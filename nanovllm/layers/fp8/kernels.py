"""Triton FP8 GEMM for W8A16 (BF16 activations × FP8 weights).

Static W8A8 uses ``torch._scaled_mm`` (cuBLASLt FP8 Tensor Cores): real FP8×FP8 GEMM with
tensor-wise activation scale ``s_x``, then per-channel ``weight_scale`` on the result.
Activations are quantized with a Triton kernel when input is contiguous bf16 on CUDA (avoids
a full ``x.float()`` temporary that hurts large prefill).
Weights are passed in the column-major layout cuBLASLt expects (``w.t().contiguous().t()``).
Inner/outer dims are padded to multiples of 16 when required by the FP8 kernel.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _quantize_bf16_to_fp8_kernel(
    x_ptr,
    out_ptr,
    inv_scale_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_om,
    stride_ok,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused bf16 load → scale in fp32 → clamp → fp8 store (no intermediate fp32 tensor)."""
    inv_s = tl.load(inv_scale_ptr)
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    for kk in range(0, tl.cdiv(K, BLOCK_K)):
        cur_k = kk * BLOCK_K + offs_k
        mask_k = cur_k < K
        mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + cur_k[None, :] * stride_xk,
            mask=mask,
            other=0.0,
        )
        x_f = a.to(tl.float32) * inv_s
        x_f = tl.clamp(x_f, -448.0, 448.0)
        fp8v = x_f.to(tl.float8e4nv)
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + cur_k[None, :] * stride_ok,
            fp8v,
            mask=mask,
        )


def _quantize_bf16_to_fp8_triton(
    x_bf16: torch.Tensor,
    input_scale: torch.Tensor,
) -> torch.Tensor:
    """``x_bf16`` [M,K] contiguous CUDA → ``float8_e4m3fn`` [M,K] contiguous."""
    assert x_bf16.is_cuda and x_bf16.dtype == torch.bfloat16 and x_bf16.is_contiguous()
    dev = x_bf16.device
    M, K = x_bf16.shape
    s_x = input_scale.to(device=dev, dtype=torch.float32).reshape(())
    inv_scale = torch.reciprocal(s_x)
    out = torch.empty((M, K), device=dev, dtype=torch.float8_e4m3fn)
    BLOCK_M, BLOCK_K = 64, 256
    grid = (triton.cdiv(M, BLOCK_M),)
    _quantize_bf16_to_fp8_kernel[grid](
        x_bf16,
        out,
        inv_scale,
        M,
        K,
        x_bf16.stride(0),
        x_bf16.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )
    return out


def _quantize_activation_to_fp8(
    x: torch.Tensor,
    input_scale: torch.Tensor,
    dev: torch.device,
) -> torch.Tensor:
    """Activation → fp8 with tensor-wise ``input_scale``; fast path for contiguous bf16 on CUDA."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    s_x = input_scale.to(device=dev, dtype=torch.float32).reshape(())
    if (
        x.dtype == torch.bfloat16
        and x.is_cuda
        and x.is_contiguous()
        and x.device == dev
    ):
        return _quantize_bf16_to_fp8_triton(x, input_scale)
    x_f = x.float() if x.dtype != torch.float32 else x
    return (x_f / s_x).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 256},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128},
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=8,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def w8a16_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    w_scale_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Activation BF16, weight FP8 (cast to bf16 in K loop). Per-channel weight scale."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    w_scale = tl.load(w_scale_ptr + offs_bn)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for kk in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = (kk * BLOCK_SIZE_K + offs_k) < K
        a_bf16 = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b_fp8 = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        b_bf16 = b_fp8.to(tl.bfloat16)
        acc += tl.dot(a_bf16, b_bf16, out_dtype=tl.float32)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = acc * w_scale[None, :]
    if bias_ptr is not None:
        c += tl.load(bias_ptr + offs_bn)[None, :]

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    tl.store(
        c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :],
        c.to(tl.bfloat16),
        mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N),
    )


def launch_w8a16_gemm(
    x_bf16: torch.Tensor,
    w_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """x: [M, K] bf16, w: [K, N] fp8, weight_scale: [N] float32."""
    assert x_bf16.is_cuda and w_fp8.is_cuda
    M, K = x_bf16.shape
    N = w_fp8.shape[1]
    out = torch.empty((M, N), device=x_bf16.device, dtype=torch.bfloat16)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    w8a16_gemm_kernel[grid](
        x_bf16,
        w_fp8,
        out,
        weight_scale,
        bias,
        M,
        N,
        K,
        x_bf16.stride(0),
        x_bf16.stride(1),
        w_fp8.stride(0),
        w_fp8.stride(1),
        out.stride(0),
        out.stride(1),
        GROUP_SIZE_M=8,
    )
    return out


def launch_w8a8_static_gemm(
    x: torch.Tensor,
    input_scale: torch.Tensor,
    w_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    w_fp8_nk: torch.Tensor | None = None,
) -> torch.Tensor:
    """PTQ W8A8: ``x_fp8`` @ ``w_fp8`` via cuBLASLt FP8 GEMM, then ``* weight_scale`` (+ bias).

    Same scaling semantics as the previous FP32 matmul reference: quantize activations with
    ``input_scale``, apply ``_scaled_mm`` with scale ``s_x`` vs ``1``, multiply columns by
    ``weight_scale``. Uses ``use_fast_accum`` for throughput (slightly lower precision).

    ``w_fp8_nk``: optional contiguous ``[N, K]`` uint8/fp8 storage of ``w_fp8.T`` (see
    ``qweight_nk`` in parallel layers). When set and no padding is needed, avoids per-forward
    ``w.t().contiguous()`` allocations that dominate decode latency.
    """
    assert x.is_cuda and w_fp8.is_cuda
    assert x.dtype in (torch.float32, torch.bfloat16)
    assert w_fp8.dtype == torch.float8_e4m3fn
    dev = x.device
    s_x = input_scale.to(device=dev, dtype=torch.float32).reshape(())
    w_s = weight_scale.to(device=dev, dtype=torch.float32)
    x_fp8 = _quantize_activation_to_fp8(x, input_scale, dev)

    K = x_fp8.shape[1]
    K_w, N = w_fp8.shape
    assert K == K_w

    K_pad = (K + 15) // 16 * 16
    N_pad = (N + 15) // 16 * 16
    if K_pad != K or N_pad != N:
        x_fp8 = F.pad(x_fp8, (0, K_pad - K), value=0)
        w_use = F.pad(w_fp8, (0, N_pad - N, 0, K_pad - K), value=0)
        mat2 = w_use.t().contiguous().t()
    elif w_fp8_nk is not None:
        assert w_fp8_nk.shape == (N, K)
        mat2 = w_fp8_nk.view(torch.float8_e4m3fn).t()
    else:
        mat2 = w_fp8.t().contiguous().t()
    sb = torch.ones((), device=dev, dtype=torch.float32)

    out = torch._scaled_mm(
        x_fp8,
        mat2,
        s_x,
        sb,
        out_dtype=torch.bfloat16,
        use_fast_accum=True,
    )
    out = out[:, :N].float() * w_s
    if bias is not None:
        out = out + bias.to(device=dev, dtype=torch.float32)
    return out.to(torch.bfloat16)
