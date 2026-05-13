from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 256}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, num_stages=3, num_warps=8),
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
    assert x_bf16.is_cuda and w_fp8.is_cuda
    M, K = x_bf16.shape
    N = w_fp8.shape[1]
    out = torch.empty((M, N), device=x_bf16.device, dtype=torch.bfloat16)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
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


def to_scaled_mm_weight(w_fp8: torch.Tensor) -> torch.Tensor:
    assert w_fp8.is_cuda
    k, n = w_fp8.shape
    out = torch.empty_strided((k, n), (1, k), device=w_fp8.device, dtype=torch.float8_e4m3fn)
    out.copy_(w_fp8)
    return out


@triton.jit
def act_quant_fp8_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    scale = tl.maximum(tl.load(scale_ptr), 1.0e-12)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = x / scale
    x = tl.minimum(tl.maximum(x, -448.0), 448.0)
    tl.store(out_ptr + offsets, x, mask=mask)


def quantize_activation_torch(x: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
    x_scale = input_scale.reshape(()).to(torch.float32).clamp(min=1e-12)
    return (x.float() / x_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def quantize_activation_triton(x: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    grid = (triton.cdiv(x.numel(), 1024),)
    act_quant_fp8_kernel[grid](x, out, input_scale.reshape(()), x.numel(), BLOCK_SIZE=1024)
    return out


def quantize_activation_w8a8(x: torch.Tensor, input_scale: torch.Tensor, backend: str = "torch") -> torch.Tensor:
    if backend == "torch":
        return quantize_activation_torch(x, input_scale)
    if backend == "triton":
        return quantize_activation_triton(x, input_scale)
    raise ValueError(f"Unsupported W8A8 activation quant backend: {backend!r}")


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}, num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def w8a8_fused_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    input_scale_ptr,
    weight_scale_ptr,
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
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    input_scale = tl.maximum(tl.load(input_scale_ptr), 1.0e-12)
    weight_scale = tl.load(weight_scale_ptr)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for kk in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offsets = kk * BLOCK_SIZE_K + offs_k
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K), other=0.0).to(tl.float32)
        a = a / input_scale
        a = tl.minimum(tl.maximum(a, -448.0), 448.0).to(tl.float8e4nv)
        b = tl.load(b_ptrs, mask=(k_offsets[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = acc * input_scale * weight_scale
    if bias_ptr is not None:
        c += tl.load(bias_ptr + offs_n)[None, :]
    tl.store(
        c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :],
        c.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def launch_w8a8_fused_gemm_experimental(
    x: torch.Tensor,
    w_fp8_col_major: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    assert x.is_cuda and w_fp8_col_major.is_cuda
    M, K = x.shape
    N = w_fp8_col_major.shape[1]
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    w8a8_fused_gemm_kernel[grid](
        x,
        w_fp8_col_major,
        out,
        input_scale.reshape(()),
        weight_scale.reshape(()),
        bias,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w_fp8_col_major.stride(0),
        w_fp8_col_major.stride(1),
        out.stride(0),
        out.stride(1),
        GROUP_SIZE_M=8,
    )
    return out


def launch_scaled_mm_w8a8(
    x: torch.Tensor,
    w_fp8_col_major: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    act_quant_backend: str = "torch",
) -> torch.Tensor:
    assert x.is_cuda and w_fp8_col_major.is_cuda
    x_scale = input_scale.reshape(()).to(torch.float32).clamp(min=1e-12)
    x_fp8 = quantize_activation_w8a8(x, input_scale, act_quant_backend)
    out = torch._scaled_mm(
        x_fp8,
        w_fp8_col_major,
        scale_a=x_scale,
        scale_b=weight_scale.reshape(()).to(torch.float32),
        out_dtype=torch.bfloat16,
    )
    if bias is not None:
        out = out + bias
    return out
