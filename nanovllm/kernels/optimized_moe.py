from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    token_start = token_block_id * BLOCK_M
    token_end = min((token_block_id + 1) * BLOCK_M, token_num)

    dim_start = dim_block_id * BLOCK_DIM
    dim_end = min((dim_block_id + 1) * BLOCK_DIM, hidden_dim)

    offs_dim = dim_start + tl.arange(0, BLOCK_DIM)

    for token_index in range(token_start, token_end):
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        input_t_ptr = input_ptr + token_index * input_stride_0 + offs_dim
        for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
            tmp = tl.load(input_t_ptr + i * input_stride_1, mask=offs_dim < dim_end, other=0.0)
            accumulator += tmp
        store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim
        tl.store(store_t_ptr, accumulator.to(input_ptr.dtype.element_ty), mask=offs_dim < dim_end)


@triton.jit
def fused_moe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if even_Ks:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(c_ptr.dtype.element_ty)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def launch_moe_sum_reduce(input_tensor, output_tensor) -> None:
    token_num, topk_num, hidden_dim = input_tensor.shape
    block_m = 16
    block_dim = min(triton.next_power_of_2(hidden_dim), 1024)
    grid = (triton.cdiv(token_num, block_m), triton.cdiv(hidden_dim, block_dim))
    moe_sum_reduce_kernel[grid](
        input_tensor,
        input_tensor.stride(0),
        input_tensor.stride(1),
        input_tensor.stride(2),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        token_num,
        topk_num,
        hidden_dim,
        BLOCK_M=block_m,
        BLOCK_DIM=block_dim,
        NUM_STAGE=1,
    )


def launch_fused_moe_kernel(
    hidden_states,
    weight,
    output,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    *,
    apply_router_weight: bool,
    top_k: int,
    config: dict[str, int],
    compute_type,
) -> None:
    num_tokens = hidden_states.shape[0]
    output_dim = weight.shape[1]
    input_dim = weight.shape[2]
    em = sorted_token_ids.shape[0]
    output_stride_n = output.stride(2) if output.dim() == 3 else output.stride(1)
    grid = (
        triton.cdiv(em, config["BLOCK_SIZE_M"]) * triton.cdiv(output_dim, config["BLOCK_SIZE_N"]),
    )
    fused_moe_kernel[grid](
        hidden_states,
        weight,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        output_dim,
        input_dim,
        em,
        num_tokens * top_k,
        hidden_states.stride(0),
        hidden_states.stride(1),
        weight.stride(0),
        weight.stride(2),
        weight.stride(1),
        output.stride(0),
        output_stride_n,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        MUL_ROUTED_WEIGHT=apply_router_weight,
        top_k=top_k,
        compute_type=compute_type,
        even_Ks=input_dim % config["BLOCK_SIZE_K"] == 0,
    )


@triton.jit
def token_moe_w13_kernel(
    x_ptr,
    w13_ptr,
    gate_up_ptr,
    topk_ids_ptr,
    M: tl.constexpr,
    I: tl.constexpr,
    H: tl.constexpr,
    stride_xm,
    stride_xh,
    stride_we,
    stride_wn,
    stride_wh,
    stride_gm,
    stride_gn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token_id = tl.program_id(0)
    n_block = tl.program_id(1)
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    expert_id = tl.load(topk_ids_ptr + token_id).to(tl.int64)

    gate_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_base = x_ptr + token_id * stride_xm
    gate_base = w13_ptr + expert_id * stride_we
    up_base = gate_base + I * stride_wn
    for k in range(0, tl.cdiv(H, BLOCK_K)):
        k_offsets = k * BLOCK_K + offs_k
        x = tl.load(x_base + k_offsets * stride_xh, mask=k_offsets < H, other=0.0)
        gate_w = tl.load(
            gate_base + offs_n[:, None] * stride_wn + k_offsets[None, :] * stride_wh,
            mask=(offs_n[:, None] < I) & (k_offsets[None, :] < H),
            other=0.0,
        )
        up_w = tl.load(
            up_base + offs_n[:, None] * stride_wn + k_offsets[None, :] * stride_wh,
            mask=(offs_n[:, None] < I) & (k_offsets[None, :] < H),
            other=0.0,
        )
        gate_acc += tl.sum(gate_w * x[None, :], axis=1)
        up_acc += tl.sum(up_w * x[None, :], axis=1)

    tl.store(
        gate_up_ptr + token_id * stride_gm + offs_n * stride_gn,
        gate_acc.to(gate_up_ptr.dtype.element_ty),
        mask=offs_n < I,
    )
    tl.store(
        gate_up_ptr + token_id * stride_gm + (I + offs_n) * stride_gn,
        up_acc.to(gate_up_ptr.dtype.element_ty),
        mask=offs_n < I,
    )


@triton.jit
def token_moe_w2_kernel(
    activated_ptr,
    w2_ptr,
    output_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    M: tl.constexpr,
    I: tl.constexpr,
    H: tl.constexpr,
    stride_am,
    stride_ai,
    stride_we,
    stride_wh,
    stride_wi,
    stride_om,
    stride_oh,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token_id = tl.program_id(0)
    n_block = tl.program_id(1)
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    expert_id = tl.load(topk_ids_ptr + token_id).to(tl.int64)
    route_weight = tl.load(topk_weights_ptr + token_id).to(tl.float32)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    a_base = activated_ptr + token_id * stride_am
    w_base = w2_ptr + expert_id * stride_we
    for k in range(0, tl.cdiv(I, BLOCK_K)):
        k_offsets = k * BLOCK_K + offs_k
        a = tl.load(a_base + k_offsets * stride_ai, mask=k_offsets < I, other=0.0)
        w = tl.load(
            w_base + offs_n[:, None] * stride_wh + k_offsets[None, :] * stride_wi,
            mask=(offs_n[:, None] < H) & (k_offsets[None, :] < I),
            other=0.0,
        )
        acc += tl.sum(w * a[None, :], axis=1)

    acc = acc * route_weight
    tl.store(
        output_ptr + token_id * stride_om + offs_n * stride_oh,
        acc.to(output_ptr.dtype.element_ty),
        mask=offs_n < H,
    )


def launch_token_moe_graph_kernel(hidden_states, topk_ids, topk_weights, w13, w2, output, *, intermediate_size: int) -> None:
    m_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    gate_up = torch.empty((m_tokens, 2 * intermediate_size), device=hidden_states.device, dtype=hidden_states.dtype)
    block_n = 32
    block_k = 64
    grid_w13 = (m_tokens, triton.cdiv(intermediate_size, block_n))
    token_moe_w13_kernel[grid_w13](
        hidden_states,
        w13,
        gate_up,
        topk_ids,
        m_tokens,
        intermediate_size,
        hidden_size,
        hidden_states.stride(0),
        hidden_states.stride(1),
        w13.stride(0),
        w13.stride(1),
        w13.stride(2),
        gate_up.stride(0),
        gate_up.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    gate, up = gate_up.chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate) * up
    activated = activated.contiguous()
    grid_w2 = (m_tokens, triton.cdiv(hidden_size, block_n))
    token_moe_w2_kernel[grid_w2](
        activated,
        w2,
        output,
        topk_ids,
        topk_weights,
        m_tokens,
        intermediate_size,
        hidden_size,
        activated.stride(0),
        activated.stride(1),
        w2.stride(0),
        w2.stride(1),
        w2.stride(2),
        output.stride(0),
        output.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
