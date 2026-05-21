"""Mixed KV cache kernels.

The maintained quantized KV format is K-int8/V-FP8:

* K uses symmetric int8 with per-token/per-head/group-32 scales.
* V uses FP8 E4M3 with per-token/per-head scales.

Full FP8 KV, V-only FP8, and fake quantization probes were removed from the runtime. The
ablation rationale is documented in ``opt/kv_cache_quant.md``.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _store_kvcache_k_int8_v_fp8_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    slot_mapping_ptr,
    H: tl.constexpr,
    HD: tl.constexpr,
    K_GROUP_SIZE: tl.constexpr,
    K_GROUPS: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    group_idx = tl.program_id(2)
    group_offsets = tl.arange(0, K_GROUP_SIZE)
    head_offsets = group_idx * K_GROUP_SIZE + group_offsets
    key_offsets = token_idx * key_stride + head_idx * HD + head_offsets
    key_values = tl.load(key_ptr + key_offsets).to(tl.float32)
    key_absmax = tl.max(tl.abs(key_values), axis=0)
    key_scale = tl.maximum(key_absmax / 127.0, 1.0e-8)
    key_q = (key_values / key_scale).to(tl.int8)
    slot = tl.load(slot_mapping_ptr + token_idx)
    cache_offsets = slot * H * HD + head_idx * HD + head_offsets
    tl.store(k_cache_ptr + cache_offsets, key_q)
    tl.store(k_scale_cache_ptr + (slot * H + head_idx) * K_GROUPS + group_idx, key_scale)

    if group_idx == 0:
        v_offsets = tl.arange(0, HD)
        value_offsets = token_idx * value_stride + head_idx * HD + v_offsets
        value_values = tl.load(value_ptr + value_offsets).to(tl.float32)
        value_absmax = tl.max(tl.abs(value_values), axis=0)
        value_scale = tl.maximum(value_absmax / 448.0, 1.0e-8)
        value_q = (value_values / value_scale).to(tl.float8e4nv)
        value_cache_offsets = slot * H * HD + head_idx * HD + v_offsets
        tl.store(v_cache_ptr + value_cache_offsets, value_q)
        tl.store(v_scale_cache_ptr + slot * H + head_idx, value_scale)


def store_kvcache_k_int8_v_fp8_triton(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_group_size: int = 32,
) -> None:
    N, num_heads, head_dim = key.shape
    assert value.shape == key.shape
    assert head_dim % k_group_size == 0
    assert k_cache.dtype == torch.int8
    assert v_cache.dtype == torch.float8_e4m3fn
    assert k_scale_cache.shape == (*k_cache.shape[:3], head_dim // k_group_size)
    assert v_scale_cache.shape == v_cache.shape[:3]
    assert slot_mapping.numel() == N
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    _store_kvcache_k_int8_v_fp8_kernel[(N, num_heads, head_dim // k_group_size)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        slot_mapping,
        num_heads,
        head_dim,
        k_group_size,
        head_dim // k_group_size,
    )


@triton.jit
def _store_kvcache_fp8_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    slot_mapping_ptr,
    H: tl.constexpr,
    HD: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offsets = tl.arange(0, HD)
    key_offsets = token_idx * key_stride + head_idx * HD + offsets
    value_offsets = token_idx * value_stride + head_idx * HD + offsets
    key_values = tl.load(key_ptr + key_offsets).to(tl.float32)
    value_values = tl.load(value_ptr + value_offsets).to(tl.float32)
    key_absmax = tl.max(tl.abs(key_values), axis=0)
    value_absmax = tl.max(tl.abs(value_values), axis=0)
    key_scale = tl.maximum(key_absmax / 448.0, 1.0e-8)
    value_scale = tl.maximum(value_absmax / 448.0, 1.0e-8)
    key_q = (key_values / key_scale).to(tl.float8e4nv)
    value_q = (value_values / value_scale).to(tl.float8e4nv)
    slot = tl.load(slot_mapping_ptr + token_idx)
    cache_offsets = slot * H * HD + head_idx * HD + offsets
    tl.store(k_cache_ptr + cache_offsets, key_q)
    tl.store(v_cache_ptr + cache_offsets, value_q)
    tl.store(k_scale_cache_ptr + slot * H + head_idx, key_scale)
    tl.store(v_scale_cache_ptr + slot * H + head_idx, value_scale)


def store_kvcache_fp8_triton(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    N, num_heads, head_dim = key.shape
    assert value.shape == key.shape
    assert k_cache.dtype == torch.float8_e4m3fn
    assert v_cache.dtype == torch.float8_e4m3fn
    assert k_scale_cache.shape == k_cache.shape[:3]
    assert v_scale_cache.shape == v_cache.shape[:3]
    assert slot_mapping.numel() == N
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    _store_kvcache_fp8_kernel[(N, num_heads)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        slot_mapping,
        num_heads,
        head_dim,
    )


def dequant_fp8_kvcache_slice(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert k_cache.dtype == torch.float8_e4m3fn
    assert v_cache.dtype == torch.float8_e4m3fn
    assert k_scale_cache.shape == k_cache.shape[:3]
    assert v_scale_cache.shape == v_cache.shape[:3]
    k_bf16 = (k_cache.float() * k_scale_cache.float().unsqueeze(-1)).to(torch.bfloat16)
    v_bf16 = (v_cache.float() * v_scale_cache.float().unsqueeze(-1)).to(torch.bfloat16)
    return k_bf16, v_bf16


def dequant_fp8_kvcache_gather_decode(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    output_dtype: torch.dtype,
    k_out: torch.Tensor | None = None,
    v_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert k_cache.dtype == torch.float8_e4m3fn and v_cache.dtype == torch.float8_e4m3fn
    assert k_scale_cache.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert v_scale_cache.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert block_tables.dtype == torch.int32 and context_lens.dtype == torch.int32
    assert output_dtype in (torch.float16, torch.bfloat16)
    assert k_cache.shape == v_cache.shape
    assert k_scale_cache.shape == k_cache.shape[:3]
    assert v_scale_cache.shape == v_cache.shape[:3]
    batch_size = context_lens.numel()
    max_context_len = int(context_lens.max().item())
    _, block_size, num_heads, head_dim = k_cache.shape
    output_shape = (batch_size, max_context_len, num_heads, head_dim)
    if k_out is None or v_out is None:
        k_out = torch.empty(output_shape, dtype=output_dtype, device=k_cache.device)
        v_out = torch.empty_like(k_out)
    else:
        assert k_out.shape == output_shape and v_out.shape == output_shape
        assert k_out.dtype == output_dtype and v_out.dtype == output_dtype
        assert k_out.device == k_cache.device and v_out.device == v_cache.device
    block_tokens = int(os.environ.get("NANOVLLM_K_INT8_V_FP8_GATHER_BLOCK_TOKENS", "16"))
    assert block_tokens in {1, 2, 4, 8, 16, 32, 64}
    grid = (batch_size, triton.cdiv(max_context_len, block_tokens), num_heads)
    _dequant_k_int8_vcache_fp8_gather_decode_kernel[grid](
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        block_tables,
        context_lens,
        k_out,
        v_out,
        block_tables.stride(0),
        max_context_len,
        block_size,
        num_heads,
        head_dim,
        1,
        head_dim,
        block_tokens,
    )
    return k_out, v_out


def dequant_k_int8_vcache_fp8_slice(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert k_cache.dtype == torch.int8
    assert v_cache.dtype == torch.float8_e4m3fn
    k_group_size = k_cache.shape[-1] // k_scale_cache.shape[-1]
    k_grouped = k_cache.float().reshape(*k_cache.shape[:-1], k_scale_cache.shape[-1], k_group_size)
    k_bf16 = k_grouped.mul(k_scale_cache.float().unsqueeze(-1)).reshape_as(k_cache).to(torch.bfloat16)
    v_bf16 = (v_cache.float() * v_scale_cache.float().unsqueeze(-1)).to(torch.bfloat16)
    return k_bf16, v_bf16


@triton.jit
def _dequant_k_int8_vcache_fp8_gather_decode_kernel(
    k_cache_ptr,
    v_cache_ptr,
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    k_out_ptr,
    v_out_ptr,
    BLOCK_TABLE_STRIDE: tl.constexpr,
    MAX_CONTEXT_LEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    H: tl.constexpr,
    HD: tl.constexpr,
    K_GROUPS: tl.constexpr,
    K_GROUP_SIZE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    token_block_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    token_offsets = token_block_idx * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    head_offsets = tl.arange(0, HD)
    context_len = tl.load(context_lens_ptr + batch_idx)
    valid_tokens = token_offsets < context_len
    logical_block_idx = token_offsets // BLOCK_SIZE
    block_offset = token_offsets - logical_block_idx * BLOCK_SIZE
    physical_block = tl.load(
        block_tables_ptr + batch_idx * BLOCK_TABLE_STRIDE + logical_block_idx,
        mask=valid_tokens,
        other=0,
    )
    cache_offsets = ((physical_block[:, None] * BLOCK_SIZE + block_offset[:, None]) * H + head_idx) * HD + head_offsets[None, :]
    scale_base = (physical_block * BLOCK_SIZE + block_offset) * H + head_idx
    k_scale_offsets = scale_base[:, None] * K_GROUPS + head_offsets[None, :] // K_GROUP_SIZE
    v_scale_offsets = scale_base
    valid = valid_tokens[:, None]
    key = tl.load(k_cache_ptr + cache_offsets, mask=valid, other=0.0).to(tl.float32)
    value = tl.load(v_cache_ptr + cache_offsets, mask=valid, other=0.0).to(tl.float32)
    key_scale = tl.load(k_scale_cache_ptr + k_scale_offsets, mask=valid, other=0.0).to(tl.float32)
    value_scale = tl.load(v_scale_cache_ptr + v_scale_offsets, mask=valid_tokens, other=0.0).to(tl.float32)
    out_offsets = ((batch_idx * MAX_CONTEXT_LEN + token_offsets[:, None]) * H + head_idx) * HD + head_offsets[None, :]
    tl.store(k_out_ptr + out_offsets, key * key_scale, mask=valid)
    tl.store(v_out_ptr + out_offsets, value * value_scale[:, None], mask=valid)


def dequant_k_int8_vcache_fp8_gather_decode(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    output_dtype: torch.dtype,
    k_out: torch.Tensor | None = None,
    v_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert k_cache.dtype == torch.int8 and v_cache.dtype == torch.float8_e4m3fn
    assert k_scale_cache.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert v_scale_cache.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert block_tables.dtype == torch.int32 and context_lens.dtype == torch.int32
    assert output_dtype in (torch.float16, torch.bfloat16)
    assert k_cache.shape == v_cache.shape
    assert v_scale_cache.shape == v_cache.shape[:3]
    batch_size = context_lens.numel()
    max_context_len = int(context_lens.max().item())
    _, block_size, num_heads, head_dim = k_cache.shape
    k_groups = k_scale_cache.shape[-1]
    assert k_scale_cache.shape == (*k_cache.shape[:3], k_groups)
    assert head_dim % k_groups == 0
    output_shape = (batch_size, max_context_len, num_heads, head_dim)
    if k_out is None or v_out is None:
        k_out = torch.empty(output_shape, dtype=output_dtype, device=k_cache.device)
        v_out = torch.empty_like(k_out)
    else:
        assert k_out.shape == output_shape and v_out.shape == output_shape
        assert k_out.dtype == output_dtype and v_out.dtype == output_dtype
        assert k_out.device == k_cache.device and v_out.device == v_cache.device
    block_tokens = int(os.environ.get("NANOVLLM_K_INT8_V_FP8_GATHER_BLOCK_TOKENS", "16"))
    assert block_tokens in {1, 2, 4, 8, 16, 32, 64}
    grid = (batch_size, triton.cdiv(max_context_len, block_tokens), num_heads)
    _dequant_k_int8_vcache_fp8_gather_decode_kernel[grid](
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        block_tables,
        context_lens,
        k_out,
        v_out,
        block_tables.stride(0),
        max_context_len,
        block_size,
        num_heads,
        head_dim,
        k_groups,
        head_dim // k_groups,
        block_tokens,
    )
    return k_out, v_out
