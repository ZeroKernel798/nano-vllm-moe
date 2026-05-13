"""Experimental KV cache pack kernels (storage format; not wired into Attention yet).

``store_kvcache_fp8`` mirrors the slot layout of ``layers.attention.store_kvcache`` but writes
``float8_e4m3fn`` using per-slot/per-head dynamic scales (``amax / 448``).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


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
    idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offsets = tl.arange(0, HD)
    key_offsets = idx * key_stride + head_idx * HD + offsets
    value_offsets = idx * value_stride + head_idx * HD + offsets
    key_bf16 = tl.load(key_ptr + key_offsets)
    value_bf16 = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * H * HD + head_idx * HD + offsets
    key_absmax = tl.max(tl.abs(key_bf16.to(tl.float32)), axis=0)
    value_absmax = tl.max(tl.abs(value_bf16.to(tl.float32)), axis=0)
    key_scale = tl.maximum(key_absmax / 448.0, 1.0e-8)
    value_scale = tl.maximum(value_absmax / 448.0, 1.0e-8)
    key_inv_scale = 1.0 / key_scale
    value_inv_scale = 1.0 / value_scale
    key_f32 = key_bf16.to(tl.float32) * key_inv_scale
    value_f32 = value_bf16.to(tl.float32) * value_inv_scale
    k_fp8 = key_f32.to(tl.float8e4nv)
    v_fp8 = value_f32.to(tl.float8e4nv)
    tl.store(k_cache_ptr + cache_offsets, k_fp8)
    tl.store(v_cache_ptr + cache_offsets, v_fp8)
    scale_offset = slot * H + head_idx
    tl.store(k_scale_cache_ptr + scale_offset, key_scale)
    tl.store(v_scale_cache_ptr + scale_offset, value_scale)


def store_kvcache_fp8(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Write BF16 ``key``/``value`` into FP8 caches using per-slot/per-head dynamic scales.

    ``k_cache``/``v_cache`` must be ``torch.float8_e4m3fn``, shape ``[num_blocks, block_size, H, D]``.
    Scale caches must be float tensors with shape ``[num_blocks, block_size, H]``.
    """
    assert k_cache.dtype == torch.float8_e4m3fn and v_cache.dtype == torch.float8_e4m3fn
    assert k_scale_cache.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert v_scale_cache.dtype in (torch.float16, torch.bfloat16, torch.float32)
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert k_scale_cache.shape == k_cache.shape[:3]
    assert v_scale_cache.shape == v_cache.shape[:3]
    assert slot_mapping.numel() == N
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


def dequant_kvcache_fp8_slice(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference dequant for tests: entire cache tensors BF16 (not for production decode)."""
    assert k_cache.dtype == torch.float8_e4m3fn
    key_scale = k_scale_cache.float().unsqueeze(-1)
    value_scale = v_scale_cache.float().unsqueeze(-1)
    k_bf16 = (k_cache.float() * key_scale).to(torch.bfloat16)
    v_bf16 = (v_cache.float() * value_scale).to(torch.bfloat16)
    return k_bf16, v_bf16


@triton.jit
def _dequant_kvcache_fp8_gather_decode_kernel(
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
    scale_offset = (physical_block * BLOCK_SIZE + block_offset) * H + head_idx
    valid = valid_tokens[:, None]
    key = tl.load(k_cache_ptr + cache_offsets, mask=valid, other=0.0).to(tl.float32)
    value = tl.load(v_cache_ptr + cache_offsets, mask=valid, other=0.0).to(tl.float32)
    key_scale = tl.load(k_scale_cache_ptr + scale_offset, mask=valid_tokens, other=0.0).to(tl.float32)
    value_scale = tl.load(v_scale_cache_ptr + scale_offset, mask=valid_tokens, other=0.0).to(tl.float32)
    out_offsets = ((batch_idx * MAX_CONTEXT_LEN + token_offsets[:, None]) * H + head_idx) * HD + head_offsets[None, :]
    tl.store(k_out_ptr + out_offsets, key * key_scale[:, None], mask=valid)
    tl.store(v_out_ptr + out_offsets, value * value_scale[:, None], mask=valid)


def dequant_kvcache_fp8_gather_decode(
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
    """Gather only decode-visible FP8 KV tokens and dequantize into contiguous cache tensors.

    Output shape is ``[batch, max_context_len, num_kv_heads, head_dim]`` so it can be passed to
    ``flash_attn_with_kvcache`` without a paged ``block_table``. This is an intermediate step
    toward a native FP8 paged decode kernel; it avoids dequantizing unreferenced cache blocks.
    """
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
    block_tokens = int(os.environ.get("NANOVLLM_FP8_KV_GATHER_BLOCK_TOKENS", "16"))
    assert block_tokens in {1, 2, 4, 8, 16, 32, 64}
    grid = (batch_size, triton.cdiv(max_context_len, block_tokens), num_heads)
    _dequant_kvcache_fp8_gather_decode_kernel[grid](
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
        block_tokens,
    )
    return k_out, v_out
