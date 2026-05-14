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


def store_kvcache_k_int8_v_fp8(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_group_size: int = 32,
) -> None:
    """Experimental: store K as symmetric int8 group-quantized and V as FP8 E4M3."""
    N, num_heads, head_dim = key.shape
    assert value.shape == key.shape
    assert head_dim % k_group_size == 0
    assert k_cache.dtype == torch.int8
    assert v_cache.dtype == torch.float8_e4m3fn
    assert k_scale_cache.shape == (*k_cache.shape[:3], head_dim // k_group_size)
    assert v_scale_cache.shape == v_cache.shape[:3]
    slots = slot_mapping.long()

    key_grouped = key.float().reshape(N, num_heads, head_dim // k_group_size, k_group_size)
    k_scale = key_grouped.abs().amax(dim=-1, keepdim=True).div(127.0).clamp(min=1.0e-8)
    k_q = key_grouped.div(k_scale).round().clamp(-127.0, 127.0).to(torch.int8).reshape(N, num_heads, head_dim)
    k_cache.view(-1, num_heads, head_dim)[slots] = k_q
    k_scale_cache.view(-1, num_heads, head_dim // k_group_size)[slots] = k_scale.squeeze(-1).to(k_scale_cache.dtype)

    v_scale = value.float().abs().amax(dim=-1, keepdim=True).div(448.0).clamp(min=1.0e-8)
    v_cache.view(-1, num_heads, head_dim)[slots] = value.float().div(v_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    v_scale_cache.view(-1, num_heads)[slots] = v_scale.squeeze(-1).to(v_scale_cache.dtype)


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


def store_kvcache_v_fp8(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Debug/experimental: store K in BF16 cache and V in FP8 cache."""
    N, num_heads, head_dim = key.shape
    assert value.shape == key.shape
    assert k_cache.dtype in (torch.float16, torch.bfloat16)
    assert v_cache.dtype == torch.float8_e4m3fn
    assert v_scale_cache.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert slot_mapping.numel() == N
    k_cache.view(-1, num_heads, head_dim)[slot_mapping.long()] = key
    scale = value.float().abs().amax(dim=-1, keepdim=True).div(448.0).clamp(min=1.0e-8)
    v_cache.view(-1, num_heads, head_dim)[slot_mapping.long()] = value.float().div(scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    v_scale_cache.view(-1, num_heads)[slot_mapping.long()] = scale.squeeze(-1).to(v_scale_cache.dtype)


def dequant_vcache_fp8_slice(v_cache: torch.Tensor, v_scale_cache: torch.Tensor) -> torch.Tensor:
    assert v_cache.dtype == torch.float8_e4m3fn
    return (v_cache.float() * v_scale_cache.float().unsqueeze(-1)).to(torch.bfloat16)


def store_kvcache_fake_fp8(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    mode: str,
) -> None:
    """Debug-only: quantize K and/or V through FP8, dequantize, then store in BF16 cache."""
    if mode not in {"k", "v", "both"}:
        raise ValueError(f"Unsupported fake FP8 KV store mode: {mode!r}")
    N, num_heads, head_dim = key.shape
    assert value.shape == key.shape
    assert k_cache.dtype in (torch.float16, torch.bfloat16)
    assert v_cache.dtype in (torch.float16, torch.bfloat16)
    assert slot_mapping.numel() == N

    def fake_fp8_roundtrip(x: torch.Tensor) -> torch.Tensor:
        quant_format = os.environ.get("NANOVLLM_KV_FAKE_FP8_FORMAT", "e4m3").strip().lower()
        group_size = int(os.environ.get("NANOVLLM_KV_FAKE_FP8_GROUP_SIZE", "0"))
        if quant_format == "e4m3":
            quant_dtype = torch.float8_e4m3fn
            max_value = 448.0
        elif quant_format == "e5m2":
            quant_dtype = torch.float8_e5m2
            max_value = 57344.0
        elif quant_format == "int8":
            quant_dtype = torch.int8
            max_value = 127.0
        else:
            raise ValueError(f"Unsupported fake KV quant format: {quant_format!r}")

        x_float = x.float()
        if group_size and group_size < x.shape[-1]:
            if x.shape[-1] % group_size != 0:
                raise ValueError(f"head_dim={x.shape[-1]} must be divisible by group_size={group_size}")
            grouped = x_float.reshape(*x.shape[:-1], x.shape[-1] // group_size, group_size)
            scale = grouped.abs().amax(dim=-1, keepdim=True).div(max_value).clamp(min=1.0e-8)
            rounded = grouped.div(scale).round().clamp(-max_value, max_value).to(quant_dtype).float().mul(scale) if quant_format == "int8" else grouped.div(scale).clamp(-max_value, max_value).to(quant_dtype).float().mul(scale)
            return rounded.reshape_as(x_float).to(x.dtype)
        scale = x_float.abs().amax(dim=-1, keepdim=True).div(max_value).clamp(min=1.0e-8)
        return (x_float.div(scale).round().clamp(-max_value, max_value).to(quant_dtype).float().mul(scale) if quant_format == "int8" else x_float.div(scale).clamp(-max_value, max_value).to(quant_dtype).float().mul(scale)).to(x.dtype)

    stored_key = fake_fp8_roundtrip(key) if mode in {"k", "both"} else key
    stored_value = fake_fp8_roundtrip(value) if mode in {"v", "both"} else value
    k_cache.view(-1, num_heads, head_dim)[slot_mapping.long()] = stored_key
    v_cache.view(-1, num_heads, head_dim)[slot_mapping.long()] = stored_value


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
