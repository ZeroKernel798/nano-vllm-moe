"""Experimental KV cache pack kernels (storage format; not wired into Attention yet).

``store_kvcache_fp8`` mirrors the slot layout of ``layers.attention.store_kvcache`` but writes
``float8_e4m3fn`` using a scalar ``scale`` (same as ``amax / 448`` style static scaling).
"""

from __future__ import annotations

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
    slot_mapping_ptr,
    inv_scale_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    inv_scale = tl.load(inv_scale_ptr)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key_bf16 = tl.load(key_ptr + key_offsets)
    value_bf16 = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    key_f32 = key_bf16.to(tl.float32) * inv_scale
    value_f32 = value_bf16.to(tl.float32) * inv_scale
    k_fp8 = key_f32.to(tl.float8e4nv)
    v_fp8 = value_f32.to(tl.float8e4nv)
    tl.store(k_cache_ptr + cache_offsets, k_fp8)
    tl.store(v_cache_ptr + cache_offsets, v_fp8)


def store_kvcache_fp8(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    """Write BF16 ``key``/``value`` into FP8 caches using ``scale`` (float tensor, typically scalar).

    ``k_cache``/``v_cache`` must be ``torch.float8_e4m3fn``, shape ``[num_blocks, block_size, H, D]``.
    Quantization: ``(x.float() / scale).to(float8_e4m3fn)`` — kernel uses ``inv_scale = 1/scale``.
    """
    assert k_cache.dtype == torch.float8_e4m3fn and v_cache.dtype == torch.float8_e4m3fn
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    inv_scale = (1.0 / scale.float()).to(device=key.device, dtype=torch.float32)
    _store_kvcache_fp8_kernel[(N,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        inv_scale,
        D,
    )


def dequant_kvcache_fp8_slice(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference dequant for tests: entire cache tensors BF16 (not for production decode)."""
    assert k_cache.dtype == torch.float8_e4m3fn
    s = scale.float().view(1, 1, 1, 1)
    k_bf16 = (k_cache.float() * s).to(torch.bfloat16)
    v_bf16 = (v_cache.float() * s).to(torch.bfloat16)
    return k_bf16, v_bf16
