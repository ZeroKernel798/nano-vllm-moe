"""Multi-head attention with paged KV, FlashAttention, and experimental FP8 KV cache."""

import os

import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from torch import nn

from nanovllm.layers.fp8_paged_attention import fp8_paged_attention_decode, k_int8_v_fp8_paged_attention_decode
from nanovllm.layers.kv_cache_kernels import (
    dequant_kvcache_fp8_gather_decode,
    dequant_k_int8_vcache_fp8_gather_decode,
    dequant_k_int8_vcache_fp8_slice,
    dequant_kvcache_fp8_slice,
    dequant_vcache_fp8_slice,
    store_kvcache_fake_fp8,
    store_kvcache_fp8,
    store_kvcache_k_int8_v_fp8,
    store_kvcache_k_int8_v_fp8_triton,
    store_kvcache_v_fp8,
)
from nanovllm.utils.context import get_context
from nanovllm.utils.kv_cache_profile import timed_kv_cache_profile


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale_cache = self.v_scale_cache = torch.tensor([])
        self._fp8_kv_gather_workspace: tuple[torch.Tensor, torch.Tensor] | None = None

    def _get_fp8_kv_gather_workspace(
        self,
        batch_size: int,
        max_context_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch_size, max_context_len, self.num_kv_heads, self.head_dim)
        workspace = self._fp8_kv_gather_workspace
        if (
            workspace is None
            or workspace[0].shape != shape
            or workspace[0].dtype != dtype
            or workspace[0].device != device
        ):
            k_workspace = torch.empty(shape, dtype=dtype, device=device)
            v_workspace = torch.empty_like(k_workspace)
            self._fp8_kv_gather_workspace = (k_workspace, v_workspace)
        return self._fp8_kv_gather_workspace

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        fp8_cache = k_cache.dtype == torch.float8_e4m3fn if k_cache.numel() else False
        v_fp8_cache = v_cache.dtype == torch.float8_e4m3fn if v_cache.numel() else False
        k_int8_cache = k_cache.dtype == torch.int8 if k_cache.numel() else False
        kv_profile = os.environ.get("NANOVLLM_KV_CACHE_PROFILE", "0") == "1"
        fake_fp8_store_mode = os.environ.get("NANOVLLM_KV_FAKE_FP8_STORE", "").strip().lower()
        fp8_decode_backend = os.environ.get("NANOVLLM_FP8_KV_DECODE", "native")
        if k_cache.numel() and v_cache.numel():
            if fp8_cache:
                store = lambda: store_kvcache_fp8(
                    k, v, k_cache, v_cache, self.k_scale_cache, self.v_scale_cache, context.slot_mapping
                )
                if kv_profile:
                    timed_kv_cache_profile("fp8_store", store)
                else:
                    store()
            elif k_int8_cache and v_fp8_cache:
                if os.environ.get("NANOVLLM_K_INT8_V_FP8_STORE", "triton").strip().lower() == "torch":
                    store = lambda: store_kvcache_k_int8_v_fp8(
                        k, v, k_cache, v_cache, self.k_scale_cache, self.v_scale_cache, context.slot_mapping
                    )
                    store_profile_name = "k_int8_v_fp8_store_torch"
                else:
                    store = lambda: store_kvcache_k_int8_v_fp8_triton(
                        k, v, k_cache, v_cache, self.k_scale_cache, self.v_scale_cache, context.slot_mapping
                    )
                    store_profile_name = "k_int8_v_fp8_store_triton"
                if kv_profile:
                    timed_kv_cache_profile(store_profile_name, store)
                else:
                    store()
            elif v_fp8_cache:
                store = lambda: store_kvcache_v_fp8(k, v, k_cache, v_cache, self.v_scale_cache, context.slot_mapping)
                if kv_profile:
                    timed_kv_cache_profile("v_fp8_store", store)
                else:
                    store()
            else:
                if fake_fp8_store_mode:
                    store = lambda: store_kvcache_fake_fp8(k, v, k_cache, v_cache, context.slot_mapping, fake_fp8_store_mode)
                    profile_name = f"bf16_store_fake_fp8_{fake_fp8_store_mode}"
                else:
                    store = lambda: store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
                    profile_name = "bf16_store"
                if kv_profile:
                    timed_kv_cache_profile(profile_name, store)
                else:
                    store()
        if context.is_prefill:
            if context.block_tables is not None:  # prefix cache
                if fp8_cache:
                    dequant = lambda: dequant_kvcache_fp8_slice(
                        k_cache, v_cache, self.k_scale_cache, self.v_scale_cache
                    )
                    if kv_profile:
                        k_cache, v_cache = timed_kv_cache_profile("fp8_prefill_dequant_full_cache", dequant)
                    else:
                        k_cache, v_cache = dequant()
                elif k_int8_cache and v_fp8_cache:
                    dequant = lambda: dequant_k_int8_vcache_fp8_slice(
                        k_cache, v_cache, self.k_scale_cache, self.v_scale_cache
                    )
                    if kv_profile:
                        k_cache, v_cache = timed_kv_cache_profile("k_int8_v_fp8_prefill_dequant_full_cache", dequant)
                    else:
                        k_cache, v_cache = dequant()
                elif v_fp8_cache:
                    dequant_v = lambda: dequant_vcache_fp8_slice(v_cache, self.v_scale_cache)
                    if kv_profile:
                        v_cache = timed_kv_cache_profile("v_fp8_prefill_dequant_full_cache", dequant_v)
                    else:
                        v_cache = dequant_v()
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        else:  # decode
            decode_block_table = context.block_tables
            if k_int8_cache and v_fp8_cache:
                if fp8_decode_backend == "native" and q.size(0) == 1 and context.context_lens.numel() == 1:
                    block_tokens = int(os.environ.get("NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS", "64"))

                    def native_decode():
                        return k_int8_v_fp8_paged_attention_decode(
                            q[0],
                            k_cache,
                            v_cache,
                            self.k_scale_cache,
                            self.v_scale_cache,
                            context.block_tables[0],
                            context.context_lens[:1],
                            self.scale,
                            block_tokens=block_tokens,
                        ).unsqueeze(0)

                    if kv_profile:
                        o = timed_kv_cache_profile("k_int8_v_fp8_decode_native_paged_attention", native_decode)
                    else:
                        o = native_decode()
                    o = o.view(-1, self.num_heads * self.head_dim)
                    return o
                if fp8_decode_backend in {"native", "gather_dequant"}:
                    def dequant():
                        max_context_len = int(context.context_lens.max().item())
                        k_workspace, v_workspace = self._get_fp8_kv_gather_workspace(
                            context.context_lens.numel(), max_context_len, q.dtype, q.device
                        )
                        return dequant_k_int8_vcache_fp8_gather_decode(
                            k_cache,
                            v_cache,
                            self.k_scale_cache,
                            self.v_scale_cache,
                            context.block_tables,
                            context.context_lens,
                            q.dtype,
                            k_workspace,
                            v_workspace,
                        )
                    if kv_profile:
                        k_cache, v_cache = timed_kv_cache_profile("k_int8_v_fp8_decode_gather_dequant", dequant)
                    else:
                        k_cache, v_cache = dequant()
                    decode_block_table = None
                else:
                    dequant = lambda: dequant_k_int8_vcache_fp8_slice(
                        k_cache, v_cache, self.k_scale_cache, self.v_scale_cache
                    )
                    if kv_profile:
                        k_cache, v_cache = timed_kv_cache_profile("k_int8_v_fp8_decode_dequant_full_cache", dequant)
                    else:
                        k_cache, v_cache = dequant()
            elif v_fp8_cache and not fp8_cache:
                dequant_v = lambda: dequant_vcache_fp8_slice(v_cache, self.v_scale_cache)
                if kv_profile:
                    v_cache = timed_kv_cache_profile("v_fp8_decode_dequant_full_cache", dequant_v)
                else:
                    v_cache = dequant_v()
            if fp8_cache:
                if fp8_decode_backend == "native" and q.size(0) == 1 and context.context_lens.numel() == 1:
                    block_tokens = int(os.environ.get("NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS", "64"))

                    def native_decode():
                        return fp8_paged_attention_decode(
                            q[0],
                            k_cache,
                            v_cache,
                            self.k_scale_cache,
                            self.v_scale_cache,
                            context.block_tables[0],
                            context.context_lens[:1],
                            self.scale,
                            block_tokens=block_tokens,
                        ).unsqueeze(0)

                    if kv_profile:
                        o = timed_kv_cache_profile("fp8_decode_native_paged_attention", native_decode)
                    else:
                        o = native_decode()
                    o = o.view(-1, self.num_heads * self.head_dim)
                    return o
                if fp8_decode_backend in {"native", "gather_dequant"}:
                    def dequant():
                        max_context_len = int(context.context_lens.max().item())
                        k_workspace, v_workspace = self._get_fp8_kv_gather_workspace(
                            context.context_lens.numel(), max_context_len, q.dtype, q.device
                        )
                        return dequant_kvcache_fp8_gather_decode(
                            k_cache,
                            v_cache,
                            self.k_scale_cache,
                            self.v_scale_cache,
                            context.block_tables,
                            context.context_lens,
                            q.dtype,
                            k_workspace,
                            v_workspace,
                        )
                    if kv_profile:
                        k_cache, v_cache = timed_kv_cache_profile("fp8_decode_gather_dequant", dequant)
                    else:
                        k_cache, v_cache = dequant()
                    decode_block_table = None
                else:
                    dequant = lambda: dequant_kvcache_fp8_slice(
                        k_cache, v_cache, self.k_scale_cache, self.v_scale_cache
                    )
                    if kv_profile:
                        k_cache, v_cache = timed_kv_cache_profile("fp8_decode_dequant_full_cache", dequant)
                    else:
                        k_cache, v_cache = dequant()
            flash_decode = lambda: flash_attn_with_kvcache(
                q.unsqueeze(1), k_cache, v_cache, cache_seqlens=context.context_lens,
                block_table=decode_block_table, softmax_scale=self.scale, causal=True,
            )
            if kv_profile:
                o = timed_kv_cache_profile("decode_flash_attn", flash_decode)
            else:
                o = flash_decode()
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
