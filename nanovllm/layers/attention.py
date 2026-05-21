"""Multi-head attention with paged KV, FlashAttention, and mixed KV cache."""

import os

import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from torch import nn

from nanovllm.layers.fp8_paged_attention import k_int8_v_fp8_paged_attention_decode
from nanovllm.layers.kv_cache_kernels import (
    dequant_fp8_kvcache_gather_decode,
    dequant_fp8_kvcache_slice,
    dequant_k_int8_vcache_fp8_gather_decode,
    dequant_k_int8_vcache_fp8_slice,
    store_kvcache_fp8_triton,
    store_kvcache_k_int8_v_fp8_triton,
)
from nanovllm.utils.context import get_context
from nanovllm.utils.kv_cache import k_int8_group_size_from_env, k_int8_recent_bf16_window_from_env
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
        self.k_bf16_shadow_cache = torch.tensor([])
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
        k_bf16_shadow_cache = self.k_bf16_shadow_cache
        v_fp8_cache = v_cache.dtype == torch.float8_e4m3fn if v_cache.numel() else False
        k_int8_cache = k_cache.dtype == torch.int8 if k_cache.numel() else False
        mixed_kv_cache = k_int8_cache and v_fp8_cache
        full_fp8_kv_cache = v_fp8_cache and k_cache.dtype == torch.float8_e4m3fn if k_cache.numel() else False
        kv_profile = os.environ.get("NANOVLLM_KV_CACHE_PROFILE", "0") == "1"
        fp8_decode_backend = os.environ.get("NANOVLLM_FP8_KV_DECODE", "native")
        if k_cache.numel() and v_cache.numel():
            if mixed_kv_cache:
                k_group_size = k_int8_group_size_from_env(self.head_dim)
                store = lambda: store_kvcache_k_int8_v_fp8_triton(
                    k,
                    v,
                    k_cache,
                    v_cache,
                    self.k_scale_cache,
                    self.v_scale_cache,
                    context.slot_mapping,
                    k_group_size=k_group_size,
                )
                if kv_profile:
                    timed_kv_cache_profile("k_int8_v_fp8_store_triton", store)
                else:
                    store()
                if k_bf16_shadow_cache.numel():
                    shadow_store = lambda: store_kvcache(k, k, k_bf16_shadow_cache, k_bf16_shadow_cache, context.slot_mapping)
                    if kv_profile:
                        timed_kv_cache_profile("k_bf16_shadow_store", shadow_store)
                    else:
                        shadow_store()
            elif full_fp8_kv_cache:
                store = lambda: store_kvcache_fp8_triton(
                    k,
                    v,
                    k_cache,
                    v_cache,
                    self.k_scale_cache,
                    self.v_scale_cache,
                    context.slot_mapping,
                )
                if kv_profile:
                    timed_kv_cache_profile("fp8_kv_store_triton", store)
                else:
                    store()
            else:
                store = lambda: store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
                if kv_profile:
                    timed_kv_cache_profile("bf16_store", store)
                else:
                    store()
        if context.is_prefill:
            if context.block_tables is not None:  # prefix cache
                if mixed_kv_cache or full_fp8_kv_cache:
                    def dequant():
                        if context.block_tables.shape[0] == 1:
                            context_lens = torch.tensor(
                                [context.max_seqlen_k], dtype=torch.int32, device=q.device
                            )
                            k_workspace, v_workspace = self._get_fp8_kv_gather_workspace(
                                1, context.max_seqlen_k, q.dtype, q.device
                            )
                            gather_fn = dequant_k_int8_vcache_fp8_gather_decode if mixed_kv_cache else dequant_fp8_kvcache_gather_decode
                            return gather_fn(
                                k_cache,
                                v_cache,
                                self.k_scale_cache,
                                self.v_scale_cache,
                                context.block_tables,
                                context_lens,
                                q.dtype,
                                k_workspace,
                                v_workspace,
                            )
                        if mixed_kv_cache:
                            return dequant_k_int8_vcache_fp8_slice(
                                k_cache, v_cache, self.k_scale_cache, self.v_scale_cache
                            )
                        return dequant_fp8_kvcache_slice(
                            k_cache, v_cache, self.k_scale_cache, self.v_scale_cache
                        )

                    if kv_profile:
                        k_cache, v_cache = timed_kv_cache_profile("k_int8_v_fp8_prefill_dequant_full_cache", dequant)
                    else:
                        k_cache, v_cache = dequant()
                    if context.block_tables.shape[0] == 1:
                        k_cache = k_cache[0, : context.max_seqlen_k]
                        v_cache = v_cache[0, : context.max_seqlen_k]
                        block_tables = None
                    else:
                        block_tables = context.block_tables
                else:
                    block_tables = context.block_tables
                k, v = k_cache, v_cache
            else:
                block_tables = None
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
                block_table=block_tables,
            )
        else:  # decode
            decode_block_table = context.block_tables
            if mixed_kv_cache:
                if fp8_decode_backend == "native" and q.size(0) == 1 and context.context_lens.numel() == 1:
                    block_tokens = int(os.environ.get("NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS", "32"))

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
                    recent_k_window = k_int8_recent_bf16_window_from_env()
                    if recent_k_window > 0 and k_bf16_shadow_cache.numel():
                        context_lens_cpu = context.context_lens.tolist()
                        for batch_idx, context_len in enumerate(context_lens_cpu):
                            start = max(0, int(context_len) - recent_k_window)
                            if start >= int(context_len):
                                continue
                            token_offsets = torch.arange(
                                start,
                                int(context_len),
                                device=context.block_tables.device,
                                dtype=torch.int64,
                            )
                            block_size = k_bf16_shadow_cache.shape[1]
                            logical_blocks = token_offsets // block_size
                            block_offsets = token_offsets % block_size
                            physical_blocks = context.block_tables[batch_idx, logical_blocks].long()
                            k_cache[batch_idx, start:int(context_len)].copy_(
                                k_bf16_shadow_cache[physical_blocks, block_offsets]
                            )
                    decode_block_table = None
                else:
                    raise ValueError(
                        "NANOVLLM_FP8_KV_DECODE must be one of {'native', 'gather_dequant'} "
                        f"for k_int8_v_fp8, got {fp8_decode_backend!r}"
                    )
            elif full_fp8_kv_cache:
                def dequant():
                    max_context_len = int(context.context_lens.max().item())
                    k_workspace, v_workspace = self._get_fp8_kv_gather_workspace(
                        context.context_lens.numel(), max_context_len, q.dtype, q.device
                    )
                    return dequant_fp8_kvcache_gather_decode(
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
                    k_cache, v_cache = timed_kv_cache_profile("fp8_kv_decode_gather_dequant", dequant)
                else:
                    k_cache, v_cache = dequant()
                decode_block_table = None
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
