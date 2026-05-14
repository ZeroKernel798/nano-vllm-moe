from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_paged_attention_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_table_ptr,
    context_len_ptr,
    out_ptr,
    sm_scale: tl.constexpr,
    BLOCK_TABLE_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    head_idx = tl.program_id(0)
    offsets = tl.arange(0, HEAD_DIM)
    token_offsets = tl.arange(0, BLOCK_TOKENS)
    context_len = tl.load(context_len_ptr)
    kv_head_idx = head_idx // (NUM_HEADS // NUM_KV_HEADS)
    q = tl.load(q_ptr + head_idx * HEAD_DIM + offsets).to(tl.float32)

    m_i = tl.full((), -3.4028234663852886e38, tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((HEAD_DIM,), tl.float32)

    for token_block_start in range(0, context_len, BLOCK_TOKENS):
        token_idx = token_block_start + token_offsets
        valid_tokens = token_idx < context_len
        logical_block_idx = token_idx // BLOCK_SIZE
        block_offset = token_idx - logical_block_idx * BLOCK_SIZE
        physical_block = tl.load(
            block_table_ptr + logical_block_idx,
            mask=valid_tokens,
            other=0,
        )
        cache_offsets = ((physical_block[:, None] * BLOCK_SIZE + block_offset[:, None]) * NUM_KV_HEADS + kv_head_idx) * HEAD_DIM + offsets[None, :]
        scale_offsets = (physical_block * BLOCK_SIZE + block_offset) * NUM_KV_HEADS + kv_head_idx
        k = tl.load(k_cache_ptr + cache_offsets, mask=valid_tokens[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_cache_ptr + cache_offsets, mask=valid_tokens[:, None], other=0.0).to(tl.float32)
        k_scale = tl.load(k_scale_ptr + scale_offsets, mask=valid_tokens, other=0.0).to(tl.float32)
        v_scale = tl.load(v_scale_ptr + scale_offsets, mask=valid_tokens, other=0.0).to(tl.float32)
        k = k * k_scale[:, None]
        v = v * v_scale[:, None]
        scores = tl.sum(k * q[None, :], axis=1) * sm_scale
        scores = tl.where(valid_tokens, scores, -3.4028234663852886e38)
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
        l_i = l_new

    out = acc / l_i
    tl.store(out_ptr + head_idx * HEAD_DIM + offsets, out)


def fp8_paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_len: torch.Tensor,
    softmax_scale: float,
    block_tokens: int = 32,
) -> torch.Tensor:
    """Decode-only FP8 paged attention for one sequence.

    This isolated prototype supports ``q`` with shape ``[num_heads, head_dim]`` and FP8 KV cache
    with shape ``[num_blocks, block_size, num_kv_heads, head_dim]``. It returns one BF16/FP16 output
    row per query head and is intended to be compared against ``gather_dequant + FlashAttention``.
    """
    assert q.dim() == 2
    assert k_cache.dtype == torch.float8_e4m3fn and v_cache.dtype == torch.float8_e4m3fn
    assert k_cache.shape == v_cache.shape
    assert k_scale_cache.shape == k_cache.shape[:3]
    assert v_scale_cache.shape == v_cache.shape[:3]
    assert block_table.dim() == 1 and block_table.dtype == torch.int32
    assert context_len.numel() == 1 and context_len.dtype == torch.int32
    assert q.dtype in (torch.float16, torch.bfloat16)
    num_heads, head_dim = q.shape
    _, block_size, num_kv_heads, cache_head_dim = k_cache.shape
    assert head_dim == cache_head_dim
    assert num_heads % num_kv_heads == 0
    assert block_tokens in {16, 32, 64}
    out = torch.empty_like(q)
    _fp8_paged_attention_decode_kernel[(num_heads,)](
        q,
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        block_table,
        context_len,
        out,
        float(softmax_scale),
        block_table.stride(0),
        block_size,
        num_heads,
        num_kv_heads,
        head_dim,
        block_tokens,
    )
    return out


@triton.jit
def _k_int8_v_fp8_paged_attention_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_table_ptr,
    context_len_ptr,
    out_ptr,
    sm_scale: tl.constexpr,
    BLOCK_TABLE_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    K_GROUPS: tl.constexpr,
    K_GROUP_SIZE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    head_idx = tl.program_id(0)
    offsets = tl.arange(0, HEAD_DIM)
    token_offsets = tl.arange(0, BLOCK_TOKENS)
    context_len = tl.load(context_len_ptr)
    kv_head_idx = head_idx // (NUM_HEADS // NUM_KV_HEADS)
    q = tl.load(q_ptr + head_idx * HEAD_DIM + offsets).to(tl.float32)

    m_i = tl.full((), -3.4028234663852886e38, tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((HEAD_DIM,), tl.float32)

    for token_block_start in range(0, context_len, BLOCK_TOKENS):
        token_idx = token_block_start + token_offsets
        valid_tokens = token_idx < context_len
        logical_block_idx = token_idx // BLOCK_SIZE
        block_offset = token_idx - logical_block_idx * BLOCK_SIZE
        physical_block = tl.load(
            block_table_ptr + logical_block_idx,
            mask=valid_tokens,
            other=0,
        )
        cache_offsets = ((physical_block[:, None] * BLOCK_SIZE + block_offset[:, None]) * NUM_KV_HEADS + kv_head_idx) * HEAD_DIM + offsets[None, :]
        scale_base = (physical_block * BLOCK_SIZE + block_offset) * NUM_KV_HEADS + kv_head_idx
        k_scale_offsets = scale_base[:, None] * K_GROUPS + offsets[None, :] // K_GROUP_SIZE
        v_scale_offsets = scale_base
        k = tl.load(k_cache_ptr + cache_offsets, mask=valid_tokens[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_cache_ptr + cache_offsets, mask=valid_tokens[:, None], other=0.0).to(tl.float32)
        k_scale = tl.load(k_scale_ptr + k_scale_offsets, mask=valid_tokens[:, None], other=0.0).to(tl.float32)
        v_scale = tl.load(v_scale_ptr + v_scale_offsets, mask=valid_tokens, other=0.0).to(tl.float32)
        k = k * k_scale
        v = v * v_scale[:, None]
        scores = tl.sum(k * q[None, :], axis=1) * sm_scale
        scores = tl.where(valid_tokens, scores, -3.4028234663852886e38)
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
        l_i = l_new

    out = acc / l_i
    tl.store(out_ptr + head_idx * HEAD_DIM + offsets, out)


def k_int8_v_fp8_paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_len: torch.Tensor,
    softmax_scale: float,
    block_tokens: int = 32,
) -> torch.Tensor:
    assert q.dim() == 2
    assert k_cache.dtype == torch.int8 and v_cache.dtype == torch.float8_e4m3fn
    assert k_cache.shape == v_cache.shape
    assert v_scale_cache.shape == v_cache.shape[:3]
    assert block_table.dim() == 1 and block_table.dtype == torch.int32
    assert context_len.numel() == 1 and context_len.dtype == torch.int32
    assert q.dtype in (torch.float16, torch.bfloat16)
    num_heads, head_dim = q.shape
    _, block_size, num_kv_heads, cache_head_dim = k_cache.shape
    k_groups = k_scale_cache.shape[-1]
    assert head_dim == cache_head_dim
    assert k_scale_cache.shape == (*k_cache.shape[:3], k_groups)
    assert head_dim % k_groups == 0
    assert num_heads % num_kv_heads == 0
    assert block_tokens in {16, 32, 64}
    out = torch.empty_like(q)
    _k_int8_v_fp8_paged_attention_decode_kernel[(num_heads,)](
        q,
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        block_table,
        context_len,
        out,
        float(softmax_scale),
        block_table.stride(0),
        block_size,
        num_heads,
        num_kv_heads,
        head_dim,
        k_groups,
        head_dim // k_groups,
        block_tokens,
    )
    return out
