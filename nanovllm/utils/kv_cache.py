"""KV cache dtype helpers and roadmap for quantized KV.

Why FlashAttention blocks a one-line FP8 switch
-----------------------------------------------
``flash_attn.flash_attn_with_kvcache`` / ``flash_attn_varlen_func`` (CUDA backend) expect
``k_cache`` / ``v_cache`` in **float16 or bfloat16** with contiguous last dimension. There is
no supported path in the stock wheel to pass **FP8 paged KV** into the same kernels without
either:

1. **Full-sequence dequant** bf16 → FA each decode step (**O(seq × layers)** extra work and
   bandwidth), which negates decode wins unless sequence length is tiny, or
2. **Native FP8 KV inside the attention kernel** (custom CUDA / upstream FlashAttention support),
   or
3. A **decode-only** attention implementation (e.g. Triton) that reads FP8 from the paged
   layout directly in registers.

So “reliable” GPU KV quantization in this codebase is a **multi-phase** project: memory
accounting and storage format first, then either upstream kernel support or a dedicated decode
path.

Recommended phases
-----------------
**Phase A — Infra (done here):** ``Config.kv_cache_dtype``, bytes-per-element for block budget,
runtime guard so only implemented modes run.

**Phase B — Storage format:** FP8 (or int8) buffers with the **same paged shape** as today
``[num_blocks, block_size, num_kv_heads, head_dim]``, plus optional scale tensor(s). Implement
``store_kvcache_*`` that quantizes on write (see ``layers/kv_cache_kernels.py`` sketch).

**Phase C — Prefill:** Prefix / long prefill may require **dequant** of cached KV to bf16 for
``flash_attn_varlen_func`` *or* a varlen kernel that accepts FP8 — measure cost; prefill is often
more tolerant than decode.

**Phase D — Decode:** Either integrate a library that supports FP8 KV + paged attention on your
GPU, or add a **batch=1 / small-batch** Triton decode that matches GQA layout and block tables.

Until Phase D is satisfied, keep ``kv_cache_dtype="bf16"`` for production inference.
"""

from __future__ import annotations

KV_CACHE_DTYPES_BF16 = frozenset({"bf16", "bfloat16"})
KV_CACHE_DTYPES_FP8 = frozenset({"fp8", "fp8_e4m3", "float8_e4m3fn"})


def normalize_kv_cache_dtype(name: str) -> str:
    n = name.strip().lower()
    if n in ("bf16", "bfloat16"):
        return "bf16"
    if n in KV_CACHE_DTYPES_FP8:
        return "fp8_e4m3"
    raise ValueError(f"Unknown kv_cache_dtype: {name!r}")


def kv_cache_bytes_per_element(kv_cache_dtype: str) -> int:
    """Storage bytes per scalar in one K or V element (per head dim component)."""
    n = normalize_kv_cache_dtype(kv_cache_dtype)
    if n == "bf16":
        return 2
    if n == "fp8_e4m3":
        return 1
    raise AssertionError


def kv_cache_runtime_supported(kv_cache_dtype: str) -> bool:
    """Whether the rest of nano-vllm (Attention + FlashAttention) can run this mode."""
    n = normalize_kv_cache_dtype(kv_cache_dtype)
    return n == "bf16"


def assert_kv_cache_runtime_supported(kv_cache_dtype: str) -> None:
    if not kv_cache_runtime_supported(kv_cache_dtype):
        raise NotImplementedError(
            f'kv_cache_dtype={kv_cache_dtype!r} is not implemented yet. '
            f'Only "bf16" works with the current FlashAttention path. '
            f'See nanovllm.utils.kv_cache module docstring for the rollout plan.'
        )
