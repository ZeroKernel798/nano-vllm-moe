"""KV cache dtype helpers.

Only two storage modes are kept in the active code path:

* ``bf16``: production/reference FlashAttention KV cache.
* ``k_int8_v_fp8``: mixed KV quantization, with K stored as group-wise int8 and V stored as
  FP8 E4M3.
* ``fp8``: debug-only full FP8 KV with per-token/per-head K/V scales.

Full FP8 KV, V-only FP8, and fake-quant diagnostic paths were removed from the runtime. Their
reasoning is documented in ``opt/kv_cache_quant.md``: full FP8 K caused attention-score drift,
while V-only FP8 was useful for ablation but not enough for the final memory story.
"""

from __future__ import annotations

import os

KV_CACHE_DTYPES_BF16 = frozenset({"bf16", "bfloat16"})
KV_CACHE_DTYPES_K_INT8_V_FP8 = frozenset({"k_int8_v_fp8", "int8_k_fp8_v", "kv_int8_fp8"})
KV_CACHE_DTYPES_FP8 = frozenset({"fp8", "fp8_e4m3", "fp8_kv", "kv_fp8"})
KV_CACHE_SCALE_DTYPES = {
    "fp32": "float32",
    "float32": "float32",
    "fp16": "float16",
    "float16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
}


def normalize_kv_cache_dtype(name: str) -> str:
    n = name.strip().lower()
    if n in ("bf16", "bfloat16"):
        return "bf16"
    if n in KV_CACHE_DTYPES_K_INT8_V_FP8:
        return "k_int8_v_fp8"
    if n in KV_CACHE_DTYPES_FP8:
        return "fp8"
    raise ValueError(f"Unknown kv_cache_dtype: {name!r}")


def kv_cache_scale_dtype_bytes_per_element(kv_cache_scale_dtype: str) -> int:
    n = normalize_kv_cache_scale_dtype(kv_cache_scale_dtype)
    if n == "float32":
        return 4
    if n in {"float16", "bfloat16"}:
        return 2
    raise AssertionError(f"Unhandled kv_cache_scale_dtype: {n}")


def k_int8_group_size_from_env(head_dim: int) -> int:
    group_size = int(os.environ.get("NANOVLLM_K_INT8_GROUP_SIZE", "32"))
    if group_size <= 0 or head_dim % group_size != 0:
        raise ValueError(
            "NANOVLLM_K_INT8_GROUP_SIZE must be a positive divisor of head_dim, "
            f"got group_size={group_size}, head_dim={head_dim}."
        )
    return group_size


def k_int8_recent_bf16_window_from_env() -> int:
    window = int(os.environ.get("NANOVLLM_K_INT8_RECENT_BF16_WINDOW", "0"))
    if window < 0:
        raise ValueError(f"NANOVLLM_K_INT8_RECENT_BF16_WINDOW must be >= 0, got {window}.")
    return window


def normalize_kv_cache_scale_dtype(name: str) -> str:
    n = name.strip().lower()
    if n in KV_CACHE_SCALE_DTYPES:
        return KV_CACHE_SCALE_DTYPES[n]
    raise ValueError(f"Unknown kv_cache_scale_dtype: {name!r}")


def kv_cache_bytes_per_element(kv_cache_dtype: str) -> int:
    """Storage bytes per scalar in one K or V element (per head dim component)."""
    n = normalize_kv_cache_dtype(kv_cache_dtype)
    if n == "bf16":
        return 2
    if n == "k_int8_v_fp8":
        return 1
    if n == "fp8":
        return 1
    raise AssertionError


def kv_cache_runtime_supported(kv_cache_dtype: str) -> bool:
    """Whether the rest of nano-vllm (Attention + FlashAttention) can run this mode."""
    n = normalize_kv_cache_dtype(kv_cache_dtype)
    return n == "bf16"


def assert_kv_cache_runtime_supported(kv_cache_dtype: str, experimental_fp8: bool = False) -> None:
    if not kv_cache_runtime_supported(kv_cache_dtype) and not experimental_fp8:
        raise NotImplementedError(
            f'kv_cache_dtype={kv_cache_dtype!r} is not implemented yet. '
            f'Only "bf16" is production-supported by default. '
            f'Pass experimental_kv_cache_fp8=True to use the K-int8/V-FP8 mixed KV or debug full-FP8 path. '
            f'See opt/kv_cache_quant.md for the current route.'
        )
