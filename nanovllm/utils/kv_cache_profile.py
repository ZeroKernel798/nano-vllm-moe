from __future__ import annotations

from collections import defaultdict
from time import perf_counter
from typing import Any, Callable, TypeVar

import torch

T = TypeVar("T")

_PROFILE: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "total_s": 0.0})


def reset_kv_cache_profile() -> None:
    _PROFILE.clear()


def record_kv_cache_profile(name: str, elapsed_s: float) -> None:
    item = _PROFILE[name]
    item["count"] += 1.0
    item["total_s"] += float(elapsed_s)


def get_kv_cache_profile() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, item in sorted(_PROFILE.items()):
        count = int(item["count"])
        total_s = float(item["total_s"])
        out[name] = {
            "count": count,
            "total_s": total_s,
            "avg_ms": total_s * 1000.0 / count if count else 0.0,
        }
    return out


def timed_kv_cache_profile(name: str, fn: Callable[[], T]) -> T:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    record_kv_cache_profile(name, perf_counter() - start)
    return result
