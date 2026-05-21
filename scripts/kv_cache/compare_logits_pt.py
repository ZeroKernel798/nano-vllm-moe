"""Offline comparison for logits traces saved by kv_cache_logits_dump.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    return len(set(a.topk(k).indices.tolist()) & set(b.topk(k).indices.tolist())) / float(k)


def _compare_logits(left: torch.Tensor, right: torch.Tensor, top_k: int, atol: float, rtol: float) -> dict[str, Any]:
    rows = []
    row_count = min(left.shape[0], right.shape[0])
    for row_idx in range(row_count):
        a = left[row_idx].float()
        b = right[row_idx].float()
        diff = (a - b).abs()
        rows.append(
            {
                "row": row_idx,
                "cosine": float(F.cosine_similarity(a, b, dim=0).item()),
                "max_abs": float(diff.max().item()),
                "mean_abs": float(diff.mean().item()),
                "top_k_overlap": _topk_overlap(a, b, top_k),
                "argmax_match": bool(a.argmax().item() == b.argmax().item()),
                "allclose": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
            }
        )

    def mean(name: str) -> float:
        return sum(float(row[name]) for row in rows) / len(rows) if rows else 0.0

    return {
        "num_compared_rows": row_count,
        "shape_left": list(left.shape),
        "shape_right": list(right.shape),
        "allclose": bool(torch.allclose(left[:row_count].float(), right[:row_count].float(), atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
        "cosine_mean": mean("cosine"),
        "cosine_min": min((float(row["cosine"]) for row in rows), default=0.0),
        "max_abs_max": max((float(row["max_abs"]) for row in rows), default=0.0),
        "mean_abs_mean": mean("mean_abs"),
        "top_k_overlap_mean": mean("top_k_overlap"),
        "argmax_match_rate": sum(1 for row in rows if row["argmax_match"]) / max(len(rows), 1),
        "row_allclose_rate": sum(1 for row in rows if row["allclose"]) / max(len(rows), 1),
        "rows": rows,
    }


def _compare_tokens(left: list[int], right: list[int]) -> dict[str, Any]:
    n = min(len(left), len(right))
    matches = [left[i] == right[i] for i in range(n)]
    first_mismatch = next((i for i, ok in enumerate(matches) if not ok), None)
    return {
        "left_generated_token_ids": left,
        "right_generated_token_ids": right,
        "compared_tokens": n,
        "token_match_rate": sum(matches) / max(n, 1),
        "exact_match": len(left) == len(right) and all(matches),
        "first_mismatch_index": first_mismatch,
    }


def _load(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-pt", required=True)
    parser.add_argument("--right-pt", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    left = _load(args.left_pt)
    right = _load(args.right_pt)
    summary = {
        "left_path": args.left_pt,
        "right_path": args.right_pt,
        "left_metadata": left["metadata"],
        "right_metadata": right["metadata"],
        "prompt_match": left["prompt_token_ids"] == right["prompt_token_ids"],
        "token_comparison": _compare_tokens(left["generated_token_ids"], right["generated_token_ids"]),
        "logits_comparison": _compare_logits(left["logits"], right["logits"], args.top_k, args.atol, args.rtol),
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
