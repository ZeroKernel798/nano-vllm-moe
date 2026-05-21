"""Run group-size KV quality gates with one process per backend trace."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def _split_ints(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def _run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_cmd(args: argparse.Namespace, backend: str, input_len: int, seed: int, out_pt: Path, out_json: Path) -> list[str]:
    max_model_len = args.max_model_len
    max_num_batched_tokens = args.max_num_batched_tokens
    if input_len == args.long_context_len and args.output_len > 0:
        max_model_len = max(max_model_len, input_len)
        max_num_batched_tokens = max(max_num_batched_tokens, input_len)
    return [
        sys.executable,
        str(SCRIPT_DIR / "kv_cache_logits_dump.py"),
        "--model-path",
        args.model_path,
        "--backend",
        backend,
        "--input-len",
        str(input_len),
        "--output-len",
        str(args.output_len),
        "--max-model-len",
        str(max_model_len),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--kv-cache-scale-dtype",
        args.kv_cache_scale_dtype,
        "--native-block-tokens",
        str(args.native_block_tokens),
        "--k-group-size",
        str(args.k_group_size),
        "--gather-block-tokens",
        str(args.gather_block_tokens),
        "--seed",
        str(seed),
        "--vocab-range",
        str(args.vocab_range),
        "--output-pt",
        str(out_pt),
        "--output-json",
        str(out_json),
    ]


def _compare_cmd(left_pt: Path, right_pt: Path, out_json: Path, top_k: int) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "compare_logits_pt.py"),
        "--left-pt",
        str(left_pt),
        "--right-pt",
        str(right_pt),
        "--top-k",
        str(top_k),
        "--output-json",
        str(out_json),
    ]


def _case(args: argparse.Namespace, input_len: int, seed: int, backend: str, output_dir: Path) -> dict[str, Any]:
    stem = f"len{input_len}_seed{seed}_{backend}_g{args.k_group_size}"
    bf16_pt = output_dir / f"len{input_len}_seed{seed}_bf16.pt"
    bf16_json = output_dir / f"len{input_len}_seed{seed}_bf16.json"
    if not bf16_pt.exists() or not bf16_json.exists():
        code = _run(
            _dump_cmd(args, "bf16", input_len, seed, bf16_pt, bf16_json),
            output_dir / f"len{input_len}_seed{seed}_bf16.log",
        )
        if code != 0:
            return {
                "ok": False,
                "input_len": input_len,
                "seed": seed,
                "backend": backend,
                "stage": "bf16_dump",
                "returncode": code,
                "log_path": str(output_dir / f"len{input_len}_seed{seed}_bf16.log"),
            }

    quant_pt = output_dir / f"{stem}.pt"
    quant_json = output_dir / f"{stem}.json"
    code = _run(_dump_cmd(args, backend, input_len, seed, quant_pt, quant_json), output_dir / f"{stem}.log")
    if code != 0:
        return {
            "ok": False,
            "input_len": input_len,
            "seed": seed,
            "backend": backend,
            "stage": "quant_dump",
            "returncode": code,
            "log_path": str(output_dir / f"{stem}.log"),
        }

    compare_json = output_dir / f"bf16_vs_{stem}.json"
    code = _run(_compare_cmd(bf16_pt, quant_pt, compare_json, args.top_k), output_dir / f"bf16_vs_{stem}.log")
    if code != 0:
        return {
            "ok": False,
            "input_len": input_len,
            "seed": seed,
            "backend": backend,
            "stage": "compare",
            "returncode": code,
            "log_path": str(output_dir / f"bf16_vs_{stem}.log"),
        }

    comparison = _load_json(compare_json)
    quant_meta = _load_json(quant_json)
    assert comparison is not None and quant_meta is not None
    token_cmp = comparison["token_comparison"]
    logits_cmp = comparison["logits_comparison"]
    return {
        "ok": True,
        "input_len": input_len,
        "seed": seed,
        "backend": backend,
        "k_group_size": args.k_group_size,
        "exact_match": token_cmp["exact_match"],
        "token_match_rate": token_cmp["token_match_rate"],
        "first_mismatch_index": token_cmp["first_mismatch_index"],
        "cosine_min": logits_cmp["cosine_min"],
        "cosine_mean": logits_cmp["cosine_mean"],
        "argmax_match_rate": logits_cmp["argmax_match_rate"],
        "top_k_overlap_mean": logits_cmp["top_k_overlap_mean"],
        "num_kvcache_blocks": quant_meta["num_kvcache_blocks"],
        "kv_cache_total_bytes_per_block": quant_meta["kv_cache_total_bytes_per_block"],
        "comparison_json": str(compare_json),
        "quant_json": str(quant_json),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run K-int8/V-FP8 group-size logits gates")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-lens", default="8192,12288,16384")
    parser.add_argument("--long-context-len", type=int, default=32768)
    parser.add_argument("--output-len", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--backends", default="native,gather_dequant")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--kv-cache-scale-dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--k-group-size", type=int, default=16)
    parser.add_argument("--native-block-tokens", type=int, default=32, choices=(16, 32, 64))
    parser.add_argument("--gather-block-tokens", type=int, default=16, choices=(1, 2, 4, 8, 16, 32, 64))
    parser.add_argument("--vocab-range", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    args.model_path = str(Path(args.model_path).expanduser())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for input_len in _split_ints(args.input_lens):
        for seed in _split_ints(args.seeds):
            for backend in [item.strip() for item in args.backends.split(",") if item.strip()]:
                row = _case(args, input_len, seed, backend, output_dir)
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = {"rows": rows}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fieldnames = [
        "ok",
        "input_len",
        "seed",
        "backend",
        "k_group_size",
        "exact_match",
        "token_match_rate",
        "first_mismatch_index",
        "cosine_min",
        "cosine_mean",
        "argmax_match_rate",
        "top_k_overlap_mean",
        "num_kvcache_blocks",
        "kv_cache_total_bytes_per_block",
        "stage",
        "returncode",
        "log_path",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
