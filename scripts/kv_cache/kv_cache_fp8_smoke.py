"""Smoke-test experimental FP8 KV cache storage and decode backends.

This script compares BF16 KV cache against FP8 storage with a selectable decode backend. The
default FP8 backend is native paged decode for batch=1, with gather-dequant fallback for unsupported
shapes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm import LLM, SamplingParams


def _cuda_memory() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {}
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "free_bytes": int(free),
        "total_bytes": int(total),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _run_once(args: argparse.Namespace, kv_cache_dtype: str, experimental: bool) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    random.seed(args.seed)
    prompt_token_ids = [[random.randint(0, args.vocab_range) for _ in range(args.input_len)] for _ in range(args.num_seqs)]
    sampling_params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.output_len)
    old_decode_backend = os.environ.get("NANOVLLM_FP8_KV_DECODE")
    old_block_tokens = os.environ.get("NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS")
    if experimental:
        os.environ["NANOVLLM_FP8_KV_DECODE"] = args.fp8_decode_backend
        os.environ["NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS"] = str(args.native_block_tokens)

    llm = None
    try:
        before = _cuda_memory()
        llm = LLM(
            args.model_path,
            enforce_eager=True,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tp_size=args.tp_size,
            ep_size=args.ep_size,
            kv_cache_dtype=kv_cache_dtype,
            experimental_kv_cache_fp8=experimental,
            kv_cache_scale_dtype=args.kv_cache_scale_dtype,
        )
        after_load = _cuda_memory()
        t0 = perf_counter()
        output = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = perf_counter() - t0
        after_generate = _cuda_memory()

        runner = llm.model_runner
        config = runner.config
        kv_cache = runner.kv_cache
        scale_storage_bytes = 0
        if getattr(runner, "k_scale_cache", torch.tensor([])).numel():
            scale_storage_bytes += runner.k_scale_cache.numel() * runner.k_scale_cache.element_size()
        if getattr(runner, "v_scale_cache", torch.tensor([])).numel():
            scale_storage_bytes += runner.v_scale_cache.numel() * runner.v_scale_cache.element_size()
        kv_cache_data_bytes = int(kv_cache.numel() * kv_cache.element_size())
        return {
            "kv_cache_dtype": kv_cache_dtype,
            "kv_cache_scale_dtype": config.kv_cache_scale_dtype,
            "experimental_kv_cache_fp8": experimental,
            "fp8_decode_backend": args.fp8_decode_backend if experimental else None,
            "native_block_tokens": args.native_block_tokens if experimental else None,
            "wall_time_s": wall,
            "num_kvcache_blocks": int(config.num_kvcache_blocks),
            "kvcache_block_size": int(config.kvcache_block_size),
            "kv_cache_shape": list(kv_cache.shape),
            "kv_cache_torch_dtype": str(kv_cache.dtype),
            "kv_cache_element_size": int(kv_cache.element_size()),
            "kv_cache_data_storage_bytes": kv_cache_data_bytes,
            "kv_cache_scale_storage_bytes": int(scale_storage_bytes),
            "kv_cache_total_storage_bytes": int(kv_cache_data_bytes + scale_storage_bytes),
            "kv_cache_data_bytes_per_block": kv_cache_data_bytes / config.num_kvcache_blocks,
            "kv_cache_total_bytes_per_block": (kv_cache_data_bytes + scale_storage_bytes) / config.num_kvcache_blocks,
            "memory_before": before,
            "memory_after_load": after_load,
            "memory_after_generate": after_generate,
            "stats": output["stats"],
            "results": output["results"],
        }
    finally:
        if llm is not None:
            llm.exit()
            del llm
        if old_decode_backend is None:
            os.environ.pop("NANOVLLM_FP8_KV_DECODE", None)
        else:
            os.environ["NANOVLLM_FP8_KV_DECODE"] = old_decode_backend
        if old_block_tokens is None:
            os.environ.pop("NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS", None)
        else:
            os.environ["NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS"] = old_block_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _compare_results(bf16: list[list[int]], fp8: list[list[int]]) -> dict[str, Any]:
    total = 0
    same = 0
    first_mismatch: dict[str, int] | None = None
    for seq_idx, (left, right) in enumerate(zip(bf16, fp8)):
        for token_idx, (a, b) in enumerate(zip(left, right)):
            total += 1
            if a == b:
                same += 1
            elif first_mismatch is None:
                first_mismatch = {"seq_idx": seq_idx, "token_idx": token_idx, "bf16": int(a), "fp8": int(b)}
        if len(left) != len(right) and first_mismatch is None:
            first_mismatch = {"seq_idx": seq_idx, "token_idx": min(len(left), len(right)), "bf16_len": len(left), "fp8_len": len(right)}
    return {
        "num_sequences": len(bf16),
        "total_compared_tokens": total,
        "matching_tokens": same,
        "token_match_rate": (same / total) if total else 1.0,
        "exact_sequence_match": bf16 == fp8,
        "first_mismatch": first_mismatch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental FP8 KV cache memory/accuracy smoke")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--label", default="kv_cache_fp8_smoke")
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--kv-cache-scale-dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--fp8-decode-backend", default="native", choices=("native", "gather_dequant", "full_dequant"))
    parser.add_argument("--native-block-tokens", type=int, default=64, choices=(16, 32, 64))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-range", type=int, default=10000)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    args.model_path = os.path.expanduser(args.model_path)

    bf16 = _run_once(args, "bf16", False)
    fp8 = _run_once(args, "fp8_e4m3", True)
    comparison = _compare_results(bf16["results"], fp8["results"])
    summary = {
        "label": args.label,
        "model_path": args.model_path,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_seqs": args.num_seqs,
        "bf16": bf16,
        "fp8_e4m3": fp8,
        "comparison": comparison,
        "data_storage_ratio_fp8_over_bf16": fp8["kv_cache_data_storage_bytes"] / bf16["kv_cache_data_storage_bytes"],
        "total_storage_ratio_fp8_over_bf16": fp8["kv_cache_total_storage_bytes"] / bf16["kv_cache_total_storage_bytes"],
        "data_bytes_per_block_ratio_fp8_over_bf16": fp8["kv_cache_data_bytes_per_block"] / bf16["kv_cache_data_bytes_per_block"],
        "total_bytes_per_block_ratio_fp8_over_bf16": fp8["kv_cache_total_bytes_per_block"] / bf16["kv_cache_total_bytes_per_block"],
        "block_ratio_fp8_over_bf16": fp8["num_kvcache_blocks"] / bf16["num_kvcache_blocks"],
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
