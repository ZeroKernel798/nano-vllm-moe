"""Logits-level validation for the maintained K-int8/V-FP8 KV cache path."""

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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context


BACKENDS = {"native", "gather_dequant"}


def _cache_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size()) if tensor.numel() else 0


def _timed_cuda(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, perf_counter() - start


def _run_trace(
    args: argparse.Namespace,
    prompt_token_ids: list[int],
    kv_cache_dtype: str,
    experimental: bool,
    decode_backend: str | None = None,
) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    old_decode_backend = os.environ.get("NANOVLLM_FP8_KV_DECODE")
    old_block_tokens = os.environ.get("NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS")
    if experimental:
        if decode_backend not in BACKENDS:
            raise ValueError(f"Unsupported mixed KV decode backend: {decode_backend!r}")
        os.environ["NANOVLLM_FP8_KV_DECODE"] = decode_backend
        os.environ["NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS"] = str(args.native_block_tokens)

    llm = None
    try:
        llm = LLM(
            args.model_path,
            enforce_eager=True,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tp_size=1,
            ep_size=1,
            kv_cache_dtype=kv_cache_dtype,
            experimental_kv_cache_fp8=experimental,
            kv_cache_scale_dtype=args.kv_cache_scale_dtype,
        )

        sampling_params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.output_len)
        seq = Sequence(prompt_token_ids, sampling_params)
        llm.scheduler.add(seq)
        llm.model_runner.call("reset_kv_cache_profile")

        step_records: list[dict[str, Any]] = []
        logits_trace: list[torch.Tensor] = []
        generated: list[int] = []
        total_start = perf_counter()

        while not llm.scheduler.is_finished():
            seqs, is_prefill = llm.scheduler.schedule()
            phase = "prefill" if is_prefill else "decode"
            if is_prefill:
                (input_ids, positions), prepare_time = _timed_cuda(lambda: llm.model_runner.prepare_prefill(seqs))
            else:
                (input_ids, positions), prepare_time = _timed_cuda(lambda: llm.model_runner.prepare_decode(seqs))
            logits, model_time = _timed_cuda(lambda: llm.model_runner.run_model(input_ids, positions, is_prefill))
            if logits is None:
                raise RuntimeError("logits are None; this script expects tp_size=ep_size=1")
            temperatures, sample_prepare_time = _timed_cuda(lambda: llm.model_runner.prepare_sample(seqs))
            token_ids, sample_time = _timed_cuda(lambda: llm.model_runner.sampler(logits, temperatures).tolist())
            logits_trace.append(logits[-len(seqs) :].detach().float().cpu())
            generated.extend(int(token_id) for token_id in token_ids)

            post_start = perf_counter()
            llm.scheduler.postprocess(seqs, token_ids, is_prefill)
            reset_context()
            post_time = perf_counter() - post_start
            step_records.append(
                {
                    "phase": phase,
                    "input_tokens": int(input_ids.numel()),
                    "prepare_ms": prepare_time * 1000.0,
                    "model_ms": model_time * 1000.0,
                    "sample_prepare_ms": sample_prepare_time * 1000.0,
                    "sample_ms": sample_time * 1000.0,
                    "postprocess_ms": post_time * 1000.0,
                    "token_ids": [int(token_id) for token_id in token_ids],
                }
            )

        runner = llm.model_runner
        num_blocks = int(runner.config.num_kvcache_blocks)
        kv_data_bytes = _cache_bytes(runner.kv_cache)
        if kv_data_bytes == 0:
            kv_data_bytes = _cache_bytes(runner.k_cache_storage) + _cache_bytes(runner.v_cache_storage)
        scale_bytes = _cache_bytes(runner.k_scale_cache) + _cache_bytes(runner.v_scale_cache)
        result = {
            "kv_cache_dtype": kv_cache_dtype,
            "decode_backend": decode_backend if experimental else None,
            "native_block_tokens": args.native_block_tokens if experimental else None,
            "kv_cache_scale_dtype": args.kv_cache_scale_dtype,
            "num_kvcache_blocks": num_blocks,
            "kv_cache_data_storage_bytes": kv_data_bytes,
            "kv_cache_scale_storage_bytes": scale_bytes,
            "kv_cache_total_storage_bytes": kv_data_bytes + scale_bytes,
            "kv_cache_data_bytes_per_block": kv_data_bytes / max(num_blocks, 1),
            "kv_cache_total_bytes_per_block": (kv_data_bytes + scale_bytes) / max(num_blocks, 1),
            "generated_token_ids": generated,
            "steps": step_records,
            "total_time_s": perf_counter() - total_start,
            "kv_cache_profile": llm.model_runner.call("get_kv_cache_profile"),
            "logits_trace": logits_trace,
        }
        return result
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


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    return len(set(a.topk(k).indices.tolist()) & set(b.topk(k).indices.tolist())) / float(k)


def _compare_logits(left_trace: list[torch.Tensor], right_trace: list[torch.Tensor], top_k: int) -> dict[str, Any]:
    rows = []
    for step_idx, (left_step, right_step) in enumerate(zip(left_trace, right_trace)):
        for row_idx in range(min(left_step.shape[0], right_step.shape[0])):
            left = left_step[row_idx]
            right = right_step[row_idx]
            diff = (left - right).abs()
            rows.append(
                {
                    "step": step_idx,
                    "row": row_idx,
                    "cosine": float(F.cosine_similarity(left, right, dim=0).item()),
                    "max_abs": float(diff.max().item()),
                    "mean_abs": float(diff.mean().item()),
                    "top_k_overlap": _topk_overlap(left, right, top_k),
                    "argmax_match": bool(left.argmax().item() == right.argmax().item()),
                }
            )
    if not rows:
        return {"num_compared_rows": 0, "steps": []}

    def mean(name: str) -> float:
        return sum(float(row[name]) for row in rows) / len(rows)

    return {
        "num_compared_rows": len(rows),
        "cosine_mean": mean("cosine"),
        "cosine_min": min(float(row["cosine"]) for row in rows),
        "max_abs_max": max(float(row["max_abs"]) for row in rows),
        "mean_abs_mean": mean("mean_abs"),
        "top_k_overlap_mean": mean("top_k_overlap"),
        "argmax_match_rate": sum(1 for row in rows if row["argmax_match"]) / len(rows),
        "steps": rows,
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


def _strip_logits(result: dict[str, Any]) -> dict[str, Any]:
    out = dict(result)
    out.pop("logits_trace", None)
    return out


def _parse_backends(value: str) -> list[str]:
    backends = [backend.strip() for backend in value.split(",") if backend.strip()]
    unknown = sorted(set(backends) - BACKENDS)
    if unknown:
        raise ValueError(f"Unknown mixed KV decode backends: {unknown}; valid={sorted(BACKENDS)}")
    return backends


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--label", default="kv_cache_mixed_logits")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--kv-cache-scale-dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument(
        "--fp8-decode-backends",
        default="native,gather_dequant",
        help="Comma-separated mixed KV decode backends: native,gather_dequant",
    )
    parser.add_argument("--native-block-tokens", type=int, default=32, choices=(16, 32, 64))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-range", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    args.model_path = os.path.expanduser(args.model_path)

    random.seed(args.seed)
    prompt_token_ids = [random.randint(0, args.vocab_range) for _ in range(args.input_len)]
    bf16 = _run_trace(args, prompt_token_ids, "bf16", False)
    backends = _parse_backends(args.fp8_decode_backends)
    mixed_results = {
        backend: _run_trace(args, prompt_token_ids, "k_int8_v_fp8", True, backend)
        for backend in backends
    }
    comparisons = {
        backend: {
            "token_comparison": _compare_tokens(bf16["generated_token_ids"], result["generated_token_ids"]),
            "logits_comparison": _compare_logits(bf16["logits_trace"], result["logits_trace"], args.top_k),
        }
        for backend, result in mixed_results.items()
    }
    storage_ratios = {
        backend: {
            "data_bytes_per_block": result["kv_cache_data_bytes_per_block"] / bf16["kv_cache_data_bytes_per_block"],
            "total_bytes_per_block": result["kv_cache_total_bytes_per_block"] / bf16["kv_cache_total_bytes_per_block"],
            "block_count": result["num_kvcache_blocks"] / bf16["num_kvcache_blocks"],
        }
        for backend, result in mixed_results.items()
    }
    summary = {
        "label": args.label,
        "model_path": args.model_path,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "seed": args.seed,
        "top_k": args.top_k,
        "decode_backends": backends,
        "bf16": _strip_logits(bf16),
        "k_int8_v_fp8": {backend: _strip_logits(result) for backend, result in mixed_results.items()},
        "comparisons_vs_bf16": comparisons,
        "storage_ratios_quant_over_bf16": storage_ratios,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
