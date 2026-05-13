"""Logits-level validation for experimental FP8 KV cache storage.

This compares the default BF16 KV cache path against the intentionally slow FP8-store/BF16-dequant
path. It records per-step logits error, top-k overlap, generated-token match, coarse stage timings,
and KV cache storage accounting. The script is validation-only; FP8 decode is expected to be slower
until a native FP8 paged attention/read kernel exists.
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context


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


def _new_sequence(prompt_token_ids: list[int], sampling_params: SamplingParams) -> Sequence:
    return Sequence(prompt_token_ids, sampling_params)


def _run_trace(args: argparse.Namespace, prompt_token_ids: list[int], kv_cache_dtype: str, experimental: bool) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    before = _cuda_memory()
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
    after_load = _cuda_memory()

    sampling_params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.output_len)
    seq = _new_sequence(prompt_token_ids, sampling_params)
    llm.scheduler.add(seq)
    llm.model_runner.call("reset_kv_cache_profile")

    step_records: list[dict[str, Any]] = []
    logits_trace: list[torch.Tensor] = []
    generated: list[int] = []
    total_start = perf_counter()

    while not llm.scheduler.is_finished():
        seqs, is_prefill = llm.scheduler.schedule()
        phase = "prefill" if is_prefill else "decode"
        step: dict[str, Any] = {"phase": phase, "num_seqs": len(seqs)}

        if is_prefill:
            (input_ids, positions), prepare_time = _timed_cuda(lambda: llm.model_runner.prepare_prefill(seqs))
        else:
            (input_ids, positions), prepare_time = _timed_cuda(lambda: llm.model_runner.prepare_decode(seqs))
        step["input_tokens"] = int(input_ids.numel())
        step["prepare_ms"] = prepare_time * 1000.0

        logits, model_time = _timed_cuda(lambda: llm.model_runner.run_model(input_ids, positions, is_prefill))
        step["model_ms"] = model_time * 1000.0
        if logits is None:
            raise RuntimeError("logits are None on rank 0; this validation script expects tp_size=ep_size=1")

        temperatures, sample_prepare_time = _timed_cuda(lambda: llm.model_runner.prepare_sample(seqs))
        token_ids, sample_time = _timed_cuda(lambda: llm.model_runner.sampler(logits, temperatures).tolist())
        step["sample_prepare_ms"] = sample_prepare_time * 1000.0
        step["sample_ms"] = sample_time * 1000.0
        step["token_ids"] = [int(token_id) for token_id in token_ids]
        step["logit_rows"] = int(logits.shape[0])
        step["vocab_size"] = int(logits.shape[-1])
        logits_trace.append(logits[-len(seqs) :].detach().float().cpu())
        generated.extend(int(token_id) for token_id in token_ids)

        post_start = perf_counter()
        llm.scheduler.postprocess(seqs, token_ids)
        reset_context()
        step["postprocess_ms"] = (perf_counter() - post_start) * 1000.0
        step["total_ms"] = sum(
            step[name]
            for name in ("prepare_ms", "model_ms", "sample_prepare_ms", "sample_ms", "postprocess_ms")
        )
        step_records.append(step)

    total_time = perf_counter() - total_start
    after_generate = _cuda_memory()
    kv_cache_profile = llm.model_runner.call("get_kv_cache_profile")
    config = llm.model_runner.config
    kv_data_bytes = _cache_bytes(llm.model_runner.kv_cache)
    scale_bytes = _cache_bytes(llm.model_runner.k_scale_cache) + _cache_bytes(llm.model_runner.v_scale_cache)
    num_blocks = int(config.num_kvcache_blocks)
    result = {
        "kv_cache_dtype": kv_cache_dtype,
        "experimental_kv_cache_fp8": experimental,
        "kv_cache_scale_dtype": args.kv_cache_scale_dtype,
        "num_kvcache_blocks": num_blocks,
        "kv_cache_data_storage_bytes": kv_data_bytes,
        "kv_cache_scale_storage_bytes": scale_bytes,
        "kv_cache_total_storage_bytes": kv_data_bytes + scale_bytes,
        "kv_cache_data_bytes_per_block": kv_data_bytes / max(num_blocks, 1),
        "kv_cache_total_bytes_per_block": (kv_data_bytes + scale_bytes) / max(num_blocks, 1),
        "generated_token_ids": generated,
        "steps": step_records,
        "total_time_s": total_time,
        "memory_before": before,
        "memory_after_load": after_load,
        "memory_after_generate": after_generate,
        "kv_cache_profile": kv_cache_profile,
    }
    llm.exit()
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result["logits_trace"] = logits_trace
    return result


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    a_idx = set(a.topk(k).indices.tolist())
    b_idx = set(b.topk(k).indices.tolist())
    return len(a_idx & b_idx) / float(k)


def _compare_logits(bf16_trace: list[torch.Tensor], fp8_trace: list[torch.Tensor], top_k: int) -> dict[str, Any]:
    step_metrics = []
    count = min(len(bf16_trace), len(fp8_trace))
    for step_idx in range(count):
        bf16_step = bf16_trace[step_idx]
        fp8_step = fp8_trace[step_idx]
        row_count = min(bf16_step.shape[0], fp8_step.shape[0])
        for row_idx in range(row_count):
            bf16_logits = bf16_step[row_idx]
            fp8_logits = fp8_step[row_idx]
            diff = (bf16_logits - fp8_logits).abs()
            step_metrics.append(
                {
                    "step": step_idx,
                    "row": row_idx,
                    "cosine": float(F.cosine_similarity(bf16_logits, fp8_logits, dim=0).item()),
                    "max_abs": float(diff.max().item()),
                    "mean_abs": float(diff.mean().item()),
                    "top_k_overlap": _topk_overlap(bf16_logits, fp8_logits, top_k),
                    "bf16_argmax": int(bf16_logits.argmax().item()),
                    "fp8_argmax": int(fp8_logits.argmax().item()),
                    "argmax_match": bool(bf16_logits.argmax().item() == fp8_logits.argmax().item()),
                }
            )
    if not step_metrics:
        return {"num_compared_rows": 0, "steps": []}

    def mean(name: str) -> float:
        return sum(float(metric[name]) for metric in step_metrics) / len(step_metrics)

    return {
        "num_compared_rows": len(step_metrics),
        "cosine_mean": mean("cosine"),
        "cosine_min": min(float(metric["cosine"]) for metric in step_metrics),
        "max_abs_max": max(float(metric["max_abs"]) for metric in step_metrics),
        "mean_abs_mean": mean("mean_abs"),
        "top_k_overlap_mean": mean("top_k_overlap"),
        "argmax_match_rate": sum(1 for metric in step_metrics if metric["argmax_match"]) / len(step_metrics),
        "steps": step_metrics,
    }


def _strip_logits(result: dict[str, Any]) -> dict[str, Any]:
    result = dict(result)
    result.pop("logits_trace", None)
    return result


def _compare_tokens(bf16: list[int], fp8: list[int]) -> dict[str, Any]:
    n = min(len(bf16), len(fp8))
    matches = [bf16[i] == fp8[i] for i in range(n)]
    first_mismatch = next((i for i, ok in enumerate(matches) if not ok), None)
    return {
        "bf16_generated_token_ids": bf16,
        "fp8_generated_token_ids": fp8,
        "compared_tokens": n,
        "token_match_rate": sum(matches) / max(n, 1),
        "exact_match": len(bf16) == len(fp8) and all(matches),
        "first_mismatch_index": first_mismatch,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--label", default="kv_cache_fp8_logits")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--kv-cache-scale-dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-range", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    args.model_path = os.path.expanduser(args.model_path)

    random.seed(args.seed)
    prompt_token_ids = [random.randint(0, args.vocab_range) for _ in range(args.input_len)]
    bf16 = _run_trace(args, prompt_token_ids, "bf16", False)
    fp8 = _run_trace(args, prompt_token_ids, "fp8_e4m3", True)
    logits_comparison = _compare_logits(bf16["logits_trace"], fp8["logits_trace"], args.top_k)
    token_comparison = _compare_tokens(bf16["generated_token_ids"], fp8["generated_token_ids"])

    summary = {
        "label": args.label,
        "model_path": args.model_path,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "seed": args.seed,
        "top_k": args.top_k,
        "bf16": _strip_logits(bf16),
        "fp8_e4m3": _strip_logits(fp8),
        "token_comparison": token_comparison,
        "logits_comparison": logits_comparison,
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
