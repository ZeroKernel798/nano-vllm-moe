"""Accuracy suite for mixed K-int8/V-FP8 KV cache.

Runs BF16 KV and mixed KV on deterministic random token prompts across seeds/prompt ids, then
summarizes divergence, logits similarity, top-k overlap, and decode timing. This script is intended
to build a stable correctness baseline for the maintained mixed KV path; it does not require
token-exact match for long autoregressive outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cache_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size()) if tensor.numel() else 0


def _timed_cuda(fn):
    _cuda_sync()
    start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    if start is None:
        import time

        t0 = time.perf_counter()
        result = fn()
        return result, (time.perf_counter() - t0) * 1000.0
    start.record()
    result = fn()
    end.record()
    _cuda_sync()
    return result, float(start.elapsed_time(end))


def _run_trace(args: argparse.Namespace, prompt_token_ids: list[int], kv_cache_dtype: str, experimental: bool) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    old_decode_backend = os.environ.get("NANOVLLM_FP8_KV_DECODE")
    old_block_tokens = os.environ.get("NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS")
    if experimental:
        os.environ["NANOVLLM_FP8_KV_DECODE"] = args.fp8_decode_backend
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

        steps: list[dict[str, Any]] = []
        logits_trace: list[torch.Tensor] = []
        generated: list[int] = []
        while not llm.scheduler.is_finished():
            seqs, is_prefill = llm.scheduler.schedule()
            phase = "prefill" if is_prefill else "decode"
            if is_prefill:
                (input_ids, positions), prepare_ms = _timed_cuda(lambda: llm.model_runner.prepare_prefill(seqs))
            else:
                (input_ids, positions), prepare_ms = _timed_cuda(lambda: llm.model_runner.prepare_decode(seqs))
            logits, model_ms = _timed_cuda(lambda: llm.model_runner.run_model(input_ids, positions, is_prefill))
            temperatures, sample_prepare_ms = _timed_cuda(lambda: llm.model_runner.prepare_sample(seqs))
            token_ids, sample_ms = _timed_cuda(lambda: llm.model_runner.sampler(logits, temperatures).tolist())
            logits_trace.append(logits[-len(seqs) :].detach().float().cpu())
            generated.extend(int(token_id) for token_id in token_ids)
            llm.scheduler.postprocess(seqs, token_ids, is_prefill)
            reset_context()
            steps.append(
                {
                    "phase": phase,
                    "input_tokens": int(input_ids.numel()),
                    "model_ms": model_ms,
                    "prepare_ms": prepare_ms,
                    "sample_prepare_ms": sample_prepare_ms,
                    "sample_ms": sample_ms,
                    "token_ids": [int(token_id) for token_id in token_ids],
                }
            )

        profile = llm.model_runner.call("get_kv_cache_profile")
        num_blocks = int(llm.model_runner.config.num_kvcache_blocks)
        kv_data_bytes = _cache_bytes(llm.model_runner.kv_cache)
        if not kv_data_bytes:
            kv_data_bytes = _cache_bytes(llm.model_runner.k_cache_storage) + _cache_bytes(llm.model_runner.v_cache_storage)
        scale_bytes = _cache_bytes(llm.model_runner.k_scale_cache) + _cache_bytes(llm.model_runner.v_scale_cache)
        return {
            "kv_cache_dtype": kv_cache_dtype,
            "fp8_decode_backend": args.fp8_decode_backend if experimental else None,
            "native_block_tokens": args.native_block_tokens if experimental else None,
            "generated_token_ids": generated,
            "steps": steps,
            "kv_cache_profile": profile,
            "num_kvcache_blocks": num_blocks,
            "kv_cache_total_storage_bytes": kv_data_bytes + scale_bytes,
            "kv_cache_total_bytes_per_block": (kv_data_bytes + scale_bytes) / max(num_blocks, 1),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
            "logits_trace": logits_trace,
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


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    return len(set(a.topk(k).indices.tolist()) & set(b.topk(k).indices.tolist())) / float(k)


def _compare(bf16: dict[str, Any], fp8: dict[str, Any], top_k: int) -> dict[str, Any]:
    bf16_tokens = bf16["generated_token_ids"]
    fp8_tokens = fp8["generated_token_ids"]
    n_tokens = min(len(bf16_tokens), len(fp8_tokens))
    token_matches = [bf16_tokens[i] == fp8_tokens[i] for i in range(n_tokens)]
    first_mismatch = next((i for i, ok in enumerate(token_matches) if not ok), None)

    rows: list[dict[str, Any]] = []
    for step_idx, (bf16_step, fp8_step) in enumerate(zip(bf16["logits_trace"], fp8["logits_trace"])):
        row_count = min(bf16_step.shape[0], fp8_step.shape[0])
        for row_idx in range(row_count):
            a = bf16_step[row_idx]
            b = fp8_step[row_idx]
            diff = (a - b).abs()
            rows.append(
                {
                    "step": step_idx,
                    "row": row_idx,
                    "cosine": float(F.cosine_similarity(a, b, dim=0).item()),
                    "max_abs": float(diff.max().item()),
                    "mean_abs": float(diff.mean().item()),
                    "top_k_overlap": _topk_overlap(a, b, top_k),
                    "argmax_match": bool(a.argmax().item() == b.argmax().item()),
                    "before_divergence": first_mismatch is None or step_idx <= first_mismatch,
                }
            )
    before_rows = [row for row in rows if row["before_divergence"]]

    def avg(items: list[dict[str, Any]], key: str) -> float:
        return mean(float(item[key]) for item in items) if items else 0.0

    return {
        "token_match_rate": sum(token_matches) / max(n_tokens, 1),
        "exact_match": len(bf16_tokens) == len(fp8_tokens) and all(token_matches),
        "first_mismatch_index": first_mismatch,
        "num_logits_rows": len(rows),
        "cosine_mean": avg(rows, "cosine"),
        "cosine_min": min((float(row["cosine"]) for row in rows), default=0.0),
        "top_k_overlap_mean": avg(rows, "top_k_overlap"),
        "argmax_match_rate": sum(1 for row in rows if row["argmax_match"]) / max(len(rows), 1),
        "pre_divergence_rows": len(before_rows),
        "pre_divergence_cosine_mean": avg(before_rows, "cosine"),
        "pre_divergence_top_k_overlap_mean": avg(before_rows, "top_k_overlap"),
        "pre_divergence_argmax_match_rate": sum(1 for row in before_rows if row["argmax_match"]) / max(len(before_rows), 1),
    }


def _decode_summary(run: dict[str, Any]) -> dict[str, float]:
    decode_steps = [step for step in run["steps"] if step["phase"] == "decode"]
    model_ms = sum(float(step["model_ms"]) for step in decode_steps)
    total_ms = sum(
        float(step["model_ms"] + step["prepare_ms"] + step["sample_prepare_ms"] + step["sample_ms"])
        for step in decode_steps
    )
    tokens = len(decode_steps)
    return {
        "decode_tokens": float(tokens),
        "model_tps": tokens / (model_ms / 1000.0) if model_ms else 0.0,
        "total_tps": tokens / (total_ms / 1000.0) if total_ms else 0.0,
        "model_ms_per_token": model_ms / max(tokens, 1),
        "total_ms_per_token": total_ms / max(tokens, 1),
    }


def _strip(run: dict[str, Any]) -> dict[str, Any]:
    out = dict(run)
    out.pop("logits_trace", None)
    return out


def _make_prompt(seed: int, prompt_id: int, input_len: int, vocab_range: int) -> list[int]:
    rng = random.Random(seed * 1000003 + prompt_id * 9176)
    return [rng.randint(0, vocab_range) for _ in range(input_len)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-len", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=2304)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2304)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--kv-cache-dtype", default="k_int8_v_fp8", choices=("k_int8_v_fp8",))
    parser.add_argument("--kv-cache-scale-dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--fp8-decode-backend", default="native", choices=("native", "gather_dequant"))
    parser.add_argument("--native-block-tokens", type=int, default=32, choices=(16, 32, 64))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--prompt-ids", default="0,1,2")
    parser.add_argument("--vocab-range", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    args.model_path = os.path.expanduser(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(item) for item in args.seeds.split(",") if item]
    prompt_ids = [int(item) for item in args.prompt_ids.split(",") if item]

    records = []
    for seed in seeds:
        for prompt_id in prompt_ids:
            prompt = _make_prompt(seed, prompt_id, args.input_len, args.vocab_range)
            bf16 = _run_trace(args, prompt, "bf16", False)
            quant = _run_trace(args, prompt, args.kv_cache_dtype, True)
            comparison = _compare(bf16, quant, args.top_k)
            record = {
                "seed": seed,
                "prompt_id": prompt_id,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "comparison": comparison,
                "bf16_decode": _decode_summary(bf16),
                "quant_decode": _decode_summary(quant),
                "bf16": _strip(bf16),
                args.kv_cache_dtype: _strip(quant),
            }
            records.append(record)
            (output_dir / f"case_seed{seed}_prompt{prompt_id}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    def avg(path: list[str]) -> float:
        vals = []
        for record in records:
            item: Any = record
            for key in path:
                item = item[key]
            vals.append(float(item))
        return mean(vals) if vals else 0.0

    first_mismatches = [record["comparison"]["first_mismatch_index"] for record in records]
    summary = {
        "num_cases": len(records),
        "input_len": args.input_len,
        "output_len": args.output_len,
        "seeds": seeds,
        "prompt_ids": prompt_ids,
        "kv_cache_dtype": args.kv_cache_dtype,
        "fp8_decode_backend": args.fp8_decode_backend,
        "native_block_tokens": args.native_block_tokens,
        "exact_match_rate": sum(1 for record in records if record["comparison"]["exact_match"]) / max(len(records), 1),
        "token_match_rate_mean": avg(["comparison", "token_match_rate"]),
        "first_mismatch_indices": first_mismatches,
        "cosine_mean": avg(["comparison", "cosine_mean"]),
        "top_k_overlap_mean": avg(["comparison", "top_k_overlap_mean"]),
        "argmax_match_rate_mean": avg(["comparison", "argmax_match_rate"]),
        "pre_divergence_cosine_mean": avg(["comparison", "pre_divergence_cosine_mean"]),
        "pre_divergence_top_k_overlap_mean": avg(["comparison", "pre_divergence_top_k_overlap_mean"]),
        "pre_divergence_argmax_match_rate_mean": avg(["comparison", "pre_divergence_argmax_match_rate"]),
        "bf16_model_tps_mean": avg(["bf16_decode", "model_tps"]),
        "quant_model_tps_mean": avg(["quant_decode", "model_tps"]),
        "model_tps_ratio_quant_over_bf16": avg(["quant_decode", "model_tps"]) / max(avg(["bf16_decode", "model_tps"]), 1.0e-9),
        "total_bytes_per_block_ratio_quant_over_bf16": avg([args.kv_cache_dtype, "kv_cache_total_bytes_per_block"]) / max(avg(["bf16", "kv_cache_total_bytes_per_block"]), 1.0e-9),
        "block_ratio_quant_over_bf16": avg([args.kv_cache_dtype, "num_kvcache_blocks"]) / max(avg(["bf16", "num_kvcache_blocks"]), 1.0e-9),
        "records": records,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    (output_dir / "summary.json").write_text(text + "\n", encoding="utf-8")
    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
