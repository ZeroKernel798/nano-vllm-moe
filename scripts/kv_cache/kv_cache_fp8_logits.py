"""Logits-level validation for experimental FP8 KV cache storage.

This compares the default BF16 KV cache path against FP8-store decode backends. It records
per-step logits error, top-k overlap, generated-token match, coarse stage timings, and KV cache
storage accounting. The native backend reads FP8 KV directly in a paged attention prototype;
gather-dequant only dequantizes decode-visible tokens before FlashAttention; full-dequant
dequantizes the whole cache before FlashAttention.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
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


BACKENDS = {"native", "gather_dequant", "full_dequant"}
FAKE_FP8_STORE_MODES = {"k", "v", "both"}


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


def _run_trace(
    args: argparse.Namespace,
    prompt_token_ids: list[int],
    kv_cache_dtype: str,
    experimental: bool,
    fp8_decode_backend: str | None = None,
    fake_fp8_store_mode: str | None = None,
) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    old_decode_backend = os.environ.get("NANOVLLM_FP8_KV_DECODE")
    old_block_tokens = os.environ.get("NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS")
    old_fake_fp8_store = os.environ.get("NANOVLLM_KV_FAKE_FP8_STORE")
    old_fake_fp8_format = os.environ.get("NANOVLLM_KV_FAKE_FP8_FORMAT")
    old_fake_fp8_group_size = os.environ.get("NANOVLLM_KV_FAKE_FP8_GROUP_SIZE")
    os.environ["NANOVLLM_KV_FAKE_FP8_FORMAT"] = args.fake_fp8_format
    os.environ["NANOVLLM_KV_FAKE_FP8_GROUP_SIZE"] = str(args.fake_fp8_group_size)
    if fake_fp8_store_mode:
        os.environ["NANOVLLM_KV_FAKE_FP8_STORE"] = fake_fp8_store_mode
    else:
        os.environ.pop("NANOVLLM_KV_FAKE_FP8_STORE", None)
    if experimental:
        assert fp8_decode_backend is not None
        os.environ["NANOVLLM_FP8_KV_DECODE"] = fp8_decode_backend
        os.environ["NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS"] = str(args.native_block_tokens)

    llm = None
    try:
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
            llm.scheduler.postprocess(seqs, token_ids, is_prefill)
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
        runner = llm.model_runner
        config = runner.config
        kv_data_bytes = _cache_bytes(runner.kv_cache)
        if kv_data_bytes == 0 and hasattr(runner, "k_cache_storage") and hasattr(runner, "v_cache_storage"):
            kv_data_bytes = _cache_bytes(runner.k_cache_storage) + _cache_bytes(runner.v_cache_storage)
        scale_bytes = _cache_bytes(runner.k_scale_cache) + _cache_bytes(runner.v_scale_cache)
        num_blocks = int(config.num_kvcache_blocks)
        result = {
            "kv_cache_dtype": kv_cache_dtype,
            "experimental_kv_cache_fp8": experimental,
            "fp8_decode_backend": fp8_decode_backend if experimental else None,
            "fake_fp8_store_mode": fake_fp8_store_mode,
            "fake_fp8_format": args.fake_fp8_format if fake_fp8_store_mode else None,
            "fake_fp8_group_size": args.fake_fp8_group_size if fake_fp8_store_mode else None,
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
            "total_time_s": total_time,
            "memory_before": before,
            "memory_after_load": after_load,
            "memory_after_generate": after_generate,
            "kv_cache_profile": kv_cache_profile,
        }
        result["logits_trace"] = logits_trace
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
        if old_fake_fp8_store is None:
            os.environ.pop("NANOVLLM_KV_FAKE_FP8_STORE", None)
        else:
            os.environ["NANOVLLM_KV_FAKE_FP8_STORE"] = old_fake_fp8_store
        if old_fake_fp8_format is None:
            os.environ.pop("NANOVLLM_KV_FAKE_FP8_FORMAT", None)
        else:
            os.environ["NANOVLLM_KV_FAKE_FP8_FORMAT"] = old_fake_fp8_format
        if old_fake_fp8_group_size is None:
            os.environ.pop("NANOVLLM_KV_FAKE_FP8_GROUP_SIZE", None)
        else:
            os.environ["NANOVLLM_KV_FAKE_FP8_GROUP_SIZE"] = old_fake_fp8_group_size
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    a_idx = set(a.topk(k).indices.tolist())
    b_idx = set(b.topk(k).indices.tolist())
    return len(a_idx & b_idx) / float(k)


def _compare_logits(
    left_trace: list[torch.Tensor],
    right_trace: list[torch.Tensor],
    top_k: int,
    left_label: str = "left",
    right_label: str = "right",
) -> dict[str, Any]:
    step_metrics = []
    count = min(len(left_trace), len(right_trace))
    for step_idx in range(count):
        left_step = left_trace[step_idx]
        right_step = right_trace[step_idx]
        row_count = min(left_step.shape[0], right_step.shape[0])
        for row_idx in range(row_count):
            left_logits = left_step[row_idx]
            right_logits = right_step[row_idx]
            diff = (left_logits - right_logits).abs()
            left_argmax = int(left_logits.argmax().item())
            right_argmax = int(right_logits.argmax().item())
            step_metrics.append(
                {
                    "step": step_idx,
                    "row": row_idx,
                    "cosine": float(F.cosine_similarity(left_logits, right_logits, dim=0).item()),
                    "max_abs": float(diff.max().item()),
                    "mean_abs": float(diff.mean().item()),
                    "top_k_overlap": _topk_overlap(left_logits, right_logits, top_k),
                    "left_label": left_label,
                    "right_label": right_label,
                    "left_argmax": left_argmax,
                    "right_argmax": right_argmax,
                    "argmax_match": bool(left_argmax == right_argmax),
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


def _parse_backends(value: str) -> list[str]:
    backends = [backend.strip() for backend in value.split(",") if backend.strip()]
    unknown = sorted(set(backends) - BACKENDS)
    if unknown:
        raise ValueError(f"Unknown FP8 decode backends: {unknown}; valid={sorted(BACKENDS)}")
    return backends


def _parse_fake_modes(value: str) -> list[str]:
    modes = [mode.strip() for mode in value.split(",") if mode.strip()]
    unknown = sorted(set(modes) - FAKE_FP8_STORE_MODES)
    if unknown:
        raise ValueError(f"Unknown fake FP8 store modes: {unknown}; valid={sorted(FAKE_FP8_STORE_MODES)}")
    return modes


def _jsonable_for_torch_save(result: dict[str, Any]) -> dict[str, Any]:
    return result


def _run_single_and_save(args: argparse.Namespace, run_kind: str) -> None:
    random.seed(args.seed)
    prompt_token_ids = [random.randint(0, args.vocab_range) for _ in range(args.input_len)]
    if run_kind == "bf16":
        result = _run_trace(args, prompt_token_ids, "bf16", False)
    elif run_kind.startswith("fake_"):
        fake_mode = run_kind.removeprefix("fake_")
        if fake_mode not in FAKE_FP8_STORE_MODES:
            raise ValueError(f"Unknown fake FP8 store mode: {fake_mode!r}")
        result = _run_trace(args, prompt_token_ids, "bf16", False, fake_fp8_store_mode=fake_mode)
    elif run_kind == "v_fp8":
        result = _run_trace(args, prompt_token_ids, "fp8_v_only", True, "v_fp8")
    elif run_kind in BACKENDS:
        result = _run_trace(args, prompt_token_ids, "fp8_e4m3", True, run_kind)
    else:
        raise ValueError(f"Unknown single run kind: {run_kind!r}")
    Path(args.single_run_output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(_jsonable_for_torch_save(result), args.single_run_output)


def _isolated_cmd(args: argparse.Namespace, run_kind: str, output_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-path",
        args.model_path,
        "--label",
        f"{args.label}_{run_kind}",
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--kv-cache-scale-dtype",
        args.kv_cache_scale_dtype,
        "--native-block-tokens",
        str(args.native_block_tokens),
        "--seed",
        str(args.seed),
        "--vocab-range",
        str(args.vocab_range),
        "--top-k",
        str(args.top_k),
        "--fake-fp8-format",
        args.fake_fp8_format,
        "--fake-fp8-group-size",
        str(args.fake_fp8_group_size),
        "--single-run-kind",
        run_kind,
        "--single-run-output",
        str(output_path),
    ]
    return cmd


def _run_isolated(args: argparse.Namespace, backends: list[str]) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any] | None, dict[str, dict[str, Any]]]:
    if not args.output_json:
        raise ValueError("--isolated-runs requires --output-json so temporary traces have a stable directory")
    work_dir = Path(args.output_json).with_suffix("")
    work_dir.mkdir(parents=True, exist_ok=True)
    fake_modes = _parse_fake_modes(args.fake_fp8_store_modes)
    run_kinds = ["bf16", *(f"fake_{mode}" for mode in fake_modes)]
    if args.include_v_fp8:
        run_kinds.append("v_fp8")
    run_kinds.extend(backends)
    outputs = {kind: work_dir / f"{kind}.pt" for kind in run_kinds}
    for kind in run_kinds:
        cmd = _isolated_cmd(args, kind, outputs[kind])
        print(f"### isolated_run:{kind}")
        print(" ".join(cmd))
        status = subprocess.call(cmd)
        print(f"### isolated_run:{kind}:exit_code={status}")
        if status:
            raise SystemExit(status)
    bf16 = torch.load(outputs["bf16"], map_location="cpu", weights_only=False)
    fake_results = {mode: torch.load(outputs[f"fake_{mode}"], map_location="cpu", weights_only=False) for mode in fake_modes}
    v_fp8_result = torch.load(outputs["v_fp8"], map_location="cpu", weights_only=False) if args.include_v_fp8 else None
    fp8_results = {backend: torch.load(outputs[backend], map_location="cpu", weights_only=False) for backend in backends}
    return bf16, fake_results, v_fp8_result, fp8_results


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
    parser.add_argument(
        "--fp8-decode-backends",
        default="native,gather_dequant,full_dequant",
        help="Comma-separated FP8 decode backends: native,gather_dequant,full_dequant",
    )
    parser.add_argument(
        "--fake-fp8-store-modes",
        default="",
        help="Comma-separated debug BF16-cache fake FP8 store modes: k,v,both",
    )
    parser.add_argument("--native-block-tokens", type=int, default=64, choices=(16, 32, 64))
    parser.add_argument("--include-v-fp8", action="store_true", help="Also run real K-BF16/V-FP8 cache mode")
    parser.add_argument("--fake-fp8-format", default="e4m3", choices=("e4m3", "e5m2", "int8"))
    parser.add_argument("--fake-fp8-group-size", type=int, default=0, help="0 means per-vector scale; otherwise per-group scale over head_dim")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-range", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-json")
    parser.add_argument("--isolated-runs", action="store_true", help="Run each BF16/FP8 backend in a fresh subprocess")
    parser.add_argument("--single-run-kind")
    parser.add_argument("--single-run-output")
    args = parser.parse_args()
    args.model_path = os.path.expanduser(args.model_path)

    if args.single_run_kind:
        if not args.single_run_output:
            raise ValueError("--single-run-kind requires --single-run-output")
        _run_single_and_save(args, args.single_run_kind)
        return

    backends = _parse_backends(args.fp8_decode_backends)
    fake_modes = _parse_fake_modes(args.fake_fp8_store_modes)
    if args.isolated_runs:
        bf16, fake_results, v_fp8_result, fp8_results = _run_isolated(args, backends)
    else:
        random.seed(args.seed)
        prompt_token_ids = [random.randint(0, args.vocab_range) for _ in range(args.input_len)]
        bf16 = _run_trace(args, prompt_token_ids, "bf16", False)
        fake_results = {mode: _run_trace(args, prompt_token_ids, "bf16", False, fake_fp8_store_mode=mode) for mode in fake_modes}
        v_fp8_result = _run_trace(args, prompt_token_ids, "fp8_v_only", True, "v_fp8") if args.include_v_fp8 else None
        fp8_results = {backend: _run_trace(args, prompt_token_ids, "fp8_e4m3", True, backend) for backend in backends}
    fake_comparisons = {
        mode: {
            "token_comparison": _compare_tokens(bf16["generated_token_ids"], fake["generated_token_ids"]),
            "logits_comparison": _compare_logits(bf16["logits_trace"], fake["logits_trace"], args.top_k, "bf16", f"fake_{mode}"),
        }
        for mode, fake in fake_results.items()
    }
    v_fp8_comparison = None
    if v_fp8_result is not None:
        v_fp8_comparison = {
            "token_comparison": _compare_tokens(bf16["generated_token_ids"], v_fp8_result["generated_token_ids"]),
            "logits_comparison": _compare_logits(bf16["logits_trace"], v_fp8_result["logits_trace"], args.top_k, "bf16", "v_fp8"),
        }
    comparisons = {
        backend: {
            "token_comparison": _compare_tokens(bf16["generated_token_ids"], fp8["generated_token_ids"]),
            "logits_comparison": _compare_logits(bf16["logits_trace"], fp8["logits_trace"], args.top_k, "bf16", backend),
        }
        for backend, fp8 in fp8_results.items()
    }
    pairwise_fp8_comparisons = {
        f"{left}_vs_{right}": {
            "token_comparison": _compare_tokens(
                fp8_results[left]["generated_token_ids"], fp8_results[right]["generated_token_ids"]
            ),
            "logits_comparison": _compare_logits(
                fp8_results[left]["logits_trace"], fp8_results[right]["logits_trace"], args.top_k, left, right
            ),
        }
        for left_index, left in enumerate(fp8_results)
        for right in list(fp8_results)[left_index + 1 :]
    }

    summary = {
        "label": args.label,
        "model_path": args.model_path,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "seed": args.seed,
        "top_k": args.top_k,
        "fp8_decode_backends": list(fp8_results),
        "fake_fp8_store_modes": list(fake_results),
        "fake_fp8_format": args.fake_fp8_format,
        "fake_fp8_group_size": args.fake_fp8_group_size,
        "bf16": _strip_logits(bf16),
        "fake_fp8_store": {mode: _strip_logits(result) for mode, result in fake_results.items()},
        "v_fp8": _strip_logits(v_fp8_result) if v_fp8_result is not None else None,
        "fp8_e4m3": {backend: _strip_logits(result) for backend, result in fp8_results.items()},
        "fake_fp8_comparisons_vs_bf16": fake_comparisons,
        "v_fp8_comparison_vs_bf16": v_fp8_comparison,
        "comparisons_vs_bf16": comparisons,
        "pairwise_fp8_comparisons": pairwise_fp8_comparisons,
        "storage_ratios_v_fp8_over_bf16": (
            {
                "data_bytes_per_block": v_fp8_result["kv_cache_data_bytes_per_block"] / bf16["kv_cache_data_bytes_per_block"],
                "total_bytes_per_block": v_fp8_result["kv_cache_total_bytes_per_block"] / bf16["kv_cache_total_bytes_per_block"],
                "block_count": v_fp8_result["num_kvcache_blocks"] / bf16["num_kvcache_blocks"],
            }
            if v_fp8_result is not None else None
        ),
        "storage_ratios_fp8_over_bf16": {
            backend: {
                "data_bytes_per_block": result["kv_cache_data_bytes_per_block"] / bf16["kv_cache_data_bytes_per_block"],
                "total_bytes_per_block": result["kv_cache_total_bytes_per_block"] / bf16["kv_cache_total_bytes_per_block"],
                "block_count": result["num_kvcache_blocks"] / bf16["num_kvcache_blocks"],
            }
            for backend, result in fp8_results.items()
        },
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
