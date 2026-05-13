from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from random import Random
from statistics import mean, stdev

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm import LLM, SamplingParams


def parse_backends(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_ep_backends(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@contextmanager
def temporary_env(updates: dict[str, str | None]):
    old_values = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def make_prompt_token_ids(args, repeat_idx: int) -> list[list[int]]:
    if args.fixed_prompts:
        rng = Random(args.seed)
    else:
        rng = Random(args.seed + repeat_idx)
    return [
        [rng.randint(0, args.vocab_size - 1) for _ in range(rng.randint(args.min_input_len, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]


def make_sampling_params(args, repeat_idx: int) -> list[SamplingParams]:
    if args.fixed_prompts:
        rng = Random(args.seed + 9973)
    else:
        rng = Random(args.seed + repeat_idx + 9973)
    return [
        SamplingParams(
            temperature=args.temperature,
            ignore_eos=True,
            max_tokens=rng.randint(args.min_output_len, args.max_output_len),
        )
        for _ in range(args.num_seqs)
    ]


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def make_failure_result(args, backend: str, ep_backend: str, repeat_idx: int, exc: Exception) -> dict:
    return {
        "backend": backend,
        "ep_backend": ep_backend,
        "repeat": repeat_idx,
        "tp_size": args.tp_size,
        "ep_size": args.ep_size,
        "ok": False,
        "discarded": repeat_idx < args.discard_first,
        "error": repr(exc),
    }


def run_backend_once(args, backend: str, ep_backend: str, repeat_idx: int) -> dict:
    print("\n" + "=" * 70)
    print(
        f"Testing MoE backend={backend} EP backend={ep_backend} "
        f"TP={args.tp_size} EP={args.ep_size} repeat={repeat_idx}"
    )
    print("=" * 70)
    path = os.path.expanduser(args.model_path)
    llm = None
    try:
        with temporary_env({"NANOVLLM_MOE_PROFILE": "1" if args.moe_profile else None}):
            llm = LLM(
                path,
                enforce_eager=args.enforce_eager,
                tp_size=args.tp_size,
                ep_size=args.ep_size,
                max_model_len=args.max_model_len,
                max_num_batched_tokens=args.max_num_batched_tokens,
                max_num_seqs=args.max_num_seqs,
                gpu_memory_utilization=args.gpu_memory_utilization,
                moe_backend=backend,
                moe_ep_backend=ep_backend,
            )
            prompt_token_ids = make_prompt_token_ids(args, repeat_idx)
            sampling_params = make_sampling_params(args, repeat_idx)

            for _ in range(args.warmup_runs):
                llm.generate(["Warmup"], SamplingParams(max_tokens=args.warmup_tokens), use_tqdm=False)
            if args.moe_profile:
                llm.model_runner.call("reset_moe_profile")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            out = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            moe_profile = llm.model_runner.call("get_moe_profile") if args.moe_profile else {}
        wall = time.perf_counter() - start
        stats = out["stats"]
        total_in = sum(len(prompt) for prompt in prompt_token_ids)
        total_out = sum(len(tokens) for tokens in out["results"])
        result = {
            "backend": backend,
            "ep_backend": ep_backend,
            "repeat": repeat_idx,
            "tp_size": args.tp_size,
            "ep_size": args.ep_size,
            "ok": True,
            "discarded": repeat_idx < args.discard_first,
            "wall": wall,
            "prefill_tps": stats["prefill_tps"],
            "decode_tps": stats["decode_tps"],
            "total_in": total_in,
            "total_out": total_out,
            "prompt_lens": [len(prompt) for prompt in prompt_token_ids],
            "output_lens": [sp.max_tokens for sp in sampling_params],
            "moe_profile": moe_profile,
            "error": "",
        }
        print("✅", result)
        return result
    except Exception as exc:
        result = make_failure_result(args, backend, ep_backend, repeat_idx, exc)
        print("❌", result)
        return result
    finally:
        if llm is not None:
            llm.exit()
            del llm
        elif dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        clear_cuda()
        time.sleep(args.sleep_after_backend)


def summarize_backend(backend: str, ep_backend: str, results: list[dict]) -> dict:
    ok_results = [
        item
        for item in results
        if item["backend"] == backend
        and item["ep_backend"] == ep_backend
        and item["ok"]
        and not item.get("discarded", False)
    ]
    total_results = [r for r in results if r["backend"] == backend and r["ep_backend"] == ep_backend]
    summary = {"backend": backend, "ep_backend": ep_backend, "ok_runs": len(ok_results), "total_runs": len(total_results)}
    summary["discarded_runs"] = len([r for r in total_results if r.get("discarded", False)])
    for metric in ["wall", "prefill_tps", "decode_tps"]:
        values = [item[metric] for item in ok_results]
        if values:
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_stdev"] = stdev(values) if len(values) > 1 else 0.0
    return summary


def format_result(result: dict) -> str:
    if not result["ok"]:
        return (
            f"backend={result['backend']}, ep_backend={result['ep_backend']}, repeat={result['repeat']}, TP={result['tp_size']}, "
            f"EP={result['ep_size']}, FAILED: {result['error']}"
        )
    return (
        f"backend={result['backend']}, ep_backend={result['ep_backend']}, repeat={result['repeat']}, "
        f"TP={result['tp_size']}, EP={result['ep_size']}, "
        f"wall={result['wall']:.3f}s, prefill_tps={result['prefill_tps']:.2f}, "
        f"decode_tps={result['decode_tps']:.2f}, total_in={result['total_in']}, total_out={result['total_out']}"
    )


def write_text_summary(args, results: list[dict], summaries: list[dict]) -> None:
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("MoE backend benchmark\n")
        f.write(f"model={args.model_path}\n")
        f.write(
            f"tp={args.tp_size}, ep={args.ep_size}, ep_backends={args.ep_backends}, "
            f"repeat={args.repeat}, discard_first={args.discard_first}, fixed_prompts={args.fixed_prompts}\n"
        )
        for result in results:
            f.write(format_result(result) + "\n")
        f.write("\nSummary\n")
        for summary in summaries:
            f.write(json.dumps(summary, sort_keys=True) + "\n")


def write_json(path: str, args, results: list[dict], summaries: list[dict]) -> None:
    if not path:
        return
    payload = {
        "config": {
            "model_path": args.model_path,
            "backends": parse_backends(args.backends),
            "ep_backends": parse_ep_backends(args.ep_backends),
            "tp_size": args.tp_size,
            "ep_size": args.ep_size,
            "repeat": args.repeat,
            "discard_first": args.discard_first,
            "fixed_prompts": args.fixed_prompts,
            "warmup_runs": args.warmup_runs,
            "warmup_tokens": args.warmup_tokens,
            "moe_profile": args.moe_profile,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "num_seqs": args.num_seqs,
        },
        "results": results,
        "summary": summaries,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_csv(path: str, results: list[dict]) -> None:
    if not path:
        return
    fieldnames = [
        "backend",
        "ep_backend",
        "repeat",
        "tp_size",
        "ep_size",
        "ok",
        "discarded",
        "wall",
        "prefill_tps",
        "decode_tps",
        "total_in",
        "total_out",
        "error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MoE expert backends under the same EP setup")
    parser.add_argument("--model-path", type=str, default="/home/ubuntu/project/models/qwen/Qwen1.5-MoE-A2.7B-Chat")
    parser.add_argument("--backends", type=str, default="eager,optimized,fused")
    parser.add_argument("--ep-backends", type=str, default="torch")
    parser.add_argument("--output-file", type=str, default="moe_backend_benchmark_summary.txt")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--discard-first", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=64)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--min-input-len", type=int, default=4)
    parser.add_argument("--max-input-len", type=int, default=4)
    parser.add_argument("--min-output-len", type=int, default=2)
    parser.add_argument("--max-output-len", type=int, default=2)
    parser.add_argument("--warmup-tokens", type=int, default=2)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--fixed-prompts", action="store_true")
    parser.add_argument("--moe-profile", action="store_true")
    parser.add_argument("--sleep-after-backend", type=float, default=3.0)
    args = parser.parse_args()

    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if args.discard_first < 0 or args.discard_first >= args.repeat:
        raise ValueError("--discard-first must be >= 0 and < --repeat")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")

    results = []
    for backend in parse_backends(args.backends):
        for ep_backend in parse_ep_backends(args.ep_backends):
            for repeat_idx in range(args.repeat):
                results.append(run_backend_once(args, backend, ep_backend, repeat_idx))

    summaries = [
        summarize_backend(backend, ep_backend, results)
        for backend in parse_backends(args.backends)
        for ep_backend in parse_ep_backends(args.ep_backends)
    ]
    write_text_summary(args, results, summaries)
    write_json(args.output_json, args, results, summaries)
    write_csv(args.output_csv, results)

    print(f"\nSummary saved to {args.output_file}")
    if args.output_json:
        print(f"JSON saved to {args.output_json}")
    if args.output_csv:
        print(f"CSV saved to {args.output_csv}")


if __name__ == "__main__":
    main()
