from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from random import randint, seed

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm import LLM, SamplingParams


def parse_backends(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_backend(args, backend: str, output_file: str) -> None:
    print("\n" + "=" * 70)
    print(f"Testing MoE backend={backend} TP={args.tp_size} EP={args.ep_size}")
    print("=" * 70)
    seed(args.seed)
    path = os.path.expanduser(args.model_path)
    llm = None
    try:
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
        )
        prompt_token_ids = [
            [randint(0, 10000) for _ in range(randint(args.min_input_len, args.max_input_len))]
            for _ in range(args.num_seqs)
        ]
        sampling_params = [
            SamplingParams(
                temperature=args.temperature,
                ignore_eos=True,
                max_tokens=randint(args.min_output_len, args.max_output_len),
            )
            for _ in range(args.num_seqs)
        ]

        llm.generate(["Warmup"], SamplingParams(max_tokens=args.warmup_tokens), use_tqdm=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - start
        stats = out["stats"]
        total_in = sum(len(prompt) for prompt in prompt_token_ids)
        total_out = sum(len(tokens) for tokens in out["results"])
        result = (
            f"backend={backend}, TP={args.tp_size}, EP={args.ep_size}, "
            f"wall={wall:.3f}s, prefill_tps={stats['prefill_tps']:.2f}, "
            f"decode_tps={stats['decode_tps']:.2f}, total_in={total_in}, total_out={total_out}"
        )
        print("✅", result)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(result + "\n")
    except Exception as exc:
        result = f"backend={backend}, TP={args.tp_size}, EP={args.ep_size}, FAILED: {exc!r}"
        print("❌", result)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(result + "\n")
    finally:
        if llm is not None:
            llm.exit()
            del llm
        elif dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        clear_cuda()
        time.sleep(args.sleep_after_backend)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MoE expert backends under the same EP setup")
    parser.add_argument("--model-path", type=str, default="/home/ubuntu/project/models/qwen/Qwen1.5-MoE-A2.7B-Chat")
    parser.add_argument("--backends", type=str, default="transformers,mini_sglang,fused")
    parser.add_argument("--output-file", type=str, default="moe_backend_benchmark_summary.txt")
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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep-after-backend", type=float, default=3.0)
    args = parser.parse_args()

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("MoE backend benchmark\n")
        f.write(f"model={args.model_path}\n")
        f.write(f"tp={args.tp_size}, ep={args.ep_size}\n")

    for backend in parse_backends(args.backends):
        run_backend(args, backend, args.output_file)

    print(f"\nSummary saved to {args.output_file}")


if __name__ == "__main__":
    main()
