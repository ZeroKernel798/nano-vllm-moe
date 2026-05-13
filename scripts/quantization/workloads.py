from __future__ import annotations

import argparse
import random
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from common import cuda_memory_snapshot, nvidia_smi_query, reset_cuda_peak
from nanovllm import LLM, SamplingParams


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=16)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--min-input-len", type=int, default=4)
    parser.add_argument("--min-output-len", type=int, default=2)
    parser.add_argument("--random-lens", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)


def build_llm(model_path: str, args: argparse.Namespace) -> LLM:
    return LLM(
        str(Path(model_path).expanduser()),
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


def make_token_batch(args: argparse.Namespace, repeat_index: int) -> tuple[list[list[int]], list[SamplingParams]]:
    rng = random.Random(args.seed + repeat_index)
    if args.random_lens:
        min_input = min(args.min_input_len, args.input_len)
        min_output = min(args.min_output_len, args.output_len)
        prompts = [
            [rng.randint(0, args.vocab_size - 1) for _ in range(rng.randint(min_input, args.input_len))]
            for _ in range(args.num_seqs)
        ]
        sampling = [
            SamplingParams(
                temperature=args.temperature,
                ignore_eos=True,
                max_tokens=rng.randint(min_output, args.output_len),
            )
            for _ in range(args.num_seqs)
        ]
        return prompts, sampling
    prompts = [[rng.randint(0, args.vocab_size - 1) for _ in range(args.input_len)] for _ in range(args.num_seqs)]
    sampling_param = SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=args.output_len)
    return prompts, [sampling_param] * args.num_seqs


def run_generation_once(llm: LLM, args: argparse.Namespace, repeat_index: int) -> dict[str, Any]:
    prompts, sampling = make_token_batch(args, repeat_index)
    reset_cuda_peak()
    start = perf_counter()
    output = llm.generate(prompts, sampling, use_tqdm=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = perf_counter() - start
    stats = output["stats"]
    total_gen_tokens = sum(len(tokens) for tokens in output["results"])
    total_prompt_tokens = sum(len(tokens) for tokens in prompts)
    return {
        "repeat": repeat_index,
        "wall_time_s": wall,
        "prefill_tps": stats["prefill_tps"],
        "decode_tps": stats["decode_tps"],
        "avg_ttft_ms": stats["avg_ttft_ms"],
        "total_prompt_tokens": total_prompt_tokens,
        "total_gen_tokens": total_gen_tokens,
        "end_to_end_tps": (total_prompt_tokens + total_gen_tokens) / max(wall, 1e-6),
        "memory": cuda_memory_snapshot("run_"),
        "nvidia_smi": nvidia_smi_query(),
    }
