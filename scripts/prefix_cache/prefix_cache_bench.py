#!/usr/bin/env python3
"""Prefix Cache Benchmark for nano-vllm-moe.

Run each scenario separately to avoid process group conflicts.

Usage:
  python prefix_cache_bench.py --model <path> --scenario no-reuse
  python prefix_cache_bench.py --model <path> --scenario shared-prefix
  python prefix_cache_bench.py --model <path> --scenario partial-shared
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def generate_prompts_shared_prefix(num_requests: int, prefix_len: int, unique_len: int) -> list[list[int]]:
    prefix = list(range(100, 100 + prefix_len))
    prompts = []
    for i in range(num_requests):
        unique = list(range(1000 + i * unique_len, 1000 + i * unique_len + unique_len))
        prompts.append(prefix + unique)
    return prompts


def generate_prompts_partial_shared(num_requests: int, shared_base_len: int, variation_len: int, unique_len: int) -> list[list[int]]:
    prompts = []
    for i in range(num_requests):
        base = list(range(100, 100 + shared_base_len))
        variation = list(range(500 + i * variation_len, 500 + i * variation_len + variation_len))
        unique = list(range(2000 + i * unique_len, 2000 + i * unique_len + unique_len))
        prompts.append(base + variation + unique)
    return prompts


def generate_prompts_no_reuse(num_requests: int, prompt_len: int) -> list[list[int]]:
    prompts = []
    for i in range(num_requests):
        prompt = list(range(100 + i * prompt_len, 100 + i * prompt_len + prompt_len))
        prompts.append(prompt)
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Prefix Cache Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--num-requests", type=int, default=4, help="Number of requests")
    parser.add_argument("--prefix-len", type=int, default=512, help="Shared prefix length")
    parser.add_argument("--unique-len", type=int, default=128, help="Unique suffix length")
    parser.add_argument("--max-tokens", type=int, default=16, help="Max output tokens")
    parser.add_argument("--block-size", type=int, default=256, help="KV cache block size")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--scenario", type=str, required=True,
                        choices=["no-reuse", "shared-prefix", "partial-shared"],
                        help="Which scenario to run")
    args = parser.parse_args()

    from nanovllm import LLM, SamplingParams

    if args.scenario == "no-reuse":
        prompts = generate_prompts_no_reuse(args.num_requests, args.prefix_len + args.unique_len)
    elif args.scenario == "shared-prefix":
        prompts = generate_prompts_shared_prefix(args.num_requests, args.prefix_len, args.unique_len)
    elif args.scenario == "partial-shared":
        prompts = generate_prompts_partial_shared(
            args.num_requests, args.prefix_len // 2, args.prefix_len // 2, args.unique_len
        )
    else:
        raise ValueError(f"Unknown scenario: {args.scenario}")

    print(f"\n{'='*60}")
    print(f"Scenario: {args.scenario}")
    print(f"{'='*60}")
    print(f"Generated {len(prompts)} prompts")
    print(f"Average prompt length: {sum(len(p) for p in prompts) // len(prompts)}")

    llm = LLM(
        model=args.model,
        enforce_eager=True,
        kvcache_block_size=args.block_size,
        gpu_memory_utilization=0.9,
    )

    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    result = llm.generate(prompts, sampling_params, use_tqdm=True)

    stats = result["stats"]
    prefix_cache = stats.get("prefix_cache", {})

    print(f"\nResults:")
    print(f"  Total time: {stats['total_time']:.3f}s")
    print(f"  Avg TTFT: {stats['avg_ttft_ms']:.1f}ms")
    print(f"  Prefill TPS: {stats['prefill_tps']:.1f}")
    print(f"  Decode TPS: {stats['decode_tps']:.1f}")
    print(f"  Prefix cache queries: {prefix_cache.get('num_queries', 0)}")
    print(f"  Prefix cache hit blocks: {prefix_cache.get('total_hit_blocks', 0)}")
    print(f"  Token hit rate: {prefix_cache.get('token_hit_rate', 0.0):.3f}")
    print(f"  Block hit rate: {prefix_cache.get('block_hit_rate', 0.0):.3f}")

    output = {
        "scenario": args.scenario,
        "num_requests": len(prompts),
        "avg_prompt_len": sum(len(p) for p in prompts) // len(prompts),
        "total_time_s": stats["total_time"],
        "avg_ttft_ms": stats["avg_ttft_ms"],
        "prefill_tps": stats["prefill_tps"],
        "decode_tps": stats["decode_tps"],
        "prefix_cache": prefix_cache,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
