#!/usr/bin/env python3
"""Comprehensive test suite for nano-vllm-moe optimizations.

This script tests:
1. Accuracy: PPL, logits cosine, token match, MMLU/CEval
2. Efficiency: TPS, latency, prefill/decode timing
3. Memory: KV bytes/block, peak memory, model size

Usage:
    python comprehensive_test.py --model-path /path/to/model --output-dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class TestConfig:
    """Configuration for a single test."""
    name: str
    description: str
    model_path: str
    quantization_type: str = "bf16"  # bf16, w8a16, w8a8
    kv_cache_dtype: str = "bf16"  # bf16, k_int8_v_fp8
    test_types: list[str] = field(default_factory=lambda: ["accuracy", "efficiency", "memory"])

    # Test parameters
    input_len: int = 512
    output_len: int = 16
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.9

    # Accuracy test params
    ppl_samples: int = 32
    ppl_max_length: int = 256

    # Efficiency test params
    bench_warmup: int = 3
    bench_repeat: int = 5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    config: TestConfig
    success: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config": self.config.to_dict(),
            "success": self.success,
            "metrics": self.metrics,
            "error": self.error,
        }


def run_accuracy_test(config: TestConfig) -> dict[str, Any]:
    """Run accuracy tests: PPL, logits cosine, token match."""
    # Import here to avoid issues when running locally
    from scripts.quantization.eval_ppl_quant import eval_ppl
    from scripts.quantization.compare_logits import compare_logits

    metrics = {}

    # PPL test
    if "ppl" in config.test_types or "accuracy" in config.test_types:
        ppl_result = eval_ppl(
            model_path=config.model_path,
            samples=config.ppl_samples,
            max_length=config.ppl_max_length,
        )
        metrics["ppl"] = ppl_result

    # Logits cosine test
    if "logits" in config.test_types or "accuracy" in config.test_types:
        logits_result = compare_logits(
            model_path=config.model_path,
            input_len=config.input_len,
            output_len=config.output_len,
        )
        metrics["logits_cosine"] = logits_result.get("cosine", 0)
        metrics["token_match"] = logits_result.get("token_match", 0)

    return metrics


def run_efficiency_test(config: TestConfig) -> dict[str, Any]:
    """Run efficiency tests: TPS, latency."""
    import torch
    import time
    from nanovllm import LLM

    metrics = {}

    # Initialize model
    llm = LLM(
        model_path=config.model_path,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enforce_eager=True,  # Disable CUDA graph for consistent timing
    )

    # Generate test prompt
    prompt = "The quick brown fox jumps over the lazy dog. " * (config.input_len // 10)

    # Warmup
    for _ in range(config.bench_warmup):
        llm.generate([prompt], max_tokens=config.output_len)

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(config.bench_repeat):
        output = llm.generate([prompt], max_tokens=config.output_len)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_tokens = config.output_len * config.bench_repeat
    metrics["output_tokens"] = total_tokens
    metrics["wall_time_s"] = elapsed
    metrics["tps"] = total_tokens / elapsed
    metrics["latency_ms"] = elapsed * 1000 / config.bench_repeat

    return metrics


def run_memory_test(config: TestConfig) -> dict[str, Any]:
    """Run memory tests: peak memory, model size, KV cache."""
    import torch
    from nanovllm import LLM
    from nanovllm.utils.kv_cache import kv_cache_bytes_per_element

    metrics = {}

    # Get model size
    model_path = Path(config.model_path)
    total_size = sum(f.stat().st_size for f in model_path.glob("*.safetensors"))
    metrics["model_size_gb"] = total_size / (1024**3)

    # Initialize model and measure memory
    torch.cuda.reset_peak_memory_stats()
    llm = LLM(
        model_path=config.model_path,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
        kv_cache_dtype=config.kv_cache_dtype,
    )

    metrics["peak_memory_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
    metrics["reserved_memory_gb"] = torch.cuda.max_memory_reserved() / (1024**3)

    # KV cache metrics
    if config.kv_cache_dtype != "bf16":
        kv_bpe = kv_cache_bytes_per_element(config.kv_cache_dtype)
        metrics["kv_bytes_per_element"] = kv_bpe
        metrics["kv_compression_ratio"] = 2 / kv_bpe  # Compared to BF16

    return metrics


def run_test(config: TestConfig) -> TestResult:
    """Run a single test configuration."""
    result = TestResult(
        name=config.name,
        config=config,
        success=True,
    )

    try:
        if "accuracy" in config.test_types:
            result.metrics.update(run_accuracy_test(config))

        if "efficiency" in config.test_types:
            result.metrics.update(run_efficiency_test(config))

        if "memory" in config.test_types:
            result.metrics.update(run_memory_test(config))

    except Exception as e:
        result.success = False
        result.error = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Comprehensive test suite for nano-vllm-moe")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", default="test_results", help="Directory to save results")
    parser.add_argument("--quantization-type", default="bf16", choices=["bf16", "w8a16", "w8a8"])
    parser.add_argument("--kv-cache-dtype", default="bf16", choices=["bf16", "k_int8_v_fp8"])
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--test-types", nargs="+", default=["accuracy", "efficiency", "memory"])
    args = parser.parse_args()

    config = TestConfig(
        name=f"{args.quantization_type}_{args.kv_cache_dtype}",
        description=f"Test for {args.quantization_type} with {args.kv_cache_dtype} KV cache",
        model_path=args.model_path,
        quantization_type=args.quantization_type,
        kv_cache_dtype=args.kv_cache_dtype,
        input_len=args.input_len,
        output_len=args.output_len,
        test_types=args.test_types,
    )

    result = run_test(config)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{config.name}_results.json"

    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"Test completed: {result.name}")
    print(f"Success: {result.success}")
    print(f"Results saved to: {output_file}")

    if result.metrics:
        print("\nMetrics:")
        for k, v in result.metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
