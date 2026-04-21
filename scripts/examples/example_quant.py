"""End-to-end inference comparison: BF16 vs FP8-W8A16 vs INT8-W8A8.

Usage:
  python scripts/examples/example_quant.py
  python scripts/examples/example_quant.py --scheme fp8_w8a16
  python scripts/examples/example_quant.py --scheme int8_w8a8
  python scripts/examples/example_quant.py --scheme all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nanovllm import LLM, SamplingParams

MODEL_BASE = "/workspace/models/qwen/Qwen1.5-0.5B-Chat"
MODEL_FP8 = "/workspace/models/qwen/Qwen1.5-0.5B-Chat-FP8"
MODEL_INT8 = "/workspace/models/qwen/Qwen1.5-0.5B-Chat-INT8"
MODEL_INT8_STATIC = "/workspace/models/qwen/Qwen1.5-0.5B-Chat-INT8-Static"

PROMPTS = [
    "introduce yourself in one sentence",
    "what is 17 times 24?",
    "write a haiku about the ocean",
]


def run_inference(model_path: str, label: str, max_tokens: int = 128) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}  ({model_path})")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model_path,
        enforce_eager=True,   # simplest path, avoid CUDA graph issues
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        gpu_memory_utilization=0.5,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    formatted = []
    for p in PROMPTS:
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
        formatted.append(text)

    t0 = time.perf_counter()
    results = llm.generate(formatted, sampling)
    elapsed = time.perf_counter() - t0

    for prompt, tokens in zip(PROMPTS, results["results"]):
        answer = llm.tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"\nQ: {prompt}")
        print(f"A: {answer}")

    s = results["stats"]
    print(f"\n--- Perf ({label}) ---")
    print(f"  Total: {elapsed:.2f}s  |  Decode TPS: {s.get('decode_tps', 0):.1f}")

    del llm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scheme", default="all",
        choices=["all", "bf16", "fp8_w8a16", "int8_w8a8", "int8_w8a8_static"],
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    runs: list[tuple[str, str]] = []
    if args.scheme in ("all", "bf16"):
        runs.append((MODEL_BASE, "BF16 baseline"))
    if args.scheme in ("all", "fp8_w8a16"):
        runs.append((MODEL_FP8, "FP8 W8A16"))
    if args.scheme in ("all", "int8_w8a8"):
        runs.append((MODEL_INT8, "INT8 W8A16"))
    if args.scheme in ("all", "int8_w8a8_static"):
        runs.append((MODEL_INT8_STATIC, "INT8 W8A8 static"))

    for path, label in runs:
        run_inference(path, label, args.max_tokens)


if __name__ == "__main__":
    main()
