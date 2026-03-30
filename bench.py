"""
End-to-end generation benchmark (mixed prefill + decode).

Compare BF16 vs FP8 checkpoints by running this script with different ``--model-path``
(same hyperparameters). For P/D isolation use ``pd_bench.py`` instead.

Examples::

    python bench.py --model-path /path/to/bf16-instruct --label bf16
    python bench.py --model-path /path/to/fp8-w8a8 --label w8a8
    python bench.py --model-path /path/to/fp8-w8a16 --label w8a16
    python bench.py --model-path /root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct --label bf16 --warmup \
    --num-seqs 256 \
    --input-len 1024 \
    --output-len 1024 \
    --max-model-len 4096 \
    --seed 0 \
    --random-lens
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


def _try_read_quant_config(model_dir: str) -> dict | None:
    p = Path(model_dir) / "config.json"
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("quantization_config")
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="nano-vllm mixed generation throughput")
    ap.add_argument(
        "--model-path",
        type=str,
        default="/root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
    )
    ap.add_argument("--label", type=str, default="", help="Tag printed in the report line")
    ap.add_argument("--num-seqs", type=int, default=256, help="Parallel sequences in one batch")
    ap.add_argument("--input-len", type=int, default=512, help="Prompt length (tokens) when not --random-lens")
    ap.add_argument("--output-len", type=int, default=512, help="max_tokens per seq when not --random-lens")
    ap.add_argument(
        "--random-lens",
        action="store_true",
        help="Random input in [100, input-len] and output in [100, output-len] (less reproducible)",
    )
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--ep-size", type=int, default=1)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--warmup",
        action="store_true",
        help="Run one untimed full batch before the timed run (recommended for stable GPU clocks)",
    )
    args = ap.parse_args()

    path = os.path.expanduser(args.model_path)
    random.seed(args.seed)

    qc = _try_read_quant_config(path)
    tag = args.label or Path(path).name
    print(f"model_path={path}")
    print(f"label={tag}")
    if qc:
        print(f"quantization_config={qc}")

    llm = LLM(
        path,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
    )

    n = args.num_seqs
    if args.random_lens:
        prompt_token_ids = [
            [random.randint(0, 10000) for _ in range(random.randint(100, args.input_len))]
            for _ in range(n)
        ]
        sampling_params = [
            SamplingParams(
                temperature=0.6,
                ignore_eos=True,
                max_tokens=random.randint(100, args.output_len),
            )
            for _ in range(n)
        ]
    else:
        prompt_token_ids = [
            [random.randint(0, 10000) for _ in range(args.input_len)] for _ in range(n)
        ]
        sp = SamplingParams(
            temperature=0.6, ignore_eos=True, max_tokens=args.output_len
        )
        sampling_params = [sp] * n

    if args.warmup:
        llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = perf_counter()
    out = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = perf_counter() - t0

    s = out["stats"]
    total_gen = sum(len(toks) for toks in out["results"])
    print(
        f"\n[{tag}] wall_time={wall:.3f}s  "
        f"prefill_tps={s['prefill_tps']:.2f}  decode_tps={s['decode_tps']:.2f}  "
        f"avg_ttft_ms={s['avg_ttft_ms']:.2f}  "
        f"total_gen_tokens={total_gen}"
    )


if __name__ == "__main__":
    main()
