from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from safetensors import safe_open

from common import emit_result, print_result, runtime_metadata


def load_tensor(model_path: str, name: str) -> torch.Tensor:
    for file in sorted(Path(model_path).glob("*.safetensors")):
        with safe_open(str(file), "pt", "cpu") as handle:
            if name in handle.keys():
                return handle.get_tensor(name)
    raise KeyError(f"{name!r} not found in {model_path}")


def cuda_time(fn, warmup: int, repeat: int) -> tuple[float, Any]:
    result = None
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()
    start = perf_counter()
    for _ in range(repeat):
        result = fn()
    torch.cuda.synchronize()
    return (perf_counter() - start) * 1000.0 / repeat, result


def benchmark_weight(args: argparse.Namespace, weight_name: str) -> dict[str, Any]:
    weight = load_tensor(args.bf16_model_path, f"{weight_name}.weight").to(torch.bfloat16).cuda()
    n, k = weight.shape
    expected_k = args.k or k
    if k != expected_k:
        raise ValueError(f"{weight_name} K mismatch: weight.shape={tuple(weight.shape)}, expected K={args.k}")
    x = torch.randn((args.m, k), device="cuda", dtype=torch.bfloat16)
    flops = 2.0 * args.m * k * n
    ms, out = cuda_time(lambda: torch.nn.functional.linear(x, weight), args.warmup, args.repeat)
    return {
        "weight_name": weight_name,
        "shape": {"m": args.m, "k": k, "n": n},
        "repeat": args.repeat,
        "warmup": args.warmup,
        "backend": "bf16_torch",
        "ms": ms,
        "tflops": flops / (ms / 1000.0) / 1e12,
        "output_checksum": out.float().sum().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbenchmark native BF16 linear layers")
    parser.add_argument("--bf16-model-path", required=True)
    parser.add_argument("--label", default="bf16_linear_microbench")
    parser.add_argument("--weight-name", action="append", required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, default=0, help="Expected input dimension; inferred from weight when omitted")
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output-json")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BF16 microbench")
    result = runtime_metadata(args.bf16_model_path, args.label)
    result["task"] = "bf16_linear_microbench"
    result["args"] = vars(args)
    result["cases"] = [benchmark_weight(args, name) for name in args.weight_name]
    emit_result(args, result)

    for case in result["cases"]:
        print_result(
            case,
            [
                "weight_name",
                "shape.m",
                "shape.k",
                "shape.n",
                "ms",
                "tflops",
            ],
        )


if __name__ == "__main__":
    main()
