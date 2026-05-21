from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from safetensors import safe_open

from common import emit_result, print_result, runtime_metadata
from nanovllm.quantization.kernels import launch_w8a16_gemm


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
    elapsed_ms = (perf_counter() - start) * 1000.0 / repeat
    return elapsed_ms, result


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0).item()


def benchmark_case(args: argparse.Namespace, weight_name: str, m: int) -> dict[str, Any]:
    qweight_u8 = load_tensor(args.model_path, f"{weight_name}.qweight")
    weight_scale = load_tensor(args.model_path, f"{weight_name}.weight_scale").to(torch.float32).cuda()
    qweight = qweight_u8.view(torch.float8_e4m3fn).t().contiguous().cuda()
    k, n = qweight.shape
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)

    w_bf16 = qweight.to(torch.bfloat16) * weight_scale.to(torch.bfloat16).unsqueeze(0)
    baseline_ms, baseline_out = cuda_time(lambda: torch.mm(x, w_bf16), args.warmup, args.repeat)
    dequant_mm_ms, dequant_out = cuda_time(
        lambda: torch.mm(x, qweight.to(torch.bfloat16) * weight_scale.to(torch.bfloat16).unsqueeze(0)),
        args.warmup,
        args.repeat,
    )
    triton_ms, triton_out = cuda_time(
        lambda: launch_w8a16_gemm(x, qweight, weight_scale, None),
        args.warmup,
        args.repeat,
    )
    flops = 2.0 * m * k * n
    max_abs = (triton_out.float() - baseline_out.float()).abs().max().item()
    mean_abs = (triton_out.float() - baseline_out.float()).abs().mean().item()
    result = {
        "weight_name": weight_name,
        "shape": {"m": m, "k": k, "n": n},
        "repeat": args.repeat,
        "warmup": args.warmup,
        "ms": {
            "bf16_cached_mm": baseline_ms,
            "w8a16_dequant_mm": dequant_mm_ms,
            "w8a16_triton": triton_ms,
        },
        "tflops": {
            "bf16_cached_mm": flops / (baseline_ms / 1000.0) / 1e12,
            "w8a16_dequant_mm": flops / (dequant_mm_ms / 1000.0) / 1e12,
            "w8a16_triton": flops / (triton_ms / 1000.0) / 1e12,
        },
        "ratios": {
            "triton_vs_dequant_mm": triton_ms / max(dequant_mm_ms, 1e-9),
            "triton_vs_cached_bf16_mm": triton_ms / max(baseline_ms, 1e-9),
        },
        "error_vs_cached_bf16_mm": {
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "cosine": cosine(triton_out, baseline_out),
        },
        "sample": {
            "dequant_vs_cached_max_abs": (dequant_out.float() - baseline_out.float()).abs().max().item(),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbenchmark FP8 W8A16 linear kernels")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--label", default="w8a16_linear_microbench")
    parser.add_argument("--weight-name", action="append", required=True)
    parser.add_argument("--m", action="append", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output-json")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for W8A16 microbench")

    result = runtime_metadata(args.model_path, args.label)
    result["task"] = "w8a16_linear_microbench"
    result["args"] = vars(args)
    result["cases"] = [
        benchmark_case(args, weight_name, m)
        for weight_name in args.weight_name
        for m in args.m
    ]
    emit_result(args, result)

    for case in result["cases"]:
        print_result(
            case,
            [
                "weight_name",
                "shape.m",
                "shape.k",
                "shape.n",
                "ms.bf16_cached_mm",
                "ms.w8a16_dequant_mm",
                "ms.w8a16_triton",
                "ratios.triton_vs_dequant_mm",
                "ratios.triton_vs_cached_bf16_mm",
                "error_vs_cached_bf16_mm.max_abs",
                "error_vs_cached_bf16_mm.cosine",
            ],
        )


if __name__ == "__main__":
    main()
