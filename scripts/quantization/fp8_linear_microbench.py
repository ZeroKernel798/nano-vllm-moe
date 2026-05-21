from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from safetensors import safe_open

from common import emit_result, print_result, runtime_metadata
from nanovllm.quantization.cuda_ext import launch_w8a8_cutlass_gemm
from nanovllm.quantization.kernels import launch_scaled_mm_w8a8, to_scaled_mm_weight
from nanovllm.quantization.w8a8_jit import launch_w8a8_cuda_ptx_jit

FP8_MAX = 448.0


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


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0).item()


def summarize_backend(ms: float, out: torch.Tensor, reference: torch.Tensor, flops: float, bf16_ms: float) -> dict[str, Any]:
    return {
        "ms": ms,
        "tflops": flops / (ms / 1000.0) / 1e12,
        "ratio_vs_bf16_dequant": ms / max(bf16_ms, 1e-9),
        "error_vs_bf16_dequant": {
            "max_abs": (out.float() - reference.float()).abs().max().item(),
            "mean_abs": (out.float() - reference.float()).abs().mean().item(),
            "cosine": cosine(out, reference),
        },
    }


def benchmark_weight(args: argparse.Namespace, weight_name: str) -> dict[str, Any]:
    qweight_u8 = load_tensor(args.w8a8_model_path, f"{weight_name}.qweight")
    weight_scale = load_tensor(args.w8a8_model_path, f"{weight_name}.weight_scale").to(torch.float32).cuda()
    input_scale = load_tensor(args.w8a8_model_path, f"{weight_name}.input_scale").to(torch.float32).cuda()
    qweight = qweight_u8.view(torch.float8_e4m3fn).t().contiguous().cuda()

    expected_k = args.k or qweight.shape[0]
    if qweight.shape[0] != expected_k:
        raise ValueError(f"{weight_name} K mismatch: qweight.shape={tuple(qweight.shape)}, expected K={args.k}")
    k, n = qweight.shape
    x = torch.randn((args.m, k), device="cuda", dtype=torch.bfloat16)
    w_bf16 = qweight.to(torch.bfloat16) * weight_scale.to(torch.bfloat16).unsqueeze(0)

    tensor_scale = weight_scale.max().clamp(min=1e-12)
    rescale = (weight_scale / tensor_scale).reshape(1, -1)
    scaled_weight = (qweight.to(torch.float32) * rescale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    w_torch = to_scaled_mm_weight(scaled_weight)
    w_cutlass = w_torch
    w_ptx = scaled_weight.t().contiguous()
    scale_b = tensor_scale.reshape(()).to(torch.float32)

    bf16_ms, bf16_out = cuda_time(lambda: torch.mm(x, w_bf16), args.warmup, args.repeat)
    flops = 2.0 * args.m * k * n
    backend_results: dict[str, Any] = {}
    outputs: dict[str, torch.Tensor] = {}

    def run_torch() -> torch.Tensor:
        return launch_scaled_mm_w8a8(x, w_torch, input_scale, scale_b, None, act_quant_backend="torch")

    def run_ptx() -> torch.Tensor:
        return launch_w8a8_cuda_ptx_jit(x, w_ptx, input_scale, scale_b, None)

    def run_cutlass() -> torch.Tensor:
        return launch_w8a8_cutlass_gemm(x, w_cutlass, input_scale, scale_b, None)

    def run_auto() -> torch.Tensor:
        if args.m <= args.auto_threshold:
            return run_ptx()
        return run_cutlass()

    runners = {
        "torch": run_torch,
        "ptx": run_ptx,
        "cutlass": run_cutlass,
        "auto": run_auto,
    }
    for backend in args.backend:
        ms, out = cuda_time(runners[backend], args.warmup, args.repeat)
        outputs[backend] = out
        backend_results[backend] = summarize_backend(ms, out, bf16_out, flops, bf16_ms)

    backend_compare: dict[str, Any] = {}
    if "ptx" in backend_results and "cutlass" in backend_results:
        backend_compare["selected_for_m"] = "ptx" if args.m <= args.auto_threshold else "cutlass"
        backend_compare["auto_threshold"] = args.auto_threshold
        backend_compare["ptx_speedup_vs_cutlass"] = backend_results["cutlass"]["ms"] / max(
            backend_results["ptx"]["ms"], 1e-9
        )
        backend_compare["ptx_vs_cutlass_max_abs"] = (outputs["ptx"].float() - outputs["cutlass"].float()).abs().max().item()
        backend_compare["ptx_vs_cutlass_mean_abs"] = (
            outputs["ptx"].float() - outputs["cutlass"].float()
        ).abs().mean().item()
        backend_compare["ptx_vs_cutlass_cosine"] = cosine(outputs["ptx"], outputs["cutlass"])

    primary = backend_results[args.backend[0]]
    return {
        "weight_name": weight_name,
        "shape": {"m": args.m, "k": k, "n": n},
        "repeat": args.repeat,
        "warmup": args.warmup,
        "backends": args.backend,
        "primary_backend": args.backend[0],
        "ms": {
            "bf16_dequant_mm": bf16_ms,
            "w8a8_full": primary["ms"],
        },
        "tflops": {
            "bf16_dequant_mm": flops / (bf16_ms / 1000.0) / 1e12,
            "w8a8_full": primary["tflops"],
        },
        "ratios": {
            "w8a8_full_vs_bf16_dequant": primary["ratio_vs_bf16_dequant"],
        },
        "error_vs_bf16_dequant": primary["error_vs_bf16_dequant"],
        "w8a8_by_backend": backend_results,
        "backend_compare": backend_compare,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbenchmark FP8 W8A8 linear backends")
    parser.add_argument("--w8a8-model-path", required=True)
    parser.add_argument("--label", default="fp8_linear_microbench")
    parser.add_argument("--weight-name", action="append", required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, default=0, help="Expected input dimension; inferred from qweight when omitted")
    parser.add_argument("--backend", action="append", choices=["torch", "ptx", "cutlass", "auto"], default=None)
    parser.add_argument("--auto-threshold", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output-json")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--output-csv")
    args = parser.parse_args()
    if args.backend is None:
        args.backend = ["torch", "ptx", "cutlass", "auto"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FP8 microbench")
    result = runtime_metadata(args.w8a8_model_path, args.label)
    result["task"] = "fp8_linear_microbench"
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
                "ms.bf16_dequant_mm",
                "w8a8_by_backend.torch.ms",
                "w8a8_by_backend.ptx.ms",
                "w8a8_by_backend.cutlass.ms",
                "w8a8_by_backend.auto.ms",
                "backend_compare.selected_for_m",
                "backend_compare.ptx_speedup_vs_cutlass",
                "backend_compare.ptx_vs_cutlass_cosine",
            ],
        )


if __name__ == "__main__":
    main()
