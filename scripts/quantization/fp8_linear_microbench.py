from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from safetensors import safe_open

from common import emit_result, print_result, runtime_metadata
from nanovllm.quantization.kernels import launch_w8a8_fused_gemm_experimental, quantize_activation_w8a8

FP8_MAX = 448.0
FUSED_TRITON_WARNING = (
    "fused_triton is experimental and can trigger a Triton compiler abort on SM89; "
    "pass --allow-fused-triton to run it."
)


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


def to_scaled_mm_weight(w_fp8: torch.Tensor) -> torch.Tensor:
    k, n = w_fp8.shape
    out = torch.empty_strided((k, n), (1, k), device=w_fp8.device, dtype=torch.float8_e4m3fn)
    out.copy_(w_fp8)
    return out


def quantize_activation(x: torch.Tensor, input_scale: torch.Tensor, backend: str) -> torch.Tensor:
    return quantize_activation_w8a8(x, input_scale, backend)


def benchmark_weight(args: argparse.Namespace, weight_name: str) -> dict[str, Any]:
    qweight_u8 = load_tensor(args.w8a8_model_path, f"{weight_name}.qweight")
    weight_scale = load_tensor(args.w8a8_model_path, f"{weight_name}.weight_scale").to(torch.float32)
    input_scale = load_tensor(args.w8a8_model_path, f"{weight_name}.input_scale").to(torch.float32)
    qweight = qweight_u8.view(torch.float8_e4m3fn).t().contiguous().cuda()
    weight_scale = weight_scale.cuda()
    input_scale = input_scale.cuda()

    expected_k = args.k or qweight.shape[0]
    if qweight.shape[0] != expected_k:
        raise ValueError(f"{weight_name} K mismatch: qweight.shape={tuple(qweight.shape)}, expected K={args.k}")
    k, n = qweight.shape
    x = torch.randn((args.m, k), device="cuda", dtype=torch.bfloat16)
    w_bf16 = qweight.to(torch.bfloat16) * weight_scale.to(torch.bfloat16).unsqueeze(0)

    tensor_scale = weight_scale.max().clamp(min=1e-12)
    rescale = (weight_scale / tensor_scale).reshape(1, -1)
    scaled_weight = (qweight.to(torch.float32) * rescale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    w_scaled_mm = to_scaled_mm_weight(scaled_weight)

    scale_a = input_scale.reshape(()).to(torch.float32).clamp(min=1e-12)
    scale_b = tensor_scale.reshape(()).to(torch.float32)

    bf16_ms, bf16_out = cuda_time(lambda: torch.mm(x, w_bf16), args.warmup, args.repeat)
    w8a16_dequant_ms, _ = cuda_time(
        lambda: torch.mm(x, qweight.to(torch.bfloat16) * weight_scale.to(torch.bfloat16).unsqueeze(0)),
        args.warmup,
        args.repeat,
    )
    flops = 2.0 * args.m * k * n
    backend_results: dict[str, Any] = {}
    full_out_by_backend: dict[str, torch.Tensor] = {}
    for backend in args.act_quant_backend:
        if backend == "fused_triton":
            if not args.allow_fused_triton:
                backend_results[backend] = {
                    "skipped": True,
                    "reason": FUSED_TRITON_WARNING,
                    "ms": {"activation_quant": None, "scaled_mm_only": None, "full": None},
                    "tflops": {"scaled_mm_only": None, "full": None},
                    "ratios": {
                        "activation_quant_pct_of_full": None,
                        "scaled_mm_pct_of_full": None,
                        "full_vs_bf16": None,
                        "full_vs_w8a16": None,
                    },
                    "error_vs_bf16_dequant": None,
                }
                continue
            fused_ms, fused_out = cuda_time(
                lambda: launch_w8a8_fused_gemm_experimental(x, w_scaled_mm, input_scale, scale_b, None),
                args.warmup,
                args.repeat,
            )
            full_out_by_backend[backend] = fused_out
            max_abs_error = (fused_out.float() - bf16_out.float()).abs().max().item()
            mean_abs_error = (fused_out.float() - bf16_out.float()).abs().mean().item()
            cosine = torch.nn.functional.cosine_similarity(
                fused_out.float().flatten(), bf16_out.float().flatten(), dim=0
            ).item()
            backend_results[backend] = {
                "ms": {
                    "activation_quant": None,
                    "scaled_mm_only": None,
                    "full": fused_ms,
                },
                "tflops": {
                    "scaled_mm_only": None,
                    "full": flops / (fused_ms / 1000.0) / 1e12,
                },
                "ratios": {
                    "activation_quant_pct_of_full": None,
                    "scaled_mm_pct_of_full": None,
                    "full_vs_bf16": fused_ms / max(bf16_ms, 1e-9),
                    "full_vs_w8a16": fused_ms / max(w8a16_dequant_ms, 1e-9),
                },
                "error_vs_bf16_dequant": {
                    "max_abs": max_abs_error,
                    "mean_abs": mean_abs_error,
                    "cosine": cosine,
                },
            }
            continue
        x_fp8 = quantize_activation(x, input_scale, backend)
        activation_quant_ms, last_x_fp8 = cuda_time(
            lambda backend=backend: quantize_activation(x, input_scale, backend), args.warmup, args.repeat
        )
        scaled_mm_only_ms, _ = cuda_time(
            lambda x_fp8=x_fp8: torch._scaled_mm(
                x_fp8, w_scaled_mm, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16
            ),
            args.warmup,
            args.repeat,
        )
        w8a8_full_ms, full_out = cuda_time(
            lambda backend=backend: torch._scaled_mm(
                quantize_activation(x, input_scale, backend),
                w_scaled_mm,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            ),
            args.warmup,
            args.repeat,
        )
        del last_x_fp8
        full_out_by_backend[backend] = full_out
        max_abs_error = (full_out.float() - bf16_out.float()).abs().max().item()
        mean_abs_error = (full_out.float() - bf16_out.float()).abs().mean().item()
        cosine = torch.nn.functional.cosine_similarity(
            full_out.float().flatten(), bf16_out.float().flatten(), dim=0
        ).item()
        backend_results[backend] = {
            "ms": {
                "activation_quant": activation_quant_ms,
                "scaled_mm_only": scaled_mm_only_ms,
                "full": w8a8_full_ms,
            },
            "tflops": {
                "scaled_mm_only": flops / (scaled_mm_only_ms / 1000.0) / 1e12,
                "full": flops / (w8a8_full_ms / 1000.0) / 1e12,
            },
            "ratios": {
                "activation_quant_pct_of_full": activation_quant_ms / max(w8a8_full_ms, 1e-9) * 100.0,
                "scaled_mm_pct_of_full": scaled_mm_only_ms / max(w8a8_full_ms, 1e-9) * 100.0,
                "full_vs_bf16": w8a8_full_ms / max(bf16_ms, 1e-9),
                "full_vs_w8a16": w8a8_full_ms / max(w8a16_dequant_ms, 1e-9),
            },
            "error_vs_bf16_dequant": {
                "max_abs": max_abs_error,
                "mean_abs": mean_abs_error,
                "cosine": cosine,
            },
        }

    if "torch" in full_out_by_backend and "triton" in full_out_by_backend:
        triton_out = full_out_by_backend["triton"].float()
        torch_out = full_out_by_backend["torch"].float()
        backend_compare = {
            "triton_vs_torch_full_max_abs": (triton_out - torch_out).abs().max().item(),
            "triton_vs_torch_full_mean_abs": (triton_out - torch_out).abs().mean().item(),
            "triton_vs_torch_full_cosine": torch.nn.functional.cosine_similarity(
                triton_out.flatten(), torch_out.flatten(), dim=0
            ).item(),
            "triton_activation_quant_speedup_vs_torch": backend_results["torch"]["ms"]["activation_quant"]
            / max(backend_results["triton"]["ms"]["activation_quant"], 1e-9),
            "triton_full_speedup_vs_torch": backend_results["torch"]["ms"]["full"]
            / max(backend_results["triton"]["ms"]["full"], 1e-9),
        }
    else:
        backend_compare = {}
    if "fused_triton" in full_out_by_backend and "triton" in full_out_by_backend:
        fused_out = full_out_by_backend["fused_triton"].float()
        triton_out = full_out_by_backend["triton"].float()
        backend_compare.update(
            {
                "fused_vs_triton_full_max_abs": (fused_out - triton_out).abs().max().item(),
                "fused_vs_triton_full_mean_abs": (fused_out - triton_out).abs().mean().item(),
                "fused_vs_triton_full_cosine": torch.nn.functional.cosine_similarity(
                    fused_out.flatten(), triton_out.flatten(), dim=0
                ).item(),
                "fused_full_speedup_vs_triton": backend_results["triton"]["ms"]["full"]
                / max(backend_results["fused_triton"]["ms"]["full"], 1e-9),
            }
        )

    primary_backend = args.act_quant_backend[0]
    primary = backend_results[primary_backend]

    return {
        "weight_name": weight_name,
        "shape": {"m": args.m, "k": k, "n": n},
        "repeat": args.repeat,
        "warmup": args.warmup,
        "act_quant_backends": args.act_quant_backend,
        "primary_act_quant_backend": primary_backend,
        "ms": {
            "bf16_mm": bf16_ms,
            "w8a16_dequant_mm": w8a16_dequant_ms,
            "w8a8_activation_quant": primary["ms"]["activation_quant"],
            "w8a8_scaled_mm_only": primary["ms"]["scaled_mm_only"],
            "w8a8_full": primary["ms"]["full"],
        },
        "tflops": {
            "bf16_mm": flops / (bf16_ms / 1000.0) / 1e12,
            "w8a16_dequant_mm": flops / (w8a16_dequant_ms / 1000.0) / 1e12,
            "w8a8_scaled_mm_only": primary["tflops"]["scaled_mm_only"],
            "w8a8_full": primary["tflops"]["full"],
        },
        "ratios": {
            "activation_quant_pct_of_full": primary["ratios"]["activation_quant_pct_of_full"],
            "scaled_mm_pct_of_full": primary["ratios"]["scaled_mm_pct_of_full"],
            "w8a8_full_vs_bf16": primary["ratios"]["full_vs_bf16"],
            "w8a8_full_vs_w8a16": primary["ratios"]["full_vs_w8a16"],
        },
        "error_vs_bf16_dequant": primary["error_vs_bf16_dequant"],
        "w8a8_by_act_quant_backend": backend_results,
        "act_quant_backend_compare": backend_compare,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbenchmark FP8 W8A8 linear components")
    parser.add_argument("--w8a8-model-path", required=True)
    parser.add_argument("--label", default="fp8_linear_microbench")
    parser.add_argument("--weight-name", action="append", required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, default=0, help="Expected input dimension; inferred from qweight when omitted")
    parser.add_argument("--allow-fused-triton", action="store_true", help="Run experimental fused_triton backend")
    parser.add_argument(
        "--act-quant-backend",
        action="append",
        choices=["torch", "triton", "fused_triton"],
        default=["torch"],
    )
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output-json")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

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
                "ms.bf16_mm",
                "ms.w8a16_dequant_mm",
                "ms.w8a8_activation_quant",
                "ms.w8a8_scaled_mm_only",
                "ms.w8a8_full",
                "ratios.activation_quant_pct_of_full",
                "ratios.scaled_mm_pct_of_full",
                "ratios.w8a8_full_vs_bf16",
                "ratios.w8a8_full_vs_w8a16",
                "w8a8_by_act_quant_backend.torch.ms.activation_quant",
                "w8a8_by_act_quant_backend.triton.ms.activation_quant",
                "w8a8_by_act_quant_backend.fused_triton.ms.full",
                "act_quant_backend_compare.triton_activation_quant_speedup_vs_torch",
                "act_quant_backend_compare.triton_full_speedup_vs_torch",
                "act_quant_backend_compare.fused_full_speedup_vs_triton",
                "act_quant_backend_compare.fused_vs_triton_full_cosine",
            ],
        )


if __name__ == "__main__":
    main()
