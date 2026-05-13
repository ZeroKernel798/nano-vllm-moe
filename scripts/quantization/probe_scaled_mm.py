from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from common import nvidia_smi_query, write_json

FP8_MAX = 448.0


def quantize_tensorwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (x.float().abs().amax() / FP8_MAX).clamp(min=1e-12).to(torch.float32)
    q = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale.reshape(())


def make_col_major(w: torch.Tensor) -> torch.Tensor:
    k, n = w.shape
    out = torch.empty_strided((k, n), (1, k), device=w.device, dtype=w.dtype)
    out.copy_(w)
    return out


def bench(fn, repeat: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (perf_counter() - start) / repeat


def try_scaled_mm(a_fp8: torch.Tensor, b_fp8: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> dict[str, Any]:
    try:
        out = torch._scaled_mm(a_fp8, b_fp8, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        return {
            "ok": True,
            "shape": list(out.shape),
            "dtype": str(out.dtype),
            "max_abs": float(out.float().abs().max().item()),
        }
    except Exception as exc:  # noqa: BLE001 - probe should report exact failure
        return {"ok": False, "error": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe torch._scaled_mm FP8 support on this GPU")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if not hasattr(torch, "_scaled_mm"):
        raise SystemExit("torch._scaled_mm is not available")

    torch.manual_seed(args.seed)
    device = "cuda"
    a_bf16 = torch.randn((args.m, args.k), device=device, dtype=torch.bfloat16)
    w_bf16 = torch.randn((args.k, args.n), device=device, dtype=torch.bfloat16)
    a_fp8, scale_a = quantize_tensorwise(a_bf16)
    w_fp8, scale_b = quantize_tensorwise(w_bf16)
    w_fp8_col_major = make_col_major(w_fp8)

    scale_b_vec = torch.full((args.n,), scale_b.item(), device=device, dtype=torch.float32)
    scale_b_1xn = scale_b_vec.reshape(1, args.n)
    scale_a_vec_m = torch.full((args.m,), scale_a.item(), device=device, dtype=torch.float32)
    scale_a_mx1 = scale_a_vec_m.reshape(args.m, 1)

    cases = {
        "row_major_scalar_scalar": (a_fp8, w_fp8, scale_a, scale_b),
        "col_major_scalar_scalar": (a_fp8, w_fp8_col_major, scale_a, scale_b),
        "col_major_scalar_vec_n": (a_fp8, w_fp8_col_major, scale_a, scale_b_vec),
        "col_major_scalar_1xn": (a_fp8, w_fp8_col_major, scale_a, scale_b_1xn),
        "col_major_vec_m_scalar": (a_fp8, w_fp8_col_major, scale_a_vec_m, scale_b),
        "col_major_mx1_scalar": (a_fp8, w_fp8_col_major, scale_a_mx1, scale_b),
    }

    result: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": nvidia_smi_query(),
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "cases": {},
    }
    ref = a_bf16.float() @ w_bf16.float()
    for name, (a, b, sa, sb) in cases.items():
        case = try_scaled_mm(a, b, sa, sb)
        if case["ok"]:
            out = torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
            diff = (out.float() - ref).abs()
            case["mean_abs_error"] = float(diff.mean().item())
            case["max_abs_error"] = float(diff.max().item())
            case["time_s"] = bench(lambda: torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16), args.repeat, args.warmup)
        result["cases"][name] = case
        print(f"{name}: {case}")

    bf16_time = bench(lambda: a_bf16 @ w_bf16, args.repeat, args.warmup)
    result["bf16_matmul_time_s"] = bf16_time
    print(f"bf16_matmul_time_s={bf16_time}")
    write_json(args.output_json, result)


if __name__ == "__main__":
    main()
