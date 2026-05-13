from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.nn.functional as F
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm.executor.moe.config import make_moe_parallel_config
from nanovllm.executor.moe.experts import OptimizedExperts, EagerExperts, FusedExperts
from nanovllm.executor.moe.kernel import MoEKernel
from nanovllm.executor.moe.prepare_finalize import NoEPPrepareFinalize


BACKENDS = {
    "eager": EagerExperts,
    "optimized": OptimizedExperts,
    "fused": FusedExperts,
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def make_inputs(args, num_tokens: int, device: torch.device, dtype: torch.dtype):
    torch.manual_seed(args.seed + num_tokens)
    x = torch.randn(num_tokens, args.hidden_size, device=device, dtype=dtype) * args.input_scale
    w13 = torch.randn(args.num_experts, 2 * args.intermediate_size, args.hidden_size, device=device, dtype=dtype) * args.weight_scale
    w2 = torch.randn(args.num_experts, args.hidden_size, args.intermediate_size, device=device, dtype=dtype) * args.weight_scale
    topk_ids = torch.randint(0, args.num_experts, (num_tokens, args.top_k), device=device, dtype=torch.long)
    topk_weights = torch.rand(num_tokens, args.top_k, device=device, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return x, topk_weights, topk_ids, w13, w2


def make_kernel(backend: str, args):
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported backend={backend}; choices={sorted(BACKENDS)}")
    parallel_config = make_moe_parallel_config(
        tp_group=None,
        ep_group=None,
        global_num_experts=args.num_experts,
        intermediate_size=args.intermediate_size,
    )
    return MoEKernel(
        prepare_finalize=NoEPPrepareFinalize(parallel_config),
        experts=BACKENDS[backend](),
        parallel_config=parallel_config,
    )


def run_once(kernel, inputs, dtype: torch.dtype):
    x, topk_weights, topk_ids, w13, w2 = inputs
    return kernel(x, topk_weights, topk_ids, w13, w2, model_dtype=dtype)


def cuda_time(fn, device: torch.device) -> tuple[float, torch.Tensor]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0, out


def compare(reference: torch.Tensor, output: torch.Tensor) -> dict[str, float]:
    ref = reference.float().flatten()
    out = output.float().flatten()
    return {
        "max_abs_vs_eager": (out - ref).abs().max().item(),
        "mean_abs_vs_eager": (out - ref).abs().mean().item(),
        "cosine_vs_eager": F.cosine_similarity(out, ref, dim=0).item(),
    }


def estimate_moe_flops(num_tokens: int, top_k: int, hidden_size: int, intermediate_size: int) -> int:
    return num_tokens * top_k * 6 * hidden_size * intermediate_size


def estimate_moe_bytes(
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    dtype_bytes: int,
    output_bytes: int,
) -> int:
    expert_tokens = num_tokens * top_k
    active_experts = min(num_experts, expert_tokens)
    activation_bytes = expert_tokens * (
        hidden_size * dtype_bytes
        + 2 * intermediate_size * dtype_bytes
        + intermediate_size * dtype_bytes
        + hidden_size * output_bytes
    )
    weight_bytes = active_experts * (
        2 * intermediate_size * hidden_size * dtype_bytes
        + hidden_size * intermediate_size * dtype_bytes
    )
    return activation_bytes + weight_bytes


def apply_model_config(args) -> None:
    if not args.model_config:
        return
    config = AutoConfig.from_pretrained(args.model_config, trust_remote_code=True)
    args.hidden_size = int(config.hidden_size)
    args.intermediate_size = int(getattr(config, "moe_intermediate_size"))
    args.num_experts = int(config.num_experts)
    args.top_k = int(config.num_experts_per_tok)
    args.model_type = getattr(config, "model_type", "unknown")


def bench_case(
    args,
    backend: str,
    num_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    inputs,
    eager_out: torch.Tensor | None,
):
    kernel = make_kernel(backend, args)
    for _ in range(args.warmup):
        run_once(kernel, inputs, dtype)
    samples = []
    output = None
    for _ in range(args.repeat):
        elapsed_ms, output = cuda_time(lambda: run_once(kernel, inputs, dtype), device)
        samples.append(elapsed_ms)
    mean_ms = mean(samples)
    flops = estimate_moe_flops(num_tokens, args.top_k, args.hidden_size, args.intermediate_size)
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    estimated_bytes = estimate_moe_bytes(
        num_tokens,
        args.top_k,
        args.hidden_size,
        args.intermediate_size,
        args.num_experts,
        dtype_bytes,
        args.output_bytes,
    )
    effective_tflops = flops / max(mean_ms / 1000.0, 1e-12) / 1e12
    effective_tbps = estimated_bytes / max(mean_ms / 1000.0, 1e-12) / 1e12
    result = {
        "backend": backend,
        "num_tokens": num_tokens,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "model_config": args.model_config,
        "model_type": getattr(args, "model_type", "synthetic"),
        "repeat": args.repeat,
        "mean_ms": mean_ms,
        "stdev_ms": stdev(samples) if len(samples) > 1 else 0.0,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "tokens_per_s": num_tokens / max(mean_ms, 1e-9) * 1000.0,
        "expert_tokens_per_s": num_tokens * args.top_k / max(mean_ms, 1e-9) * 1000.0,
        "estimated_flops": flops,
        "estimated_bytes": estimated_bytes,
        "effective_tflops": effective_tflops,
        "effective_tbps": effective_tbps,
        "peak_tflops": args.peak_tflops,
        "peak_tbps": args.peak_tbps,
        "mfu_percent": effective_tflops / args.peak_tflops * 100.0 if args.peak_tflops > 0 else 0.0,
        "mbu_percent": effective_tbps / args.peak_tbps * 100.0 if args.peak_tbps > 0 else 0.0,
    }
    if eager_out is not None and output is not None:
        result.update(compare(eager_out, output))
    return result


def write_csv(path: str, rows: list[dict]) -> None:
    if not path:
        return
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic single-device MoE local compute benchmark")
    parser.add_argument("--backends", default="eager,optimized,fused")
    parser.add_argument("--num-tokens", default="1,8,32,128,512,1024,2048,4096,8192")
    parser.add_argument("--model-config", default="")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--weight-scale", type=float, default=0.1)
    parser.add_argument("--peak-tflops", type=float, default=330.0)
    parser.add_argument("--peak-tbps", type=float, default=1.0)
    parser.add_argument("--output-bytes", type=int, default=4)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    apply_model_config(args)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    dtype = getattr(torch, args.dtype)
    rows = []
    for num_tokens in parse_int_csv(args.num_tokens):
        inputs = make_inputs(args, num_tokens, device, dtype)
        eager_kernel = make_kernel("eager", args)
        eager_out = run_once(eager_kernel, inputs, dtype)
        for backend in parse_csv(args.backends):
            row = bench_case(args, backend, num_tokens, device, dtype, inputs, eager_out)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "results": rows}, f, indent=2)
    write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
