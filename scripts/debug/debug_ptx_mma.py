#!/usr/bin/env python3
"""Debug script for W8A8 PTX MMA kernel fragment layout.

This script creates synthetic test cases to verify the MMA fragment layout
without the complexity of real model weights.

Usage:
    python debug_ptx_mma.py --m 16 --n 256 --k 256
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

# Force CUDA PTX backend
os.environ["NANOVLLM_FP8_W8A8_RUNTIME_BACKEND"] = "cuda_ptx"
os.environ["NANOVLLM_W8A8_JIT_KERNEL"] = "ptx_mma"

from nanovllm.quantization.w8a8_jit import launch_w8a8_cuda_ptx_jit


def create_test_matrices(m: int, n: int, k: int, seed: int = 42, pattern: str = "ramp"):
    """Create test matrices with known values."""
    torch.manual_seed(seed)

    if pattern == "ones":
        x_bf16 = torch.ones((m, k), dtype=torch.bfloat16, device="cuda")
        w_fp8_values = torch.ones((n, k), dtype=torch.float32, device="cuda")
        input_scale_value = 1.0
        weight_scale_value = 1.0
    elif pattern == "identity":
        x_values = torch.zeros((m, k), dtype=torch.float32, device="cuda")
        w_fp8_values = torch.zeros((n, k), dtype=torch.float32, device="cuda")
        for i in range(min(m, k)):
            x_values[i, i] = 1.0
        for i in range(min(n, k)):
            w_fp8_values[i, i] = 1.0
        x_bf16 = x_values.to(torch.bfloat16)
        input_scale_value = 1.0
        weight_scale_value = 1.0
    elif pattern == "xones_wramp":
        x_bf16 = torch.ones((m, k), dtype=torch.bfloat16, device="cuda")
        w_fp8_values = torch.arange(k, dtype=torch.float32, device="cuda").repeat(n, 1) / 16.0
        input_scale_value = 1.0
        weight_scale_value = 16.0
    elif pattern == "xramp_wones":
        x_values = torch.arange(k, dtype=torch.float32, device="cuda").repeat(m, 1) / 16.0
        x_bf16 = x_values.to(torch.bfloat16)
        w_fp8_values = torch.ones((n, k), dtype=torch.float32, device="cuda")
        input_scale_value = 16.0
        weight_scale_value = 1.0
    elif pattern == "ramp":
        x_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        w_fp8_values = torch.arange(k, dtype=torch.float32, device="cuda").repeat(n, 1) / 16.0
        input_scale_value = 16.0
        weight_scale_value = 16.0
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    w_fp8 = w_fp8_values.to(torch.float8_e4m3fn)

    # Scales
    input_scale = torch.tensor([input_scale_value], dtype=torch.float32, device="cuda")
    weight_scale = torch.tensor([weight_scale_value], dtype=torch.float32, device="cuda")

    return x_bf16, w_fp8, input_scale, weight_scale


def reference_gemm(x_bf16: torch.Tensor, w_fp8: torch.Tensor,
                   input_scale: torch.Tensor, weight_scale: torch.Tensor):
    """Reference GEMM using BF16 dequant."""
    # Dequantize weight to BF16
    w_bf16 = w_fp8.to(torch.bfloat16) * weight_scale.to(torch.bfloat16)

    # Quantize activation
    x_quant = (x_bf16.float() / input_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    x_dequant = x_quant.to(torch.bfloat16) * input_scale.to(torch.bfloat16)

    # GEMM
    out = torch.mm(x_dequant, w_bf16.t())  # w_bf16 is (N, K), need transpose for (K, N)

    return out


def test_ptx_mma(m: int, n: int, k: int, pattern: str = "ramp"):
    """Test PTX MMA kernel with synthetic data."""
    print(f"\nTesting PTX MMA: M={m}, N={n}, K={k}")
    print("-" * 50)

    # Only test with K divisible by 32 for PTX MMA
    if k % 32 != 0:
        print(f"K must be divisible by 32 for PTX MMA, got K={k}")
        return

    x_bf16, w_fp8, input_scale, weight_scale = create_test_matrices(m, n, k, pattern=pattern)

    # Reference result
    with torch.no_grad():
        ref_out = reference_gemm(x_bf16, w_fp8, input_scale, weight_scale)

    # PTX MMA result
    try:
        ptx_out = launch_w8a8_cuda_ptx_jit(x_bf16, w_fp8, input_scale, weight_scale, None)

        # Compare
        diff = (ptx_out.float() - ref_out.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Cosine similarity
        cosine = torch.nn.functional.cosine_similarity(
            ptx_out.float().flatten(), ref_out.float().flatten(), dim=0
        ).item()

        print(f"Reference output shape: {ref_out.shape}")
        print(f"PTX MMA output shape: {ptx_out.shape}")
        print(f"Max absolute error: {max_diff:.6f}")
        print(f"Mean absolute error: {mean_diff:.6f}")
        print(f"Cosine similarity: {cosine:.6f}")

        print(f"Pattern: {pattern}")
        print("\nFirst row comparison:")
        print(f"  Reference: {ref_out[0, :min(8, n)].tolist()}")
        print(f"  PTX MMA:   {ptx_out[0, :min(8, n)].tolist()}")
        print("First column comparison:")
        print(f"  Reference: {ref_out[:min(8, m), 0].tolist()}")
        print(f"  PTX MMA:   {ptx_out[:min(8, m), 0].tolist()}")

        if cosine > 0.999:
            print("\n✅ PTX MMA PASSED")
        else:
            print("\n❌ PTX MMA FAILED - numerical error detected")

    except Exception as e:
        print(f"❌ PTX MMA FAILED with error: {e}")
        import traceback
        traceback.print_exc()


def test_fragment_layout():
    """Test individual fragment loading patterns."""
    print("\n" + "=" * 60)
    print("Fragment Layout Debug")
    print("=" * 60)

    # Test with small K=32 to isolate fragment issues
    m, n, k = 16, 8, 32

    print(f"\nTest case: M={m}, N={n}, K={k}")
    print("Expected thread layout for m16n8k32 FP8 MMA is validated by patterns below.")

    for pattern in ("ones", "identity", "xones_wramp", "xramp_wones", "ramp"):
        test_ptx_mma(m, n, k, pattern=pattern)


def main():
    parser = argparse.ArgumentParser(description="Debug PTX MMA fragment layout")
    parser.add_argument("--m", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--n", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--k", type=int, nargs="+", default=[256, 512])
    parser.add_argument(
        "--pattern",
        choices=["ramp", "ones", "identity", "xones_wramp", "xramp_wones"],
        default="ramp",
    )
    parser.add_argument("--debug-fragment", action="store_true", help="Run fragment layout debug")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required")
        return

    if args.debug_fragment:
        test_fragment_layout()
        return

    # Test various shapes
    for m in args.m:
        for n in args.n:
            for k in args.k:
                if k % 32 == 0:  # Only test valid K for PTX MMA
                    test_ptx_mma(m, n, k, pattern=args.pattern)


if __name__ == "__main__":
    main()
