from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import sys

import torch
from torch.utils.cpp_extension import load


def _ensure_env_path() -> None:
    """Ensure Python env bin is in PATH for CUDA compilation."""
    env_bin = str(Path(sys.executable).resolve().parent)
    path_items = os.environ.get("PATH", "").split(os.pathsep)
    if env_bin not in path_items:
        os.environ["PATH"] = os.pathsep.join([env_bin, *path_items])


@lru_cache(maxsize=1)
def load_w8a16_extension():
    _ensure_env_path()
    root = Path(__file__).resolve().parents[2]
    sources = [
        root / "nanovllm" / "quantization" / "csrc" / "w8a16_ext.cpp",
        root / "nanovllm" / "quantization" / "csrc" / "w8a16_kernel.cu",
    ]
    return load(
        name="nanovllm_w8a16_ext",
        sources=[str(path) for path in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        with_cuda=True,
        verbose=False,
    )


@lru_cache(maxsize=1)
def load_w8a8_cutlass_extension():
    """Load CUTLASS FP8 GEMM extension for W8A8."""
    _ensure_env_path()
    root = Path(__file__).resolve().parents[2]
    cutlass_include = root / "third_party" / "cutlass" / "include"
    sources = [
        root / "nanovllm" / "quantization" / "csrc" / "w8a8_cutlass_ext.cpp",
        root / "nanovllm" / "quantization" / "csrc" / "w8a8_cutlass_kernel.cu",
    ]
    return load(
        name="nanovllm_w8a8_cutlass_ext",
        sources=[str(path) for path in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-I/usr/local/cuda/include",
            f"-I{cutlass_include}",
        ],
        with_cuda=True,
        verbose=False,
    )


def launch_w8a16_cuda_ptx(
    x_bf16: torch.Tensor,
    w_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if not x_bf16.is_cuda or not w_fp8.is_cuda:
        raise ValueError("W8A16 CUDA/PTX backend requires CUDA tensors")
    ext = load_w8a16_extension()
    return ext.w8a16_cuda_forward(x_bf16, w_fp8, weight_scale, bias)


def launch_w8a8_cutlass_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Launch CUTLASS FP8 GEMM for W8A8.

    Args:
        x: BF16 activation [M, K]
        w: FP8 weight [N, K] column-major
        x_scale: float scalar for activation quantization
        w_scale: float scalar for weight
        bias: optional BF16 bias [N]

    Returns:
        BF16 output [M, N]
    """
    if not x.is_cuda or not w.is_cuda:
        raise ValueError("W8A8 CUTLASS backend requires CUDA tensors")
    ext = load_w8a8_cutlass_extension()
    return ext.w8a8_cutlass_gemm(x, w, x_scale, w_scale, bias)
