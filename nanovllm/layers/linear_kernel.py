"""LinearKernel abstraction for quantized GEMM backends.

Design rationale (from quant_layer_audit.md):
- All three existing quantization schemes (AWQ INT4, INT8 W8A8, FP8 W8A16/W8A8)
  share the same high-level contract: given an activation tensor ``x`` and
  pre-loaded weight buffers, produce an output tensor.
- The buffer naming and layout differ significantly across schemes, so rather
  than forcing a unified buffer contract, each ``LinearKernel`` subclass owns
  its own buffer names and packs/unpacks them internally.
- TP all_reduce is **not** the kernel's responsibility; it is handled by the
  surrounding parallel linear layer (RowParallelLinear pattern).
- ``group_size`` is included as an optional constructor argument to accommodate
  AWQ-style per-group quantization.  FP8 and INT8 kernels ignore it.

Stage 4 migration plan:
  1. Implement ``AWQKernel``, ``FP8Kernel``, ``W8A8Kernel``.
  2. Replace direct ``F.linear`` / triton calls in the parallel linear
     classes with ``self.kernel.forward(x, self.quant_buffers())``.
  3. Add per-kernel numerical tests in ``tests/quant/`` before migration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class LinearKernel(ABC):
    """Abstract base for a quantized linear kernel.

    A ``LinearKernel`` is **not** an ``nn.Module``; it is a stateless
    compute strategy that the surrounding ``nn.Module`` delegates to.
    All weight buffers remain registered on the parent module.

    Subclasses must implement ``forward`` which accepts the activation
    tensor ``x`` and the quantization buffers as keyword arguments.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, **buffers: torch.Tensor) -> torch.Tensor:
        """Compute the quantized GEMM.

        Parameters
        ----------
        x:
            Activation tensor, shape ``[*, K]``.  The kernel is responsible
            for reshaping/casting as required by the hardware instruction.
        **buffers:
            Quantization buffers owned by the parent module.  Each kernel
            subclass documents which keyword names it expects.

        Returns
        -------
        Tensor of shape ``[*, N]``, dtype matches ``x`` (or ``bfloat16``
        if the hardware forces it; caller is responsible for the cast-back).
        """

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


class DenseKernel(LinearKernel):
    """Unquantized (FP16/BF16) kernel: thin wrapper around ``F.linear``.

    Expected buffers
    ----------------
    weight : Tensor [N, K]
        Float weight matrix (already sharded for the local TP rank).
    bias : Tensor [N] or None
        Optional bias.
    """

    def forward(self, x: torch.Tensor, **buffers: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        weight = buffers["weight"]
        bias = buffers.get("bias")
        return F.linear(x, weight, bias)


class AWQKernel(LinearKernel):
    """AWQ INT4 kernel: unpack qweight on-the-fly then ``F.linear``.

    Expected buffers
    ----------------
    qweight  : Tensor [K, N/8] int32
        Packed INT4 weights (vLLM/AutoAWQ physical layout with interleave).
    qzeros   : Tensor [K//group_size, N/8] int32
        Packed zero-points.
    scales   : Tensor [K//group_size, N] float16
        Per-group scales.
    bias     : Tensor [N] or None
        Optional bias.

    Notes
    -----
    ``group_size`` is a constructor argument (default 128).
    The unpack logic lives in ``unpack_awq_int4`` (quant_linear.py).
    This class is intentionally **not** referencing that function yet;
    full migration happens in Stage 4 after the test baseline is locked.
    """

    def __init__(self, group_size: int = 128) -> None:
        self.group_size = group_size

    def forward(self, x: torch.Tensor, **buffers: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "AWQKernel.forward is a Stage 4 placeholder.  "
            "Use AWQRowParallelLinear / AWQMergedColumnParallelLinear directly for now."
        )

    def extra_repr(self) -> str:
        return f"group_size={self.group_size}"


class FP8Kernel(LinearKernel):
    """FP8 kernel supporting W8A16 and static W8A8 schemes.

    Expected buffers (W8A16)
    ------------------------
    qweight      : Tensor [K, N] uint8  (view as float8_e4m3fn internally)
    weight_scale : Tensor [N] float32   (per-channel)
    bias         : Tensor [N] or None

    Additional buffers for W8A8 static
    -----------------------------------
    input_scale  : Tensor [1] float32
    qweight_nk   : Tensor [N, K] uint8  (transposed cache, may be None)

    Notes
    -----
    Full migration is a Stage 4 task.  This class is a typed placeholder.
    """

    def __init__(self, fp8_scheme: str = "w8a16") -> None:
        assert fp8_scheme in ("w8a16", "w8a8_static"), f"Unknown scheme: {fp8_scheme}"
        self.fp8_scheme = fp8_scheme

    def forward(self, x: torch.Tensor, **buffers: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "FP8Kernel.forward is a Stage 4 placeholder.  "
            "Use FP8RowParallelLinear / FP8QKVParallelLinear directly for now."
        )

    def extra_repr(self) -> str:
        return f"fp8_scheme={self.fp8_scheme!r}"


class W8A8Kernel(LinearKernel):
    """INT8 W8A8 dynamic-activation kernel (Triton _w8a8_linear_kernel).

    Expected buffers
    ----------------
    qweight_kn   : Tensor [K, N] int8   (note: buffer name differs from FP8!)
    weight_scales : Tensor [N] float16  (per-channel; note: differs from FP8 float32)
    bias          : Tensor [N] or None

    Notes
    -----
    Buffer name ``qweight_kn`` differs from FP8's ``qweight``; this asymmetry
    is preserved intentionally until Stage 4 unification.
    Full migration is a Stage 4 task.
    """

    def forward(self, x: torch.Tensor, **buffers: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "W8A8Kernel.forward is a Stage 4 placeholder.  "
            "Use Int8RowParallelLinear / Int8QKVParallelLinear directly for now."
        )


__all__ = [
    "LinearKernel",
    "DenseKernel",
    "AWQKernel",
    "FP8Kernel",
    "W8A8Kernel",
]
