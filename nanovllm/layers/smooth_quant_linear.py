"""INT8 quantized parallel linear layers.

Two inference schemes share the same checkpoint format:

W8A16 (weight-only)
  - Weights stored as int8 with per-output-channel float32 scale.
  - Activations remain in bfloat16.
  - Compute: dequant weight → BF16 → standard cuBLAS GEMM.
  - No calibration required.

W8A8 static
  - Weights: same as W8A16.
  - Activations: quantized with a pre-computed per-layer scalar ``input_scale``
    (calibrated offline; eliminates per-token abs-max overhead).
  - Compute: for large batch (M > 32) → cuBLAS INT8 GEMM (``torch._int_mm``
    with M-padding to next multiple of 32, required by RTX 3080/GA102);
    for small batch (M ≤ 32) → BF16 fallback (padding cost exceeds benefit).
  - Requires calibration step in ``scripts/quantize/quantize.py``.

Checkpoint suffix naming (aligned with FP8 layers):
  ``qweight``      [N, K] int8    (saved) / [K, N] int8 (in-memory, K-major)
  ``weight_scale`` [N]    float32
  ``input_scale``  [1]    float32  (W8A8 static only)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist


# ---------------------------------------------------------------------------
# INT8 GEMM helpers
# ---------------------------------------------------------------------------

# RTX 3080 (GA102) cuBLAS INT8 constraint: M must be a multiple of 32 and > 16.
_INT8_MM_MIN_M = 32


def _int8_mm_padded(x_int8: torch.Tensor, qweight: torch.Tensor) -> torch.Tensor:
    """``torch._int_mm`` with M-padding to satisfy cuBLAS INT8 alignment.

    Args:
        x_int8:  [M, K] int8 — quantized activations
        qweight: [K, N] int8 — quantized weight (K-major)
    Returns:
        [M, N] int32
    """
    M = x_int8.shape[0]
    pad_m = (_INT8_MM_MIN_M - M % _INT8_MM_MIN_M) % _INT8_MM_MIN_M
    if M <= 16:  # cuBLAS requires M > 16 as well
        pad_m = _INT8_MM_MIN_M - M
    if pad_m > 0:
        x_int8 = F.pad(x_int8.float(), (0, 0, 0, pad_m)).to(torch.int8)
    out = torch._int_mm(x_int8, qweight)
    return out[:M]


def _divide(n: int, d: int) -> int:
    assert n % d == 0, f"{n} is not divisible by {d}"
    return n // d


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Int8LinearBase(nn.Module):
    """Common INT8 quantized linear base.

    Parameters
    ----------
    int8_scheme : ``"w8a16"`` or ``"w8a8_static"``
        Selects the inference path (see module docstring).

    Buffers
    -------
    qweight      : [K, N] int8   — weight in K-major layout (transposed from HF)
    weight_scale : [N]    float32 — per-output-channel dequant scale
    input_scale  : [1]    float32 — activation scale  (``w8a8_static`` only)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        int8_scheme: str = "w8a16",
        tp_dim=None,
        tp_group=None,
        **_kwargs,
    ):
        super().__init__()
        assert int8_scheme in ("w8a16", "w8a8_static"), \
            f"Unknown int8_scheme: {int8_scheme!r}"
        self.input_size = input_size
        self.output_size = output_size
        self.int8_scheme = int8_scheme
        self.tp_group = tp_group
        self.tp_size = (
            dist.get_world_size(tp_group) if tp_group is not None
            else (dist.get_world_size() if dist.is_initialized() else 1)
        )
        self.tp_rank = (
            dist.get_rank(tp_group) if tp_group is not None
            else (dist.get_rank() if dist.is_initialized() else 0)
        )

    def _init_quant_buffers(self, in_features: int, out_features: int):
        self.register_buffer("qweight", torch.zeros((in_features, out_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.zeros((out_features,), dtype=torch.float32))
        if self.int8_scheme == "w8a8_static":
            self.register_buffer("input_scale", torch.ones(1, dtype=torch.float32))

    # ------------------------------------------------------------------
    # W8A16: dequant weight → BF16 → cuBLAS GEMM
    # ------------------------------------------------------------------
    def _forward_w8a16(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_2d = x.view(-1, x.shape[-1]).to(torch.bfloat16)
        # qweight [K,N] → F.linear expects [N,K]: use .t()
        w = self.qweight.to(torch.bfloat16).t()               # [N, K]
        w = w * self.weight_scale.to(torch.bfloat16).unsqueeze(1)
        bias = getattr(self, "bias", None)
        out = F.linear(x_2d, w,
                       bias.to(torch.bfloat16) if bias is not None else None)
        return out.to(orig_dtype).view(*x.shape[:-1], -1)

    # ------------------------------------------------------------------
    # W8A8 static: calibrated input_scale, INT8 GEMM for large M
    # ------------------------------------------------------------------
    def _forward_w8a8_static(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_2d = x.view(-1, x.shape[-1])
        M = x_2d.shape[0]

        # Quantize activation with pre-computed scale (single scalar divide)
        x_int8 = x_2d.float().div(self.input_scale).clamp(-128, 127).round().to(torch.int8)

        if M > _INT8_MM_MIN_M:
            # cuBLAS INT8 GEMM path (DP4A Tensor Cores)
            out_i32 = _int8_mm_padded(x_int8, self.qweight)   # [M, N] int32
            out = out_i32.float() * self.input_scale * self.weight_scale.unsqueeze(0)
        else:
            # BF16 fallback: avoid padding overhead for small batch (decode)
            w = self.qweight.to(torch.bfloat16).t()
            w = w * self.weight_scale.to(torch.bfloat16).unsqueeze(1)
            out = F.linear(x_2d.to(torch.bfloat16), w).float()

        bias = getattr(self, "bias", None)
        if bias is not None:
            out = out + bias.float()
        return out.to(orig_dtype).view(*x.shape[:-1], -1)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.int8_scheme == "w8a8_static":
            return self._forward_w8a8_static(x)
        return self._forward_w8a16(x)


# ---------------------------------------------------------------------------
# Parallel variants
# ---------------------------------------------------------------------------

class Int8MergedColumnParallelLinear(Int8LinearBase):
    """Gate+up fused MLP projection (INT8)."""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        int8_scheme: str = "w8a16",
        tp_group=None,
        **kwargs,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), int8_scheme, tp_dim=1,
                         tp_group=tp_group, **kwargs)
        self._init_quant_buffers(input_size, sum(output_sizes) // self.tp_size)

        self.qweight.weight_loader = self._qweight_loader
        self.weight_scale.weight_loader = self._scale_loader
        if int8_scheme == "w8a8_static":
            self.input_scale.weight_loader = self._input_scale_loader

        self.bias = nn.Parameter(torch.empty(sum(output_sizes) // self.tp_size)) if bias else None

    def _qweight_loader(self, param, loaded_weight, shard_id: int):
        offset = sum(self.output_sizes[:shard_id]) // self.tp_size
        size = self.output_sizes[shard_id] // self.tp_size
        # loaded: [N_shard, K] → transpose to [K, N_shard] (K-major)
        param.data.narrow(1, offset, size).copy_(loaded_weight.t().contiguous())

    def _scale_loader(self, param, loaded_weight, shard_id: int):
        offset = sum(self.output_sizes[:shard_id]) // self.tp_size
        size = self.output_sizes[shard_id] // self.tp_size
        param.data.narrow(0, offset, size).copy_(loaded_weight.view(-1).to(torch.float32))

    def _input_scale_loader(self, param, loaded_weight, shard_id):
        flat = loaded_weight.view(-1)
        param.data.copy_(flat[:1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)


class Int8QKVParallelLinear(Int8LinearBase):
    """QKV fused attention projection (INT8)."""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        bias: bool = False,
        int8_scheme: str = "w8a16",
        tp_group=None,
        **kwargs,
    ):
        self.head_size = head_size
        tp_size = (
            dist.get_world_size(tp_group) if tp_group is not None
            else (dist.get_world_size() if dist.is_initialized() else 1)
        )
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = total_num_kv_heads // tp_size
        out_f = (self.num_heads + 2 * self.num_kv_heads) * head_size
        super().__init__(hidden_size, out_f, int8_scheme, tp_dim=1, tp_group=tp_group, **kwargs)
        self._init_quant_buffers(hidden_size, out_f)

        self.qweight.weight_loader = self._qweight_loader
        self.weight_scale.weight_loader = self._scale_loader
        if int8_scheme == "w8a8_static":
            self.input_scale.weight_loader = self._input_scale_loader

        if bias:
            self.bias = nn.Parameter(torch.empty(out_f))
            self.bias.weight_loader = self._bias_loader
        else:
            self.register_parameter("bias", None)

    def _qkv_slice(self, shard_id: str):
        if shard_id == "q":
            return self.num_heads * self.head_size, 0
        elif shard_id == "k":
            return self.num_kv_heads * self.head_size, self.num_heads * self.head_size
        else:
            return (self.num_kv_heads * self.head_size,
                    (self.num_heads + self.num_kv_heads) * self.head_size)

    def _qweight_loader(self, param, loaded_weight, shard_id):
        size, offset = self._qkv_slice(shard_id)
        param.data.narrow(1, offset, size).copy_(loaded_weight.t().contiguous())

    def _scale_loader(self, param, loaded_weight, shard_id):
        size, offset = self._qkv_slice(shard_id)
        param.data.narrow(0, offset, size).copy_(loaded_weight.view(-1).to(torch.float32))

    def _bias_loader(self, param, loaded_weight, shard_id):
        size, offset = self._qkv_slice(shard_id)
        param.data.narrow(0, offset, size).copy_(loaded_weight.view(-1))

    def _input_scale_loader(self, param, loaded_weight, shard_id):
        flat = loaded_weight.view(-1)
        param.data.copy_(flat[:1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)


class Int8RowParallelLinear(Int8LinearBase):
    """Output projection (INT8, all-reduce for TP > 1)."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        reduce_results: bool = True,
        int8_scheme: str = "w8a16",
        tp_group=None,
        **kwargs,
    ):
        super().__init__(input_size, output_size, int8_scheme, tp_dim=0, tp_group=tp_group, **kwargs)
        self.reduce_results = reduce_results
        self._init_quant_buffers(_divide(input_size, self.tp_size), output_size)

        self.qweight.weight_loader = lambda p, w, *_: p.data.copy_(w.t().contiguous())
        self.weight_scale.weight_loader = lambda p, w, *_: p.data.copy_(w.view(-1).to(torch.float32))
        if int8_scheme == "w8a8_static":
            self.input_scale.weight_loader = lambda p, w, *_: p.data.copy_(w.view(-1)[:1])

        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._forward(x)
        if self.tp_size > 1 and self.reduce_results:
            dist.all_reduce(out, group=self.tp_group)
        return out
