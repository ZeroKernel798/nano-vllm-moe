"""FP8 parallel linear layers: W8A16 (bf16 act) and static W8A8 (fp8 act + calibrated input scale)."""

from __future__ import annotations

import torch
from torch import nn
import torch.distributed as dist

from nanovllm.layers.fp8.kernels import launch_w8a16_gemm, launch_w8a8_static_gemm


def divide(numerator: int, denominator: int) -> int:
    assert numerator % denominator == 0
    return numerator // denominator


def _matmul_fp8_static_ptq(
    x_2d: torch.Tensor,
    input_scale: torch.Tensor,
    qweight_uint8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    qweight_nk: torch.Tensor | None,
) -> torch.Tensor:
    """W8A8 static linear: cuBLASLt FP8 ``_scaled_mm`` in ``launch_w8a8_static_gemm``."""
    dev = x_2d.device
    x = x_2d
    if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.device != dev:
        x = x_2d.to(device=dev, dtype=torch.bfloat16).contiguous()
    w_fp8 = qweight_uint8.to(device=dev).view(torch.float8_e4m3fn)
    nk = qweight_nk.to(device=dev) if qweight_nk is not None else None
    return launch_w8a8_static_gemm(x, input_scale, w_fp8, weight_scale, bias, w_fp8_nk=nk)


def _load_weight_scale(param: torch.Tensor, loaded: torch.Tensor, offset: int, size: int) -> None:
    """Scalar or per-channel weight scale into param[offset:offset+size]."""
    flat = loaded.view(-1)
    if flat.numel() == 1:
        param.data.narrow(0, offset, size).fill_(flat.item())
    else:
        param.data.narrow(0, offset, size).copy_(flat)


class FP8LinearBase(nn.Module):
    """Shared buffers: qweight [K,N] uint8, weight_scale [N]; optional input_scale [1] for W8A8 static."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        fp8_scheme: str,
        tp_group=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fp8_scheme = fp8_scheme
        self.tp_size = (
            dist.get_world_size(tp_group)
            if tp_group is not None
            else (dist.get_world_size() if dist.is_initialized() else 1)
        )
        self.tp_group = tp_group

    def _ensure_qweight_nk(self) -> None:
        """Fill ``qweight_nk`` once after ``qweight`` is loaded (static inference)."""
        if self.fp8_scheme != "w8a8_static" or getattr(self, "_qweight_nk_synced", False):
            return
        with torch.no_grad():
            w = self.qweight.view(torch.float8_e4m3fn)
            self.qweight_nk.copy_(w.t().contiguous().view(torch.uint8))
        self._qweight_nk_synced = True

    def _init_quant_buffers(self, in_features: int, out_features: int) -> None:
        self.register_buffer("qweight", torch.zeros((in_features, out_features), dtype=torch.uint8))
        self.register_buffer("weight_scale", torch.zeros((out_features,), dtype=torch.float32))
        if self.fp8_scheme == "w8a8_static":
            self.register_buffer("input_scale", torch.ones(1, dtype=torch.float32))
            # [N, K] contiguous row-major copy of qweight.T — avoids per-forward O(KN) transpose alloc.
            self.register_buffer("qweight_nk", torch.empty((out_features, in_features), dtype=torch.uint8))
            self._qweight_nk_synced = False

    def _forward_w8a16(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = x.dtype
        x_2d = x.view(-1, x.shape[-1])
        if x_2d.dtype != torch.bfloat16:
            x_2d = x_2d.to(torch.bfloat16)
        w_fp8 = self.qweight.view(torch.float8_e4m3fn)
        out = launch_w8a16_gemm(x_2d, w_fp8, self.weight_scale, getattr(self, "bias", None))
        return out.view(*x.shape[:-1], -1).to(target_dtype)

    def _forward_w8a8_static(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = x.dtype
        x_2d = x.view(-1, x.shape[-1])
        self._ensure_qweight_nk()
        out = _matmul_fp8_static_ptq(
            x_2d,
            self.input_scale,
            self.qweight,
            self.weight_scale,
            getattr(self, "bias", None),
            self.qweight_nk,
        )
        return out.view(*x.shape[:-1], -1).to(target_dtype)

    def forward_fp8(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp8_scheme == "w8a16":
            return self._forward_w8a16(x)
        if self.fp8_scheme == "w8a8_static":
            return self._forward_w8a8_static(x)
        raise ValueError(f"Unknown fp8_scheme: {self.fp8_scheme}")


class FP8MergedColumnParallelLinear(FP8LinearBase):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        fp8_scheme: str = "w8a16",
        tp_group=None,
        **kwargs,
    ) -> None:
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), fp8_scheme, tp_group=tp_group, **kwargs)
        self._init_quant_buffers(input_size, sum(output_sizes) // self.tp_size)
        self.bias = nn.Parameter(torch.empty(sum(output_sizes) // self.tp_size)) if bias else None

        self.qweight.weight_loader = self.qweight_loader
        self.weight_scale.weight_loader = self.weight_scale_loader
        if fp8_scheme == "w8a8_static":
            self.input_scale.weight_loader = self.input_scale_loader

    def qweight_loader(self, param, loaded_weight, shard_id):
        offset = sum(self.output_sizes[:shard_id]) // self.tp_size
        size = self.output_sizes[shard_id] // self.tp_size
        param.data.narrow(1, offset, size).copy_(loaded_weight.t().contiguous())

    def weight_scale_loader(self, param, loaded_weight, shard_id):
        offset = sum(self.output_sizes[:shard_id]) // self.tp_size
        size = self.output_sizes[shard_id] // self.tp_size
        _load_weight_scale(param, loaded_weight, offset, size)

    def input_scale_loader(self, param, loaded_weight, shard_id):
        flat = loaded_weight.view(-1)
        if flat.numel() == 1:
            param.data.copy_(flat.view(1))
        else:
            param.data.copy_(flat[:1])

    def forward(self, x):  # noqa: D102
        return self.forward_fp8(x)


class FP8QKVParallelLinear(FP8LinearBase):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        bias: bool = False,
        fp8_scheme: str = "w8a16",
        tp_group=None,
        **kwargs,
    ) -> None:
        self.head_size = head_size
        tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = total_num_kv_heads // tp_size
        out_f = (self.num_heads + 2 * self.num_kv_heads) * head_size
        super().__init__(hidden_size, out_f, fp8_scheme, tp_group=tp_group, **kwargs)
        self._init_quant_buffers(hidden_size, out_f)

        self.qweight.weight_loader = self.qweight_loader
        self.weight_scale.weight_loader = self.weight_scale_loader
        if fp8_scheme == "w8a8_static":
            self.input_scale.weight_loader = self.input_scale_loader

    def qweight_loader(self, param, loaded_weight, shard_id):
        if shard_id == "q":
            size, offset = self.num_heads * self.head_size, 0
        elif shard_id == "k":
            size, offset = self.num_kv_heads * self.head_size, self.num_heads * self.head_size
        else:
            size, offset = self.num_kv_heads * self.head_size, (self.num_heads + self.num_kv_heads) * self.head_size
        param.data.narrow(1, offset, size).copy_(loaded_weight.t().contiguous())

    def weight_scale_loader(self, param, loaded_weight, shard_id):
        if shard_id == "q":
            size, offset = self.num_heads * self.head_size, 0
        elif shard_id == "k":
            size, offset = self.num_kv_heads * self.head_size, self.num_heads * self.head_size
        else:
            size, offset = self.num_kv_heads * self.head_size, (self.num_heads + self.num_kv_heads) * self.head_size
        _load_weight_scale(param, loaded_weight, offset, size)

    def input_scale_loader(self, param, loaded_weight, shard_id):
        flat = loaded_weight.view(-1)
        if flat.numel() == 1:
            param.data.copy_(flat.view(1))
        else:
            param.data.copy_(flat[:1])

    def forward(self, x):  # noqa: D102
        return self.forward_fp8(x)


class FP8RowParallelLinear(FP8LinearBase):
    """Row-parallel: GEMM without bias in kernel; bias (full output dim) applied after optional all_reduce."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        reduce_results: bool = True,
        fp8_scheme: str = "w8a16",
        tp_group=None,
        **kwargs,
    ) -> None:
        super().__init__(input_size, output_size, fp8_scheme, tp_group=tp_group, **kwargs)
        self.reduce_results = reduce_results
        self._init_quant_buffers(divide(input_size, self.tp_size), output_size)
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None

        self.qweight.weight_loader = lambda p, w, *args: p.data.copy_(w.t().contiguous())
        self.weight_scale.weight_loader = lambda p, w, *args: _load_weight_scale(p, w, 0, p.shape[0])
        if fp8_scheme == "w8a8_static":
            self.input_scale.weight_loader = lambda p, w, *args: p.data.copy_(w.view(-1)[:1])

    def forward_fp8(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = x.dtype
        x_2d = x.view(-1, x.shape[-1])
        if self.fp8_scheme == "w8a16":
            if x_2d.dtype != torch.bfloat16:
                x_2d = x_2d.to(torch.bfloat16)
            w_fp8 = self.qweight.view(torch.float8_e4m3fn)
            out = launch_w8a16_gemm(x_2d, w_fp8, self.weight_scale, None)
            return out.view(*x.shape[:-1], -1).to(target_dtype)
        if self.fp8_scheme == "w8a8_static":
            self._ensure_qweight_nk()
            out = _matmul_fp8_static_ptq(
                x_2d, self.input_scale, self.qweight, self.weight_scale, None, self.qweight_nk
            )
            return out.view(*x.shape[:-1], -1).to(target_dtype)
        raise ValueError(f"Unknown fp8_scheme: {self.fp8_scheme}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward_fp8(x)
        if self.tp_size > 1 and self.reduce_results:
            dist.all_reduce(out, group=self.tp_group)
        if self.bias is not None:
            return out + self.bias
        return out
