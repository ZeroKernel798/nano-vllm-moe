from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import torch
from torch import nn

from nanovllm.quantization.kernels import launch_scaled_mm_w8a8, launch_w8a16_gemm, to_scaled_mm_weight
from nanovllm.quantization.base_config import QuantizationConfig, QuantizeMethodBase


_SUPPORTED_FP8_TYPES = {"fp8_w8a16", "fp8_w8a8_static"}
_FP8_BACKEND_LOGGED: set[str] = set()
_SUPPORTED_W8A8_ACT_QUANT_BACKENDS = {"torch", "triton"}


def get_w8a8_act_quant_backend() -> str:
    backend = os.environ.get("NANOVLLM_FP8_W8A8_ACT_QUANT", "torch").strip().lower()
    if backend not in _SUPPORTED_W8A8_ACT_QUANT_BACKENDS:
        raise ValueError(
            "NANOVLLM_FP8_W8A8_ACT_QUANT must be one of "
            f"{sorted(_SUPPORTED_W8A8_ACT_QUANT_BACKENDS)}, got {backend!r}"
        )
    return backend


def is_fp8_config(config: Any) -> bool:
    return getattr(config, "quantization_type", None) in _SUPPORTED_FP8_TYPES


@dataclass(frozen=True)
class Fp8Config(QuantizationConfig):
    quantization_type: str = "fp8_w8a16"

    def __post_init__(self) -> None:
        if self.quantization_type not in _SUPPORTED_FP8_TYPES:
            raise ValueError(f"Unsupported FP8 quantization_type={self.quantization_type!r}")

    def get_name(self) -> str:
        return "fp8"

    def get_quant_method(self, layer: nn.Module, prefix: str = "") -> QuantizeMethodBase | None:
        del layer, prefix
        return Fp8LinearMethod(self)


class Fp8LinearMethod(QuantizeMethodBase):
    def __init__(self, quant_config: Fp8Config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: nn.Module,
        *,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> None:
        del input_size, output_size, params_dtype
        output_size_per_partition = sum(output_partition_sizes)
        layer.register_buffer(
            "qweight",
            torch.empty((input_size_per_partition, output_size_per_partition), dtype=torch.uint8),
        )
        layer.register_buffer(
            "weight_scale",
            torch.empty((output_size_per_partition,), dtype=torch.float32),
        )
        if self.quant_config.quantization_type == "fp8_w8a8_static":
            layer.register_buffer("input_scale", torch.empty((), dtype=torch.float32))
            layer.input_scale.weight_loader = self._make_input_scale_loader(layer)
        layer.qweight.weight_loader = self._make_qweight_loader(layer)
        layer.weight_scale.weight_loader = self._make_weight_scale_loader(layer)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if not layer.qweight.is_cuda:
            return
        device_index = layer.qweight.device.index if layer.qweight.device.index is not None else 0
        capability = torch.cuda.get_device_capability(device_index)
        if self.quant_config.quantization_type == "fp8_w8a8_static":
            if not hasattr(torch, "_scaled_mm"):
                raise RuntimeError("fp8_w8a8_static requires torch._scaled_mm")
            qweight = layer.qweight.view(torch.float8_e4m3fn)
            weight_scale = layer.weight_scale.to(torch.float32)
            tensor_scale = weight_scale.max().clamp(min=1e-12)
            rescale = (weight_scale / tensor_scale).reshape(1, -1)
            scaled_weight = (qweight.to(torch.float32) * rescale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            layer.register_buffer("qweight_scaled_mm", to_scaled_mm_weight(scaled_weight))
            layer.register_buffer("weight_scale_scaled_mm", tensor_scale.reshape(()).to(torch.float32))
            layer._fp8_backend = "w8a8_scaled_mm"
            layer._fp8_w8a8_act_quant_backend = get_w8a8_act_quant_backend()
            log_key = f"w8a8_scaled_mm_{layer._fp8_w8a8_act_quant_backend}"
            if log_key not in _FP8_BACKEND_LOGGED:
                print(
                    "[nanovllm] FP8 W8A8 static using torch._scaled_mm "
                    f"tensor-wise weight scale and {layer._fp8_w8a8_act_quant_backend} activation quant"
                )
                _FP8_BACKEND_LOGGED.add(log_key)
            return
        if capability >= (9, 0):
            layer._fp8_backend = "w8a16_triton"
            if "triton" not in _FP8_BACKEND_LOGGED:
                print("[nanovllm] FP8 W8A16 using Triton weight-only kernel")
                _FP8_BACKEND_LOGGED.add("triton")
            return
        layer._fp8_backend = "w8a16_dequant_matmul"
        if "dequant_matmul" not in _FP8_BACKEND_LOGGED:
            print("[nanovllm] FP8 W8A16 using per-forward BF16 dequant matmul")
            _FP8_BACKEND_LOGGED.add("dequant_matmul")

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        target_dtype = x.dtype
        x_2d = x.view(-1, x.shape[-1])
        if x_2d.dtype != torch.bfloat16:
            x_2d = x_2d.to(torch.bfloat16)
        w_fp8 = layer.qweight.view(torch.float8_e4m3fn)
        backend = getattr(layer, "_fp8_backend", None)
        if x.is_cuda and backend == "w8a8_scaled_mm":
            out = launch_scaled_mm_w8a8(
                x_2d,
                layer.qweight_scaled_mm,
                layer.input_scale,
                layer.weight_scale_scaled_mm,
                bias,
                getattr(layer, "_fp8_w8a8_act_quant_backend", "torch"),
            )
            return out.view(*x.shape[:-1], -1).to(target_dtype)
        if x.is_cuda and backend == "w8a16_triton":
            out = launch_w8a16_gemm(x_2d, w_fp8, layer.weight_scale, bias)
            return out.view(*x.shape[:-1], -1).to(target_dtype)
        if backend == "w8a16_dequant_matmul" or not x.is_cuda:
            w_bf16 = w_fp8.to(torch.bfloat16) * layer.weight_scale.to(torch.bfloat16).unsqueeze(0)
            out = torch.mm(x_2d, w_bf16)
            if bias is not None:
                out = out + bias
            return out.view(*x.shape[:-1], -1).to(target_dtype)
        raise RuntimeError(f"FP8 backend is not initialized: {backend!r}")

    def _make_qweight_loader(self, layer: nn.Module):
        def loader(param: torch.Tensor, loaded_weight: torch.Tensor, shard_id=None) -> None:
            loaded_weight = self._select_loaded_weight(layer, loaded_weight, shard_id)
            offset, size = self._local_output_offset_size(layer, shard_id, loaded_weight.shape[0])
            param.data.narrow(1, offset, size).copy_(loaded_weight.t().contiguous())

        return loader

    def _make_weight_scale_loader(self, layer: nn.Module):
        def loader(param: torch.Tensor, loaded_weight: torch.Tensor, shard_id=None) -> None:
            loaded_weight = self._select_loaded_scale(layer, loaded_weight, shard_id)
            offset, size = self._local_output_offset_size(layer, shard_id, loaded_weight.numel())
            flat = loaded_weight.view(-1).to(torch.float32)
            if flat.numel() == 1:
                param.data.narrow(0, offset, size).fill_(flat.item())
            else:
                param.data.narrow(0, offset, size).copy_(flat)

        return loader

    def _make_input_scale_loader(self, layer: nn.Module):
        del layer

        def loader(param: torch.Tensor, loaded_weight: torch.Tensor, shard_id=None) -> None:
            del shard_id
            param.data.copy_(loaded_weight.reshape(()).to(torch.float32))

        return loader

    @staticmethod
    def _local_output_offset_size(layer: nn.Module, shard_id, loaded_output_size: int) -> tuple[int, int]:
        if hasattr(layer, "get_quant_output_offset_size") and shard_id is not None:
            return layer.get_quant_output_offset_size(shard_id)
        return 0, loaded_output_size

    @staticmethod
    def _select_loaded_weight(layer: nn.Module, loaded_weight: torch.Tensor, shard_id) -> torch.Tensor:
        if getattr(layer, "tp_dim", None) == 0:
            shard_size = loaded_weight.shape[0] // layer.tp_size
            start = layer.tp_rank * shard_size
            return loaded_weight.narrow(0, start, shard_size)
        if getattr(layer, "tp_dim", None) == 1:
            shard_size = loaded_weight.shape[1] // layer.tp_size
            start = layer.tp_rank * shard_size
            return loaded_weight.narrow(1, start, shard_size)
        return loaded_weight

    @staticmethod
    def _select_loaded_scale(layer: nn.Module, loaded_weight: torch.Tensor, shard_id) -> torch.Tensor:
        del shard_id
        if getattr(layer, "tp_dim", None) == 0 and loaded_weight.numel() > 1:
            shard_size = loaded_weight.numel() // layer.tp_size
            start = layer.tp_rank * shard_size
            return loaded_weight.view(-1).narrow(0, start, shard_size)
        return loaded_weight
