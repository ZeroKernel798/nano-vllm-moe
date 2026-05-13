from __future__ import annotations

from torch import nn

from nanovllm.quantization.base_config import QuantizationConfig
from nanovllm.quantization.fp8 import Fp8Config, is_fp8_config


def get_quant_config(hf_config) -> QuantizationConfig | None:
    if is_fp8_config(hf_config):
        return Fp8Config(getattr(hf_config, "quantization_type"))
    return None


def process_weights_after_loading(model: nn.Module) -> None:
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            quant_method.process_weights_after_loading(module)
