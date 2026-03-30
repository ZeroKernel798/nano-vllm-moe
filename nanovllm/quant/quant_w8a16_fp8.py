"""
Export Llama (dense) linear layers to FP8 E4M3 weights + tensor-wise scale (W8A16 for nano-vllm).

W8A16 here: FP8 weights, BF16 activations at inference (``fp8_scheme: "w8a16"``).
Each targeted ``*.weight`` is replaced by ``*.qweight`` (uint8 view of fp8) and
``*.weight_scale`` (scalar broadcast to per-output channels in the loader).

Does **not** write ``input_scale`` (that is W8A8 static only).

Layer filter: any ``state_dict`` key with ``"weight"`` and a substring in
``("proj", "fc", "gate_up")`` — covers ``q/k/v/o_proj``, ``gate_proj`` / ``up_proj``,
fused ``gate_up_proj``, ``down_proj``, etc. Norm / embed / ``lm_head`` stay BF16.

Requires: transformers, safetensors, torch with CUDA (recommended for load speed).
"""

from __future__ import annotations

import argparse
import os

import torch
import tqdm
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

FP8_MAX = 448.0

_KEYWORDS = ("proj", "fc", "gate_up")


def _should_quantize_weight(name: str) -> bool:
    return "weight" in name and any(k in name for k in _KEYWORDS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pure per-tensor-max FP8 export (W8A16) for nano-vllm")
    p.add_argument(
        "--model-path",
        type=str,
        default="/root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
    )
    p.add_argument(
        "--output-path",
        type=str,
        default="/root/autodl-tmp/models/Llama-3.1-Pure-FP8-W8A16",
    )
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    print("Loading BF16 model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )

    fp8_state_dict: dict[str, torch.Tensor] = {}
    model_state = model.state_dict()

    for name, param in tqdm.tqdm(model_state.items(), desc="FP8 (W8A16) export"):
        if _should_quantize_weight(name):
            weight_max = param.abs().max().to(torch.float32)
            scale = (weight_max / FP8_MAX).clamp(min=1e-12)
            qweight = (param.to(torch.float32) / scale).to(torch.float8_e4m3fn)
            base_name = name.replace(".weight", "")
            fp8_state_dict[f"{base_name}.qweight"] = qweight.view(torch.uint8)
            fp8_state_dict[f"{base_name}.weight_scale"] = scale.reshape(1).to(torch.float32)
        else:
            fp8_state_dict[name] = param

    out_file = os.path.join(args.output_path, "model.safetensors")
    save_file(fp8_state_dict, out_file)
    tokenizer.save_pretrained(args.output_path)

    model.config.quantization_config = {"quant_method": "fp8", "fp8_scheme": "w8a16"}
    model.config.save_pretrained(args.output_path)

    print(f"Done. Wrote {out_file}")
    print('config.json: quantization_config = {"quant_method": "fp8", "fp8_scheme": "w8a16"}')


if __name__ == "__main__":
    main()
