from __future__ import annotations

import json
import shutil
import tempfile
from glob import glob
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from common import read_model_config


def copy_hf_metadata(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    skip = {"model.safetensors", "model.safetensors.index.json"}
    for item in src.iterdir():
        if item.name in skip or item.suffix == ".safetensors":
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def write_dequant_config(src: Path, dst: Path) -> None:
    config = read_model_config(src)
    config.pop("quantization_type", None)
    config.pop("quantization_config", None)
    with (dst / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
        file.write("\n")


def dequantize_fp8_w8a16(src: Path, dst: Path, dtype: torch.dtype) -> dict[str, int]:
    copy_hf_metadata(src, dst)
    write_dequant_config(src, dst)
    tensors: dict[str, torch.Tensor] = {}
    qweights: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    copied = 0
    dequantized = 0
    for file in sorted(glob(str(src / "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                if key.endswith(".qweight"):
                    qweights[key[: -len(".qweight")]] = tensor
                elif key.endswith(".weight_scale"):
                    scales[key[: -len(".weight_scale")]] = tensor
                elif key.endswith(".input_scale"):
                    continue
                else:
                    tensors[key] = tensor
                    copied += 1
    for base, qweight in qweights.items():
        if base not in scales:
            raise ValueError(f"Missing weight_scale for {base}.qweight")
        scale = scales[base].to(torch.float32)
        weight = qweight.view(torch.float8_e4m3fn).to(dtype) * scale.to(dtype).unsqueeze(1)
        tensors[f"{base}.weight"] = weight.contiguous()
        dequantized += 1
    save_file(tensors, str(dst / "model.safetensors"))
    return {"dequantized_tensors": dequantized, "copied_tensors": copied}


def prepare_eval_model(
    model_path: str,
    quant_format: str,
    dtype: torch.dtype,
) -> tuple[str, tempfile.TemporaryDirectory[str] | None, dict[str, Any]]:
    src = Path(model_path).expanduser()
    config = read_model_config(src)
    detected = config.get("quantization_type")
    selected = detected if quant_format == "auto" else quant_format
    if selected in (None, "", "none"):
        return str(src), None, {"quant_format": selected or "none", "proxy": None}
    if selected not in {"fp8_w8a16", "fp8_w8a8_static"}:
        raise ValueError(f"Unsupported quant_format={selected!r}; add an adapter for this format")
    tempdir = tempfile.TemporaryDirectory(prefix="nanovllm_ppl_dequant_")
    stats = dequantize_fp8_w8a16(src, Path(tempdir.name), dtype)
    stats.update({"quant_format": selected, "proxy": f"{selected}_to_bf16_hf"})
    return tempdir.name, tempdir, stats
