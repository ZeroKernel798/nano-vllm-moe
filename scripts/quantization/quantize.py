"""Offline FP8 exporters for the vLLM-style quantization interface."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

FP8_MAX = 448.0
_SKIP_KEYS = ("embed_tokens", "lm_head")
_SUPPORTED_SCHEMES = ("fp8_w8a16", "fp8_w8a8_static")


@dataclass
class ExportStats:
    total_tensors: int = 0
    quantized_tensors: int = 0
    skipped_tensors: int = 0


@dataclass
class CalibrationConfig:
    dataset: str
    dataset_config: str
    split: str
    text_column: str
    cache_dir: str
    text_file: str
    samples: int
    max_length: int
    batch_size: int
    dtype: str
    device: str
    trust_remote_code: bool
    skip: bool


def should_quantize(weight_name: str, tensor: torch.Tensor) -> bool:
    return tensor.dim() == 2 and weight_name.endswith(".weight") and not any(key in weight_name for key in _SKIP_KEYS)


def quantize_fp8_per_output_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight_fp32 = weight.float()
    scale = weight_fp32.abs().max(dim=1).values.clamp(min=1e-8) / FP8_MAX
    qweight = (weight_fp32 / scale.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return qweight.view(torch.uint8), scale.to(torch.float32)


def default_input_scale() -> torch.Tensor:
    return torch.tensor(1.0, dtype=torch.float32)


def copy_metadata(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    for item in os.listdir(src):
        if item.endswith(".safetensors") or item in {"model.safetensors.index.json"}:
            continue
        src_path = os.path.join(src, item)
        dst_path = os.path.join(dst, item)
        if os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)


def update_config(output_path: str, scheme: str, manifest: dict[str, Any]) -> None:
    config_path = Path(output_path) / "config.json"
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    config["quantization_type"] = scheme
    config["quantization_config"] = {
        "format": scheme,
        "weight_dtype": "float8_e4m3fn",
        "weight_scale": "per_output_channel",
        "activation_dtype": "bfloat16" if scheme == "fp8_w8a16" else "float8_e4m3fn",
        "activation_scale": None if scheme == "fp8_w8a16" else "static_per_linear",
        "runtime_status": "implemented" if scheme == "fp8_w8a16" else "export_contract_only",
        "manifest_file": "quantization_manifest.json",
    }
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
        file.write("\n")
    with (Path(output_path) / "quantization_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")


def load_calibration_texts(config: CalibrationConfig) -> list[str]:
    if config.text_file:
        texts = [line.strip() for line in Path(config.text_file).read_text(encoding="utf-8").splitlines()]
        return [text for text in texts if text][: config.samples]
    from datasets import load_dataset

    load_kwargs = {}
    if config.cache_dir:
        load_kwargs["cache_dir"] = str(Path(config.cache_dir).expanduser())
    raw = load_dataset(config.dataset, config.dataset_config, split=config.split, **load_kwargs)
    texts: list[str] = []
    for text in raw[config.text_column]:
        if text and text.strip():
            texts.append(text.strip())
        if len(texts) >= config.samples:
            break
    return texts


def collect_input_scales(model_path: str, bases: list[str], config: CalibrationConfig) -> dict[str, torch.Tensor]:
    if config.skip:
        return {base: default_input_scale() for base in bases}
    print(
        "Calibrating fp8_w8a8_static input scales "
        f"[dataset={config.dataset}/{config.dataset_config}, split={config.split}, samples={config.samples}, max_length={config.max_length}]"
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, config.dtype)
    use_cuda = torch.cuda.is_available() and config.device.startswith("cuda")
    device = config.device if use_cuda else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=config.trust_remote_code)
    model = model.to(device)
    model.eval()

    target_bases = set(bases)
    max_abs: dict[str, float] = {base: 0.0 for base in bases}
    handles = []

    def make_hook(base: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not inputs:
                return
            activation = inputs[0]
            if not torch.is_tensor(activation):
                return
            value = activation.detach().abs().float().max().item()
            if value > max_abs[base]:
                max_abs[base] = value

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in target_bases:
            handles.append(module.register_forward_pre_hook(make_hook(name)))

    missing_hooks = sorted(target_bases - {name for name, module in model.named_modules() if isinstance(module, nn.Linear) and name in target_bases})
    if missing_hooks:
        print(f"  [WARN] Missing calibration hooks for {len(missing_hooks)} linear modules; examples={missing_hooks[:5]}")

    texts = load_calibration_texts(config)
    if not texts:
        raise ValueError("No calibration texts loaded")
    with torch.inference_mode():
        for start in range(0, len(texts), config.batch_size):
            batch = texts[start : start + config.batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            model(**encoded)
            if start == 0 or (start + config.batch_size) % 32 == 0:
                print(f"  [calib] processed={min(start + config.batch_size, len(texts))}/{len(texts)}")

    for handle in handles:
        handle.remove()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scales = {
        base: torch.tensor(max(max_abs[base], 1e-8) / FP8_MAX, dtype=torch.float32)
        for base in bases
    }
    values = [scale.item() for scale in scales.values()]
    print(
        "  [calib] input_scale stats: "
        f"count={len(values)}, min={min(values):.6e}, max={max(values):.6e}, mean={sum(values) / len(values):.6e}"
    )
    return scales


def quantize(model_path: str, output_path: str, scheme: str, calibration: CalibrationConfig) -> None:
    if scheme not in _SUPPORTED_SCHEMES:
        raise ValueError(f"Unsupported scheme={scheme!r}")
    print(f"Quantizing {model_path} → {output_path} [scheme={scheme}]")
    copy_metadata(model_path, output_path)
    output_tensors: dict[str, torch.Tensor] = {}
    stats = ExportStats()
    quantized_names: list[str] = []
    skipped_names: list[str] = []
    quantized_bases: list[str] = []

    for file in sorted(glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                stats.total_tensors += 1
                if should_quantize(key, tensor):
                    base = key[: -len(".weight")]
                    qweight, weight_scale = quantize_fp8_per_output_channel(tensor)
                    output_tensors[f"{base}.qweight"] = qweight
                    output_tensors[f"{base}.weight_scale"] = weight_scale
                    quantized_bases.append(base)
                    stats.quantized_tensors += 1
                    quantized_names.append(key)
                    if stats.quantized_tensors <= 3 or stats.quantized_tensors % 50 == 0:
                        print(f"  [Q] {key} {tuple(tensor.shape)} -> qweight {tuple(qweight.shape)}")
                else:
                    output_tensors[key] = tensor
                    stats.skipped_tensors += 1
                    skipped_names.append(key)

    input_scales: dict[str, torch.Tensor] = {}
    if scheme == "fp8_w8a8_static":
        input_scales = collect_input_scales(model_path, quantized_bases, calibration)
        for base in quantized_bases:
            output_tensors[f"{base}.input_scale"] = input_scales.get(base, default_input_scale())

    output_file = os.path.join(output_path, "model.safetensors")
    save_file(output_tensors, output_file)
    manifest = {
        "scheme": scheme,
        "source_model_path": model_path,
        "output_path": output_path,
        "total_tensors": stats.total_tensors,
        "quantized_tensors": stats.quantized_tensors,
        "skipped_tensors": stats.skipped_tensors,
        "quantized_examples": quantized_names[:20],
        "skipped_examples": skipped_names[:20],
        "scale_tensors": ["weight_scale"] if scheme == "fp8_w8a16" else ["weight_scale", "input_scale"],
        "calibration": None
        if scheme == "fp8_w8a16"
        else {
            "granularity": "per_linear_static_per_tensor",
            "dataset": calibration.dataset,
            "dataset_config": calibration.dataset_config,
            "split": calibration.split,
            "text_column": calibration.text_column,
            "cache_dir": calibration.cache_dir,
            "text_file": calibration.text_file,
            "samples": calibration.samples,
            "max_length": calibration.max_length,
            "batch_size": calibration.batch_size,
            "dtype": calibration.dtype,
            "device": calibration.device,
            "skip": calibration.skip,
            "input_scale_min": min((scale.item() for scale in input_scales.values()), default=None),
            "input_scale_max": max((scale.item() for scale in input_scales.values()), default=None),
        },
        "notes": "fp8_w8a8_static exports calibrated per-linear static input_scale; runtime kernel is not implemented yet."
        if scheme == "fp8_w8a8_static"
        else "fp8_w8a16 runtime is supported by nano-vllm.",
    }
    update_config(output_path, scheme, manifest)
    print(f"Saved {output_file}")
    print(f"Quantized {stats.quantized_tensors}/{stats.total_tensors} tensors")
    print(f"config.json: quantization_type={scheme!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="nano-vllm-moe FP8 exporter")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--scheme", default="fp8_w8a16", choices=_SUPPORTED_SCHEMES)
    parser.add_argument("--calib-dataset", default="wikitext")
    parser.add_argument("--calib-dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--calib-split", default="validation")
    parser.add_argument("--calib-text-column", default="text")
    parser.add_argument("--calib-cache-dir", default="")
    parser.add_argument("--calib-text-file", default="")
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--calib-max-length", type=int, default=512)
    parser.add_argument("--calib-batch-size", type=int, default=4)
    parser.add_argument("--calib-dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--calib-device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true", help="Write input_scale=1.0 placeholders for fp8_w8a8_static")
    args = parser.parse_args()
    calibration = CalibrationConfig(
        dataset=args.calib_dataset,
        dataset_config=args.calib_dataset_config,
        split=args.calib_split,
        text_column=args.calib_text_column,
        cache_dir=args.calib_cache_dir,
        text_file=args.calib_text_file,
        samples=args.calib_samples,
        max_length=args.calib_max_length,
        batch_size=args.calib_batch_size,
        dtype=args.calib_dtype,
        device=args.calib_device,
        trust_remote_code=args.trust_remote_code,
        skip=args.skip_calibration,
    )
    quantize(args.model_path, args.output_path, args.scheme, calibration)


if __name__ == "__main__":
    main()
