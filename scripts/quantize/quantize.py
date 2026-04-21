"""Offline PTQ quantization script for nano-vllm-moe.

Supported schemes
-----------------
``fp8_w8a16``
    FP8 weight-only.  No calibration.  Loads with ``Qwen2ForCausalLMFP8``.

``int8_w8a16``
    INT8 weight-only.  No calibration.  Loads with ``Qwen2ForCausalLMInt8``.

``int8_w8a8_static``
    INT8 W8A8 with a calibrated per-layer activation scale.  Runs a short
    calibration pass using the HuggingFace model and WikiText-2 sentences
    (or a custom file) to estimate per-layer ``input_scale``.  Loads with
    ``Qwen2ForCausalLMInt8`` (static path selected automatically from config).

Output format (all schemes share the same key structure)
---------------------------------------------------------
  *.qweight        → [N, K] uint8 (FP8) or int8 (INT8)  — per linear weight
  *.weight_scale   → [N]    float32                      — per-output-channel
  *.input_scale    → [1]    float32  (w8a8_static only)  — per-layer activation
  All other tensors (biases, layernorm weights, embeddings) are copied as-is.

Usage examples
--------------
  python scripts/quantize/quantize.py \\
      --model-path /workspace/models/qwen/Qwen1.5-0.5B-Chat \\
      --output-path /workspace/models/qwen/Qwen1.5-0.5B-Chat-FP8 \\
      --scheme fp8_w8a16

  python scripts/quantize/quantize.py \\
      --model-path /workspace/models/qwen/Qwen1.5-0.5B-Chat \\
      --output-path /workspace/models/qwen/Qwen1.5-0.5B-Chat-INT8 \\
      --scheme int8_w8a16

  python scripts/quantize/quantize.py \\
      --model-path /workspace/models/qwen/Qwen1.5-0.5B-Chat \\
      --output-path /workspace/models/qwen/Qwen1.5-0.5B-Chat-INT8-Static \\
      --scheme int8_w8a8_static \\
      --calib-samples 64 \\
      --calib-seqlen 512
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from glob import glob
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

FP8_MAX = 448.0
INT8_MAX = 127.0

_QUANTIZE_SUFFIXES = (".weight",)
_SKIP_KEYS = ("embed_tokens", "lm_head")


def _should_quantize(weight_name: str, tensor: torch.Tensor) -> bool:
    if tensor.dim() != 2:
        return False
    if any(s in weight_name for s in _SKIP_KEYS):
        return False
    return weight_name.endswith(".weight")


def _quantize_fp8(w: torch.Tensor):
    w_f = w.float()
    scale = w_f.abs().max(dim=1).values.clamp(min=1e-8) / FP8_MAX
    w_fp8 = (w_f / scale.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return w_fp8.view(torch.uint8), scale.to(torch.float32)


def _quantize_int8(w: torch.Tensor):
    w_f = w.float()
    scale = w_f.abs().max(dim=1).values.clamp(min=1e-8) / INT8_MAX
    w_int8 = (w_f / scale.unsqueeze(1)).clamp(-128, 127).round().to(torch.int8)
    return w_int8, scale.to(torch.float32)


# ---------------------------------------------------------------------------
# Calibration for W8A8 static (INT8 only)
# ---------------------------------------------------------------------------

def _calibrate_input_scales(
    model_path: str,
    calib_samples: int,
    calib_seqlen: int,
) -> dict[str, float]:
    """Run calibration forward passes; return per-layer abs-max activation scales.

    Returns dict mapping HF module name (e.g. ``model.layers.0.self_attn.q_proj``)
    to ``float`` scale = abs_max / 127.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("transformers is required for calibration")

    print(f"  [calibrate] loading BF16 model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).cuda()
    model.eval()

    # ---- build calibration dataset ----
    # Try local file first, then a tiny built-in corpus (no network needed).
    _BUILTIN = [
        "The quick brown fox jumps over the lazy dog and runs away.",
        "Machine learning models require careful calibration for quantization.",
        "Natural language processing enables computers to understand human text.",
        "Deep neural networks consist of many layers of interconnected nodes.",
        "The weather today is sunny with a chance of rain in the afternoon.",
        "Scientists discovered a new species of bacteria in the deep ocean.",
        "Attention mechanisms allow models to focus on relevant parts of input.",
        "Transformer architectures have revolutionized modern AI applications.",
        "The stock market experienced significant volatility during the quarter.",
        "Programming languages provide abstractions to control computer hardware.",
        "Large language models are trained on vast amounts of text data.",
        "Quantum computing promises exponential speedup for certain algorithms.",
        "The history of mathematics spans thousands of years across cultures.",
        "Climate change poses significant challenges to global ecosystems.",
        "Philosophers have debated the nature of consciousness for centuries.",
    ]
    # Repeat corpus to reach calib_samples
    texts = (_BUILTIN * ((calib_samples // len(_BUILTIN)) + 1))[:calib_samples]
    print(f"  [calibrate] using built-in synthetic corpus ({len(texts)} samples).")

    print(f"  [calibrate] running {len(texts)} samples (seqlen≤{calib_seqlen}) ...")

    act_max: dict[str, float] = {}
    hooks = []

    def _make_hook(name: str):
        def _hook(module, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            val = float(x.detach().float().abs().max().item())
            act_max[name] = max(act_max.get(name, 0.0), val)
        return _hook

    # Register on all nn.Linear modules whose names end in _proj
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "_proj" in name:
            hooks.append(mod.register_forward_hook(_make_hook(name)))

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=calib_seqlen,
            ).to("cuda")
            model(**enc)

    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()

    # Convert to scales: scale = abs_max / 127
    scales = {k: max(v, 1e-8) / INT8_MAX for k, v in act_max.items()}
    print(f"  [calibrate] collected input_scale for {len(scales)} modules.")
    return scales


# ---------------------------------------------------------------------------
# Main quantization loop
# ---------------------------------------------------------------------------

def quantize(
    model_path: str,
    output_path: str,
    scheme: str,
    calib_samples: int = 64,
    calib_seqlen: int = 512,
) -> None:
    assert scheme in ("fp8_w8a16", "int8_w8a16", "int8_w8a8_static"), \
        f"Unknown scheme: {scheme}"

    # Select quantizer and quantization_type tag
    if scheme == "fp8_w8a16":
        quantize_fn = _quantize_fp8
        quant_tag = "fp8_w8a16"
    elif scheme == "int8_w8a16":
        quantize_fn = _quantize_int8
        quant_tag = "int8_w8a16"
    else:  # int8_w8a8_static
        quantize_fn = _quantize_int8
        quant_tag = "int8_w8a8_static"

    os.makedirs(output_path, exist_ok=True)

    # ---- optional calibration ----
    input_scales: dict[str, float] = {}
    if scheme == "int8_w8a8_static":
        input_scales = _calibrate_input_scales(model_path, calib_samples, calib_seqlen)

    # ---- quantize safetensors ----
    safetensor_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    assert safetensor_files, f"No safetensors found in {model_path}"

    print(f"\nQuantizing {model_path} → {output_path}  [scheme={scheme}]")
    n_quantized = n_total = 0

    for src_file in safetensor_files:
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(src_file, "pt", "cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                n_total += 1
                if _should_quantize(key, tensor):
                    qw, ws = quantize_fn(tensor)
                    base = key[:-len(".weight")]
                    out_tensors[base + ".qweight"] = qw
                    out_tensors[base + ".weight_scale"] = ws
                    n_quantized += 1
                    if n_quantized <= 3 or n_quantized % 50 == 0:
                        print(f"  [Q] {key}  {tuple(tensor.shape)} → "
                              f"qweight {tuple(qw.shape)}")
                    # Attach input_scale for static W8A8
                    if scheme == "int8_w8a8_static":
                        # Map checkpoint key like 'model.layers.0.self_attn.q_proj.weight'
                        # to module name 'model.layers.0.self_attn.q_proj'
                        module_name = base  # e.g. 'model.layers.0.self_attn.q_proj'
                        # Look up calibrated scale; default 1.0 if not found
                        cal_scale = input_scales.get(module_name, 1.0)
                        out_tensors[base + ".input_scale"] = torch.tensor(
                            [cal_scale], dtype=torch.float32
                        )
                else:
                    out_tensors[key] = tensor

        dst_file = os.path.join(output_path, os.path.basename(src_file))
        save_file(out_tensors, dst_file)
        print(f"  Saved {dst_file}")

    print(f"\nQuantized {n_quantized}/{n_total} tensors.")

    # ---- copy non-safetensors (tokenizer, generation_config, etc.) ----
    for fname in os.listdir(model_path):
        if fname.endswith(".safetensors") or fname.endswith(".index.json"):
            continue
        src = os.path.join(model_path, fname)
        dst = os.path.join(output_path, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    # ---- write modified config.json ----
    config_src = os.path.join(model_path, "config.json")
    config_dst = os.path.join(output_path, "config.json")
    with open(config_src) as fh:
        cfg = json.load(fh)
    cfg["quantization_type"] = quant_tag
    with open(config_dst, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"  config.json: quantization_type={quant_tag!r}")

    # ---- rebuild shard index if multi-shard ----
    index_src = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index_src):
        with open(index_src) as fh:
            index = json.load(fh)
        new_map: dict[str, str] = {}
        for key, shard in index["weight_map"].items():
            if key.endswith(".weight") and _should_quantize_key(key):
                base = key[:-len(".weight")]
                new_map[base + ".qweight"] = shard
                new_map[base + ".weight_scale"] = shard
                if scheme == "int8_w8a8_static":
                    new_map[base + ".input_scale"] = shard
            else:
                new_map[key] = shard
        index["weight_map"] = new_map
        with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as fh:
            json.dump(index, fh, indent=2)
        print("  Updated model.safetensors.index.json")

    print(f"\nDone. Load with:\n  LLM('{output_path}', ...)")


def _should_quantize_key(key: str) -> bool:
    if not key.endswith(".weight"):
        return False
    if any(s in key for s in _SKIP_KEYS):
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nano-vllm-moe offline PTQ quantizer")
    parser.add_argument("--model-path", required=True, help="Input BF16 checkpoint directory")
    parser.add_argument("--output-path", required=True, help="Output quantized checkpoint directory")
    parser.add_argument(
        "--scheme", default="int8_w8a16",
        choices=["fp8_w8a16", "int8_w8a16", "int8_w8a8_static"],
        help="Quantization scheme (default: int8_w8a16)",
    )
    parser.add_argument("--calib-samples", type=int, default=64,
                        help="Number of calibration samples for int8_w8a8_static")
    parser.add_argument("--calib-seqlen", type=int, default=512,
                        help="Max token length per calibration sample")
    args = parser.parse_args()
    quantize(args.model_path, args.output_path, args.scheme,
             args.calib_samples, args.calib_seqlen)
