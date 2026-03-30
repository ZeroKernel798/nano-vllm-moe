"""
Static FP8 (E4M3) export for nano-vllm W8A8 inference.

Uses NVIDIA ModelOpt calibration on wikitext-2-raw-v1, then exports
``qweight`` / ``weight_scale`` / ``input_scale`` for transformer Linears.
``embed_tokens`` and ``lm_head`` stay as BF16 ``.weight`` (nano-vllm does not FP8-load those modules).

Requires: transformers, datasets, safetensors, nvidia-modelopt, torch with CUDA.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import tqdm
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.quantization as mtq

FP8_MAX = 448.0


def _is_quantized_linear(m: torch.nn.Module) -> bool:
    return hasattr(m, "weight_quantizer") and hasattr(m, "input_quantizer") and hasattr(m, "weight")


def _export_fp8_for_nano_module(name: str) -> bool:
    """nano-vllm keeps ``embed_tokens`` / ``lm_head`` as BF16 ``nn.Parameter`` + ``F.linear``.

    If we export ``qweight`` only, the checkpoint has no ``*.weight`` for these modules and
    ``load_model`` never fills ``lm_head`` / ``embed_tokens`` (random logits / embeddings).
    """
    return "lm_head" not in name and "embed_tokens" not in name


def _quantizer_amax_tensor(tq) -> torch.Tensor | None:
    if hasattr(tq, "_amax") and getattr(tq, "_amax", None) is not None:
        return tq._amax.float()
    a = getattr(tq, "amax", None)
    if a is not None:
        return a.float()
    return None


def _export_quantized_linear_tensors(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """HF Linear weight [out, in] -> ``qweight`` as uint8 **same layout as ``.weight``** ([out, in]).

    ``load_model`` / ``FP8*ParallelLinear`` apply ``.t()`` when copying into the internal
    ``[in, out]`` buffer; export must not transpose (see ``quant_w8a16_fp8.py``).
    """
    w = module.weight.data.float()
    wq = module.weight_quantizer
    iq = module.input_quantizer
    w_amax = _quantizer_amax_tensor(wq)
    if w_amax is None:
        w_amax = w.abs().max().detach().float().clamp(min=1e-12)
    else:
        w_amax = w_amax.clamp(min=1e-12)
    i_amax = _quantizer_amax_tensor(iq)
    if i_amax is None:
        i_amax = torch.tensor(1.0, dtype=torch.float32)
    else:
        i_amax = i_amax.clamp(min=1e-12)

    if w_amax.numel() == 1:
        w_scale = (w_amax / FP8_MAX).item()
        q_hf = (w / w_scale).to(torch.float8_e4m3fn)
        weight_scale = torch.full((w.shape[0],), w_scale, dtype=torch.float32)
    else:
        s = (w_amax.reshape(-1) / FP8_MAX).clamp(min=1e-12)
        if s.numel() == 1:
            s = s.expand(w.shape[0])
        else:
            s = s[: w.shape[0]]
        q_hf = (w / s.unsqueeze(1)).to(torch.float8_e4m3fn)
        weight_scale = s.to(torch.float32)

    input_scale = (i_amax / FP8_MAX).reshape(-1)[:1].to(torch.float32)

    return {
        "qweight": q_hf.contiguous().view(torch.uint8),
        "weight_scale": weight_scale,
        "input_scale": input_scale,
    }


def _fuse_packed_input_scales(
    exported: dict[str, torch.Tensor], num_hidden_layers: int
) -> None:
    """Align ``input_scale`` for HF shards that map to one nano-vllm buffer.

    ``LlamaForCausalLM`` has separate ``q_proj`` / ``k_proj`` / ``v_proj`` Linear modules,
    each with its own calibrated ``input_quantizer``, but nano stores a **single**
    ``qkv_proj.input_scale``. The loader overwrites that buffer once per shard; without
    fusion, only the last shard's scale survives and Q/K (or gate vs up) use wrong scales.

    We take the **max** of the per-shard scales (same ``x``, conservative for E4M3 range).
    """
    for i in range(num_hidden_layers):
        qk = f"model.layers.{i}.self_attn.q_proj.input_scale"
        kk = f"model.layers.{i}.self_attn.k_proj.input_scale"
        vk = f"model.layers.{i}.self_attn.v_proj.input_scale"
        if qk in exported and kk in exported and vk in exported:
            s = torch.max(
                torch.stack(
                    [
                        exported[qk].reshape(-1),
                        exported[kk].reshape(-1),
                        exported[vk].reshape(-1),
                    ]
                )
            )
            t = s.view(1).cpu()
            exported[qk] = t.clone()
            exported[kk] = t.clone()
            exported[vk] = t.clone()
        gk = f"model.layers.{i}.mlp.gate_proj.input_scale"
        uk = f"model.layers.{i}.mlp.up_proj.input_scale"
        if gk in exported and uk in exported:
            s = torch.max(exported[gk].reshape(-1), exported[uk].reshape(-1))
            t = s.view(1).cpu()
            exported[gk] = t.clone()
            exported[uk] = t.clone()


def build_nano_fp8_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Replace QuantLinear ``.weight`` with ``qweight`` + scales (except embed / LM head)."""
    exported: dict[str, torch.Tensor] = {}
    quant_prefixes: set[str] = set()

    for name, mod in model.named_modules():
        if not _is_quantized_linear(mod):
            continue
        if not _export_fp8_for_nano_module(name):
            continue
        tensors = _export_quantized_linear_tensors(mod)
        prefix = name
        quant_prefixes.add(prefix)
        exported[f"{prefix}.qweight"] = tensors["qweight"].cpu()
        exported[f"{prefix}.weight_scale"] = tensors["weight_scale"].cpu()
        exported[f"{prefix}.input_scale"] = tensors["input_scale"].cpu()

    for key, tensor in model.state_dict().items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if key in exported:
            continue
        parts = key.rsplit(".", 1)
        if len(parts) == 2:
            base, leaf = parts
            if leaf == "weight" and base in quant_prefixes:
                continue
        t = tensor.cpu()
        if ("embed_tokens" in key or "lm_head" in key) and key.endswith(".weight"):
            t = t.to(torch.bfloat16)
        exported[key] = t

    nl = getattr(model.config, "num_hidden_layers", None)
    if isinstance(nl, int) and nl > 0:
        _fuse_packed_input_scales(exported, nl)

    return exported


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ModelOpt FP8 static export for nano-vllm W8A8")
    p.add_argument(
        "--model-path",
        type=str,
        default="/root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
    )
    p.add_argument(
        "--output-path",
        type=str,
        default="/root/autodl-tmp/models/Llama-3.1-ModelOpt-FP8-W8A8",
    )
    p.add_argument("--calib-samples", type=int, default=128, help="Number of wikitext lines for calibration")
    p.add_argument("--max-length", type=int, default=512)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    print("Loading wikitext-2-raw-v1 for calibration...")
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    samples: list[str] = []
    for line in traindata["text"]:
        if len(line.strip()) > 50:
            samples.append(line.strip())
        if len(samples) >= args.calib_samples:
            break

    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    def calibration_loop(m: torch.nn.Module) -> None:
        m.eval()
        dev = next(m.parameters()).device
        with torch.no_grad():
            for text in tqdm.tqdm(samples, desc="Calibrating"):
                batch = tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=args.max_length,
                    truncation=True,
                )
                inputs = {k: v.to(dev) for k, v in batch.items()}
                m(**inputs)

    print("Applying ModelOpt FP8_DEFAULT_CFG ...")
    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, calibration_loop)

    print("Building nano-vllm FP8 state dict (layers FP8; embed + lm_head BF16 weights) ...")
    fp8_state = build_nano_fp8_state_dict(model)

    out_file = os.path.join(args.output_path, "model.safetensors")
    save_file(fp8_state, out_file)
    tokenizer.save_pretrained(args.output_path)

    quant_cfg = {
        "quant_method": "fp8",
        "fp8_scheme": "w8a8_static",
        "activation_scheme": "static",
        "producer": "nvidia-modelopt",
    }
    model.config.quantization_config = quant_cfg
    model.config.save_pretrained(args.output_path)

    cfg_path = os.path.join(args.output_path, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["quantization_config"] = quant_cfg
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    print(f"Done. Wrote {out_file} and config with fp8_scheme=w8a8_static.")


if __name__ == "__main__":
    main()
