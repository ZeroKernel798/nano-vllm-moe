"""WikiText-2 perplexity for nano-vllm INT8 exports (W8A16 / W8A8 static).

Loads ``*.safetensors`` with ``qweight`` + ``weight_scale`` (+ optional
``input_scale``), reconstructs ``Linear.weight`` in BF16 by dequantizing,
then runs the standard sliding-window perplexity computation.

Interpretation
--------------
- **W8A16**: Weight-only error vs BF16.  This PPL is a faithful proxy
  because activations stay in BF16 in both eval and real inference.

- **W8A8 static**: Real inference also quantizes activations; this script
  evaluates with **BF16 activations** (dequant proxy), so **PPL is slightly
  optimistic** for the activation quantization component.  Still useful to
  compare weight quantization quality vs W8A16 / BF16.

Example::

    python scripts/eval/eval_ppl_nano_int8.py \\
        --model-path /workspace/models/qwen/Qwen1.5-0.5B-Chat-INT8 \\
        --max-length 2048 --stride 512 --split test

    python scripts/eval/eval_ppl_nano_int8.py \\
        --model-path /workspace/models/qwen/Qwen1.5-0.5B-Chat-INT8-Static \\
        --max-length 2048 --stride 512 --split test
"""

from __future__ import annotations

import argparse
import glob
import os

import torch
from safetensors.torch import load_file
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _merge_safetensors(model_dir: str) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No *.safetensors under {model_dir!r}")
    for f in files:
        out.update(load_file(f))
    return out


def dequant_int8_state_dict(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Replace ``*.qweight`` + ``*.weight_scale`` with ``*.weight`` (BF16)."""
    out: dict[str, torch.Tensor] = {}
    qkeys = [k for k in raw if k.endswith(".qweight")]

    for qk in qkeys:
        base = qk[: -len(".qweight")]
        sk = base + ".weight_scale"
        if sk not in raw:
            raise KeyError(f"Missing {sk} for {qk}")
        q = raw[qk].float()                    # [N, K] int8 → float
        ws = raw[sk].float()                   # [N]
        if ws.numel() == 1:
            w_fp = q * ws.reshape(())
        else:
            w_fp = q * ws.view(-1, 1)
        out[base + ".weight"] = w_fp.to(torch.bfloat16)

    # Copy through all non-quantization tensors
    for k, v in raw.items():
        if k.endswith(".qweight") or k.endswith(".weight_scale") or k.endswith(".input_scale"):
            continue
        if k not in out:
            out[k] = v

    return out


def main() -> None:
    p = argparse.ArgumentParser(description="WikiText PPL for nano INT8 export (dequant → HF BF16)")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--split", type=str, default="test", choices=("test", "validation"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    path = os.path.expanduser(args.model_path)
    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device = args.device if use_cuda else "cpu"

    print(f"Loading safetensors from {path} ...")
    raw = _merge_safetensors(path)
    sd = dequant_int8_state_dict(raw)

    print("Building HF model from config + dequantized weights ...")
    config = AutoConfig.from_pretrained(path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.bfloat16, trust_remote_code=args.trust_remote_code
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if getattr(model.config, "tie_word_embeddings", False):
        try:
            model.lm_head.weight = model.model.embed_tokens.weight
        except Exception:
            pass
    if missing:
        print(f"warning: missing keys ({len(missing)}): {missing[:6]}{'...' if len(missing) > 6 else ''}")

    if use_cuda:
        model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=args.trust_remote_code)

    max_pos = getattr(model.config, "max_position_embeddings", 4096)
    max_length = min(args.max_length, max_pos)
    stride = min(args.stride, max_length)

    print(f"Loading wikitext-2-raw-v1 split={args.split!r} ...")
    try:
        from datasets import load_dataset
        raw_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=args.split)
        text = "\n\n".join(t for t in raw_ds["text"] if t)
    except Exception as exc:
        raise SystemExit(
            f"Cannot load wikitext dataset: {exc}\n"
            "Make sure the 'datasets' package is installed and network is available, "
            "or use --split with a local copy."
        ) from exc
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    print(f"Tokenized length: {seq_len}, chunk={max_length}, stride={stride}")

    input_ids = encodings.input_ids.to(device)
    nlls: list[torch.Tensor] = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride), desc="windows"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        if trg_len <= 0:
            break
        chunk = input_ids[:, begin_loc:end_loc]
        target_ids = chunk.clone()
        target_ids[:, :-trg_len] = -100
        with torch.inference_mode():
            out = model(chunk, labels=target_ids)
            nlls.append(out.loss * trg_len)
        prev_end_loc = end_loc
        if end_loc >= seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / prev_end_loc).item()
    print(f"\nWikiText-2-raw-v1 ({args.split})  perplexity ≈ {ppl:.4f}  (INT8 → dequant BF16 eval)")


if __name__ == "__main__":
    main()
