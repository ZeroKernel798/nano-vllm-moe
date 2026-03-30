"""
WikiText-2 (wikitext-2-raw-v1) sliding-window perplexity for HuggingFace causal LMs.

Use cases
---------
- **BF16 / FP16 baseline**: point ``--model-path`` at a standard HF checkpoint
  (``torch_dtype`` matches weights).

- **nano-vllm FP8 exports** (``qweight`` / ``weight_scale``): use ``eval_ppl_nano_fp8.py`` in this
  repo (dequantizes weights to BF16, then same WikiText loop). See that script for W8A8 vs W8A16
  interpretation.

W8A16 vs W8A8 (this repo)
-------------------------
- **W8A16** (``fp8_scheme: "w8a16"``): FP8 weights, BF16 activations — export e.g. with
  ``nanovllm/quant/quant_w8a16_fp8.py`` (sets ``quantization_config`` in ``config.json``).
- **W8A8 static** (``fp8_scheme: "w8a8_static"``): FP8 weights + static FP8 activations —
  ``nanovllm/quant/quant_w8a8_fp8.py``.

If you meant **A8W16** (activations 8-bit, weights BF16), that is **not** the same as W8A16
and is not implemented as an ``fp8_scheme`` here.

Example
-------
::

    python eval_ppl_wikitext.py --model-path /path/to/Llama-3.1-8B-Instruct \\
        --dtype bfloat16 --max-length 2048 --stride 512

"""

from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="WikiText-2 raw perplexity (HF causal LM)")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    p.add_argument("--max-length", type=int, default=2048, help="Chunk length (<= model max pos)")
    p.add_argument("--stride", type=int, default=512, help="Window stride (overlap = max_length - stride)")
    p.add_argument("--split", type=str, default="test", choices=("test", "validation"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    path = os.path.expanduser(args.model_path)
    dtype = getattr(torch, args.dtype)
    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device = args.device if use_cuda else "cpu"

    print(f"Loading tokenizer/model from: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=args.trust_remote_code)
    load_kw = dict(torch_dtype=dtype, trust_remote_code=args.trust_remote_code)
    if use_cuda:
        load_kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(path, **load_kw)
    if not use_cuda:
        model = model.to(device)
    model.eval()

    max_pos = getattr(model.config, "max_position_embeddings", 4096)
    max_length = min(args.max_length, max_pos)
    stride = min(args.stride, max_length)

    print(f"Loading wikitext-2-raw-v1 split={args.split!r} ...")
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=args.split)
    text = "\n\n".join(t for t in raw["text"] if t)
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    print(f"Tokenized length: {seq_len} tokens, chunk max_length={max_length}, stride={stride}")

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
            # Mean NLL over non-ignored positions; scale to sum over trg_len tokens
            nlls.append(out.loss * trg_len)
        prev_end_loc = end_loc
        if end_loc >= seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    denom = prev_end_loc
    ppl = torch.exp(total_nll / denom).item()
    print(f"\nWikiText-2-raw-v1 ({args.split})  perplexity ≈ {ppl:.4f}")
    print(f"(aggregated negative log-likelihood / {denom} tokens)")


if __name__ == "__main__":
    main()
