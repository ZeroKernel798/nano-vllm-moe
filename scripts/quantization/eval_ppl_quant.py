from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import cuda_memory_snapshot, emit_result, print_result, runtime_metadata
from ppl_adapters import prepare_eval_model
from ppl_data import add_dataset_args, load_eval_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic quantization perplexity evaluation")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--quant-format", default="auto", help="auto, none, fp8_w8a16; future adapters can add awq/gptq")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    add_dataset_args(parser)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=0, help="Limit tokenized length for smoke tests")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    model_path = str(Path(args.model_path).expanduser())
    label = args.label or Path(model_path).name
    dtype = getattr(torch, args.dtype)
    result = runtime_metadata(model_path, label)
    result["task"] = "perplexity"
    result["args"] = vars(args)
    start = perf_counter()

    eval_model_path, tempdir, adapter_stats = prepare_eval_model(model_path, args.quant_format, dtype)
    result["adapter"] = adapter_stats
    try:
        use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
        device = args.device if use_cuda else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(eval_model_path, trust_remote_code=args.trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            eval_model_path,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        model = model.to(device)
        model.eval()
        result["memory_after_load"] = cuda_memory_snapshot("after_load_")

        text = load_eval_text(args)
        encodings = tokenizer(text, return_tensors="pt")
        if args.max_tokens and encodings.input_ids.size(1) > args.max_tokens:
            encodings.input_ids = encodings.input_ids[:, : args.max_tokens]
        input_ids = encodings.input_ids.to(device)
        seq_len = input_ids.size(1)
        max_pos = getattr(model.config, "max_position_embeddings", args.max_length)
        max_length = min(args.max_length, max_pos, seq_len)
        stride = min(args.stride, max_length)

        nlls: list[torch.Tensor] = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride), desc="ppl_windows"):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            if trg_len <= 0:
                break
            chunk = input_ids[:, begin_loc:end_loc]
            target_ids = chunk.clone()
            target_ids[:, :-trg_len] = -100
            with torch.inference_mode():
                output = model(chunk, labels=target_ids)
                nlls.append(output.loss * trg_len)
            prev_end_loc = end_loc
            if end_loc >= seq_len:
                break
        total_nll = torch.stack(nlls).sum()
        ppl = torch.exp(total_nll / prev_end_loc).item()
        result["metrics"] = {
            "perplexity": ppl,
            "tokens": prev_end_loc,
            "windows": len(nlls),
            "max_length": max_length,
            "stride": stride,
            "elapsed_s": perf_counter() - start,
        }
        result["memory_after_eval"] = cuda_memory_snapshot("after_eval_")
    finally:
        if tempdir is not None:
            tempdir.cleanup()

    emit_result(args, result)
    print_result(
        result,
        ["label", "task", "adapter.quant_format", "adapter.proxy", "metrics.perplexity", "metrics.tokens", "metrics.elapsed_s"],
    )


if __name__ == "__main__":
    main()
