from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import cuda_memory_snapshot, emit_result, print_result, runtime_metadata
from ppl_adapters import prepare_eval_model


def default_prompts() -> list[str]:
    return [
        "The capital of France is",
        "In a distant future, humans and machines",
        "Quantization changes model weights, but the generated text should",
    ]


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file:
        prompts = [line.strip() for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines()]
        return [prompt for prompt in prompts if prompt]
    if args.prompt:
        return args.prompt
    return default_prompts()


def load_model(model_path: str, dtype: torch.dtype, device: str, trust_remote_code: bool) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model = model.to(device)
    model.eval()
    return model


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    a_top = set(a.topk(k).indices.tolist())
    b_top = set(b.topk(k).indices.tolist())
    return len(a_top & b_top) / max(k, 1)


def compare_prompt(
    baseline_model: AutoModelForCausalLM,
    quant_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_prompt_tokens)
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device) if "attention_mask" in encoded else None
    with torch.inference_mode():
        baseline_logits = baseline_model(input_ids=input_ids, attention_mask=attention_mask).logits
        quant_logits = quant_model(input_ids=input_ids, attention_mask=attention_mask).logits
    b = baseline_logits[:, -1, :].float().squeeze(0)
    q = quant_logits[:, -1, :].float().squeeze(0)
    diff = (q - b).abs()
    baseline_top1 = int(b.argmax().item())
    quant_top1 = int(q.argmax().item())
    generated_baseline = baseline_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_quant = quant_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    baseline_new = generated_baseline[0, input_ids.shape[1] :].tolist()
    quant_new = generated_quant[0, input_ids.shape[1] :].tolist()
    aligned = sum(1 for left, right in zip(baseline_new, quant_new) if left == right)
    denom = max(len(baseline_new), len(quant_new), 1)
    return {
        "prompt": prompt,
        "prompt_tokens": int(input_ids.numel()),
        "logits": {
            "max_abs_error": float(diff.max().item()),
            "mean_abs_error": float(diff.mean().item()),
            "cosine": float(F.cosine_similarity(b, q, dim=0).item()),
            "top1_match": baseline_top1 == quant_top1,
            "baseline_top1_token_id": baseline_top1,
            "quant_top1_token_id": quant_top1,
            "top5_overlap": topk_overlap(b, q, 5),
            "top10_overlap": topk_overlap(b, q, 10),
        },
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "baseline_token_ids": baseline_new,
            "quant_token_ids": quant_new,
            "token_match_count": aligned,
            "token_match_ratio": aligned / denom,
            "exact_match": baseline_new == quant_new,
            "baseline_text": tokenizer.decode(baseline_new, skip_special_tokens=False),
            "quant_text": tokenizer.decode(quant_new, skip_special_tokens=False),
        },
    }


def summarize(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    if not comparisons:
        return {}
    return {
        "num_prompts": len(comparisons),
        "avg_max_abs_error": sum(item["logits"]["max_abs_error"] for item in comparisons) / len(comparisons),
        "avg_mean_abs_error": sum(item["logits"]["mean_abs_error"] for item in comparisons) / len(comparisons),
        "avg_cosine": sum(item["logits"]["cosine"] for item in comparisons) / len(comparisons),
        "top1_match_rate": sum(1 for item in comparisons if item["logits"]["top1_match"]) / len(comparisons),
        "avg_top5_overlap": sum(item["logits"]["top5_overlap"] for item in comparisons) / len(comparisons),
        "avg_top10_overlap": sum(item["logits"]["top10_overlap"] for item in comparisons) / len(comparisons),
        "generation_exact_match_rate": sum(1 for item in comparisons if item["generation"]["exact_match"]) / len(comparisons),
        "avg_generation_token_match_ratio": sum(item["generation"]["token_match_ratio"] for item in comparisons) / len(comparisons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BF16 vs quant logits and greedy generated tokens")
    parser.add_argument("--baseline-model-path", required=True)
    parser.add_argument("--model-path", required=True, help="Quantized model path")
    parser.add_argument("--label", default="")
    parser.add_argument("--quant-format", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file")
    parser.add_argument("--max-prompt-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    model_path = str(Path(args.model_path).expanduser())
    baseline_model_path = str(Path(args.baseline_model_path).expanduser())
    label = args.label or Path(model_path).name
    dtype = getattr(torch, args.dtype)
    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device = args.device if use_cuda else "cpu"

    start = perf_counter()
    result = runtime_metadata(model_path, label)
    result["task"] = "logits_token_correctness"
    result["args"] = vars(args)
    result["baseline_model_path"] = baseline_model_path
    result["baseline_quant"] = runtime_metadata(baseline_model_path, "baseline")["quant"]

    eval_model_path, tempdir, adapter_stats = prepare_eval_model(model_path, args.quant_format, dtype)
    result["adapter"] = adapter_stats
    try:
        tokenizer = AutoTokenizer.from_pretrained(baseline_model_path, trust_remote_code=args.trust_remote_code)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        baseline_model = load_model(baseline_model_path, dtype, device, args.trust_remote_code)
        quant_model = load_model(eval_model_path, dtype, device, args.trust_remote_code)
        result["memory_after_load"] = cuda_memory_snapshot("after_load_")
        prompts = load_prompts(args)
        comparisons = [compare_prompt(baseline_model, quant_model, tokenizer, prompt, args, device) for prompt in prompts]
        result["comparisons"] = comparisons
        result["summary"] = summarize(comparisons)
        result["elapsed_s"] = perf_counter() - start
        result["memory_after_compare"] = cuda_memory_snapshot("after_compare_")
    finally:
        if tempdir is not None:
            tempdir.cleanup()

    emit_result(args, result)
    print_result(
        result,
        [
            "label",
            "task",
            "adapter.quant_format",
            "adapter.proxy",
            "summary.num_prompts",
            "summary.avg_cosine",
            "summary.top1_match_rate",
            "summary.avg_top10_overlap",
            "summary.generation_exact_match_rate",
            "summary.avg_generation_token_match_ratio",
        ],
    )


if __name__ == "__main__":
    main()
