from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import cuda_memory_snapshot, emit_result, print_result, runtime_metadata
from ppl_adapters import prepare_eval_model


def extract_number(text: str) -> str | None:
    marker_match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if marker_match:
        return marker_match.group(1).replace(",", "")
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def canonical_number(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
    ds = load_dataset("gsm8k", "main", split=args.split, cache_dir=args.dataset_cache_dir or None)
    examples = []
    for index, row in enumerate(ds):
        answer_number = canonical_number(extract_number(row["answer"]))
        if answer_number is None:
            continue
        examples.append({"index": index, "question": row["question"], "answer": row["answer"], "answer_number": answer_number})
        if args.max_samples and len(examples) >= args.max_samples:
            break
    return examples


def format_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer: Let's think step by step."


def run_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
    device: str,
) -> list[dict[str, Any]]:
    outputs = []
    for example in examples:
        prompt = format_prompt(example["question"])
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_prompt_tokens)
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device) if "attention_mask" in encoded else None
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0, input_ids.shape[1] :].cpu().tolist()
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred_number = canonical_number(extract_number(text))
        outputs.append(
            {
                "index": example["index"],
                "answer_number": example["answer_number"],
                "prediction_number": pred_number,
                "correct": pred_number == example["answer_number"],
                "generated_text": text,
            }
        )
    return outputs


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def summarize(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(outputs)
    valid = sum(1 for item in outputs if item["prediction_number"] is not None)
    correct = sum(1 for item in outputs if item["correct"])
    return {
        "total": total,
        "valid_predictions": valid,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "valid_rate": valid / total if total else 0.0,
    }


def compare_outputs(baseline: list[dict[str, Any]], quant: list[dict[str, Any]]) -> dict[str, Any]:
    total = min(len(baseline), len(quant))
    same_number = sum(
        1 for left, right in zip(baseline, quant) if left["prediction_number"] == right["prediction_number"]
    )
    flips = [
        {
            "index": left["index"],
            "answer_number": left["answer_number"],
            "baseline_prediction": left["prediction_number"],
            "quant_prediction": right["prediction_number"],
            "baseline_correct": left["correct"],
            "quant_correct": right["correct"],
        }
        for left, right in zip(baseline, quant)
        if left["prediction_number"] != right["prediction_number"]
    ]
    return {
        "total": total,
        "same_prediction_rate": same_number / total if total else 0.0,
        "num_prediction_flips": len(flips),
        "flips_sample": flips[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory-safe GSM8K greedy numeric generation probe")
    parser.add_argument("--baseline-model-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--quant-format", default="auto")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-cache-dir", default="")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
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
    result["task"] = "gsm8k_greedy_numeric"
    result["args"] = vars(args)
    result["baseline_model_path"] = baseline_model_path
    result["baseline_quant"] = runtime_metadata(baseline_model_path, "baseline")["quant"]

    examples = load_examples(args)
    result["example_count"] = len(examples)
    eval_model_path, tempdir, adapter_stats = prepare_eval_model(model_path, args.quant_format, dtype)
    result["adapter"] = adapter_stats
    try:
        tokenizer = AutoTokenizer.from_pretrained(baseline_model_path, trust_remote_code=args.trust_remote_code)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        baseline_model = AutoModelForCausalLM.from_pretrained(
            baseline_model_path, torch_dtype=dtype, trust_remote_code=args.trust_remote_code
        ).to(device)
        baseline_model.eval()
        baseline_outputs = run_model(baseline_model, tokenizer, examples, args, device)
        result["memory_after_baseline"] = cuda_memory_snapshot("after_baseline_")
        del baseline_model
        release_cuda_memory()

        quant_model = AutoModelForCausalLM.from_pretrained(
            eval_model_path, torch_dtype=dtype, trust_remote_code=args.trust_remote_code
        ).to(device)
        quant_model.eval()
        quant_outputs = run_model(quant_model, tokenizer, examples, args, device)
        result["memory_after_quant"] = cuda_memory_snapshot("after_quant_")
        del quant_model
        release_cuda_memory()

        result["baseline_summary"] = summarize(baseline_outputs)
        result["quant_summary"] = summarize(quant_outputs)
        result["comparison"] = compare_outputs(baseline_outputs, quant_outputs)
        result["outputs_sample"] = [
            {"baseline": left, "quant": right}
            for left, right in list(zip(baseline_outputs, quant_outputs))[:20]
        ]
        result["elapsed_s"] = perf_counter() - start
    finally:
        if tempdir is not None:
            tempdir.cleanup()

    emit_result(args, result)
    print_result(
        result,
        [
            "label",
            "task",
            "example_count",
            "baseline_summary.accuracy",
            "quant_summary.accuracy",
            "comparison.same_prediction_rate",
            "comparison.num_prediction_flips",
            "elapsed_s",
        ],
    )


if __name__ == "__main__":
    main()
