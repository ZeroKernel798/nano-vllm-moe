from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import cuda_memory_snapshot, emit_result, print_result, runtime_metadata
from ppl_adapters import prepare_eval_model


def parse_configs(value: str) -> list[str | None]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or [None]


def normalize_answer(answer: Any, labels: list[str]) -> str | None:
    if isinstance(answer, int):
        return labels[answer] if 0 <= answer < len(labels) else None
    text = str(answer).strip()
    if not text:
        return None
    if text.isdigit():
        idx = int(text)
        return labels[idx] if 0 <= idx < len(labels) else None
    first = text[0].upper()
    return first if first in labels else None


def normalize_example(example: dict[str, Any], dataset_kind: str) -> dict[str, Any] | None:
    labels = ["A", "B", "C", "D"]
    if dataset_kind == "mmlu":
        choices = example.get("choices")
        if not isinstance(choices, list) or len(choices) < 4:
            return None
        answer = normalize_answer(example.get("answer"), labels)
        question = str(example.get("question", "")).strip()
    elif dataset_kind == "ceval":
        choices = [str(example.get(label, "")).strip() for label in labels]
        answer = normalize_answer(example.get("answer"), labels)
        question = str(example.get("question", "")).strip()
    else:
        raise ValueError(f"Unsupported dataset_kind={dataset_kind!r}")
    if not question or answer is None or any(not choice for choice in choices[:4]):
        return None
    return {"question": question, "choices": choices[:4], "answer": answer}


def load_choice_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    configs = parse_configs(args.dataset_config)
    per_config = max(args.max_samples_per_config, 0)
    for config in configs:
        kwargs = {"split": args.split, "cache_dir": args.dataset_cache_dir or None}
        if config is not None:
            dataset = load_dataset(args.dataset, config, **kwargs)
            config_name = config
        else:
            dataset = load_dataset(args.dataset, **kwargs)
            config_name = "default"
        kept = 0
        for row in dataset:
            item = normalize_example(row, args.dataset_kind)
            if item is None:
                continue
            item["config"] = config_name
            examples.append(item)
            kept += 1
            if per_config and kept >= per_config:
                break
        if args.max_samples and len(examples) >= args.max_samples:
            examples = examples[: args.max_samples]
            break
    return examples


def format_prompt(example: dict[str, Any]) -> str:
    lines = [example["question"]]
    for label, choice in zip(["A", "B", "C", "D"], example["choices"]):
        lines.append(f"{label}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def choice_token_ids(tokenizer: AutoTokenizer, labels: list[str]) -> dict[str, list[int]]:
    variants: dict[str, list[int]] = {}
    for label in labels:
        ids: set[int] = set()
        for text in (label, f" {label}", f"\n{label}"):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if encoded:
                ids.add(encoded[-1])
        variants[label] = sorted(ids)
    return variants


def score_examples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
    device: str,
) -> list[dict[str, Any]]:
    labels = ["A", "B", "C", "D"]
    label_token_ids = choice_token_ids(tokenizer, labels)
    outputs: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        prompt = format_prompt(example)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_prompt_tokens)
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device) if "attention_mask" in encoded else None
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1].float().cpu()
        scores = {label: max(float(logits[token_id].item()) for token_id in label_token_ids[label]) for label in labels}
        pred = max(labels, key=lambda label: scores[label])
        outputs.append(
            {
                "index": index,
                "config": example["config"],
                "answer": example["answer"],
                "prediction": pred,
                "correct": pred == example["answer"],
                "scores": scores,
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
    correct = sum(1 for item in outputs if item["correct"])
    by_config: dict[str, dict[str, int]] = {}
    for item in outputs:
        bucket = by_config.setdefault(item["config"], {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += int(item["correct"])
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_config": {
            name: {**stats, "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0}
            for name, stats in sorted(by_config.items())
        },
    }


def compare_outputs(baseline: list[dict[str, Any]], quant: list[dict[str, Any]]) -> dict[str, Any]:
    total = min(len(baseline), len(quant))
    agreements = sum(1 for left, right in zip(baseline, quant) if left["prediction"] == right["prediction"])
    flips = [
        {
            "index": left["index"],
            "config": left["config"],
            "answer": left["answer"],
            "baseline_prediction": left["prediction"],
            "quant_prediction": right["prediction"],
            "baseline_correct": left["correct"],
            "quant_correct": right["correct"],
        }
        for left, right in zip(baseline, quant)
        if left["prediction"] != right["prediction"]
    ]
    return {
        "total": total,
        "prediction_agreement": agreements / total if total else 0.0,
        "num_prediction_flips": len(flips),
        "flips_sample": flips[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory-safe MMLU/CEval logit-rank multiple-choice gate")
    parser.add_argument("--baseline-model-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--quant-format", default="auto")
    parser.add_argument("--dataset-kind", choices=("mmlu", "ceval"), default="mmlu")
    parser.add_argument("--dataset", default="cais/mmlu")
    parser.add_argument("--dataset-config", default="abstract_algebra")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-cache-dir", default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-samples-per-config", type=int, default=100)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
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
    result["task"] = "choice_logit_rank"
    result["args"] = vars(args)
    result["baseline_model_path"] = baseline_model_path
    result["baseline_quant"] = runtime_metadata(baseline_model_path, "baseline")["quant"]

    examples = load_choice_examples(args)
    if not examples:
        raise RuntimeError("No multiple-choice examples loaded")
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
        baseline_outputs = score_examples(baseline_model, tokenizer, examples, args, device)
        result["memory_after_baseline"] = cuda_memory_snapshot("after_baseline_")
        del baseline_model
        release_cuda_memory()

        quant_model = AutoModelForCausalLM.from_pretrained(
            eval_model_path, torch_dtype=dtype, trust_remote_code=args.trust_remote_code
        ).to(device)
        quant_model.eval()
        quant_outputs = score_examples(quant_model, tokenizer, examples, args, device)
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
            "comparison.prediction_agreement",
            "comparison.num_prediction_flips",
            "elapsed_s",
        ],
    )


if __name__ == "__main__":
    main()
