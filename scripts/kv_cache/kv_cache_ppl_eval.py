"""Teacher-forced KV-cache NLL/PPL evaluation.

This evaluates KV-cache quantization through the normal nano-vLLM scheduler:
prefill a prefix, then decode one token at a time while forcing the next
ground-truth token into the KV cache. This avoids measuring a full forward pass
that never exercises decode-time KV-cache reads.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context


BACKENDS = {"bf16", "native", "gather_dequant", "fp8_gather"}


def _split_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _split_ints(text: str) -> list[int]:
    return [int(item) for item in _split_csv(text)]


def _cache_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size()) if tensor.numel() else 0


def _timed_cuda(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, perf_counter() - start


def _read_text_file(path: Path, text_column: str) -> str:
    if path.suffix == ".jsonl":
        texts: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            value = obj.get(text_column, "")
            if value:
                texts.append(str(value))
        return "\n\n".join(texts)
    return path.read_text(encoding="utf-8")


def _load_text_from_path(path: str, split: str, text_column: str, max_rows: int) -> str:
    root = Path(path).expanduser()
    if root.is_file():
        return _read_text_file(root, text_column)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset path does not exist: {root}")

    parquet_files = sorted(root.glob(f"{split}-*.parquet")) or sorted(root.glob("*.parquet"))
    if parquet_files:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("datasets is required to read parquet datasets") from exc
        raw = load_dataset("parquet", data_files={split: [str(p) for p in parquet_files]}, split=split)
        texts = [str(row.get(text_column, "")) for idx, row in enumerate(raw) if max_rows <= 0 or idx < max_rows]
        return "\n\n".join(text for text in texts if text)

    jsonl_files = sorted(root.glob(f"{split}*.jsonl")) or sorted(root.glob("*.jsonl"))
    if jsonl_files:
        texts = []
        row_count = 0
        for file in jsonl_files:
            for line in file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                value = obj.get(text_column, "")
                if value:
                    texts.append(str(value))
                    row_count += 1
                    if 0 < max_rows <= row_count:
                        return "\n\n".join(texts)
        return "\n\n".join(texts)

    text_files = sorted(root.glob(f"{split}*.txt")) or sorted(root.glob("*.txt"))
    if text_files:
        return "\n\n".join(file.read_text(encoding="utf-8") for file in text_files)

    raise FileNotFoundError(f"no parquet/jsonl/txt files found under {root}")


def _load_token_ids(args: argparse.Namespace, tokenizer) -> list[int]:
    text = _load_text_from_path(args.dataset_path, args.split, args.text_column, args.max_rows)
    if args.max_chars and len(text) > args.max_chars:
        text = text[: args.max_chars]
    token_ids = tokenizer.encode(text)
    if args.max_tokens and len(token_ids) > args.max_tokens:
        token_ids = token_ids[: args.max_tokens]
    if len(token_ids) < 2:
        raise RuntimeError("not enough tokens loaded for PPL evaluation")
    return [int(token_id) for token_id in token_ids]


def _init_llm(args: argparse.Namespace, backend: str) -> LLM:
    experimental = backend != "bf16"
    kv_cache_dtype = "fp8" if backend == "fp8_gather" else "k_int8_v_fp8"
    return LLM(
        args.model_path,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tp_size=1,
        ep_size=1,
        kv_cache_dtype=kv_cache_dtype if experimental else "bf16",
        experimental_kv_cache_fp8=experimental,
        kv_cache_scale_dtype=args.kv_cache_scale_dtype,
    )


def _force_postprocess(llm: LLM, seqs: list[Sequence], forced_token_ids: list[int], is_prefill: bool) -> None:
    llm.scheduler.postprocess(seqs, forced_token_ids, is_prefill)
    reset_context()


def _run_window(args: argparse.Namespace, llm: LLM, tokens: list[int], start: int, context_len: int, eval_tokens: int) -> dict[str, Any]:
    prompt = tokens[start : start + context_len]
    targets = tokens[start + context_len : start + context_len + eval_tokens]
    if len(prompt) != context_len or len(targets) != eval_tokens:
        raise ValueError("window is shorter than requested context/eval length")

    sampling_params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=eval_tokens)
    seq = Sequence(prompt, sampling_params)
    llm.scheduler.add(seq)

    nll_sum = 0.0
    token_count = 0
    argmax_match = 0
    target_rank_sum = 0.0
    top5_contains = 0
    top10_contains = 0
    model_time_s = 0.0
    prefill_steps = 0
    decode_steps = 0
    target_idx = 0

    while not llm.scheduler.is_finished():
        seqs, is_prefill = llm.scheduler.schedule()
        if len(seqs) != 1:
            raise RuntimeError("PPL eval expects max_num_seqs=1")
        if is_prefill:
            (input_ids, positions), _ = _timed_cuda(lambda: llm.model_runner.prepare_prefill(seqs))
            prefill_steps += 1
        else:
            (input_ids, positions), _ = _timed_cuda(lambda: llm.model_runner.prepare_decode(seqs))
            decode_steps += 1
        logits, model_time = _timed_cuda(lambda: llm.model_runner.run_model(input_ids, positions, is_prefill))
        model_time_s += model_time
        if logits is None:
            raise RuntimeError("logits are None; this script expects tp_size=ep_size=1")

        append_now = not (is_prefill and seq.num_cached_tokens + seq.num_scheduled_tokens < seq.num_tokens)
        if append_now:
            target = targets[target_idx]
            row = logits[-1].float()
            target_tensor = torch.tensor([target], dtype=torch.long, device=row.device)
            nll = F.cross_entropy(row.unsqueeze(0), target_tensor, reduction="sum")
            nll_sum += float(nll.item())
            token_count += 1
            if int(torch.argmax(row).item()) == int(target):
                argmax_match += 1
            rank = int((row > row[target]).sum().item()) + 1
            target_rank_sum += float(rank)
            topk = torch.topk(row, k=min(10, row.numel())).indices.tolist()
            if int(target) in topk[:5]:
                top5_contains += 1
            if int(target) in topk:
                top10_contains += 1
            forced_token = target
            target_idx += 1
        else:
            forced_token = int(torch.argmax(logits[-1].float()).item())

        _force_postprocess(llm, seqs, [forced_token], is_prefill)

    return {
        "start": start,
        "context_len": context_len,
        "eval_tokens": token_count,
        "nll_sum": nll_sum,
        "mean_nll": nll_sum / max(token_count, 1),
        "ppl": math.exp(nll_sum / max(token_count, 1)),
        "argmax_match_rate": argmax_match / max(token_count, 1),
        "target_rank_mean": target_rank_sum / max(token_count, 1),
        "target_in_top5_rate": top5_contains / max(token_count, 1),
        "target_in_top10_rate": top10_contains / max(token_count, 1),
        "prefill_steps": prefill_steps,
        "decode_steps": decode_steps,
        "model_time_s": model_time_s,
    }


def _run_backend(args: argparse.Namespace, token_ids: list[int], backend: str) -> dict[str, Any]:
    experimental = backend != "bf16"
    old_decode_backend = os.environ.get("NANOVLLM_FP8_KV_DECODE")
    old_block_tokens = os.environ.get("NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS")
    old_group_size = os.environ.get("NANOVLLM_K_INT8_GROUP_SIZE")
    old_gather_block_tokens = os.environ.get("NANOVLLM_K_INT8_V_FP8_GATHER_BLOCK_TOKENS")
    if experimental and backend != "fp8_gather":
        os.environ["NANOVLLM_FP8_KV_DECODE"] = backend
        os.environ["NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS"] = str(args.native_block_tokens)
        os.environ["NANOVLLM_K_INT8_GROUP_SIZE"] = str(args.k_group_size)
        os.environ["NANOVLLM_K_INT8_V_FP8_GATHER_BLOCK_TOKENS"] = str(args.gather_block_tokens)
    elif backend == "fp8_gather":
        os.environ["NANOVLLM_K_INT8_V_FP8_GATHER_BLOCK_TOKENS"] = str(args.gather_block_tokens)

    llm = None
    start_time = perf_counter()
    try:
        llm = _init_llm(args, backend)
        rows: list[dict[str, Any]] = []
        for context_len in _split_ints(args.context_lens):
            available = len(token_ids) - context_len - args.eval_tokens
            if available < 0:
                rows.append({"ok": False, "backend": backend, "context_len": context_len, "error": "not_enough_tokens"})
                continue
            starts = list(range(0, available + 1, args.window_stride))[: args.max_windows]
            for start in starts:
                row = _run_window(args, llm, token_ids, start, context_len, args.eval_tokens)
                row["ok"] = True
                row["backend"] = backend
                row["k_group_size"] = args.k_group_size if experimental else None
                row["native_block_tokens"] = args.native_block_tokens if experimental else None
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)

        ok_rows = [row for row in rows if row.get("ok")]
        nll_sum = sum(float(row["nll_sum"]) for row in ok_rows)
        total_tokens = sum(int(row["eval_tokens"]) for row in ok_rows)
        runner = llm.model_runner
        num_blocks = int(runner.config.num_kvcache_blocks)
        kv_data_bytes = _cache_bytes(runner.kv_cache)
        if kv_data_bytes == 0:
            kv_data_bytes = _cache_bytes(runner.k_cache_storage) + _cache_bytes(runner.v_cache_storage)
        scale_bytes = _cache_bytes(runner.k_scale_cache) + _cache_bytes(runner.v_scale_cache)
        return {
            "backend": backend,
            "ok": bool(ok_rows),
            "k_group_size": args.k_group_size if experimental and backend != "fp8_gather" else None,
            "native_block_tokens": args.native_block_tokens if experimental and backend != "fp8_gather" else None,
            "gather_block_tokens": args.gather_block_tokens if experimental else None,
            "total_eval_tokens": total_tokens,
            "windows": len(ok_rows),
            "nll_sum": nll_sum,
            "mean_nll": nll_sum / max(total_tokens, 1),
            "ppl": math.exp(nll_sum / max(total_tokens, 1)) if total_tokens else float("inf"),
            "argmax_match_rate": sum(float(row["argmax_match_rate"]) * int(row["eval_tokens"]) for row in ok_rows) / max(total_tokens, 1),
            "target_rank_mean": sum(float(row["target_rank_mean"]) * int(row["eval_tokens"]) for row in ok_rows) / max(total_tokens, 1),
            "target_in_top5_rate": sum(float(row["target_in_top5_rate"]) * int(row["eval_tokens"]) for row in ok_rows) / max(total_tokens, 1),
            "target_in_top10_rate": sum(float(row["target_in_top10_rate"]) * int(row["eval_tokens"]) for row in ok_rows) / max(total_tokens, 1),
            "num_kvcache_blocks": num_blocks,
            "kv_cache_data_storage_bytes": kv_data_bytes,
            "kv_cache_scale_storage_bytes": scale_bytes,
            "kv_cache_total_storage_bytes": kv_data_bytes + scale_bytes,
            "kv_cache_total_bytes_per_block": (kv_data_bytes + scale_bytes) / max(num_blocks, 1),
            "elapsed_s": perf_counter() - start_time,
            "rows": rows,
        }
    finally:
        if llm is not None:
            llm.exit()
            del llm
        for name, old_value in (
            ("NANOVLLM_FP8_KV_DECODE", old_decode_backend),
            ("NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS", old_block_tokens),
            ("NANOVLLM_K_INT8_GROUP_SIZE", old_group_size),
            ("NANOVLLM_K_INT8_V_FP8_GATHER_BLOCK_TOKENS", old_gather_block_tokens),
        ):
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher-forced KV-cache NLL/PPL eval")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--context-lens", default="4096,8192")
    parser.add_argument("--eval-tokens", type=int, default=128)
    parser.add_argument("--max-windows", type=int, default=2)
    parser.add_argument("--window-stride", type=int, default=2048)
    parser.add_argument("--backends", default="bf16,native")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--kv-cache-scale-dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--k-group-size", type=int, default=8)
    parser.add_argument("--native-block-tokens", type=int, default=32, choices=(16, 32, 64))
    parser.add_argument("--gather-block-tokens", type=int, default=16, choices=(1, 2, 4, 8, 16, 32, 64))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    args.model_path = str(Path(args.model_path).expanduser())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=args.trust_remote_code)
    token_ids = _load_token_ids(args, tokenizer)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summaries = []
    for backend in _split_csv(args.backends):
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; choices={sorted(BACKENDS)}")
        summary = _run_backend(args, token_ids, backend)
        summaries.append(summary)
        (output_dir / f"{backend}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    by_backend = {item["backend"]: item for item in summaries}
    bf16 = by_backend.get("bf16")
    comparison_rows = []
    for item in summaries:
        row = {
            "backend": item["backend"],
            "k_group_size": item.get("k_group_size"),
            "windows": item["windows"],
            "total_eval_tokens": item["total_eval_tokens"],
            "mean_nll": item["mean_nll"],
            "ppl": item["ppl"],
            "argmax_match_rate": item["argmax_match_rate"],
            "target_rank_mean": item["target_rank_mean"],
            "target_in_top5_rate": item["target_in_top5_rate"],
            "target_in_top10_rate": item["target_in_top10_rate"],
            "kv_cache_total_bytes_per_block": item["kv_cache_total_bytes_per_block"],
            "num_kvcache_blocks": item["num_kvcache_blocks"],
        }
        if bf16 and item["backend"] != "bf16":
            row["delta_nll"] = item["mean_nll"] - bf16["mean_nll"]
            row["relative_ppl_regression"] = item["ppl"] / bf16["ppl"] - 1.0
            row["block_ratio_over_bf16"] = item["num_kvcache_blocks"] / max(bf16["num_kvcache_blocks"], 1)
            row["bytes_per_block_ratio_over_bf16"] = item["kv_cache_total_bytes_per_block"] / max(bf16["kv_cache_total_bytes_per_block"], 1.0e-9)
        comparison_rows.append(row)

    result = {
        "args": vars(args),
        "token_count": len(token_ids),
        "summaries": summaries,
        "comparison": comparison_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({key for row in comparison_rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(json.dumps({"comparison": comparison_rows, "summary_json": str(output_dir / "summary.json")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
