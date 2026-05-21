"""Capacity sweep for BF16 KV vs K-int8/V-FP8 mixed KV.

The driver mode launches every case in a fresh Python process so CUDA allocator
state and OOM failures do not poison later cases. The child ``--case-mode``
path writes one JSON record and exits with a non-zero status on failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm import LLM, SamplingParams


def _split_ints(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def _cuda_memory() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {}
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "free_bytes": int(free),
        "total_bytes": int(total),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _cache_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size()) if tensor.numel() else 0


def _effective_input_len(requested_input_len: int, output_len: int, max_model_len: int) -> int:
    return min(requested_input_len, max_model_len - output_len)


def _make_prompts(num_seqs: int, input_len: int, seed: int, vocab_range: int) -> list[list[int]]:
    prompts = []
    for seq_idx in range(num_seqs):
        rng = random.Random(seed * 1000003 + seq_idx * 9176)
        prompts.append([rng.randint(0, vocab_range) for _ in range(input_len)])
    return prompts


def _storage_summary(llm: LLM) -> dict[str, Any]:
    runner = llm.model_runner
    kv_data_bytes = _cache_bytes(runner.kv_cache)
    if kv_data_bytes == 0:
        kv_data_bytes = _cache_bytes(runner.k_cache_storage) + _cache_bytes(runner.v_cache_storage)
    scale_bytes = _cache_bytes(runner.k_scale_cache) + _cache_bytes(runner.v_scale_cache)
    num_blocks = int(runner.config.num_kvcache_blocks)
    return {
        "num_kvcache_blocks": num_blocks,
        "kvcache_block_size": int(runner.config.kvcache_block_size),
        "kv_cache_data_storage_bytes": kv_data_bytes,
        "kv_cache_scale_storage_bytes": scale_bytes,
        "kv_cache_total_storage_bytes": kv_data_bytes + scale_bytes,
        "kv_cache_total_bytes_per_block": (kv_data_bytes + scale_bytes) / max(num_blocks, 1),
        "free_kvcache_blocks_after_generate": len(llm.scheduler.block_manager.free_block_ids),
        "used_kvcache_blocks_after_generate": len(llm.scheduler.block_manager.used_block_ids),
    }


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    max_model_len = args.max_model_len
    if max_model_len <= 0:
        from transformers import AutoConfig

        max_model_len = int(AutoConfig.from_pretrained(args.model_path).max_position_embeddings)
    effective_input_len = _effective_input_len(args.requested_input_len, args.output_len, max_model_len)
    if effective_input_len <= 0:
        raise ValueError(
            "effective input length is non-positive: "
            f"requested_input_len={args.requested_input_len}, output_len={args.output_len}, "
            f"max_model_len={max_model_len}"
        )

    old_decode_backend = os.environ.get("NANOVLLM_FP8_KV_DECODE")
    old_block_tokens = os.environ.get("NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS")
    old_group_size = os.environ.get("NANOVLLM_K_INT8_GROUP_SIZE")
    old_gather_block_tokens = os.environ.get("NANOVLLM_K_INT8_V_FP8_GATHER_BLOCK_TOKENS")
    if args.mode == "k_int8_v_fp8":
        os.environ["NANOVLLM_FP8_KV_DECODE"] = args.fp8_decode_backend
        os.environ["NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS"] = str(args.native_block_tokens)
        os.environ["NANOVLLM_K_INT8_GROUP_SIZE"] = str(args.k_group_size)
        os.environ["NANOVLLM_K_INT8_V_FP8_GATHER_BLOCK_TOKENS"] = str(args.gather_block_tokens)

    llm = None
    try:
        prompts = _make_prompts(args.num_seqs, effective_input_len, args.seed, args.vocab_range)
        sampling_params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.output_len)
        before = _cuda_memory()
        llm = LLM(
            args.model_path,
            enforce_eager=True,
            max_model_len=max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tp_size=1,
            ep_size=1,
            kv_cache_dtype=args.mode,
            experimental_kv_cache_fp8=args.mode == "k_int8_v_fp8",
            kv_cache_scale_dtype=args.kv_cache_scale_dtype,
            chunked_prefill_policy=args.chunked_prefill_policy,
        )
        after_load = _cuda_memory()
        t0 = perf_counter()
        output = llm.generate(prompts, sampling_params, use_tqdm=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_time_s = perf_counter() - t0
        after_generate = _cuda_memory()
        storage = _storage_summary(llm)
        required_prompt_blocks = args.num_seqs * ((effective_input_len + storage["kvcache_block_size"] - 1) // storage["kvcache_block_size"])
        required_total_blocks = args.num_seqs * (
            (effective_input_len + args.output_len + storage["kvcache_block_size"] - 1) // storage["kvcache_block_size"]
        )
        results = output["results"]
        return {
            "ok": True,
            "mode": args.mode,
            "num_seqs": args.num_seqs,
            "requested_input_len": args.requested_input_len,
            "effective_input_len": effective_input_len,
            "output_len": args.output_len,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "kv_cache_scale_dtype": args.kv_cache_scale_dtype,
            "fp8_decode_backend": args.fp8_decode_backend if args.mode == "k_int8_v_fp8" else None,
            "k_group_size": args.k_group_size if args.mode == "k_int8_v_fp8" else None,
            "native_block_tokens": args.native_block_tokens if args.mode == "k_int8_v_fp8" else None,
            "chunked_prefill_policy": args.chunked_prefill_policy,
            "required_prompt_blocks": required_prompt_blocks,
            "required_total_blocks": required_total_blocks,
            "wall_time_s": wall_time_s,
            "generated_sequences": len(results),
            "generated_tokens": sum(len(row) for row in results),
            "stats": output["stats"],
            "storage": storage,
            "memory_before": before,
            "memory_after_load": after_load,
            "memory_after_generate": after_generate,
        }
    finally:
        if llm is not None:
            llm.exit()
            del llm
        for key, value in (
            ("NANOVLLM_FP8_KV_DECODE", old_decode_backend),
            ("NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS", old_block_tokens),
            ("NANOVLLM_K_INT8_GROUP_SIZE", old_group_size),
            ("NANOVLLM_K_INT8_V_FP8_GATHER_BLOCK_TOKENS", old_gather_block_tokens),
        ):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _case_main(args: argparse.Namespace) -> int:
    try:
        result = run_case(args)
        code = 0
    except Exception as exc:
        result = {
            "ok": False,
            "mode": args.mode,
            "num_seqs": args.num_seqs,
            "requested_input_len": args.requested_input_len,
            "output_len": args.output_len,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=20),
            "memory_after_error": _cuda_memory(),
        }
        code = 1
    if args.output_json:
        _write_json(Path(args.output_json), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return code


def _run_child(base_args: argparse.Namespace, mode: str, input_len: int, num_seqs: int, output_json: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--case-mode",
        "--model-path",
        base_args.model_path,
        "--mode",
        mode,
        "--requested-input-len",
        str(input_len),
        "--output-len",
        str(base_args.output_len),
        "--num-seqs",
        str(num_seqs),
        "--max-model-len",
        str(base_args.max_model_len),
        "--max-num-batched-tokens",
        str(base_args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(base_args.gpu_memory_utilization),
        "--kv-cache-scale-dtype",
        base_args.kv_cache_scale_dtype,
        "--fp8-decode-backend",
        base_args.fp8_decode_backend,
        "--k-group-size",
        str(base_args.k_group_size),
        "--native-block-tokens",
        str(base_args.native_block_tokens),
        "--gather-block-tokens",
        str(base_args.gather_block_tokens),
        "--chunked-prefill-policy",
        base_args.chunked_prefill_policy,
        "--seed",
        str(base_args.seed),
        "--vocab-range",
        str(base_args.vocab_range),
        "--output-json",
        str(output_json),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if output_json.exists():
        record = json.loads(output_json.read_text(encoding="utf-8"))
    else:
        record = {
            "ok": False,
            "mode": mode,
            "num_seqs": num_seqs,
            "requested_input_len": input_len,
            "output_len": base_args.output_len,
            "error_type": "MissingOutput",
            "error": f"child exited {proc.returncode} without writing {output_json}",
        }
    record["child_returncode"] = proc.returncode
    record["child_log_tail"] = proc.stdout[-4000:]
    _write_json(output_json, record)
    return record


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    max_ok: dict[str, dict[str, int]] = {}
    for record in records:
        input_key = str(record["requested_input_len"])
        mode = record["mode"]
        max_ok.setdefault(input_key, {}).setdefault(mode, 0)
        if record.get("ok"):
            max_ok[input_key][mode] = max(max_ok[input_key][mode], int(record["num_seqs"]))

    rows = []
    for input_key in sorted(max_ok, key=lambda x: int(x)):
        bf16 = max_ok[input_key].get("bf16", 0)
        mixed = max_ok[input_key].get("k_int8_v_fp8", 0)
        rows.append(
            {
                "requested_input_len": int(input_key),
                "bf16_max_ok_n": bf16,
                "k_int8_v_fp8_max_ok_n": mixed,
                "capacity_ratio_mixed_over_bf16": (mixed / bf16) if bf16 else None,
            }
        )
    return {"rows": rows, "records": records}


def _driver_main(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_lens = _split_ints(args.input_lens)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    records: list[dict[str, Any]] = []
    for input_len in input_lens:
        for mode in modes:
            for num_seqs in range(args.min_n, args.max_n + 1):
                case_json = output_dir / f"case_len{input_len}_{mode}_n{num_seqs}.json"
                record = _run_child(args, mode, input_len, num_seqs, case_json)
                records.append(record)
                print(
                    f"{mode} input={input_len} n={num_seqs} "
                    f"ok={record.get('ok')} returncode={record.get('child_returncode')}",
                    flush=True,
                )
                if not record.get("ok") and args.stop_on_first_fail:
                    break
    summary = _summarize(records)
    _write_json(output_dir / "summary.json", summary)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "requested_input_len",
                "bf16_max_ok_n",
                "k_int8_v_fp8_max_ok_n",
                "capacity_ratio_mixed_over_bf16",
            ],
        )
        writer.writeheader()
        writer.writerows(summary["rows"])
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="BF16 vs K-int8/V-FP8 KV concurrency capacity sweep")
    parser.add_argument("--case-mode", action="store_true")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mode", default="bf16", choices=("bf16", "k_int8_v_fp8"))
    parser.add_argument("--modes", default="bf16,k_int8_v_fp8")
    parser.add_argument("--input-lens", default="2048,4096,8192,16384,32768")
    parser.add_argument("--requested-input-len", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--min-n", type=int, default=1)
    parser.add_argument("--max-n", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--kv-cache-scale-dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--fp8-decode-backend", default="native", choices=("native", "gather_dequant"))
    parser.add_argument("--k-group-size", type=int, default=16)
    parser.add_argument("--native-block-tokens", type=int, default=32, choices=(16, 32, 64))
    parser.add_argument("--gather-block-tokens", type=int, default=16, choices=(1, 2, 4, 8, 16, 32, 64))
    parser.add_argument("--chunked-prefill-policy", default="prefill_first", choices=("prefill_first", "decode_first"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-range", type=int, default=10000)
    parser.add_argument("--stop-on-first-fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json")
    parser.add_argument("--output-dir", default=".remote-logs/kv_cache_concurrency_sweep")
    args = parser.parse_args()
    args.model_path = os.path.expanduser(args.model_path)
    if args.case_mode:
        raise SystemExit(_case_main(args))
    raise SystemExit(_driver_main(args))


if __name__ == "__main__":
    main()
