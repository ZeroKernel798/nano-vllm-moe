"""Single-process KV cache memory measurement.

Run one KV cache mode per process so BF16 and quantized KV measurements do not
share CUDA allocator state. The workload is intentionally fixed-shape: same
model, prompt length, output length, and request count across modes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm import LLM, SamplingParams


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


def _nvidia_smi_memory() -> list[dict[str, str]]:
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory,name",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError:
        return []
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) == 3:
            rows.append({"pid": parts[0], "used_memory_mib": parts[1], "name": parts[2]})
    return rows


def _cache_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size()) if tensor.numel() else 0


def _kv_storage(llm: LLM) -> dict[str, Any]:
    runner = llm.model_runner
    kv_cache = runner.kv_cache
    kv_data_bytes = _cache_bytes(kv_cache)
    if kv_data_bytes == 0:
        kv_data_bytes = _cache_bytes(runner.k_cache_storage) + _cache_bytes(runner.v_cache_storage)
    scale_bytes = _cache_bytes(runner.k_scale_cache) + _cache_bytes(runner.v_scale_cache)
    num_blocks = int(runner.config.num_kvcache_blocks)
    return {
        "num_kvcache_blocks": num_blocks,
        "kvcache_block_size": int(runner.config.kvcache_block_size),
        "kv_cache_shape": list(kv_cache.shape),
        "k_cache_shape": list(getattr(runner, "k_cache_storage", kv_cache).shape),
        "v_cache_shape": list(getattr(runner, "v_cache_storage", kv_cache).shape),
        "kv_cache_dtype": str(kv_cache.dtype),
        "k_cache_dtype": str(getattr(runner, "k_cache_storage", kv_cache).dtype),
        "v_cache_dtype": str(getattr(runner, "v_cache_storage", kv_cache).dtype),
        "kv_cache_data_storage_bytes": kv_data_bytes,
        "kv_cache_scale_storage_bytes": scale_bytes,
        "kv_cache_total_storage_bytes": kv_data_bytes + scale_bytes,
        "kv_cache_total_bytes_per_block": (kv_data_bytes + scale_bytes) / max(num_blocks, 1),
        "free_kvcache_blocks_after_generate": len(llm.scheduler.block_manager.free_block_ids),
        "used_kvcache_blocks_after_generate": len(llm.scheduler.block_manager.used_block_ids),
    }


def _make_prompts(args: argparse.Namespace) -> list[list[int]]:
    prompts = []
    for seq_idx in range(args.num_seqs):
        rng = random.Random(args.seed * 1000003 + seq_idx * 9176)
        prompts.append([rng.randint(0, args.vocab_range) for _ in range(args.input_len)])
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-mode KV cache memory case")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mode", default="bf16", choices=("bf16", "k_int8_v_fp8", "fp8"))
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=8192)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--num-kvcache-blocks", type=int, default=-1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--kv-cache-scale-dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--fp8-decode-backend", default="native", choices=("native", "gather_dequant"))
    parser.add_argument("--k-group-size", type=int, default=32)
    parser.add_argument("--native-block-tokens", type=int, default=32, choices=(16, 32, 64))
    parser.add_argument("--gather-block-tokens", type=int, default=16, choices=(1, 2, 4, 8, 16, 32, 64))
    parser.add_argument("--chunked-prefill-policy", default="prefill_first", choices=("prefill_first", "decode_first"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-range", type=int, default=10000)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    args.model_path = str(Path(args.model_path).expanduser())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    old_env = {
        "NANOVLLM_FP8_KV_DECODE": os.environ.get("NANOVLLM_FP8_KV_DECODE"),
        "NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS": os.environ.get("NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS"),
        "NANOVLLM_K_INT8_GROUP_SIZE": os.environ.get("NANOVLLM_K_INT8_GROUP_SIZE"),
        "NANOVLLM_K_INT8_V_FP8_GATHER_BLOCK_TOKENS": os.environ.get("NANOVLLM_K_INT8_V_FP8_GATHER_BLOCK_TOKENS"),
    }
    if args.mode != "bf16":
        os.environ["NANOVLLM_FP8_KV_DECODE"] = args.fp8_decode_backend
        os.environ["NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS"] = str(args.native_block_tokens)
        os.environ["NANOVLLM_K_INT8_GROUP_SIZE"] = str(args.k_group_size)
        os.environ["NANOVLLM_K_INT8_V_FP8_GATHER_BLOCK_TOKENS"] = str(args.gather_block_tokens)

    llm = None
    result: dict[str, Any] = {"args": vars(args), "mode": args.mode, "pid": os.getpid()}
    try:
        result["memory_before_load"] = _cuda_memory()
        result["nvidia_smi_before_load"] = _nvidia_smi_memory()
        llm = LLM(
            args.model_path,
            enforce_eager=True,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            num_kvcache_blocks=args.num_kvcache_blocks,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tp_size=1,
            ep_size=1,
            kv_cache_dtype=args.mode,
            experimental_kv_cache_fp8=args.mode != "bf16",
            kv_cache_scale_dtype=args.kv_cache_scale_dtype,
            chunked_prefill_policy=args.chunked_prefill_policy,
        )
        result["memory_after_load"] = _cuda_memory()
        result["nvidia_smi_after_load"] = _nvidia_smi_memory()
        result["kv_storage_after_load"] = _kv_storage(llm)

        prompts = _make_prompts(args)
        sampling_params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.output_len)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        output = llm.generate(prompts, sampling_params, use_tqdm=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result["generate_wall_time_s"] = perf_counter() - start
        result["memory_after_generate"] = _cuda_memory()
        result["nvidia_smi_after_generate"] = _nvidia_smi_memory()
        result["kv_storage_after_generate"] = _kv_storage(llm)
        result["generated_sequences"] = len(output["results"])
        result["generated_tokens"] = sum(len(row) for row in output["results"])
        result["stats"] = output["stats"]
        result["ok"] = True
    except Exception as exc:
        result.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "memory_after_error": _cuda_memory(),
                "nvidia_smi_after_error": _nvidia_smi_memory(),
            }
        )
        raise
    finally:
        if llm is not None:
            llm.exit()
            del llm
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
