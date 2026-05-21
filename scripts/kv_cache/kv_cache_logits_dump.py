"""Dump one backend logits trace to a torch checkpoint for offline comparison."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context


BACKENDS = {"bf16", "native", "gather_dequant"}


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


def _make_prompt(seed: int, input_len: int, vocab_range: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(0, vocab_range) for _ in range(input_len)]


def run(args: argparse.Namespace, prompt_token_ids: list[int]) -> tuple[dict[str, Any], torch.Tensor]:
    backend = args.backend
    experimental = backend != "bf16"
    old_decode_backend = os.environ.get("NANOVLLM_FP8_KV_DECODE")
    old_block_tokens = os.environ.get("NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS")
    old_group_size = os.environ.get("NANOVLLM_K_INT8_GROUP_SIZE")
    old_gather_block_tokens = os.environ.get("NANOVLLM_K_INT8_V_FP8_GATHER_BLOCK_TOKENS")
    if experimental:
        os.environ["NANOVLLM_FP8_KV_DECODE"] = backend
        os.environ["NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS"] = str(args.native_block_tokens)
        os.environ["NANOVLLM_K_INT8_GROUP_SIZE"] = str(args.k_group_size)
        os.environ["NANOVLLM_K_INT8_V_FP8_GATHER_BLOCK_TOKENS"] = str(args.gather_block_tokens)

    llm = None
    try:
        llm = LLM(
            args.model_path,
            enforce_eager=True,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tp_size=1,
            ep_size=1,
            kv_cache_dtype="k_int8_v_fp8" if experimental else "bf16",
            experimental_kv_cache_fp8=experimental,
            kv_cache_scale_dtype=args.kv_cache_scale_dtype,
        )

        sampling_params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.output_len)
        seq = Sequence(prompt_token_ids, sampling_params)
        llm.scheduler.add(seq)
        llm.model_runner.call("reset_kv_cache_profile")

        step_records: list[dict[str, Any]] = []
        logits_trace: list[torch.Tensor] = []
        generated: list[int] = []
        total_start = perf_counter()

        while not llm.scheduler.is_finished():
            seqs, is_prefill = llm.scheduler.schedule()
            phase = "prefill" if is_prefill else "decode"
            if is_prefill:
                (input_ids, positions), prepare_time = _timed_cuda(lambda: llm.model_runner.prepare_prefill(seqs))
            else:
                (input_ids, positions), prepare_time = _timed_cuda(lambda: llm.model_runner.prepare_decode(seqs))
            logits, model_time = _timed_cuda(lambda: llm.model_runner.run_model(input_ids, positions, is_prefill))
            if logits is None:
                raise RuntimeError("logits are None; this script expects tp_size=ep_size=1")
            temperatures, sample_prepare_time = _timed_cuda(lambda: llm.model_runner.prepare_sample(seqs))
            token_ids, sample_time = _timed_cuda(lambda: llm.model_runner.sampler(logits, temperatures).tolist())

            appended_indices = [
                idx
                for idx, seq in enumerate(seqs)
                if not (is_prefill and seq.num_cached_tokens + seq.num_scheduled_tokens < seq.num_tokens)
            ]
            if appended_indices:
                logits_rows = logits[-len(seqs) :]
                logits_trace.append(logits_rows[appended_indices].detach().float().cpu())
                generated.extend(int(token_ids[idx]) for idx in appended_indices)

            post_start = perf_counter()
            llm.scheduler.postprocess(seqs, token_ids, is_prefill)
            reset_context()
            step_records.append(
                {
                    "phase": phase,
                    "input_tokens": int(input_ids.numel()),
                    "prepare_ms": prepare_time * 1000.0,
                    "model_ms": model_time * 1000.0,
                    "sample_prepare_ms": sample_prepare_time * 1000.0,
                    "sample_ms": sample_time * 1000.0,
                    "postprocess_ms": (perf_counter() - post_start) * 1000.0,
                    "token_ids": [int(token_id) for token_id in token_ids],
                }
            )

        runner = llm.model_runner
        num_blocks = int(runner.config.num_kvcache_blocks)
        kv_data_bytes = _cache_bytes(runner.kv_cache)
        if kv_data_bytes == 0:
            kv_data_bytes = _cache_bytes(runner.k_cache_storage) + _cache_bytes(runner.v_cache_storage)
        scale_bytes = _cache_bytes(runner.k_scale_cache) + _cache_bytes(runner.v_scale_cache)
        if not logits_trace:
            raise RuntimeError("no generation logits were recorded")
        trace = torch.cat(logits_trace, dim=0).contiguous()
        meta = {
            "backend": backend,
            "model_path": args.model_path,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "seed": args.seed,
            "vocab_range": args.vocab_range,
            "native_block_tokens": args.native_block_tokens if experimental else None,
            "k_group_size": args.k_group_size if experimental else None,
            "gather_block_tokens": args.gather_block_tokens if experimental else None,
            "kv_cache_scale_dtype": args.kv_cache_scale_dtype,
            "num_kvcache_blocks": num_blocks,
            "kv_cache_data_storage_bytes": kv_data_bytes,
            "kv_cache_scale_storage_bytes": scale_bytes,
            "kv_cache_total_storage_bytes": kv_data_bytes + scale_bytes,
            "kv_cache_data_bytes_per_block": kv_data_bytes / max(num_blocks, 1),
            "kv_cache_total_bytes_per_block": (kv_data_bytes + scale_bytes) / max(num_blocks, 1),
            "generated_token_ids": generated,
            "steps": step_records,
            "total_time_s": perf_counter() - total_start,
            "logits_shape": list(trace.shape),
            "logits_dtype": str(trace.dtype),
            "kv_cache_profile": llm.model_runner.call("get_kv_cache_profile"),
        }
        return meta, trace
    finally:
        if llm is not None:
            llm.exit()
            del llm
        if old_decode_backend is None:
            os.environ.pop("NANOVLLM_FP8_KV_DECODE", None)
        else:
            os.environ["NANOVLLM_FP8_KV_DECODE"] = old_decode_backend
        if old_block_tokens is None:
            os.environ.pop("NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS", None)
        else:
            os.environ["NANOVLLM_FP8_KV_NATIVE_BLOCK_TOKENS"] = old_block_tokens
        if old_group_size is None:
            os.environ.pop("NANOVLLM_K_INT8_GROUP_SIZE", None)
        else:
            os.environ["NANOVLLM_K_INT8_GROUP_SIZE"] = old_group_size
        if old_gather_block_tokens is None:
            os.environ.pop("NANOVLLM_K_INT8_V_FP8_GATHER_BLOCK_TOKENS", None)
        else:
            os.environ["NANOVLLM_K_INT8_V_FP8_GATHER_BLOCK_TOKENS"] = old_gather_block_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--kv-cache-scale-dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--native-block-tokens", type=int, default=32, choices=(16, 32, 64))
    parser.add_argument("--k-group-size", type=int, default=32)
    parser.add_argument("--gather-block-tokens", type=int, default=16, choices=(1, 2, 4, 8, 16, 32, 64))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-range", type=int, default=10000)
    parser.add_argument("--output-pt", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    args.model_path = os.path.expanduser(args.model_path)

    prompt_token_ids = _make_prompt(args.seed, args.input_len, args.vocab_range)
    meta, trace = run(args, prompt_token_ids)
    out_pt = Path(args.output_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": meta,
            "prompt_token_ids": prompt_token_ids,
            "generated_token_ids": meta["generated_token_ids"],
            "logits": trace,
        },
        out_pt,
    )
    text = json.dumps({**meta, "trace_path": str(out_pt)}, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
