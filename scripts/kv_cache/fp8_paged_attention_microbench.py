from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm.layers.fp8_paged_attention import fp8_paged_attention_decode
from nanovllm.layers.kv_cache_kernels import dequant_kvcache_fp8_gather_decode, store_kvcache_fp8


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn, repeat: int, warmup: int):
    for _ in range(warmup):
        out = fn()
    _sync()
    times: list[float] = []
    out = None
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        _sync()
        times.append(float(start.elapsed_time(end)))
    return out, times


def _make_fp8_cache(args: argparse.Namespace):
    dtype = getattr(torch, args.dtype)
    num_blocks = (args.context_len + args.block_size - 1) // args.block_size
    total_tokens = num_blocks * args.block_size
    k = torch.randn(total_tokens, args.num_kv_heads, args.head_dim, dtype=dtype, device="cuda")
    v = torch.randn_like(k)
    k_cache = torch.empty(num_blocks, args.block_size, args.num_kv_heads, args.head_dim, dtype=torch.float8_e4m3fn, device="cuda")
    v_cache = torch.empty_like(k_cache)
    k_scale = torch.empty(num_blocks, args.block_size, args.num_kv_heads, dtype=torch.float16, device="cuda")
    v_scale = torch.empty_like(k_scale)
    slot_mapping = torch.arange(total_tokens, dtype=torch.int64, device="cuda")
    store_kvcache_fp8(k, v, k_cache, v_cache, k_scale, v_scale, slot_mapping)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda")
    context_len = torch.tensor([args.context_len], dtype=torch.int32, device="cuda")
    return k_cache, v_cache, k_scale, v_scale, block_table, context_len


def main() -> None:
    parser = argparse.ArgumentParser(description="Native FP8 paged attention microbench")
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--block-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16"))
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.set_default_device("cuda")
    dtype = getattr(torch, args.dtype)
    k_cache, v_cache, k_scale, v_scale, block_table, context_len = _make_fp8_cache(args)
    q = torch.randn(args.num_heads, args.head_dim, dtype=dtype, device="cuda")
    scale = args.head_dim ** -0.5

    def native():
        return fp8_paged_attention_decode(q, k_cache, v_cache, k_scale, v_scale, block_table, context_len, scale, args.block_tokens)

    def gather_ref():
        k_ref, v_ref = dequant_kvcache_fp8_gather_decode(
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_table.view(1, -1),
            context_len,
            dtype,
        )
        kv_group = args.num_heads // args.num_kv_heads
        k_expanded = k_ref[0].repeat_interleave(kv_group, dim=1).transpose(0, 1).float()
        v_expanded = v_ref[0].repeat_interleave(kv_group, dim=1).transpose(0, 1).float()
        scores = torch.einsum("hd,hld->hl", q.float(), k_expanded) * scale
        probs = torch.softmax(scores, dim=-1)
        return torch.einsum("hl,hld->hd", probs, v_expanded).to(dtype)

    native_out, native_times = _time(native, args.repeat, args.warmup)
    ref_out, ref_times = _time(gather_ref, args.repeat, args.warmup)
    diff = (native_out.float() - ref_out.float()).abs()
    summary = {
        "context_len": args.context_len,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "block_tokens": args.block_tokens,
        "native_avg_ms": sum(native_times) / len(native_times),
        "ref_avg_ms": sum(ref_times) / len(ref_times),
        "cosine": float(F.cosine_similarity(native_out.float().flatten(), ref_out.float().flatten(), dim=0).item()),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "native_times_ms": native_times,
        "ref_times_ms": ref_times,
    }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
