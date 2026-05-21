"""Direct correctness check for the K-int8/V-FP8 native paged attention kernel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm.layers.fp8_paged_attention import k_int8_v_fp8_paged_attention_decode
from nanovllm.layers.kv_cache_kernels import store_kvcache_k_int8_v_fp8_triton


def _make_block_table(num_blocks: int, shuffle: bool, device: torch.device) -> torch.Tensor:
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device)
    if shuffle:
        block_table = block_table[torch.randperm(num_blocks, device=device)]
    return block_table


def _reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_len: int,
    softmax_scale: float,
) -> torch.Tensor:
    block_size = k_cache.shape[1]
    num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    head_group = num_heads // num_kv_heads
    positions = torch.arange(context_len, device=q.device)
    logical_blocks = positions // block_size
    block_offsets = positions % block_size
    physical_blocks = block_table[logical_blocks].long()
    k = k_cache[physical_blocks, block_offsets].float()
    v = v_cache[physical_blocks, block_offsets].float()
    k_scales = k_scale_cache[physical_blocks, block_offsets].float()
    v_scales = v_scale_cache[physical_blocks, block_offsets].float()
    k_group_size = head_dim // k_scales.shape[-1]
    k = (k.reshape(context_len, num_kv_heads, k_scales.shape[-1], k_group_size) * k_scales.unsqueeze(-1)).reshape(
        context_len, num_kv_heads, head_dim
    )
    v = v * v_scales.unsqueeze(-1)

    out = torch.empty((num_heads, head_dim), dtype=torch.float32, device=q.device)
    qf = q.float()
    for head_idx in range(num_heads):
        kv_head_idx = head_idx // head_group
        scores = (k[:, kv_head_idx] * qf[head_idx]).sum(dim=-1) * softmax_scale
        probs = torch.softmax(scores, dim=0)
        out[head_idx] = probs @ v[:, kv_head_idx]
    return out.to(q.dtype)


def _compare(native: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    native_f = native.float()
    ref_f = reference.float()
    diff = (native_f - ref_f).abs()
    cos = F.cosine_similarity(native_f, ref_f, dim=-1)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "cosine_min": float(cos.min().item()),
        "cosine_mean": float(cos.mean().item()),
        "allclose_atol_1e_2_rtol_1e_2": bool(torch.allclose(native_f, ref_f, atol=1.0e-2, rtol=1.0e-2)),
    }


def run_case(args: argparse.Namespace, context_len: int, block_tokens: int) -> dict[str, Any]:
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    block_size = args.block_size
    num_blocks = (context_len + block_size - 1) // block_size
    num_kv_heads = args.num_kv_heads
    num_heads = args.num_heads
    head_dim = args.head_dim
    k_groups = head_dim // args.k_group_size
    assert head_dim % args.k_group_size == 0
    dtype = getattr(torch, args.dtype)

    q = torch.randn((num_heads, head_dim), device=device, dtype=dtype)
    k_cache = torch.randint(
        -127,
        128,
        (num_blocks, block_size, num_kv_heads, head_dim),
        device=device,
        dtype=torch.int8,
    )
    k_scale_cache = (torch.rand((num_blocks, block_size, num_kv_heads, k_groups), device=device) * 0.03 + 0.001).to(dtype)
    v = torch.randn((num_blocks, block_size, num_kv_heads, head_dim), device=device, dtype=torch.float32) * 2.0
    v_scale_cache = (torch.rand((num_blocks, block_size, num_kv_heads), device=device) * 0.03 + 0.001).to(dtype)
    v_cache = (v / v_scale_cache.float().unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    block_table = _make_block_table(num_blocks, args.shuffle_blocks, device)
    context_len_t = torch.tensor([context_len], dtype=torch.int32, device=device)

    native = k_int8_v_fp8_paged_attention_decode(
        q,
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        block_table,
        context_len_t,
        args.softmax_scale,
        block_tokens=block_tokens,
    )
    ref = _reference(q, k_cache, v_cache, k_scale_cache, v_scale_cache, block_table, context_len, args.softmax_scale)
    torch.cuda.synchronize()
    return {
        "context_len": context_len,
        "block_tokens": block_tokens,
        "num_blocks": num_blocks,
        "shuffle_blocks": args.shuffle_blocks,
        **_compare(native, ref),
    }


def run_store_case(args: argparse.Namespace, context_len: int, block_tokens: int) -> dict[str, Any]:
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    block_size = args.block_size
    num_blocks = (context_len + block_size - 1) // block_size
    num_kv_heads = args.num_kv_heads
    num_heads = args.num_heads
    head_dim = args.head_dim
    k_groups = head_dim // args.k_group_size
    assert head_dim % args.k_group_size == 0
    dtype = getattr(torch, args.dtype)

    q = torch.randn((num_heads, head_dim), device=device, dtype=dtype)
    key = torch.randn((context_len, num_kv_heads, head_dim), device=device, dtype=dtype)
    value = torch.randn((context_len, num_kv_heads, head_dim), device=device, dtype=dtype)
    block_table = _make_block_table(num_blocks, args.shuffle_blocks, device)
    positions = torch.arange(context_len, device=device)
    logical_blocks = positions // block_size
    block_offsets = positions % block_size
    slot_mapping = (block_table[logical_blocks].long() * block_size + block_offsets).to(torch.int32)

    k_cache = torch.empty((num_blocks, block_size, num_kv_heads, head_dim), device=device, dtype=torch.int8)
    v_cache = torch.empty((num_blocks, block_size, num_kv_heads, head_dim), device=device, dtype=torch.float8_e4m3fn)
    k_scale_cache = torch.empty((num_blocks, block_size, num_kv_heads, k_groups), device=device, dtype=dtype)
    v_scale_cache = torch.empty((num_blocks, block_size, num_kv_heads), device=device, dtype=dtype)
    store_kvcache_k_int8_v_fp8_triton(key, value, k_cache, v_cache, k_scale_cache, v_scale_cache, slot_mapping)

    context_len_t = torch.tensor([context_len], dtype=torch.int32, device=device)
    native = k_int8_v_fp8_paged_attention_decode(
        q,
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        block_table,
        context_len_t,
        args.softmax_scale,
        block_tokens=block_tokens,
    )
    ref = _reference(q, k_cache, v_cache, k_scale_cache, v_scale_cache, block_table, context_len, args.softmax_scale)
    torch.cuda.synchronize()
    return {
        "context_len": context_len,
        "block_tokens": block_tokens,
        "num_blocks": num_blocks,
        "shuffle_blocks": args.shuffle_blocks,
        "include_store": True,
        **_compare(native, ref),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-lens", default="1024,8192")
    parser.add_argument("--block-tokens", default="16,32,64")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-heads", type=int, default=28)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--k-group-size", type=int, default=32)
    parser.add_argument("--softmax-scale", type=float, default=0.08838834764831845)
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16"))
    parser.add_argument("--shuffle-blocks", action="store_true")
    parser.add_argument("--include-store", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    context_lens = [int(x) for x in args.context_lens.split(",") if x]
    block_tokens = [int(x) for x in args.block_tokens.split(",") if x]
    run = run_store_case if args.include_store else run_case
    records = [run(args, context_len, bt) for context_len in context_lens for bt in block_tokens]
    summary = {
        "seed": args.seed,
        "dtype": args.dtype,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "block_size": args.block_size,
        "include_store": args.include_store,
        "records": records,
    }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
