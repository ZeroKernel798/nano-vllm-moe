from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm.executor.moe.config import make_moe_parallel_config
from nanovllm.executor.moe.prepare_finalize import TorchAllToAllPrepareFinalize


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_backends(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_inputs(
    *,
    rank: int,
    device: torch.device,
    num_tokens: int,
    hidden_size: int,
    top_k: int,
    global_num_experts: int,
    seed: int,
    imbalance: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + rank * 1009 + num_tokens)
    x_cpu = torch.randn(num_tokens, hidden_size, generator=generator, dtype=torch.float32) * 0.1 + rank
    if imbalance == "balanced":
        ids = torch.arange(num_tokens * top_k, dtype=torch.long).reshape(num_tokens, top_k) % global_num_experts
        ids = (ids + rank) % global_num_experts
    elif imbalance == "skewed":
        hot_experts = max(1, global_num_experts // 4)
        ids = torch.randint(0, hot_experts, (num_tokens, top_k), generator=generator, dtype=torch.long)
    else:
        ids = torch.randint(0, global_num_experts, (num_tokens, top_k), generator=generator, dtype=torch.long)
    weights = torch.rand(num_tokens, top_k, generator=generator, dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return x_cpu.to(device), weights.to(device), ids.to(device)


def make_backend(name: str, parallel_config):
    if name == "torch":
        return TorchAllToAllPrepareFinalize(parallel_config)
    raise ValueError(f"Unsupported backend={name!r}")

def run_one_iteration(prepare_finalize, x, topk_weights, topk_ids, model_dtype: torch.dtype, device: torch.device):
    sync_device(device)
    t0 = time.perf_counter()
    prepared = prepare_finalize.prepare(x, topk_weights, topk_ids)
    sync_device(device)
    t1 = time.perf_counter()
    expert_out = prepared.hidden_states.to(model_dtype) * prepared.topk_weights[:, None].to(model_dtype)
    expert_out = expert_out + prepared.topk_ids.to(model_dtype)[:, None]
    sync_device(device)
    t2 = time.perf_counter()
    actual = prepare_finalize.finalize(
        expert_out,
        prepared,
        output_shape=x.shape,
        model_dtype=model_dtype,
        reduce_tp=False,
    )
    sync_device(device)
    t3 = time.perf_counter()
    local_ids = topk_ids % prepare_finalize.parallel_config.local_num_experts
    expected = (x.to(model_dtype)[:, None, :] * topk_weights[:, :, None].to(model_dtype))
    expected = expected + local_ids.to(model_dtype)[:, :, None]
    expected = expected.sum(dim=1)
    max_abs = float((actual - expected).abs().max().item())
    ctx = prepared.ctx
    return {
        "prepare_ms": (t1 - t0) * 1000.0,
        "synthetic_expert_ms": (t2 - t1) * 1000.0,
        "finalize_ms": (t3 - t2) * 1000.0,
        "total_ms": (t3 - t0) * 1000.0,
        "max_abs": max_abs,
        "send_counts": list(ctx.get("s_list", [])),
        "recv_counts": list(ctx.get("r_list", [])),
    }


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "stdev": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize_records(records: list[dict]) -> dict:
    active = [item for item in records if not item["discarded"]]
    out = {"ok_runs": len(active), "discarded_runs": len(records) - len(active), "total_runs": len(records)}
    for metric in ["prepare_ms", "synthetic_expert_ms", "finalize_ms", "total_ms", "max_abs"]:
        out[metric] = summarize([float(item[metric]) for item in active])
    if active:
        send_totals = [sum(item["send_counts"]) for item in active]
        recv_totals = [sum(item["recv_counts"]) for item in active]
        out["send_total"] = summarize(send_totals)
        out["recv_total"] = summarize(recv_totals)
        out["send_counts_sample"] = active[-1]["send_counts"]
        out["recv_counts_sample"] = active[-1]["recv_counts"]
    return out


def worker(rank: int, args, init_method: str, queue) -> None:
    world_size = args.ep_size
    backend = "nccl" if args.device == "cuda" else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size, init_method=init_method)
    try:
        if args.device == "cuda":
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
            model_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        else:
            device = torch.device("cpu")
            model_dtype = torch.float32
        parallel_config = make_moe_parallel_config(
            tp_group=None,
            ep_group=dist.group.WORLD,
            global_num_experts=args.global_num_experts,
            intermediate_size=args.intermediate_size,
        )
        local_payload = []
        for num_tokens in parse_csv_ints(args.num_tokens):
            x, topk_weights, topk_ids = make_inputs(
                rank=rank,
                device=device,
                num_tokens=num_tokens,
                hidden_size=args.hidden_size,
                top_k=args.top_k,
                global_num_experts=args.global_num_experts,
                seed=args.seed,
                imbalance=args.imbalance,
            )
            for backend_name in parse_backends(args.backends):
                prepare_finalize = make_backend(backend_name, parallel_config)
                records = []
                for repeat_idx in range(args.repeat):
                    record = run_one_iteration(prepare_finalize, x, topk_weights, topk_ids, model_dtype, device)
                    record.update(
                        {
                            "rank": rank,
                            "backend": backend_name,
                            "num_tokens": num_tokens,
                            "repeat": repeat_idx,
                            "discarded": repeat_idx < args.discard_first,
                        }
                    )
                    records.append(record)
                local_payload.append(
                    {
                        "rank": rank,
                        "backend": backend_name,
                        "num_tokens": num_tokens,
                        "summary": summarize_records(records),
                        "records": records if args.keep_records else [],
                    }
                )
        gathered = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_payload, gathered, dst=0)
        if rank == 0:
            queue.put(gathered)
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated EP prepare/finalize microbench")
    parser.add_argument("--backends", type=str, default="torch")
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--num-tokens", type=str, default="4,64,256")
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--global-num-experts", type=int, default=60)
    parser.add_argument("--intermediate-size", type=int, default=1408)
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--discard-first", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--imbalance", choices=["random", "balanced", "skewed"], default="random")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--keep-records", action="store_true")
    args = parser.parse_args()

    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if args.discard_first < 0 or args.discard_first >= args.repeat:
        raise ValueError("--discard-first must be >= 0 and < --repeat")
    if args.global_num_experts % args.ep_size != 0:
        raise ValueError("--global-num-experts must be divisible by --ep-size")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    init_method = f"tcp://127.0.0.1:{find_free_port()}"
    mp.spawn(worker, args=(args, init_method, queue), nprocs=args.ep_size, join=True)
    gathered = queue.get()
    rank_payloads = [item for rank_items in gathered for item in rank_items]
    payload = {
        "config": vars(args),
        "rank_payloads": rank_payloads,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    for item in rank_payloads:
        summary = item["summary"]
        print(
            f"rank={item['rank']} backend={item['backend']} tokens={item['num_tokens']} "
            f"prepare={summary['prepare_ms']['mean']:.4f}ms finalize={summary['finalize_ms']['mean']:.4f}ms "
            f"total={summary['total_ms']['mean']:.4f}ms max_abs={summary['max_abs']['max']:.3e}"
        )


if __name__ == "__main__":
    main()
