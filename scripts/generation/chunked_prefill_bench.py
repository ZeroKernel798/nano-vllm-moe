from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
import torch.nn.functional as F
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm import LLM, SamplingParams


@dataclass
class StepRecord:
    step: int
    phase: str
    elapsed_ms: float
    num_tokens: int
    outputs: list[tuple[int, int]]


def make_prompt(token_count: int, vocab_size: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(10, vocab_size - 1) for _ in range(token_count)]


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_llm(args: argparse.Namespace, chunk_size: int, policy: str) -> LLM:
    return LLM(
        args.model_path,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=chunk_size,
        max_num_seqs=args.max_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        chunked_prefill_policy=policy,
    )


def get_model_max_len(model_path: str) -> int:
    return int(AutoConfig.from_pretrained(model_path).max_position_embeddings)


def choose_phase3_b_prompt_tokens(args: argparse.Namespace) -> int:
    if args.phase3_b_prompt_tokens > 0:
        return args.phase3_b_prompt_tokens
    model_max_len = get_model_max_len(args.model_path)
    configured_max_len = model_max_len if args.max_model_len <= 0 else min(args.max_model_len, model_max_len)
    reserve = args.phase3_a_prompt_tokens + args.phase3_a_output_tokens + args.phase3_b_output_tokens
    usable = max(1, configured_max_len - reserve)
    block = args.auto_b_prompt_block_size
    if block > 1:
        usable = max(block, (usable // block) * block)
    if args.auto_b_prompt_max_tokens > 0:
        usable = min(usable, args.auto_b_prompt_max_tokens)
    if args.auto_b_prompt_min_tokens > 0 and usable < args.auto_b_prompt_min_tokens:
        raise ValueError(
            f"auto-selected B prompt {usable} is below --auto-b-prompt-min-tokens "
            f"{args.auto_b_prompt_min_tokens}; configured_max_len={configured_max_len}, reserve={reserve}"
        )
    return usable


def release_llm(llm: LLM) -> None:
    llm.exit()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_engine_trace(llm: LLM) -> tuple[list[StepRecord], dict[int, list[int]], float | None, float, float]:
    records: list[StepRecord] = []
    outputs: dict[int, list[int]] = {}
    first_output_ms = None
    start = perf_counter()
    step = 0
    while not llm.is_finished():
        synchronize()
        t0 = perf_counter()
        step_outputs, num_tokens = llm.step()
        synchronize()
        elapsed_ms = (perf_counter() - t0) * 1000
        phase = "prefill" if num_tokens > 0 else "decode"
        for seq_id, token_ids in step_outputs:
            outputs[seq_id] = list(token_ids)
            if first_output_ms is None:
                first_output_ms = (perf_counter() - start) * 1000
        records.append(
            StepRecord(
                step=step,
                phase=phase,
                elapsed_ms=elapsed_ms,
                num_tokens=num_tokens,
                outputs=[(seq_id, len(token_ids)) for seq_id, token_ids in step_outputs],
            )
        )
        step += 1
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    return records, outputs, first_output_ms, (perf_counter() - start) * 1000, peak_memory_gb


def run_engine_trace_with_logits(llm: LLM) -> tuple[list[StepRecord], dict[int, list[int]], torch.Tensor | None, float | None, float, float]:
    records: list[StepRecord] = []
    outputs: dict[int, list[int]] = {}
    last_logits = None
    first_output_ms = None
    start = perf_counter()
    step = 0
    while not llm.is_finished():
        seqs, is_prefill = llm.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        synchronize()
        t0 = perf_counter()
        token_ids, logits = llm.model_runner.call("run_with_logits", seqs, is_prefill)
        synchronize()
        if logits is not None:
            last_logits = logits
        llm.scheduler.postprocess(seqs, token_ids, is_prefill)
        step_outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        for seq_id, output_ids in step_outputs:
            outputs[seq_id] = list(output_ids)
            if first_output_ms is None:
                first_output_ms = (perf_counter() - start) * 1000
        records.append(
            StepRecord(
                step=step,
                phase="prefill" if num_tokens > 0 else "decode",
                elapsed_ms=(perf_counter() - t0) * 1000,
                num_tokens=num_tokens,
                outputs=[(seq_id, len(output_ids)) for seq_id, output_ids in step_outputs],
            )
        )
        step += 1
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    return records, outputs, last_logits, first_output_ms, (perf_counter() - start) * 1000, peak_memory_gb


def prefill_step_count(records: list[StepRecord]) -> int:
    return sum(1 for record in records if record.phase == "prefill")


def first_decode_ms(records: list[StepRecord]) -> float | None:
    total = 0.0
    for record in records:
        total += record.elapsed_ms
        if record.phase == "decode":
            return total
    return None


def phase1(args: argparse.Namespace) -> dict:
    prompt = make_prompt(args.phase1_tokens, args.vocab_size, args.seed)
    params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)

    chunked = build_llm(args, args.phase1_chunk, "prefill_first")
    chunked.add_request(prompt, params)
    chunked_records, chunked_outputs, chunked_logits, chunked_ttft_ms, chunked_wall_ms, chunked_peak_gb = run_engine_trace_with_logits(chunked)
    release_llm(chunked)

    baseline = build_llm(args, args.phase1_tokens, "prefill_first")
    baseline.add_request(prompt, params)
    baseline_records, baseline_outputs, baseline_logits, baseline_ttft_ms, baseline_wall_ms, baseline_peak_gb = run_engine_trace_with_logits(baseline)
    release_llm(baseline)

    cosine = None
    if chunked_logits is not None and baseline_logits is not None:
        cosine = float(F.cosine_similarity(chunked_logits[0], baseline_logits[0], dim=0).item())
    chunked_result = next(iter(chunked_outputs.values()), None)
    baseline_result = next(iter(baseline_outputs.values()), None)
    token_match = chunked_result == baseline_result

    return {
        "phase": "phase1_smoke",
        "tokens": args.phase1_tokens,
        "chunk": args.phase1_chunk,
        "expected_prefill_steps": (args.phase1_tokens + args.phase1_chunk - 1) // args.phase1_chunk,
        "chunked_prefill_steps": prefill_step_count(chunked_records),
        "baseline_prefill_steps": prefill_step_count(baseline_records),
        "chunked_ttft_ms": chunked_ttft_ms,
        "baseline_ttft_ms": baseline_ttft_ms,
        "chunked_wall_ms": chunked_wall_ms,
        "baseline_wall_ms": baseline_wall_ms,
        "chunked_peak_memory_gb": chunked_peak_gb,
        "baseline_peak_memory_gb": baseline_peak_gb,
        "output_token_match": token_match,
        "logits_cosine": cosine,
        "pass": prefill_step_count(chunked_records) == (args.phase1_tokens + args.phase1_chunk - 1) // args.phase1_chunk and bool(token_match) and (cosine is None or cosine > args.phase1_min_cosine),
    }


def phase2(args: argparse.Namespace) -> list[dict]:
    rows = []
    prompt = make_prompt(args.phase2_tokens, args.vocab_size, args.seed + 100)
    params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    for chunk in args.phase2_chunks:
        llm = build_llm(args, chunk, "prefill_first")
        llm.add_request(prompt, params)
        records, _, ttft_ms, wall_ms, peak_memory_gb = run_engine_trace(llm)
        rows.append(
            {
                "phase": "phase2_baseline",
                "tokens": args.phase2_tokens,
                "chunk": chunk,
                "policy": "prefill_first",
                "prefill_steps": prefill_step_count(records),
                "ttft_ms": ttft_ms,
                "wall_ms": wall_ms,
                "peak_memory_gb": peak_memory_gb,
            }
        )
        release_llm(llm)
    return rows


def summarize_itl(token_times_ms: list[float]) -> dict:
    if len(token_times_ms) < 2:
        return {"avg_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None, "count": 0}
    intervals = [b - a for a, b in zip(token_times_ms, token_times_ms[1:])]
    sorted_intervals = sorted(intervals)

    def percentile(q: float) -> float:
        idx = min(len(sorted_intervals) - 1, int(q * (len(sorted_intervals) - 1)))
        return sorted_intervals[idx]

    return {
        "avg_ms": mean(intervals),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": max(intervals),
        "count": len(intervals),
    }


def phase3_one(args: argparse.Namespace, chunk: int, policy: str) -> dict:
    llm = build_llm(args, chunk, policy)
    params_a = SamplingParams(temperature=0.0, max_tokens=args.phase3_a_output_tokens, ignore_eos=True)
    params_b = SamplingParams(temperature=0.0, max_tokens=args.phase3_b_output_tokens, ignore_eos=True)
    prompt_a = make_prompt(args.phase3_a_prompt_tokens, args.vocab_size, args.seed + 200)
    prompt_b = make_prompt(args.phase3_b_prompt_tokens_resolved, args.vocab_size, args.seed + 300)

    llm.add_request(prompt_a, params_a)
    seq_a_id = llm.scheduler.waiting[-1].seq_id
    injected = False
    seq_b_id = None
    seq_a_token_times: list[float] = []
    seq_b_ttft_ms = None
    seq_b_inject_ms = None
    start = perf_counter()
    step_records: list[StepRecord] = []
    step = 0

    while not llm.is_finished():
        synchronize()
        t0 = perf_counter()
        outputs, num_tokens = llm.step()
        synchronize()
        now_ms = (perf_counter() - start) * 1000
        elapsed_ms = (perf_counter() - t0) * 1000
        phase = "prefill" if num_tokens > 0 else "decode"
        step_records.append(
            StepRecord(step, phase, elapsed_ms, num_tokens, [(seq_id, len(token_ids)) for seq_id, token_ids in outputs])
        )
        for seq_id, token_ids in outputs:
            if seq_id == seq_a_id and len(token_ids) > len(seq_a_token_times):
                seq_a_token_times.extend([now_ms] * (len(token_ids) - len(seq_a_token_times)))
            if seq_b_id is not None and seq_id == seq_b_id and token_ids and seq_b_ttft_ms is None:
                seq_b_ttft_ms = now_ms - seq_b_inject_ms if seq_b_inject_ms is not None else now_ms
        for seq in llm.scheduler.running:
            if seq.seq_id == seq_a_id and seq.num_completion_tokens > len(seq_a_token_times):
                seq_a_token_times.extend([now_ms] * (seq.num_completion_tokens - len(seq_a_token_times)))
            if seq_b_id is not None and seq.seq_id == seq_b_id and seq.num_completion_tokens > 0 and seq_b_ttft_ms is None:
                seq_b_ttft_ms = now_ms - seq_b_inject_ms if seq_b_inject_ms is not None else now_ms
        if not injected and len(seq_a_token_times) >= args.phase3_inject_after_tokens:
            llm.add_request(prompt_b, params_b)
            seq_b_id = llm.scheduler.waiting[-1].seq_id
            seq_b_inject_ms = now_ms
            injected = True
        step += 1

    itl = summarize_itl(seq_a_token_times)
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    release_llm(llm)
    return {
        "phase": "phase3_interference",
        "chunk": chunk,
        "policy": policy,
        "seq_a_prompt_tokens": args.phase3_a_prompt_tokens,
        "seq_a_output_tokens": args.phase3_a_output_tokens,
        "seq_b_prompt_tokens": args.phase3_b_prompt_tokens_resolved,
        "seq_b_output_tokens": args.phase3_b_output_tokens,
        "inject_after_tokens": args.phase3_inject_after_tokens,
        "seq_a_itl_avg_ms": itl["avg_ms"],
        "seq_a_itl_p95_ms": itl["p95_ms"],
        "seq_a_itl_p99_ms": itl["p99_ms"],
        "seq_a_itl_max_ms": itl["max_ms"],
        "seq_a_itl_count": itl["count"],
        "seq_b_ttft_ms": seq_b_ttft_ms,
        "prefill_steps": prefill_step_count(step_records),
        "total_steps": len(step_records),
        "peak_memory_gb": peak_memory_gb,
    }


def phase3(args: argparse.Namespace) -> list[dict]:
    rows = [] if args.phase3_skip_no_chunk else [phase3_one(args, args.phase3_no_chunk, "prefill_first")]
    for chunk in args.phase3_chunks:
        rows.append(phase3_one(args, chunk, "prefill_first"))
        rows.append(phase3_one(args, chunk, "decode_first"))
    return rows


def write_outputs(rows: list[dict], output_json: Path | None, output_csv: Path | None) -> None:
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-phase chunked prefill benchmark")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--phases", default="1,2,3", help="Comma-separated phases: 1,2,3")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Use 0 to select the HF config max_position_embeddings")
    parser.add_argument("--max-seqs", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phase1-tokens", type=int, default=800)
    parser.add_argument("--phase1-chunk", type=int, default=256)
    parser.add_argument("--phase1-min-cosine", type=float, default=0.99)
    parser.add_argument("--phase2-tokens", type=int, default=2048)
    parser.add_argument("--phase2-chunks", type=lambda s: [int(x) for x in s.split(",")], default="2048,1024,512,256,128")
    parser.add_argument("--phase3-a-prompt-tokens", type=int, default=20)
    parser.add_argument("--phase3-a-output-tokens", type=int, default=100)
    parser.add_argument("--phase3-b-prompt-tokens", type=int, default=2000, help="Use 0 to auto-select the largest safe prompt below max_model_len")
    parser.add_argument("--phase3-b-output-tokens", type=int, default=1)
    parser.add_argument("--auto-b-prompt-block-size", type=int, default=256)
    parser.add_argument("--auto-b-prompt-max-tokens", type=int, default=0)
    parser.add_argument("--auto-b-prompt-min-tokens", type=int, default=0)
    parser.add_argument("--phase3-inject-after-tokens", type=int, default=20)
    parser.add_argument("--phase3-no-chunk", type=int, default=2048)
    parser.add_argument("--phase3-skip-no-chunk", action="store_true")
    parser.add_argument("--phase3-chunk", type=int, default=512)
    parser.add_argument("--phase3-chunks", type=lambda s: [int(x) for x in s.split(",")], default="512")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.phase3_b_prompt_tokens_resolved = choose_phase3_b_prompt_tokens(args)
    selected = {phase.strip() for phase in args.phases.split(",") if phase.strip()}
    rows: list[dict] = []
    if "1" in selected:
        rows.append(phase1(args))
    if "2" in selected:
        rows.extend(phase2(args))
    if "3" in selected:
        rows.extend(phase3(args))
    write_outputs(rows, args.output_json, args.output_csv)


if __name__ == "__main__":
    main()
