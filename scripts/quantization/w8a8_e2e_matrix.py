from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm import LLM, SamplingParams


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def make_prompts(num_seqs: int, input_len: int, vocab_size: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randint(10, vocab_size - 1) for _ in range(input_len)] for _ in range(num_seqs)]


def output_hash(outputs: list[list[int]]) -> str:
    payload = json.dumps(outputs, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def compare_outputs(reference: list[list[int]] | None, outputs: list[list[int]]) -> dict[str, Any]:
    if reference is None:
        return {
            "output_token_match_vs_bf16": None,
            "output_token_agreement_ratio": None,
            "first_mismatch_index": None,
        }
    total = 0
    match = 0
    first_mismatch = None
    for ref_seq, out_seq in zip(reference, outputs, strict=False):
        limit = min(len(ref_seq), len(out_seq))
        for idx in range(limit):
            total += 1
            if ref_seq[idx] == out_seq[idx]:
                match += 1
            elif first_mismatch is None:
                first_mismatch = idx
        for idx in range(limit, max(len(ref_seq), len(out_seq))):
            total += 1
            if first_mismatch is None:
                first_mismatch = idx
    exact = reference == outputs
    return {
        "output_token_match_vs_bf16": exact,
        "output_token_agreement_ratio": match / total if total else None,
        "first_mismatch_index": first_mismatch,
    }


def peak_memory_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024**3)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_generation_trace(llm: LLM, prompts: list[list[int]], sampling: SamplingParams) -> dict[str, Any]:
    for prompt in prompts:
        llm.add_request(prompt, sampling)

    outputs: dict[int, list[int]] = {}
    prefill_tokens = 0
    prefill_time = 0.0
    decode_tokens = 0
    decode_time = 0.0
    first_token_latency_s = None
    start = perf_counter()
    while not llm.is_finished():
        synchronize()
        step_start = perf_counter()
        step_outputs, num_tokens = llm.step()
        synchronize()
        step_time = perf_counter() - step_start
        if num_tokens > 0:
            prefill_tokens += num_tokens
            prefill_time += step_time
        else:
            decode_tokens += abs(num_tokens)
            decode_time += step_time
            if first_token_latency_s is None:
                first_token_latency_s = perf_counter() - start
        for seq_id, token_ids in step_outputs:
            outputs[seq_id] = list(token_ids)

    wall = perf_counter() - start
    final_outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
    return {
        "outputs": final_outputs,
        "wall_time_s": wall,
        "prefill_tokens": prefill_tokens,
        "prefill_time_s": prefill_time,
        "decode_tokens": decode_tokens,
        "decode_time_s": decode_time,
        "prefill_tok_s": prefill_tokens / max(prefill_time, 1e-6),
        "decode_tok_s": decode_tokens / max(decode_time, 1e-6),
        "first_token_latency_ms": (first_token_latency_s * 1000) if first_token_latency_s is not None else None,
    }


def run_case(
    *,
    model_path: str,
    model_label: str,
    phase: str,
    input_len: int,
    output_len: int,
    args: argparse.Namespace,
    reference_outputs: list[list[int]] | None,
) -> tuple[dict[str, Any], list[list[int]] | None]:
    prompt_seed = args.seed + input_len
    config = AutoConfig.from_pretrained(model_path)
    max_position = int(getattr(config, "max_position_embeddings", input_len + output_len))
    requested_input_len = input_len
    requested_len = input_len + output_len
    if requested_len > max_position:
        if args.fit_input_to_max_position and output_len < max_position:
            input_len = max_position - output_len
            requested_len = max_position
        else:
            return (
                {
                    "model_label": model_label,
                    "model_path": model_path,
                    "phase": phase,
                    "requested_input_len": requested_input_len,
                    "input_len": input_len,
                    "output_len": output_len,
                    "requested_len": requested_len,
                    "max_position_embeddings": max_position,
                    "status": "skipped",
                    "error": f"requested_len={requested_len} exceeds max_position_embeddings={max_position}",
                },
                None,
            )

    prompts = make_prompts(args.num_seqs, input_len, args.vocab_size, prompt_seed)
    sampling = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=output_len)
    max_model_len = requested_len
    max_num_batched_tokens = max(input_len, args.max_num_batched_tokens)

    row: dict[str, Any] = {
        "model_label": model_label,
        "model_path": model_path,
        "phase": phase,
        "requested_input_len": requested_input_len,
        "input_len": input_len,
        "output_len": output_len,
        "requested_len": requested_input_len + output_len,
        "effective_len": requested_len,
        "max_position_embeddings": max_position,
        "num_seqs": args.num_seqs,
        "status": "ok",
    }
    llm = None
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        llm = LLM(
            model_path,
            enforce_eager=args.enforce_eager,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=args.num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tp_size=args.tp_size,
            ep_size=args.ep_size,
        )
        for warmup_idx in range(args.warmup):
            warm_prompts = make_prompts(args.num_seqs, input_len, args.vocab_size, prompt_seed - warmup_idx - 1)
            run_generation_trace(llm, warm_prompts, sampling)
            synchronize()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        trace = run_generation_trace(llm, prompts, sampling)
        outputs = trace["outputs"]
        total_prompt_tokens = sum(len(tokens) for tokens in prompts)
        total_gen_tokens = sum(len(tokens) for tokens in outputs)
        row.update(
            {
                "wall_time_s": trace["wall_time_s"],
                "prefill_tok_s": trace["prefill_tok_s"],
                "decode_tok_s": trace["decode_tok_s"],
                "total_tok_s": (total_prompt_tokens + total_gen_tokens) / max(trace["wall_time_s"], 1e-6),
                "first_token_latency_ms": trace["first_token_latency_ms"],
                "peak_cuda_memory_gb": peak_memory_gb(),
                "total_prompt_tokens": total_prompt_tokens,
                "total_gen_tokens": total_gen_tokens,
                "measured_prefill_tokens": trace["prefill_tokens"],
                "measured_decode_tokens": trace["decode_tokens"],
                "output_sha256": output_hash(outputs),
            }
        )
        row.update(compare_outputs(reference_outputs, outputs))
        return row, outputs
    except Exception as exc:  # noqa: BLE001 - benchmark should record failures and continue.
        row.update(
            {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(limit=6),
                "peak_cuda_memory_gb": peak_memory_gb(),
            }
        )
        return row, None
    finally:
        if llm is not None:
            llm.exit()
            del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump({"rows": rows}, file, indent=2, ensure_ascii=False)
        file.write("\n")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="BF16 vs W8A8 end-to-end throughput matrix")
    parser.add_argument("--bf16-model-path", required=True)
    parser.add_argument("--w8a8-model-path", required=True)
    parser.add_argument("--model-size-label", required=True)
    parser.add_argument("--input-lens", default="1024,2048,4096,8192,16384,32768")
    parser.add_argument("--decode-output-len", type=int, default=128)
    parser.add_argument("--mixed-output-len", type=int, default=128)
    parser.add_argument("--prefill-output-len", type=int, default=1)
    parser.add_argument("--phases", default="prefill,decode,mixed")
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--fit-input-to-max-position",
        action="store_true",
        help=(
            "If input_len + output_len exceeds max_position_embeddings, reduce input_len "
            "so the generated sequence fits in the model context. This makes 32K mean "
            "total context budget instead of prompt tokens only."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    input_lens = parse_int_list(args.input_lens)
    phases = [item.strip() for item in args.phases.split(",") if item.strip()]
    output_len_by_phase = {
        "prefill": args.prefill_output_len,
        "decode": args.decode_output_len,
        "mixed": args.mixed_output_len,
    }

    for phase in phases:
        if phase not in output_len_by_phase:
            raise ValueError(f"Unsupported phase={phase!r}")
        for input_len in input_lens:
            output_len = output_len_by_phase[phase]
            bf16_row, bf16_outputs = run_case(
                model_path=args.bf16_model_path,
                model_label=f"{args.model_size_label}_bf16",
                phase=phase,
                input_len=input_len,
                output_len=output_len,
                args=args,
                reference_outputs=None,
            )
            rows.append(bf16_row)
            write_outputs(Path(args.output_dir), rows)

            old_backend = os.environ.get("NANOVLLM_FP8_W8A8_RUNTIME_BACKEND")
            os.environ["NANOVLLM_FP8_W8A8_RUNTIME_BACKEND"] = "cutlass"
            try:
                w8a8_row, _ = run_case(
                    model_path=args.w8a8_model_path,
                    model_label=f"{args.model_size_label}_w8a8_cutlass",
                    phase=phase,
                    input_len=input_len,
                    output_len=output_len,
                    args=args,
                    reference_outputs=bf16_outputs,
                )
            finally:
                if old_backend is None:
                    os.environ.pop("NANOVLLM_FP8_W8A8_RUNTIME_BACKEND", None)
                else:
                    os.environ["NANOVLLM_FP8_W8A8_RUNTIME_BACKEND"] = old_backend
            rows.append(w8a8_row)
            write_outputs(Path(args.output_dir), rows)
            print(
                f"{phase} input={input_len} output={output_len} "
                f"bf16={bf16_row.get('status')} w8a8={w8a8_row.get('status')}"
            )


if __name__ == "__main__":
    main()
