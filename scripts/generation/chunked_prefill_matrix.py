from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

from chunked_prefill_bench import phase3_one


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def run_case(base_args: argparse.Namespace, mode: str, b_tokens: int, chunk: int, policy: str) -> dict:
    args = SimpleNamespace(**vars(base_args))
    args.phase3_b_prompt_tokens_resolved = b_tokens
    args.phase3_b_prompt_tokens = b_tokens
    args.max_model_len = max(args.max_model_len, b_tokens + args.phase3_b_output_tokens)
    try:
        row = phase3_one(args, chunk, policy)
        row["mode"] = mode
        row["status"] = "ok"
        row["error"] = ""
        return row
    except Exception as exc:  # keep the matrix moving after OOM or compile/runtime failures
        return {
            "phase": "phase3_interference",
            "mode": mode,
            "chunk": chunk,
            "policy": policy,
            "seq_a_prompt_tokens": args.phase3_a_prompt_tokens,
            "seq_a_output_tokens": args.phase3_a_output_tokens,
            "seq_b_prompt_tokens": b_tokens,
            "seq_b_output_tokens": args.phase3_b_output_tokens,
            "inject_after_tokens": args.phase3_inject_after_tokens,
            "status": "error",
            "error": repr(exc),
            "traceback_tail": "\n".join(traceback.format_exc().splitlines()[-8:]),
        }


def write_outputs(rows: list[dict], output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = sorted({key for row in rows for key in row})
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunked prefill A/B interference matrix")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-label", default="")
    parser.add_argument("--b-prompt-tokens", type=parse_int_list, default="2048,4096,8192,16384,32768")
    parser.add_argument("--decode-first-chunks", type=parse_int_list, default="128,256,512,1024")
    parser.add_argument("--prefill-first-chunk", type=int, default=128)
    parser.add_argument("--a-prompt-tokens", type=int, default=32)
    parser.add_argument("--a-output-tokens", type=int, default=128)
    parser.add_argument("--b-output-tokens", type=int, default=1)
    parser.add_argument("--inject-after-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-seqs", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    # Names expected by chunked_prefill_bench.phase3_one.
    args.phase3_a_prompt_tokens = args.a_prompt_tokens
    args.phase3_a_output_tokens = args.a_output_tokens
    args.phase3_b_output_tokens = args.b_output_tokens
    args.phase3_inject_after_tokens = args.inject_after_tokens

    rows: list[dict] = []
    for b_tokens in args.b_prompt_tokens:
        no_chunk_size = max(args.max_model_len, b_tokens + args.b_output_tokens)
        rows.append(run_case(args, "no_chunk", b_tokens, no_chunk_size, "prefill_first"))
        write_outputs(rows, args.output_json, args.output_csv)

        rows.append(run_case(args, "chunk_prefill_first_128", b_tokens, args.prefill_first_chunk, "prefill_first"))
        write_outputs(rows, args.output_json, args.output_csv)

        for chunk in args.decode_first_chunks:
            rows.append(run_case(args, f"chunk_decode_first_{chunk}", b_tokens, chunk, "decode_first"))
            write_outputs(rows, args.output_json, args.output_csv)

    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
