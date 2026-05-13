from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import now_tag

SCRIPT_DIR = Path(__file__).resolve().parent


def run_step(name: str, cmd: list[str], keep_going: bool) -> int:
    print(f"\n### quant_eval:{name}")
    print(" ".join(cmd))
    status = subprocess.call(cmd)
    print(f"### quant_eval:{name}:exit_code={status}")
    if status and not keep_going:
        raise SystemExit(status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generic quantization eval suite")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-inspect", action="store_true")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--baseline-model-path", default="", help="Baseline BF16/HF model path for logits/token comparison")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=16)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--bench-repeat", type=int, default=1)
    parser.add_argument("--bench-warmup", type=int, default=0)
    parser.add_argument("--ppl-max-length", type=int, default=512)
    parser.add_argument("--ppl-stride", type=int, default=256)
    parser.add_argument("--ppl-max-tokens", type=int, default=2048)
    parser.add_argument("--ppl-text-file", default="")
    parser.add_argument("--dataset-cache-dir", default="")
    parser.add_argument("--quant-format", default="auto")
    parser.add_argument("--compare-max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f".remote-logs/quantization/{now_tag()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    jsonl_path = output_dir / "results.jsonl"
    common = ["--model-path", args.model_path, "--label", args.label, "--output-jsonl", str(jsonl_path), "--output-csv", str(csv_path)]
    status_codes: dict[str, int] = {}

    if not args.skip_inspect:
        status_codes["inspect"] = run_step(
            "inspect",
            [args.python, str(SCRIPT_DIR / "inspect_checkpoint.py"), *common, "--output-json", str(output_dir / "inspect.json")],
            args.keep_going,
        )
    gen_args = [
        "--num-seqs", str(args.num_seqs),
        "--input-len", str(args.input_len),
        "--output-len", str(args.output_len),
        "--max-model-len", str(args.max_model_len),
        "--tp-size", str(args.tp_size),
        "--ep-size", str(args.ep_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]
    if args.enforce_eager:
        gen_args.append("--enforce-eager")
    if not args.skip_bench:
        status_codes["bench"] = run_step(
            "bench",
            [args.python, str(SCRIPT_DIR / "bench_quant.py"), *common, *gen_args, "--repeat", str(args.bench_repeat), "--warmup", str(args.bench_warmup), "--output-json", str(output_dir / "bench.json")],
            args.keep_going,
        )
    if not args.skip_memory:
        status_codes["memory"] = run_step(
            "memory",
            [args.python, str(SCRIPT_DIR / "memory_quant.py"), *common, *gen_args, "--output-json", str(output_dir / "memory.json")],
            args.keep_going,
        )
    if not args.skip_ppl:
        ppl_args = ["--quant-format", args.quant_format, "--max-length", str(args.ppl_max_length), "--stride", str(args.ppl_stride), "--max-tokens", str(args.ppl_max_tokens)]
        if args.ppl_text_file:
            ppl_args.extend(["--text-file", args.ppl_text_file])
        if args.dataset_cache_dir:
            ppl_args.extend(["--dataset-cache-dir", args.dataset_cache_dir])
        status_codes["ppl"] = run_step(
            "ppl",
            [args.python, str(SCRIPT_DIR / "eval_ppl_quant.py"), *common, *ppl_args, "--output-json", str(output_dir / "ppl.json")],
            args.keep_going,
        )

    if not args.skip_compare:
        if not args.baseline_model_path:
            if not args.keep_going:
                raise SystemExit("--baseline-model-path is required unless --skip-compare is set")
            print("### quant_eval:compare:skipped missing --baseline-model-path")
        else:
            status_codes["compare"] = run_step(
                "compare",
                [
                    args.python,
                    str(SCRIPT_DIR / "compare_logits.py"),
                    "--baseline-model-path",
                    args.baseline_model_path,
                    *common,
                    "--quant-format",
                    args.quant_format,
                    "--max-new-tokens",
                    str(args.compare_max_new_tokens),
                    "--output-json",
                    str(output_dir / "compare.json"),
                ],
                args.keep_going,
            )

    print(f"\noutput_dir={output_dir}")
    print(f"csv={csv_path}")
    print(f"jsonl={jsonl_path}")
    print(f"status_codes={status_codes}")
    if any(status_codes.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
