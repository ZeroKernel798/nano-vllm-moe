from __future__ import annotations

import argparse
from pathlib import Path

from common import cuda_memory_snapshot, emit_result, print_result, runtime_metadata
from workloads import add_generation_args, build_llm, run_generation_once


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic quantization throughput benchmark")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--label", default="")
    add_generation_args(parser)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0, help="Number of untimed warmup runs")
    parser.add_argument("--output-json")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    model_path = str(Path(args.model_path).expanduser())
    label = args.label or Path(model_path).name
    result = runtime_metadata(model_path, label)
    result["task"] = "throughput"
    result["args"] = vars(args)
    result["memory_before_load"] = cuda_memory_snapshot("before_load_")

    llm = build_llm(model_path, args)
    result["memory_after_load"] = cuda_memory_snapshot("after_load_")

    for warmup_index in range(args.warmup):
        run_generation_once(llm, args, -warmup_index - 1)
    result["memory_after_warmup"] = cuda_memory_snapshot("after_warmup_")

    runs = [run_generation_once(llm, args, index) for index in range(args.repeat)]
    result["runs"] = runs
    result["summary"] = {
        key: sum(run[key] for run in runs) / len(runs)
        for key in ("wall_time_s", "prefill_tps", "decode_tps", "avg_ttft_ms", "end_to_end_tps")
    }
    result["summary"]["total_gen_tokens_last"] = runs[-1]["total_gen_tokens"]
    result["summary"]["total_prompt_tokens_last"] = runs[-1]["total_prompt_tokens"]

    emit_result(args, result)
    print_result(
        result,
        [
            "label",
            "task",
            "summary.wall_time_s",
            "summary.prefill_tps",
            "summary.decode_tps",
            "summary.avg_ttft_ms",
            "summary.end_to_end_tps",
        ],
    )


if __name__ == "__main__":
    main()
