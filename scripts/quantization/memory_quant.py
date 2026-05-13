from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import cuda_memory_snapshot, emit_result, nvidia_smi_query, print_result, reset_cuda_peak, runtime_metadata
from workloads import add_generation_args, build_llm, make_token_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic quantization memory benchmark")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--label", default="")
    add_generation_args(parser)
    parser.add_argument("--output-json")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    model_path = str(Path(args.model_path).expanduser())
    label = args.label or Path(model_path).name
    result = runtime_metadata(model_path, label)
    result["task"] = "memory"
    result["args"] = vars(args)

    reset_cuda_peak()
    result["before_load"] = cuda_memory_snapshot("before_load_")
    result["nvidia_smi_before_load"] = nvidia_smi_query()
    llm = build_llm(model_path, args)
    result["after_load"] = cuda_memory_snapshot("after_load_")
    result["nvidia_smi_after_load"] = nvidia_smi_query()

    prompts, sampling = make_token_batch(args, 0)
    reset_cuda_peak()
    output = llm.generate(prompts, sampling, use_tqdm=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    result["after_generate"] = cuda_memory_snapshot("after_generate_")
    result["nvidia_smi_after_generate"] = nvidia_smi_query()
    result["generated_tokens"] = sum(len(tokens) for tokens in output["results"])
    result["stats"] = output["stats"]

    emit_result(args, result)
    print_result(result, ["label", "task", "model_size_bytes", "generated_tokens", "stats.prefill_tps", "stats.decode_tps"])


if __name__ == "__main__":
    main()
