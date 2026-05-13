from __future__ import annotations

import argparse
from collections import Counter
from glob import glob
from pathlib import Path

from safetensors import safe_open

from common import append_csv, append_jsonl, print_result, read_model_config, runtime_metadata, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect generic quantized checkpoint contract")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--output-json")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    model_path = str(Path(args.model_path).expanduser())
    label = args.label or Path(model_path).name
    result = runtime_metadata(model_path, label)
    result["task"] = "inspect_checkpoint"
    config = read_model_config(model_path)
    quantization_type = config.get("quantization_type")
    result["config_fields"] = {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "quantization_type": quantization_type,
        "quantization_config": config.get("quantization_config"),
    }

    suffix_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    qweight_bases: set[str] = set()
    scale_bases: set[str] = set()
    input_scale_bases: set[str] = set()
    tensor_count = 0
    examples: list[dict] = []
    safetensor_files = sorted(glob(str(Path(model_path) / "*.safetensors")))
    for file in safetensor_files:
        with safe_open(file, "pt", "cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                tensor_count += 1
                dtype_counts[str(tensor.dtype)] += 1
                suffix = key.rsplit(".", 1)[-1]
                suffix_counts[suffix] += 1
                if key.endswith(".qweight"):
                    qweight_bases.add(key[: -len(".qweight")])
                elif key.endswith(".weight_scale"):
                    scale_bases.add(key[: -len(".weight_scale")])
                elif key.endswith(".input_scale"):
                    input_scale_bases.add(key[: -len(".input_scale")])
                if len(examples) < 12:
                    examples.append({"name": key, "shape": list(tensor.shape), "dtype": str(tensor.dtype)})

    missing_scales = sorted(qweight_bases - scale_bases)
    orphan_scales = sorted(scale_bases - qweight_bases)
    missing_input_scales = sorted(qweight_bases - input_scale_bases) if quantization_type == "fp8_w8a8_static" else []
    orphan_input_scales = sorted(input_scale_bases - qweight_bases)
    is_contract_healthy = not missing_scales and not orphan_scales and not missing_input_scales and not orphan_input_scales
    result["checkpoint"] = {
        "safetensor_files": len(safetensor_files),
        "tensor_count": tensor_count,
        "suffix_counts": dict(suffix_counts),
        "dtype_counts": dict(dtype_counts),
        "qweight_count": len(qweight_bases),
        "weight_scale_count": len(scale_bases),
        "input_scale_count": len(input_scale_bases),
        "missing_scales": missing_scales[:50],
        "orphan_scales": orphan_scales[:50],
        "missing_input_scales": missing_input_scales[:50],
        "orphan_input_scales": orphan_input_scales[:50],
        "is_contract_healthy": is_contract_healthy,
        "examples": examples,
    }

    write_json(args.output_json, result)
    append_jsonl(args.output_jsonl, result)
    append_csv(args.output_csv, result)
    print_result(
        result,
        [
            "label",
            "task",
            "config_fields.quantization_type",
            "checkpoint.tensor_count",
            "checkpoint.qweight_count",
            "checkpoint.weight_scale_count",
            "checkpoint.input_scale_count",
            "checkpoint.is_contract_healthy",
        ],
    )


if __name__ == "__main__":
    main()
