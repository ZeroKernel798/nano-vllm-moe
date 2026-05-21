from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def case_key(case: dict[str, Any]) -> tuple[int, str]:
    return int(case["shape"]["m"]), str(case["weight_name"])


def short_name(weight_name: str) -> str:
    return weight_name.rsplit(".", 1)[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize native BF16, FP8 Torch, and FP8 CUTLASS W8A8 linear runs")
    parser.add_argument("--bf16-dir", required=True)
    parser.add_argument("--w8a8-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    bf16_cases: dict[tuple[int, str], dict[str, Any]] = {}
    for path in sorted(Path(args.bf16_dir).glob("m*.json")):
        data = load_json(path)
        for case in data.get("cases", []):
            bf16_cases[case_key(case)] = case

    rows: list[dict[str, Any]] = []
    for path in sorted(Path(args.w8a8_dir).glob("m*.json")):
        data = load_json(path)
        for case in data.get("cases", []):
            key = case_key(case)
            bf16_case = bf16_cases.get(key)
            if bf16_case is None:
                continue
            backends = case["w8a8_by_backend"]
            bf16_ms = float(bf16_case["ms"])
            fp8_torch_ms = float(backends["torch"]["ms"])
            cutlass_ms = float(backends["cutlass"]["ms"])
            rows.append(
                {
                    "m": key[0],
                    "case": short_name(key[1]),
                    "weight_name": key[1],
                    "k": case["shape"]["k"],
                    "n": case["shape"]["n"],
                    "bf16_torch_ms": bf16_ms,
                    "fp8_torch_ms": fp8_torch_ms,
                    "fp8_cutlass_ms": cutlass_ms,
                    "fp8_torch_vs_bf16": fp8_torch_ms / max(bf16_ms, 1e-12),
                    "fp8_cutlass_vs_bf16": cutlass_ms / max(bf16_ms, 1e-12),
                    "cutlass_vs_fp8_torch": cutlass_ms / max(fp8_torch_ms, 1e-12),
                    "fp8_torch_cosine_vs_dequant_bf16": backends["torch"]["error_vs_bf16_dequant"]["cosine"],
                    "fp8_cutlass_cosine_vs_dequant_bf16": backends["cutlass"]["error_vs_bf16_dequant"]["cosine"],
                }
            )

    rows.sort(key=lambda row: (int(row["m"]), str(row["case"])))
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_json).open("w", encoding="utf-8") as file:
        json.dump({"rows": rows}, file, indent=2, ensure_ascii=False)
        file.write("\n")
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
