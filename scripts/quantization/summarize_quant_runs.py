from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from common import flatten_dict


def load_records(paths: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        p = Path(path)
        if p.suffix == ".jsonl":
            with p.open(encoding="utf-8") as file:
                records.extend(json.loads(line) for line in file if line.strip())
        else:
            with p.open(encoding="utf-8") as file:
                records.append(json.load(file))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize quant eval JSON/JSONL runs")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    records = load_records(args.inputs)
    rows = [flatten_dict(record) for record in records]
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    selected = [
        "label",
        "task",
        "quant.quantization_type",
        "summary.prefill_tps",
        "summary.decode_tps",
        "summary.wall_time_s",
        "metrics.perplexity",
        "model_size_bytes",
    ]
    print("\t".join(selected))
    for row in rows:
        print("\t".join(str(row.get(key, "")) for key in selected))


if __name__ == "__main__":
    main()
