from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_parent(path: str | os.PathLike[str] | None) -> None:
    if path:
        Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def read_json(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def read_model_config(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    return read_json(Path(model_path) / "config.json") or {}


def quant_metadata(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    config = read_model_config(model_path)
    return {
        "quantization_type": config.get("quantization_type"),
        "quantization_config": config.get("quantization_config"),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
    }


def path_size_bytes(path: str | os.PathLike[str]) -> int:
    root = Path(path)
    if root.is_file():
        return root.stat().st_size
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def git_metadata() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "is_dirty": bool(run(["git", "status", "--porcelain"])),
    }


def gpu_summary() -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "capability": f"{props.major}.{props.minor}",
                "total_memory_bytes": props.total_memory,
            }
        )
    return devices


def runtime_metadata(model_path: str, label: str) -> dict[str, Any]:
    return {
        "timestamp": now_tag(),
        "label": label,
        "model_path": str(Path(model_path).expanduser()),
        "model_size_bytes": path_size_bytes(model_path),
        "quant": quant_metadata(model_path),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": gpu_summary(),
        "git": git_metadata(),
    }


def cuda_memory_snapshot(prefix: str = "") -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {f"{prefix}cuda_available": False}
    torch.cuda.synchronize()
    result: dict[str, Any] = {f"{prefix}cuda_available": True}
    for index in range(torch.cuda.device_count()):
        result[f"{prefix}gpu{index}_allocated_bytes"] = torch.cuda.memory_allocated(index)
        result[f"{prefix}gpu{index}_reserved_bytes"] = torch.cuda.memory_reserved(index)
        result[f"{prefix}gpu{index}_max_allocated_bytes"] = torch.cuda.max_memory_allocated(index)
        result[f"{prefix}gpu{index}_max_reserved_bytes"] = torch.cuda.max_memory_reserved(index)
    return result


def reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def nvidia_smi_query() -> list[dict[str, str]]:
    fields = "index,name,compute_cap,memory.used,memory.total,utilization.gpu"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    names = fields.split(",")
    for line in output.strip().splitlines():
        values = [part.strip() for part in line.split(",")]
        rows.append(dict(zip(names, values, strict=False)))
    return rows


def write_json(path: str | os.PathLike[str] | None, data: dict[str, Any]) -> None:
    if not path:
        return
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def append_jsonl(path: str | os.PathLike[str] | None, data: dict[str, Any]) -> None:
    if not path:
        return
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, name))
        elif isinstance(value, list):
            flat[name] = json.dumps(value, ensure_ascii=False)
        else:
            flat[name] = value
    return flat


def append_csv(path: str | os.PathLike[str] | None, data: dict[str, Any]) -> None:
    if not path:
        return
    ensure_parent(path)
    flat = flatten_dict(data)
    csv_path = Path(path)
    rows: list[dict[str, Any]] = []
    fields: list[str] = []
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            fields = list(reader.fieldnames or [])
            rows.extend(dict(row) for row in reader)
    for key in flat:
        if key not in fields:
            fields.append(key)
    rows.append(flat)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def emit_result(args: Any, result: dict[str, Any]) -> None:
    write_json(getattr(args, "output_json", None), result)
    append_jsonl(getattr(args, "output_jsonl", None), result)
    append_csv(getattr(args, "output_csv", None), result)


def print_result(data: dict[str, Any], keys: list[str]) -> None:
    for key in keys:
        value: Any = data
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None:
            print(f"{key}={value}")
