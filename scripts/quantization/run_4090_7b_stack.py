from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import now_tag


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
KV_SCRIPT_DIR = REPO_ROOT / "scripts" / "kv_cache"


def run_step(name: str, cmd: list[str], keep_going: bool) -> int:
    print(f"\n### quant_4090_7b:{name}")
    print(" ".join(cmd))
    status = subprocess.call(cmd)
    print(f"### quant_4090_7b:{name}:exit_code={status}")
    if status and not keep_going:
        raise SystemExit(status)
    return status


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")


def suite_cmd(args: argparse.Namespace, model_path: str, label: str, output_dir: Path, quant_format: str) -> list[str]:
    cmd = [
        args.python,
        str(SCRIPT_DIR / "run_quant_suite.py"),
        "--model-path",
        model_path,
        "--label",
        label,
        "--output-dir",
        str(output_dir),
        "--python",
        args.python,
        "--num-seqs",
        str(args.num_seqs),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--max-model-len",
        str(args.max_model_len),
        "--tp-size",
        str(args.tp_size),
        "--ep-size",
        str(args.ep_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--bench-repeat",
        str(args.bench_repeat),
        "--bench-warmup",
        str(args.bench_warmup),
        "--ppl-max-length",
        str(args.ppl_max_length),
        "--ppl-stride",
        str(args.ppl_stride),
        "--ppl-max-tokens",
        str(args.ppl_max_tokens),
        "--quant-format",
        quant_format,
        "--compare-max-new-tokens",
        str(args.compare_max_new_tokens),
    ]
    if args.dataset_cache_dir:
        cmd.extend(["--dataset-cache-dir", args.dataset_cache_dir])
    if args.ppl_text_file:
        cmd.extend(["--ppl-text-file", args.ppl_text_file])
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.skip_ppl:
        cmd.append("--skip-ppl")
    if args.skip_compare or label == "bf16":
        cmd.append("--skip-compare")
    else:
        cmd.extend(["--baseline-model-path", args.bf16_model_path])
    if label == "bf16":
        cmd.append("--skip-inspect")
    if args.keep_going:
        cmd.append("--keep-going")
    return cmd


def kv_cmd(args: argparse.Namespace, model_path: str, label: str, output_json: Path, input_len: int) -> list[str]:
    return [
        args.python,
        str(KV_SCRIPT_DIR / "kv_cache_fp8_smoke.py"),
        "--model-path",
        model_path,
        "--label",
        label,
        "--input-len",
        str(input_len),
        "--output-len",
        str(args.kv_output_len),
        "--max-model-len",
        str(max(input_len + args.kv_output_len, args.max_model_len)),
        "--max-num-batched-tokens",
        str(max(input_len + args.kv_output_len, args.max_model_len)),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--fp8-decode-backend",
        args.fp8_decode_backend,
        "--native-block-tokens",
        str(args.native_block_tokens),
        "--kv-cache-scale-dtype",
        args.kv_cache_scale_dtype,
        "--seed",
        str(args.seed),
        "--output-json",
        str(output_json),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RTX 4090 7B quantization stack suite")
    parser.add_argument("--bf16-model-path", required=True)
    parser.add_argument("--w8a16-model-path", required=True)
    parser.add_argument("--w8a8-model-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--stages", default="bf16,w8a16,kv", help="Comma-separated: bf16,w8a16,kv,w8a8")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--bench-repeat", type=int, default=1)
    parser.add_argument("--bench-warmup", type=int, default=1)
    parser.add_argument("--ppl-max-length", type=int, default=512)
    parser.add_argument("--ppl-stride", type=int, default=256)
    parser.add_argument("--ppl-max-tokens", type=int, default=1024)
    parser.add_argument("--ppl-text-file", default="")
    parser.add_argument("--dataset-cache-dir", default="")
    parser.add_argument("--compare-max-new-tokens", type=int, default=8)
    parser.add_argument("--kv-input-lens", default="8192,16384,32512")
    parser.add_argument("--kv-output-len", type=int, default=16)
    parser.add_argument("--kv-cache-dtype", default="k_int8_v_fp8", choices=("fp8_e4m3", "fp8_v_only", "k_int8_v_fp8"))
    parser.add_argument("--fp8-decode-backend", default="native", choices=("native", "gather_dequant", "full_dequant"))
    parser.add_argument("--native-block-tokens", type=int, default=64, choices=(16, 32, 64))
    parser.add_argument("--kv-cache-scale-dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--seed", type=int, default=20260513)
    add_generation_args(parser)
    args = parser.parse_args()

    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    output_dir = Path(args.output_dir or f".remote-logs/quantization/4090_7b_stack_{now_tag()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    status_codes: dict[str, int] = {}
    if "bf16" in stages:
        status_codes["bf16"] = run_step(
            "bf16",
            suite_cmd(args, args.bf16_model_path, "bf16", output_dir / "bf16", "none"),
            args.keep_going,
        )
    if "w8a16" in stages:
        status_codes["w8a16"] = run_step(
            "w8a16",
            suite_cmd(args, args.w8a16_model_path, "w8a16", output_dir / "w8a16", "fp8_w8a16"),
            args.keep_going,
        )
    if "kv" in stages:
        for input_len in [int(item) for item in args.kv_input_lens.split(",") if item.strip()]:
            name = f"w8a16_{args.kv_cache_dtype}_{input_len}"
            status_codes[name] = run_step(
                name,
                kv_cmd(args, args.w8a16_model_path, name, output_dir / f"{name}.json", input_len),
                args.keep_going,
            )
    if "w8a8" in stages:
        if not args.w8a8_model_path:
            raise SystemExit("--w8a8-model-path is required for stage w8a8")
        status_codes["w8a8"] = run_step(
            "w8a8",
            suite_cmd(args, args.w8a8_model_path, "w8a8", output_dir / "w8a8", "fp8_w8a8_static"),
            args.keep_going,
        )

    manifest = {
        "output_dir": str(output_dir),
        "stages": sorted(stages),
        "status_codes": status_codes,
        "args": vars(args),
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    if any(status_codes.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
