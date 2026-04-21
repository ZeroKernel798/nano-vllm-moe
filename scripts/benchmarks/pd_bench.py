import argparse
import os
import gc
import sys
from pathlib import Path
import torch
from random import randint

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm import LLM, SamplingParams


def run_bench(llm, args, tp, ep, bs, seq_len, mode, output_file):
    """
    mode: 'prefill' 或 'decode'
    在同一 LLM 实例上跑单组参数（不再加载/卸载模型）。
    """
    label = f"[{mode.upper()}] TP={tp}, EP={ep}, BS={bs}, Len={seq_len}"
    print(f"\n🚀 Running {label}...")

    try:
        if mode == "prefill":
            prompt_token_ids = [
                [randint(0, 1000) for _ in range(seq_len)] for _ in range(bs)
            ]
            sampling_params = SamplingParams(max_tokens=1, ignore_eos=True)
        else:
            prompt_token_ids = [[1, 2, 3, 4] for _ in range(bs)]
            sampling_params = SamplingParams(
                max_tokens=args.decode_len, ignore_eos=True
            )

        llm.generate([[1, 2, 3]], SamplingParams(max_tokens=1), use_tqdm=False)

        torch.cuda.synchronize()
        test_res = llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
        torch.cuda.synchronize()

        s = test_res["stats"]
        tps = s["prefill_tps"] if mode == "prefill" else s["decode_tps"]
        latency = (
            s["avg_ttft_ms"]
            if mode == "prefill"
            else (s["total_time"] * 1000 / args.decode_len)
        )

        res_line = (
            f"| {mode.capitalize()} | TP={tp},EP={ep},BS={bs},L={seq_len} | "
            f"{tps:.2f} tok/s | {latency:.2f} ms |"
        )
        print(f"✅ {res_line}")
        with open(output_file, "a") as f:
            f.write(res_line + "\n")

    except Exception as e:
        print(f"❌ {label} Failed: {e}")


def main(args):
    output_file = "pd_separate_report.md"
    with open(output_file, "w") as f:
        f.write("# InferX P/D 分离测试报告\n\n")
        f.write("| 阶段 | 测试配置 | 吞吐量 (Throughput) | 延迟 (Latency) |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")

    # configs = [(1, 1), (2, 1), (1, 2), (2, 2), (4, 1)]
    configs = [(1, 1)]

    prefill_scenarios = [
        (32, 512),
        (32, 1024),
        (64, 512),
        (64, 1024),
        (128, 512),
        (128, 1024),
        (256, 512),
        (256, 1024),
        (32, 4096),
        (64, 4096),
        (128, 4096),
        (256, 4096),
    ]
    decode_batches = [1, 32, 128, 256]

    path = os.path.expanduser(args.model_path)

    for tp, ep in configs:
        print(f"\n{'=' * 60}\nLoading model: {path}  TP={tp}, EP={ep}\n{'=' * 60}")
        llm = LLM(path, tp_size=tp, ep_size=ep, enforce_eager=False)
        try:
            print("\n" + " Starting Prefill Benchmark ".center(60, "="))
            for bs, seq in prefill_scenarios:
                run_bench(llm, args, tp, ep, bs, seq, "prefill", output_file)

            print("\n" + " Starting Decode Benchmark ".center(60, "="))
            for bs in decode_batches:
                run_bench(llm, args, tp, ep, bs, 4, "decode", output_file)
        finally:
            llm.exit()
            del llm
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/autodl-tmp/models/Llama-3.1-Pure-FP8-W8A16",
    )
    parser.add_argument(
        "--decode-len",
        type=int,
        default=128,
        help="Decode模式下生成的长度",
    )
    args = parser.parse_args()
    main(args)
