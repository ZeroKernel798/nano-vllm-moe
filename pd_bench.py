import argparse
import os
import time
import gc
import torch
from random import randint
from nanovllm import LLM, SamplingParams

def run_bench(args, tp, ep, bs, seq_len, mode, output_file):
    """
    mode: 'prefill' 或 'decode'
    """
    label = f"[{mode.upper()}] TP={tp}, EP={ep}, BS={bs}, Len={seq_len}"
    print(f"\n🚀 Running {label}...")

    try:
        path = os.path.expanduser(args.model_path)
        print(f"Loading model from: {path}")
        llm = LLM(path, tp_size=tp, ep_size=ep, enforce_eager=False)

        if mode == 'prefill':
            # Prefill 模式：长输入，只输出 1 个 token
            prompt_token_ids = [[randint(0, 1000) for _ in range(seq_len)] for _ in range(bs)]
            sampling_params = SamplingParams(max_tokens=1, ignore_eos=True)
        else:
            # Decode 模式：极短输入，长输出
            prompt_token_ids = [[1, 2, 3, 4] for _ in range(bs)] 
            sampling_params = SamplingParams(max_tokens=args.decode_len, ignore_eos=True)

        llm.generate([[1, 2, 3]], SamplingParams(max_tokens=1), use_tqdm=False)

        torch.cuda.synchronize()
        test_res = llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
        torch.cuda.synchronize()

        s = test_res["stats"]
        tps = s["prefill_tps"] if mode == 'prefill' else s["decode_tps"]
        latency = s["avg_ttft_ms"] if mode == 'prefill' else (s["total_time"] * 1000 / args.decode_len)

        # | Mode | Config | Throughput | Latency/Step |
        res_line = f"| {mode.capitalize()} | TP={tp},EP={ep},BS={bs},L={seq_len} | {tps:.2f} tok/s | {latency:.2f} ms |"
        print(f"✅ {res_line}")
        with open(output_file, "a") as f:
            f.write(res_line + "\n")

        # 清理
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5) 

    except Exception as e:
        print(f"❌ {label} Failed: {e}")

def main(args):
    output_file = "pd_separate_report.md"
    with open(output_file, "w") as f:
        f.write("# InferX P/D 分离测试报告\n\n")
        f.write("| 阶段 | 测试配置 | 吞吐量 (Throughput) | 延迟 (Latency) |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")

    # 并行策略组合
    configs = [(1, 1), (2, 1), (1, 2), (2, 2), (4, 1)]

    # Prefill  (关注长序列处理能力)
    print("\n" + " Starting Prefill Benchmark ".center(60, "="))
    prefill_scenarios = [
        (32, 512),   # 标准长度
        (32, 1024),  # 长文本
        (64, 512),   # 标准长度
        (64, 1024),  # 长文本
        (128, 512),   # 标准长度
        (128, 1024),  # 长文本
        (256, 512),   # 标准长度
        (256, 1024),  # 长文本
        (32, 4096),    # 极限单序列
        (64, 4096),   # 极限单序列
    ]
    for bs, seq in prefill_scenarios:
        for tp, ep in configs:
            run_bench(args, tp, ep, bs, seq, 'prefill', output_file)

    #  Decode (关注 Batch Size 吞吐能力) ---
    print("\n" + " Starting Decode Benchmark ".center(60, "="))
    decode_batches = [1, 32, 128, 256]
    for bs in decode_batches:
        for tp, ep in configs:
            # Decode 模式下 seq_len 固定为极小值
            run_bench(args, tp, ep, bs, 4, 'decode', output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default="/home/zerokernel_ac/huggingface/qwen/Qwen1.5-MoE-A2.7B-Chat"
    )
    parser.add_argument("--decode-len", type=int, default=128, help="Decode模式下生成的长度")
    args = parser.parse_args()
    main(args)