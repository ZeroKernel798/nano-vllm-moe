import argparse
import os
import time
import gc
from random import randint, seed
import torch

from nanovllm import LLM, SamplingParams

def run_single_test(args, tp, ep, output_file):
    """
    运行单个 (TP, EP) 配置的压测
    """
    print(f"\n" + "="*50)
    print(f"🚀 Testing Config: TP={tp}, EP={ep} (Total Cards: {tp*ep})")
    print("="*50)
    
    try:
        # 初始化引擎
        llm = LLM(
            args.model_path, 
            enforce_eager=args.enforce_eager, 
            tp_size=tp, 
            ep_size=ep
        )

        # 构造负载
        seed(0)
        prompt_token_ids = [
            [randint(0, 10000) for _ in range(randint(128, args.max_input_len))]
            for _ in range(args.num_seqs)
        ]
        sampling_params = [
            SamplingParams(
                temperature=0.6, 
                ignore_eos=True, 
                max_tokens=randint(128, args.max_output_len)
            )
            for _ in range(args.num_seqs)
        ]

        # Warmup
        llm.generate(["Warmup"], SamplingParams(max_tokens=8))

        # 正式计时
        torch.cuda.synchronize()
        t0 = time.time()
        llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        t1 = time.time()

        # 计算数据
        total_time = t1 - t0
        total_in = sum(len(p) for p in prompt_token_ids)
        total_out = sum(sp.max_tokens for sp in sampling_params)
        out_tp = total_out / total_time
        total_tp = (total_in + total_out) / total_time

        result_line = f"TP={tp}, EP={ep}, Cards={tp*ep}, Out_TP={out_tp:.2f}, Total_TP={total_tp:.2f}, Time={total_time:.2f}s"
        print(f"✅ Result: {result_line}")

        # 写入文件 (即时保存，防止崩溃丢失)
        with open(output_file, "a") as f:
            f.write(result_line + "\n")

        llm.exit() 
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2) # 给系统一点缓冲时间来释放端口

        return total_tp

    except Exception as e:
        print(f"❌ Config TP={tp}, EP={ep} Failed: {e}")
        return None

def main(args):
    sweep_configs = [
        (1, 1),           # 1卡
        (1, 2), (2, 1),   # 2卡 (EP模式 vs TP模式)
        (4, 1), (2, 2), (1, 4) # 4卡 (全EP vs 全TP vs 混合模式)
    ]

    output_file = "benchmark_summary.txt"
    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, "w") as f:
        f.write(f"InferX MoE Parallelism Sweep Report\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Seqs: {args.num_seqs}, Max_In: {args.max_input_len}, Max_Out: {args.max_output_len}\n")
        f.write("-" * 60 + "\n")

    baseline_tp = None
    
    for tp, ep in sweep_configs:
        current_tp = run_single_test(args, tp, ep, output_file)
        
        if tp == 1 and ep == 1:
            baseline_tp = current_tp

    print(f"\n✨ All tests completed. Summary saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nano-vllm MoE Sweep Benchmark")
    parser.add_argument("--model-path", type=str, default="/home/zerokernel_ac/huggingface/qwen/Qwen1.5-MoE-A2.7B-Chat")
    
    parser.add_argument("--num-seqs", type=int, default=256) 
    parser.add_argument("--max-input-len", type=int, default=512)
    parser.add_argument("--max-output-len", type=int, default=512)
    
    parser.add_argument("--enforce-eager", type=bool, default=True)
    
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)

    args = parser.parse_args()
    
    main(args)