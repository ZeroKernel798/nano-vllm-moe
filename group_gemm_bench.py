import argparse
import os
import time
from random import randint, seed

from nanovllm import LLM, SamplingParams


def main(args):
    # 1. 初始化设置
    seed(0)
    num_seqs = args.num_seqs
    max_input_len = args.max_input_len
    max_output_len = args.max_output_len

    path = os.path.expanduser(args.model_path)
    print(f"Loading model from: {path}")
    
    # 2. 初始化 LLM 引擎
    llm = LLM(
        path, 
        enforce_eager=args.enforce_eager, 
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # 3. 构造极高压力的随机负载
    print(f"Generating {num_seqs} random sequences for benchmarking...")
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    
    # 强制忽略 EOS，跑满设定的随机 max_tokens
    sampling_params = [
        SamplingParams(
            temperature=0.6, 
            ignore_eos=True, 
            max_tokens=randint(100, max_output_len)
        )
        for _ in range(num_seqs)
    ]

    # 4. Warmup (预热)
    # 这一步非常关键：让 Triton 编译 Kernel，并预先分配好 CUDA 显存
    print("Warming up the model (compiling Triton kernels)...")
    llm.generate(["Benchmark warmup: "], SamplingParams(max_tokens=8))

    # 5. 正式 Benchmark 计时
    print("Starting benchmark...")
    torch.cuda.synchronize() # 确保预热完全结束
    t0 = time.time()
    
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    
    torch.cuda.synchronize() # 确保生成完全结束
    t1 = time.time()
    
    # 6. 计算吞吐量
    total_time = t1 - t0
    total_input_tokens = sum(len(p) for p in prompt_token_ids)
    total_output_tokens = sum(sp.max_tokens for sp in sampling_params)
    
    # 输出吞吐（Output Throughput）：衡量 Decode 阶段的生成速度
    output_throughput = total_output_tokens / total_time
    # 总吞吐（Total Throughput）：包含 Prefill 阶段的综合处理速度
    total_throughput = (total_input_tokens + total_output_tokens) / total_time

    print("-" * 50)
    print("🚀 Benchmark Results:")
    print(f"Time Taken:          {total_time:.2f} s")
    print(f"Total Input Tokens:  {total_input_tokens} tok")
    print(f"Total Output Tokens: {total_output_tokens} tok")
    print(f"Output Throughput:   {output_throughput:.2f} tok/s")
    print(f"Total Throughput:    {total_throughput:.2f} tok/s")
    print("-" * 50)


if __name__ == "__main__":
    import torch # 需要在顶层或 main 里 import 用于 synchronize
    
    parser = argparse.ArgumentParser(description="nano-vllm MoE Benchmark")
    parser.add_argument(
        "--model-path", type=str, default="/home/zerokernel_ac/huggingface/qwen/Qwen1.5-MoE-A2.7B-Chat"
    )
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=4)
    parser.add_argument("--enforce-eager", type=bool, default=True)
    
    # 增加压测参数配置
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    
    args = parser.parse_args()
    main(args)