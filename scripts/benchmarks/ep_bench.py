import argparse
import os
import time
import sys
from pathlib import Path
from random import randint, seed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanovllm import LLM, SamplingParams


def main(args):
    seed(0)
    num_seqs = args.num_seqs
    max_input_len = args.max_input_len
    max_output_len = args.max_output_len

    path = os.path.expanduser(args.model_path)
    print(f"Loading model from: {path}")
    
    llm = LLM(
        path, 
        enforce_eager=args.enforce_eager, 
        tp_size=args.tp_size,    # 传给 Config
        ep_size=args.ep_size,    # 传给 Config
    )

    print(f"Generating {num_seqs} random sequences for benchmarking...")
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(128, max_input_len))]
        for _ in range(num_seqs)
    ]
    
    sampling_params = [
        SamplingParams(
            temperature=0.6, 
            ignore_eos=True, 
            max_tokens=randint(128, max_output_len)
        )
        for _ in range(num_seqs)
    ]

    print("Warming up the model (compiling Triton kernels)...")
    llm.generate(["Benchmark warmup: "], SamplingParams(max_tokens=8))

    print("Starting benchmark...")
    torch.cuda.synchronize()
    t0 = time.time()
    
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    
    torch.cuda.synchronize()
    t1 = time.time()
    
    total_time = t1 - t0
    total_input_tokens = sum(len(p) for p in prompt_token_ids)
    total_output_tokens = sum(sp.max_tokens for sp in sampling_params)
    
    output_throughput = total_output_tokens / total_time
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
    import torch 
    
    parser = argparse.ArgumentParser(description="nano-vllm MoE Benchmark")
    parser.add_argument(
        "--model-path", type=str, default="/home/zerokernel_ac/huggingface/qwen/Qwen1.5-MoE-A2.7B-Chat"
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--enforce-eager", type=bool, default=True)
    
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    
    args = parser.parse_args()
    main(args)