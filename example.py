import argparse
import os

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


def main(args):
    path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path, 
        enforce_eager=args.enforce_eager, 
        tp_size=args.tp_size,   
        ep_size=args.ep_size,   
    )

    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_tokens
    )
    prompts = [
        "introduce yourself",
        # "list all prime numbers within 100",
        # "你好 请问你是谁",
        # "What is 123 + 456",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for prompt in prompts
    ]
    response = llm.generate(prompts, sampling_params)

    for prompt, output_tokens in zip(prompts, response['results']):
        # 手动调用 llm.tokenizer.decode
        text = llm.tokenizer.decode(output_tokens)
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {text!r}")

    # 性能数据 
    s = response['stats']
    print(f"\n" + "="*30)
    print(f"📊 推理性能报告:")
    print(f"⏱️  总耗时: {s['total_time']:.2f}s")
    print(f"🚀 Prefill 吞吐: {s['prefill_tps']:.2f} tok/s")
    print(f"⚡ Decode 吞吐: {s['decode_tps']:.2f} tok/s")
    print(f"⏳ 平均首词延迟 (TTFT): {s['avg_ttft_ms']:.2f} ms")
    print("="*30)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="nano vllm")
    # argparse.add_argument(
    #     "--model-path", type=str, default="/root/autodl-tmp/models/Llama-3.1-SmoothQuant-INT8"
    # )
    argparse.add_argument(
        "--model-path", type=str, default="/root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    )
    # argparse.add_argument(
    #     "--model-path", type=str, default="/root/autodl-tmp/models/Llama-3.1-Pure-FP8"
    # )
    # argparse.add_argument(
    #     "--model-path", type=str, default="/root/autodl-tmp/models/Llama-3.1-ModelOpt-FP8"
    # )

    argparse.add_argument("--tp-size", type=int, default=1)
    argparse.add_argument("--ep-size", type=int, default=1)
    argparse.add_argument("--enforce-eager", type=bool, default=True)
    argparse.add_argument("--temperature", type=float, default=0.6)
    argparse.add_argument("--max-tokens", type=int, default=256)
    args = argparse.parse_args()

    main(args)
