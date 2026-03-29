import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset

model_path = '/root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct'
output_path = '/root/autodl-tmp/models/Llama-3.1-AWQ-INT4'

print("📚 正在加载本地 Wikitext-2 进行校准...")
traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

# 筛选并清洗数据 只取有内容的行，凑够 128 条（对齐 QuaRot 实验标准）
samples = []
for line in traindata['text']:
    if len(line.strip()) > 50:  # 只要长度大于50个字符的有效文本
        samples.append(line.strip())
    if len(samples) >= 128:
        break

print("🚀 正在加载原模型 (FP16)...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

quant_config = {
    "zero_point": True,
    "q_group_size": 128,  
    "w_bit": 4,           # 4-bit 量化
    "version": "GEMM"
}

print("⚖️ 开始执行 AWQ-INT4 量化 (使用本地数据)...")
model.quantize(tokenizer, quant_config=quant_config, calib_data=samples)

print(f"💾 正在保存量化权重至: {output_path}")
model.save_quantized(output_path)
tokenizer.save_pretrained(output_path)

print("\n🎉 成功！现在你可以去对比 QuaRot 了。")