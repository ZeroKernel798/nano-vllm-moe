from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer
import torch

# 指向你刚才生成的 INT4 文件夹
model_path = '/root/autodl-tmp/models/Llama-3.1-AWQ-INT4'

print("🚀 正在加载 AWQ-INT4 模型 (这会非常快)...")
# 加载量化模型，fuse_layers=True 可以加速推理
model = AutoAWQForCausalLM.from_quantized(model_path, fuse_layers=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 准备测试指令
prompt = "introduce yourself"
# Llama-3.1 需要特定的 Chat Template，这里简单处理
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("\n🤖 模型回复：")
_ = model.generate(**inputs, streamer=streamer, max_new_tokens=100)