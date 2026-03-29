import os
import torch
import tqdm
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from smoothquant.smooth import smooth_lm
from smoothquant.calibration import get_act_scales
from safetensors.torch import save_file

model_path = '/root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct'
output_path = '/root/autodl-tmp/models/Llama-3.1-SmoothQuant-INT8'
temp_data_path = "temp_calib_data.jsonl" # 临时中转文件
if not os.path.exists(output_path): os.makedirs(output_path)

print("📚 正在加载本地 Wikitext-2 进行校准...")
traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

samples = []
for line in traindata['text']:
    if len(line.strip()) > 50:  
        samples.append(line.strip())
    if len(samples) >= 128:
        break

print(f"📝 正在生成临时校准文件: {temp_data_path}")
with open(temp_data_path, "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps({"text": s}) + "\n")

print("🚀 正在加载原模型 (BF16)...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)

print("⚖️ 正在进行激活值校准 (使用对齐后的 128 条数据)...")

try:
    act_scales = get_act_scales(
        model, 
        tokenizer, 
        temp_data_path,  
        num_samples=128, 
        seq_len=512
    )
except TypeError:
    act_scales = get_act_scales(model, tokenizer, temp_data_path, 128, 512)

print("🪄 正在执行 SmoothQuant 变换 (alpha=0.5)...")
smooth_lm(model, act_scales, alpha=0.5)

print("💾 正在执行 Int8 量化并导出...")
int8_state_dict = {}
model_state = model.state_dict()

for name, param in tqdm.tqdm(model_state.items(), desc="Exporting"):
    if "weight" in name and any(x in name for x in ["proj", "fc", "gate_up"]):
        weight_scales = param.abs().max(dim=-1, keepdim=True)[0].to(torch.float16) / 127.0
        
        qweight = (param / weight_scales).round().clamp(-128, 127).to(torch.int8)
        
        base_name = name.replace(".weight", "")
        int8_state_dict[f"{base_name}.qweight"] = qweight
        int8_state_dict[f"{base_name}.weight_scales"] = weight_scales
    else:
        int8_state_dict[name] = param

save_file(int8_state_dict, os.path.join(output_path, "model.safetensors"))
tokenizer.save_pretrained(output_path)

model.config.quantization_config = {"quant_method": "smoothquant"}
model.config.model_type = "llama"
model.config.save_pretrained(output_path)

if os.path.exists(temp_data_path): os.remove(temp_data_path)
print(f"\n🎉 转换完成！输出路径: {output_path}")

