import os
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

model_path = '/root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct'
output_path = '/root/autodl-tmp/models/Llama-3.1-Pure-FP8' 
if not os.path.exists(output_path): os.makedirs(output_path)

print("🚀 加载原模型 (BF16)...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)

print("💾 正在执行纯净 FP8 (E4M3) 量化...")
fp8_state_dict = {}
model_state = model.state_dict()

FP8_MAX = 448.0

for name, param in tqdm.tqdm(model_state.items(), desc="FP8 Exporting"):
    if "weight" in name and any(x in name for x in ["proj", "fc", "gate_up"]):
        weight_max = param.abs().max().to(torch.float32)
        scale = weight_max / FP8_MAX
        
        qweight = (param.to(torch.float32) / scale).to(torch.float8_e4m3fn)
        base_name = name.replace(".weight", "")
        
        fp8_state_dict[f"{base_name}.qweight"] = qweight.view(torch.uint8)
        fp8_state_dict[f"{base_name}.weight_scale"] = torch.tensor([scale], dtype=torch.float32)
    else:
        # 非线性层保持原样
        fp8_state_dict[name] = param

print("📦 保存模型与配置...")
save_file(fp8_state_dict, os.path.join(output_path, "model.safetensors"))
tokenizer.save_pretrained(output_path)

model.config.quantization_config = {"quant_method": "fp8"}
model.config.model_type = "llama"
model.config.save_pretrained(output_path)

print(f"\n🎉 纯净 FP8 转换完成！路径: {output_path}")