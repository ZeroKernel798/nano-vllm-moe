import os
import torch
import tqdm
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import modelopt.torch.quantization as mtq
from safetensors.torch import save_file

# --- 配置 ---
model_path = '/root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct'
output_path = '/root/autodl-tmp/models/Llama-3.1-ModelOpt-FP8'
if not os.path.exists(output_path): os.makedirs(output_path)

# 1. 数据准备 (模仿你提供的逻辑)
print("📚 正在准备 Wikitext-2 校准样本...")
traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
samples = []
for line in traindata['text']:
    if len(line.strip()) > 50:
        samples.append(line.strip())
    if len(samples) >= 128:
        break

# 2. 加载模型
print("🚀 正在加载原模型 (BF16)...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# 3. 定义 ModelOpt 所需的 Forward Loop
# ModelOpt 会在 forward 过程中通过观察算子自动捕获每一层的激活值分布
def calibration_loop(model):
    print("⚖️ 开始执行 PTQ 校准推理...")
    with torch.no_grad():
        for text in tqdm.tqdm(samples, desc="Calibrating"):
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to("cuda")
            model(**inputs)

# 4. 执行量化配置
# 使用 ModelOpt 的标准 FP8 配置 (E4M3 格式)
# 这会自动处理权重和激活值的 Scaling
print("🪄 正在应用 ModelOpt FP8 静态量化配置...")
config = mtq.FP8_DEFAULT_CFG

# 核心：量化并校准
# 该函数会先修改模型结构插入量化节点，然后运行 calibration_loop 统计 Scale
model = mtq.quantize(model, config, calibration_loop)

# 5. 导出状态字典
print("💾 正在导出 FP8 权重与 Scaling Factors...")
fp8_state_dict = {}
model_state = model.state_dict()

for name, param in tqdm.tqdm(model_state.items(), desc="Exporting"):
    # ModelOpt 转换后的权重通常是 float8_e4m3fn 类型
    if param.dtype == torch.float8_e4m3fn:
        # 为了兼容性，我们可以将其 view 为 uint8 保存，或者直接保存（取决于你的推理后端）
        fp8_state_dict[name] = param.view(torch.uint8)
    else:
        fp8_state_dict[name] = param

# 6. 保存文件
save_file(fp8_state_dict, os.path.join(output_path, "model.safetensors"))
tokenizer.save_pretrained(output_path)

# ---------------------------------------------------------
# 7. 更新 Config (非常科学的建议)
# 在 config.json 中指明量化方式，方便 vLLM 或 SGLang 自动识别
# ---------------------------------------------------------
print("📝 正在更新模型配置...")
model.config.quantization_config = {
    "quant_method": "fp8",
    "activation_scheme": "static", # 明确标注是静态量化
    "kv_cache_dtype": "fp8",       # 如果你打算以后量化 KV Cache
    "producer": "nvidia-modelopt"
}
model.config.save_pretrained(output_path)

print(f"\n🎉 FP8 静态量化完成！输出路径: {output_path}")