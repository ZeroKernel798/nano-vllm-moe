from modelscope import snapshot_download
import os

# 推荐 DeepSeek MoE 16B，逻辑最硬核，ModelScope 必有
model_id = 'deepseek-ai/deepseek-moe-16b-chat'
cache_dir = '/root/autodl-tmp/models'

print(f"开始从 ModelScope 下载 {model_id}...")

model_dir = snapshot_download(
    model_id, 
    cache_dir=cache_dir, 
    revision='master'
)

print(f"\n✅ 下载完成！路径: {model_dir}")