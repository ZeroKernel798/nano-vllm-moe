import argparse
from modelscope import snapshot_download
import os

os.environ['MODELSCOPE_PARALLEL_DOWNLOAD'] = '8'

def download_model():
    parser = argparse.ArgumentParser(description="Llama-3.2 对话版下载工具")
    
    # 修改点：将默认 ID 改为 Instruct 版本
    # 如果显存足够（建议 8G 以上），可以将 default 改为 "LLM-Research/Llama-3.2-3B-Instruct"
    parser.add_argument(
        "--model-id",
        "--model",
        dest="model_id",
        type=str,
        default="qwen/Qwen1.5-MoE-A2.7B-Chat",
        help="ModelScope 上的模型 ID (建议使用 Instruct 结尾的对话版)"
    )
    
    parser.add_argument(
        "--cache-dir",
        "--path",
        dest="cache_dir",
        type=str,
        default="/workspace/models/",
        help="模型存储路径"
    )
    
    args = parser.parse_args()

    print(f"🚀 准备从 ModelScope 下载对话模型: {args.model_id}")
    print(f"📂 存储路径设定为: {args.cache_dir}")

    try:
        # 下载模型快照
        model_dir = snapshot_download(
            args.model_id, 
            cache_dir=args.cache_dir,
            ignore_file_pattern='original/.*'
        )
        print(f"\n✅ 下载完成！")
        print(f"📍 模型路径: {model_dir}")
        print(f"💡 现在你可以使用该路径配合 vLLM 或 Transformers 进行对话了。")
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")

if __name__ == "__main__":
    download_model()


