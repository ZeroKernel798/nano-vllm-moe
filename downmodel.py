import argparse
from modelscope import snapshot_download

def download_model():
    parser = argparse.ArgumentParser(description="Llama-3.2 下载工具")
    
    # 修改为 Llama-3.2-1B 的 ModelScope ID
    parser.add_argument(
        "--model-id", 
        type=str, 
        default="LLM-Research/Llama-3.2-1B",
        help="ModelScope 上的模型 ID"
    )
    
    # 保持你之前的存储路径逻辑
    parser.add_argument(
        "--cache-dir", 
        type=str, 
        default="/home/zerokernel_ac/huggingface",
        help="模型存储路径"
    )
    
    args = parser.parse_args()

    print(f"🚀 准备从 ModelScope 下载: {args.model_id}")
    print(f"📂 存储路径设定为: {args.cache_dir}")

    try:
        # 下载模型快照
        model_dir = snapshot_download(
            args.model_id, 
            cache_dir=args.cache_dir
        )
        print(f"\n✅ 下载完成！模型已存至: {model_dir}")
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")

if __name__ == "__main__":
    download_model()