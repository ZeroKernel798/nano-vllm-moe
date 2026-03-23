import argparse
from modelscope import snapshot_download

def download_model():
    parser = argparse.ArgumentParser(description="模型下载工具")
    
    parser.add_argument(
        "--model-id", 
        type=str, 
        default="deepseek-ai/deepseek-moe-16b-chat",
        help="ModelScope 上的模型 ID"
    )
    
    parser.add_argument(
        "--cache-dir", 
        type=str, 
        default="/root/autodl-tmp/models",
        help="模型存储路径"
    )
    
    args = parser.parse_args()

    print(f"🚀 准备从 ModelScope 下载: {args.model_id}")
    print(f"📂 存储路径设定为: {args.cache_dir}")

    try:
        model_dir = snapshot_download(
            args.model_id, 
            cache_dir=args.cache_dir, 
            revision='master'
        )
        print(f"\n✅ 下载完成！模型已存至: {model_dir}")
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")

if __name__ == "__main__":
    download_model()