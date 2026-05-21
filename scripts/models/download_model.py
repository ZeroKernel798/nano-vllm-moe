from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a model to an explicit local directory.")
    parser.add_argument(
        "--model-id",
        default="qwen/Qwen2.5-7B-Instruct",
        help="ModelScope model id or Hugging Face repo id.",
    )
    parser.add_argument(
        "--local-dir",
        default="/root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct",
        help="Final model directory. Use a fresh path when correcting a mislabeled checkpoint.",
    )
    parser.add_argument(
        "--cache-dir",
        default="/root/autodl-tmp/.cache",
        help="Download cache directory. Put this on the data disk on AutoDL.",
    )
    parser.add_argument(
        "--source",
        choices=("modelscope", "huggingface"),
        default="modelscope",
        help="Download backend. ModelScope is usually better on AutoDL/mainland network.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default="https://hf-mirror.com",
        help="HF endpoint used only with --source huggingface.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir).expanduser()
    cache_dir = Path(args.cache_dir).expanduser()
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model_id={args.model_id}")
    print(f"local_dir={local_dir}")
    print(f"cache_dir={cache_dir}")

    if args.source == "modelscope":
        from modelscope import snapshot_download

        model_dir = snapshot_download(
            model_id=args.model_id,
            local_dir=str(local_dir),
            cache_dir=str(cache_dir),
        )
    else:
        os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(
            repo_id=args.model_id,
            local_dir=str(local_dir),
            cache_dir=str(cache_dir),
        )

    print(f"Download complete: {model_dir}")


if __name__ == "__main__":
    main()
