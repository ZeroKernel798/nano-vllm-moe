from __future__ import annotations

import argparse
from pathlib import Path


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--text-file")
    parser.add_argument("--dataset-cache-dir", default="", help="HuggingFace datasets cache directory")


def load_eval_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8")
    from datasets import load_dataset

    load_kwargs = {}
    if args.dataset_cache_dir:
        load_kwargs["cache_dir"] = str(Path(args.dataset_cache_dir).expanduser())
    raw = load_dataset(args.dataset, args.dataset_config, split=args.split, **load_kwargs)
    return "\n\n".join(text for text in raw[args.text_column] if text)
