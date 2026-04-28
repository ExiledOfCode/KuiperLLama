#!/usr/bin/env python3
"""Download Qwen/Qwen3-8B from ModelScope into the local models directory."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_OUTPUT = ROOT_DIR / "models" / "Qwen3-8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Qwen3-8B from ModelScope.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="ModelScope model id.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Local output directory, default: KuiperLLama/models/Qwen3-8B.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency `modelscope`. Install with: pip install modelscope"
        ) from exc

    print("==== ModelScope Download ====")
    print(f"model_id: {args.model_id}")
    print(f"output  : {args.output}")
    # local_dir is supported by current ModelScope versions. If older versions ignore it,
    # cache_dir still keeps the download under the project model directory.
    path = snapshot_download(
        args.model_id,
        local_dir=str(args.output),
        cache_dir=str(args.output.parent),
    )
    print(f"downloaded to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
