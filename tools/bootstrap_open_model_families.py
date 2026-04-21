#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
TOOLS_DIR = ROOT_DIR / "tools"

RUNTIME_INCLUDE_PATTERNS = [
    ".gitattributes",
    "README.md",
    "LICENSE*",
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "merges.txt",
    "vocab.json",
    "chat_template.jinja",
    "added_tokens.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
]


MODEL_SPECS = {
    "llama3_2_1b_instruct": {
        "repo": "unsloth/Llama-3.2-1B-Instruct",
        "dir": MODELS_DIR / "Llama-3.2-1B-Instruct",
        "exporter": TOOLS_DIR / "export_llama_family.py",
        "output": "Llama-3.2-1B-Instruct-bf16.bin",
        "max_seq_len": 4096,
        "dtype": "bf16",
        "include": RUNTIME_INCLUDE_PATTERNS,
    },
    "tinyllama_1_1b_chat": {
        "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "dir": MODELS_DIR / "TinyLlama-1.1B-Chat-v1.0",
        "exporter": TOOLS_DIR / "export_llama_family.py",
        "output": "TinyLlama-1.1B-Chat-v1.0-bf16.bin",
        "max_seq_len": 2048,
        "dtype": "bf16",
        "include": RUNTIME_INCLUDE_PATTERNS,
    },
    "smollm2_1_7b_instruct": {
        "repo": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "dir": MODELS_DIR / "SmolLM2-1.7B-Instruct",
        "exporter": TOOLS_DIR / "export_llama_family.py",
        "output": "SmolLM2-1.7B-Instruct-bf16.bin",
        "max_seq_len": 4096,
        "dtype": "bf16",
        "include": RUNTIME_INCLUDE_PATTERNS,
    },
    "deepseek_r1_distill_qwen_1_5b": {
        "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "dir": MODELS_DIR / "DeepSeek-R1-Distill-Qwen-1.5B",
        "exporter": TOOLS_DIR / "export_qwen2_family.py",
        "output": "DeepSeek-R1-Distill-Qwen-1.5B-bf16.bin",
        "max_seq_len": 4096,
        "dtype": "bf16",
        "include": RUNTIME_INCLUDE_PATTERNS,
    },
}


def run_command(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def download_model(repo_id: str, target_dir: Path, include_patterns: list[str] | None = None) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "hf",
        "download",
        repo_id,
        "--local-dir",
        str(target_dir),
    ]
    if include_patterns:
        cmd.append("--include")
        cmd.extend(include_patterns)
    run_command(cmd)


def export_model(spec: dict) -> None:
    output_path = spec["dir"] / spec["output"]
    cmd = [
        sys.executable,
        str(spec["exporter"]),
        "--model-dir",
        str(spec["dir"]),
        "--output",
        str(output_path),
        "--max-seq-len",
        str(spec["max_seq_len"]),
        "--overwrite",
    ]
    if "dtype" in spec:
        cmd.extend(["--dtype", spec["dtype"]])
    run_command(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and export predefined open model families.")
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(MODEL_SPECS.keys()),
        help="Only process the selected model id. Can be passed multiple times.",
    )
    parser.add_argument("--skip-download", action="store_true", help="Reuse existing local model dirs.")
    parser.add_argument("--skip-export", action="store_true", help="Only download models.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_ids = args.model or list(MODEL_SPECS.keys())

    for model_id in selected_ids:
        spec = MODEL_SPECS[model_id]
        print(f"==== {model_id} ====")
        if not args.skip_download:
            download_model(spec["repo"], spec["dir"], spec.get("include"))
        if not args.skip_export:
            export_model(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
