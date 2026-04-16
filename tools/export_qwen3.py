#!/usr/bin/env python3
"""
Export Qwen3 HuggingFace safetensors weights to KuiperLLama .bin format.

Usage examples:
  python3 tools/export_qwen3.py models/Qwen3-1.7B/Qwen3-1.7B.bin --hf=models/Qwen3-1.7B
  python3 tools/export_qwen3.py --output models/Qwen3-1.7B/Qwen3-1.7B-bf16.bin --model-dir=models/Qwen3-1.7B --dtype=bf16
  python3 tools/export_qwen3.py --output models/Qwen3-1.7B/Qwen3-1.7B.bin --model-dir=models/Qwen3-1.7B

Output weight order follows `kuiper/source/model/qwen3.cpp`:
  1) input_layernorm (all layers)
  2) post_attention_layernorm (all layers)
  3) final norm
  4) embed tokens
  5) q_proj (all layers)
  6) q_norm (all layers)
  7) k_proj (all layers)
  8) k_norm (all layers)
  9) v_proj (all layers)
  10) o_proj (all layers)
  11) gate_proj (all layers)
  12) down_proj (all layers)
  13) up_proj (all layers)
  14) lm_head
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT_DIR / "models" / "Qwen3-1.7B"
DEFAULT_OUTPUT = DEFAULT_MODEL_DIR / "Qwen3-1.7B.bin"
MODEL_FILE_MAGIC = 0x4B4D444C  # "KMDL"
MODEL_FILE_VERSION = 1
WEIGHT_TYPE_FP32 = 0
WEIGHT_TYPE_INT8 = 1
WEIGHT_TYPE_BF16 = 2


def serialize_fp32(file_obj, tensor: torch.Tensor) -> None:
    """Write one tensor in fp32 contiguous layout."""
    data = tensor.detach().to(torch.float32).contiguous().view(-1).cpu().numpy()
    file_obj.write(data.tobytes())


def serialize_bf16(file_obj, tensor: torch.Tensor) -> None:
    """Write one tensor in bf16 contiguous layout."""
    data = tensor.detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy()
    file_obj.write(data.tobytes())


def read_config(model_dir: Path) -> Dict:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_header(config: Dict, max_seq_len: int) -> bytes:
    hidden_size = int(config["hidden_size"])
    head_num = int(config["num_attention_heads"])
    kv_head_num = int(config["num_key_value_heads"])
    vocab_size = int(config["vocab_size"])
    layer_num = int(config["num_hidden_layers"])
    intermediate_size = int(config["intermediate_size"])
    head_dim = int(config.get("head_dim", hidden_size // head_num))
    dim = head_num * head_dim

    # ModelConfig layout (QWEN3_SUPPORT):
    # int32_t dim, hidden_dim, layer_num, head_num, kv_head_num, vocab_size, seq_len, immediate_dim_
    return struct.pack(
        "iiiiiiii",
        dim,
        hidden_size,
        layer_num,
        head_num,
        kv_head_num,
        vocab_size,
        int(max_seq_len),
        intermediate_size,
    )


def build_file_header(dtype_name: str) -> bytes:
    if dtype_name == "bf16":
        weight_type = WEIGHT_TYPE_BF16
    else:
        weight_type = WEIGHT_TYPE_FP32
    return struct.pack("IIii", MODEL_FILE_MAGIC, MODEL_FILE_VERSION, weight_type, 0)


def build_weight_order(config: Dict) -> List[str]:
    n_layers = int(config["num_hidden_layers"])
    names: List[str] = []

    names.extend([f"model.layers.{i}.input_layernorm.weight" for i in range(n_layers)])
    names.extend([f"model.layers.{i}.post_attention_layernorm.weight" for i in range(n_layers)])
    names.append("model.norm.weight")
    names.append("model.embed_tokens.weight")

    names.extend([f"model.layers.{i}.self_attn.q_proj.weight" for i in range(n_layers)])
    names.extend([f"model.layers.{i}.self_attn.q_norm.weight" for i in range(n_layers)])
    names.extend([f"model.layers.{i}.self_attn.k_proj.weight" for i in range(n_layers)])
    names.extend([f"model.layers.{i}.self_attn.k_norm.weight" for i in range(n_layers)])
    names.extend([f"model.layers.{i}.self_attn.v_proj.weight" for i in range(n_layers)])
    names.extend([f"model.layers.{i}.self_attn.o_proj.weight" for i in range(n_layers)])

    names.extend([f"model.layers.{i}.mlp.gate_proj.weight" for i in range(n_layers)])
    names.extend([f"model.layers.{i}.mlp.down_proj.weight" for i in range(n_layers)])
    names.extend([f"model.layers.{i}.mlp.up_proj.weight" for i in range(n_layers)])

    # lm_head is required by Kuiper qwen3 loader; for tied embeddings we reuse embed_tokens.
    names.append("lm_head.weight")
    return names


def load_weight_map(model_dir: Path) -> Dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index_data.get("weight_map", {})
        if not weight_map:
            raise ValueError(f"Invalid index file (missing weight_map): {index_path}")
        return {str(k): str(v) for k, v in weight_map.items()}

    # Single-file safetensors fallback.
    single_file = model_dir / "model.safetensors"
    if not single_file.exists():
        raise FileNotFoundError(
            f"Neither sharded index nor model.safetensors found under: {model_dir}"
        )

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError(
            "Missing dependency `safetensors`. Install with: pip install safetensors"
        ) from exc

    with safe_open(str(single_file), framework="pt", device="cpu") as reader:
        return {key: single_file.name for key in reader.keys()}


class SafeTensorReader:
    def __init__(self, model_dir: Path, weight_map: Dict[str, str]):
        self.model_dir = model_dir
        self.weight_map = weight_map
        self._handles: Dict[str, object] = {}

        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError(
                "Missing dependency `safetensors`. Install with: pip install safetensors"
            ) from exc
        self._safe_open = safe_open

        shard_files = sorted(set(weight_map.values()))
        for file_name in shard_files:
            file_path = self.model_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Missing shard file: {file_path}")
            self._handles[file_name] = self._safe_open(str(file_path), framework="pt", device="cpu")

    def has_key(self, key: str) -> bool:
        return key in self.weight_map

    def get_tensor(self, key: str) -> torch.Tensor:
        if key not in self.weight_map:
            raise KeyError(f"Tensor key not found: {key}")
        file_name = self.weight_map[key]
        return self._handles[file_name].get_tensor(key)


def resolve_lm_head(reader: SafeTensorReader, embed_key: str, lm_head_key: str) -> torch.Tensor:
    if reader.has_key(lm_head_key):
        return reader.get_tensor(lm_head_key)
    # Tied embeddings model: lm_head is absent in sharded weights.
    return reader.get_tensor(embed_key)


def validate_required_keys(reader: SafeTensorReader, ordered_names: Iterable[str]) -> List[str]:
    missing: List[str] = []
    for name in ordered_names:
        if name == "lm_head.weight":
            # tied embeddings may not have standalone lm_head.weight
            if not reader.has_key(name) and not reader.has_key("model.embed_tokens.weight"):
                missing.append(name)
        elif not reader.has_key(name):
            missing.append(name)
    return missing


def validate_required_keys_from_map(weight_map: Dict[str, str], ordered_names: Iterable[str]) -> List[str]:
    missing: List[str] = []
    for name in ordered_names:
        if name == "lm_head.weight":
            if name not in weight_map and "model.embed_tokens.weight" not in weight_map:
                missing.append(name)
        elif name not in weight_map:
            missing.append(name)
    return missing


def export_qwen3_bin(
    model_dir: Path,
    output_path: Path,
    max_seq_len: int,
    dtype_name: str,
    overwrite: bool = False,
    check_only: bool = False,
) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if output_path.exists() and not overwrite and not check_only:
        raise FileExistsError(
            f"Output file already exists: {output_path}\n"
            "Use --overwrite to replace it."
        )

    config = read_config(model_dir)
    order = build_weight_order(config)
    weight_map = load_weight_map(model_dir)

    if check_only:
        missing = validate_required_keys_from_map(weight_map, order)
    else:
        reader = SafeTensorReader(model_dir, weight_map)
        missing = validate_required_keys(reader, order)
    if missing:
        preview = "\n".join(missing[:20])
        suffix = "\n...(truncated)" if len(missing) > 20 else ""
        raise KeyError(f"Missing required tensor keys ({len(missing)}):\n{preview}{suffix}")

    print("==== Qwen3 Export ====")
    print(f"model_dir     : {model_dir}")
    print(f"output_path   : {output_path}")
    print(f"max_seq_len   : {max_seq_len}")
    print(f"num_layers    : {config['num_hidden_layers']}")
    print(f"hidden_size   : {config['hidden_size']}")
    print(f"head_num      : {config['num_attention_heads']}")
    print(f"kv_head_num   : {config['num_key_value_heads']}")
    print(f"intermediate  : {config['intermediate_size']}")
    print(f"weight_dtype  : {dtype_name}")

    if check_only:
        print("[check-only] required keys and config are valid.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_header = build_file_header(dtype_name)
    header = build_header(config, max_seq_len)
    serializer = serialize_bf16 if dtype_name == "bf16" else serialize_fp32

    with output_path.open("wb") as f:
        f.write(file_header)
        f.write(header)

        total = len(order)
        for idx, name in enumerate(order, start=1):
            if name == "lm_head.weight":
                tensor = resolve_lm_head(reader, "model.embed_tokens.weight", "lm_head.weight")
            else:
                tensor = reader.get_tensor(name)
            serializer(f, tensor)
            if idx % 20 == 0 or idx == total:
                print(f"[{idx:>3}/{total}] {name}")

    print(f"wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Qwen3 HuggingFace safetensors to KuiperLLama .bin"
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        type=Path,
        help="the output filepath, compatible with export_qwen2.py style",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .bin path. If omitted, use positional filepath or the default output path.",
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--hf",
        type=Path,
        default=None,
        help="HuggingFace/local model directory containing config.json and safetensors files.",
    )
    group.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Alias of --hf.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="Exported max_seq_len in bin header (default: 256).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=("fp32", "bf16"),
        help="Stored weight dtype in output file (default: fp32).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate config/keys only, do not write output file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.max_seq_len <= 0:
        print("Error: --max-seq-len must be > 0", file=sys.stderr)
        return 2

    model_dir = args.hf or args.model_dir or DEFAULT_MODEL_DIR
    output_path = args.output or args.filepath or DEFAULT_OUTPUT

    try:
        export_qwen3_bin(
            model_dir=model_dir,
            output_path=output_path,
            max_seq_len=args.max_seq_len,
            dtype_name=args.dtype,
            overwrite=args.overwrite,
            check_only=args.check_only,
        )
    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
