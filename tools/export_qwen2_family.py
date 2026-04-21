#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch


MODEL_FILE_MAGIC = 0x4B4D444C  # "KMDL"
MODEL_FILE_VERSION = 1
WEIGHT_TYPE_FP32 = 0
WEIGHT_TYPE_BF16 = 2


def serialize_fp32(file_obj, tensor: torch.Tensor) -> None:
    data = tensor.detach().to(torch.float32).contiguous().view(-1).cpu().numpy()
    file_obj.write(data.tobytes())


def serialize_bf16(file_obj, tensor: torch.Tensor) -> None:
    data = tensor.detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy()
    file_obj.write(data.tobytes())


def read_config(model_dir: Path) -> Dict:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_weight_map(model_dir: Path) -> Dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        if not weight_map:
            raise ValueError(f"Invalid weight map: {index_path}")
        return {str(k): str(v) for k, v in weight_map.items()}

    single_file = model_dir / "model.safetensors"
    if not single_file.exists():
        raise FileNotFoundError(f"Missing model.safetensors under {model_dir}")

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError("Missing dependency `safetensors`. Install with: pip install safetensors") from exc

    with safe_open(str(single_file), framework="pt", device="cpu") as reader:
        return {key: single_file.name for key in reader.keys()}


class SafeTensorReader:
    def __init__(self, model_dir: Path, weight_map: Dict[str, str]):
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError("Missing dependency `safetensors`. Install with: pip install safetensors") from exc

        self.model_dir = model_dir
        self.weight_map = weight_map
        self._handles = {}
        for file_name in sorted(set(weight_map.values())):
            file_path = model_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Missing shard file: {file_path}")
            self._handles[file_name] = safe_open(str(file_path), framework="pt", device="cpu")

    def has_key(self, key: str) -> bool:
        return key in self.weight_map

    def get_tensor(self, key: str) -> torch.Tensor:
        if key not in self.weight_map:
            raise KeyError(f"Tensor key not found: {key}")
        return self._handles[self.weight_map[key]].get_tensor(key)


def is_shared_classifier(config: Dict, reader: SafeTensorReader) -> bool:
    tie = config.get("tie_word_embeddings")
    if tie is not None:
        return bool(tie)
    return not reader.has_key("lm_head.weight")


def build_model_config(config: Dict, max_seq_len: int, shared_classifier: bool) -> bytes:
    vocab_size = int(config["vocab_size"])
    if not shared_classifier:
        vocab_size = -vocab_size
    return struct.pack(
        "iiiiiii",
        int(config["hidden_size"]),
        int(config["intermediate_size"]),
        int(config["num_hidden_layers"]),
        int(config["num_attention_heads"]),
        int(config["num_key_value_heads"]),
        vocab_size,
        int(max_seq_len),
    )


def build_file_header(dtype_name: str) -> bytes:
    weight_type = WEIGHT_TYPE_BF16 if dtype_name == "bf16" else WEIGHT_TYPE_FP32
    return struct.pack("IIii", MODEL_FILE_MAGIC, MODEL_FILE_VERSION, weight_type, 0)


def build_weight_order(config: Dict) -> List[str]:
    n_layers = int(config["num_hidden_layers"])
    names = ["model.embed_tokens.weight"]
    names.extend([f"model.layers.{i}.input_layernorm.weight" for i in range(n_layers)])
    names.extend([f"model.layers.{i}.self_attn.q_proj.weight" for i in range(n_layers)])
    names.extend([f"model.layers.{i}.self_attn.q_proj.bias" for i in range(n_layers)])
    names.extend([f"model.layers.{i}.self_attn.k_proj.weight" for i in range(n_layers)])
    names.extend([f"model.layers.{i}.self_attn.k_proj.bias" for i in range(n_layers)])
    names.extend([f"model.layers.{i}.self_attn.v_proj.weight" for i in range(n_layers)])
    names.extend([f"model.layers.{i}.self_attn.v_proj.bias" for i in range(n_layers)])
    names.extend([f"model.layers.{i}.self_attn.o_proj.weight" for i in range(n_layers)])
    names.extend([f"model.layers.{i}.post_attention_layernorm.weight" for i in range(n_layers)])
    names.extend([f"model.layers.{i}.mlp.gate_proj.weight" for i in range(n_layers)])
    names.extend([f"model.layers.{i}.mlp.down_proj.weight" for i in range(n_layers)])
    names.extend([f"model.layers.{i}.mlp.up_proj.weight" for i in range(n_layers)])
    names.append("model.norm.weight")
    return names


def validate_required_keys(reader: SafeTensorReader, ordered_names: Iterable[str], shared_classifier: bool) -> List[str]:
    missing = [name for name in ordered_names if not reader.has_key(name)]
    if not shared_classifier and not reader.has_key("lm_head.weight"):
        missing.append("lm_head.weight")
    return missing


def build_rope_cache(max_seq_len: int, head_size: int, rope_theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
    dims = torch.arange(head_size, dtype=torch.float32).unsqueeze(0)
    freqs = positions * torch.pow(torch.tensor(float(rope_theta), dtype=torch.float32), -dims / head_size)
    return torch.cos(freqs), torch.sin(freqs)


def export_qwen2_family(model_dir: Path, output_path: Path, max_seq_len: int, dtype_name: str,
                        overwrite: bool = False, check_only: bool = False) -> None:
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be > 0")
    if output_path.exists() and not overwrite and not check_only:
        raise FileExistsError(f"Output file already exists: {output_path}")

    config = read_config(model_dir)
    weight_map = load_weight_map(model_dir)
    reader = SafeTensorReader(model_dir, weight_map)
    shared_classifier = is_shared_classifier(config, reader)
    ordered_names = build_weight_order(config)
    missing = validate_required_keys(reader, ordered_names, shared_classifier)
    if missing:
        preview = "\n".join(missing[:20])
        suffix = "\n...(truncated)" if len(missing) > 20 else ""
        raise KeyError(f"Missing required tensor keys ({len(missing)}):\n{preview}{suffix}")

    head_size = int(config["hidden_size"]) // int(config["num_attention_heads"])
    rope_theta = float(config.get("rope_theta") or 1000000.0)
    cos_cache, sin_cache = build_rope_cache(max_seq_len=max_seq_len, head_size=head_size, rope_theta=rope_theta)

    print("==== Qwen2 Family Export ====")
    print(f"model_dir     : {model_dir}")
    print(f"output_path   : {output_path}")
    print(f"max_seq_len   : {max_seq_len}")
    print(f"weight_dtype  : {dtype_name}")
    print(f"rope_theta    : {rope_theta}")
    print(f"shared_cls    : {shared_classifier}")

    if check_only:
        print("[check-only] required keys and config are valid.")
        return

    serializer = serialize_bf16 if dtype_name == "bf16" else serialize_fp32
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(build_file_header(dtype_name))
        f.write(build_model_config(config, max_seq_len, shared_classifier))
        for index, name in enumerate(ordered_names, start=1):
            serializer(f, reader.get_tensor(name))
            if index % 20 == 0 or index == len(ordered_names):
                print(f"[{index:>3}/{len(ordered_names)}] {name}")
        serializer(f, cos_cache)
        serializer(f, sin_cache)
        print("[cache] rope cos/sin written")
        if not shared_classifier:
            serializer(f, reader.get_tensor("lm_head.weight"))
            print("[lm_head] written")
    print(f"wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Qwen2-compatible HuggingFace weights to Kuiper .bin")
    parser.add_argument("--model-dir", type=Path, required=True, help="Local model directory with config + safetensors.")
    parser.add_argument("--output", type=Path, required=True, help="Output .bin file path.")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Exported max_seq_len in header.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=("fp32", "bf16"),
                        help="Stored weight dtype.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists.")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        export_qwen2_family(
            model_dir=args.model_dir,
            output_path=args.output,
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
