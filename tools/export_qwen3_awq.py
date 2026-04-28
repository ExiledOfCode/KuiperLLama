#!/usr/bin/env python3
"""
Export Qwen3 HuggingFace/ModelScope safetensors weights to KuiperLLama AWQ INT4 .bin.

This is a weight-only W4A16 path:
  - Linear weights are packed as uint4 pairs in row-major order.
  - Per-group fp32 scale and uint4 zero point are stored after each packed tensor.
  - Non-Linear tensors, such as embeddings and RMSNorm weights, are stored as bf16/fp32.

The calibration pass collects mean absolute input activations for target Linear layers.
Quantization then chooses a clipping ratio by minimizing activation-weighted reconstruction
error for each output-row/group. This keeps inference simple: no runtime activation scaling
kernel is required, and MatMul dequantizes with (q4 - zero) * scale.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

import export_qwen3 as qwen3


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT_DIR / "models" / "Qwen3-8B"
DEFAULT_OUTPUT = DEFAULT_MODEL_DIR / "Qwen3-8B-awq-int4.bin"

WEIGHT_TYPE_AWQ_INT4 = 3


DEFAULT_CALIBRATION_PROMPTS = [
    "请简要介绍大语言模型推理系统中KV Cache的作用。",
    "用C++写一个函数，计算vector<float>中所有元素的平均值。",
    "解释CUDA中线程块、线程束和共享内存之间的关系。",
    "请总结Transformer中Self-Attention和前馈网络的主要计算步骤。",
    "给出一个关于本地部署聊天机器人的用户问题和回答示例。",
    "分析模型量化为什么能够降低显存占用，以及可能带来的精度损失。",
    "请用三点说明软件系统中前后端分离架构的优点。",
    "如果模型上下文长度不足，推理服务应该如何处理用户输入？",
]


def build_awq_file_header(non_param_dtype: str) -> bytes:
    return struct.pack(
        "IIii",
        qwen3.MODEL_FILE_MAGIC,
        qwen3.MODEL_FILE_VERSION,
        WEIGHT_TYPE_AWQ_INT4,
        qwen3.data_type_value(non_param_dtype),
    )


def read_calibration_prompts(path: Optional[Path]) -> List[str]:
    if path is None:
        return DEFAULT_CALIBRATION_PROMPTS
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(raw)
        if isinstance(payload, list):
            prompts = [str(item) for item in payload]
        else:
            prompts = [str(item) for item in payload.get("prompts", [])]
    else:
        prompts = [line.strip() for line in raw.splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No calibration prompts found in {path}")
    return prompts


def module_name_from_weight_name(weight_name: str) -> str:
    return weight_name[:-7] if weight_name.endswith(".weight") else weight_name


def target_module_names(config: Dict) -> List[str]:
    return [
        module_name_from_weight_name(name)
        for name in qwen3.build_weight_order(config)
        if qwen3.should_quantize_tensor(name)
    ]


def collect_activation_stats(
    model_dir: Path,
    config: Dict,
    prompts: List[str],
    max_length: int,
    device: str,
    dtype_name: str,
    max_gpu_memory: Optional[str],
    max_cpu_memory: Optional[str],
) -> Dict[str, torch.Tensor]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Missing dependency `transformers`.") from exc

    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    if device == "cpu":
        dtype = torch.float32

    print("==== AWQ calibration ====")
    print(f"model_dir      : {model_dir}")
    print(f"calib_samples  : {len(prompts)}")
    print(f"calib_max_len  : {max_length}")
    print(f"device         : {device}")
    print(f"dtype          : {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    max_memory = None
    if device == "auto" and (max_gpu_memory or max_cpu_memory):
        max_memory = {}
        if max_gpu_memory and torch.cuda.is_available():
            max_memory[0] = max_gpu_memory
        if max_cpu_memory:
            max_memory["cpu"] = max_cpu_memory

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        device_map="auto" if device == "auto" else None,
        max_memory=max_memory,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if device not in ("auto", "cpu"):
        model.to(device)
    model.eval()

    targets = set(target_module_names(config))
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs, _output):
            if not inputs:
                return
            x = inputs[0].detach().to(torch.float32)
            if x.dim() == 1:
                x = x.view(1, -1)
            x = x.reshape(-1, x.shape[-1]).abs().mean(dim=0).cpu()
            if name not in sums:
                sums[name] = x
                counts[name] = 1
            else:
                sums[name] += x
                counts[name] += 1
        return hook

    for name, module in model.named_modules():
        if name in targets:
            handles.append(module.register_forward_hook(make_hook(name)))

    if not handles:
        raise RuntimeError("No target Linear modules were found for calibration.")

    with torch.no_grad():
        for idx, prompt in enumerate(prompts, start=1):
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            input_device = next(model.parameters()).device
            encoded = {k: v.to(input_device) for k, v in encoded.items()}
            model(**encoded)
            print(f"[{idx:>3}/{len(prompts)}] calibrated prompt tokens={encoded['input_ids'].numel()}")

    for handle in handles:
        handle.remove()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    stats = {f"{name}.weight": sums[name] / max(1, counts[name]) for name in sums}
    missing = [f"{name}.weight" for name in targets if f"{name}.weight" not in stats]
    if missing:
        print(f"[WARN] Missing activation stats for {len(missing)} tensors; they will use uniform weights.")
    return stats


def pack_uint4(values: torch.Tensor) -> torch.Tensor:
    flat = values.contiguous().view(-1).to(torch.uint8)
    if flat.numel() % 2 != 0:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8)])
    low = flat[0::2] & 0x0F
    high = (flat[1::2] & 0x0F) << 4
    return (low | high).contiguous()


def quantize_awq_int4(
    tensor: torch.Tensor,
    group_size: int,
    activation_stat: Optional[torch.Tensor],
    min_clip: float,
    search_steps: int,
    chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    data = tensor.detach().to(torch.float32).contiguous()
    if data.dim() != 2:
        raise ValueError(f"AWQ int4 only supports 2D Linear weights, got shape={tuple(data.shape)}")
    out_dim, in_dim = data.shape
    if in_dim % group_size != 0:
        raise ValueError(
            f"AWQ group_size must divide in_features. shape={tuple(data.shape)} group_size={group_size}"
        )
    if data.numel() % 2 != 0:
        raise ValueError(f"Packed int4 requires an even number of weights, got {data.numel()}")

    n_groups = in_dim // group_size
    ratios = torch.linspace(1.0, min_clip, steps=max(1, search_steps), dtype=torch.float32)
    if activation_stat is None or activation_stat.numel() != in_dim:
        act = torch.ones(in_dim, dtype=torch.float32)
    else:
        act = activation_stat.detach().to(torch.float32).cpu()
        act = act / torch.clamp(act.mean(), min=1e-6)
        act = torch.clamp(act, min=0.05, max=20.0)
    act = act.view(1, n_groups, group_size)

    q_chunks: list[torch.Tensor] = []
    scale_chunks: list[torch.Tensor] = []
    zero_chunks: list[torch.Tensor] = []
    max_err = 0.0

    for start in range(0, out_dim, chunk_rows):
        end = min(out_dim, start + chunk_rows)
        grouped = data[start:end].view(end - start, n_groups, group_size)
        best_err = torch.full((end - start, n_groups), float("inf"), dtype=torch.float32)
        best_q = torch.zeros_like(grouped, dtype=torch.uint8)
        best_scale = torch.ones((end - start, n_groups), dtype=torch.float32)
        best_zero = torch.zeros((end - start, n_groups), dtype=torch.uint8)

        base_min = grouped.amin(dim=-1, keepdim=True)
        base_max = grouped.amax(dim=-1, keepdim=True)
        center = (base_min + base_max) * 0.5
        half_range = torch.clamp((base_max - base_min) * 0.5, min=1e-6)
        for ratio in ratios:
            clip_half_range = torch.clamp(half_range * float(ratio), min=1e-6)
            clip_min = center - clip_half_range
            clip_max = center + clip_half_range
            scale = torch.clamp((clip_max - clip_min) / 15.0, min=1e-6)
            zero = torch.round(-clip_min / scale).clamp(0, 15)
            q = torch.round(grouped / scale + zero).clamp(0, 15).to(torch.uint8)
            dequant = (q.to(torch.float32) - zero) * scale
            err = ((dequant - grouped) ** 2 * act).mean(dim=-1)
            mask = err < best_err
            best_err = torch.where(mask, err, best_err)
            best_scale = torch.where(mask, scale.squeeze(-1), best_scale)
            best_zero = torch.where(mask, zero.squeeze(-1).to(torch.uint8), best_zero)
            best_q = torch.where(mask.unsqueeze(-1), q, best_q)

        dequant = (best_q.to(torch.float32) - best_zero.to(torch.float32).unsqueeze(-1)) * best_scale.unsqueeze(-1)
        max_err = max(max_err, float((dequant - grouped).abs().max().item()))
        q_chunks.append(best_q.reshape(end - start, in_dim))
        scale_chunks.append(best_scale)
        zero_chunks.append(best_zero)

    q_all = torch.cat(q_chunks, dim=0)
    scales = torch.cat(scale_chunks, dim=0).contiguous().view(-1)
    zeros = torch.cat(zero_chunks, dim=0).contiguous().view(-1)
    packed = pack_uint4(q_all)
    return packed, scales, zeros, max_err


def serialize_uint8(file_obj, tensor: torch.Tensor) -> None:
    file_obj.write(tensor.detach().to(torch.uint8).contiguous().cpu().numpy().tobytes())


def export_qwen3_awq(
    model_dir: Path,
    output_path: Path,
    max_seq_len: int,
    group_size: int,
    non_param_dtype: str,
    calib_prompts: Optional[Path],
    calib_max_len: int,
    calib_device: str,
    calib_dtype: str,
    calib_max_gpu_memory: Optional[str],
    calib_max_cpu_memory: Optional[str],
    min_clip: float,
    search_steps: int,
    chunk_rows: int,
    overwrite: bool,
    check_only: bool,
    skip_calibration: bool,
) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if output_path.exists() and not overwrite and not check_only:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite.")
    if group_size <= 0:
        raise ValueError("--group-size must be > 0")

    config = qwen3.read_config(model_dir)
    order = qwen3.build_weight_order(config)
    weight_map = qwen3.load_weight_map(model_dir)
    missing = qwen3.validate_required_keys_from_map(weight_map, order)
    if missing:
        preview = "\n".join(missing[:20])
        raise KeyError(f"Missing required tensor keys ({len(missing)}):\n{preview}")
    if check_only:
        print("[check-only] config and safetensor keys are valid.")
        return

    activation_stats: Dict[str, torch.Tensor] = {}
    if not skip_calibration:
        activation_stats = collect_activation_stats(
            model_dir=model_dir,
            config=config,
            prompts=read_calibration_prompts(calib_prompts),
            max_length=calib_max_len,
            device=calib_device,
            dtype_name=calib_dtype,
            max_gpu_memory=calib_max_gpu_memory,
            max_cpu_memory=calib_max_cpu_memory,
        )
    else:
        print("[WARN] --skip-calibration is enabled; export will use uniform activation weights.")

    print("==== Qwen3 AWQ Export ====")
    print(f"model_dir      : {model_dir}")
    print(f"output_path    : {output_path}")
    print(f"max_seq_len    : {max_seq_len}")
    print(f"num_layers     : {config['num_hidden_layers']}")
    print(f"hidden_size    : {config['hidden_size']}")
    print(f"group_size     : {group_size}")
    print(f"non_param_dtype: {non_param_dtype}")
    print(f"min_clip       : {min_clip}")
    print(f"search_steps   : {search_steps}")

    reader = qwen3.SafeTensorReader(model_dir, weight_map)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_header = build_awq_file_header(non_param_dtype)
    header = qwen3.build_header(config, max_seq_len)
    non_param_serializer = qwen3.serialize_bf16 if non_param_dtype == "bf16" else qwen3.serialize_fp32

    max_errors = []
    with output_path.open("wb") as f:
        f.write(file_header)
        f.write(header)
        f.write(struct.pack("i", group_size))

        total = len(order)
        for idx, name in enumerate(order, start=1):
            tensor = (
                qwen3.resolve_lm_head(reader, "model.embed_tokens.weight", "lm_head.weight")
                if name == "lm_head.weight"
                else reader.get_tensor(name)
            )
            if qwen3.should_quantize_tensor(name):
                packed, scales, zeros, maxerr = quantize_awq_int4(
                    tensor=tensor,
                    group_size=group_size,
                    activation_stat=activation_stats.get(name),
                    min_clip=min_clip,
                    search_steps=search_steps,
                    chunk_rows=chunk_rows,
                )
                serialize_uint8(f, packed)
                qwen3.serialize_fp32(f, scales)
                serialize_uint8(f, zeros)
                max_errors.append((maxerr, name, tuple(tensor.shape)))
                action = f"awq_int4 packed={packed.numel()} maxerr={maxerr:.6g}"
            else:
                non_param_serializer(f, tensor)
                action = "stored"
            if idx % 10 == 0 or idx == total:
                print(f"[{idx:>3}/{total}] {action} {name}")

    if max_errors:
        max_errors.sort(reverse=True)
        maxerr, name, shape = max_errors[0]
        print(f"max AWQ int4 group error: {maxerr:.6g} at {name} shape={shape}")
    print(f"wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Qwen3 AWQ INT4 .bin for KuiperLLama.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--non-param-dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--calib-prompts", type=Path, default=None)
    parser.add_argument("--calib-max-len", type=int, default=512)
    parser.add_argument("--calib-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--calib-dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--calib-max-gpu-memory", type=str, default=None)
    parser.add_argument("--calib-max-cpu-memory", type=str, default=None)
    parser.add_argument("--min-clip", type=float, default=0.80)
    parser.add_argument("--search-steps", type=int, default=8)
    parser.add_argument("--chunk-rows", type=int, default=128)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        export_qwen3_awq(
            model_dir=args.model_dir,
            output_path=args.output,
            max_seq_len=args.max_seq_len,
            group_size=args.group_size,
            non_param_dtype=args.non_param_dtype,
            calib_prompts=args.calib_prompts,
            calib_max_len=args.calib_max_len,
            calib_device=args.calib_device,
            calib_dtype=args.calib_dtype,
            calib_max_gpu_memory=args.calib_max_gpu_memory,
            calib_max_cpu_memory=args.calib_max_cpu_memory,
            min_clip=args.min_clip,
            search_steps=args.search_steps,
            chunk_rows=args.chunk_rows,
            overwrite=args.overwrite,
            check_only=args.check_only,
            skip_calibration=args.skip_calibration,
        )
    except Exception as exc:
        print(f"AWQ export failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
