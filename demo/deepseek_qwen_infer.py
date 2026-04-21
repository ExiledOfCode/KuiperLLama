#!/usr/bin/env python3
from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


KMDL_MAGIC = 0x4B4D444C
DEFAULT_SYSTEM_PROMPT = "你是一个乐于助人的中文 AI 助手。请用简洁、自然的中文回答。"
DEEPSEEK_MARKERS = (
    "<｜begin▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<think>",
)


def read_seq_len(checkpoint_path: str) -> int | None:
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    with path.open("rb") as fh:
        data = fh.read(64)
    if len(data) < 28:
        return None
    magic = int.from_bytes(data[:4], "little", signed=False)
    offset = 16 if magic == KMDL_MAGIC else 0
    if len(data) < offset + 28:
        return None
    values = [
        int.from_bytes(data[offset + i * 4: offset + (i + 1) * 4], "little", signed=True)
        for i in range(7)
    ]
    seq_len = values[6]
    return seq_len if 0 < seq_len < 1_000_000 else None


def normalize_prompt(prompt: str) -> str:
    text = str(prompt or "").strip()
    if not text:
        return ""
    if any(marker in text for marker in DEEPSEEK_MARKERS):
        return text
    return (
        "<｜begin▁of▁sentence｜>"
        f"{DEFAULT_SYSTEM_PROMPT}"
        f"<｜User｜>{text}"
        "<｜Assistant｜><think>\n"
    )


def sanitize_response(text: str) -> str:
    value = str(text or "")
    for marker in ("<｜end▁of▁sentence｜>",):
        pos = value.find(marker)
        if pos != -1:
            value = value[:pos]
    return value.strip()


class DeepSeekQwenInfer:
    def __init__(self, checkpoint_path: str, tokenizer_path: str, max_new_tokens: int, temperature: float):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.model_dir = Path(tokenizer_path).resolve().parent
        self.seq_len = read_seq_len(checkpoint_path) or 4096
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.temperature = max(0.0, float(temperature))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            dtype=self.dtype,
            trust_remote_code=False,
        )
        if self.device == "cuda":
            self.model = self.model.cuda()
        self.model.eval()

    def _effective_max_new_tokens(self, prompt_token_count: int, requested_max_new_tokens: int) -> int:
        requested = max(1, int(requested_max_new_tokens))
        if prompt_token_count >= self.seq_len:
            raise ValueError(
                f"当前请求超过模型上下文长度：prompt_tokens={prompt_token_count}, "
                f"max_token={requested}, seq_len={self.seq_len}。请减少历史轮数或降低 max_token。"
            )
        available = max(1, self.seq_len - prompt_token_count)
        if prompt_token_count + requested > self.seq_len:
            print(
                (
                    f"[WARN] max_new_tokens clamped from {requested} to {available} "
                    f"because prompt_tokens={prompt_token_count} and seq_len={self.seq_len}"
                ),
                file=sys.stderr,
                flush=True,
            )
            return available
        return requested

    def generate(self, prompt: str, max_new_tokens: int | None = None, temperature: float | None = None) -> tuple[str, int, float]:
        model_prompt = normalize_prompt(prompt)
        inputs = self.tokenizer(model_prompt, return_tensors="pt", add_special_tokens=False)
        prompt_token_count = int(inputs["input_ids"].shape[1])
        effective_max_new_tokens = self._effective_max_new_tokens(
            prompt_token_count,
            self.max_new_tokens if max_new_tokens is None else max_new_tokens,
        )
        temperature_value = self.temperature if temperature is None else max(0.0, float(temperature))

        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        generation_kwargs = {
            "max_new_tokens": effective_max_new_tokens,
            "do_sample": temperature_value > 0.0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature_value > 0.0:
            generation_kwargs["temperature"] = temperature_value

        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        duration = time.perf_counter() - start
        generated_ids = output_ids[0][prompt_token_count:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        if "<think>" in model_prompt and "<think>" not in response:
            response = "<think>\n" + response
        response = sanitize_response(response)
        return response, int(generated_ids.shape[0]), duration


def print_result(response: str, steps: int, duration: float, print_stats: bool) -> None:
    print("[RESPONSE_START]")
    print(response)
    print("[RESPONSE_END]")
    sys.stdout.flush()
    if print_stats:
        speed = (steps / duration) if duration > 0 else 0.0
        print(f"[STATS] steps={steps} duration={duration:.6f} steps_per_s={speed:.6f}", file=sys.stderr)
        sys.stderr.flush()


def run_single_shot(checkpoint_path: str, tokenizer_path: str, prompt: str, max_new_tokens: int, temperature: float) -> int:
    engine = DeepSeekQwenInfer(checkpoint_path, tokenizer_path, max_new_tokens, temperature)
    try:
        response, steps, duration = engine.generate(prompt)
    except Exception as exc:
        print(str(exc))
        return 0
    print_result(response, steps, duration, True)
    return 0


def run_serve(checkpoint_path: str, tokenizer_path: str, max_new_tokens: int, temperature: float) -> int:
    engine = DeepSeekQwenInfer(checkpoint_path, tokenizer_path, max_new_tokens, temperature)
    print("[READY]")
    sys.stdout.flush()

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.rstrip("\n")
        if line == "[EXIT]":
            break
        if line == "[CANCEL]":
            continue
        if line != "[PROMPT_START]":
            continue

        prompt_lines = []
        while True:
            line = sys.stdin.readline()
            if not line:
                return 0
            line = line.rstrip("\n")
            if line == "[PROMPT_END]":
                break
            prompt_lines.append(line)

        prompt = "\n".join(prompt_lines)
        try:
            response, steps, duration = engine.generate(prompt)
        except Exception as exc:
            response = str(exc)
            steps = 0
            duration = 0.0
        print_result(response, steps, duration, False)
    return 0


def parse_max_new_tokens(raw: str | None) -> int:
    if raw is None:
        return 128
    try:
        value = int(raw)
    except ValueError:
        return 128
    return value if value > 0 else 128


def parse_temperature(raw: str | None) -> float:
    if raw is None:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return value if value >= 0.0 else 0.0


def print_usage(program: str) -> None:
    print("Usage:", file=sys.stderr)
    print(f"  {program} <checkpoint_path> <tokenizer_path> <prompt> [max_new_tokens] [temperature]", file=sys.stderr)
    print(f"  {program} --serve <checkpoint_path> <tokenizer_path> [max_new_tokens] [temperature]", file=sys.stderr)


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "--serve":
        if len(argv) < 4 or len(argv) > 6:
            print_usage(argv[0])
            return 1
        checkpoint_path = argv[2]
        tokenizer_path = argv[3]
        max_new_tokens = parse_max_new_tokens(argv[4] if len(argv) >= 5 else None)
        temperature = parse_temperature(argv[5] if len(argv) >= 6 else None)
        return run_serve(checkpoint_path, tokenizer_path, max_new_tokens, temperature)

    if len(argv) < 4 or len(argv) > 6:
        print_usage(argv[0])
        return 1
    checkpoint_path = argv[1]
    tokenizer_path = argv[2]
    prompt = argv[3]
    max_new_tokens = parse_max_new_tokens(argv[4] if len(argv) >= 5 else None)
    temperature = parse_temperature(argv[5] if len(argv) >= 6 else None)
    return run_single_shot(checkpoint_path, tokenizer_path, prompt, max_new_tokens, temperature)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
