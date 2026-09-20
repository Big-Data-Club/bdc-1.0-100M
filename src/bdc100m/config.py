from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    intermediate_size: int = 2048
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_word_embeddings: bool = True

    def validate(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        head_dim = self.hidden_size // self.num_attention_heads
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be >= 2")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")


@dataclass(slots=True)
class TrainingConfig:
    train_data: str = "data/train.bin"
    val_data: str = "data/val.bin"
    output_dir: str = "outputs/bdc-1.0-100M"
    seed: int = 1337
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 32
    max_steps: int = 20000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dtype: str = "bfloat16"
    compile: bool = False
    log_interval: int = 10
    eval_interval: int = 500
    eval_batches: int = 50
    checkpoint_interval: int = 1000
    keep_last_checkpoints: int = 3
    resume_from: str | None = None

    def validate(self) -> None:
        if self.micro_batch_size < 1:
            raise ValueError("micro_batch_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.warmup_steps < 0 or self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < max_steps")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16, or bfloat16")


@dataclass(slots=True)
class RunConfig:
    model: ModelConfig
    training: TrainingConfig

    def validate(self) -> None:
        self.model.validate()
        self.training.validate()


def _reject_unknown(section_name: str, cls: type, values: dict[str, Any]) -> None:
    allowed = set(cls.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in '{section_name}': {sorted(unknown)}")


def load_config(path: str | Path) -> RunConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    model_raw = raw.get("model", {})
    train_raw = raw.get("training", {})
    _reject_unknown("model", ModelConfig, model_raw)
    _reject_unknown("training", TrainingConfig, train_raw)

    cfg = RunConfig(ModelConfig(**model_raw), TrainingConfig(**train_raw))
    cfg.validate()
    return cfg
