from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr(
    step: int,
    max_steps: int,
    warmup_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def human_count(n: int) -> str:
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}" if unit else str(n)
        n /= 1000
    return f"{n:.2f}P"
