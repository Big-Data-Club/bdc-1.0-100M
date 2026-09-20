from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    keep_last: int = 3,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"step-{step:08d}.pt"
    tmp = out / f".{target.name}.tmp"

    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

    torch.save(state, tmp)
    os.replace(tmp, target)

    checkpoints = sorted(out.glob("step-*.pt"))
    if keep_last > 0 and len(checkpoints) > keep_last:
        for old in checkpoints[: -keep_last]:
            old.unlink(missing_ok=True)
    return target


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> int:
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "cuda_rng_state_all" in state:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng_state_all"]])
    return int(state.get("step", 0))
