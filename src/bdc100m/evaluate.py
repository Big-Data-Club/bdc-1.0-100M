from __future__ import annotations

import argparse
import math

import torch

from .checkpoint import load_checkpoint
from .config import load_config
from .data import TokenBin
from .model import BDCModel
from .train import evaluate, resolve_dtype


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a BDCModel checkpoint")
    p.add_argument("--config", default="configs/bdc_1_0_100m.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batches", type=int, default=100)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(cfg.training.dtype, device)
    model = BDCModel(cfg.model).to(device)
    step = load_checkpoint(args.checkpoint, model, optimizer=None, map_location=device)
    val = TokenBin(cfg.training.val_data)
    gen = torch.Generator(device="cpu").manual_seed(cfg.training.seed + 999_999)
    loss = evaluate(
        model,
        val,
        args.batches,
        cfg.training.micro_batch_size,
        cfg.model.max_seq_len,
        device,
        dtype,
        gen,
    )
    print(f"step={step} val_loss={loss:.6f} perplexity={math.exp(min(loss, 20.0)):.3f}")


if __name__ == "__main__":
    main()
