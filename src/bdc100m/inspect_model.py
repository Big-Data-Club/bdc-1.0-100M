from __future__ import annotations

import argparse

from .config import load_config
from .model import BDCModel
from .utils import human_count


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect model shape and parameter count")
    p.add_argument("--config", default="configs/bdc_1_0_100m.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    model = BDCModel(cfg.model)
    total = model.num_parameters()
    print(model)
    print(f"\nParameters: {total:,} ({human_count(total)})")
    print(f"Context length: {cfg.model.max_seq_len}")
    print(f"Vocabulary size: {cfg.model.vocab_size}")


if __name__ == "__main__":
    main()
