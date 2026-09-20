#!/usr/bin/env python
"""Create synthetic token files and run a tiny debug training job without downloading data."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    data = Path("data")
    data.mkdir(exist_ok=True)
    rng = np.random.default_rng(1337)
    rng.integers(0, 512, size=100_000, dtype=np.uint32).tofile(data / "train.bin")
    rng.integers(0, 512, size=20_000, dtype=np.uint32).tofile(data / "val.bin")
    subprocess.run(
        [sys.executable, "-m", "bdc100m.train", "--config", "configs/debug.yaml"],
        check=True,
    )


if __name__ == "__main__":
    main()
