from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TokenBin:
    """Memory-mapped contiguous uint32 token stream."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Token file not found: {self.path}")
        self.tokens = np.memmap(self.path, dtype=np.uint32, mode="r")
        if len(self.tokens) < 2:
            raise ValueError(f"Token file is too small: {self.path}")

    def __len__(self) -> int:
        return len(self.tokens)

    def get_batch(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.tokens) - seq_len - 1
        if max_start < 0:
            raise ValueError(
                f"Dataset has {len(self.tokens)} tokens but needs at least {seq_len + 1}"
            )
        starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator).tolist()
        arr = np.stack([np.asarray(self.tokens[s : s + seq_len + 1], dtype=np.int64) for s in starts])
        batch = torch.from_numpy(arr)
        x = batch[:, :-1].to(device=device, non_blocking=True)
        y = batch[:, 1:].to(device=device, non_blocking=True)
        return x, y
