from __future__ import annotations

import numpy as np
import torch

from bdc100m.data import TokenBin


def test_token_bin_batch(tmp_path) -> None:
    path = tmp_path / "tokens.bin"
    np.arange(1000, dtype=np.uint32).tofile(path)
    ds = TokenBin(path)
    gen = torch.Generator().manual_seed(1)
    x, y = ds.get_batch(4, 16, torch.device("cpu"), gen)
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    assert torch.all(y[:, :-1] == x[:, 1:])
