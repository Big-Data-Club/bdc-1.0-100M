from __future__ import annotations

import pytest

from bdc100m.config import ModelConfig


def test_invalid_gqa_ratio() -> None:
    cfg = ModelConfig(num_attention_heads=12, num_key_value_heads=5)
    with pytest.raises(ValueError):
        cfg.validate()
