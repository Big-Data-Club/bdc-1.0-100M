from __future__ import annotations

import torch

from bdc100m.config import ModelConfig
from bdc100m.model import BDCModel


def test_100m_parameter_count() -> None:
    cfg = ModelConfig()
    model = BDCModel(cfg)
    assert model.num_parameters() == 100_092_672


def test_forward_and_loss() -> None:
    cfg = ModelConfig(
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=176,
        max_seq_len=32,
    )
    model = BDCModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)


def test_tied_embeddings() -> None:
    model = BDCModel(ModelConfig(vocab_size=256))
    assert model.lm_head.weight.data_ptr() == model.tok_embeddings.weight.data_ptr()
