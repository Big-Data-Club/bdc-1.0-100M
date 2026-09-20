from __future__ import annotations

import argparse

import torch
from tokenizers import Tokenizer

from .checkpoint import load_checkpoint
from .config import load_config
from .model import BDCModel


def main() -> None:
    p = argparse.ArgumentParser(description="Generate text from a BDCModel checkpoint")
    p.add_argument("--config", default="configs/bdc_1_0_100m.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    if tokenizer.get_vocab_size() != cfg.model.vocab_size:
        raise ValueError(
            f"Tokenizer vocab={tokenizer.get_vocab_size()} != model vocab={cfg.model.vocab_size}"
        )

    model = BDCModel(cfg.model).to(device)
    step = load_checkpoint(args.checkpoint, model, optimizer=None, map_location=device)
    model.eval()

    encoded = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    eos_id = tokenizer.token_to_id("<eos>")
    out = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=eos_id,
    )
    print(f"[checkpoint step {step}]")
    print(tokenizer.decode(out[0].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
