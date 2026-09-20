#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def main() -> None:
    p = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer from a HF dataset")
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument("--name", default=None, help="Optional Hugging Face dataset config/subset name")
    p.add_argument("--split", default="train")
    p.add_argument("--text-column", default="text")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--max-docs", type=int, default=200000)
    p.add_argument("--output-dir", default="artifacts/tokenizer")
    args = p.parse_args()

    ds = load_dataset(args.dataset, args.name, split=args.split, streaming=True)

    def texts():
        for i, row in enumerate(ds):
            if i >= args.max_docs:
                break
            text = row.get(args.text_column)
            if isinstance(text, str) and text.strip():
                yield text

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts(), trainer=trainer, length=args.max_docs)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out / "tokenizer.json"))
    print(f"saved={out / 'tokenizer.json'} vocab={tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
