#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


def write_split(
    dataset: str,
    name: str | None,
    split: str,
    text_column: str,
    tokenizer: Tokenizer,
    output: Path,
    max_docs: int | None,
) -> dict:
    ds = load_dataset(dataset, name, split=split, streaming=True)
    eos = tokenizer.token_to_id("<eos>")
    if eos is None:
        raise ValueError("Tokenizer must contain <eos>")

    output.parent.mkdir(parents=True, exist_ok=True)
    docs = 0
    tokens = 0
    with output.open("wb") as f:
        for row in tqdm(ds, desc=f"tokenizing {split}"):
            if max_docs is not None and docs >= max_docs:
                break
            text = row.get(text_column)
            if not isinstance(text, str) or not text.strip():
                continue
            ids = tokenizer.encode(text).ids + [eos]
            np.asarray(ids, dtype=np.uint32).tofile(f)
            docs += 1
            tokens += len(ids)
    return {"split": split, "documents": docs, "tokens": tokens, "path": str(output)}


def main() -> None:
    p = argparse.ArgumentParser(description="Tokenize a HF dataset into memory-mapped uint32 .bin files")
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument("--name", default=None)
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="validation")
    p.add_argument("--text-column", default="text")
    p.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    p.add_argument("--output-dir", default="data")
    p.add_argument("--max-train-docs", type=int, default=None)
    p.add_argument("--max-val-docs", type=int, default=10000)
    args = p.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    out = Path(args.output_dir)
    train_meta = write_split(
        args.dataset,
        args.name,
        args.train_split,
        args.text_column,
        tok,
        out / "train.bin",
        args.max_train_docs,
    )
    val_meta = write_split(
        args.dataset,
        args.name,
        args.val_split,
        args.text_column,
        tok,
        out / "val.bin",
        args.max_val_docs,
    )
    meta = {
        "dataset": args.dataset,
        "name": args.name,
        "vocab_size": tok.get_vocab_size(),
        "dtype": "uint32",
        "train": train_meta,
        "validation": val_meta,
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
