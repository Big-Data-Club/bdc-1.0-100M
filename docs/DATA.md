# Data format

Training uses a simple contiguous token stream stored as raw `uint32` integers.

```text
data/
├── train.bin
├── val.bin
└── metadata.json
```

Each source document is tokenized and terminated with `<eos>`. Documents are concatenated into one token stream. During pretraining, batches sample random contiguous windows of `max_seq_len + 1` tokens; the first `max_seq_len` tokens are inputs and the shifted sequence is the target.

This format is intentionally simple:

- memory-mapped, so the complete corpus does not need to fit in RAM;
- low Python overhead during training;
- easy to shard later when scaling to multi-node training.

For serious web-scale corpora, the next step is to add quality filtering, exact/fuzzy deduplication, PII filtering, and multiple `.bin` shards instead of one file.
