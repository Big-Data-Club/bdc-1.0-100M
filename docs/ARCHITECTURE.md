# Architecture

BDC-1.0-100M is a decoder-only, pre-norm Transformer designed to be small enough for local research while retaining the core components used by modern LLMs.

## Default 100M configuration

| Component | Value |
|---|---:|
| Vocabulary | 32,000 |
| Layers | 12 |
| Hidden size | 768 |
| Query heads | 12 |
| KV heads | 4 |
| Head dimension | 64 |
| FFN hidden size | 2,048 |
| Context length | 1,024 |
| Positional encoding | RoPE |
| Attention | Grouped-query attention (GQA) |
| Normalization | RMSNorm, pre-norm |
| MLP | SwiGLU |
| Linear biases | None |
| Embedding / LM head | Tied |
| Parameters | 100,092,672 |

## Block

```text
x
├─ RMSNorm ─ GQA + RoPE ─────────┐
└────────────────────────────── + ── h
                                  │
h ├─ RMSNorm ─ SwiGLU FFN ───────┐
  └──────────────────────────── + ── output
```

## Parameter budget

With tied token embeddings:

- Token embedding: `32,000 × 768 = 24,576,000`
- Per block attention: `1,572,864`
- Per block SwiGLU MLP: `4,718,592`
- Per block RMSNorms: `1,536`
- 12 blocks: `75,515,904`
- Final RMSNorm: `768`
- Total: **100,092,672 parameters**

The output LM head shares the token embedding weights, so it adds no independent parameters.
