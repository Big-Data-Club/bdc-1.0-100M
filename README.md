# BDC-1.0-100M

A clean, research-oriented repository for **pretraining a ~100M parameter decoder-only language model from scratch**.

The default model is deliberately small enough for a modest GPU, while keeping modern LLM building blocks: **RoPE, RMSNorm, grouped-query attention (GQA), SwiGLU, pre-norm residual blocks, tied embeddings, PyTorch SDPA, bf16/fp16 AMP, checkpoint/resume, and optional DDP**.

## Architecture

```text
Vocabulary            32,000
Layers                     12
Hidden size                768
Attention heads             12
KV heads                     4
Head dimension              64
SwiGLU hidden            2,048
Context                  1,024
Parameters          100,092,672
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the exact parameter budget.

## Repository layout

```text
bdc-1.0-100M/
├── configs/
│   ├── bdc_1_0_100m.yaml     # real ~100M run
│   └── debug.yaml             # tiny CPU/GPU smoke run
├── apptainer/
│   └── Apptainer.def          # GPU-ready immutable container
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATA.md
│   ├── APPTAINER.md
│   └── SLURM.md
├── scripts/
│   ├── train_tokenizer.py
│   ├── prepare_data.py
│   ├── smoke_test.py
│   ├── build_apptainer.sh
│   ├── run_apptainer.sh
│   ├── test_apptainer_gpu.sh
│   ├── train_single_gpu.sh
│   └── train_ddp.sh
├── slurm/
│   ├── train.sbatch                       # host uv environment
│   ├── train_multinode.sbatch             # host uv, multi-node
│   ├── train_apptainer.sbatch             # Apptainer, 1 node
│   └── train_multinode_apptainer.sbatch   # Apptainer, multi-node
├── logs/                      # Slurm stdout/stderr (gitignored)
├── src/bdc100m/
│   ├── config.py
│   ├── model.py
│   ├── data.py
│   ├── checkpoint.py
│   ├── distributed.py
│   ├── train.py
│   ├── evaluate.py
│   ├── generate.py
│   └── inspect_model.py
├── tests/
├── pyproject.toml
├── Makefile
└── README.md
```

## 1. Environment with `uv`

```bash
uv sync --extra dev
```

If you want optional Weights & Biases later:

```bash
uv sync --extra dev --extra tracking
```

## 2. Verify the model before downloading data

```bash
uv run bdc-inspect --config configs/bdc_1_0_100m.yaml
uv run pytest
```

Expected parameter count:

```text
100,092,672
```

You can also run an end-to-end synthetic smoke test:

```bash
uv run python scripts/smoke_test.py
```

This creates random token files and trains the tiny debug model for 20 steps. It verifies model forward/backward, optimizer, evaluation, logging, and checkpointing without any external dataset.

## 3. Train a tokenizer

For the first PoC, TinyStories is intentionally easy to debug:

```bash
uv run python scripts/train_tokenizer.py \
  --dataset roneneldan/TinyStories \
  --split train \
  --vocab-size 32000 \
  --max-docs 200000 \
  --output-dir artifacts/tokenizer
```

For a Vietnamese/English model later, train the tokenizer on a representative mixed corpus rather than TinyStories only.

## 4. Prepare tokenized training data

```bash
uv run python scripts/prepare_data.py \
  --dataset roneneldan/TinyStories \
  --tokenizer artifacts/tokenizer/tokenizer.json \
  --output-dir data
```

This creates raw memory-mapped token streams:

```text
data/train.bin
data/val.bin
data/metadata.json
```

## 5. Pretrain on one GPU

```bash
uv run bdc-train --config configs/bdc_1_0_100m.yaml
```

or:

```bash
./scripts/train_single_gpu.sh
```

The conservative default is:

```text
micro batch         2 sequences
sequence length     1,024
accumulation        32
world size          1
-------------------------
effective batch     65,536 tokens / optimizer step
```

If VRAM allows it, increase `micro_batch_size` first. Gradient accumulation can then be reduced while keeping a similar global token batch.

## 6. Resume training

Set in `configs/bdc_1_0_100m.yaml`:

```yaml
resume_from: outputs/bdc-1.0-100M/step-00001000.pt
```

Then run the same training command.

Checkpoints contain model weights, optimizer state, step, config, and RNG state.

## 7. Evaluate

```bash
uv run bdc-eval \
  --config configs/bdc_1_0_100m.yaml \
  --checkpoint outputs/bdc-1.0-100M/step-00001000.pt \
  --batches 100
```

## 8. Generate

```bash
uv run bdc-generate \
  --config configs/bdc_1_0_100m.yaml \
  --checkpoint outputs/bdc-1.0-100M/step-00001000.pt \
  --tokenizer artifacts/tokenizer/tokenizer.json \
  --prompt "Once upon a time" \
  --max-new-tokens 128
```

## 9. Multi-GPU DDP

The same training loop supports single-node DDP:

```bash
NPROC_PER_NODE=2 ./scripts/train_ddp.sh
```

The effective token batch automatically scales with `world_size`.

## 10. Apptainer GPU image

Build an immutable training image from the repository root:

```bash
./scripts/build_apptainer.sh
```

The default image is `images/bdc-1.0-100M.sif` and is based on
`pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime`. Test GPU passthrough inside an
interactive GPU allocation:

```bash
./scripts/test_apptainer_gpu.sh
```

Run training directly in the image:

```bash
./scripts/run_apptainer.sh \
  bdc-train --config configs/bdc_1_0_100m.yaml
```

Or submit the containerized job to Slurm:

```bash
sbatch --partition=gpu --gres=gpu:4 slurm/train_apptainer.sbatch
```

Two-node example:

```bash
sbatch --nodes=2 --gres=gpu:4 --partition=gpu \
  slurm/train_multinode_apptainer.sbatch
```

See [`docs/APPTAINER.md`](docs/APPTAINER.md) for image builds, CUDA base
overrides, external dataset binds, scratch/cache handling, and multi-node NCCL.

## 11. Submit with Slurm (host `uv` environment)

Prepare the `uv` environment once on a login/build node. Create the project environment once on a login/build node:

```bash
uv sync --extra dev
```

`uv sync` will generate `uv.lock`. Commit that lock file in your Git repository; after that, use `uv sync --frozen --extra dev` for reproducible cluster deployments.

The default Slurm job requests one GPU and automatically detects how many GPUs Slurm exposes to the job:

```bash
sbatch --partition=gpu --gres=gpu:1 slurm/train.sbatch
```

For a 4-GPU node, override the resource request without editing the file:

```bash
sbatch --partition=gpu --gres=gpu:4 slurm/train.sbatch
```

If your cluster requires an account/QoS:

```bash
sbatch \
  --partition=gpu \
  --account=YOUR_ACCOUNT \
  --qos=YOUR_QOS \
  --gres=gpu:4 \
  slurm/train.sbatch
```

Use a different config through the exported `CONFIG` variable:

```bash
CONFIG=configs/bdc_1_0_100m.yaml \
  sbatch --partition=gpu --gres=gpu:4 slurm/train.sbatch
```

Logs are written as:

```text
logs/bdc-1.0-100M-<job_id>.out
logs/bdc-1.0-100M-<job_id>.err
```

A multi-node template is also included:

```bash
sbatch --nodes=2 --gres=gpu:4 --partition=gpu slurm/train_multinode.sbatch
```

The Slurm launchers deliberately do **not** hard-code `partition`, `account`, CUDA modules, InfiniBand interface names, or other site-specific settings. See [`docs/SLURM.md`](docs/SLURM.md) for cluster setup, module loading, NCCL tuning, multi-node launch, and troubleshooting.

## Training outputs

```text
outputs/bdc-1.0-100M/
├── metrics.jsonl
├── step-00001000.pt
├── step-00002000.pt
└── ...
```

`metrics.jsonl` records train loss, LR, gradient norm, throughput, validation loss and perplexity and is intentionally tool-agnostic.

## Practical GPU starting points

These are starting configurations, not guarantees; activation memory depends on PyTorch/CUDA/kernel versions.

| VRAM | Suggested starting point |
|---:|---|
| 8–12 GB | `micro_batch_size: 1`, fp16/bf16, seq 1024 |
| 16 GB | `micro_batch_size: 1–2` |
| 24 GB | `micro_batch_size: 2–4` |
| 48 GB+ | increase micro-batch and/or context |

If OOM occurs, reduce `micro_batch_size` before reducing sequence length. Preserve the desired global token batch with gradient accumulation.

## What this repository intentionally does not hide

The repository uses a small amount of direct PyTorch instead of a large training framework so that the important mechanics stay visible:

- exact architecture implementation;
- causal language-model objective;
- GQA head expansion;
- RoPE application;
- mixed precision;
- gradient accumulation;
- gradient clipping;
- cosine learning-rate schedule;
- checkpoint/resume;
- DDP synchronization.

That makes it suitable both as a PoC and as a base for research extensions such as FlashAttention-specific kernels, activation checkpointing, FSDP, dataset sharding, sequence packing, distributed checkpointing, profiling, MFU measurement, and multi-node training.

## Recommended development path

1. Run `pytest` and the synthetic smoke test.
2. Train the debug model on a small TinyStories subset.
3. Train the 100M model for a short run and verify loss decreases.
4. Replace TinyStories with a cleaned Vietnamese/English corpus.
5. Benchmark tokenizer fertility and validation perplexity by language.
6. Add larger context / FSDP / multi-node only after the single-GPU baseline is reproducible.

## License

Apache-2.0.
# bdc-1.0-100M
