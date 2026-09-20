# Slurm training guide

> **Containerized jobs:** for the recommended Apptainer workflow, see [`APPTAINER.md`](APPTAINER.md) and use `slurm/train_apptainer.sbatch` or `slurm/train_multinode_apptainer.sbatch`.

`BDC-1.0-100M` supports single-node and multi-node GPU training through PyTorch DDP and `torchrun`.

The repository intentionally keeps scheduler-specific settings outside the Python training code. The same `bdc100m.train` entry point is therefore used locally, under `torchrun`, and under Slurm.

## 1. Assumptions

Before submitting a job:

- the repository is on a filesystem visible to the allocated compute node(s);
- `data/train.bin` and `data/val.bin` already exist;
- `uv` is available on the compute nodes;
- the project `.venv` has already been created;
- the installed PyTorch build is compatible with the cluster CUDA driver.

Prepare the environment on a login/build node:

```bash
uv sync --extra dev
uv run --no-sync bdc-inspect --config configs/bdc_1_0_100m.yaml
uv run --no-sync pytest
```

The first `uv sync` generates `uv.lock`; commit that file to your Git repository. Subsequent deployments can use `uv sync --frozen --extra dev`. Compute nodes are commonly firewalled from the Internet, so the Slurm scripts use `uv run --no-sync`: jobs reuse the prepared `.venv` instead of resolving or downloading dependencies at job start.

If your site explicitly allows dependency resolution from compute nodes, submit with `BDC_UV_SYNC=1` for the single-node launcher. For reproducible production jobs, preparing the environment ahead of time is preferred.

## 2. Single-node jobs

The portable default is one GPU:

```bash
sbatch --partition=gpu --gres=gpu:1 slurm/train.sbatch
```

Request multiple GPUs by overriding `--gres`:

```bash
sbatch --partition=gpu --gres=gpu:4 slurm/train.sbatch
```

The launcher counts the GPUs visible inside the allocation and starts:

```text
torchrun --standalone --nproc_per_node=<visible GPUs>
```

You normally do not need to set `NPROC_PER_NODE`. It is available as an escape hatch if a cluster exposes more GPUs than you want PyTorch to use.

Example:

```bash
NPROC_PER_NODE=2 sbatch --partition=gpu --gres=gpu:4 slurm/train.sbatch
```

## 3. Partition, account, QoS, and GPU type

These values are site-specific and are deliberately not committed to the repository.

Typical examples:

```bash
sbatch \
  --partition=gpu \
  --account=my_project \
  --qos=normal \
  --gres=gpu:a100:4 \
  slurm/train.sbatch
```

or on clusters using generic resources without a GPU model:

```bash
sbatch --partition=accelerated --gres=gpu:4 slurm/train.sbatch
```

Command-line `sbatch` options override the matching `#SBATCH` defaults in the script.

## 4. Environment modules

Clusters often require CUDA/compiler modules. Load them before submitting if the site exports the environment into jobs:

```bash
module load cuda/12.4
module load gcc/12
sbatch --partition=gpu slurm/train.sbatch
```

If your cluster resets modules for batch jobs, add the required `module load ...` lines near the top of the local copy of the `.sbatch` file. Do not copy a module list from another cluster blindly; CUDA and compiler module names are site-specific.

After allocation, the job prints `nvidia-smi`, the hostname, visible GPUs, config path, and process count into the Slurm log. This makes failed jobs much easier to diagnose.

## 5. Training config and checkpoints

The default config is:

```text
configs/bdc_1_0_100m.yaml
```

Override it without editing the launcher:

```bash
CONFIG=configs/bdc_1_0_100m.yaml \
  sbatch --partition=gpu --gres=gpu:4 slurm/train.sbatch
```

Checkpoint output defaults to:

```text
outputs/bdc-1.0-100M/
```

To resume, set `training.resume_from` in a config copy:

```yaml
training:
  resume_from: outputs/bdc-1.0-100M/step-00010000.pt
```

Then submit the same Slurm command.

## 6. Global batch size under DDP

The training loop computes the effective tokens per optimizer step as:

```text
micro_batch_size
× max_seq_len
× gradient_accumulation_steps
× world_size
```

The default configuration on one GPU is:

```text
2 × 1024 × 32 × 1 = 65,536 tokens / optimizer step
```

On four GPUs, leaving the config unchanged gives:

```text
2 × 1024 × 32 × 4 = 262,144 tokens / optimizer step
```

If you want to preserve the one-GPU global batch while scaling to four GPUs, reduce `gradient_accumulation_steps` from `32` to `8`.

This matters because changing global batch can change optimization behavior; multi-GPU scaling should not be treated only as a throughput switch.

## 7. Multi-node DDP

A two-node/four-GPU-per-node template is included in `slurm/train_multinode.sbatch`.

Example:

```bash
sbatch \
  --nodes=2 \
  --gres=gpu:4 \
  --partition=gpu \
  slurm/train_multinode.sbatch
```

The launcher:

1. selects the first allocated hostname as `MASTER_ADDR`;
2. derives a per-job `MASTER_PORT` from the Slurm job ID;
3. starts one `srun` task per node;
4. starts one `torchrun` worker per visible GPU on each node;
5. lets `torchrun` populate `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` for the Python DDP code.

All nodes must see the same repository, environment, training data, and output directory through shared storage.

## 8. NCCL and networking

The launchers set only:

```bash
NCCL_DEBUG=WARN
```

They intentionally do not hard-code network-interface or InfiniBand settings. Those depend on the cluster.

Useful site-specific variables include:

```bash
NCCL_SOCKET_IFNAME=ib0
NCCL_IB_DISABLE=0
NCCL_DEBUG=INFO
```

Only set them when you know the cluster topology. A wrong `NCCL_SOCKET_IFNAME` can make a healthy multi-node job hang during process-group initialization.

For debugging, increase verbosity for one run:

```bash
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL \
  sbatch --nodes=2 --gres=gpu:4 slurm/train_multinode.sbatch
```

## 9. Monitoring jobs

Common commands:

```bash
squeue -u "$USER"
sacct -j <job_id> --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode
```

Follow stdout:

```bash
tail -f logs/bdc-1.0-100M-<job_id>.out
```

The training process also writes machine-readable metrics to:

```text
outputs/bdc-1.0-100M/metrics.jsonl
```

## 10. Common failures

### `uv: command not found`

The batch environment does not expose your `uv` installation. Load the relevant module, add the user-local binary directory to `PATH`, or install `uv` in a shared software environment used by the job.

### `.venv not found`

Run from the repository root:

```bash
uv sync
```

before submitting the job.

### CUDA is unavailable inside the job

Check that the Slurm request actually allocates a GPU and inspect the `nvidia-smi` output in the job log.

### CUDA out of memory

Reduce `training.micro_batch_size` first. Preserve the desired global token batch by increasing gradient accumulation if necessary.

### Multi-node job hangs at startup

Check node-to-node networking and NCCL configuration. Start with `NCCL_DEBUG=INFO`, verify `MASTER_ADDR` is reachable by every allocated node, and use the interface settings recommended by the cluster administrators.
