# Apptainer on Slurm

BDC-1.0-100M includes a GPU-ready Apptainer definition and Slurm launchers. The
container contains Python, CUDA-enabled PyTorch and the installed `bdc100m`
package. Training data, configs, checkpoints, and logs remain outside the image.
This keeps the SIF immutable while allowing long-running jobs to write outputs.

## Layout

```text
apptainer/Apptainer.def
images/bdc-1.0-100M.sif          # generated, not committed
scripts/build_apptainer.sh
scripts/run_apptainer.sh
scripts/test_apptainer_gpu.sh
slurm/train_apptainer.sbatch
slurm/train_multinode_apptainer.sbatch
```

## 1. Load Apptainer

Clusters commonly provide Apptainer through environment modules:

```bash
module avail apptainer
module load apptainer
apptainer --version
```

The exact module name is site-specific.

## 2. Build the SIF

Build from the repository root because `%files` in the definition copies the
project package into the image:

```bash
./scripts/build_apptainer.sh
```

Default output:

```text
images/bdc-1.0-100M.sif
```

If your site requires fakeroot:

```bash
BDC_APPTAINER_BUILD_FLAGS="--fakeroot" ./scripts/build_apptainer.sh
```

If the cluster does not permit local image builds, build the SIF on a compatible
Linux build host and copy the resulting `.sif` to shared storage.

### Override the CUDA/PyTorch base

The default is:

```text
pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
```

Override without changing the definition file:

```bash
BDC_BASE_IMAGE="pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime" \
  ./scripts/build_apptainer.sh
```

Choose a CUDA runtime compatible with the NVIDIA driver installed on the
cluster. At runtime Apptainer `--nv` exposes the allocated NVIDIA devices and
host driver libraries to the container.

## 3. GPU sanity check

Run this on a GPU node or inside an interactive GPU allocation:

```bash
./scripts/test_apptainer_gpu.sh
```

Or directly:

```bash
apptainer exec --nv images/bdc-1.0-100M.sif python -c \
  'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())'
```

## 4. Run interactively

The wrapper binds the repository to `/workspace` so relative paths in the YAML
config resolve to the host `data/` and `outputs/` directories:

```bash
./scripts/run_apptainer.sh \
  bdc-train --config configs/bdc_1_0_100m.yaml
```

Additional storage can be bound with:

```bash
BDC_EXTRA_BINDS="/scratch/$USER:/scratch/$USER" \
  ./scripts/run_apptainer.sh bdc-inspect --config configs/bdc_1_0_100m.yaml
```

## 5. Slurm: one node

One GPU:

```bash
sbatch \
  --partition=gpu \
  --gres=gpu:1 \
  slurm/train_apptainer.sbatch
```

Four GPUs on one node:

```bash
sbatch \
  --partition=gpu \
  --gres=gpu:4 \
  slurm/train_apptainer.sbatch
```

Use an image stored elsewhere:

```bash
BDC_IMAGE=/shared/containers/bdc-1.0-100M.sif \
  sbatch --partition=gpu --gres=gpu:4 slurm/train_apptainer.sbatch
```

The job uses `torchrun --standalone` and derives `--nproc_per_node` from the GPUs
Slurm exposes to the allocation.

## 6. Slurm: multiple nodes

Example with two nodes and four GPUs per node:

```bash
sbatch \
  --nodes=2 \
  --gres=gpu:4 \
  --partition=gpu \
  slurm/train_multinode_apptainer.sbatch
```

The launcher chooses the first allocated host as `MASTER_ADDR`, creates a stable
per-job `MASTER_PORT`, starts one Slurm task per node, and then starts one
`torchrun` process per allocated GPU.

For InfiniBand / RoCE clusters, site-specific NCCL settings can be passed through
without modifying the script, for example:

```bash
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
sbatch ... slurm/train_multinode_apptainer.sbatch
```

Do not copy network-interface values from another cluster blindly; use the
interface names recommended by your cluster administrators.

## 7. External datasets and checkpoint storage

For large corpora, do not put data into the SIF. Keep it on high-throughput
shared/local storage and bind it into the container. Example:

```bash
export BDC_EXTRA_BINDS="/scratch/$USER/bdc-data:/datasets,/scratch/$USER/bdc-output:/runs"
```

Then point a training config at paths such as:

```yaml
training:
  train_data: /datasets/train.bin
  val_data: /datasets/val.bin
  output_dir: /runs/bdc-1.0-100M
```

This lets the same immutable SIF be reused across experiments.

## 8. Reproducibility model

The SIF freezes the Python/CUDA userspace and BDC source at image-build time.
The host provides the NVIDIA kernel driver through `apptainer --nv`. Configs,
data and output directories are deliberately external to the image. If source
code changes, rebuild the SIF before a production experiment so the code in the
image and experiment metadata stay aligned.
