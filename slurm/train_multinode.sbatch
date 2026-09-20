#!/usr/bin/env bash
#SBATCH --job-name=bdc-1.0-100M
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Multi-node DDP template. Override site-specific resources on the sbatch command line.
# Example:
#   sbatch --nodes=2 --gres=gpu:4 --partition=gpu --account=my_account \
#     slurm/train_multinode.sbatch

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export CONFIG="${CONFIG:-configs/bdc_1_0_100m.yaml}"
cd "$PROJECT_ROOT"
mkdir -p logs

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not available in PATH on the compute nodes." >&2
  exit 1
fi
if [[ ! -d .venv ]]; then
  echo "ERROR: .venv not found. Run 'uv sync' on a login/build node first." >&2
  exit 1
fi

export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
export MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "BDC-1.0-100M multi-node Slurm job"
echo "job_id=$SLURM_JOB_ID nodes=$SLURM_NNODES"
echo "master=$MASTER_ADDR:$MASTER_PORT"
echo "project_root=$PROJECT_ROOT"
echo "config=$CONFIG"
echo "============================================================"

# One Slurm task per node. Each task launches one torchrun worker per GPU visible
# on that node. bdc100m.distributed reads torchrun's RANK/LOCAL_RANK/WORLD_SIZE.
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -lc '
  set -euo pipefail
  cd "$PROJECT_ROOT"

  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    visible="${CUDA_VISIBLE_DEVICES// /}"
    if [[ "$visible" == "NoDevFiles" || "$visible" == "-1" ]]; then
      NPROC_PER_NODE=0
    else
      NPROC_PER_NODE=$(awk -F"," "{print NF}" <<< "$visible")
    fi
  elif [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ ]]; then
    NPROC_PER_NODE="$SLURM_GPUS_ON_NODE"
  else
    NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
  fi

  if [[ "$NPROC_PER_NODE" -lt 1 ]]; then
    echo "ERROR: no GPU visible on $(hostname)" >&2
    exit 1
  fi

  CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( CPUS_PER_TASK / NPROC_PER_NODE ))}"
  if [[ "$OMP_NUM_THREADS" -lt 1 ]]; then export OMP_NUM_THREADS=1; fi

  if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SLURM_TMPDIR/torchinductor}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SLURM_TMPDIR/triton}"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
  fi

  echo "node=$(hostname) node_rank=$SLURM_PROCID gpus=$NPROC_PER_NODE master=$MASTER_ADDR:$MASTER_PORT"

  exec uv run --no-sync torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank="$SLURM_PROCID" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    -m bdc100m.train \
    --config "$CONFIG"
'
