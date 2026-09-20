#!/usr/bin/env bash
#SBATCH --job-name=bdc-1.0-100M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Portable single-node Slurm launcher for BDC-1.0-100M.
# Override site-specific resources on the sbatch command line, for example:
#   sbatch --partition=gpu --account=my_account --gres=gpu:4 slurm/train.sbatch

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG="${CONFIG:-configs/bdc_1_0_100m.yaml}"
cd "$PROJECT_ROOT"

mkdir -p logs

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not available in PATH on the compute node." >&2
  echo "Load your cluster Python/uv module or activate an environment that provides uv." >&2
  exit 1
fi

# Compute nodes are often offline. By default we use the already-created .venv
# without resolving/downloading dependencies. Set BDC_UV_SYNC=1 only if compute
# nodes are allowed to access the package index/cache.
if [[ "${BDC_UV_SYNC:-0}" == "1" ]]; then
  uv sync
fi

if [[ ! -d .venv ]]; then
  echo "ERROR: .venv not found in $PROJECT_ROOT" >&2
  echo "Run 'uv sync' once on a login/build node before sbatch." >&2
  exit 1
fi

# Detect the number of GPUs visible inside the Slurm allocation.
detect_gpu_count() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local visible="${CUDA_VISIBLE_DEVICES// /}"
    if [[ "$visible" == "NoDevFiles" || "$visible" == "-1" ]]; then
      echo 0
      return
    fi
    awk -F',' '{print NF}' <<< "$visible"
    return
  fi
  if [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ ]]; then
    echo "$SLURM_GPUS_ON_NODE"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L | wc -l
    return
  fi
  echo 0
}

NPROC_PER_NODE="${NPROC_PER_NODE:-$(detect_gpu_count)}"
if [[ "$NPROC_PER_NODE" -lt 1 ]]; then
  echo "ERROR: no GPU is visible in this Slurm allocation." >&2
  exit 1
fi

# Conservative NCCL defaults. Site-specific values can be exported before sbatch.
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( CPUS_PER_TASK / NPROC_PER_NODE ))}"
if [[ "$OMP_NUM_THREADS" -lt 1 ]]; then export OMP_NUM_THREADS=1; fi
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1

# Keep caches off a slow shared home directory when node-local scratch exists.
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SLURM_TMPDIR/torchinductor}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SLURM_TMPDIR/triton}"
  mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
fi

echo "============================================================"
echo "BDC-1.0-100M Slurm job"
echo "job_id=${SLURM_JOB_ID:-local}"
echo "host=$(hostname)"
echo "project_root=$PROJECT_ROOT"
echo "config=$CONFIG"
echo "nproc_per_node=$NPROC_PER_NODE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"
nvidia-smi || true

exec uv run --no-sync torchrun \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  -m bdc100m.train \
  --config "$CONFIG"
