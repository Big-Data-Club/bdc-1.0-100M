#!/usr/bin/env bash
#SBATCH --job-name=bdc-1.0-100M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Single-node GPU training inside Apptainer.
# Example:
#   sbatch --partition=gpu --gres=gpu:4 slurm/train_apptainer.sbatch
# Override image/config without editing this file:
#   BDC_IMAGE=/shared/images/bdc-1.0-100M.sif CONFIG=configs/foo.yaml sbatch ...

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG="${CONFIG:-configs/bdc_1_0_100m.yaml}"
IMAGE="${BDC_IMAGE:-$PROJECT_ROOT/images/bdc-1.0-100M.sif}"
cd "$PROJECT_ROOT"
mkdir -p logs

if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: apptainer is not available on $(hostname). Load the cluster Apptainer module." >&2
  exit 1
fi
if [[ ! -f "$IMAGE" ]]; then
  echo "ERROR: Apptainer image not found: $IMAGE" >&2
  exit 1
fi

# Detect GPUs granted by Slurm, not total GPUs installed in the node.
detect_gpu_count() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local visible="${CUDA_VISIBLE_DEVICES// /}"
    if [[ "$visible" == "NoDevFiles" || "$visible" == "-1" ]]; then echo 0; return; fi
    awk -F',' '{print NF}' <<< "$visible"
    return
  fi
  if [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ ]]; then echo "$SLURM_GPUS_ON_NODE"; return; fi
  nvidia-smi -L 2>/dev/null | wc -l
}

NPROC_PER_NODE="${NPROC_PER_NODE:-$(detect_gpu_count)}"
if [[ "$NPROC_PER_NODE" -lt 1 ]]; then
  echo "ERROR: no GPU is visible in this Slurm allocation." >&2
  exit 1
fi

CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( CPUS_PER_TASK / NPROC_PER_NODE ))}"
if [[ "$OMP_NUM_THREADS" -lt 1 ]]; then export OMP_NUM_THREADS=1; fi
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1

APPTAINER_ARGS=(--nv --bind "$PROJECT_ROOT:/workspace")
if [[ -n "${SLURM_TMPDIR:-}" && -d "${SLURM_TMPDIR}" ]]; then
  APPTAINER_ARGS+=(--bind "$SLURM_TMPDIR:$SLURM_TMPDIR")
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SLURM_TMPDIR/torchinductor}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SLURM_TMPDIR/triton}"
  export HF_HOME="${HF_HOME:-$SLURM_TMPDIR/huggingface}"
  mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$HF_HOME"
fi
if [[ -n "${BDC_EXTRA_BINDS:-}" ]]; then
  APPTAINER_ARGS+=(--bind "$BDC_EXTRA_BINDS")
fi

# Make selected host variables explicit inside the container.
export APPTAINERENV_OMP_NUM_THREADS="$OMP_NUM_THREADS"
export APPTAINERENV_NCCL_DEBUG="$NCCL_DEBUG"
export APPTAINERENV_PYTHONUNBUFFERED=1
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
[[ -n "${TORCHINDUCTOR_CACHE_DIR:-}" ]] && export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR"
[[ -n "${TRITON_CACHE_DIR:-}" ]] && export APPTAINERENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
[[ -n "${HF_HOME:-}" ]] && export APPTAINERENV_HF_HOME="$HF_HOME"

echo "============================================================"
echo "BDC-1.0-100M / Slurm + Apptainer"
echo "job_id=${SLURM_JOB_ID:-local}"
echo "host=$(hostname)"
echo "image=$IMAGE"
echo "config=$CONFIG"
echo "nproc_per_node=$NPROC_PER_NODE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"

apptainer exec "${APPTAINER_ARGS[@]}" "$IMAGE" nvidia-smi || true

exec apptainer exec \
  "${APPTAINER_ARGS[@]}" \
  "$IMAGE" \
  bash -lc 'cd /workspace && exec torchrun \
    --standalone \
    --nproc_per_node="$1" \
    -m bdc100m.train \
    --config "$2"' bash "$NPROC_PER_NODE" "$CONFIG"
