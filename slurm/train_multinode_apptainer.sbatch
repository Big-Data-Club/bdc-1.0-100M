#!/usr/bin/env bash
#SBATCH --job-name=bdc-1.0-100M
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Multi-node DDP inside Apptainer. One Slurm task is launched per node and each
# task starts one torchrun worker per GPU allocated on that node.

set -euo pipefail

export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export CONFIG="${CONFIG:-configs/bdc_1_0_100m.yaml}"
export BDC_IMAGE="${BDC_IMAGE:-$PROJECT_ROOT/images/bdc-1.0-100M.sif}"
cd "$PROJECT_ROOT"
mkdir -p logs

if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: apptainer is not available. Load the cluster Apptainer module." >&2
  exit 1
fi
export APPTAINER_BIN="$(command -v apptainer)"
if [[ ! -f "$BDC_IMAGE" ]]; then
  echo "ERROR: Apptainer image not found: $BDC_IMAGE" >&2
  exit 1
fi

export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
export MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1
export APPTAINERENV_MASTER_ADDR="$MASTER_ADDR"
export APPTAINERENV_MASTER_PORT="$MASTER_PORT"
export APPTAINERENV_NCCL_DEBUG="$NCCL_DEBUG"
export APPTAINERENV_PYTHONUNBUFFERED=1
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"

# Propagate optional network tuning selected by the cluster administrator/user.
for name in NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_DISABLE NCCL_NET_GDR_LEVEL; do
  if [[ -n "${!name:-}" ]]; then
    export "APPTAINERENV_${name}=${!name}"
  fi
done

echo "============================================================"
echo "BDC-1.0-100M / multi-node Slurm + Apptainer"
echo "job_id=$SLURM_JOB_ID nodes=$SLURM_NNODES"
echo "master=$MASTER_ADDR:$MASTER_PORT"
echo "image=$BDC_IMAGE"
echo "config=$CONFIG"
echo "============================================================"

srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -lc '
  set -euo pipefail
  cd "$PROJECT_ROOT"

  if [[ ! -x "$APPTAINER_BIN" ]]; then
    echo "ERROR: apptainer executable is not accessible on $(hostname): $APPTAINER_BIN" >&2
    exit 1
  fi

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
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( CPUS_PER_TASK / NPROC_PER_NODE ))}"
  if [[ "$OMP_NUM_THREADS" -lt 1 ]]; then OMP_NUM_THREADS=1; fi
  export OMP_NUM_THREADS
  export APPTAINERENV_OMP_NUM_THREADS="$OMP_NUM_THREADS"

  APPTAINER_ARGS=(--nv --bind "$PROJECT_ROOT:/workspace")
  if [[ -n "${SLURM_TMPDIR:-}" && -d "$SLURM_TMPDIR" ]]; then
    APPTAINER_ARGS+=(--bind "$SLURM_TMPDIR:$SLURM_TMPDIR")
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SLURM_TMPDIR/torchinductor}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SLURM_TMPDIR/triton}"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
    export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR"
    export APPTAINERENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
  fi
  if [[ -n "${BDC_EXTRA_BINDS:-}" ]]; then
    APPTAINER_ARGS+=(--bind "$BDC_EXTRA_BINDS")
  fi

  echo "node=$(hostname) node_rank=$SLURM_PROCID gpus=$NPROC_PER_NODE master=$MASTER_ADDR:$MASTER_PORT"

  exec "$APPTAINER_BIN" exec "${APPTAINER_ARGS[@]}" "$BDC_IMAGE" \
    bash -lc '\''cd /workspace && exec torchrun \
      --nnodes="$1" \
      --nproc_per_node="$2" \
      --node_rank="$3" \
      --master_addr="$4" \
      --master_port="$5" \
      -m bdc100m.train \
      --config "$6"'\'' bash \
      "$SLURM_NNODES" "$NPROC_PER_NODE" "$SLURM_PROCID" "$MASTER_ADDR" "$MASTER_PORT" "$CONFIG"
'
