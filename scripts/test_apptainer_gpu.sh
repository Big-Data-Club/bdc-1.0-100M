#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${BDC_IMAGE:-$PROJECT_ROOT/images/bdc-1.0-100M.sif}"

if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: apptainer is not available in PATH." >&2
  exit 1
fi
if [[ ! -f "$IMAGE" ]]; then
  echo "ERROR: image not found: $IMAGE" >&2
  exit 1
fi

apptainer exec --nv "$IMAGE" python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda runtime:', torch.version.cuda)
print('gpu count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'gpu[{i}]:', torch.cuda.get_device_name(i))
if not torch.cuda.is_available():
    raise SystemExit('CUDA is not available inside the container')
PY
