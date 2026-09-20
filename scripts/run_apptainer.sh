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
  echo "Build it with ./scripts/build_apptainer.sh" >&2
  exit 1
fi

NV_ARGS=()
if [[ "${BDC_APPTAINER_NV:-1}" == "1" ]]; then
  NV_ARGS+=(--nv)
fi

BIND_ARGS=(--bind "$PROJECT_ROOT:/workspace")
if [[ -n "${BDC_EXTRA_BINDS:-}" ]]; then
  BIND_ARGS+=(--bind "$BDC_EXTRA_BINDS")
fi

if [[ "$#" -eq 0 ]]; then
  set -- bdc-inspect --config configs/bdc_1_0_100m.yaml
fi

exec apptainer exec \
  "${NV_ARGS[@]}" \
  "${BIND_ARGS[@]}" \
  "$IMAGE" \
  bash -lc 'cd /workspace && exec "$@"' bash "$@"
