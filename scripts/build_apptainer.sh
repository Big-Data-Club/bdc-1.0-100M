#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEF_FILE="${BDC_APPTAINER_DEF:-$PROJECT_ROOT/apptainer/Apptainer.def}"
IMAGE="${BDC_IMAGE:-$PROJECT_ROOT/images/bdc-1.0-100M.sif}"
BASE_IMAGE="${BDC_BASE_IMAGE:-pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime}"

if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: apptainer is not available in PATH." >&2
  exit 1
fi

mkdir -p "$(dirname "$IMAGE")"
cd "$PROJECT_ROOT"

# Optional site-specific flags, e.g.:
#   BDC_APPTAINER_BUILD_FLAGS="--fakeroot" ./scripts/build_apptainer.sh
# shellcheck disable=SC2206
BUILD_FLAGS=(${BDC_APPTAINER_BUILD_FLAGS:-})

echo "Building BDC-1.0-100M Apptainer image"
echo "  definition: $DEF_FILE"
echo "  base image:  $BASE_IMAGE"
echo "  output:      $IMAGE"

apptainer build \
  "${BUILD_FLAGS[@]}" \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  "$IMAGE" \
  "$DEF_FILE"

echo "Built: $IMAGE"
