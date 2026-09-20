#!/usr/bin/env bash
set -euo pipefail
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
uv run torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m bdc100m.train --config configs/bdc_1_0_100m.yaml
