.PHONY: sync test smoke inspect train apptainer-build apptainer-test apptainer-inspect slurm-apptainer

sync:
	uv sync --extra dev

test:
	uv run pytest

smoke:
	uv run python scripts/smoke_test.py

inspect:
	uv run bdc-inspect --config configs/bdc_1_0_100m.yaml

train:
	uv run bdc-train --config configs/bdc_1_0_100m.yaml

apptainer-build:
	./scripts/build_apptainer.sh

apptainer-test:
	./scripts/test_apptainer_gpu.sh

apptainer-inspect:
	BDC_APPTAINER_NV=0 ./scripts/run_apptainer.sh bdc-inspect --config configs/bdc_1_0_100m.yaml

slurm-apptainer:
	sbatch slurm/train_apptainer.sbatch
