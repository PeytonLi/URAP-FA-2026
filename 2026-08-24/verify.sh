#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="$ROOT_DIR/.matplotlib"
.venv/bin/ruff check src tests scripts
.venv/bin/pytest -q
bash -n scripts/*.sh
.venv/bin/python -m multimodel_bias.cli_iat \
  --run-id validation_aug24 --model-config config/models_explicit_aug24.json \
  --models all --repeats 1 --dry-run
.venv/bin/python -m multimodel_bias.cli_decodingtrust \
  --run-id validation_aug24 --model-config config/models_explicit_aug24.json \
  --models all --race-focus --topics-per-group 4 --dry-run
.venv/bin/python -m multimodel_bias.cli_bold \
  --run-id validation_aug24 --model-config config/models_bold_aug24.json \
  --models all --race-focus --samples-per-category 4 --dry-run
.venv/bin/python scripts/audit_artifact_privacy.py --root "$ROOT_DIR"
if [[ -f ARTIFACT_MANIFEST.csv ]]; then
  .venv/bin/python scripts/freeze_artifacts.py --root "$ROOT_DIR" --verify
fi
