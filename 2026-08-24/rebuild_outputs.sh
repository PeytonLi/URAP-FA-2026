#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  printf 'Missing main environment. Run ./scripts/setup.sh first.\n' >&2
  exit 1
fi
if [[ ! -x .venv-bold/bin/python ]]; then
  printf 'Missing BOLD scorer environment. Run ./scripts/setup_bold_scorer.sh first.\n' >&2
  exit 1
fi

export MPLCONFIGDIR="$ROOT_DIR/.matplotlib"
./scripts/verify.sh
.venv/bin/python scripts/build_iat8_results.py \
  --iterations 10 --bootstrap-samples 100000 \
  --output-run-id iat12_eight_model_expanded_10x_20260824_v1
.venv/bin/python scripts/reanalyze_decodingtrust.py \
  --run-id decodingtrust_race_eight_model_pilot_20260824_v2 \
  --model-config config/models_explicit_aug24.json
.venv-bold/bin/python -m multimodel_bias.cli_bold_score \
  --run-id bold_race_eight_model_expanded_20260824_v3 \
  --model-config config/models_bold_aug24.json --batch-size 16
.venv/bin/python scripts/build_aug24_report.py \
  --iat-run-id iat12_eight_model_expanded_10x_20260824_v1 \
  --decodingtrust-run-id decodingtrust_race_eight_model_pilot_20260824_v2 \
  --bold-run-id bold_race_eight_model_expanded_20260824_v3 \
  --phase-slug expanded --phase-label "Expanded IAT/BOLD; DecodingTrust pilot" \
  --model-config config/models_explicit_aug24.json --write-main-results
.venv/bin/python scripts/verify_completed_results.py
.venv/bin/python scripts/capture_environment.py
.venv/bin/python scripts/build_study_manifest.py
.venv/bin/python scripts/sanitize_results.py --root results
.venv/bin/python scripts/sanitize_results.py --root reference
.venv/bin/python scripts/audit_artifact_privacy.py --root "$ROOT_DIR"
.venv/bin/python scripts/freeze_artifacts.py --root "$ROOT_DIR"
.venv/bin/python scripts/freeze_artifacts.py --root "$ROOT_DIR" --verify
