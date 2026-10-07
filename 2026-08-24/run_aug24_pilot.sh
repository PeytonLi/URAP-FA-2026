#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export MPLCONFIGDIR="$ROOT_DIR/.matplotlib"

MINIMUM_BALANCE_USD="${MINIMUM_BALANCE_USD:-13.99}"
MODEL_CONFIG="config/models_explicit_aug24.json"

.venv/bin/python scripts/validate_openrouter_routes.py \
  --model-config "$MODEL_CONFIG" \
  --output results/compatibility/openrouter_route_snapshot_reproduction.json

.venv/bin/python -m multimodel_bias.cli_iat \
  --run-id iat12_new5_pilot_10x_20260824_v1 \
  --model-config "$MODEL_CONFIG" \
  --models gpt56_luna,mimo_v25,minimax_m3,step37_flash,grok46 \
  --repeats 1 --base-seed 17082026 --max-workers 5 \
  --minimum-request-interval 6.5 --max-run-cost 0.5 \
  --minimum-balance "$MINIMUM_BALANCE_USD" --bootstrap-samples 20000 \
  --max-retries 2 --timeout 60

.venv/bin/python scripts/build_iat8_results.py \
  --iterations 1 --bootstrap-samples 20000 \
  --output-run-id iat12_eight_model_pilot_1x_20260824_v1

.venv/bin/python -m multimodel_bias.cli_decodingtrust \
  --run-id decodingtrust_race_eight_model_pilot_20260824_v2 \
  --model-config "$MODEL_CONFIG" --race-focus --topics-per-group 4 \
  --base-seed 24082026 --max-workers 8 --minimum-request-interval 6.5 \
  --max-run-cost 1 --minimum-balance "$MINIMUM_BALANCE_USD" \
  --max-retries 2 --timeout 60 --retry-empty-responses

.venv/bin/python -m multimodel_bias.cli_bold \
  --run-id bold_race_eight_model_pilot_20260824_v2 \
  --model-config "$MODEL_CONFIG" --race-focus --samples-per-category 4 \
  --base-seed 24082026 --max-workers 8 --minimum-request-interval 6.5 \
  --max-run-cost 0.75 --minimum-balance "$MINIMUM_BALANCE_USD" \
  --max-retries 2 --timeout 60 --retry-empty-responses

.venv-bold/bin/python -m multimodel_bias.cli_bold_score \
  --run-id bold_race_eight_model_pilot_20260824_v2 \
  --model-config "$MODEL_CONFIG" --batch-size 16
