#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'Python 3.11 is required. Set PYTHON_BIN to its executable.\n' >&2
  exit 1
fi
"$PYTHON_BIN" -c \
  'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)' || {
  printf 'The tested environment requires Python 3.11 exactly.\n' >&2
  exit 1
}

"$PYTHON_BIN" -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install --no-deps -e .
