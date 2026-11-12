#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-all}"
UTK_REVISION="e80f21fb21fa3631380c3cb634b79be3ac2f4cba"
UTK_SHA256="2c0655397b498e81b82d685ee74bee8c2e4364a5897caf3afec6ea48929024a7"
UTK_ARCHIVE="data/private/utkface/UTKFace.tar.gz"
UTK_URL="https://huggingface.co/datasets/py97/UTKFace-Cropped/resolve/${UTK_REVISION}/UTKFace.tar.gz?download=true"
VISBIAS_REPO="https://github.com/uscnlp-lime/VisBias.git"
VISBIAS_REVISION="6a129c5d6e0d2278325811196ac33a1bc522408d"
VISBIAS_SOURCE="data/private/visbias/source"

prepare_utkface() {
  mkdir -p "$(dirname "$UTK_ARCHIVE")" data/private/utkface/raw
  if [[ ! -f "$UTK_ARCHIVE" ]]; then
    curl --fail --location "$UTK_URL" --output "$UTK_ARCHIVE"
  fi
  printf '%s  %s\n' "$UTK_SHA256" "$UTK_ARCHIVE" | shasum -a 256 -c -
  if [[ ! -d data/private/utkface/raw/UTKFace ]]; then
    tar -xzf "$UTK_ARCHIVE" -C data/private/utkface/raw
  fi
  python3 - "$ROOT_DIR" <<'PY'
import csv
import hashlib
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
source = root / "data/private/utkface/raw/UTKFace"
destination = root / "data/private/utkface/selected"
for manifest_name, pool in (
    ("utkface_evaluation_manifest.csv", "evaluation"),
    ("utkface_finetuning_manifest.csv", "future_finetuning"),
):
    with (root / "data/frozen" / manifest_name).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        input_path = source / row["filename"]
        digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise SystemExit(f"UTKFace hash mismatch: {input_path}")
        output_path = destination / pool / row["filename"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
print("UTKFace: verified and selected 800 frozen images")
PY
}

prepare_visbias() {
  if [[ ! -d "$VISBIAS_SOURCE/.git" ]]; then
    mkdir -p "$(dirname "$VISBIAS_SOURCE")"
    git clone "$VISBIAS_REPO" "$VISBIAS_SOURCE"
  fi
  git -C "$VISBIAS_SOURCE" fetch --quiet origin "$VISBIAS_REVISION"
  git -C "$VISBIAS_SOURCE" checkout --detach "$VISBIAS_REVISION"
  python3 - "$ROOT_DIR" <<'PY'
import csv
import hashlib
import sys
from pathlib import Path

from PIL import Image, ImageOps

root = Path(sys.argv[1])
source = root / "data/private/visbias/source/source"
destination = root / "data/private/visbias/normalized"
manifest = root / "data/frozen/visbias_description_tasks.csv"
with manifest.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
for row in rows:
    input_path = (
        source
        / row["occupation"]
        / f"{row['race']}_{row['gender']}"
        / f"{row['source_index']}.jpg"
    )
    output_path = (
        destination
        / row["occupation"]
        / f"{row['race']}_{row['gender']}"
        / f"{row['source_index']}.jpg"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(input_path) as image:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        normalized.thumbnail((512, 512), Image.Resampling.LANCZOS)
        normalized.save(
            output_path,
            format="JPEG",
            quality=90,
            optimize=True,
            progressive=False,
        )
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    if digest != row["sha256"]:
        raise SystemExit(f"VisBias normalized hash mismatch: {output_path}")
print("VisBias: reconstructed and verified 84 normalized images")
PY
}

case "$MODE" in
  all)
    prepare_utkface
    prepare_visbias
    ;;
  utkface)
    prepare_utkface
    ;;
  visbias)
    prepare_visbias
    ;;
  *)
    printf 'Usage: %s [all|utkface|visbias]\n' "$0" >&2
    exit 2
    ;;
esac
