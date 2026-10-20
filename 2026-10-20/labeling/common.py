"""Shared helpers: manifest loading, image access, JSONL draft stores."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKDIR = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "2026-09-29" / "data" / "selected_images.csv"
IMAGE_ROOT = REPO / "2026-09-29" / "private" / "dataset"
PRIVATE = WORKDIR / "data" / "private" / "target_review"

LABEL_SPLITS = ("train", "development")  # test and iat12 stay untouched


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def load_manifest() -> list[dict]:
    return read_csv(MANIFEST)


def labelable_rows(manifest=None) -> list[dict]:
    rows = manifest if manifest is not None else load_manifest()
    return [r for r in rows if r.get("screening_status") == "eligible"
            and r.get("split") in LABEL_SPLITS]


def image_file(row: dict) -> Path:
    return IMAGE_ROOT / row["image_path"]


def verify_image_file(row: dict, pixel_check: bool = True) -> bool:
    """True when the local file's PNG bytes match the manifest image_sha256,
    OR (pixel_check) its decoded pixels match pixel_sha256. The Linux rebuild
    produces identical pixels but different PNG container bytes."""
    path = image_file(row)
    if not path.is_file():
        return False
    if sha(path) == row["image_sha256"]:
        return True
    if not pixel_check:
        return False
    from PIL import Image
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        digest = hashlib.sha256(
            f"{rgb.width}x{rgb.height}|RGB|".encode() + rgb.tobytes()
        ).hexdigest()
    return digest == row.get("pixel_sha256")


class JsonlStore:
    """Append-only, resumable store keyed on a caller-chosen field."""

    def __init__(self, path: Path, key: str):
        self.path = Path(path)
        self.key = key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._records[str(rec[self.key])] = rec

    def __contains__(self, k) -> bool:
        return str(k) in self._records

    def __len__(self) -> int:
        return len(self._records)

    def get(self, k):
        return self._records.get(str(k))

    def values(self) -> list[dict]:
        return list(self._records.values())

    def append(self, rec: dict) -> None:
        k = str(rec[self.key])
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        self._records[k] = rec
