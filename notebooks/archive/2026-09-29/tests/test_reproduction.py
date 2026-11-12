"""Offline replay must retain the published images and split assignments."""
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import curation  # noqa: E402


def test_exact_frozen_selection_and_balance(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "data", tmp_path / "data")
    shutil.copytree(ROOT / "config", tmp_path / "config")
    expected = (ROOT / "data/selected_images.csv").read_bytes()
    monkeypatch.setattr(curation, "ROOT", tmp_path)
    config = json.loads((tmp_path / "config/curation.json").read_text())
    curation.select(config, materialize=False)
    assert (tmp_path / "data/selected_images.csv").read_bytes() == expected
    curation.validate(config, skip_image_files=True)
    report = json.loads((tmp_path / "data/validation.json").read_text())
    assert report["passed"] and report["selected_images"] == 2400
    assert report["image_files_verified"] is False
    with (tmp_path / "data/selected_images.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert Counter(r["dataset"] for r in rows) == {"utkface": 1000, "fairface": 1400}
    cells = Counter((r["dataset"], r["source_race"], r["source_gender"]) for r in rows)
    assert len(cells) == 24 and set(cells.values()) == {100}
    counts = Counter(r["split"] for r in rows)
    assert counts == {"train": 1440, "development": 480, "test": 480}


def test_validation_does_not_claim_missing_image_bytes(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "data", tmp_path / "data")
    shutil.copytree(ROOT / "config", tmp_path / "config")
    monkeypatch.setattr(curation, "ROOT", tmp_path)
    config = json.loads((tmp_path / "config/curation.json").read_text())
    with pytest.raises(ValueError, match="Release validation failed"):
        curation.validate(config)
    report = json.loads((tmp_path / "data/validation.json").read_text())
    assert report["passed"] is False
    assert any("Image absent or drifted" in e for e in report["errors"])
