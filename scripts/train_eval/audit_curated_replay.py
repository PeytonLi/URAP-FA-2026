"""Audit upstream selection replay in a temporary copy; never rewrite the pool."""
from __future__ import annotations

import contextlib
import importlib.metadata
import io
import json
import platform
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import study


def main():
    config = study.dataset_config()
    source = study.REPO / Path(config["manifest_path"]).parents[1]
    sys.path.insert(0, str(source / "src"))
    import curation

    expected_path = study.REPO / config["manifest_path"]
    expected = study.read_csv(expected_path)
    original_root = curation.ROOT
    with tempfile.TemporaryDirectory(prefix="urap-curation-replay-") as folder:
        temporary_root = Path(folder)
        shutil.copytree(source / "data", temporary_root / "data")
        shutil.copytree(source / "config", temporary_root / "config")
        try:
            curation.ROOT = temporary_root
            upstream_config = json.loads((temporary_root / "config/curation.json").read_text())
            with contextlib.redirect_stdout(io.StringIO()):
                curation.select(upstream_config, materialize=False)
            replay_path = temporary_root / "data/selected_images.csv"
            replay = study.read_csv(replay_path)
            replay_sha256 = study.sha(replay_path)
        finally:
            curation.ROOT = original_root

    fields_changed = Counter()
    examples = []
    if len(expected) == len(replay):
        for original, regenerated in zip(expected, replay):
            for field in original.keys() | regenerated.keys():
                if original.get(field) != regenerated.get(field):
                    fields_changed[field] += 1
                    if len(examples) < 3:
                        examples.append({"image_id": original["image_id"], "field": field,
                                         "committed": original.get(field),
                                         "replayed": regenerated.get(field)})
    normalized = [{**row, "image_path": row["image_path"].replace("\\", "/")} for row in replay]
    membership_and_splits_match = (
        {row["image_id"]: row["split"] for row in expected}
        == {row["image_id"]: row["split"] for row in replay}
    )
    versions = {name: importlib.metadata.version(name)
                for name in ("Pillow", "pyarrow", "numpy", "scipy", "huggingface_hub")}
    report = {
        "checked_at": study.now(), "source_commit": config["source_commit"],
        "committed_sha256": study.sha(expected_path), "replayed_sha256": replay_sha256,
        "byte_identical": study.sha(expected_path) == replay_sha256,
        "expected_rows": len(expected), "replayed_rows": len(replay),
        "membership_and_splits_match": membership_and_splits_match,
        "ordered_rows_identical_after_image_path_separator_normalization": expected == normalized,
        "fields_changed": dict(fields_changed), "examples": examples,
        "python": platform.python_version(), "platform": platform.system(),
        "dependency_versions": versions, "committed_source_rewritten": False,
    }
    study.save_json(study.ROOT / "results/curation_replay_audit.json", report)
    print(json.dumps(report, indent=2))
    return 0 if expected == normalized else 2


if __name__ == "__main__":
    raise SystemExit(main())
