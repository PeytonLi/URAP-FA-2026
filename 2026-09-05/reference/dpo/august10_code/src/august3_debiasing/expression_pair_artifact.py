from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .constants import (
    EXPRESSION_PAIR_AUDIT_VERSION,
    EXPRESSION_PAIR_MINIMUM_CONFIDENCE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_expression_pair_artifact(
    metadata_csv: Path, manifest_path: Path
) -> dict[str, Any]:
    metadata_csv = metadata_csv.resolve()
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("expression-pair audit manifest is not complete")
    if manifest.get("released_metadata_is_partial") is not False:
        raise ValueError("expression-pair metadata is marked partial")
    if manifest.get("policy_version") != EXPRESSION_PAIR_AUDIT_VERSION:
        raise ValueError("expression-pair audit policy version is missing or stale")
    if manifest.get("race_or_gender_metadata_in_prompt") is not False:
        raise ValueError("expression-pair audit did not prove blinded prompts")
    try:
        minimum_confidence = float(manifest["minimum_confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("expression-pair audit confidence is invalid") from exc
    if (
        not math.isfinite(minimum_confidence)
        or minimum_confidence < EXPRESSION_PAIR_MINIMUM_CONFIDENCE
        or minimum_confidence > 1.0
    ):
        raise ValueError("expression-pair audit confidence is below policy")
    if manifest.get("accepted_metadata_csv_sha256") != sha256_file(metadata_csv):
        raise ValueError("expression-pair metadata CSV differs from its manifest")

    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        row_count = sum(1 for _ in csv.DictReader(handle))
    if row_count == 0 or row_count % 2:
        raise ValueError("expression-pair metadata must contain complete nonempty pairs")
    selected_pairs = int(manifest.get("selected_pairs", -1))
    if selected_pairs != row_count // 2:
        raise ValueError("expression-pair manifest selected-pair count differs from CSV")
    if int(manifest.get("accepted_pairs", selected_pairs)) != selected_pairs:
        raise ValueError("expression-pair manifest accepted-pair count is inconsistent")
    target = int(manifest.get("target_accepted_pairs", 0))
    if target > 0 and selected_pairs != target:
        raise ValueError("expression-pair artifact did not fill its declared target")
    return manifest
