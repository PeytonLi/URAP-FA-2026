#!/usr/bin/env python3
"""Copy selected, immutable source artifacts and record their byte hashes."""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OLD = WORKSPACE / "multimodel-bias-benchmark-august31"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    records = []

    def copy(source: Path, destination: Path, purpose: str) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and digest(destination) != digest(source):
            raise ValueError(f"Refusing to overwrite different artifact: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
        records.append(
            {
                "source_absolute_path": str(source),
                "package_path": str(destination.relative_to(ROOT)),
                "sha256": digest(destination),
                "purpose": purpose,
            }
        )

    for folder in (
        "src",
        "tests",
        "config",
        "data/frozen",
        "data/iat12",
        "data/bold",
        "data/decodingtrust",
        "reference/august24",
    ):
        for source in sorted((OLD / folder).rglob("*")):
            if not source.is_file() or "__pycache__" in source.parts or source.name == ".DS_Store":
                continue
            copy(
                source, ROOT / source.relative_to(OLD), "frozen protocol or reusable implementation"
            )
    for name in (
        "requirements.lock",
        "requirements-bold.lock",
        "requirements-report.lock",
        "LICENSE",
    ):
        copy(OLD / name, ROOT / name, "environment or license")
    copy(
        OLD / "pyproject.toml", ROOT / "reference/august31_pyproject.toml", "historical environment"
    )
    for name in (
        "FINE_TUNING_HISTORY.md",
        "docs/METHODS_AUG31.md",
        "docs/DATA_DICTIONARY.md",
        "docs/GROK_OPEN_WEIGHT_REVIEW.md",
        "docs/IMAGE_BENCHMARK_REVIEW.md",
    ):
        copy(OLD / name, ROOT / "reference/august31" / name, "historical methods; claims to verify")
    for sub in ("utkface_iat", "visbias", "archived_significance", "utkface_data"):
        for source in sorted((OLD / "results" / sub).rglob("*")):
            if source.is_file() and source.suffix in {".csv", ".json"}:
                copy(
                    source,
                    ROOT / "reference/august31" / source.relative_to(OLD),
                    "archived row-level evidence",
                )
    # Only materialize evaluation and future-pool images named by frozen manifests.
    for manifest in (ROOT / "data/frozen/utkface").glob("*image_manifest.csv"):
        with manifest.open(newline="") as handle:
            for row in csv.DictReader(handle):
                relative = row.get("source_relative_path")
                if relative:
                    source = (OLD / "data/frozen/utkface" / relative).resolve()
                    copy(source, (manifest.parent / relative).resolve(), "private frozen input")
    manifest = ROOT / "data/frozen/utkface/utkface_iat_evaluation_manifest.csv"
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            source = (OLD / "data/frozen/utkface" / row["source_relative_path"]).resolve()
            copy(
                source,
                (manifest.parent / row["source_relative_path"]).resolve(),
                "private evaluation image",
            )
    for source in sorted((OLD / "data/private/visbias/normalized").rglob("*.jpg")):
        copy(source, ROOT / source.relative_to(OLD), "private archived VisBias input")
    aug26 = WORKSPACE / "gemma3-bias-presentation-august26"
    for sub in ("results/decodingtrust", "reference"):
        for source in sorted((aug26 / sub).rglob("*")):
            if source.is_file() and source.suffix in {".csv", ".json", ".md", ".py"}:
                copy(
                    source,
                    ROOT / "reference/august26" / source.relative_to(aug26),
                    "Gemma hosted text / pinned local IAT evidence",
                )
    guide = WORKSPACE / "RA_Evidence_Request_Variable_Glossary (1).docx"
    copy(
        guide,
        ROOT / "reference/RA_Evidence_Request_Variable_Glossary.docx",
        "researcher evidence guideline",
    )
    out = ROOT / "reference/COPIED_ARTIFACTS.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Copied and checked {len(records)} source records")


if __name__ == "__main__":
    main()
