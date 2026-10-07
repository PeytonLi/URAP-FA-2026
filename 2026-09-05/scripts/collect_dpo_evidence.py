#!/usr/bin/env python3
"""Collect actual training inputs, outputs and source code, without checkpoints/secrets."""
from __future__ import annotations

import shutil

from common import ROOT, sha, write_csv


def main():
    records = []

    def copy(source, target, purpose):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha(target) != sha(source):
            raise ValueError(f"Source changed: {source}")
        if not target.exists():
            shutil.copy2(source, target)
        records.append({"source_absolute_path": str(source),
                        "package_path": str(target.relative_to(ROOT)), "sha256": sha(target),
                        "purpose": purpose})

    old = ROOT.parent / "gemma3-fairface-debiasing-august10"
    for folder in ("src", "scripts", "config", "docs", "evidence"):
        for path in sorted((old / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".csv", ".md"}:
                copy(path, ROOT / "reference/dpo/august10_code" / path.relative_to(old),
                     "historical pipeline source; not authorization to retrain")
    copy(old / "requirements-runpod.txt", ROOT / "reference/dpo/august10_code/requirements-runpod.txt",
         "historical training dependencies")
    for folder in ("august10_negative_expression_pilot_v4",
                   "august10_first_run_neutral12_retrospective_v1",
                   "august10_dpo_lora_overfit_diagnostic_v2"):
        for path in sorted((old / "results" / folder).rglob("*")):
            if path.is_file() and path.suffix in {".csv", ".json", ".md"}:
                copy(path, ROOT / "reference/dpo/august10" / path.relative_to(old / "results"),
                     "observed DPO data and pre/post evaluation")
    folder = ROOT.parent / "gemma3-fairface-debiasing-july19/results/" \
             "research_july19_ai_only_20260720_104900/train_ai_only"
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix in {".csv", ".json"}:
            copy(path, ROOT / "reference/dpo/july19" / path.name, "FairFace held-out and failure evidence")
    old = ROOT.parent / "gemma3-fairface-debiasing-july27"
    for folder in ("results/july19_vs_july27_comparison", "docs"):
        for path in sorted((old / folder).rglob("*")):
            if path.is_file() and path.suffix in {".csv", ".json", ".md"}:
                copy(path, ROOT / "reference/dpo/july27" / path.relative_to(old), "failure context")
    old = ROOT.parent / "gemma3-fairface-debiasing-july13"
    for folder in ("results/april29_reanalysis_20260713", "docs"):
        for path in sorted((old / folder).rglob("*")):
            if path.is_file() and path.suffix in {".csv", ".json", ".md"}:
                copy(path, ROOT / "reference/dpo/july13" / path.relative_to(old), "earlier failure context")
    write_csv(ROOT / "reference/DPO_COPIED_ARTIFACTS.csv", records)
    print(f"Verified {len(records)} DPO evidence/code artifacts")


if __name__ == "__main__":
    main()
