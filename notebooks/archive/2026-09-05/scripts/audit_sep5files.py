#!/usr/bin/env python3
"""Audit the user's three CSV exports against original row-level records."""

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
import shutil

from common import ROOT, read_csv, sha, write_csv, write_json

FILES = {"IAT": "iat12", "DecodingTrust": "decodingtrust", "BOLD": "bold"}


def difference_kind(field, exported, canonical):
    if exported == canonical:
        return "identical"
    if exported.lower() in {"true", "false"} and exported.lower() == canonical.lower():
        return "boolean_case"
    if exported.strip() == canonical.strip():
        return "whitespace_changed"
    if field == "response" and canonical.startswith("'") and exported == canonical[1:]:
        return "leading_apostrophe_removed"
    try:
        a, b = Decimal(exported), Decimal(canonical)
        return "numeric_format" if a == b else "numeric_precision_changed"
    except InvalidOperation:
        return "unexplained_text_difference"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--import-attachments", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "data/sep5files"
    destination.mkdir(parents=True, exist_ok=True)
    source_manifest = destination / "source_manifest.json"
    originals = json.loads(source_manifest.read_text()) if source_manifest.exists() else {}
    inventory, diffs = [], []
    for label, subfolder in FILES.items():
        filename = f"Significance Tests - {label}responses.csv"
        original = ROOT.parent / "sep5files" / filename
        target = destination / filename
        if args.import_attachments:
            if target.exists() and sha(target) != sha(original):
                raise ValueError(f"Attachment changed; do not overwrite frozen copy: {filename}")
            if not target.exists():
                shutil.copy2(original, target)
            originals[filename] = dict(original_absolute_path=str(original), sha256=sha(original))
        rows = read_csv(target)
        source = ROOT / "reference/august24/results" / subfolder / "responses.csv"
        archived = read_csv(source)
        lookup = {r["task_id"]: r for r in archived}
        ids = {r["task_id"] for r in rows}
        if len(ids) != len(rows) or ids != set(lookup):
            raise ValueError(f"Missing/duplicated tasks: {filename}")
        if set(rows[0]) != set(archived[0]):
            raise ValueError(f"Unexpected columns: {filename}")
        counts = Counter()
        for r in rows:
            for field, value in r.items():
                canonical = lookup[r["task_id"]][field]
                kind = difference_kind(field, value, canonical)
                if kind != "identical":
                    if kind == "numeric_precision_changed" and field not in {
                        "cost_usd",
                        "latency_seconds",
                    }:
                        raise ValueError(f"Unexpected numeric change: {filename} {field}")
                    counts[kind] += 1
                    diffs.append(
                        dict(
                            file=filename,
                            task_id=r["task_id"],
                            field=field,
                            difference_kind=kind,
                            exported_value=value,
                            canonical_value=canonical,
                        )
                    )
        if counts["unexplained_text_difference"]:
            raise ValueError(f"Unexplained attachment differences: {filename}")
        inventory.append(
            dict(
                benchmark=label,
                filename=filename,
                package_path=str(target.relative_to(ROOT)),
                original_absolute_path=originals[filename]["original_absolute_path"],
                source_sha256=sha(target),
                canonical_path=str(source.relative_to(ROOT)),
                canonical_sha256=sha(source),
                rows=len(rows),
                columns=len(rows[0]),
                unique_task_ids=len(ids),
                models=sorted({r["model_key"] for r in rows}),
                run_ids=sorted({r["run_id"] for r in rows}),
                status_counts=dict(Counter(r["status"] for r in rows)),
                differences=dict(counts),
                prompt_hash_mismatches=sum(
                    hashlib.sha256(r["prompt"].encode()).hexdigest() != r["prompt_sha256"]
                    for r in rows
                )
                if "prompt" in rows[0]
                else "prompt omitted; see frozen protocol",
                exact_input_text_replay="Use canonical_path; uploaded spreadsheet exports are not byte-identical",
            )
        )
    write_json(ROOT / "report/SEP5FILES_AUDIT.json", inventory)
    if args.import_attachments:
        write_json(source_manifest, originals)
    write_csv(ROOT / "report/SEP5FILES_DIFFERENCES.csv", diffs)
    print(
        json.dumps(
            [
                dict(
                    file=r["filename"],
                    rows=r["rows"],
                    columns=r["columns"],
                    differences=r["differences"],
                )
                for r in inventory
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
