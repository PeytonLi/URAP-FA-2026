#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from august10_research.judgment_planner import (  # noqa: E402
    PlannerPolicy,
    plan_judgment_batch,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prefilter candidates and plan a resumable Claude judging batch."
    )
    parser.add_argument("--candidate-csv", required=True, type=Path)
    parser.add_argument("--eligible-csv", action="append", type=Path, default=[])
    parser.add_argument("--exclusions-csv", action="append", type=Path, default=[])
    parser.add_argument(
        "--audit-root",
        action="append",
        type=Path,
        default=[],
        help="Discover completed paid Claude audit directories recursively.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--gate-config",
        type=Path,
        default=ROOT_DIR / "config/pilot_gate.json",
    )
    parser.add_argument("--max-iat-pairs", type=int, default=20)
    parser.add_argument("--max-profile-pairs", type=int, default=20)
    parser.add_argument("--seed", default="august10-claude-batch-v1")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    if args.max_iat_pairs < 0 or args.max_profile_pairs < 0:
        raise ValueError("batch pair limits must be nonnegative")

    candidates = read_csv(args.candidate_csv)
    if not candidates:
        raise ValueError("candidate CSV is empty")
    discovered_audits: list[Path] = []
    for root in args.audit_root:
        for manifest_path in sorted(root.rglob("claude_pair_audit_manifest.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") == "complete" and not manifest.get("dry_run", False):
                discovered_audits.append(manifest_path.parent)
    eligible_paths = list(args.eligible_csv)
    exclusion_paths = list(args.exclusions_csv)
    for directory in discovered_audits:
        eligible_paths.append(directory / "claude_judged_pairs_eligible.csv")
        exclusion_paths.append(directory / "claude_pair_exclusions.csv")
    eligible_paths = list(dict.fromkeys(path.resolve() for path in eligible_paths))
    exclusion_paths = list(dict.fromkeys(path.resolve() for path in exclusion_paths))
    eligible = [row for path in eligible_paths for row in read_csv(path)]
    exclusions = [row for path in exclusion_paths for row in read_csv(path)]
    attempted = {
        str(row.get("pair_id", "")).strip()
        for row in eligible + exclusions
        if str(row.get("pair_id", "")).strip()
    }
    gate_policy = json.loads(args.gate_config.read_text(encoding="utf-8"))["dataset"]
    planner_policy = PlannerPolicy(
        target_iat_blocks=(
            gate_policy["min_train_iat_blocks"]
            + gate_policy["min_development_iat_blocks"]
        ),
        target_profile_blocks=(
            gate_policy["min_train_profile_blocks"]
            + gate_policy["min_development_profile_blocks"]
        ),
        min_negative_iat_blocks=gate_policy["min_negative_iat_blocks"],
        preserve_ready_profile_matches_for_iat=True,
        seed=args.seed,
    )
    plan = plan_judgment_batch(
        candidates,
        eligible,
        attempted,
        max_iat_pairs=args.max_iat_pairs,
        max_profile_pairs=args.max_profile_pairs,
        policy=planner_policy,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(candidates[0])
    output_files = {}
    for name, rows in (
        ("mechanically_valid_candidates.csv", plan["mechanically_valid_rows"]),
        ("mechanical_exclusions.csv", plan["mechanical_exclusions"]),
        ("claude_iat_batch.csv", plan["planned_iat_rows"]),
        ("claude_profile_batch.csv", plan["planned_profile_rows"]),
    ):
        path = args.output_dir / name
        output_fields = fields if name != "mechanical_exclusions.csv" else ["pair_id", "reasons"]
        write_csv(path, rows, output_fields)
        output_files[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}

    manifest = {
        "protocol_version": "august10-claude-judgment-planner-v1",
        "status": "ready" if plan["planned_iat_rows"] or plan["planned_profile_rows"] else "empty",
        "source_candidate_csv": str(args.candidate_csv.resolve()),
        "source_candidate_csv_sha256": sha256_file(args.candidate_csv),
        "source_candidate_rows": len(candidates),
        "eligible_sources": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in eligible_paths
        ],
        "exclusion_sources": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in exclusion_paths
        ],
        "discovered_paid_audit_dirs": [
            str(path.resolve()) for path in discovered_audits
        ],
        "attempted_pair_count": plan["attempted_pair_count"],
        "unattempted_mechanically_valid_rows": plan[
            "unattempted_mechanically_valid_rows"
        ],
        "inventory": plan["inventory"],
        "seed": args.seed,
        "planner_policy": {
            key: getattr(planner_policy, key)
            for key in planner_policy.__dataclass_fields__
        },
        "outputs": output_files,
    }
    manifest_path = args.output_dir / "judgment_plan.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
