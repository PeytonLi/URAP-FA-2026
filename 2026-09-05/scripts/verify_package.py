#!/usr/bin/env python3
"""Verify checksums and privacy; --write freezes the proposed public artifact set."""

import argparse
import json
import re
import xml.etree.ElementTree as ET

from common import ROOT, read_csv, sha, utcnow, write_json

EXCLUDED_PARTS = {
    ".venv",
    ".venv-bold",
    ".model-cache",
    "runtime",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".matplotlib",
    ".git",
}
EXCLUDED_SUFFIXES = {".pyc", ".safetensors", ".bin", ".tgz", ".gz", ".docx"}


def publishable(p):
    relative = p.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts):
        return False
    if str(relative).startswith(("data/private/", "data/iat12/images/")):
        return False
    if p.suffix in EXCLUDED_SUFFIXES or p.name in {
        ".DS_Store",
        ".env",
        "runtime_preflight_unexecuted_tasks.json",
    }:
        return False
    if relative.parts[0] == "results" and "plots" in relative.parts:
        return False
    if str(relative).startswith("results/runpod/") and p.name not in {
        "download_verification.json",
        "cleanup_verification.json",
    }:
        return False
    if str(relative) == "results/preflight/access.json":
        return False
    if str(relative) in {"report/RESEARCH_REPORT_2PAGE.pdf", "report/MAIN_REPORT.md"}:
        return False
    return True


def privacy_scan(files):
    violations = []
    patterns = (
        re.compile(r"sk-(?:ant-api\d+-|or-v1-)[A-Za-z0-9_-]{25,}"),
        re.compile(r"-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----"),
        re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]{80,}"),
    )
    for p in files:
        if p.suffix not in {".py", ".json", ".jsonl", ".csv", ".md", ".txt", ".toml", ".sh"}:
            continue
        text = p.read_text(errors="replace")
        if any(pattern.search(text) for pattern in patterns):
            violations.append(str(p.relative_to(ROOT)))
    return violations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    checksum = ROOT / "SHA256SUMS"
    if not args.write:
        failures = []
        count = 0
        for line in checksum.read_text().splitlines():
            digest, relative = line.split("  ", 1)
            p = ROOT / relative
            count += 1
            if not p.is_file() or sha(p) != digest:
                failures.append(relative)
        if failures:
            raise ValueError("Checksum failures: " + ", ".join(failures))
        print(f"Verified {count} distributable artifacts")
        return
    report = ROOT / "report"
    junit = ROOT / "runtime/pytest.xml"
    test_summary = None
    if junit.exists():
        suite = ET.parse(junit).getroot().find("testsuite")
        test_summary = {
            key: int(suite.attrib[key]) for key in ("tests", "failures", "errors", "skipped")
        }
        if test_summary["failures"] or test_summary["errors"]:
            raise ValueError("Unit tests have unresolved failures")
    replay_path = report / "PUBLIC_REPLAY_CHECK.json"
    replay = json.loads(replay_path.read_text()) if replay_path.exists() else None
    if replay is not None and not replay["all_numeric_csv_and_figure_hashes_match"]:
        raise ValueError("Independent public replay has unresolved differences")
    source_checks = []
    for provenance in (
        ROOT / "reference/COPIED_ARTIFACTS.csv",
        ROOT / "reference/DPO_COPIED_ARTIFACTS.csv",
    ):
        for r in read_csv(provenance):
            p = ROOT / r["package_path"]
            # Added workflow tests live outside copied historical files; historical bytes must agree.
            source_checks.append(
                dict(path=r["package_path"], matches=p.exists() and sha(p) == r["sha256"])
            )
    failures = [r["path"] for r in source_checks if not r["matches"]]
    if failures:
        raise ValueError("Copied source/reference drift: " + ", ".join(failures))
    local = json.loads((ROOT / "results/gemma_local/manifest.json").read_text())
    if (
        not local.get("base_hashes_unchanged")
        or local.get("adapter_loaded")
        or local["completed_tasks"] != 956
    ):
        raise ValueError("Incomplete local Gemma run or base integrity failure")
    for r in read_csv(ROOT / "results/analysis/all_significance_tests.csv"):
        for key in ("raw_p_value", "bh_adjusted_q_value"):
            v = float(r[key])
            if not 0 <= v <= 1:
                raise ValueError(f"Invalid {key}: {r['benchmark']} {r['model_key']}")
    files = [p for p in sorted(ROOT.rglob("*")) if p.is_file() and publishable(p)]
    violations = privacy_scan(files)
    if violations:
        raise ValueError("Potential secret/image payload found in: " + ", ".join(violations))
    proposed = report / "PROPOSED_GITHUB_FILES.txt"
    paths = [str(p.relative_to(ROOT)) for p in files if p != checksum]
    if str(proposed.relative_to(ROOT)) not in paths:
        paths.append(str(proposed.relative_to(ROOT)))
    if "report/INTEGRITY.json" not in paths:
        paths.append("report/INTEGRITY.json")
    paths.append("SHA256SUMS")
    proposed.write_text("\n".join(sorted(set(paths))) + "\n")
    write_json(
        report / "INTEGRITY.json",
        dict(
            checked_at=utcnow(),
            copied_artifacts_verified=len(source_checks),
            local_unit_tests=test_summary,
            isolated_public_replay_verified=bool(
                replay and replay["all_numeric_csv_and_figure_hashes_match"]
            ),
            secret_and_image_payload_scan_pass=True,
            base_hashes_unchanged=True,
            planned_local_tasks=956,
            completed_local_tasks=local["completed_tasks"],
            local_valid_tasks=local["valid_tasks"],
            github_publication_authorized=True,
            github_release_ref="sep5-evidence-v1",
            publication_note="Integrity generated before release push; verify the named GitHub ref for publication status",
            private_images_excluded=True,
            neural_scoring_manifest="results/gemma_bold/scorer_manifest.json",
            checksums_cover="proposed public files except SHA256SUMS itself; private/runtime files have separate manifests",
            remaining_limits=[
                "Exact historical sampled baseline replay failed; frozen original preserved",
                "Exact historical executed code commits and immutable teacher/judge API snapshots unavailable",
                "Private image bundle needed for fresh image inference, not offline score/figure replay",
                "No successful DPO adapter or independent human capability validation",
                "36/48 local Gemma DecodingTrust outputs truncated: full-response estimate not interpretable",
            ],
        ),
    )
    files = [ROOT / name for name in sorted(set(paths)) if name != "SHA256SUMS"]
    checksum.write_text("\n".join(f"{sha(p)}  {p.relative_to(ROOT)}" for p in files) + "\n")
    print(
        f"Frozen {len(files)} distributable files; verified {len(source_checks)} historical copies; no upload"
    )


if __name__ == "__main__":
    main()
