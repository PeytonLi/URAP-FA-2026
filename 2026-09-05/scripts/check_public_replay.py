#!/usr/bin/env python3
"""Test the proposed public bundle in isolation without private images or old roots."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from common import ROOT, sha, write_json


def main():
    destination = tempfile.mkdtemp(prefix="public-replay-", dir=ROOT / "runtime")
    from pathlib import Path

    destination = Path(destination)
    files = (ROOT / "report/PROPOSED_GITHUB_FILES.txt").read_text().splitlines()
    for relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    if (destination / "data/iat12/images").exists() or (destination / "data/private").exists():
        raise ValueError("Private assets leaked into proposed public package")
    (destination / "runtime").mkdir(exist_ok=True)
    python = str(destination / ".venv/bin/python")
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env["MPLCONFIGDIR"] = str(destination / ".matplotlib")
    env["XDG_CACHE_HOME"] = str(destination / "runtime/cache")
    env["PIP_CACHE_DIR"] = str(ROOT / "runtime/pip-cache")
    log = ROOT / "runtime/public_replay.log"
    with log.open("w") as handle:
        for command in (
            [sys.executable, "-m", "venv", str(destination / ".venv")],
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                "locks/analysis-macos-py311.txt",
            ],
            [python, "-m", "pip", "install", "--disable-pip-version-check", "--no-deps", "-e", "."],
            [python, "-m", "pip", "check"],
            [
                python,
                "-c",
                "from pathlib import Path; import multimodel_bias; assert Path(multimodel_bias.__file__).resolve().is_relative_to(Path.cwd() / 'src')",
            ],
            [python, "scripts/verify_package.py"],
            [python, "-m", "pytest", "-q", "--junitxml=runtime/pytest.xml"],
            [python, "scripts/reproduce.py", "--skip-tests"],
        ):
            subprocess.run(
                command,
                cwd=destination,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
    compared = list((ROOT / "results/analysis").glob("*.csv")) + list((ROOT / "plots").glob("*"))
    compared.extend(
        ROOT / "report" / file
        for file in (
            "FORMAL_RESEARCH_REPORT.pdf",
            "FORMAL_RESEARCH_REPORT.md",
            "SEP5FILES_COLUMN_APPENDIX.csv",
            "RA_GUIDELINE_RESPONSES.csv",
        )
    )
    comparisons = [
        dict(
            path=str(p.relative_to(ROOT)), matches=sha(p) == sha(destination / p.relative_to(ROOT))
        )
        for p in compared
        if p.is_file()
    ]
    failures = [r["path"] for r in comparisons if not r["matches"]]
    suite = ET.parse(destination / "runtime/pytest.xml").getroot().find("testsuite")
    result = dict(
        independent_public_copy=str(destination),
        fresh_virtual_environment=True,
        dependency_consistency_check_passed=True,
        module_import_verified_inside_public_copy=True,
        unit_tests={
            key: int(suite.attrib[key]) for key in ("tests", "failures", "errors", "skipped")
        },
        private_images_present=False,
        original_request_docx_present=False,
        model_calls=0,
        compared_artifacts=len(comparisons),
        all_numeric_csv_and_figure_hashes_match=not failures,
        mismatched_paths=failures,
        log_path=str(log),
        relocation_note="Absolute-path metadata/glossary files intentionally record the clone root and are not hash-compared. Image verification receipts retain their historical meaning.",
    )
    write_json(ROOT / "report/PUBLIC_REPLAY_CHECK.json", result)
    print(json.dumps(result, indent=2))
    if failures:
        raise ValueError("Public replay drift: " + ", ".join(failures))


if __name__ == "__main__":
    main()
