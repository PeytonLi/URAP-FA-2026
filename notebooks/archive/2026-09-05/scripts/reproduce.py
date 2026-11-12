#!/usr/bin/env python3
"""Offline replay of statistics, figures and evidence report from preserved responses."""

import argparse
import subprocess
import sys

from common import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    if not args.skip_tests:
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)
    for script in (
        "analyze_evidence.py",
        "plot_evidence.py",
        "build_evidence_records.py",
        "build_report.py",
        "audit_sep5files.py",
        "build_formal_report.py",
    ):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)
    print("Offline replay complete. No model calls or training were performed.")


if __name__ == "__main__":
    main()
