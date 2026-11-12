from __future__ import annotations

import argparse
import json
from pathlib import Path

from .significance import build_all_significance_outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build explicit p/q-value figures from frozen August 24 results."
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("reference/august24/results"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/archived_significance"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=31082026)
    args = parser.parse_args()
    result = build_all_significance_outputs(
        args.reference_root,
        args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
