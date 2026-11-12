"""Rebuild Prompt 9 word-count graphs and primary statistical tests from frozen paired responses."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path[:0] = [
    str(ROOT),
    str(REPO / "2026-10-27/analysis"),
    str(REPO / "2026-10-20/analysis"),
    str(REPO / "2026-10-03"),
]

import iat_significance_v4 as stats
import variety
from finetune_significance import holm

PAIR = ROOT / "prompt_sweep_B/prompt_9/paired_responses.csv"
OUT = ROOT / "prompt_9_rebuild"
SWEEP_SOURCES = stats.FINAL_SOURCES | {"fallback_invalid"}


def main():
    if OUT.exists():
        raise SystemExit(f"Refuse to overwrite existing {OUT}")
    rows = list(csv.DictReader(PAIR.open(newline="", encoding="utf-8")))
    base = [r for r in rows if r["condition"] == "base"]
    tuned = [r for r in rows if r["condition"] == "tuned"]
    for side in (base, tuned):
        for r in side:
            r.pop("condition", None)
    stats.require(len(base) == 600 and len(tuned) == 600, "Prompt 9 needs 600 paired rows per condition")

    tests, per_image, _ = stats.analyze(base, tuned, SWEEP_SOURCES)
    for row in tests:
        row["family"] = "negative_control_" + row["family"]
        row["prompt"] = "9"
        row["run_prompt_id"] = "p9"
        row["interpretation"] = "no_face_gray_image_control"
    # Holm within this prompt's primary family only (rebuild is single-prompt).
    primary = [r for r in tests if r["family"] == "negative_control_primary"]
    for row, p in zip(primary, holm([r["raw_p"] for r in primary]), strict=True):
        row["holm_p"] = p
        row["multiplicity_family"] = "negative_control_primary_prompt_9_rebuild"

    OUT.mkdir(parents=True)
    variety.write_csv(OUT / "tests.csv", tests)
    flat = [{"prompt": row["prompt"], "run_prompt_id": row["run_prompt_id"],
             "interpretation": row["interpretation"], **stats.flat_rows([row])[0]} for row in tests]
    variety.write_csv(OUT / "statistics_table.csv", flat)

    # Graphs from temporary response CSVs matching variety.main expectations.
    tmp = OUT / "_tmp"
    tmp.mkdir()
    for label, data in (("base", base), ("tuned", tuned)):
        path = tmp / f"{label}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    variety.main([("base", tmp / "base.csv"), ("tuned", tmp / "tuned.csv")],
                 OUT / "graphs", title_prefix="Prompt 9")

    primary_out = [r for r in flat if r["Family"] == "negative_control_primary"]
    (OUT / "PRIMARY.md").write_text(
        "# Prompt 9 primary tests (rebuild)\n\n"
        + "\n".join(
            f"- **{r['Test']}**: change={r['Change']}, 95% CI={r['95% CI']}, "
            f"raw p={r['raw p']}, Holm p={r['Holm p']}"
            for r in primary_out
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "out": str(OUT),
        "graph_tuned": str(OUT / "graphs/word_counts_tuned.png"),
        "graph_base": str(OUT / "graphs/word_counts_base.png"),
        "primary": primary_out,
        "diagnostics": [r for r in flat if r["Family"] == "negative_control_diagnostics"
                        and r["View"] == "sampled_exact"],
    }, indent=2))


if __name__ == "__main__":
    main()
