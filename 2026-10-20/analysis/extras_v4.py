"""DecodingTrust, BOLD and VisBias for the v4 matched-answer ORPO model.

Descriptive only (no significance tests). The untuned base and round-2 ORPO
come from the round-2 run; v4 ORPO comes from the extras-only run given on the
command line. Alica-style graphs and CSVs land in 2026-10-20/analysis/extras_v4/.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from split_panels import save_each

ROOT = Path(__file__).resolve().parents[2]
R2 = ROOT / "2026-10-03/data/private/prime_native/20261021T084030.287641_0000/recovered/results"
OUT = ROOT / "2026-10-20/analysis/extras_v4"
CONDS = ["base", "orpo_r2", "orpo_v4"]
LABELS = {"base": "BASE", "orpo_r2": "ORPO round 2", "orpo_v4": "ORPO v4"}
COLORS = {"base": "tab:blue", "orpo_r2": "seagreen", "orpo_v4": "orange"}
RACE_CATS = ["African_Americans", "Asian_Americans", "European_Americans",
             "Hispanic_and_Latino_Americans"]
REGIMES = ["benign", "untargeted", "targeted"]
ATTRS = ["income", "education"]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load(v4):
    roots = {"base": (R2, "base"), "orpo_r2": (R2, "orpo"), "orpo_v4": (v4, "orpo")}
    def get(kind, name):
        return {c: read_csv(r / "extra_benchmarks" / kind / name.format(m=m)) for c, (r, m) in roots.items()}
    return (get("decodingtrust", "gemma3_12b_{m}_native/responses.csv"),
            get("bold", "gemma3_12b_{m}_native/scored_responses.csv"),
            get("visbias", "visbias-gemma3_12b_{m}_native/responses.csv"))


def grouped(ax, keys, value, title, fmt="%.3f", rotation=0):
    x = np.arange(len(keys)); w = 0.27
    crowded = len(keys) > 8
    for i, c in enumerate(CONDS):
        b = ax.bar(x + (i - 1) * w, [value(c, k) for k in keys], w, label=LABELS[c], color=COLORS[c])
        ax.bar_label(b, fmt=fmt, fontsize=6 if crowded else 8, rotation=90 if crowded else 0, padding=2)
    names = [k.replace("_Americans", " Americans") for k in keys]
    if crowded:
        ax.set_xticks(x, names, rotation=45, ha="right")
    else:
        ax.set_xticks(x, names, rotation=rotation)
    ax.margins(y=0.12)
    ax.set_title(title); ax.axhline(0, color="k", lw=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)


def figure(stem, headline, panels):
    """Combined side-by-side graph plus one PNG per panel in separate/."""
    fig, axes = plt.subplots(1, len(panels), figsize=(22, 7))
    for ax, (_, _, draw) in zip(axes, panels):
        draw(ax)
    fig.suptitle(headline, fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / f"{stem}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    for name, size, draw in panels:
        save_each([(name, draw)], OUT / "separate", stem, size, headline)


def agree(rows):
    return sum(r["classification"] == "AGREE" for r in rows) / max(1, len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-results", type=Path, required=True,
                        help="recovered/results folder of the v4 extras-only run")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dt, bold, vb = load(args.v4_results)
    summary = []

    groups = sorted({r["demographic_group"] for r in dt["base"]})
    figure("decodingtrust", "DecodingTrust stereotype agreement (lower is better)", [
        ("by_regime", (8, 5), lambda ax: grouped(ax, REGIMES + ["overall"], lambda c, g: agree(
            [r for r in dt[c] if g == "overall" or r["regime"] == g]), "Agreement rate by regime")),
        ("by_group", (16, 6), lambda ax: grouped(
            ax, groups, lambda c, g: agree([r for r in dt[c] if r["demographic_group"] == g]),
            "Agreement rate by demographic group"))])
    for c in CONDS:
        for g in REGIMES + ["overall"]:
            rows = [r for r in dt[c] if g == "overall" or r["regime"] == g]
            summary.append(["decodingtrust", c, "agreement_rate_" + g, len(rows), round(agree(rows), 4)])
        summary.append(["decodingtrust", c, "no_response_rate", len(dt[c]),
                        round(sum(r["classification"] == "NO_RESPONSE" for r in dt[c]) / len(dt[c]), 4)])

    metrics = {"sentiment": lambda r: num(r["vader_compound"]),
               "toxicity": lambda r: 1.0 if str(r.get("toxic_any", "")).lower() == "true" else 0.0,
               "regard_positive": lambda r: 1.0 if r.get("regard_class", "").lower() == "positive" else 0.0}
    def cat_mean(c, g, metric="sentiment"):
        return float(np.nanmean([metrics[metric](r) for r in bold[c] if r["category"] == g]))
    domains = sorted({r["domain"] for r in bold["base"]})
    figure("bold", "BOLD continuations (smaller spread across race categories is fairer)", [
        ("by_race_category", (9, 5), lambda ax: grouped(
            ax, RACE_CATS, cat_mean, "Mean VADER sentiment by race category", rotation=15)),
        ("by_domain", (9, 5), lambda ax: grouped(ax, domains, lambda c, d: float(np.nanmean(
            [num(r["vader_compound"]) for r in bold[c] if r["domain"] == d])),
            "Mean sentiment by domain", rotation=15))])
    for c in CONDS:
        for m in metrics:
            means = [cat_mean(c, g, m) for g in RACE_CATS]
            summary.append(["bold", c, m + "_race_category_spread",
                            sum(r["category"] in RACE_CATS for r in bold[c]), round(max(means) - min(means), 4)])
            summary.append(["bold", c, m + "_overall_mean", len(bold[c]),
                            round(float(np.nanmean([metrics[m](r) for r in bold[c]])), 4)])

    comp = {c: [r for r in vb[c] if r["task_kind"] == "comparison"] for c in CONDS}
    occs = sorted({r["occupation"] for r in comp["base"]})
    def share(c, a, race):
        rows = [r for r in comp[c] if r["attribute"] == a]
        return sum(r["selected_race"] == race for r in rows) / max(1, len(rows))
    def by_attribute(ax):
        x = np.arange(len(ATTRS)); w = 0.27
        for i, c in enumerate(CONDS):
            for j, race in enumerate(["black", "white"]):
                b = ax.bar(x + (i - 1) * w + (j - 0.5) * w / 2, [share(c, a, race) for a in ATTRS], w / 2,
                           label=f"{LABELS[c]} -> {race.title()}", color=COLORS[c], alpha=1.0 if j == 0 else 0.5)
                ax.bar_label(b, fmt="%.3f", fontsize=8)
        ax.set_xticks(x, ATTRS); ax.set_title("Race selection rate by attribute"); ax.margins(y=0.12)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    figure("visbias", "VisBias matched-pair comparisons (0.5/0.5 or 'cannot determine' is unbiased)", [
        ("by_attribute", (9, 5), by_attribute),
        ("by_occupation", (10, 5), lambda ax: grouped(ax, occs, lambda c, o: (
            sum(r["selected_race"] == "black" for r in comp[c] if r["occupation"] == o)
            - sum(r["selected_race"] == "white" for r in comp[c] if r["occupation"] == o))
            / max(1, sum(r["occupation"] == o for r in comp[c])),
            "Black minus White selection share by occupation", fmt="%.2f", rotation=20))])
    for c in CONDS:
        for a in ATTRS:
            rows = [r for r in comp[c] if r["attribute"] == a]
            pairs = defaultdict(dict)
            for r in rows:
                pairs[r["matched_pair_id"]][r["order"]] = r["selected_race"]
            both = [p for p in pairs.values() if len(p) == 2]
            picked = share(c, a, "black") + share(c, a, "white")
            summary += [["visbias", c, a + "_cannot_determine_rate", len(rows),
                         round(sum(r["parsed_answer"] == "CANNOT_DETERMINE" for r in rows) / len(rows), 4)],
                        ["visbias", c, a + "_black_selection_rate", len(rows), round(share(c, a, "black"), 4)],
                        ["visbias", c, a + "_white_selection_rate", len(rows), round(share(c, a, "white"), 4)],
                        ["visbias", c, a + "_black_share_of_picks", len(rows),
                         round(share(c, a, "black") / picked, 4) if picked else ""],
                        ["visbias", c, a + "_order_consistency_rate", len(both),
                         round(sum(p["1"] == p["2"] for p in both) / len(both), 4) if both else ""]]

    with open(OUT / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark", "condition", "metric", "n", "value"])
        writer.writerows(summary)
    (OUT / "provenance.json").write_text(json.dumps({
        "base_and_orpo_r2": str(R2.relative_to(ROOT)), "orpo_v4": str(args.v4_results.resolve().relative_to(ROOT)),
        "significance_tests": False}, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
