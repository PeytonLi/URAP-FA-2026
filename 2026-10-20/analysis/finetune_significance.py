"""Paired base-vs-fine-tuned significance tests across all four benchmarks.

Advisor contract (2026-10-15): compare the same Gemma-3-12B-IT model before and
after LoRA fine-tuning on IAT-style 12-image, DecodingTrust (text), BOLD (text),
and VisBias (image+text) outputs; test the CHANGE in Black-vs-White disparities
directly; keep matched observations together; report effect sizes, group rates,
95% CIs, raw p-values, plus refusal/invalid/diversity diagnostics.

Every test is paired on the shared task identity (same prompt/image/iteration in
both conditions), so the analysis measures the tuning delta rather than
cross-model differences. Permutation tests use fixed seeds; exact enumeration is
used whenever the assignment space is small enough.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "2026-09-11" / "src"))

from multimodel_bias.uncertainty import (  # noqa: E402
    benjamini_hochberg,
    stratified_binary_label_permutation,
    wilson_interval,
)
from multimodel_bias.constants import NEGATIVE_IAT_SET  # noqa: E402

WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?")
IDENTITY_PATTERN = re.compile(
    r"\b(?:african(?:[- ]american)?|caucasian|european[- ]american|"
    r"(?:black|white)[- ]presenting|"
    r"(?:black|white)\s+(?:person|people|man|men|woman|women|male|female|"
    r"individual|adult))\b",
    re.IGNORECASE,
)

EXACT_SIGN_FLIP_LIMIT = 17          # 2^17 = 131,072 exact enumerations
MC_SAMPLES = 25_000
BASE_SEED = 20_261_020
DT_GROUPS = ("Black people", "White people", "Asians", "Latin Americans")
DT_REGIMES = ("benign", "untargeted", "targeted")
BOLD_METRICS = ("toxic_any", "sentiment_class", "regard_class", "vader_compound")
VISBIAS_DESCRIPTION_METRICS = (
    "vader_compound",
    "iat_negative_rate",
    "stereotype_hits_per_100_words",
    "identity_mention",
)


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------

def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a responses file (csv or jsonl) into dict rows."""
    if not path.exists():
        alt = path.with_suffix(".jsonl") if path.suffix == ".csv" else path.with_suffix(".csv")
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"Missing responses: {path} (or {alt})")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def sign_flip_pvalue(deltas: np.ndarray, seed: int) -> tuple[float, str]:
    """Two-sided paired permutation: is the mean of per-unit deltas != 0?

    Exact enumeration of all 2^n sign assignments when n <= 17; Monte Carlo
    otherwise. Under H0 each unit's (base, tuned) labels are exchangeable.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    deltas = deltas[np.isfinite(deltas)]
    n = len(deltas)
    if n == 0:
        return math.nan, "unavailable"
    observed = abs(float(deltas.mean()))
    tol = 1e-12
    if n <= EXACT_SIGN_FLIP_LIMIT:
        exceed = 0
        total = 1 << n
        for bits in range(total):
            signs = np.array([1.0 if (bits >> i) & 1 else -1.0 for i in range(n)])
            exceed += abs(float((signs * deltas).mean())) >= observed - tol
        return exceed / total, f"exact_sign_flip_2^{n}"
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, MC_SAMPLES, 5000):
        count = min(5000, MC_SAMPLES - start)
        signs = rng.choice((-1.0, 1.0), size=(count, n))
        exceed += int((np.abs((signs * deltas).mean(axis=1)) >= observed - tol).sum())
    return (exceed + 1) / (MC_SAMPLES + 1), f"mc_sign_flip_{MC_SAMPLES}"


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(MC_SAMPLES, len(values)))
    means = values[idx].mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar on discordant pairs."""
    n = b + c
    if n == 0:
        return math.nan
    from scipy.stats import binomtest
    return float(binomtest(min(b, c), n, 0.5).pvalue)


def holm(pvalues: list[float]) -> list[float]:
    valid = sorted(((i, p) for i, p in enumerate(pvalues) if math.isfinite(p)),
                   key=lambda t: t[1])
    out = [math.nan] * len(pvalues)
    running = 0.0
    m = len(valid)
    for rank, (i, p) in enumerate(valid):
        running = max(running, p * (m - rank))
        out[i] = min(1.0, running)
    return out


def cluster_ols_interaction(units: list[dict[str, Any]]) -> dict[str, Any]:
    """rate ~ black + tuned + black:tuned with cluster-robust (image) SE."""
    import statsmodels.api as sm
    rows = []
    for u in units:
        for cond in ("base", "tuned"):
            if math.isfinite(u[cond]):
                rows.append((u["cluster"], u["black"], int(cond == "tuned"), u[cond]))
    if len(rows) < 8:
        return {"coef": math.nan, "se": math.nan, "p": math.nan}
    clusters = [r[0] for r in rows]
    X = np.array([[1.0, r[1], r[2], r[1] * r[2]] for r in rows])
    y = np.array([r[3] for r in rows])
    fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    return {"coef": float(fit.params[3]), "se": float(fit.bse[3]),
            "p": float(fit.pvalues[3]), "n_obs": int(fit.nobs)}


# ---------------------------------------------------------------------------
# IAT-12: paired image-cluster analysis
# ---------------------------------------------------------------------------

def per_image_rates(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for r in records:
        if not truthy(r.get("valid_three")):
            continue
        key = r["image_id"]
        slot = grouped.setdefault(key, {"image_id": key, "race": r["race"],
                                        "gender": r["gender"], "neg": 0, "tot": 0})
        slot["neg"] += int(float(r.get("negative_count") or 0))
        slot["tot"] += int(float(r.get("word_count") or 0))
    return {k: {**v, "rate": v["neg"] / v["tot"] if v["tot"] else math.nan}
            for k, v in grouped.items()}


def iat_gap(rates: dict[str, dict[str, Any]]) -> float:
    black = [v["rate"] for v in rates.values() if v["race"] == "Black"]
    white = [v["rate"] for v in rates.values() if v["race"] == "White"]
    if not black or not white:
        return math.nan
    return float(np.mean(black) - np.mean(white))


def iat_did_stat(deltas: list[dict[str, Any]]) -> float:
    b = [d["delta"] for d in deltas if d["race"] == "Black" and math.isfinite(d["delta"])]
    w = [d["delta"] for d in deltas if d["race"] == "White" and math.isfinite(d["delta"])]
    if not b or not w:
        return math.nan
    return float(np.mean(b) - np.mean(w))


def iat_did_bootstrap(deltas: list[dict[str, Any]], seed: int) -> tuple[float, float]:
    """Stratified image-cluster bootstrap on the DiD estimate."""
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for d in deltas:
        if math.isfinite(d["delta"]):
            cells[(d["race"], d["gender"])].append(d["delta"])
    if any(not v for v in cells.values()) or len(cells) < 4:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(MC_SAMPLES):
        per_race: dict[str, list[float]] = defaultdict(list)
        for (race, _gender), vals in cells.items():
            idx = rng.integers(0, len(vals), size=len(vals))
            per_race[race].extend(np.asarray(vals)[idx])
        if per_race.get("Black") and per_race.get("White"):
            stats.append(np.mean(per_race["Black"]) - np.mean(per_race["White"]))
    return tuple(float(x) for x in np.percentile(stats, [2.5, 97.5]))


def analyze_iat(conditions: dict[str, Path], out: Path, seed: int
                ) -> tuple[list[dict], list[dict], list[dict]]:
    """returns (test_rows, rate_rows, quality_rows)"""
    per_cond = {name: per_image_rates(read_records(p))
                for name, p in conditions.items()}
    raw = {name: read_records(p) for name, p in conditions.items()}
    test_rows, rate_rows, quality_rows = [], [], []

    for name, recs in raw.items():
        valid = [r for r in recs if truthy(r.get("valid_three"))]
        words = Counter(w for r in valid for w in str(r.get("words") or "").split("|") if w)
        tuples = Counter(str(r.get("words") or "") for r in valid)
        total_words = sum(words.values())
        probs = np.array(list(words.values())) / total_words if total_words else np.array([])
        quality_rows.append({
            "benchmark": "iat12", "condition": name,
            "responses": len(recs),
            "valid_responses": len(valid),
            "invalid_or_refusal": len(recs) - len(valid),
            "exact_format_rate": (sum(truthy(r.get("exact_format")) for r in recs) / len(recs))
            if recs else math.nan,
            "distinct_words": len(words),
            "unique_response_tuples": len(tuples),
            "normalized_word_entropy": float(
                -(probs * np.log2(probs)).sum() / math.log2(61)) if total_words else math.nan,
            "errors": sum(1 for r in recs if str(r.get("error_type") or "")),
        })

    for name, rates in per_cond.items():
        for race in ("Black", "White"):
            vals = [v["rate"] for v in rates.values() if v["race"] == race]
            lo, hi = bootstrap_mean_ci(np.array(vals), seed + 7)
            rate_rows.append({"benchmark": "iat12", "condition": name, "group": race,
                              "metric": "negative_term_rate", "rate": float(np.mean(vals)),
                              "ci_low": lo, "ci_high": hi, "n_units": len(vals)})
        rate_rows.append({"benchmark": "iat12", "condition": name, "group": "Black-White",
                          "metric": "negative_term_rate_gap", "rate": iat_gap(rates),
                          "ci_low": math.nan, "ci_high": math.nan, "n_units": len(rates)})

    tuned = [n for n in conditions if n != "base"]
    for name in tuned:
        base, other = per_cond["base"], per_cond[name]
        shared = sorted(set(base) & set(other))
        deltas = [{"image_id": i, "race": base[i]["race"], "gender": base[i]["gender"],
                   "cluster": i, "black": int(base[i]["race"] == "Black"),
                   "base": base[i]["rate"], "tuned": other[i]["rate"],
                   "delta": other[i]["rate"] - base[i]["rate"]}
                  for i in shared
                  if math.isfinite(base[i]["rate"]) and math.isfinite(other[i]["rate"])]

        # headline: DiD — did the Black-White gap change?
        did = iat_did_stat(deltas)
        obs, p, n_perm, method = stratified_binary_label_permutation(
            deltas, "delta", "race", "gender",
            group_a="Black", group_b="White",
            samples=MC_SAMPLES, seed=seed + 101)
        lo, hi = iat_did_bootstrap(deltas, seed + 102)
        test_rows.append({
            "benchmark": "iat12", "comparison": f"{name}_vs_base", "family": "headline",
            "metric": "black_minus_white_gap_change (DiD)", "group": "Black-White",
            "unit": "stimulus image (clustered repeats)",
            "n_units": len(deltas),
            "base_rate": iat_gap(per_cond["base"]),
            "tuned_rate": iat_gap(per_cond[name]),
            "delta": did, "ci_low": lo, "ci_high": hi,
            "raw_p": p, "test": f"stratified race-label permutation on deltas ({method}, {n_perm} permutations)",
        })
        # secondary: per-race change + OLS view + slot sensitivity
        ols = cluster_ols_interaction(deltas)
        test_rows.append({
            "benchmark": "iat12", "comparison": f"{name}_vs_base", "family": "secondary",
            "metric": "black_minus_white_gap_change (OLS cluster-robust)", "group": "Black-White",
            "unit": "stimulus image", "n_units": len(deltas) * 2,
            "base_rate": math.nan, "tuned_rate": math.nan,
            "delta": ols["coef"], "ci_low": ols["coef"] - 1.96 * ols["se"],
            "ci_high": ols["coef"] + 1.96 * ols["se"],
            "raw_p": ols["p"], "test": "OLS rate~black*tuned, cluster-robust SE by image",
        })
        for race in ("Black", "White"):
            d = np.array([x["delta"] for x in deltas if x["race"] == race])
            p2, m2 = sign_flip_pvalue(d, seed + 110)
            bvals = [x["base"] for x in deltas if x["race"] == race]
            tvals = [x["tuned"] for x in deltas if x["race"] == race]
            test_rows.append({
                "benchmark": "iat12", "comparison": f"{name}_vs_base", "family": "secondary",
                "metric": "negative_term_rate_change", "group": race,
                "unit": "stimulus image", "n_units": len(d),
                "base_rate": float(np.mean(bvals)), "tuned_rate": float(np.mean(tvals)),
                "delta": float(d.mean()) if len(d) else math.nan,
                "ci_low": math.nan, "ci_high": math.nan,
                "raw_p": p2, "test": m2,
            })
        # slot-level sensitivity (caveat: correlated within image)
        base_slots = {(r["image_id"], int(float(r["iteration"]))): r
                      for r in raw["base"]}
        tuned_slots = {(r["image_id"], int(float(r["iteration"]))): r
                       for r in raw[name]}
        b10 = c01 = 0
        for key, br in base_slots.items():
            tr = tuned_slots.get(key)
            if tr is None:
                continue
            bpos = int(float(br.get("negative_count") or 0)) > 0
            tpos = int(float(tr.get("negative_count") or 0)) > 0
            b10 += bpos and not tpos
            c01 += (not bpos) and tpos
        test_rows.append({
            "benchmark": "iat12", "comparison": f"{name}_vs_base", "family": "sensitivity",
            "metric": "slot_negative_rate_change (McNemar)", "group": "all",
            "unit": "image x iteration slot (correlated within image — anti-conservative)",
            "n_units": len(base_slots), "base_rate": math.nan, "tuned_rate": math.nan,
            "delta": math.nan, "ci_low": math.nan, "ci_high": math.nan,
            "raw_p": mcnemar_exact(b10, c01),
            "test": f"exact McNemar on discordant slots (b={b10}, c={c01})",
        })
    return test_rows, rate_rows, quality_rows


# ---------------------------------------------------------------------------
# DecodingTrust: paired agreement tests
# ---------------------------------------------------------------------------

def analyze_decodingtrust(conditions: dict[str, Path], out: Path, seed: int
                          ) -> tuple[list[dict], list[dict], list[dict]]:
    raw = {name: read_records(p) for name, p in conditions.items()}
    test_rows, rate_rows, quality_rows = [], [], []

    # Pair on (regime, group, prompt_sha256): stable across conditions even
    # when task_id hashes embed a model identity.
    keyed = {}
    for name, recs in raw.items():
        table = {}
        for r in recs:
            key = (r.get("regime"), r.get("demographic_group"), r.get("prompt_sha256"))
            table[key] = r
        keyed[name] = table

    for name, recs in raw.items():
        cls = Counter(r.get("classification") or "UNSCORED" for r in recs)
        quality_rows.append({
            "benchmark": "decodingtrust", "condition": name,
            "responses": len(recs),
            "agree": cls.get("AGREE", 0), "disagree": cls.get("DISAGREE", 0),
            "no_response_refusal": cls.get("NO_RESPONSE", 0),
            "api_error": cls.get("API_ERROR", 0),
            "unscored": cls.get("UNSCORED", 0),
            "distinct_responses": len({str(r.get("response")) for r in recs}),
        })

    # group x regime agreement rates per condition
    for name, recs in raw.items():
        for regime in DT_REGIMES:
            for group in DT_GROUPS:
                sel = [r for r in recs if r.get("regime") == regime
                       and r.get("demographic_group") == group
                       and r.get("classification") != "API_ERROR"]
                if not sel:
                    continue
                rate = np.mean([r["classification"] == "AGREE" for r in sel])
                lo, hi = wilson_interval(int(sum(r["classification"] == "AGREE" for r in sel)),
                                         len(sel))
                rate_rows.append({"benchmark": "decodingtrust", "condition": name,
                                  "regime": regime, "group": group,
                                  "metric": "agreement_rate", "rate": float(rate),
                                  "ci_low": lo, "ci_high": hi, "n_units": len(sel)})

    base = keyed.get("base")
    if base is None:
        return test_rows, rate_rows, quality_rows
    for name in (n for n in conditions if n != "base"):
        other = keyed[name]
        shared = sorted(set(base) & set(other))
        pairs = [(base[t], other[t]) for t in shared
                 if base[t].get("classification") != "API_ERROR"
                 and other[t].get("classification") != "API_ERROR"]

        # headline per regime: max-min group agreement gap change
        for r_i, regime in enumerate(DT_REGIMES):
            cells = [(b, t) for b, t in pairs if b.get("regime") == regime]
            if not cells:
                continue
            groups = sorted({b["demographic_group"] for b, _ in cells})

            base_gap, tuned_gap = None, None
            for label, pick in (("base", 0), ("tuned", 1)):
                rates = [np.mean([p[pick]["classification"] == "AGREE" for p in cells
                                  if p[0]["demographic_group"] == g]) for g in groups]
                if label == "base":
                    base_gap = float(max(rates) - min(rates))
                else:
                    tuned_gap = float(max(rates) - min(rates))
            observed = tuned_gap - base_gap
            rng = np.random.default_rng(seed + 200 + r_i)
            exceed = 0
            for _ in range(MC_SAMPLES):
                swap = rng.random(len(cells)) < 0.5
                rates_b, rates_t = [], []
                for g in groups:
                    sel = [i for i, (b, _) in enumerate(cells)
                           if b["demographic_group"] == g]
                    rb = np.mean([(cells[i][1] if swap[i] else cells[i][0])
                                  ["classification"] == "AGREE" for i in sel])
                    rt = np.mean([(cells[i][0] if swap[i] else cells[i][1])
                                  ["classification"] == "AGREE" for i in sel])
                    rates_b.append(rb)
                    rates_t.append(rt)
                delta = (max(rates_t) - min(rates_t)) - (max(rates_b) - min(rates_b))
                exceed += abs(delta) >= abs(observed) - 1e-12
            test_rows.append({
                "benchmark": "decodingtrust", "comparison": f"{name}_vs_base",
                "family": "headline", "metric": "max-min group agreement gap change",
                "group": f"4-group range, {regime}", "unit": "task (paired)",
                "n_units": len(cells), "base_rate": base_gap, "tuned_rate": tuned_gap,
                "delta": observed, "ci_low": math.nan, "ci_high": math.nan,
                "raw_p": (exceed + 1) / (MC_SAMPLES + 1),
                "test": f"MC paired task-label sign-flip ({MC_SAMPLES} draws)",
            })
            # secondary: McNemar per group within regime
            for g_i, g in enumerate(groups):
                sel = [(b, t) for b, t in cells if b["demographic_group"] == g]
                b10 = sum(b["classification"] == "AGREE" and t["classification"] != "AGREE"
                          for b, t in sel)
                c01 = sum(b["classification"] != "AGREE" and t["classification"] == "AGREE"
                          for b, t in sel)
                br = np.mean([b["classification"] == "AGREE" for b, _ in sel])
                tr = np.mean([t["classification"] == "AGREE" for _, t in sel])
                test_rows.append({
                    "benchmark": "decodingtrust", "comparison": f"{name}_vs_base",
                    "family": "secondary", "metric": "agreement_rate_change",
                    "group": f"{g} ({regime})", "unit": "task",
                    "n_units": len(sel), "base_rate": float(br), "tuned_rate": float(tr),
                    "delta": float(tr - br), "ci_low": math.nan, "ci_high": math.nan,
                    "raw_p": mcnemar_exact(b10, c01),
                    "test": f"exact McNemar (b={b10}, c={c01})",
                })
    return test_rows, rate_rows, quality_rows


# ---------------------------------------------------------------------------
# BOLD: paired continuation-score tests
# ---------------------------------------------------------------------------

def analyze_bold(conditions: dict[str, Path], out: Path, seed: int
                 ) -> tuple[list[dict], list[dict], list[dict]]:
    raw = {name: read_records(p) for name, p in conditions.items()}
    test_rows, rate_rows, quality_rows = [], [], []
    keyed = {name: {(r.get("category"), r.get("prompt_sha256")): r
                    for r in recs} for name, recs in raw.items()}

    for name, recs in raw.items():
        completed = [r for r in recs if r.get("status") == "completed"]
        quality_rows.append({
            "benchmark": "bold", "condition": name, "responses": len(recs),
            "completed": len(completed), "invalid_or_refusal": len(recs) - len(completed),
            "empty_response": sum(1 for r in recs if not str(r.get("response") or "").strip()),
            "distinct_responses": len({str(r.get("response") or "") for r in completed}),
        })

    for name, recs in raw.items():
        cats = sorted({r["category"] for r in recs if r.get("category")})
        for cat in cats:
            sel = [r for r in recs if r.get("category") == cat and "toxic_any" in r
                   and r.get("status") == "completed"]
            if not sel:
                continue
            for metric in BOLD_METRICS:
                vals = [float(r[metric]) for r in sel]
                lo, hi = bootstrap_mean_ci(np.array(vals), seed + 310)
                rate_rows.append({"benchmark": "bold", "condition": name, "group": cat,
                                  "metric": metric, "rate": float(np.mean(vals)),
                                  "ci_low": lo, "ci_high": hi, "n_units": len(sel)})

    base = keyed.get("base")
    if base is None:
        return test_rows, rate_rows, quality_rows
    for name in (n for n in conditions if n != "base"):
        other = keyed[name]
        shared = sorted(set(base) & set(other))
        for m_i, metric in enumerate(BOLD_METRICS):
            pairs = [(base[t], other[t]) for t in shared
                     if metric in base[t] and metric in other[t]
                     and base[t].get("status") == "completed"
                     and other[t].get("status") == "completed"]
            cats = sorted({b["category"] for b, _ in pairs})
            # headline: max-min category mean gap change
            def group_means(pick):
                return {c: float(np.mean([float(p[pick][metric]) for p in pairs
                                          if p[0]["category"] == c])) for c in cats}
            bm, tm = group_means(0), group_means(1)
            observed = (max(tm.values()) - min(tm.values())) - (max(bm.values()) - min(bm.values()))
            deltas = np.array([float(p[1][metric]) - float(p[0][metric]) for p in pairs])
            # joint sign-flip: recompute the range change under task label swaps
            cat_of = np.array([p[0]["category"] for p in pairs])
            bv = np.array([float(p[0][metric]) for p in pairs])
            tv = np.array([float(p[1][metric]) for p in pairs])
            rng = np.random.default_rng(seed + 400 + m_i)
            exceed = 0
            for _ in range(MC_SAMPLES):
                swap = rng.random(len(pairs)) < 0.5
                nb = np.where(swap, tv, bv)
                nt = np.where(swap, bv, tv)
                rb = max(nb[cat_of == c].mean() for c in cats) - min(
                    nb[cat_of == c].mean() for c in cats)
                rt = max(nt[cat_of == c].mean() for c in cats) - min(
                    nt[cat_of == c].mean() for c in cats)
                exceed += abs(rt - rb) >= abs(observed) - 1e-12
            test_rows.append({
                "benchmark": "bold", "comparison": f"{name}_vs_base", "family": "headline",
                "metric": f"max-min category gap change ({metric})",
                "group": "4-category range", "unit": "prompt (paired)",
                "n_units": len(pairs), "base_rate": max(bm.values()) - min(bm.values()),
                "tuned_rate": max(tm.values()) - min(tm.values()),
                "delta": float(observed), "ci_low": math.nan, "ci_high": math.nan,
                "raw_p": (exceed + 1) / (MC_SAMPLES + 1),
                "test": f"MC paired prompt-label sign-flip ({MC_SAMPLES} draws)",
            })
            # secondary: per-category paired change
            for c_i, cat in enumerate(cats):
                sel = np.array([p[0]["category"] == cat for p in pairs])
                d = deltas[sel]
                p2, m2 = sign_flip_pvalue(d, seed + 500 + m_i * 10 + c_i)
                br = float(np.mean([float(p[0][metric]) for p, s in zip(pairs, sel) if s]))
                tr = float(np.mean([float(p[1][metric]) for p, s in zip(pairs, sel) if s]))
                test_rows.append({
                    "benchmark": "bold", "comparison": f"{name}_vs_base", "family": "secondary",
                    "metric": f"mean change ({metric})", "group": cat,
                    "unit": "prompt", "n_units": int(sel.sum()),
                    "base_rate": br, "tuned_rate": tr, "delta": tr - br,
                    "ci_low": math.nan, "ci_high": math.nan,
                    "raw_p": p2, "test": m2,
                })
    return test_rows, rate_rows, quality_rows


# ---------------------------------------------------------------------------
# VisBias: matched-pair description diffs + comparison-unit preference
# ---------------------------------------------------------------------------

def score_visbias_descriptions(records: list[dict[str, Any]],
                               lexicons: dict[str, set[str]]) -> list[dict[str, Any]]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    negative = {w.lower() for w in NEGATIVE_IAT_SET}
    scored = []
    for r in records:
        if r.get("task_kind") != "description":
            continue
        text = str(r.get("response") or "")
        words = [m.group(0).lower() for m in WORD_RE.finditer(text)]
        race = str(r.get("race") or "")
        hits = sum(w in lexicons.get(race, set()) for w in words)
        scored.append({**r, "word_count": len(words),
                       "vader_compound": analyzer.polarity_scores(text)["compound"],
                       "iat_negative_rate": (sum(w in negative for w in words) / len(words))
                       if words else math.nan,
                       "stereotype_hits_per_100_words": (100 * hits / len(words))
                       if words else math.nan,
                       "identity_mention": float(bool(IDENTITY_PATTERN.findall(text)))})
    return scored


def visbias_pair_diffs(scored: list[dict[str, Any]], metric: str
                       ) -> dict[str, float]:
    """matched_pair_id -> black_minus_white on the metric."""
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for r in scored:
        if r.get("status") != "completed" or not math.isfinite(float(r.get(metric) or "nan")):
            continue
        grouped[r["matched_pair_id"]][r["race"]] = float(r[metric])
    return {p: v["black"] - v["white"] for p, v in grouped.items()
            if set(v) == {"black", "white"}}


def visbias_unit_scores(records: list[dict[str, Any]]) -> dict[str, float]:
    """comparison_unit_id -> mean directional score (+1 black, -1 white, 0 other)."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r.get("task_kind") != "comparison" or r.get("status") != "completed":
            continue
        sel = r.get("selected_race")
        grouped[r["comparison_unit_id"]].append(
            1.0 if sel == "black" else -1.0 if sel == "white" else 0.0)
    return {u: float(np.mean(v)) for u, v in grouped.items()}


def analyze_visbias(conditions: dict[str, Path], lexicons_path: Path, out: Path,
                    seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    lexicons = {k: {w.lower() for w in v}
                for k, v in json.loads(lexicons_path.read_text()).items()}
    raw = {name: read_records(p) for name, p in conditions.items()}
    test_rows, rate_rows, quality_rows = [], [], []

    for name, recs in raw.items():
        comps = [r for r in recs if r.get("task_kind") == "comparison"]
        descs = [r for r in recs if r.get("task_kind") == "description"]
        answers = Counter(r.get("parsed_answer") or "" for r in comps)
        quality_rows.append({
            "benchmark": "visbias", "condition": name,
            "responses": len(recs),
            "description_responses": len(descs),
            "description_completed": sum(r.get("status") == "completed" for r in descs),
            "comparison_responses": len(comps),
            "comparison_completed": sum(r.get("status") == "completed" for r in comps),
            "cannot_determine_rate": (answers["CANNOT_DETERMINE"] / len(comps)) if comps else math.nan,
            "unclassified_rate": (answers["UNCLASSIFIED"] / len(comps)) if comps else math.nan,
            "black_selected_rate": (sum(r.get("selected_race") == "black" for r in comps)
                                    / len(comps)) if comps else math.nan,
            "white_selected_rate": (sum(r.get("selected_race") == "white" for r in comps)
                                    / len(comps)) if comps else math.nan,
            "errors": sum(1 for r in recs if str(r.get("error_type") or "")),
        })

    scored = {name: score_visbias_descriptions(recs, lexicons)
              for name, recs in raw.items()}
    units = {name: visbias_unit_scores(recs) for name, recs in raw.items()}

    for name in conditions:
        for m_i, metric in enumerate(VISBIAS_DESCRIPTION_METRICS):
            diffs = visbias_pair_diffs(scored[name], metric)
            if not diffs:
                continue
            vals = np.array(list(diffs.values()))
            lo, hi = bootstrap_mean_ci(vals, seed + 600 + m_i)
            rate_rows.append({"benchmark": "visbias_descriptions", "condition": name,
                              "group": "Black-White", "metric": metric,
                              "rate": float(vals.mean()), "ci_low": lo, "ci_high": hi,
                              "n_units": len(vals)})

    for name in (n for n in conditions if n != "base"):
        for m_i, metric in enumerate(VISBIAS_DESCRIPTION_METRICS):
            db = visbias_pair_diffs(scored["base"], metric)
            dt_ = visbias_pair_diffs(scored[name], metric)
            shared = sorted(set(db) & set(dt_))
            deltas = np.array([dt_[p] - db[p] for p in shared])
            p2, m2 = sign_flip_pvalue(deltas, seed + 700 + m_i)
            test_rows.append({
                "benchmark": "visbias_descriptions", "comparison": f"{name}_vs_base",
                "family": "headline", "metric": f"black_minus_white gap change ({metric})",
                "group": "matched image pairs", "unit": "matched_pair_id",
                "n_units": len(deltas),
                "base_rate": float(np.mean([db[p] for p in shared])) if shared else math.nan,
                "tuned_rate": float(np.mean([dt_[p] for p in shared])) if shared else math.nan,
                "delta": float(deltas.mean()) if len(deltas) else math.nan,
                "ci_low": math.nan, "ci_high": math.nan,
                "raw_p": p2, "test": m2,
            })
        ub, ut = units["base"], units[name]
        shared_u = sorted(set(ub) & set(ut))
        du = np.array([ut[u] - ub[u] for u in shared_u])
        p3, m3 = sign_flip_pvalue(du, seed + 710)
        test_rows.append({
            "benchmark": "visbias_comparisons", "comparison": f"{name}_vs_base",
            "family": "headline",
            "metric": "black-vs-white unsupported selection change",
            "group": "comparison units", "unit": "comparison_unit_id",
            "n_units": len(du),
            "base_rate": float(np.mean([ub[u] for u in shared_u])) if shared_u else math.nan,
            "tuned_rate": float(np.mean([ut[u] for u in shared_u])) if shared_u else math.nan,
            "delta": float(du.mean()) if len(du) else math.nan,
            "ci_low": math.nan, "ci_high": math.nan,
            "raw_p": p3, "test": m3,
        })
    return test_rows, rate_rows, quality_rows


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def apply_multiplicity(test_rows: list[dict[str, Any]]) -> None:
    """BH across headline tests; Holm within each benchmark's secondary family."""
    head = [r for r in test_rows if r["family"] == "headline"]
    qs = benjamini_hochberg([r["raw_p"] for r in head])
    for r, q in zip(head, qs, strict=True):
        r["bh_adjusted_q_across_headline"] = q
    by_bench: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in test_rows:
        if r["family"] == "secondary":
            by_bench[(r["benchmark"], r["comparison"])].append(r)
    for rows in by_bench.values():
        adj = holm([r["raw_p"] for r in rows])
        for r, a in zip(rows, adj, strict=True):
            r["holm_adjusted_p_within_family"] = a


def write_summary(path: Path, test_rows, rate_rows, quality_rows, meta) -> None:
    lines = ["# Base vs fine-tuned significance report", ""]
    lines.append(f"Generated: {meta['generated']}  |  seed: {BASE_SEED}  |  "
                 f"MC draws: {MC_SAMPLES}")
    lines.append("")
    lines.append("## Headline: did tuning change the disparity?")
    lines.append("")
    lines.append("| benchmark | comparison | metric | base | tuned | delta | 95% CI | raw p | BH q |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in test_rows:
        if r["family"] != "headline":
            continue
        ci = ""
        if math.isfinite(r.get("ci_low") or math.nan):
            ci = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        lines.append(
            f"| {r['benchmark']} | {r['comparison']} | {r['metric']} "
            f"| {r['base_rate']:.4f} | {r['tuned_rate']:.4f} | {r['delta']:+.4f} "
            f"| {ci} | {r['raw_p']:.4f} | {r.get('bh_adjusted_q_across_headline', math.nan):.4f} |")
    lines.append("")
    lines.append("## Secondary tests (per-group changes)")
    lines.append("")
    lines.append("| benchmark | comparison | group | metric | delta | raw p | Holm p | test |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in test_rows:
        if r["family"] != "secondary":
            continue
        lines.append(
            f"| {r['benchmark']} | {r['comparison']} | {r['group']} | {r['metric']} "
            f"| {r['delta']:+.4f} | {r['raw_p']:.4f} "
            f"| {r.get('holm_adjusted_p_within_family', math.nan):.4f} | {r['test']} |")
    lines.append("")
    lines.append("## Response quality / refusal / diversity ledger")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(quality_rows, indent=2, default=str))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_condition_args(values: list[str] | None) -> dict[str, Path]:
    out = {}
    for item in values or []:
        name, _, p = item.partition("=")
        if not p:
            raise ValueError(f"condition must be name=path, got {item}")
        out[name] = Path(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iat", nargs="*", default=None,
                    help="name=responses.csv per condition (base/sft/orpo)")
    ap.add_argument("--decodingtrust", nargs="*", default=None,
                    help="name=responses.csv per condition")
    ap.add_argument("--bold", nargs="*", default=None,
                    help="name=scored_responses.csv per condition")
    ap.add_argument("--visbias", nargs="*", default=None,
                    help="name=responses.jsonl/csv per condition")
    ap.add_argument("--visbias-lexicons", type=Path,
                    default=REPO / "2026-09-05/data/frozen/visbias/race_stereotype_lexicons.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    all_tests: list[dict] = []
    all_rates: list[dict] = []
    all_quality: list[dict] = []

    jobs = [
        ("iat", args.iat, lambda c: analyze_iat(c, out, BASE_SEED)),
        ("decodingtrust", args.decodingtrust,
         lambda c: analyze_decodingtrust(c, out, BASE_SEED + 1)),
        ("bold", args.bold, lambda c: analyze_bold(c, out, BASE_SEED + 2)),
        ("visbias", args.visbias,
         lambda c: analyze_visbias(c, args.visbias_lexicons, out, BASE_SEED + 3)),
    ]
    for name, cond_args, fn in jobs:
        conds = parse_condition_args(cond_args)
        if not conds:
            continue
        if "base" not in conds:
            raise ValueError(f"{name}: a 'base=' condition is required")
        t, r, q = fn(conds)
        all_tests += t
        all_rates += r
        all_quality += q
        print(f"[{name}] {len(t)} tests, {len(r)} rate rows, {len(q)} quality rows")

    apply_multiplicity(all_tests)
    write_csv(out / "paired_tests.csv", all_tests)
    write_csv(out / "group_rates.csv", all_rates)
    write_csv(out / "response_quality.csv", all_quality)
    meta = {"generated": __import__("datetime").datetime.now().isoformat(timespec="seconds")}
    write_summary(out / "summary.md", all_tests, all_rates, all_quality, meta)
    print(json.dumps({"out": str(out), "tests": len(all_tests),
                      "rates": len(all_rates), "quality": len(all_quality)}))


if __name__ == "__main__":
    main()
