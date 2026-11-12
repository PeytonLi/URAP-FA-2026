"""Fixed-unit tests: never promote 600 correlated attempts to 600 images."""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[0]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(REPO / "2026-10-20/analysis"))
import iat_significance_v4 as stats
import variety

FROZEN = ROOT / "analysis/iat_stats_v4"


def frozen_input():
    rows = stats.load_csv(FROZEN / "paired_responses.csv")
    return ([r for r in rows if r["condition"] == "base"],
            [r for r in rows if r["condition"] == "tuned"])


def test_source_files_audited_and_image_identity_paired():
    data, audit = stats.audit_sources(ROOT / "analysis/variety_v4/provenance.json")
    stats.validate_pair(data["base"], data["tuned"])
    assert audit["paired_images"] == 12
    assert audit["conditions"]["base"]["sampled_calls"] == 600
    assert audit["conditions"]["tuned"]["sampled_calls"] == 600
    assert audit["conditions"]["base"]["fallback_calls"] == 0
    assert audit["conditions"]["tuned"]["fallback_calls"] == 28
    assert audit["historical_screenshot_baseline_in_tuning_tests"] is False
    assert stats.digest(FROZEN / "paired_responses.csv") == json.loads(
        (FROZEN / "protocol_audit.json").read_text())["paired_inputs_sha256"]


def test_invalid_views_and_published_diversity_measures():
    base, tuned = frozen_input()
    stats.validate_pair(base, tuned)
    b = stats.per_image(base, "sampled_exact")
    t = stats.per_image(tuned, "sampled_exact")
    assert sum(r["valid"] for r in b) == 600
    assert sum(r["valid"] for r in t) == 563
    assert sum(r["valid"] for r in stats.per_image(tuned, "sampled_distinct")) == 555
    assert sum(r["valid"] for r in stats.per_image(tuned, "final_scored")) == 600
    assert np.median([r["triples"] for r in b]) == 3.5
    assert np.median([r["triples"] for r in t]) == 46.5
    assert np.mean([r["entropy"] for r in t]) == pytest.approx(0.8960344054931921)
    assert np.mean([r["negative_rate"] for r in t[:6]]) - np.mean(
        [r["negative_rate"] for r in t[6:]]) == pytest.approx(-0.008915595396532357)


def test_assumed_permutation_spaces_and_extreme_contrast():
    races = ["Black"] * 6 + ["White"] * 6
    genders = ["Female"] * 3 + ["Male"] * 3 + ["Female"] * 3 + ["Male"] * 3
    assert stats.permutation_masks(races).shape == (924, 12)
    assert stats.permutation_masks(races, genders).shape == (400, 12)
    values = np.array([1.] * 6 + [0.] * 6)
    effect, p, count = stats.race_test(values, races)
    assert (effect, p, count) == (1., 2 / 924, 924)
    _, blocked, count = stats.race_test(values, races, genders)
    assert count == 400 and blocked == pytest.approx(2 / 400)


def test_entropy_and_tv_pairwise_regression_against_existing_scorer():
    base, tuned = frozen_input()
    b = stats.per_image(base, "sampled_exact")
    t = stats.per_image(tuned, "sampled_exact")
    for label, rows in (("base", b), ("tuned", t)):
        source = base if label == "base" else tuned
        summary, images, _ = variety.summarize(source, "sampled_exact")
        assert np.mean([r["entropy"] for r in rows]) == pytest.approx(
            summary["per_image_word_entropy_mean"])
        assert [r["triples"] for r in rows] == [r["distinct_unordered_triples"] for r in images]
        gap, p, count = stats.race_test(
            np.array([r["negative_rate"] for r in rows]),
            [r["race"] for r in rows])
        assert count == 924
        assert (gap, p) == pytest.approx(
            (summary["black_minus_white_negative_rate"], summary["negative_gap_exact_p"]))
        tv, p, count = stats.race_test(np.array([r["vector"] for r in rows]),
                                       [r["race"] for r in rows])
        assert (tv, p, count) == pytest.approx(
            (summary["word_tv_image_weighted"], summary["word_tv_exact_p"], 924))


def test_rarefaction_exact_even_with_nonadjacent_repeated_words():
    answers = [
        ("Happy", "Joyful", "Happy"), ("Pain", "Grief", "Sadness"),
        ("Happy", "Joyful", "Pain"), ("Ugly", "Gross", "Pain"),
    ]
    for n in (1, 2, 3, 4):
        exact = stats.rarefied_metrics(answers, n)
        brute = []
        for subset in itertools.combinations(answers, n):
            selected = [w for triple in subset for w in triple]
            entropy = variety.entropy({w: selected.count(w) for w in set(selected)},
                                      len(stats.WORDS))
            rich = len({tuple(sorted(triple)) for triple in subset})
            brute.append((entropy, rich))
        assert exact == pytest.approx(np.mean(brute, axis=0))
    with pytest.raises(ValueError):
        stats.rarefied_metrics(answers, 0)


def test_all_primary_results_match_frozen_and_have_adjusted_p():
    b, t = frozen_input()
    rows, images, _ = stats.analyze(b, t)
    published = stats.load_csv(FROZEN / "tests.csv")
    assert len(rows) == len(published)
    assert len(images) == 12
    primary = [r for r in rows if r["family"] == "primary"]
    assert [r["test"] for r in primary] == json.loads(
        (FROZEN / "analysis_plan.json").read_text())["primary_tests"]
    assert primary[0]["change"] == pytest.approx(-0.845582262063199)
    assert primary[0]["raw_p"] == pytest.approx(4 / 924)
    assert primary[0]["holm_p"] == pytest.approx(4 / 924)
    assert primary[1]["raw_p"] == primary[2]["raw_p"] == pytest.approx(2 / 4096)
    assert primary[1]["holm_p"] == primary[2]["holm_p"] == pytest.approx(6 / 4096)
    for actual, stored in zip(rows, published, strict=True):
        assert actual["test"] == stored["test"]
        assert actual["family"] == stored["family"]
        for key in ("raw_p", "holm_p", "effect_tested", "ci_low", "ci_high"):
            assert actual[key] == pytest.approx(float(stored[key]))
    assert all(math.isfinite(r["holm_p"]) for r in rows)
    assert all(r["holm_p"] + 1e-12 >= r["raw_p"] for r in rows)


@pytest.mark.parametrize("field,value", [
    ("race", "White"),
    ("image_sha256", "different"),
    ("seed", "42"),
    ("gender", "Male"),
    ("iteration", "2"),
    ("response_source", "fallback_invalid"),
])
def test_metadata_mismatch_or_missing_slots_fails_closed(field, value):
    base, tuned = frozen_input()
    tuned[0][field] = value
    with pytest.raises(ValueError):
        stats.analyze(base, tuned)


def test_checksums_detect_tampering(tmp_path):
    audit = json.loads((FROZEN / "protocol_audit.json").read_text())
    frozen = tmp_path / "responses.csv"
    frozen.write_bytes((FROZEN / "paired_responses.csv").read_bytes() + b"\n")
    assert stats.digest(frozen) != audit["paired_inputs_sha256"]
    provenance = json.loads((FROZEN / "provenance.json").read_text())
    assert provenance["analysis_script_sha256"] == stats.digest(
        ROOT / "analysis/iat_significance_v4.py")
    for name, sha in provenance["files_sha256"].items():
        assert stats.digest(FROZEN / name) == sha
