"""The screenshot anchor must never silently become a different baseline run."""
import copy
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[0]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(REPO / "2026-10-20/analysis"))
sys.path.insert(0, str(REPO / "2026-09-11/src"))

import historical_baseline as baseline
import variety
from multimodel_bias.constants import NEGATIVE_IAT_SET
from multimodel_bias.plots import REFERENCE_WORD_ORDER

FROZEN = ROOT / "analysis/historical_baseline_v1"


def frozen_rows():
    with (FROZEN / "responses.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_reference_order_and_valence_match_all_future_charts():
    assert baseline.WORD_ORDER == REFERENCE_WORD_ORDER
    assert baseline.NEGATIVE == NEGATIVE_IAT_SET
    assert len(baseline.WORD_ORDER) == len(set(baseline.WORD_ORDER)) == 61


def test_frozen_files_match_their_checksums():
    provenance = json.loads((FROZEN / "provenance.json").read_text())
    assert provenance["source_notebook_sha256"] == baseline.SOURCE_SHA256
    assert provenance["plot_style"] == baseline.PLOT_STYLE
    assert provenance["matching_repository_plot_sha256"] == baseline.REFERENCE_PLOT_SHA256
    for name, expected in provenance["files_sha256"].items():
        assert baseline.sha256(FROZEN / name) == expected
    plot = ROOT.parents[0] / provenance["matching_repository_plot"]
    assert baseline.sha256(plot) == baseline.REFERENCE_PLOT_SHA256


def test_original_bar_counts_are_recovered_from_responses_not_an_image():
    rows = frozen_rows()
    baseline.validate_rows(rows)
    with (FROZEN / "word_counts_reference_order.csv").open(newline="") as stream:
        counts = list(csv.DictReader(stream))
    assert [r["word"] for r in counts] == list(baseline.WORD_ORDER)
    for race, field in (("Black", "black_count"), ("White", "white_count")):
        observed = baseline.counts_for([r for r in rows if r["race"] == race])
        assert sum(observed.values()) == 900
        assert all(int(r[field]) == observed[r["word"]] for r in counts)
    summary, _ = baseline.summarize(rows)
    assert summary["groups"]["Black"]["negative_count"] == 341
    assert summary["groups"]["White"]["negative_count"] == 60
    assert summary["distinct_words_union"] == 38
    assert summary["median_distinct_unordered_triples_per_image"] == 24.5
    assert summary["gap_exact_permutation_p_two_sided"] == pytest.approx(62 / 924)
    stored = json.loads((FROZEN / "metrics.json").read_text())
    assert summary == stored


def test_stats_agree_with_the_existing_image_level_scorer():
    converted = []
    for row in frozen_rows():
        words = row["response"].split(", ")
        converted.append(dict(row, valid_three="True", words="|".join(words),
                              negative_count=sum(w in baseline.NEGATIVE for w in words)))
    new, _ = baseline.summarize(frozen_rows())
    old, _, _ = variety.summarize(converted, "final_scored")
    assert new["negative_rate_gap_black_minus_white"] == pytest.approx(
        old["black_minus_white_negative_rate"])
    assert new["gap_exact_permutation_p_two_sided"] == old["negative_gap_exact_p"]
    assert new["effective_vocabulary"] == pytest.approx(old["effective_vocabulary_size"])
    assert new["total_variation"] == pytest.approx(old["word_tv_image_weighted"])
    assert new["dominant_unordered_triple_share"] == old["dominant_unordered_triple_share"]


@pytest.mark.parametrize("mutation", [
    lambda rows: rows.pop(),
    lambda rows: rows.__setitem__(0, copy.deepcopy(rows[1])),
    lambda rows: rows[0].update(race="White"),
    lambda rows: rows[0].update(image_id="another_image"),
    lambda rows: rows[0].update(response="Happy, Calm, Evil"),
    lambda rows: rows[0].update(fallback_status="sampled"),
])
def test_missing_duplicate_or_changed_data_fails_closed(mutation):
    rows = frozen_rows()
    mutation(rows)
    with pytest.raises(ValueError):
        baseline.validate_rows(rows)


def test_parser_ignores_warnings_without_manufacturing_answers():
    text = "\n".join(
        f"{r['image_filename']} [iter {r['iteration']}] -> {r['response']}"
        for r in frozen_rows()
    )
    rows = baseline.parse_stream("warning: generation flags\n" + text)
    assert len(rows) == 600
    assert {r["fallback_status"] for r in rows} == {"not_recorded"}
    with pytest.raises(ValueError, match="Invalid saved response"):
        baseline.parse_stream(text.replace("Adore", "Unknown", 1))


def test_other_notebooks_cannot_silently_replace_this_baseline(tmp_path):
    path = tmp_path / "another.ipynb"
    path.write_text('{"cells": []}')
    with pytest.raises(ValueError, match="checksum"):
        baseline.recover_notebook(path)


def test_export_never_overwrites_frozen_outputs(tmp_path):
    (tmp_path / "keep.txt").write_text("existing")
    with pytest.raises(ValueError, match="never overwrite"):
        baseline.export(frozen_rows(), {}, {}, tmp_path)
    assert (tmp_path / "keep.txt").read_text() == "existing"
