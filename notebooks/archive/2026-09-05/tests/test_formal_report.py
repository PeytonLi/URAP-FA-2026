import json

from common import ROOT, read_csv, sha
from audit_sep5files import difference_kind
from build_formal_report import SPECIAL, endpoint_label


def test_significance_labels_distinguish_metrics_in_same_regime():
    first = endpoint_label({"metric": "toxicity_flag_rate", "regime": "continuation"})
    second = endpoint_label({"metric": "vader_class_mean", "regime": "continuation"})
    assert first != second
    assert "toxicity_flag_rate" in first and "continuation" in first
    assert endpoint_label({"metric": "negative_term_race_gap"}) == "negative_term_race_gap"


def test_all_attached_columns_defined():
    fields = [c for p in (ROOT / "data/sep5files").glob("*.csv") for c in read_csv(p)[0]]
    assert len(fields) == 122
    assert set(fields) <= set(SPECIAL)


def test_export_changes_are_distinguished():
    assert difference_kind("prompt", "Example", "Example ") == "whitespace_changed"
    assert difference_kind("response", "s work", "'s work") == "leading_apostrophe_removed"
    assert difference_kind("valid_three", "TRUE", "True") == "boolean_case"
    assert difference_kind("cost_usd", "1E-3", "0.001") == "numeric_format"
    assert difference_kind("response", "changed", "original") == "unexplained_text_difference"


def test_attachments_preserved_by_hash():
    manifest = json.loads((ROOT / "data/sep5files/source_manifest.json").read_text())
    for filename, record in manifest.items():
        assert sha(ROOT / "data/sep5files" / filename) == record["sha256"]


def test_guideline_and_test_counts_not_conflated_with_responses():
    assert len(read_csv(ROOT / "report/requested_variable_definitions.csv")) == 231
    assert len(read_csv(ROOT / "results/analysis/all_significance_tests.csv")) == 157
    for path in (ROOT / "data/sep5files").glob("*.csv"):
        assert "raw_p_value" not in read_csv(path)[0]
