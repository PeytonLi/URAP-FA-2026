import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_evidence as analysis


def test_exact_sign_test_small_pool():
    assert analysis.sign_test([1] * 6) == 2 / 64
    assert analysis.sign_test([0] * 6) == 1
    assert analysis.sign_test([1, -1]) == 1


def test_topic_block_null():
    rows = [
        dict(topic=str(topic), race=race, agreement=0)
        for topic in range(4)
        for race in ("A", "B", "C", "D")
    ]
    assert analysis.group_range(rows, "agreement", "race", block="topic") == (0, 0, 0, 1)


def test_bh_accounts_for_all_metrics():
    rows = [dict(raw_p_value=p) for p in [0.01, 0.03, 0.9]]
    analysis.adjust(rows, "three tests")
    np.testing.assert_allclose([r["bh_adjusted_q_value"] for r in rows], [0.03, 0.045, 0.9])


def test_legacy_parser_not_exact_format_requirement():
    row = dict(
        model_key="gemma_local",
        image="bf14_nc.jpg",
        fairface_race="Black",
        fairface_gender="Female",
        response="1. Happy, 2. Joyful, 3. Pain",
    )
    parsed = analysis.normalize_iat([row])[0]
    assert parsed["valid_three"]
    assert parsed["negative_count"] == 1
    assert parsed["words"] == "Happy|Joyful|Pain"


def test_bold_subject_clusters_preserve_response_weighting(monkeypatch):
    monkeypatch.setattr(analysis, "N", 2000)
    rows = [
        dict(category="A", subject_name="a1", score=1),
        dict(category="A", subject_name="a1", score=1),
        dict(category="A", subject_name="a2", score=0),
        dict(category="B", subject_name="b1", score=0),
        dict(category="B", subject_name="b2", score=0),
    ]
    effect, low, high, p = analysis.subject_cluster_range(rows, "score")
    assert effect == 2 / 3
    assert 0 <= low <= high <= 1
    assert 0 < p <= 1


def test_bold_constant_cluster_null(monkeypatch):
    monkeypatch.setattr(analysis, "N", 2000)
    rows = [dict(category=c, subject_name=c + str(i), score=0) for c in "ABCD" for i in range(2)]
    assert analysis.subject_cluster_range(rows, "score") == (0, 0, 0, 1)
