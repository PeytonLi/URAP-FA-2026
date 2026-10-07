import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from curation import HashIndex, capped_allocation, expression_record, image_metrics, pixel_hash, stable


def test_capped_allocation_preserves_total_and_limits():
    caps = {"20": 100, "30": 100, "40": 30, "50": 10, "60": 6}
    got = capped_allocation(100, caps)
    assert sum(got.values()) == 100
    assert all(0 <= got[k] <= caps[k] for k in caps)
    assert got["60"] == 6
    assert got["50"] == 10
    assert got == capped_allocation(100, caps)


def test_insufficient_support_raises():
    with pytest.raises(ValueError, match="Insufficient"):
        capped_allocation(101, {"a": 100})


def test_empty_allocation():
    assert capped_allocation(0, {"a": 0}) == {"a": 0}


def test_weighted_allocation_respects_weights_and_caps():
    got = capped_allocation(12, {"a": 10, "b": 10}, {"a": 2, "b": 1})
    assert got == {"a": 8, "b": 4}


def test_low_confidence_not_forced_to_neutral():
    config = {"expression_confidence_for_class": .6, "expression_model_id": "model",
              "expression_model_revision": "rev"}
    row = {"image_id": "id", "image_sha256": "sha"}
    got = expression_record(row, "sad", .4, config, "test")
    assert got["expression_class"] == "uncertain"
    assert got["expression_label"] == "sad"
    assert expression_record(row, "sad", .9, config, "test")["expression_class"] == "negative"
    assert expression_record(row, "surprise", .9, config, "test")["expression_class"] == "uncertain"


def test_hash_index_all_radius_four_candidates_retrieved():
    base = {"perceptual_hash": "0000000000000000", "difference_hash": "0000000000000000"}
    ix = HashIndex()
    ix.add(base)
    for bits in [(0, 13, 26, 39), (12, 25, 38, 63), (1, 2, 3, 4)]:
        query = {**base, "perceptual_hash": f"{sum(1 << b for b in bits):016x}"}
        assert ix.matches(query) == [base]
    far = {**base, "perceptual_hash": "ffffffffffffffff"}
    assert ix.matches(far) == []


def test_difference_hash_also_required():
    ix = HashIndex()
    ix.add({"perceptual_hash": "0", "difference_hash": "0"})
    assert not ix.matches({"perceptual_hash": "0", "difference_hash": "ffffffffffffffff"})


def test_hash_radius_does_not_silently_exceed_index_guarantee():
    with pytest.raises(ValueError, match="supports perceptual distance"):
        HashIndex().matches({"perceptual_hash": "0", "difference_hash": "0"}, max_phash=5)


def test_metrics_do_not_reject_dark_image():
    im = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    metrics = image_metrics(im)
    assert metrics["brightness"] == 0
    assert "quality_score" not in metrics
    assert "screening_status" not in metrics


def test_pixel_hash_includes_dimensions_and_pixels():
    a = Image.new("RGB", (100, 100), "red")
    assert pixel_hash(a) == pixel_hash(a.copy())
    assert pixel_hash(a) != pixel_hash(Image.new("RGB", (50, 200), "red"))


def test_seed_changes_rank():
    assert stable("one", "image") != stable("two", "image")
