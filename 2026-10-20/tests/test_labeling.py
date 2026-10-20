"""Unit tests for the gated labeling pipeline (no teacher calls)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labeling"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                     / "2026-09-11" / "src"))

import common  # noqa: E402
import gate  # noqa: E402
import policy  # noqa: E402


def draft(chosen, rejected, expression="negative", conf=0.9, flags=()):
    return {
        "chosen": chosen, "rejected": rejected, "expression": expression,
        "visible_evidence": "furrowed brow, tight lips",
        "rejected_reason": "positive words contradict visible tension",
        "confidence": conf, "review_flags": list(flags),
    }


ROW = {"image_id": "img1", "image_sha256": "a" * 64, "pixel_sha256": "b" * 64,
       "split": "train", "image_path": "images/fairface/img1.png",
       "dataset": "fairface", "match_set_id": "m1",
       "expression_class": "negative", "source_race": "Black",
       "source_gender": "Male", "age_band": "30-39"}


def rec(d, tag="A"):
    return policy.draft_record(ROW, d, tag, "test-teacher")


class ValidationTests(unittest.TestCase):
    def test_valid_draft(self):
        r = rec(draft(["Angry", "Annoy", "Awful"], ["Happy", "Joyful", "Glad"]))
        self.assertEqual(r["chosen"], "Angry, Annoy, Awful")
        self.assertEqual(r["prompt"], policy.IAT_PROMPT)

    def test_rejects_noncanonical(self):
        with self.assertRaises(ValueError):
            rec(draft(["Angry", "Annoy", "Neutral"], ["Happy", "Joyful", "Glad"]))

    def test_rejects_duplicate(self):
        with self.assertRaises(ValueError):
            rec(draft(["Angry", "Angry", "Awful"], ["Happy", "Joyful", "Glad"]))

    def test_rejects_forbidden_chosen(self):
        with self.assertRaises(ValueError):
            rec(draft(["Sick", "Awful", "Nasty"], ["Happy", "Joyful", "Glad"]))

    def test_rejects_same_response(self):
        with self.assertRaises(ValueError):
            rec(draft(["Angry", "Annoy", "Awful"], ["Angry", "Annoy", "Awful"]))

    def test_neutral_flag_added(self):
        r = rec(draft(["Glad", "Pleasing", "Lovely"], ["Angry", "Nasty", "Awful"],
                      expression="neutral"))
        self.assertIn("neutral_vocabulary_gap", r["review_flags"])


class GateTests(unittest.TestCase):
    def test_resolve_on_identical_confident(self):
        a = rec(draft(["Angry", "Annoy", "Awful"], ["Happy", "Joyful", "Glad"]))
        b = rec(draft(["Awful", "Angry", "Annoy"], ["Glad", "Happy", "Joyful"]),
                tag="B")
        status, info = gate.gate_row(ROW, a, b)
        self.assertEqual(status, "resolved")
        self.assertEqual(gate.word_set(info["chosen"]),
                         {"Angry", "Annoy", "Awful"})

    def test_adjudicate_on_2of3(self):
        a = rec(draft(["Angry", "Annoy", "Awful"], ["Happy", "Joyful", "Glad"]))
        b = rec(draft(["Angry", "Annoy", "Nasty"], ["Glad", "Happy", "Joyful"]),
                tag="B")
        status, info = gate.gate_row(ROW, a, b)
        self.assertEqual(status, "adjudication")

    def test_unresolved_low_confidence(self):
        a = rec(draft(["Angry", "Annoy", "Awful"], ["Happy", "Joyful", "Glad"],
                      conf=0.7))
        b = rec(draft(["Angry", "Annoy", "Awful"], ["Happy", "Joyful", "Glad"]),
                tag="B")
        status, _ = gate.gate_row(ROW, a, b)
        self.assertEqual(status, "unresolved")

    def test_unresolved_neutral(self):
        a = rec(draft(["Glad", "Pleasing", "Lovely"], ["Angry", "Nasty", "Awful"],
                      expression="neutral"))
        b = rec(draft(["Glad", "Pleasing", "Lovely"], ["Angry", "Nasty", "Awful"],
                      expression="neutral"), tag="B")
        status, _ = gate.gate_row(ROW, a, b)
        self.assertEqual(status, "unresolved")

    def test_unresolved_expression_disagreement(self):
        a = rec(draft(["Angry", "Annoy", "Awful"], ["Happy", "Joyful", "Glad"],
                      expression="negative"))
        b = rec(draft(["Angry", "Annoy", "Awful"], ["Happy", "Joyful", "Glad"],
                      expression="positive"), tag="B")
        status, _ = gate.gate_row(ROW, a, b)
        self.assertEqual(status, "unresolved")

    def test_unresolved_disjoint_chosen(self):
        a = rec(draft(["Angry", "Annoy", "Awful"], ["Happy", "Joyful", "Glad"]))
        b = rec(draft(["Sadness", "Pain", "Grief"], ["Glad", "Happy", "Joyful"]),
                tag="B")
        status, _ = gate.gate_row(ROW, a, b)
        self.assertEqual(status, "unresolved")

    def test_missing_pass(self):
        status, _ = gate.gate_row(ROW, None, None)
        self.assertEqual(status, "missing")


class ManifestTests(unittest.TestCase):
    def test_manifest_loads(self):
        rows = common.load_manifest()
        self.assertEqual(len(rows), 2400)
        self.assertEqual(len(common.labelable_rows(rows)), 1920)
        splits = {r["split"] for r in rows}
        self.assertEqual(splits, {"train", "development", "test"})

    def test_no_test_rows_labelable(self):
        rows = common.labelable_rows()
        self.assertTrue(all(r["split"] != "test" for r in rows))


if __name__ == "__main__":
    unittest.main()
