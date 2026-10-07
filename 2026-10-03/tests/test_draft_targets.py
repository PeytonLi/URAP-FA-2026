import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import curated
import draft_targets as draft
import study


class DraftTargetTests(unittest.TestCase):
    @staticmethod
    def value():
        return {"chosen": ["Happy", "Joyful", "Smiling"], "rejected": ["Angry", "Pain", "Sadness"],
                "expression": "positive", "visible_evidence": "Mouth corners are raised.",
                "rejected_reason": "The alternative contradicts the smile.", "confidence": 0.8,
                "review_flags": []}

    @staticmethod
    def row():
        return {"image_id": "fixture", "image_sha256": "fixture-hash", "split": "train",
                "image_path": "images/fairface/fixture.png", "dataset": "fairface",
                "expression_class": "positive"}

    def test_only_frozen_train_and_development_images_are_eligible(self):
        rows = draft.eligible_rows(study.dataset_config())
        self.assertEqual(len(rows), 1920)
        self.assertEqual({row["split"] for row in rows}, {"train", "development"})

    def test_teacher_cannot_approve_its_own_targets(self):
        record = draft.draft_record(self.row(), {**self.value(), "review_status": "approved",
                                               "reviewer": "teacher"}, "receipt.json")
        self.assertEqual(record["review_status"], "draft")
        self.assertEqual(record["reviewer"], "")
        self.assertIn("approval_missing", curated.target_failures(record, self.row()))
        self.assertEqual(record["prompt"], study.IAT_PROMPT)

    def test_noncanonical_duplicates_identical_and_unsupported_targets_are_rejected(self):
        variants = [
            {**self.value(), "chosen": ["Neutral", "Composed", "Thoughtful"]},
            {**self.value(), "chosen": ["Happy", "Happy", "Smiling"]},
            {**self.value(), "rejected": ["Happy", "Joyful", "Smiling"]},
            {**self.value(), "chosen": ["Sick", "Evil", "Selfish"]},
            {**self.value(), "confidence": True},
            {**self.value(), "confidence": float("nan")},
        ]
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                draft.validate_draft(value)

    def test_neutral_ambiguous_and_low_confidence_flags_cannot_be_omitted(self):
        value = {**self.value(), "expression": "neutral", "confidence": 0.4}
        record = draft.draft_record(self.row(), value, "receipt")
        self.assertIn("neutral_vocabulary_gap", record["review_flags"])
        self.assertIn("low_subjective_confidence", record["review_flags"])
        self.assertIn("automated_expression_disagreement", record["review_flags"])
        self.assertEqual(record["chosen"], "Happy, Joyful, Smiling")

    def test_schema_forces_the_existing_vocabulary_not_neutral_words(self):
        chosen = draft.RESPONSE_SCHEMA["properties"]["chosen"]["items"]["enum"]
        rejected = draft.RESPONSE_SCHEMA["properties"]["rejected"]["items"]["enum"]
        self.assertNotIn("Neutral", chosen)
        self.assertFalse(set(chosen) & draft.FORBIDDEN_CHOSEN)
        self.assertEqual(rejected, list(study.IAT_WORDS))

    def test_teacher_payload_has_real_image_and_no_demographic_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "image.png"
            image.write_bytes(b"test image bytes")
            payload = draft.teacher_payload(image)
        content = payload["messages"][1]["content"]
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(content[1]["text"], "Draft one image-specific preference pair as JSON.")
        self.assertEqual(payload["max_tokens"], 640)
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertNotIn("source_race", json.dumps(payload))
        self.assertNotIn("source_gender", json.dumps(payload))

    def test_usage_accounting_does_not_assume_missing_means_free(self):
        with self.assertRaisesRegex(ValueError, "missing_usage"):
            draft.usage_cost({}, {})
        cost, _ = draft.usage_cost({"usage": {"prompt_tokens": 1000, "completion_tokens": 200,
                                             "cost": 0.01}},
                                   {"input_usd_per_mtok": 1, "output_usd_per_mtok": 5})
        self.assertEqual(cost, Decimal("0.01"))

    def test_repair_payload_keeps_image_policy_and_records_uniqueness_instruction(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "image.png"
            image.write_bytes(b"fixture")
            original = draft.teacher_payload(image)
            repair = draft.teacher_payload(image, repair=True)
        self.assertEqual(repair["messages"][0], original["messages"][0])
        self.assertEqual(repair["messages"][1]["content"][0], original["messages"][1]["content"][0])
        self.assertIn("three DIFFERENT", repair["messages"][1]["content"][1]["text"])
        self.assertEqual(repair["response_format"], original["response_format"])

    def test_budget_reserves_before_calls_survives_restart_and_blocks_overspend(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ledger.json"
            budget = draft.Budget("0.05", path)
            ticket = budget.reserve({"request_id": "fixture"})
            restarted = draft.Budget("0.05", path)
            self.assertEqual(restarted.spent, Decimal("0.05"))
            with self.assertRaisesRegex(ValueError, "budget_cap"):
                restarted.reserve({"request_id": "second"})
            budget.settle(ticket, {"request_id": "fixture", "status": "done"}, Decimal("0.002"))
            self.assertEqual(draft.Budget("0.05", path).spent, Decimal("0.002"))

    def test_unknown_request_charge_is_reserved_not_refunded(self):
        with tempfile.TemporaryDirectory() as folder:
            budget = draft.Budget("0.05", Path(folder) / "ledger.json")
            ticket = budget.reserve({"request_id": "fixture"})
            budget.settle(ticket, {"status": "transport_unknown"})
            self.assertEqual(budget.spent, Decimal("0.05"))


if __name__ == "__main__":
    unittest.main()
