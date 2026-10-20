import contextlib
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import curated
import study


class CuratedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = study.dataset_config()
        cls.rows = curated.selected_rows(cls.config)

    def test_manifest_matches_active_dataset_config(self):
        expected_splits = {key: value for key, value in
                           self.config.get("split_counts", {}).items()}
        self.assertEqual(len(self.rows), sum(expected_splits.values()))
        self.assertEqual(Counter(row["split"] for row in self.rows), expected_splits)
        expected_datasets = {key: value for key, value in
                             self.config.get("dataset_counts", {}).items()}
        if expected_datasets:
            self.assertEqual(Counter(row["dataset"] for row in self.rows), expected_datasets)
        self.assertFalse({"chosen", "rejected", "prompt"} & self.rows[0].keys())

    def test_split_changes_are_rejected(self):
        rows = [{**row} for row in self.rows]
        rows[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "split counts"):
            curated.validate_rows(rows, self.config)

    def test_split_balanced_but_broken_matching_sets_are_rejected(self):
        rows = [{**row} for row in self.rows]
        train = next(row for row in rows if row["split"] == "train")
        test = next(row for row in rows if row["split"] == "test")
        train["split"], test["split"] = test["split"], train["split"]
        with self.assertRaisesRegex(ValueError, "crosses research splits"):
            curated.validate_rows(rows, self.config)

    def test_unsafe_paths_and_benchmark_overlap_are_rejected(self):
        rows = [{**row} for row in self.rows]
        rows[0]["image_path"] = "../outside.png"
        with self.assertRaisesRegex(ValueError, "image path"):
            curated.validate_rows(rows, self.config)
        with self.assertRaisesRegex(ValueError, "benchmark overlaps"):
            curated.validate_rows(self.rows, self.config, {self.rows[0]["image_sha256"]})

    def test_targets_require_image_specific_approval_not_expression_metadata(self):
        image = next(row for row in self.rows if row["split"] == "train")
        target = {key: image[key] for key in ("image_id", "image_sha256", "split")}
        target.update(chosen="Happy, Joyful, Smiling", rejected="Pain, Grief, Sadness",
                      review_status="approved", reviewer="Fixture reviewer",
                      target_provenance="Unit-test fixture, not real study labels")
        self.assertEqual(curated.target_failures(target, image), [])
        self.assertIn("approval_missing", curated.target_failures({**target, "review_status": "draft"}, image))
        self.assertIn("image_sha256_mismatch", curated.target_failures({**target, "image_sha256": "old54"}, image))
        self.assertIn("chosen_format", curated.target_failures({**target, "chosen": None}, image))
        self.assertIn("approval_missing", curated.target_failures({**target, "reviewer": None}, image))
        self.assertIn("target_provenance_missing",
                      curated.target_failures({**target, "target_provenance": None}, image))
        test_image = next(row for row in self.rows if row["split"] == "test")
        self.assertIn("heldout_test_target_forbidden", curated.target_failures(target, test_image))

    def test_no_target_file_means_no_invented_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            targets, audit = curated.reviewed_targets(self.rows, Path(folder) / "absent.json")
        self.assertEqual(targets, {})
        self.assertEqual(audit["required_records"], 1920)
        self.assertFalse(audit["ready"])

    def test_ai_targets_require_current_user_authorization_and_keep_their_source(self):
        image = next(row for row in self.rows if row["split"] == "train")
        target = {key: image[key] for key in ("image_id", "image_sha256", "split")}
        target.update(chosen="Happy, Joyful, Smiling", rejected="Pain, Grief, Sadness",
                      review_status="approved", reviewer="Codex (AI)", reviewer_kind="ai",
                      target_provenance="AI self-approval; fixture only", draft_source="codex_direct_visual_review",
                      approval_authorization_sha256="a" * 64)
        self.assertEqual(curated.target_failures(target, image, "a" * 64), [])
        self.assertIn("ai_approval_not_authorized", curated.target_failures(target, image))
        self.assertIn("ai_approval_requires_direct_inspection",
                      curated.target_failures({**target, "draft_source": "teacher"}, image, "a" * 64))
        self.assertIn("identical_responses", curated.target_failures(
            {**target, "rejected": "Smiling, Joyful, Happy"}, image, "a" * 64))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "targets.json"
            study.save_json(path, [target])
            with patch.object(curated, "ai_approval_authorization", return_value="a" * 64):
                approved, audit = curated.reviewed_targets([image], path)
            self.assertTrue(audit["ready"])
            self.assertEqual((audit["ai_approved_records"], audit["human_approved_records"]), (1, 0))
            with patch.object(curated, "ai_approval_authorization", return_value=None):
                self.assertFalse(curated.reviewed_targets([image], path)[1]["ready"])
            self.assertEqual(approved[image["image_id"]]["reviewer_kind"], "ai")

    def test_malformed_target_id_is_rejected_without_crashing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "targets.json"
            study.save_json(path, [{"image_id": []}])
            targets, audit = curated.reviewed_targets(self.rows, path)
        self.assertEqual(targets, {})
        self.assertFalse(audit["ready"])
        self.assertEqual(audit["errors"][0]["failures"], ["image_id_not_string"])

    def test_missing_images_keep_active_pool_not_historical_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(study, "ROOT", root), patch.object(study, "dataset_config", return_value=self.config):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(curated.prepare(root / "empty-assets"), 2)
            report = json.loads((root / "data/manifest.json").read_text())
            self.assertEqual(report["dataset_id"], self.config["dataset_id"])
            self.assertEqual(len(report["missing_images"]), len(self.rows))
            self.assertFalse(report["data_ready"])
            self.assertFalse((root / "data/pairs.csv").exists())

    def test_missing_configuration_does_not_fall_back_to_legacy(self):
        with patch.object(study, "dataset_config", return_value=None), patch.object(study, "prepare_legacy") as legacy:
            with self.assertRaisesRegex(ValueError, "historical fallback"):
                study.prepare()
            legacy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
