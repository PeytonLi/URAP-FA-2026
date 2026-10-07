import json
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import study


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [row for row in study.read_csv(study.SOURCE / "all_claude_eligible_pairs.csv")
                    if row["task_type"] == "iat_3word"]
        cls.forbidden = {row["sha256"] for row in study.read_csv(
            study.REPO / "2026-09-11/data/iat12/manifest.csv")}

    def test_existing_labels_and_strict_three_word_format(self):
        self.assertEqual(len(self.rows), 67)
        self.assertTrue(all(not study.label_failures(row, self.forbidden) for row in self.rows))
        self.assertTrue(study.exact_target("Happy, Joyful, Smiling"))
        for target in ("Happy,Happy,Smiling", "Happy, Happy, Smiling", "happy, Joyful, Smiling", "Happy"):
            self.assertFalse(study.exact_target(target))

    def test_leakage_is_rejected(self):
        row = {**self.rows[0], "image_sha256": next(iter(self.forbidden))}
        self.assertIn("benchmark_image_overlap", study.label_failures(row, self.forbidden))

    def test_split_is_deterministic_balanced_and_disjoint(self):
        rows, excluded = study.match_and_split(self.rows)
        self.assertEqual((rows, excluded), study.match_and_split(list(reversed(self.rows))))
        self.assertEqual(len(rows), 54)
        self.assertEqual(len(excluded), 13)
        self.assertEqual(Counter(row["split"] for row in rows), {"train": 38, "development": 16})
        for split, count in (("train", 19), ("development", 8)):
            subset = [row for row in rows if row["split"] == split]
            self.assertEqual(Counter(row["fairface_race"] for row in subset), {"Black": count, "White": count})
            negative = [row for row in subset if row["expression_class"] == "negative"]
            self.assertEqual(len(negative), 6)
            self.assertEqual({row["expression_fine"] for row in negative}, {"sad"})
        blocks = defaultdict(list)
        for row in rows:
            blocks[row["match_block_id"]].append(row)
        for block in blocks.values():
            self.assertEqual(len(block), 2)
            self.assertEqual(len({row["split"] for row in block}), 1)
            self.assertEqual(len({row["chosen"] for row in block}), 1)
        train = {row["image_sha256"] for row in rows if row["split"] == "train"}
        dev = {row["image_sha256"] for row in rows if row["split"] == "development"}
        self.assertFalse(train & dev)
        self.assertFalse((train | dev) & self.forbidden)

    def test_export_contains_images_and_targets_not_profiles_or_race(self):
        row = study.match_and_split(self.rows)[0][0]
        for method in ("sft", "orpo"):
            example = study.training_example(row, "data:image/png;base64,TEST", method)
            branches = [example] if method == "sft" else [example["chosen"], example["rejected"]]
            for branch in branches:
                messages = branch["messages"]
                self.assertEqual(len(messages), 2)
                self.assertEqual(messages[0]["content"][0]["type"], "image_url")
                self.assertEqual(messages[0]["content"][1]["text"], study.IAT_PROMPT)
                self.assertTrue(study.exact_target(messages[1]["content"]))
            self.assertNotIn("fairface", json.dumps(example))
        with self.assertRaises(ValueError):
            study.training_example({**row, "task_type": "grounded_profile"}, "image", "sft")

    def test_missing_images_block_export(self):
        with patch.object(study, "prepare", return_value=2), patch.object(study, "read_csv") as read:
            self.assertEqual(study.export(), 2)
            read.assert_not_called()

    def test_managed_preference_schema_preserves_image_and_both_targets(self):
        row = study.match_and_split(self.rows)[0][0]
        image_url = "data:image/png;base64,TEST"
        example = study.training_example(row, image_url, "orpo", "managed")
        self.assertEqual(set(example), {"input", "preferred_output", "non_preferred_output"})
        user = example["input"]["messages"]
        self.assertEqual(len(user), 1)
        self.assertEqual(user[0]["content"][0]["image_url"]["url"], image_url)
        self.assertEqual(user[0]["content"][1]["text"], study.IAT_PROMPT)
        self.assertEqual(example["preferred_output"][0]["content"], row["chosen"])
        self.assertEqual(example["non_preferred_output"][0]["content"], row["rejected"])
        self.assertNotIn("chosen", example)
        with self.assertRaises(ValueError):
            study.training_example(row, image_url, "orpo", "unknown")


if __name__ == "__main__":
    unittest.main()
