"""Diversity cannot be manufactured by permutation, fallback or missing slots."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import variety


def fixture():
    return [{"image_id": str(i), "image_sha256": f"hash{i}", "iteration": j + 1,
             "race": "Black" if i < 6 else "White", "valid_three": "True",
             "negative_count": "3", "words": "Grief|Pain|Sadness",
             "sampled_raw": ("Grief, Pain, Sadness" if j % 2 else "Pain, Grief, Sadness"),
             "response_source": "sampled"}
            for i in range(12) for j in range(50)]


class VarietyTests(unittest.TestCase):
    def test_permutations_are_one_unordered_triple(self):
        rows = fixture()
        variety.validate_slots(rows)
        summary, images, _ = variety.summarize(rows, "sampled_exact")
        self.assertEqual(summary["distinct_unordered_triples"], 1)
        self.assertEqual(summary["distinct_words_union"], 3)
        self.assertEqual(summary["word_tv_image_weighted"], 0)
        self.assertEqual(summary["word_tv_exact_p"], 1)
        self.assertEqual(summary["permutation_assignments"], 924)
        self.assertEqual(images[0]["distinct_ordered_triples"], 2)
        self.assertEqual(images[0]["dominant_unordered_triple_share"], 1)
        self.assertEqual(images[0]["triple_entropy_slots"], 0)

    def test_unknown_words_do_not_count_as_coverage(self):
        self.assertEqual(variety.exact_sample("Calm, Neutral, Happy"), [])
        self.assertEqual(variety.exact_sample("Happy, Joyful, Smiling. explanation"), [])
        self.assertEqual(variety.exact_sample("Grief, Grief, Pain"), [])
        self.assertEqual(variety.exact_sample(" happy , joyful, SMILING "), ["Happy", "Joyful", "Smiling"])

    def test_fallback_cannot_hide_invalid_sample(self):
        rows = fixture()
        for row in rows:
            row["sampled_raw"] = "I cannot answer."
            row["response_source"] = "fallback"
        sampled, _, _ = variety.summarize(rows, "sampled_exact")
        final, _, _ = variety.summarize(rows, "final_scored")
        self.assertEqual(sampled["distinct_words_union"], 0)
        self.assertFalse(sampled["inference_available"])
        self.assertEqual(final["distinct_words_union"], 3)
        self.assertEqual(final["fallback_slots"], 600)

    def test_missing_duplicate_race_drift_and_hash_drift_fail(self):
        for mutate in (
                lambda rows: rows.pop(),
                lambda rows: rows.__setitem__(0, copy.deepcopy(rows[1])),
                lambda rows: rows[0].update(race="White"),
                lambda rows: rows[0].update(image_sha256="changed")):
            rows = fixture()
            mutate(rows)
            with self.assertRaises(ValueError):
                variety.validate_slots(rows)

    def test_unseen_training_word_is_measured_not_assumed_impossible(self):
        summary, _, _ = variety.summarize(fixture(), "sampled_exact", {"Grief", "Pain"})
        self.assertEqual(summary["responses_with_unseen_training_word_share"], 1)

    def test_entropy_uses_full_frozen_vocabulary(self):
        from math import log2
        self.assertAlmostEqual(variety.entropy({"Grief": 1, "Pain": 1, "Sadness": 1}, 61),
                               log2(3) / log2(61))


if __name__ == "__main__":
    unittest.main()
