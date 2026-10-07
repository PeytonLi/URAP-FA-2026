import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report import per_image_metrics


class ReportTests(unittest.TestCase):
    def row(self, **changes):
        return {"image_id": "example", "race": "White", "gender": "Female",
                "image_sha256": "hash", "valid_three": True, "word_count": 3,
                "negative_count": 1, "exact_format": True, "response_source": "sampled", **changes}

    def test_invalid_outputs_stay_in_attempts_not_word_denominators(self):
        rows = [self.row(), self.row(valid_three=False, word_count=0,
                                    exact_format=False, response_source="fallback_invalid")]
        item = per_image_metrics(rows)[0]
        self.assertEqual((item["attempts"], item["valid"], item["invalid"]), (2, 1, 1))
        self.assertEqual((item["negative_words"], item["positive_words"], item["counted_words"]), (1, 2, 3))
        self.assertEqual(item["negative_rate"], 1 / 3)
        self.assertEqual(item["fallback_final_outputs"], 1)

    def test_no_valid_response_is_unavailable_not_zero(self):
        item = per_image_metrics([self.row(valid_three=False, word_count=0)])[0]
        self.assertIsNone(item["negative_rate"])

    def test_metadata_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            per_image_metrics([self.row(), self.row(image_sha256="different")])

    def test_wrong_word_totals_are_rejected(self):
        with self.assertRaises(ValueError):
            per_image_metrics([self.row(word_count=2)])


if __name__ == "__main__":
    unittest.main()
