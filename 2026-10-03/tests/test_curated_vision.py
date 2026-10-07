import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import curated
import curated_vision_check
import study


class CuratedVisionTests(unittest.TestCase):
    @staticmethod
    def processor(*, corrupt_prefix=False, missing_image=False, missing_positions=False):
        import torch

        class Processor:
            image_token = "<image>"
            tokenizer = type("Tokenizer", (), {"convert_tokens_to_ids": lambda self, value: 99})()

            def apply_chat_template(self, messages, **kwargs):
                full = len(messages) == 2
                ids = [1, 99, 2, 3, 4] if full else [1, 99, 2]
                if full and corrupt_prefix:
                    ids[0] = 9
                output = {"input_ids": torch.tensor([ids]),
                          "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                          "mm_token_type_ids": torch.tensor([[0, 1, 0, 0, 0] if full else [0, 1, 0]]),
                          "image_position_ids": torch.tensor([[1, 2]])}
                if not missing_image:
                    output["pixel_values"] = torch.tensor([[0.25, 0.5]])
                if full and missing_positions:
                    output.pop("image_position_ids")
                return output

        return Processor()

    def test_native_fields_and_response_only_labels_are_retained(self):
        inputs, summary = curated_vision_check.response_inputs(
            self.processor(), Path("fixture.png"), "Happy, Joyful, Smiling")
        self.assertEqual(inputs["labels"].tolist(), [[-100, -100, -100, 3, 4]])
        self.assertIn("mm_token_type_ids", inputs)
        self.assertIn("image_position_ids", inputs)
        self.assertTrue(summary["prefix_equal"])
        self.assertEqual(summary["image_tokens"], 1)

    def test_template_prefix_mismatch_is_not_masked_or_truncated(self):
        with self.assertRaisesRegex(ValueError, "extend the inference prompt"):
            curated_vision_check.response_inputs(self.processor(corrupt_prefix=True),
                                                 Path("fixture.png"), "Happy, Joyful, Smiling")

    def test_image_fields_cannot_be_dropped(self):
        with self.assertRaisesRegex(ValueError, "image tensors"):
            curated_vision_check.response_inputs(self.processor(missing_image=True),
                                                 Path("fixture.png"), "Happy, Joyful, Smiling")
        with self.assertRaisesRegex(ValueError, "lost native processor fields"):
            curated_vision_check.response_inputs(self.processor(missing_positions=True),
                                                 Path("fixture.png"), "Happy, Joyful, Smiling")

    def test_profile_target_is_rejected_before_processing(self):
        with self.assertRaisesRegex(ValueError, "canonical three-word"):
            curated_vision_check.response_inputs(None, Path("fixture.png"), "A long invented biography.")

    def test_probe_preserves_complete_training_matching_blocks(self):
        rows = curated.selected_rows(study.dataset_config())
        selected = curated_vision_check.probe_blocks(rows)
        self.assertTrue(all(row["split"] == "train" for row in selected))
        counts = Counter(row["match_set_id"] for row in selected)
        original = Counter(row["match_set_id"] for row in rows if row["split"] == "train")
        self.assertTrue(all(count == original[block] for block, count in counts.items()))
        self.assertEqual({row["dataset"] for row in selected}, {"fairface", "utkface"})

    def test_empty_pool_does_not_fall_back_to_legacy_or_benchmark(self):
        with self.assertRaisesRegex(ValueError, "frozen training"):
            curated_vision_check.probe_blocks([])

    def test_probe_fixtures_are_format_tests_not_generated_labels(self):
        self.assertTrue(all(study.exact_target(value)
                            for value in curated_vision_check.PROBE_RESPONSES_NOT_LABELS))


if __name__ == "__main__":
    unittest.main()
