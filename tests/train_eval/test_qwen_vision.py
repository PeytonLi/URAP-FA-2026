import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen_vision_check
import test_curated_vision as fixtures


class QwenVisionTests(unittest.TestCase):
    @staticmethod
    def processor(*, invalid_grid=False, changed_grid=False, changed_pixels=False):
        import torch

        base = fixtures.CuratedVisionTests.processor()
        base.tokenizer.eos_token = "<|im_end|>"
        base.tokenizer.decode = lambda *args, **kwargs: "Happy, Joyful, Smiling<|im_end|>\n"
        original = base.apply_chat_template

        def apply(messages, **kwargs):
            output = original(messages, **kwargs)
            full = len(messages) == 2
            output["image_grid_thw"] = torch.tensor([[1, 2, 2]])
            if invalid_grid:
                output["image_grid_thw"] = torch.tensor([[1, 2]])
            if full and changed_grid:
                output["image_grid_thw"] = torch.tensor([[1, 4, 4]])
            if full and changed_pixels:
                output["pixel_values"] = torch.tensor([[0.75, 0.5]])
            return output

        base.apply_chat_template = apply
        return base

    def test_both_intended_qwen_models_have_immutable_processor_revisions(self):
        self.assertEqual({value["model"] for value in qwen_vision_check.MODELS.values()},
                         {"Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3.5-9B"})
        for value in qwen_vision_check.MODELS.values():
            self.assertRegex(value["revision"], r"^[0-9a-f]{40}$")

    def test_native_grid_pixels_and_response_only_labels_are_retained(self):
        inputs, summary = qwen_vision_check.checked_inputs(
            self.processor(), Path("fixture.png"), "Happy, Joyful, Smiling")
        self.assertEqual(inputs["labels"].tolist(), [[-100, -100, -100, 3, 4]])
        self.assertEqual(summary["image_grid_thw"], [[1, 2, 2]])
        self.assertTrue(summary["native_complete_assistant_template_used"])
        self.assertTrue(summary["supervised_response_exact"])
        self.assertIn("mm_token_type_ids", inputs)

    def test_missing_or_invalid_image_grid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "native grid"):
            qwen_vision_check.checked_inputs(self.processor(invalid_grid=True),
                                             Path("fixture.png"), "Happy, Joyful, Smiling")
        with self.assertRaisesRegex(ValueError, "native grid"):
            qwen_vision_check.checked_inputs(fixtures.CuratedVisionTests.processor(),
                                             Path("fixture.png"), "Happy, Joyful, Smiling")

    def test_changed_image_grid_is_not_silently_reused(self):
        with self.assertRaisesRegex(ValueError, "image field changed"):
            qwen_vision_check.checked_inputs(self.processor(changed_grid=True),
                                             Path("fixture.png"), "Happy, Joyful, Smiling")

    def test_changed_image_pixels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "image preprocessing"):
            qwen_vision_check.checked_inputs(self.processor(changed_pixels=True),
                                             Path("fixture.png"), "Happy, Joyful, Smiling")

    def test_extra_thinking_or_control_text_is_not_supervised_as_the_answer(self):
        processor = self.processor()
        processor.tokenizer.decode = lambda *args, **kwargs: "</think>Happy, Joyful, Smiling<|im_end|>\n"
        with self.assertRaisesRegex(ValueError, "exactly the response"):
            qwen_vision_check.checked_inputs(processor, Path("fixture.png"), "Happy, Joyful, Smiling")

    def test_qwen_thinking_switch_is_model_specific(self):
        self.assertIsNone(qwen_vision_check.MODELS["qwen3vl"]["enable_thinking"])
        self.assertIs(qwen_vision_check.MODELS["qwen35"]["enable_thinking"], False)


if __name__ == "__main__":
    unittest.main()
