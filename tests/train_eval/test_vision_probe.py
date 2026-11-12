import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import remote_vision_probe as probe
import study
import vision


class VisionProbeTests(unittest.TestCase):
    @staticmethod
    def fake_image_processor():
        class Chunk:
            def __init__(self, tokens=None, expected_tokens=None, **kwargs):
                self.tokens = tokens
                self.expected_tokens = expected_tokens
                self.length = len(tokens) if tokens is not None else expected_tokens
                self.data = kwargs.get("data")

        tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: {"BOI": 3, "IMAGE": 4, "EOI": 5}[token])
        processor = SimpleNamespace(
            tokenizer=tokenizer, image_token="IMAGE", boi_token="BOI", eoi_token="EOI",
            apply_chat_template=lambda *args, **kwargs: {
                "input_ids": [SimpleNamespace(tolist=lambda: [1, 3, 4, 4, 5, 2])],
            },
        )
        return processor, SimpleNamespace(types=SimpleNamespace(EncodedTextChunk=Chunk, ImageChunk=Chunk))

    def test_boundary_candidate_preserves_official_prompt_coordinates_and_bytes(self):
        processor, fake_tinker = self.fake_image_processor()
        path = SimpleNamespace(read_bytes=lambda: b"original-image-bytes")
        with patch.dict(sys.modules, {"tinker": fake_tinker}):
            original, tokens, original_span = vision.prompt_chunks(processor, path)
            candidate, candidate_tokens, candidate_span = vision.prompt_chunks(
                processor, path, include_image_boundaries=True,
            )
        self.assertEqual(tokens, candidate_tokens)
        self.assertEqual(sum(chunk.length for chunk in original), 6)
        self.assertEqual(sum(chunk.length for chunk in candidate), 6)
        self.assertEqual(original[1].data, candidate[1].data)
        self.assertEqual(candidate[0].tokens, [1])
        self.assertEqual(candidate[2].tokens, [2])
        self.assertEqual(original_span["image_tokens"], candidate_span["image_tokens"])
        self.assertEqual(candidate_span["image_chunk_expected_tokens"], 4)

    def test_boundary_candidate_is_not_the_default(self):
        processor, fake_tinker = self.fake_image_processor()
        with patch.dict(sys.modules, {"tinker": fake_tinker}):
            chunks, _, spans = vision.prompt_chunks(processor, SimpleNamespace(read_bytes=lambda: b"image"))
        self.assertEqual(chunks[0].tokens, [1, 3])
        self.assertEqual(chunks[2].tokens, [5, 2])
        self.assertFalse(spans["image_boundaries_in_chunk"])
        self.assertEqual(spans["image_chunk_expected_tokens"], 2)

    def test_selected_probe_rows_are_training_only_and_complete_blocks(self):
        rows = vision.probe_rows(study.ROOT / "data/private/assets")
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["split"] for row in rows}, {"train"})
        for expression in ("positive", "neutral", "negative"):
            block = [row for row in rows if row["expression_class"] == expression]
            self.assertEqual(len({row["match_block_id"] for row in block}), 1)
            self.assertEqual({row["fairface_race"] for row in block}, {"Black", "White"})

    def test_missing_source_images_do_not_produce_a_text_only_probe(self):
        with patch.object(study, "sha", return_value="mismatch"):
            with self.assertRaises(ValueError):
                vision.probe_rows(study.ROOT / "data/private/assets")

    def test_remote_scores_use_response_weights_only(self):
        datum = SimpleNamespace(loss_fn_inputs={"weights": SimpleNamespace(data=[0.0, 0.5, 0.5])})
        result = SimpleNamespace(loss_fn_outputs=[{"logprobs": SimpleNamespace(data=[float("nan"), -2.0, -4.0])}])
        self.assertEqual(probe.response_scores(result, [datum]), [-3.0])

    def test_remote_score_lengths_must_align(self):
        datum = SimpleNamespace(loss_fn_inputs={"weights": SimpleNamespace(data=[0.0, 1.0])})
        result = SimpleNamespace(loss_fn_outputs=[{"logprobs": SimpleNamespace(data=[-2.0])}])
        with self.assertRaises(ValueError):
            probe.response_scores(result, [datum])

    def test_cleanup_identity_handles_native_nested_model(self):
        self.assertEqual(probe.trainer_base_model({"training_config": {"base_model": probe.BASE}}), probe.BASE)
        self.assertIsNone(probe.trainer_base_model({"training_config": {}}))

    def test_remote_score_uses_dense_sdk_values(self):
        class DenseArray:
            ndim = 1

            def tolist(self):
                return [0.0, -2.0, -4.0]

        tensor = SimpleNamespace(data=[-2.0, -4.0], sparse_crow_indices=[0, 2],
                                 to_numpy=lambda: DenseArray())
        datum = SimpleNamespace(loss_fn_inputs={"weights": SimpleNamespace(data=[0.0, 0.5, 0.5])})
        result = SimpleNamespace(loss_fn_outputs=[{"logprobs": tensor}])
        self.assertEqual(probe.response_scores(result, [datum]), [-3.0])

    def test_sparse_values_without_position_conversion_are_rejected(self):
        with self.assertRaises(ValueError):
            probe.tensor_values(SimpleNamespace(data=[-2.0], sparse_crow_indices=[0, 1]))

    def test_conditioning_requires_effect_larger_than_identical_input_noise(self):
        check = probe.conditioning_check([-2.0, -3.0], [-2.1, -2.9], [-2.2, -3.2])
        self.assertFalse(check["image_conditioning_observed"])
        self.assertAlmostEqual(check["conditioning_threshold"], 1.0)

    def test_conditioning_observes_multiple_effects_when_repeat_is_stable(self):
        check = probe.conditioning_check([-2.0, -3.0], [-2.0, -3.0], [-2.2, -2.7])
        self.assertTrue(check["image_conditioning_observed"])
        self.assertEqual(check["image_swap_scores_exceeding_threshold"], 2)

    def test_conditioning_rejects_empty_or_nonfinite_batches(self):
        with self.assertRaises(ValueError):
            probe.conditioning_check([], [], [])
        with self.assertRaises(ValueError):
            probe.conditioning_check([-2.0], [-2.0], [float("nan")])


if __name__ == "__main__":
    unittest.main()
