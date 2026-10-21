"""Tiny random native architectures, never pretrained study weights or labels."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import native_train as native
import study


def tiny_model(name):
    import torch
    from transformers import (Gemma4Config, Gemma4ForConditionalGeneration,
                              Qwen3VLConfig, Qwen3VLForConditionalGeneration,
                              Qwen3_5Config, Qwen3_5ForConditionalGeneration)
    torch.manual_seed(123)
    text = {"vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "use_cache": False}
    if name == "gemma4":
        config = Gemma4Config(text_config={**text, "global_head_dim": 8,
            "vocab_size_per_layer_input": 128, "hidden_size_per_layer_input": 0,
            "attention_k_eq_v": True, "num_global_key_value_heads": 2,
            "layer_types": ["sliding_attention", "full_attention"], "enable_moe_block": True,
            "num_experts": 4, "top_k_experts": 2, "moe_intermediate_size": 16},
            vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
                           "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8,
                           "patch_size": 2, "pooling_kernel_size": 3, "position_embedding_size": 16},
            boi_token_id=92, eoi_token_id=93, image_token_id=95)
        return Gemma4ForConditionalGeneration(config)
    text["rope_parameters"] = {"rope_type": "default", "rope_theta": 10000,
                               "mrope_section": [1, 1, 2], "partial_rotary_factor": 1.0}
    vision = {"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
              "patch_size": 2, "temporal_patch_size": 1, "spatial_merge_size": 1,
              "out_hidden_size": 32, "num_position_embeddings": 16}
    if name == "qwen3vl":
        vision["deepstack_visual_indexes"] = [0]
        config_class, model_class = Qwen3VLConfig, Qwen3VLForConditionalGeneration
    else:
        text.update(layer_types=["linear_attention", "full_attention"], linear_key_head_dim=8,
                    linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4)
        config_class, model_class = Qwen3_5Config, Qwen3_5ForConditionalGeneration
    return model_class(config_class(text_config=text, vision_config=vision, image_token_id=95,
                                   video_token_id=96, vision_start_token_id=92, vision_end_token_id=93))


def tiny_inputs(name, rejected=False):
    import torch
    ids = torch.tensor([[1, 92, 95, 95, 95, 95, 93, 2, 7 if rejected else 3, 8 if rejected else 4]])
    labels = ids.clone()
    labels[:, :8] = -100
    inputs = {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids),
              "pixel_values": torch.rand((1, 36, 12) if name == "gemma4" else (4, 12),
                                         generator=torch.Generator().manual_seed(55)),
              "mm_token_type_ids": torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0, 0, 0]])}
    if name == "gemma4":
        inputs["image_position_ids"] = torch.tensor([[[x, y] for y in range(6) for x in range(6)]])
    else:
        inputs["image_grid_thw"] = torch.tensor([[1, 2, 2]])
    return inputs


class NativeTrainTests(unittest.TestCase):
    def test_native_architecture_gradients_frozen_bases_image_control_and_adapter_roundtrip(self):
        import torch
        from peft import PeftModel
        torch.set_num_threads(2)
        receipts = []
        for name in native.MODELS:
            for method in ("sft", "orpo", "kto"):
                with self.subTest(model=name, method=method):
                    model, metadata = native.attach_lora(tiny_model(name))
                    chosen, rejected = tiny_inputs(name), tiny_inputs(name, True)
                    control = native.image_control(model, chosen)
                    with torch.no_grad():
                        native_loss = model(**chosen, use_cache=False).loss
                        response_loss = -native.response_logprobs(model, chosen).mean()
                    torch.testing.assert_close(response_loss, native_loss, atol=1e-6, rtol=1e-6)
                    with self.assertRaisesRegex(ValueError, "Text-only training"):
                        native.response_logprobs(model, {k: v for k, v in chosen.items() if k != "pixel_values"})
                    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
                    adapters = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
                    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
                    metrics = native.batch_update(model, optimizer, [(chosen, rejected)] * 2, method)
                    parameters = dict(model.named_parameters())
                    self.assertTrue(any(not torch.equal(p, parameters[n]) for n, p in adapters.items()))
                    self.assertTrue(all(torch.equal(p, parameters[n]) for n, p in frozen.items()))
                    self.assertGreater(metrics["gradient_norm_before_clip"], 0)
                    self.assertEqual(metrics["batch_examples" if method == "sft" else "batch_pairs"], 2)
                    model.eval()
                    expected = native.response_logprobs(model, chosen).detach()
                    with tempfile.TemporaryDirectory() as directory:
                        model.save_pretrained(directory)
                        restored = PeftModel.from_pretrained(tiny_model(name), directory)
                        restored.eval()
                        actual = native.response_logprobs(restored, chosen).detach()
                        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
                    receipts.append({"architecture": name, "method": method,
                                     "random_fixture_only": True, "study_weights_loaded": False,
                                     "study_optimizer_updates": 0, "fixture_optimizer_updates": 1,
                                     "frozen_base_unchanged": True, "adapter_roundtrip": True,
                                     "response_loss_matches_native_shifted_label_loss": True,
                                     "trainable_parameters": metadata["trainable_parameters"], **control})
        print("Native architecture fixture checks: " + json.dumps(receipts), flush=True)

    def test_incomplete_full_cohort_refuses_loading_weights_or_creating_a_run(self):
        from transformers import AutoModelForImageTextToText
        with tempfile.TemporaryDirectory() as folder, patch.object(study, "ROOT", Path(folder)), \
                patch.object(native.curated, "training_rows", side_effect=ValueError("incomplete targets")), \
                patch.object(native.study, "dataset_config", return_value={"fixture": True}), \
                patch.object(AutoModelForImageTextToText, "from_pretrained") as load:
            with self.assertRaisesRegex(ValueError, "incomplete targets"):
                native.train("gemma4", "sft", 60)
            load.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
