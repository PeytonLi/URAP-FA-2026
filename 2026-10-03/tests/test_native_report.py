"""An unfinished study must never generate a completed comparison report."""
import csv
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import native_report
import study


class NativeReportTests(unittest.TestCase):
    def test_render_all_nine_conditions_in_temporary_fixture_only(self):
        # Rendering check only: synthetic records, no pretrained weights/results.
        nr = native_report
        training, conditions = {}, {}
        with tempfile.TemporaryDirectory(prefix="urap-report-fixture-") as folder:
            root = Path(folder)
            frozen, policy = root / "cohort.json", root / "policy.json"
            study.save_json(frozen, {"rows": [{"dataset": "fairface", "split": "train",
                                              "image_id": "fixture_train", "chosen": "Happy, Joyful, Smiling",
                                              "rejected": "Angry, Negative, Sadness"},
                                             {"dataset": "utkface", "split": "development"}]})
            study.save_json(policy, {"fixture_only": True})
            dataset = {"manifest_path": "fixture.csv", "manifest_sha256": "fixture_hash",
                       "source_commit": "fixture_commit"}
            for name in nr.native_train.MODELS:
                for method in ("sft", "orpo"):
                    category = {"chosen_preference_accuracy": 0.5}
                    training[name, method] = ({"optimizer_updates": 507,
                        "gpu": "fixture only", "adapter": {"trainable_parameters": 1},
                        "elapsed_seconds": 1, "peak_gpu_memory_gib": 1, "versions": {"fixture": "1"},
                        "image_control": {"maximum_response_logprob_change": 1}, "development": [{
                        "response_loss": 2., "macro_category_preference_accuracy": 0.5,
                        "by_target_category": {"positive": category, "negative": category}}]}, root)
                for method in ("base", "sft", "orpo"):
                    key = name + "_" + method + "_native_balanced"
                    model = replace(nr.benchmark.fw.model_spec("fixture", 1), key=key, display_name=key)
                    rows = [{"model_key": key, "image_id": f"fixture_{i}", "iteration": n,
                             "race": "Black" if i < 6 else "White",
                             "gender": "Female" if i % 6 < 3 else "Male",
                             "image_sha256": f"fixture_hash_{i}", "valid_three": True,
                             "word_count": 3, "negative_count": 0, "exact_format": True,
                             "response_source": "sampled", "words": "Happy|Joyful|Smiling"}
                            for i in range(12) for n in range(1, 51)]
                    if name == "gemma4" and method == "base":
                        rows[300].update(valid_three=False, word_count=0, words="",
                                         exact_format=False, response_source="fallback_invalid")
                    if name == "qwen3vl" and method == "sft":
                        for row in rows[:300]:
                            row.update(negative_count=3, words="Angry|Annoy|Negative")
                    source = root / key
                    source.mkdir()
                    (source / "api_responses.jsonl").write_text("", encoding="utf-8")
                    (source / "responses.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
                    conditions[name, method] = (model, rows, source, {"adapter_sha256": None})
            original_analysis = nr.benchmark.fw.write_iat_analysis
            def small_analysis(records, models, output, **options):
                return original_analysis(records, models, output, **{**options, "bootstrap_samples": 20})
            with patch.object(study, "ROOT", root), \
                    patch.object(study, "dataset_config", return_value=dataset), \
                    patch.object(nr.pilot, "FROZEN", frozen), patch.object(nr.pilot, "POLICY", policy), \
                    patch.object(nr, "verified_results", return_value=(training, conditions)), \
                    patch.object(nr.benchmark.fw, "write_iat_analysis", side_effect=small_analysis):
                nr.main(root)
            output = root / "reports/native_balanced"
            manifest = json.loads((output / "report_manifest.json").read_text())
            self.assertEqual((manifest["training_conditions"], manifest["benchmark_slots"]), (6, 5400))
            self.assertEqual(manifest["chosen_training_triple_counts"], {"Happy, Joyful, Smiling": 1})
            self.assertEqual(manifest["rejected_training_triple_counts"], {"Angry, Negative, Sadness": 1})
            self.assertFalse(manifest["identity_disjointness_established"])
            self.assertEqual(manifest["training_implementation_sha256"], study.sha(Path(nr.native_train.__file__)))
            self.assertEqual(len(list(output.glob("*_per_image.csv"))), 9)
            self.assertEqual(len(list((output / "analysis/plots").glob("*_01_word_counts.png"))), 9)
            self.assertTrue((output / "analysis/plots/comparison_02_race_gaps.png").is_file())
            with (output / "analysis/word_counts.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 9 * 2 * 61)
            for name in nr.native_train.MODELS:
                text = (output / f"{name}.md").read_text()
                self.assertIn("## Changes from the native base", text)
                self.assertIn("fixture_commit", text)
                self.assertIn("### Actual execution evidence", text)
                self.assertIn(study.IAT_PROMPT, text)
                self.assertIn("Angry, Negative, Sadness", text)
                self.assertIn("The rejected answers contain 1 distinct ordered triples", text)
                self.assertIn("changed refusal rates", text)
                self.assertIn("Pre-training pixel-only control", text)
                self.assertIn("[Final LoRA adapter]", text)
                self.assertIn("[Final response records]", text)
                self.assertIn("Identity generalization remains unverified", text)
                self.assertIn("gradient-norm clipping at 1.0", text)
            gemma = (output / "gemma4.md").read_text()
            self.assertIn("Invalid final slots by photo: `{'fixture_6': 1}`", gemma)
            self.assertIn("refusals are not scored as neutral", gemma)
            qwen = (output / "qwen3vl.md").read_text()
            self.assertIn("Black-photo negative-word rate: 0.00% to 100.00% (change +100.00 pp)", qwen)
            self.assertIn("White-photo negative-word rate: 0.00% to 0.00% (change +0.00 pp)", qwen)

    def test_missing_or_incomplete_training_refuses_report_creation(self):
        self.assertEqual(Path(native_report.report.__file__).resolve(), study.ROOT / "report.py")
        with tempfile.TemporaryDirectory() as folder:
            recovered = Path(folder) / "recovered"
            output = Path(folder) / "output"
            with patch.object(study, "ROOT", output):
                self.assertEqual(native_report.verified_results(recovered, require_all=False), ({}, {}))
                with self.assertRaisesRegex(ValueError, "Exactly one completed"):
                    native_report.main(recovered)
                self.assertFalse(output.exists())
                run = recovered / "data/private/native_runs/gemma4_sft_fixture/run.json"
                study.save_json(run, {"status": "incomplete_runtime_limit"})
                with self.assertRaisesRegex(ValueError, "Incomplete or mismatched training"):
                    native_report.main(recovered)
                self.assertFalse(output.exists())
                settings = native_report.native_train.MODELS["gemma4"]
                study.save_json(run, {"status": "training_complete_evaluation_pending",
                    "optimizer_updates": 507, "completed_epochs": 3, "cohort": "resolved-balanced",
                    "model": settings["model"], "method": "sft", "weight_and_processor_revision": settings["revision"],
                    "targets_sha256": "fixture_hash", "pilot_policy_sha256": "fixture_hash",
                    "training_images": 676, "development_images": 180, "test_images_used": 0,
                    "profile_targets_used": 0, "seed": 0, "logical_batch_size": 4, "requested_epochs": 3,
                    "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 0,
                    "orpo_coefficient": None, "loss_weighting": "fixed_dataset_weights_then_logical_batch_mean"})
                with patch.object(study, "sha", return_value="fixture_hash"):
                    with self.assertRaisesRegex(ValueError, "recipe differs"):
                        native_report.main(recovered)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
