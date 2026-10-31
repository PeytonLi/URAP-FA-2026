import hashlib
import json
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark
import iat_prompts
import native_benchmark
import native_train
import prime_native
import remote_native
import runpod_native
import study
from multimodel_bias.iat import Stimulus, deterministic_seed, deterministic_task_id


class PromptSweepTests(unittest.TestCase):
    def test_exact_preselected_scope_and_no_training(self):
        with patch.object(iat_prompts, "checkpoint_identity", return_value={"verified": True}):
            plan = iat_prompts.sweep_plan({("gemma3_12b", "orpo"): ({}, Path("epoch_3"))})
        self.assertEqual(plan["mode"], "prompt_sweep")
        self.assertFalse(plan["training_allowed"])
        self.assertFalse(plan["extra_benchmarks"])
        self.assertEqual(plan["methods"], ["base", "orpo"])
        self.assertEqual([p["prompt_id"] for p in plan["prompts"]], ["3", "4", "5"])
        self.assertEqual(plan["expected_sampled_attempts"], 3 * 2 * 12 * 50)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            iat_prompts.sweep_plan({})

    def test_run_plan_fails_closed_before_bootstrap_or_detach(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(remote_native, "ROOT", Path(tmp)):
            for mode in (None, "unknown", "prompt_sweep", "extras_only"):
                with self.assertRaises(ValueError), patch.object(remote_native, "bootstrap_uv") as setup:
                    remote_native.main(mode)
                setup.assert_not_called()
                with self.assertRaises(ValueError), patch.object(remote_native.subprocess, "Popen") as child:
                    remote_native.detach(mode)
                child.assert_not_called()
            path = Path(tmp) / "data/private/run_plan.json"
            path.parent.mkdir(parents=True)
            for invalid in ({}, [], {"mode": "training"}, {"mode": "extras_only"}):
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    remote_native.required_plan("prompt_sweep")
            path.write_text('{"mode":"prompt_sweep"}')
            self.assertEqual(remote_native.required_plan("prompt_sweep"), {"mode": "prompt_sweep"})

    def test_inventory_requires_real_adapter_config_and_checks_every_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for files in ({}, {"processor_config.json": "a" * 64},
                          {"../adapter_model.safetensors": "a" * 64}):
                with self.assertRaises(ValueError):
                    native_train.verify_checkpoint_inventory(root, files)
            files = {}
            for name in ("adapter_model.safetensors", "adapter_config.json"):
                (root / name).write_bytes(b"fixture")
                files[name] = study.sha(root / name)
            native_train.verify_checkpoint_inventory(root, files)
            with self.assertRaises(ValueError):
                native_train.verify_checkpoint_inventory(root, {**files, "../escape": "a" * 64})
            with self.assertRaises(ValueError):
                native_train.verify_checkpoint_inventory(root, {**files, "missing": "a" * 64})
            (root / "adapter_model.safetensors").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                native_train.verify_checkpoint_inventory(root, files)

    def test_wrong_adapter_is_not_authorized_even_with_consistent_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "epoch_3"
            checkpoint.mkdir()
            files = {}
            for name in ("adapter_model.safetensors", "adapter_config.json"):
                (checkpoint / name).write_bytes(b"wrong adapter")
                files[name] = study.sha(checkpoint / name)
            (root / "run.json").write_text(json.dumps({"checkpoints": [{"files": files}]}))
            with self.assertRaisesRegex(ValueError, "authorized"):
                iat_prompts.checkpoint_identity(checkpoint)

    def test_prompt_id_derives_text_and_refuses_mismatched_text_before_initialization(self):
        with self.assertRaisesRegex(ValueError, "Prompt text"):
            native_benchmark.NativeRunner(None, None, "gemma3_12b", 0,
                                          prompt=study.IAT_PROMPT, prompt_id="3")
        with patch.object(iat_prompts, "PROMPTS", {**iat_prompts.PROMPTS, "3": "changed"}):
            with self.assertRaisesRegex(ValueError, "Frozen"):
                iat_prompts.prompt_spec("3")

    def test_completion_requires_exact_six_conditions_and_no_new_training(self):
        summary = {"status": "prompt_sweep_complete_reports_pending",
                   "training_commands_started": 0,
                   "conditions": [{"model": "gemma3_12b", "benchmark": m, "prompt_id": p,
                                   "development": False, "exit_code": 0}
                                  for p in ("3", "4", "5") for m in ("base", "orpo")]}
        self.assertTrue(iat_prompts.complete_summary(summary))
        for changed in ({"training_commands_started": 1}, {"conditions": summary["conditions"][:-1]},
                        {"status": "training_and_benchmarks_complete_reports_pending"},
                        {"conditions": summary["conditions"] + [summary["conditions"][0]]}):
            self.assertFalse(iat_prompts.complete_summary({**summary, **changed}))

    def test_prompts_preserve_identical_vocabulary_and_order(self):
        for prompt_id in iat_prompts.PROMPT_IDS:
            spec = iat_prompts.prompt_spec(prompt_id)
            words = spec["prompt"].split("[", 1)[1].split("]", 1)[0].split(", ")
            self.assertEqual(words, list(study.IAT_WORDS))
            self.assertEqual(len(words), 61)
            self.assertEqual(spec["prompt_sha256"],
                             hashlib.sha256(spec["prompt"].encode()).hexdigest())
            self.assertNotIn("distinct", spec["prompt"])
            self.assertNotIn("unbiased", spec["prompt"])
        self.assertIn("person’s visible facial expression", iat_prompts.PROMPTS["5"])
        self.assertIn("most associate", iat_prompts.PROMPTS["4"])

    def test_rejects_unselected_or_ambiguous_ids(self):
        for value in ("1", "2", "6", 3, "", None):
            with self.assertRaises(ValueError):
                iat_prompts.prompt_spec(value)
        with self.assertRaises(ValueError):
            iat_prompts.benchmark_version("sft", "3")

    def test_task_ids_unique_across_all_3600_slots_and_seeds_paired(self):
        model = benchmark.fw.model_spec("google/gemma-3-12b-it", 1, key="fixture",
                                       display_name="fixture", base_model="fixture")
        ids = set()
        seeds = {}
        for prompt_id in iat_prompts.PROMPT_IDS:
            for method in ("base", "orpo"):
                version = iat_prompts.benchmark_version(method, prompt_id)
                for index in range(1, 13):
                    stimulus = SimpleNamespace(sha256=str(index))
                    for iteration in range(1, 51):
                        task_id = deterministic_task_id(model, stimulus, iteration, version)
                        self.assertNotIn(task_id, ids)
                        ids.add(task_id)
                        seed = deterministic_seed(0, iteration, index)
                        slot = iteration, index
                        self.assertEqual(seed, seeds.setdefault(slot, seed))
        self.assertEqual(len(ids), 3600)

    def test_stale_outputs_rejected_before_weight_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent, key = iat_prompts.output_location(root, "base", "3")
            (parent / key).mkdir(parents=True)
            with patch.object(study, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "start fresh"):
                    native_benchmark.run("gemma3_12b", "base", root, 1, prompt_id="3")

    def test_prompt_and_extras_are_mutually_exclusive_before_provider_calls(self):
        with self.assertRaises(ValueError):
            runpod_native.launch(prompt_sweep=True)
        with self.assertRaises(ValueError):
            runpod_native.launch(resume_from=Path("missing"), extras_only=["orpo"], prompt_sweep=True)
        with self.assertRaises(ValueError):
            prime_native.package_inputs(Path("unused"), prompt_sweep=True)

    def test_development_not_allowed(self):
        with self.assertRaisesRegex(ValueError, "frozen IAT-12"):
            native_benchmark.run("gemma3_12b", "base", Path("."), 1,
                                 development=True, prompt_id="3")

    def test_real_runner_logs_prompt_and_uses_native_pixels_and_fixed_decoding(self):
        import torch
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "synthetic.png"
            Image.new("RGB", (2, 2), "white").save(image)
            stimulus = Stimulus("fixture", image.name, "Black", "Male",
                                2, 2, study.sha(image), image)
            seen = {}

            class Processor:
                tokenizer = SimpleNamespace(decode=lambda *a, **k: "Happy, Love, Friend")

                def apply_chat_template(self, messages, **kwargs):
                    seen["messages"], seen["options"] = messages, kwargs
                    return {"input_ids": torch.ones((1, 2), dtype=torch.long),
                            "pixel_values": torch.ones((1, 3, 2, 2))}

            class Model:
                device, dtype = "cpu", torch.float32

                def generate(self, **kwargs):
                    seen["generation"] = kwargs
                    return torch.tensor([[1, 1, 2, 3, 4]])

            model = benchmark.fw.model_spec("google/gemma-3-12b-it", 1, key="fixture",
                                           display_name="fixture", base_model="fixture")
            config = replace(benchmark.fw.run_config(1),
                             benchmark_version=iat_prompts.benchmark_version("base", "3"))
            runner = native_benchmark.NativeRunner(
                Model(), Processor(), "gemma3_12b", time.monotonic() + 100,
                "fixture", config, [model], [stimulus], output_root=root,
                prompt=iat_prompts.PROMPTS["3"], prompt_id="3")
            runner._run_task(model, stimulus, 1, 1)
            self.assertEqual(seen["messages"][0]["content"][1]["text"], iat_prompts.PROMPTS["3"])
            for name, value in {"max_new_tokens": 30, "do_sample": True,
                                "top_k": 0, "top_p": 1.0, "temperature": 1.0}.items():
                self.assertEqual(seen["generation"][name], value)
            for path in (runner.api_path, runner.records_path):
                row = json.loads(path.read_text().splitlines()[0])
                self.assertEqual(row["prompt_id"], "3")
                self.assertEqual(row["prompt_sha256"], iat_prompts.prompt_spec("3")["prompt_sha256"])
            self.assertEqual(seen["generation"]["pixel_values"].numel(), 12)


class PromptBatchTests(unittest.TestCase):
    def test_batches_have_frozen_distinct_ids_and_plans(self):
        self.assertEqual(iat_prompts.batch_ids("A"), ("p5", "p6", "p7", "p8"))
        self.assertEqual(iat_prompts.batch_ids("B"), ("p9", "p10a", "p10b", "p11", "p12", "p13"))
        self.assertFalse(set(iat_prompts.PROMPT_IDS) & set(iat_prompts.TEMPLATES))
        with self.assertRaises(ValueError):
            iat_prompts.batch_ids("C")
        for batch, count in (("A", 4), ("B", 6)):
            with patch.object(iat_prompts, "checkpoint_identity", return_value={"verified": True}):
                plan = iat_prompts.sweep_plan({("gemma3_12b", "orpo"): ({}, Path("epoch_3"))}, batch)
            self.assertEqual(plan["batch"], batch)
            self.assertFalse(plan["training_allowed"])
            self.assertEqual(plan["expected_sampled_attempts"], count * 1200)
            self.assertEqual([p["prompt_id"] for p in plan["prompts"]], list(iat_prompts.batch_ids(batch)))
        with patch.object(iat_prompts, "TEMPLATES", {**iat_prompts.TEMPLATES, "p8": "changed"}):
            with self.assertRaisesRegex(ValueError, "Frozen"):
                iat_prompts.prompt_spec("p8")

    def test_rendering_keeps_vocabulary_and_is_paired_across_methods(self):
        for prompt_id in iat_prompts.TEMPLATES:
            text = iat_prompts.render(prompt_id, "img", "Black", 1001)
            words = text.split("[", 1)[1].split("]", 1)[0].split(", ")
            self.assertEqual(sorted(words), sorted(study.IAT_WORDS))
            self.assertNotIn("{", text)
        self.assertEqual(iat_prompts.render("p5", "i", "Black", 1).split("[", 1)[1].split(", ")[0], "Sick")
        first = iat_prompts.render("p6", "img", "White", 1001)
        self.assertEqual(first, iat_prompts.render("p6", "img", "White", 1001))
        self.assertNotEqual(first, iat_prompts.render("p6", "img", "White", 2001))
        self.assertTrue(iat_prompts.render("p10a", "i", "Black", 1).startswith("This is a photo of a Black person."))
        self.assertTrue(iat_prompts.render("p10b", "i", "Black", 1).startswith("This is a photo of a White person."))
        self.assertTrue(iat_prompts.render("p10b", "i", "White", 1).startswith("This is a photo of a Black person."))
        with self.assertRaises(ValueError):
            iat_prompts.render("p10a", "i", "Asian", 1)

    def test_describe_then_choose_scores_only_the_words_line(self):
        self.assertEqual(iat_prompts.answer_text("p7", "Description: sad and happy.\nWords: Love, Joyful, Pain"),
                         "Love, Joyful, Pain")
        self.assertEqual(iat_prompts.answer_text("p7", "Description: sad and happy."), "")
        for text in ("**Words:** Love, Joyful, Pain", "words: Love, Joyful, Pain",
                     "- Words: Love, Joyful, Pain", "Description: calm. Words: Love, Joyful, Pain",
                     "**Description:** calm\n**Words**: Love, Joyful, Pain"):
            self.assertEqual(iat_prompts.answer_text("p7", text), "Love, Joyful, Pain")
        self.assertEqual(iat_prompts.answer_text("p6", "Love, Joy"), "Love, Joy")

    def test_batch_completion_counts(self):
        ids = iat_prompts.batch_ids("B")
        summary = {"status": "prompt_sweep_complete_reports_pending", "training_commands_started": 0,
                   "conditions": [{"model": "gemma3_12b", "benchmark": m, "prompt_id": p,
                                   "development": False, "exit_code": 0}
                                  for p in ids for m in ("base", "orpo")]}
        self.assertTrue(iat_prompts.complete_summary(summary, "B"))
        self.assertFalse(iat_prompts.complete_summary(summary, "A"))
        self.assertFalse(iat_prompts.complete_summary({**summary, "conditions": summary["conditions"][:-1]}, "B"))

    def test_runner_renders_per_attempt_gray_image_and_long_answers(self):
        import torch
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "synthetic.png"
            Image.new("RGB", (4, 3), "white").save(image)
            stimulus = Stimulus("fixture", image.name, "Black", "Male", 4, 3, study.sha(image), image)
            seen = []

            class Processor:
                tokenizer = SimpleNamespace(decode=lambda *a, **k: "Description: calm.\nWords: Happy, Love, Friend")

                def apply_chat_template(self, messages, **kwargs):
                    seen.append(messages[0]["content"])
                    return {"input_ids": torch.ones((1, 2), dtype=torch.long),
                            "pixel_values": torch.ones((1, 3, 2, 2))}

            class Model:
                device, dtype = "cpu", torch.float32

                def generate(self, **kwargs):
                    seen.append(kwargs["max_new_tokens"])
                    return torch.tensor([[1, 1, 2, 3, 4]])

            model = benchmark.fw.model_spec("google/gemma-3-12b-it", 1, key="fixture",
                                           display_name="fixture", base_model="fixture")
            for prompt_id in ("p6", "p7", "p9", "p10b"):
                seen.clear()
                config = replace(benchmark.fw.run_config(1),
                                 benchmark_version=iat_prompts.benchmark_version("base", prompt_id))
                runner = native_benchmark.NativeRunner(
                    Model(), Processor(), "gemma3_12b", time.monotonic() + 100,
                    "fixture_" + prompt_id, config, [model], [stimulus], output_root=root,
                    prompt_id=prompt_id)
                record = runner._run_task(model, stimulus, 2, 1)
                text = iat_prompts.render(prompt_id, "fixture", "Black", deterministic_seed(config.base_seed, 2, 1))
                self.assertEqual(seen[0][1]["text"], text)
                call = json.loads(runner.api_path.read_text().splitlines()[0])
                row = json.loads(runner.records_path.read_text().splitlines()[0])
                for value in (call, row):
                    self.assertEqual(value["prompt_sha256"], hashlib.sha256(text.encode()).hexdigest())
                    self.assertEqual(value["prompt_template_sha256"], iat_prompts.prompt_spec(prompt_id)["prompt_sha256"])
                self.assertEqual(seen[1], 120 if prompt_id == "p7" else 30)
                if prompt_id == "p9":
                    self.assertNotEqual(seen[0][0]["path"], str(image))
                    with Image.open(seen[0][0]["path"]) as gray:
                        self.assertEqual((gray.size, gray.getpixel((0, 0))), ((4, 3), (128, 128, 128)))
                else:
                    self.assertEqual(seen[0][0]["path"], str(image))
                if prompt_id == "p7":
                    self.assertEqual(record["sampled_raw"], "Happy, Love, Friend")
                    self.assertTrue(call["generated_text"].startswith("Description:"))


if __name__ == "__main__":
    unittest.main()
