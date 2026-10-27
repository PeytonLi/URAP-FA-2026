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
import prime_native
import runpod_native
import study
from multimodel_bias.iat import Stimulus, deterministic_seed, deterministic_task_id


class PromptSweepTests(unittest.TestCase):
    def test_exact_preselected_scope_and_no_training(self):
        plan = iat_prompts.sweep_plan({("gemma3_12b", "orpo"): object()})
        self.assertEqual(plan["mode"], "prompt_sweep")
        self.assertFalse(plan["training_allowed"])
        self.assertFalse(plan["extra_benchmarks"])
        self.assertEqual(plan["methods"], ["base", "orpo"])
        self.assertEqual([p["prompt_id"] for p in plan["prompts"]], ["3", "4", "5"])
        self.assertEqual(plan["expected_sampled_attempts"], 3 * 2 * 12 * 50)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            iat_prompts.sweep_plan({})

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


if __name__ == "__main__":
    unittest.main()
