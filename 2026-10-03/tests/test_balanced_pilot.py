"""Check dataset-level balancing without loading pretrained weights or GPUs."""
import sys
import subprocess
import tarfile
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import native_train
import objectives
import pilot
import prime_native


class BalancedPilotTests(unittest.TestCase):
    def test_resume_source_refuses_unowned_paths_before_cloud_imports_or_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "owned attempt"):
                prime_native.launch(resume_from=Path(folder) / "recovered")

    def test_gpu_quote_refuses_variable_spot_prepaid_or_unknown_costs(self):
        offer = {"gpuCount": 1, "gpuMemory": 80, "stockStatus": "Available", "isSpot": False,
                 "prepaidTime": None, "prices": {"onDemand": 4.29, "currency": "USD", "isVariable": False},
                 "disk": {}, "vcpu": {}, "memory": {}}
        self.assertEqual(prime_native.safe_hourly_quote(offer), 4.29)
        disk = {"defaultCount": 1500, "pricePerUnit": 0.0001, "defaultIncludedInPrice": False}
        self.assertAlmostEqual(prime_native.safe_hourly_quote({**offer, "disk": disk}), 4.44)
        for changes in ({"prepaidTime": 1}, {"isSpot": True}, {"gpuMemory": 40},
                        {"prices": {"onDemand": 4.29, "currency": "USD", "isVariable": True}},
                        {"prices": {"onDemand": float("nan"), "currency": "USD"}},
                        {"disk": {"pricePerUnit": 0.1, "defaultCount": None}},
                        {"disk": {"pricePerUnit": 0.1, "defaultCount": float("inf")}}):
            with self.assertRaises(ValueError):
                prime_native.safe_hourly_quote({**offer, **changes})

    def test_shutdown_requires_exact_owned_absence_and_terminated_history(self):
        pods = Mock()
        pods.list.return_value.data = [SimpleNamespace(id="owned")]
        self.assertFalse(prime_native.terminated(pods, "owned"))
        pods.history.assert_not_called()
        pods.list.return_value.data = []
        pods.history.return_value.data = [SimpleNamespace(id="other", terminated_at="now")]
        self.assertFalse(prime_native.terminated(pods, "owned"))
        pods.history.return_value.data = [SimpleNamespace(id="owned", terminated_at=None)]
        self.assertFalse(prime_native.terminated(pods, "owned"))
        pods.history.return_value.data = [SimpleNamespace(id="owned", terminated_at="now")]
        self.assertTrue(prime_native.terminated(pods, "owned"))

    def test_provider_retry_handles_transient_control_plane_failure(self):
        calls = [0]
        def operation():
            calls[0] += 1
            if calls[0] < 3:
                raise RuntimeError("transient DNS")
            return "ok"
        with patch.object(prime_native.time, "sleep") as pause:
            self.assertEqual(prime_native.retry(operation, attempts=3, delay=5), "ok")
        self.assertEqual(calls[0], 3)
        self.assertEqual(pause.call_count, 2)

    def test_pod_connection_falls_back_to_provider_status_endpoint(self):
        pods = Mock()
        current = SimpleNamespace(ssh_connection=None)
        pods.get_status.return_value = [SimpleNamespace(
            pod_id="owned", ssh_connection="ubuntu@203.0.113.10 -p 22")]
        self.assertEqual(prime_native.pod_ssh_connection(pods, "owned", current),
                         "ubuntu@203.0.113.10 -p 22")
        pods.get_status.assert_called_once_with(["owned"])

    def test_remote_setup_failure_still_packages_a_failed_summary(self):
        import io
        import json
        import remote_native as remote
        with tempfile.TemporaryDirectory() as folder, patch.object(remote, "ROOT", Path(folder)), \
                patch.object(remote.sys, "stderr", io.StringIO()) as errors:
            with patch.object(remote.subprocess, "run", side_effect=RuntimeError("setup fixture")):
                with self.assertRaisesRegex(RuntimeError, "setup fixture"):
                    remote.main(expected_mode="training")
            summary = json.loads((Path(folder) / "data/private/remote_summary.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["conditions"], [])
            self.assertFalse(summary["gpu_check_passed"])
            self.assertEqual(summary["error_message"], "setup fixture")
            self.assertIn("setup fixture", errors.getvalue())
            with tarfile.open(Path(folder) / "data/private/remote_outputs.tar.gz") as archive:
                self.assertEqual(archive.getnames(), ["data/private/remote_summary.json"])

    def test_resume_scientific_imports_use_the_pinned_runtime_not_bootstrap_python(self):
        import remote_native as remote
        import study
        with tempfile.TemporaryDirectory() as folder, patch.object(remote, "ROOT", Path(folder)), \
                patch.object(remote, "ACTIVE", Path(folder) / "active.json"):
            output = Mock()
            def launch(command, **options):
                self.assertEqual(command[0], "/owned/runtime/native-venv/bin/python")
                self.assertIn("require_all=False", command[2])
                self.assertEqual(options["cwd"], Path(folder))
                self.assertIs(options["stdout"], output)
                self.assertEqual(options["stderr"], subprocess.STDOUT)
                study.save_json(Path(folder) / "data/private/verified_resume_plan.json",
                                {"training": [["gemma4", "sft"], ["gemma4", "orpo"]],
                                 "benchmarks": [["gemma4", "base"], ["gemma4", "sft"]]})
                return SimpleNamespace(pid=456, wait=Mock(return_value=0))
            with patch.object(remote.subprocess, "Popen", side_effect=launch):
                training, benchmarks = remote.verify_resume("/owned/runtime/native-venv/bin/python", output)
            self.assertEqual(training, {("gemma4", "sft"), ("gemma4", "orpo")})
            self.assertEqual(benchmarks, {("gemma4", "base"), ("gemma4", "sft")})

    def test_remote_liveness_requires_owned_pid_and_includes_an_orphaned_child(self):
        import remote_native as remote
        with patch.object(Path, "read_bytes", return_value=(str(remote.ROOT) + "/remote_native.py").encode()):
            self.assertTrue(remote.owned_pid_running(123))
            self.assertFalse(remote.owned_pid_running(True))
            self.assertFalse(remote.owned_pid_running(0))
        with patch.object(Path, "read_bytes", return_value=b"unrelated process"):
            self.assertFalse(remote.owned_pid_running(123))
        with patch.object(Path, "read_bytes", side_effect=FileNotFoundError):
            self.assertFalse(remote.owned_pid_running(123))
        with tempfile.TemporaryDirectory() as folder, patch.object(remote, "ROOT", Path(folder)), \
                patch.object(remote, "ACTIVE", Path(folder) / "active.json"), \
                patch.object(remote, "owned_pid_running", side_effect=[False, True]) as live:
            import study
            study.save_json(remote.ACTIVE, {"child_pid": 456})
            status = remote.worker_status(123)
            self.assertFalse(status["worker_running"])
            self.assertTrue(status["child_running"])
            self.assertEqual([call.args for call in live.call_args_list], [(123,), (456,)])

    def test_uv_bootstrap_falls_back_when_system_venv_is_unavailable(self):
        import io
        import remote_native as remote
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            calls = []
            def run(command, **kwargs):
                calls.append((command, kwargs))
                if len(calls) == 1:
                    raise subprocess.CalledProcessError(1, command)
                return SimpleNamespace(returncode=0)
            response = Mock()
            response.__enter__ = Mock(return_value=io.BytesIO(b"#!/bin/sh\n"))
            response.__exit__ = Mock(return_value=False)
            with patch.object(remote.subprocess, "run", side_effect=run), \
                    patch.object(remote.urllib.request, "urlopen", return_value=response):
                uv = remote.bootstrap_uv(root)
            self.assertEqual(uv, root / "runtime/uv-bin/uv")
            self.assertEqual(calls[1][0][:2], ["sh", str(root / "runtime/uv-install.sh")])
            self.assertEqual(calls[1][1]["env"]["UV_UNMANAGED_INSTALL"], str(root / "runtime/uv-bin"))

    def test_fixed_weights_give_each_category_equal_total_influence(self):
        weights = pilot.balance_weights({"positive": 639, "negative": 37})
        self.assertAlmostEqual(639 * weights["positive"], 338)
        self.assertAlmostEqual(37 * weights["negative"], 338)
        for invalid in ({"positive": 0, "negative": 37}, {"positive": True, "negative": 1},
                        {"positive": 1}, {"positive": 1.5, "negative": 2}):
            with self.assertRaises(ValueError):
                pilot.balance_weights(invalid)

    def test_sft_and_entire_orpo_pair_receive_fixed_not_batch_normalized_weights(self):
        import torch
        torch.set_num_threads(2)
        for method in ("sft", "orpo"):
            with self.subTest(method=method):
                model = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    model.weight.fill_(0.1)
                optimizer = torch.optim.SGD(model.parameters(), lr=0)
                image = torch.ones(1)
                inputs = [{"offset": n, "pixel_values": image} for n in (-2., -4., -3., -5.)]
                examples = [(inputs[0], inputs[1]), (inputs[2], inputs[3])]
                weights = [0.5, 2.0]  # Sum is deliberately not the batch size.
                def scores(unused_model, value):
                    return (model.weight.flatten() * 0.01 + value["offset"])
                terms = []
                for chosen, rejected in examples:
                    c, r = scores(model, chosen), scores(model, rejected)
                    mask = torch.ones_like(c)
                    term, _ = (objectives.sft_loss([mask], [c]) if method == "sft" else
                               objectives.orpo_loss([mask, mask], [c, r]))
                    terms.append(term)
                expected = sum(w * term for w, term in zip(weights, terms)) / len(examples)
                expected_gradient = torch.autograd.grad(expected, model.weight)[0]
                with patch.object(native_train, "response_logprobs", side_effect=scores):
                    metrics = native_train.batch_update(model, optimizer, examples, method,
                                                        example_weights=weights)
                torch.testing.assert_close(model.weight.grad, expected_gradient)
                self.assertAlmostEqual(metrics["weighted_objective"], float(expected.detach()), places=6)
                self.assertEqual(metrics["sum_example_weights"], 2.5)
                for invalid in ([0, 1], [1], [float("nan"), 1], [True, 1], ["1", 1]):
                    with self.assertRaises(ValueError):
                        native_train.batch_update(model, optimizer, examples, method, example_weights=invalid)

    def test_native_benchmark_reuses_pixels_parser_fallback_and_original_seed_schedule(self):
        import torch
        import native_benchmark as nb
        import study
        from multimodel_bias.iat import Stimulus
        processor = Mock()
        processor.apply_chat_template.return_value = {"input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long), "pixel_values": torch.ones((1, 4))}
        processor.tokenizer.decode.side_effect = ["cannot choose", "Angry, Annoy, Negative"]
        seeds = []
        def generate_ids(**unused):
            seeds.append(torch.initial_seed())
            return torch.tensor([[1, 2, 3]] if len(seeds) == 1 else [[1, 2, 4, 5, 6]])
        generate = Mock(side_effect=generate_ids)
        model = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32, generate=generate)
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "image.jpg"
            image.write_bytes(b"image fixture only; no study results")
            stimulus = Stimulus("fixture", image.name, "Black", "Female", 1, 1, study.sha(image), image)
            spec = replace(nb.benchmark.fw.model_spec("fixture", 1), provider_endpoint="prime/native-transformers")
            runner = nb.NativeRunner(model, processor, "gemma4", time.monotonic() + 60,
                                    "fixture", nb.benchmark.fw.run_config(1), [spec], [stimulus],
                                    output_root=Path(folder) / "outputs")
            record = runner._run_task(spec, stimulus, 1, 1)
            self.assertEqual(record["response_source"], "fallback")
            self.assertEqual(record["negative_count"], 3)
            self.assertEqual(record["seed"], 17083027)
            self.assertEqual(seeds, [17083027, 18083027])
            self.assertEqual(processor.apply_chat_template.call_count, 1)
            self.assertFalse(processor.apply_chat_template.call_args.kwargs["enable_thinking"])
            for call, sample in zip(generate.call_args_list, [True, False]):
                self.assertEqual(call.kwargs["do_sample"], sample)
                self.assertEqual(call.kwargs["max_new_tokens"], 30)
                self.assertEqual(call.kwargs["top_k"], 0)
                torch.testing.assert_close(call.kwargs["pixel_values"], torch.ones((1, 4)))
            raw = nb.benchmark.fw._jsonl_records(runner.api_path)
            self.assertEqual([r["stage"] for r in raw], ["sampled", "fallback"])
            self.assertEqual(raw[0]["response"]["provider"], "prime/native-transformers")
            runner.deadline = 0
            with self.assertRaises(TimeoutError):
                runner._run_task(spec, stimulus, 2, 1)
            self.assertEqual(generate.call_count, 2)

    @unittest.skipUnless(pilot.FROZEN.is_file(), "Private frozen pilot not present")
    def test_input_bundle_replays_only_retained_cohort_and_separate_benchmark_images(self):
        import study
        with tempfile.TemporaryDirectory(prefix="urap-pilot-bundle-") as folder:
            archive = Path(folder) / "inputs.tar.gz"
            recovered = prime_native.ATTEMPTS / "20261006T043801.932866_0000/recovered"
            resume = recovered if (recovered / "data/private/native_runs").is_dir() else None
            receipt = prime_native.package_inputs(archive, resume_from=resume)
            extracted = Path(folder) / "replay"
            with tarfile.open(archive) as bundle:
                bundle.extractall(extracted, filter="data")
            self.assertEqual(len(list((extracted / "2026-09-29/private/dataset/images").rglob("*.png"))), 856)
            self.assertEqual(len(list((extracted / "2026-10-03/data/private/benchmark_images").glob("*.jpg"))), 12)
            # Native tokenizer artifacts are not login credentials.
            self.assertFalse(any(("token" in Path(p).name.lower()
                                  and Path(p).name not in ("tokenizer.json", "tokenizer_config.json"))
                                 or "auth.json" in p or "pod_key" in p
                                 or p.endswith(".env") for p in receipt["files"]))
            subprocess.run(["rtk", "proxy", sys.executable, str(extracted / "2026-10-03/native_train.py"),
                            "check-data", "--cohort", "resolved-balanced"], cwd=extracted, check=True)
            if resume:
                probe = ("import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
                         "import native_report; t,b=native_report.verified_results(Path(sys.argv[1]),require_all=False); "
                         "assert len(t)==2 and len(b)==2")
                subprocess.run(["rtk", "proxy", sys.executable, "-c", probe,
                                str(extracted / "2026-10-03")], cwd=extracted, check=True)
                self.assertEqual((receipt["resumed_training_conditions"], receipt["resumed_benchmark_conditions"]), (2, 2))
            study.save_json(study.ROOT / "results/balanced_pilot_bundle_check.json", {
                "checked_at": study.now(), "training_images": 676, "development_images": 180,
                "benchmark_images": 12, "extracted_cohort_replay_passed": True,
                "reserved_test_images_included": 0, "credentials_included": False,
                "study_optimizer_updates": 0, "paid_resource_created": False,
                "archive_sha256": receipt["archive_sha256"], "file_count": len(receipt["files"])})

    def test_incomplete_adapter_refuses_native_evaluation_before_loading_weights(self):
        import native_benchmark as nb
        import study
        from transformers import AutoModelForImageTextToText, AutoProcessor
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "epoch_3"
            study.save_json(checkpoint.parent / "run.json", {"status": "incomplete_runtime_limit"})
            with patch.object(AutoModelForImageTextToText, "from_pretrained") as weights, \
                    patch.object(AutoProcessor, "from_pretrained") as processor:
                with self.assertRaisesRegex(ValueError, "Incomplete or mismatched"):
                    nb.run("gemma4", "sft", Path(folder), 60, checkpoint)
                weights.assert_not_called()
                processor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
