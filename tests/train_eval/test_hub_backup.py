"""No Hub writes or paid compute: mocked backup, privacy and trainer checks."""
import contextlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hub_backup as hub
import native_train as native
import prime_native
import study


class HubBackupTests(unittest.TestCase):
    def fixture(self, folder):
        run = Path(folder) / "gemma4_sft_20261006T120000.123456_0000"
        checkpoint = run / "epoch_3"
        checkpoint.mkdir(parents=True)
        settings = native.MODELS["gemma4"]
        study.save_json(checkpoint / "adapter_config.json", {
            "base_model_name_or_path": settings["model"], "revision": settings["revision"]})
        (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture, not study weights")
        study.save_json(checkpoint / "tokenizer_config.json", {})
        (checkpoint / "chat_template.jinja").write_text("fixture", encoding="utf-8")
        for name in (".env", "HF_TOKEN", "face.png", "targets.jsonl", "optimizer.pt", "README.md"):
            (checkpoint / name).write_bytes(b"must not be uploaded")
        record = {"path": str(checkpoint), "files": {p.name: study.sha(p) for p in checkpoint.iterdir()}}
        study.save_json(run / "run.json", {
            "model": settings["model"], "weight_and_processor_revision": settings["revision"],
            "method": "sft", "status": "training_complete_evaluation_pending", "requested_epochs": 3,
            "completed_epochs": 3, "optimizer_updates": 507, "planned_optimizer_updates": 507,
            "checkpoints": [record], "credential": "never upload", "targets_are_ground_truth": False})
        api = Mock(token="fixture-secret", endpoint="https://huggingface.co")
        api.model_info.return_value = SimpleNamespace(private=True)
        return run, checkpoint, api

    def test_auth_requires_write_and_never_exposes_provider_error_or_token(self):
        import huggingface_hub
        api = Mock()
        with patch.object(huggingface_hub, "get_token", return_value=None), \
                patch.object(huggingface_hub, "HfApi", return_value=api):
            with self.assertRaisesRegex(ValueError, "local write token"):
                hub.connect()
            api.whoami.assert_not_called()
        with patch.object(huggingface_hub, "get_token", return_value="fixture-secret"), \
                patch.object(huggingface_hub, "HfApi", return_value=api):
            for role in ("write", "fineGrained", "read"):
                api.whoami.return_value = {"type": "user", "name": "peyton", "auth": {"accessToken": {"role": role}}}
                if role == "read":
                    with self.assertRaisesRegex(ValueError, "read-only"):
                        hub.connect()
                else:
                    self.assertEqual(hub.connect(), (api, "peyton"))
            api.whoami.return_value = {"type": "user", "name": "peyton", "auth": {"type": "oauth"}}
            self.assertEqual(hub.connect(), (api, "peyton"))
            api.whoami.return_value = {"type": "user", "name": "peyton", "auth": {"type": "unknown"}}
            with self.assertRaisesRegex(ValueError, "unknown credential"):
                hub.connect()
            api.whoami.side_effect = RuntimeError("fixture-secret")
            with self.assertRaises(ValueError) as error:
                hub.connect()
            self.assertNotIn("fixture-secret", str(error.exception))

    def test_private_repo_per_run_collision_refused_and_destinations_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            run, _, api = self.fixture(folder)
            state = hub.create(run, api, "peyton")
            api.create_repo.assert_called_once_with(repo_id=state["repo_id"], repo_type="model", private=True, exist_ok=False)
            api.auth_check.assert_called_once_with(state["repo_id"], repo_type="model", write=True)
            self.assertNotEqual(hub.repo_id(run, "peyton"), hub.repo_id(run.with_name(run.name.replace("sft", "orpo")), "peyton"))
            with self.assertRaisesRegex(ValueError, "must not be overwritten"):
                hub.create(run, api, "peyton")
        with tempfile.TemporaryDirectory() as folder:
            run, _, api = self.fixture(folder)
            api.auth_check.side_effect = PermissionError("fixture-secret")
            with self.assertRaisesRegex(ValueError, "creation failed") as error:
                hub.create(run, api, "peyton")
            self.assertNotIn("fixture-secret", str(error.exception))
            self.assertFalse((run / "hub_backup.json").exists())

    def test_upload_allowlist_pinned_revision_hash_verification_and_immutable_commit(self):
        import huggingface_hub
        with tempfile.TemporaryDirectory() as folder:
            run, checkpoint, api = self.fixture(folder)
            hub.create(run, api, "peyton")
            uploaded = {}
            def upload(**options):
                self.assertFalse(options["run_as_future"])
                staging = options["folder_path"]
                uploaded.update({p.name: p.read_bytes() for p in staging.iterdir()})
                self.assertNotIn("credential", json.loads(uploaded["training_summary.json"]))
                summary = json.loads(uploaded["training_summary.json"])
                self.assertEqual(summary["checkpoint_optimizer_updates"], 507)
                self.assertEqual(summary["checkpoint_completed_epochs"], 3)
                self.assertNotIn("HF_TOKEN", summary["source_checkpoint_files"])
                self.assertEqual(set(uploaded), {"adapter_config.json", "adapter_model.safetensors",
                    "tokenizer_config.json", "chat_template.jinja", "README.md", "training_summary.json", "backup_manifest.json"})
                api.model_info.return_value = SimpleNamespace(private=True, sha="a" * 40, siblings=[
                    SimpleNamespace(rfilename=name, lfs=SimpleNamespace(sha256=study.sha(staging / name))
                                    if name == "adapter_model.safetensors" else None)
                    for name in uploaded])
                return SimpleNamespace(oid="a" * 40)
            def download(repo, name, **options):
                self.assertEqual(options["revision"], "a" * 40)
                cached = run.parent / "remote-cache" / name
                cached.parent.mkdir(exist_ok=True)
                cached.write_bytes(uploaded[name])
                return cached
            api.upload_folder.side_effect = upload
            with patch.object(hub, "connect", return_value=(api, "peyton")), \
                    patch.object(huggingface_hub, "hf_hub_download", side_effect=download) as fetched:
                result = hub.publish(run, checkpoint)
            self.assertEqual(fetched.call_count, 6)
            self.assertTrue(result["training_complete"])
            self.assertEqual(result["revision"], "a" * 40)
            self.assertEqual(json.loads((run / "hub_backup.json").read_text())["status"], "verified")
            # Corrupted remote hashes must not count as a successful backup.
            api.upload_folder.side_effect = None
            api.upload_folder.return_value = SimpleNamespace(oid="a" * 40)
            next(s for s in api.model_info.return_value.siblings
                 if s.rfilename == "adapter_model.safetensors").lfs.sha256 = "0" * 64
            with patch.object(hub, "connect", return_value=(api, "peyton")), \
                    patch.object(huggingface_hub, "hf_hub_download", side_effect=download):
                with self.assertRaisesRegex(RuntimeError, "local checkpoint retained"):
                    hub.publish(run, checkpoint)
            self.assertTrue((checkpoint / "adapter_model.safetensors").is_file())
            self.assertEqual(json.loads((run / "hub_backup.json").read_text())["status"], "failed")

    def test_wrong_base_revision_and_wrong_account_refuse_before_upload(self):
        with tempfile.TemporaryDirectory() as folder:
            run, checkpoint, api = self.fixture(folder)
            hub.create(run, api, "peyton")
            with patch.object(hub, "connect", return_value=(api, "different-account")):
                with self.assertRaisesRegex(ValueError, "does not belong"):
                    hub.publish(run, checkpoint)
            api.upload_folder.assert_not_called()
            config = checkpoint / "adapter_config.json"
            data = json.loads(config.read_text())
            study.save_json(config, {**data, "revision": "wrong"})
            with self.assertRaisesRegex(ValueError, "pinned revision mismatch"):
                hub.publish(run, checkpoint)
            api.upload_folder.assert_not_called()

    def test_public_repo_and_altered_local_checkpoint_refuse_upload(self):
        with tempfile.TemporaryDirectory() as folder:
            run, checkpoint, api = self.fixture(folder)
            hub.create(run, api, "peyton")
            api.model_info.return_value = SimpleNamespace(private=False)
            with patch.object(hub, "connect", return_value=(api, "peyton")):
                with self.assertRaisesRegex(RuntimeError, "local checkpoint retained"):
                    hub.publish(run, checkpoint)
            api.upload_folder.assert_not_called()
            (checkpoint / "adapter_model.safetensors").write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                hub.publish(run, checkpoint)

    def test_secure_credential_transfer_uses_stdin_not_arguments_or_logged_errors(self):
        with patch.object(prime_native, "command") as send:
            path = prime_native.transfer_hub_token(["ssh", "owned-host"], "/tmp/owned", "fixture-secret")
            self.assertEqual(path, "/tmp/owned/.hf-write-token")
            self.assertNotIn("fixture-secret", str(send.call_args.args))
            self.assertEqual(send.call_args.kwargs["input"], "fixture-secret")
            self.assertIn("0o600", str(send.call_args.args))
            send.side_effect = subprocess.CalledProcessError(1, "ssh", stderr="fixture-secret")
            with self.assertRaises(RuntimeError) as error:
                prime_native.transfer_hub_token(["ssh"], "/tmp/owned", "fixture-secret")
            self.assertNotIn("fixture-secret", str(error.exception))

    def test_native_trainer_saves_local_before_each_mandatory_upload_and_preserves_failure(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        torch.set_num_threads(2)
        for failure in (False, True):
            with self.subTest(upload_failure=failure), tempfile.TemporaryDirectory() as folder, contextlib.ExitStack() as stack:
                root = Path(folder)
                model = torch.nn.Linear(1, 1)
                row = {"split": "train", "image_id": "fixture_train"}
                stack.enter_context(patch.object(study, "ROOT", root))
                stack.enter_context(patch.object(native, "approved_cohort", return_value=(
                    {"dataset_id": "fixture", "manifest_sha256": "fixture", "targets_path": __file__},
                    [row | {"reviewer_kind": "fixture"}],
                    [{"image_id": "fixture_dev", "reviewer_kind": "fixture"}], root)))
                for name, value in (("is_available", True), ("is_bf16_supported", True), ("device_count", 1),
                                    ("get_device_name", "fixture-no-GPU"), ("max_memory_allocated", 0)):
                    stack.enter_context(patch.object(torch.cuda, name, return_value=value))
                stack.enter_context(patch.object(AutoProcessor, "from_pretrained"))
                stack.enter_context(patch.object(AutoModelForImageTextToText, "from_pretrained", return_value=model))
                stack.enter_context(patch.object(native, "attach_lora", return_value=(model, {})))
                stack.enter_context(patch.object(native, "image_control", return_value={"fixture": True}))
                stack.enter_context(patch.object(native, "encode_row", return_value=("chosen", "rejected")))
                stack.enter_context(patch.object(native, "batch_update", return_value={"fixture": True}))
                stack.enter_context(patch.object(native, "response_logprobs", side_effect=lambda m, branch:
                    torch.tensor([-1.0 if branch == "chosen" else -2.0])))
                stack.enter_context(patch.object(hub, "connect", return_value=(Mock(), "peyton")))
                stack.enter_context(patch.object(hub, "create", return_value={"repo_id": "peyton/fixture"}))
                stack.enter_context(patch.object(native, "save_checkpoint", side_effect=lambda m, p, o, path:
                    {"path": str(path), "files": {}}))
                def publish(run, checkpoint):
                    receipt = json.loads((run / "run.json").read_text())
                    self.assertEqual(Path(receipt["checkpoints"][-1]["path"]).name, checkpoint.name)
                    self.assertEqual(receipt["completed_epochs"], len(receipt["checkpoints"]))
                    if failure:
                        raise RuntimeError("fixture upload failure")
                uploads = stack.enter_context(patch.object(hub, "publish", side_effect=publish))
                if failure:
                    with self.assertRaisesRegex(RuntimeError, "fixture upload failure"):
                        native.train("gemma3_12b", "orpo", 60)
                else:
                    native.train("gemma3_12b", "orpo", 60)
                receipt = json.loads(next((root / "data/private/native_runs").glob("*/run.json")).read_text())
                self.assertEqual(uploads.call_count, 1 if failure else 3)
                self.assertEqual(receipt["status"], "failed" if failure else "training_complete_evaluation_pending")
                self.assertTrue(receipt["hub_backup_required"])
                self.assertEqual(len(receipt["checkpoints"]), 1 if failure else 3)


if __name__ == "__main__":
    unittest.main()
