import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prime_preflight


class PrimePreflightTests(unittest.TestCase):
    def test_queries_cannot_launch_or_upload(self):
        for command in prime_preflight.QUERIES.values():
            self.assertIn(command[:2], (("train", "models"), ("train", "configs"),
                                        ("availability", "list")))
        with self.assertRaises(ValueError):
            prime_preflight.read_catalog(Path("prime.exe"), "train_start")

    def test_explicit_executable_utf8_and_no_shell(self):
        observed = {}

        def run(command, **kwargs):
            observed.update(command=command, **kwargs)
            return SimpleNamespace(returncode=0, stdout=json.dumps({"models": []}), stderr="")

        result = prime_preflight.read_catalog(Path("working-prime.exe"), "shared_models", runner=run)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(observed["command"][:4], ["rtk", "proxy", "working-prime.exe", "--plain"])
        self.assertEqual(observed["env"]["PYTHONUTF8"], "1")
        self.assertNotIn("shell", observed)

    def test_errors_do_not_leak_raw_responses(self):
        def run(*args, **kwargs):
            return SimpleNamespace(returncode=1, stdout="HTTP 403 account-private-details",
                                   stderr="potential-secret")

        result = prime_preflight.read_catalog(Path("prime.exe"), "dedicated_cached_models", runner=run)
        self.assertEqual(result["http_status_candidates"], ["403"])
        self.assertNotIn("potential-secret", json.dumps(result))
        self.assertNotIn("account-private-details", json.dumps(result))

    def test_timeout_is_unknown_not_empty_success(self):
        def run(*args, **kwargs):
            raise subprocess.TimeoutExpired("read-only", 60)

        result = prime_preflight.read_catalog(Path("prime.exe"), "shared_models", runner=run)
        self.assertEqual(result["status"], "timeout")

    def test_catalog_membership_never_proves_modality_or_orpo(self):
        summary = prime_preflight.summarize({
            "shared_models": {"status": "ok", "payload": {"models": [{"name": "Qwen/Qwen3.5-9B"}]}},
            "dedicated_cached_models": {"status": "request_failed"},
        })
        qwen = summary["models"][2]
        self.assertTrue(qwen["listed_in_shared_catalog"])
        self.assertIsNone(qwen["listed_in_dedicated_cache_catalog"])
        self.assertFalse(qwen["image_sft_verified"])
        self.assertFalse(qwen["image_orpo_verified"])


if __name__ == "__main__":
    unittest.main()
