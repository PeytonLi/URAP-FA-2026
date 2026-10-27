import http.server
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runpod_native

REPO = Path(__file__).resolve().parents[2]


class FakeRunpod(http.server.BaseHTTPRequestHandler):
    calls: list = []  # noqa: RUF012 - shared test recorder

    def _reply(self):
        FakeRunpod.calls.append((self.command, self.path, self.headers.get("Authorization")))
        self.send_response(204 if self.command == "DELETE" else 200)
        self.end_headers()

    do_GET = do_DELETE = _reply

    def log_message(self, *args):
        pass


class WatchdogScript(unittest.TestCase):
    def setUp(self):
        FakeRunpod.calls = []
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeRunpod)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "environ").write_bytes(b"RUNPOD_POD_ID=pod-1\0RUNPOD_API_KEY=k\0")
        (self.tmp / "watchdog.sh").write_text(runpod_native.WATCHDOG_SCRIPT)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("RUNPOD_")}
        self.env.update(URAP_ENV_FILE=str(self.tmp / "environ"), URAP_WATCHDOG_POLL="1",
                        URAP_API=f"http://127.0.0.1:{self.server.server_port}")
        self.procs = []

    def tearDown(self):
        for proc in self.procs:
            proc.kill()
            proc.wait()
        self.server.shutdown()
        self.server.server_close()

    def start(self, expiry, stale, keepalive):
        proc = subprocess.Popen(["bash", str(self.tmp / "watchdog.sh"), str(expiry), str(stale),
                                 str(keepalive)], env=self.env)
        self.procs.append(proc)
        (self.tmp / "pid").write_text(str(proc.pid))
        return proc

    def deletes(self, wait=6):
        end = time.time() + wait
        while time.time() < end:
            hits = [c for c in FakeRunpod.calls if c[0] == "DELETE"]
            if hits:
                return hits
            time.sleep(0.2)
        return []

    def test_absolute_expiry_deletes_even_with_fresh_keepalive(self):
        keepalive = self.tmp / "keepalive"
        keepalive.touch()
        self.start(2, 3600, keepalive)
        self.assertEqual(self.deletes()[0][1:], ("/pods/pod-1", "Bearer k"))

    def test_stale_keepalive_deletes_before_expiry(self):
        keepalive = self.tmp / "keepalive"
        keepalive.touch()
        os.utime(keepalive, (time.time() - 120, time.time() - 120))
        self.start(3600, 60, keepalive)
        self.assertTrue(self.deletes())

    def test_fresh_keepalive_before_expiry_does_not_delete_and_check_reports_armed(self):
        keepalive = self.tmp / "keepalive"
        keepalive.touch()
        self.start(3600, 3600, keepalive)
        time.sleep(2.5)
        self.assertEqual(self.deletes(wait=0.1), [])
        out = subprocess.run(["bash", "-s", str(self.tmp)], input=runpod_native.WATCHDOG_CHECK.encode(),
                             capture_output=True, env=self.env, check=True)
        state = json.loads(out.stdout)
        self.assertTrue(state["alive"])
        self.assertEqual((state["pod_id"], state["api_status"]), ("pod-1", 200))
        self.assertTrue(3590 <= state["remaining"] <= 3600)

    def test_check_reports_dead_watchdog(self):
        (self.tmp / "pid").write_text("999999")
        out = subprocess.run(["bash", "-s", str(self.tmp)], input=runpod_native.WATCHDOG_CHECK.encode(),
                             capture_output=True, env=self.env, check=True)
        self.assertFalse(json.loads(out.stdout)["alive"])


def scripted(responses):
    seen = []

    def fake(method, path):
        seen.append((method, path))
        value = responses[(method, path)]
        value = value.pop(0) if isinstance(value, list) and len(value) > 1 else (
            value[0] if isinstance(value, list) else value)
        if isinstance(value, Exception):
            raise value
        return value
    return fake, seen


class DeletionVerification(unittest.TestCase):
    def check(self, get, listed=(200, [])):
        fake, seen = scripted({("DELETE", "/pods/p"): (204, {}), ("GET", "/pods/p"): get,
                               ("GET", "/pods"): listed})
        with patch.object(runpod_native, "http_status", fake):
            return runpod_native.delete_and_verify("p", attempts=3, sleep=lambda s: None), seen

    def test_only_404_or_terminated_and_unlisted_count_as_deleted(self):
        result, _ = self.check((404, None))
        self.assertTrue(result["verified"])
        self.assertEqual(result["delete_statuses"], [204])
        self.assertTrue(self.check((200, {"desiredStatus": "TERMINATED"}))[0]["verified"])

    def test_errors_auth_failures_and_present_states_are_unverified(self):
        for get in ((200, {"desiredStatus": "EXITED"}), (200, {"desiredStatus": "RUNNING"}),
                    (401, None), (403, None), (500, None), urllib.error.URLError("down"),
                    TimeoutError()):
            with self.subTest(get=get):
                result, _ = self.check(get)
                self.assertFalse(result["verified"])
                self.assertEqual(len(result["delete_statuses"]), 3)
        listed = self.check((200, {"desiredStatus": "TERMINATED"}), listed=(200, [{"id": "p"}]))[0]
        self.assertFalse(listed["verified"])
        self.assertFalse(self.check((200, {"desiredStatus": "TERMINATED"}),
                                    listed=(500, None))[0]["verified"])


class Reaper(unittest.TestCase):
    def dead_pid(self):
        proc = subprocess.Popen(["true"])
        proc.wait()
        return proc.pid

    def test_deletes_owned_pod_after_launcher_dies_without_verified_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "launch.json"
            receipt.write_text(json.dumps({"status": "waiting_for_ssh"}))
            fake, seen = scripted({("GET", "/pods"): (200, [{"id": "p", "name": "mine"},
                                                            {"id": "q", "name": "other"}]),
                                   ("DELETE", "/pods/p"): (204, {}), ("GET", "/pods/p"): (404, None)})
            with patch.object(runpod_native, "http_status", fake):
                outcome = runpod_native.reap("mine", self.dead_pid(), receipt, time.time() + 99,
                                             sleep=lambda s: None)
            self.assertTrue(outcome["pods"]["p"]["verified"])
            self.assertNotIn(("DELETE", "/pods/q"), seen)
            self.assertTrue((Path(tmp) / "reaper.json").exists())

    def test_noop_when_launcher_verified_termination(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "launch.json"
            receipt.write_text(json.dumps({"termination_verified": True}))
            with patch.object(runpod_native, "http_status", side_effect=AssertionError):
                self.assertEqual(runpod_native.reap("mine", self.dead_pid(), receipt, 0),
                                 {"action": "none_needed"})

    def test_deadline_triggers_cleanup_while_launcher_hangs(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "launch.json"
            fake, _ = scripted({("GET", "/pods"): (200, [{"id": "p", "name": "mine"}]),
                                ("DELETE", "/pods/p"): (204, {}), ("GET", "/pods/p"): (404, None)})
            with patch.object(runpod_native, "http_status", fake):
                outcome = runpod_native.reap("mine", os.getpid(), receipt, time.time() - 1,
                                             sleep=lambda s: None)
            self.assertTrue(outcome["pods"]["p"]["verified"])


class Budget(unittest.TestCase):
    def test_storage_and_cleanup_are_inside_the_ceiling(self):
        sweep = runpod_native.PROMPT_SWEEP_HOURS
        self.assertGreater(runpod_native.hourly_cost(0.59), 0.59)
        self.assertEqual(runpod_native.paid_seconds(sweep), sweep * 3600 + 40 * 60)
        self.assertTrue(runpod_native.affordable(0.59, 4.0, sweep))
        self.assertLessEqual(runpod_native.hourly_cost(0.59) * runpod_native.paid_seconds(sweep) / 3600, 2.3)
        self.assertFalse(runpod_native.affordable(1.10, 4.0, sweep))
        self.assertFalse(runpod_native.affordable(float("nan"), 4.0, sweep))
        self.assertFalse(runpod_native.affordable(0.59, 4.0, runpod_native.MIN_RUN_HOURS))


class SemanticGate(unittest.TestCase):
    def setUp(self):
        path = REPO / "2026-10-20/analysis/prompt_sweep.py"
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("prompt_sweep", path)
        self.sweep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.sweep)

    def layout(self, root, conditions):
        for prompt, method in conditions:
            (root / "results/native_prompt_sweep" / ("prompt_" + prompt) /
             ("gemma3_12b_" + method + "_native_balanced_prompt_" + prompt)).mkdir(parents=True)

    def test_missing_extra_or_out_of_scope_conditions_fail(self):
        full = [(p, m) for p in ("3", "4", "5") for m in ("base", "orpo")]
        for conditions, extra in ((full[:5], None), (full + [("1", "base")], None),
                                  (full, "native_benchmarks"), (full, "extra_benchmarks")):
            with self.subTest(n=len(conditions), extra=extra), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.layout(root, conditions)
                if extra:
                    (root / "results" / extra).mkdir()
                with self.assertRaises(Exception) as caught:
                    self.sweep.verify(root)
                self.assertIn("ix expected" if not extra else "outside", str(caught.exception))

    def test_complete_layout_still_audits_every_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.layout(root, [(p, m) for p in ("3", "4", "5") for m in ("base", "orpo")])
            with self.assertRaises(FileNotFoundError):
                self.sweep.verify(root)


if __name__ == "__main__":
    unittest.main()
