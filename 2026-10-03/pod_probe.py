"""Read-only live view of the running pod worker (phase + log tail)."""
import json, shlex, subprocess, sys
from pathlib import Path

ATTEMPT = Path(sys.argv[1]) if len(sys.argv) > 1 else None
ROOT = Path(__file__).resolve().parent
attempts = sorted((ROOT / "data/private/prime_native").glob("*_0000"))
folder = ATTEMPT or attempts[-1]
receipt = json.loads((folder / "launch.json").read_text())
remote = receipt["remote_workspace"]

from prime_cli.core import APIClient
from prime_cli.api.pods import PodsClient
pod = PodsClient(APIClient()).get(receipt["pod_id"])
conn = pod.ssh_connection
if isinstance(conn, list):
    conn = next((c for c in conn if c), None)
parts = shlex.split(conn)
host = next(p for p in parts if "@" in p)
port = parts[parts.index("-p") + 1] if "-p" in parts else "22"
ssh = ["ssh", "-i", str(folder / "pod_key"), "-o", "BatchMode=yes",
       "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new",
       "-o", "UserKnownHostsFile=" + str(folder / "known_hosts"), "-p", port, host]

def run(cmd, timeout=30):
    return subprocess.run([*ssh, cmd], capture_output=True, text=True, timeout=timeout)

print("== pod status:", pod.status)
r = run("cat " + remote + "/2026-10-03/data/private/remote_active.json 2>/dev/null || echo '{}'")
print("== active:", r.stdout.strip() or r.stderr.strip())
r = run("cat " + remote + "/2026-10-03/data/private/remote_summary.json 2>/dev/null || echo '{}'")
try:
    s = json.loads(r.stdout)
    print("== summary: status=%s gpu_ok=%s" % (s.get("status"), s.get("gpu_check_passed")))
    for c in s.get("conditions", []):
        print("   ", c)
except Exception:
    print("== summary raw:", r.stdout[:400], r.stderr[:200])
r = run("tail -n 40 " + remote + "/2026-10-03/data/private/remote_worker.log 2>/dev/null || true", timeout=30)
print("== worker.log tail ==")
print(r.stdout[-3500:])
