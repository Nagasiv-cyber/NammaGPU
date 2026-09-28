"""
Run this on the ASPIRE LITE to test the ALG host end-to-end.
Needs no extra installs — only Python's built-in libraries.

Usage:  python test_from_buyer.py 192.168.x.x
"""

import json
import sys
import time
import urllib.error
import urllib.request

HOST_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.10"
BASE = f"http://{HOST_IP}:8000"


def call(path, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


print("Host:", call("/"))

try:
    job = call("/run-job", "POST", {"job_type": "mnist"})
except urllib.error.HTTPError as e:
    print("Could not start job:", e.code, e.read().decode())
    sys.exit(1)

job_id = job["job_id"]
print("Started job", job_id)

while True:
    info = call(f"/job/{job_id}")
    last = info["samples"][-1] if info["samples"] else {}
    print(f"  status={info['status']:8}  gpu={last.get('gpu_util', '-')}%  "
          f"vram={last.get('mem_used_mb', '-')}MB")
    if info["status"] != "running":
        break
    time.sleep(2)

print("\nRESULT")
for key in ["status", "runtime_s", "busy_seconds", "peak_util", "peak_mem_mb"]:
    print(f"  {key}: {info.get(key)}")
print("  log:", *info["log_tail"], sep="\n    ")
