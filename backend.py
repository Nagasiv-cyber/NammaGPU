"""
GPUSetu Marketplace Backend  —  runs on the Aspire Lite

Start it with:
    python -m uvicorn backend:app --host 0.0.0.0 --port 9000

Then open the dashboard in a browser:  http://127.0.0.1:9000
"""

import json
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------------------------
# Every GPU host the marketplace knows about. Add more hosts here later.
HOSTS = [
    {"host_id": "acer-alg-rtx3050-6gb", "url": "http://192.168.137.1:8000"},
]

RATE_PER_SEC = 0.001      # price in MSTC per VERIFIED GPU-second
BUSY_THRESHOLD = 20       # same rule as the host: >= 20% utilization = real work
TIMEOUT = 3               # seconds to wait for a host before calling it offline

BASE_DIR = Path(__file__).parent
jobs = {}                 # the marketplace's job book


# ---------------------------------------------------------------------------
# 2. TALKING TO HOSTS
# ---------------------------------------------------------------------------
def host_call(host, path, method="GET", body=None):
    """Send one request to a host daemon and return its JSON answer."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        host["url"] + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def host_status(host):
    """Ask a host 'are you alive and free?'"""
    try:
        info = host_call(host, "/")
        return {**host, "gpu": info.get("gpu"), "status": info.get("status")}
    except Exception:
        return {**host, "gpu": None, "status": "offline"}


def find_host(host_id):
    for h in HOSTS:
        if h["host_id"] == host_id:
            return h
    return None


# ---------------------------------------------------------------------------
# 3. BILLING — the heart of GPUSetu
# ---------------------------------------------------------------------------
def make_bill(host_job):
    """Charge only for seconds where the GPU was measurably working."""
    samples = host_job.get("samples", [])
    verified = sum(1 for s in samples if s["gpu_util"] >= BUSY_THRESHOLD)
    if host_job.get("status") != "running" and "busy_seconds" in host_job:
        verified = host_job["busy_seconds"]          # final number from the host

    if host_job.get("status") == "running":
        # Use the HOST's clock for both times. The two laptops' clocks can differ.
        wall = (samples[-1]["time"] - host_job["started_at"]) if samples else 0
    else:
        wall = host_job.get("runtime_s", 0)

    # Safety rule: you can never be billed for more seconds than the job actually ran.
    verified = min(verified, int(wall))

    return {
        "rate_per_sec": RATE_PER_SEC,
        "wall_seconds": round(wall, 1),
        "verified_seconds": verified,
        "billed_amount": round(verified * RATE_PER_SEC, 6),
        "time_based_amount": round(wall * RATE_PER_SEC, 6),   # what hourly-style billing would charge
    }


# ---------------------------------------------------------------------------
# 4. WEB SERVER
# ---------------------------------------------------------------------------
app = FastAPI(title="GPUSetu Marketplace")


class JobRequest(BaseModel):
    job_type: str = "mnist"


@app.get("/")
def dashboard():
    return FileResponse(BASE_DIR / "static" / "dashboard.html")


@app.get("/api/hosts")
def list_hosts():
    return [host_status(h) for h in HOSTS]


@app.get("/api/live")
def live():
    """Live GPU reading from the first reachable host (for the dashboard graph)."""
    for h in HOSTS:
        try:
            reading = host_call(h, "/stats")
            reading["host_id"] = h["host_id"]
            return reading
        except Exception:
            continue
    raise HTTPException(503, "No host reachable. Check the hotspot and that the host daemon is running.")


@app.post("/api/jobs")
def submit_job(req: JobRequest):
    """MATCHING: give the job to the first idle host."""
    for h in HOSTS:
        if host_status(h)["status"] != "idle":
            continue
        try:
            started = host_call(h, "/run-job", "POST", {"job_type": req.job_type})
        except urllib.error.HTTPError:
            continue                      # host became busy a moment ago -> try the next one
        except Exception:
            continue

        job_id = uuid.uuid4().hex[:8]
        jobs[job_id] = {
            "job_id": job_id,
            "job_type": req.job_type,
            "host_id": h["host_id"],
            "host_job_id": started["job_id"],
            "source": "peer",             # Phase 5 adds "cloud" for the fallback
            "submitted_at": round(time.time(), 2),
        }
        return jobs[job_id]

    raise HTTPException(503, "Every host is busy or offline. The cloud fallback arrives in Phase 5.")


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "No such job")
    host = find_host(job["host_id"])
    try:
        host_job = host_call(host, f"/job/{job['host_job_id']}")
    except Exception:
        raise HTTPException(502, "Lost contact with the host running this job.")

    bill = make_bill(host_job)
    job["status"] = host_job["status"]      # remember the latest state for the history list
    job["bill"] = bill

    return {
        **job,
        "status": host_job["status"],
        "samples": host_job.get("samples", []),
        "peak_util": host_job.get("peak_util"),
        "peak_mem_mb": host_job.get("peak_mem_mb"),
        "log_tail": host_job.get("log_tail", []),
        "bill": bill,
    }


@app.get("/api/jobs")
def job_history():
    return sorted(jobs.values(), key=lambda j: j["submitted_at"], reverse=True)
