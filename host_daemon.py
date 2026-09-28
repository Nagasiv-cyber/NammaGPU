"""
GPUSetu Host Daemon  —  runs on the Acer ALG (RTX 3050 6GB)

Start it with:
    uvicorn host_daemon:app --host 0.0.0.0 --port 8000

Then, from the Aspire Lite, open:  http://<ALG-IP>:8000/docs
"""

import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pynvml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------------------------
HOST_ID = "acer-alg-rtx3050-6gb"
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Only these jobs are allowed to run. A buyer picks a NAME, never sends code.
ALLOWED_JOBS = {
    "mnist": BASE_DIR / "jobs" / "demo_job.py",
}

# A second counts as "GPU really working" if utilization is at least this %.
BUSY_THRESHOLD = 20

# Safety net: any job running longer than this gets killed, so the host never gets stuck.
MAX_JOB_SECONDS = 120

# ---------------------------------------------------------------------------
# 2. CONNECT TO THE GPU (NVML = NVIDIA's built-in reporting library)
# ---------------------------------------------------------------------------
pynvml.nvmlInit()
GPU = pynvml.nvmlDeviceGetHandleByIndex(0)
_name = pynvml.nvmlDeviceGetName(GPU)
GPU_NAME = _name.decode() if isinstance(_name, bytes) else _name


def read_gpu():
    """Take one 'photo' of the GPU right now."""
    util = pynvml.nvmlDeviceGetUtilizationRates(GPU)
    mem = pynvml.nvmlDeviceGetMemoryInfo(GPU)
    try:
        temp = pynvml.nvmlDeviceGetTemperature(GPU, pynvml.NVML_TEMPERATURE_GPU)
    except pynvml.NVMLError:
        temp = None
    return {
        "time": round(time.time(), 2),
        "gpu_util": util.gpu,                       # % of the GPU cores busy
        "mem_used_mb": mem.used // (1024 * 1024),   # VRAM in use
        "mem_total_mb": mem.total // (1024 * 1024), # total VRAM (~6144)
        "temp_c": temp,
    }


# ---------------------------------------------------------------------------
# 3. THE JOB BOOK — every job we ever ran, kept in memory
# ---------------------------------------------------------------------------
jobs = {}
lock = threading.Lock()   # stops two workers editing the book at the same time


def current_job_id():
    with lock:
        for jid, job in jobs.items():
            if job["status"] == "running":
                return jid
    return None


def run_and_watch(job_id, script_path):
    """Runs in a background thread: start the job, photograph the GPU every second.
    Built so that whatever goes wrong, the job ALWAYS ends with a final status."""
    job = jobs[job_id]
    log_path = LOG_DIR / f"{job_id}.log"
    proc = None
    timed_out = False
    error = None

    try:
        with open(log_path, "w") as log:
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(script_path.parent),
            )
            with lock:
                job["pid"] = proc.pid

            while proc.poll() is None:          # None means "still running"
                if time.time() - job["started_at"] > MAX_JOB_SECONDS:
                    proc.kill()                  # stuck job -> force stop
                    proc.wait()
                    timed_out = True
                    break
                try:
                    sample = read_gpu()
                    with lock:
                        job["samples"].append(sample)
                except Exception as e:           # one bad reading must not kill the watcher
                    print(f"[{job_id}] GPU read failed: {e}", flush=True)
                time.sleep(1)
    except Exception as e:
        error = str(e)
        print(f"[{job_id}] watcher error: {e}", flush=True)
        if proc and proc.poll() is None:
            proc.kill()
    finally:
        ended = time.time()
        with lock:
            samples = job["samples"]
            if error:
                job["status"] = "failed"
                job["error"] = error
            elif timed_out:
                job["status"] = "timeout"
            else:
                job["status"] = "finished" if proc and proc.returncode == 0 else "failed"
            job["ended_at"] = round(ended, 2)
            job["runtime_s"] = round(ended - job["started_at"], 1)
            job["busy_seconds"] = sum(1 for s in samples if s["gpu_util"] >= BUSY_THRESHOLD)
            job["peak_util"] = max((s["gpu_util"] for s in samples), default=0)
            job["peak_mem_mb"] = max((s["mem_used_mb"] for s in samples), default=0)


def log_tail(job_id, lines=8):
    path = LOG_DIR / f"{job_id}.log"
    if not path.exists():
        return []
    return path.read_text(errors="ignore").splitlines()[-lines:]


# ---------------------------------------------------------------------------
# 4. THE WEB SERVER — the "counters" other laptops can call
# ---------------------------------------------------------------------------
app = FastAPI(title="GPUSetu Host Daemon")

# Let the dashboard (running on the Aspire Lite) call us from a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class JobRequest(BaseModel):
    job_type: str = "mnist"


@app.get("/")
def health():
    """Is the host alive, and is it free?"""
    return {
        "host_id": HOST_ID,
        "gpu": GPU_NAME,
        "status": "busy" if current_job_id() else "idle",
    }


@app.get("/stats")
def stats():
    """Live GPU reading — the dashboard calls this every second for the graph."""
    reading = read_gpu()
    reading["host_id"] = HOST_ID
    reading["gpu"] = GPU_NAME
    reading["running_job"] = current_job_id()
    return reading


@app.post("/run-job")
def run_job(req: JobRequest):
    """Start a job. Refuses if the GPU is already busy (one GPU = one job)."""
    if req.job_type not in ALLOWED_JOBS:
        raise HTTPException(400, f"Unknown job '{req.job_type}'. Allowed: {list(ALLOWED_JOBS)}")

    if current_job_id():
        raise HTTPException(409, "Host busy")   # later: backend sees this -> cloud fallback

    job_id = uuid.uuid4().hex[:8]
    with lock:
        jobs[job_id] = {
            "job_id": job_id,
            "job_type": req.job_type,
            "host_id": HOST_ID,
            "status": "running",
            "started_at": round(time.time(), 2),
            "samples": [],
        }

    threading.Thread(
        target=run_and_watch,
        args=(job_id, ALLOWED_JOBS[req.job_type]),
        daemon=True,
    ).start()

    return {"job_id": job_id, "status": "running"}


@app.get("/job/{job_id}")
def job_status(job_id: str):
    """Full record of one job: status, runtime, busy seconds, every GPU sample."""
    with lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "No such job")
        result = dict(job)
        result["samples"] = list(job["samples"])
    result["log_tail"] = log_tail(job_id)
    return result


@app.get("/jobs")
def list_jobs():
    """Short summary of every job this host has run."""
    with lock:
        return [
            {k: v for k, v in job.items() if k != "samples"}
            for job in jobs.values()
        ]