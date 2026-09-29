"""
GPUSetu Host Daemon  —  runs on the Acer ALG (RTX 3050 6GB)

Start it with:
    uvicorn host_daemon:app --host 0.0.0.0 --port 8000

Then, from the Aspire Lite, open:  http://<ALG-IP>:8000/docs
"""

import hashlib
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pynvml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------------------------
HOST_ID = "acer-alg-rtx3050-6gb"
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_ROOT = BASE_DIR / "outputs"     # each job's deliverables: outputs/<job_id>/
OUTPUT_ROOT.mkdir(exist_ok=True)

# Only these jobs are allowed to run. A buyer picks a NAME, never sends code.
ALLOWED_JOBS = {
    "mnist": BASE_DIR / "jobs" / "demo_job.py",
    "fake": BASE_DIR / "jobs" / "fake_job.py",     # FRAUD DEMO ONLY: pretends to work
}

# A second counts as "GPU really working" if utilization is at least this %.
BUSY_THRESHOLD = 20

# Safety net: any job running longer than this gets killed, so the host never gets stuck.
MAX_JOB_SECONDS = 120

# ---------------------------------------------------------------------------
# SANDBOX: run every job inside a sealed Docker container (Phase 2)
# ---------------------------------------------------------------------------
USE_DOCKER = True                                           # set False to turn the sandbox off
DOCKER_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"

# Find docker.exe even if this terminal's PATH is out of date (e.g. an IDE opened before
# Docker was installed). You can also point to it directly with the DOCKER_EXE environment variable.
import os
_local = os.environ.get("LOCALAPPDATA", "")
DOCKER_CANDIDATES = [
    os.environ.get("DOCKER_EXE", ""),
    r"C:\Program Files\Docker\Docker\resources\bin\docker.exe",
    r"C:\ProgramData\DockerDesktop\version-bin\docker.exe",
    os.path.join(_local, r"Programs\DockerDesktop\resources\bin\docker.exe"),
    os.path.join(_local, r"Programs\Docker\Docker\resources\bin\docker.exe"),
    os.path.join(_local, r"Docker\resources\bin\docker.exe"),
]
DOCKER_EXE = shutil.which("docker") or next((p for p in DOCKER_CANDIDATES if p and Path(p).exists()), None)


def docker_ready():
    """Sandbox only if Docker is installed, running, and the PyTorch image is downloaded.
    Returns (ready, reason) so the startup message can say WHY it's off."""
    if not USE_DOCKER:
        return False, "USE_DOCKER is set to False"
    if DOCKER_EXE is None:
        return False, "Docker is not installed (docker.exe not found)"
    try:
        engine = subprocess.run([DOCKER_EXE, "info"], capture_output=True, timeout=20)
        if engine.returncode != 0:
            return False, "Docker engine not running. Open Docker Desktop and wait for it to start"
        image = subprocess.run([DOCKER_EXE, "image", "inspect", DOCKER_IMAGE], capture_output=True, timeout=20)
        if image.returncode != 0:
            return False, f"image not downloaded yet. Run: docker pull {DOCKER_IMAGE}"
    except Exception as e:
        return False, f"could not talk to Docker ({e})"
    return True, DOCKER_IMAGE


SANDBOX, SANDBOX_REASON = docker_ready()
print(f"Sandbox: {'ON (Docker, ' + SANDBOX_REASON + ')' if SANDBOX else 'OFF: ' + SANDBOX_REASON}")


def job_command(job_id, script_path, out_dir):
    """The command that runs one job: sealed container if possible, plain Python otherwise."""
    if not SANDBOX:
        return [sys.executable, str(script_path)]      # OUTPUT_DIR is passed as an environment variable
    return [
        DOCKER_EXE, "run", "--rm",
        "--name", f"gpusetu-{job_id}",
        "--gpus", "all",                              # the GPU is the ONLY thing shared with the job
        "--network", "none",                          # no internet: can't leak data or download malware
        "--read-only",                                # can't change anything inside the container
        "--tmpfs", "/tmp:rw,size=256m",               # a small scratch space that vanishes afterwards
        "-v", f"{script_path.parent}:/work:ro",       # job files visible, but read-only
        "-v", f"{out_dir}:/out:rw",                   # the ONE writable folder: where results go
        "-e", "OUTPUT_DIR=/out",
        "-w", "/work",
        "--user", "1000:1000", "-e", "HOME=/tmp",     # not the admin/root user
        "--cap-drop", "ALL",                          # no special Linux permissions
        "--security-opt", "no-new-privileges",
        "--memory", "4g", "--cpus", "4", "--pids-limit", "256",   # can't hog the laptop
        DOCKER_IMAGE,
        "python", "-u", script_path.name,
    ]


def stop_container(job_id):
    if SANDBOX:
        subprocess.run([DOCKER_EXE, "kill", f"gpusetu-{job_id}"], capture_output=True, timeout=20)

# ---------------------------------------------------------------------------
# 2. CONNECT TO THE GPU (NVML = NVIDIA's built-in reporting library)
# ---------------------------------------------------------------------------
pynvml.nvmlInit()
GPU = pynvml.nvmlDeviceGetHandleByIndex(0)
_name = pynvml.nvmlDeviceGetName(GPU)
GPU_NAME = _name.decode() if isinstance(_name, bytes) else _name

nvml_lock = threading.Lock()   # one question to the graphics card at a time
nvml_failures = 0              # failed readings in a row


def _reconnect_gpu():
    """Laptop GPUs power down when idle; reconnecting to NVML often brings readings back."""
    global GPU
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass
    pynvml.nvmlInit()
    GPU = pynvml.nvmlDeviceGetHandleByIndex(0)


def read_gpu():
    """Take one 'photo' of the GPU right now. Never crashes:
    if the GPU is asleep (laptop power saving), it reports gpu_state = 'sleeping'."""
    global nvml_failures
    reading = {
        "time": round(time.time(), 2),
        "gpu_util": 0,
        "mem_used_mb": 0,
        "mem_total_mb": 0,
        "temp_c": None,
        "gpu_state": "awake",
    }
    with nvml_lock:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(GPU)
            mem = pynvml.nvmlDeviceGetMemoryInfo(GPU)
            reading["gpu_util"] = util.gpu                          # % of GPU cores busy
            reading["mem_used_mb"] = mem.used // (1024 * 1024)      # VRAM in use
            reading["mem_total_mb"] = mem.total // (1024 * 1024)    # total VRAM (~6144)
            try:
                reading["temp_c"] = pynvml.nvmlDeviceGetTemperature(GPU, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError:
                pass
            nvml_failures = 0
        except pynvml.NVMLError:
            reading["gpu_state"] = "sleeping"                      # counted as 0% = not billed
            nvml_failures += 1
            if nvml_failures % 5 == 0:                              # every 5 misses, try reconnecting
                try:
                    _reconnect_gpu()
                except Exception:
                    pass
    return reading


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
    out_dir = OUTPUT_ROOT / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = None
    timed_out = False
    error = None

    try:
        with open(log_path, "w") as log:
            proc = subprocess.Popen(
                job_command(job_id, script_path, out_dir),
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(script_path.parent),
                env={**os.environ, "OUTPUT_DIR": str(out_dir)},
            )
            with lock:
                job["pid"] = proc.pid

            while proc.poll() is None:          # None means "still running"
                if time.time() - job["started_at"] > MAX_JOB_SECONDS:
                    stop_container(job_id)       # stuck job -> force stop (container first)
                    proc.kill()
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
            stop_container(job_id)
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
            # The host's BILLING CLAIM: "my GPU worked for the whole job".
            # A normal rental would charge this. GPUSetu checks it against the telemetry.
            job["claimed_seconds"] = int(job["runtime_s"])
            job["artifacts"] = list_artifacts(out_dir)
            job["busy_seconds"] = sum(1 for s in samples if s["gpu_util"] >= BUSY_THRESHOLD)
            job["peak_util"] = max((s["gpu_util"] for s in samples), default=0)
            job["peak_mem_mb"] = max((s["mem_used_mb"] or 0 for s in samples), default=0)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_artifacts(out_dir):
    """Every file the job produced, with size and SHA-256 fingerprint."""
    try:
        return [{"name": p.name, "size_bytes": p.stat().st_size, "sha256": sha256_of(p)}
                for p in sorted(out_dir.iterdir()) if p.is_file()]
    except Exception as e:
        print(f"Could not list outputs in {out_dir}: {e}", flush=True)
        return []


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
        "sandbox": "docker" if SANDBOX else "none",
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
            "sandbox": "docker" if SANDBOX else "none",
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


@app.get("/job/{job_id}/artifact/{name}")
def job_artifact(job_id: str, name: str):
    """Download one file the job produced (only names the job actually listed)."""
    with lock:
        job = jobs.get(job_id)
        allowed = {a["name"] for a in (job or {}).get("artifacts", [])}
    if name not in allowed:
        raise HTTPException(404, "No such file for this job")
    return FileResponse(OUTPUT_ROOT / job_id / name, filename=name)


@app.get("/jobs")
def list_jobs():
    """Short summary of every job this host has run."""
    with lock:
        return [
            {k: v for k, v in job.items() if k != "samples"}
            for job in jobs.values()
        ]
