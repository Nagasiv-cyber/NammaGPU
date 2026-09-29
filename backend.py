"""
GPUSetu Marketplace Backend  —  runs on the Aspire Lite

Start it with:
    python -m uvicorn backend:app --host 0.0.0.0 --port 9000

Then open the dashboard in a browser:  http://127.0.0.1:9000

If chain_config.json exists and "enabled" is true, every job is paid on MST testnet:
  lock payment in escrow -> run job -> settle for verified seconds.
Otherwise it runs exactly like Phase 3 (billing shown, nothing on chain).
"""

import json
import threading
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

RATE_PER_SEC = 0.001      # price per VERIFIED GPU-second (replaced by the contract's rate if chain is on)
SYMBOL = "MSTC"
BUSY_THRESHOLD = 20       # same rule as the host: >= 20% utilization = real work
TIMEOUT = 3               # seconds to wait for a host before calling it offline

BASE_DIR = Path(__file__).parent
jobs = {}                 # the marketplace's job book
settle_guard = threading.Lock()   # makes sure a job is settled exactly once
match_guard = threading.Lock()    # two requests can't grab the same host at once
reserved = set()                  # hosts promised to a job that is still being set up

# ---------------------------------------------------------------------------
# 2. BLOCKCHAIN (optional)
# ---------------------------------------------------------------------------
chain = None
chain_error = None
LOCK_AMOUNT = 0

try:
    from chain import Chain, load_config
    _cfg = load_config()
    if _cfg and _cfg.get("enabled") and _cfg.get("contract_address"):
        chain = Chain(_cfg)
        RATE_PER_SEC = chain.rate_per_sec()
        SYMBOL = _cfg.get("symbol", "tMSTC")
        LOCK_AMOUNT = _cfg["lock_amount"]
        for h in HOSTS:                      # demo: the one host uses the "host" test wallet
            h.setdefault("wallet", chain.address("host"))
        print(f"Chain ON: contract {_cfg['contract_address']}, rate {RATE_PER_SEC} {SYMBOL}/sec")
    else:
        print("Chain OFF: no chain_config.json, not enabled, or no contract deployed yet.")
except Exception as e:                      # never let chain problems stop the marketplace
    chain = None
    chain_error = str(e)
    print(f"Chain OFF because of an error: {e}")


# ---------------------------------------------------------------------------
# 3. TALKING TO HOSTS
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
    """Ask a host 'are you alive and free?' One retry, so a single slow reply isn't 'offline'."""
    for attempt in range(2):
        try:
            info = host_call(host, "/")
            return {"host_id": host["host_id"], "url": host["url"],
                    "gpu": info.get("gpu"), "status": info.get("status")}
        except Exception as e:
            last_error = e
    print(f"Host {host['host_id']} unreachable: {last_error}")
    return {"host_id": host["host_id"], "url": host["url"], "gpu": None, "status": "offline"}


def find_host(host_id):
    for h in HOSTS:
        if h["host_id"] == host_id:
            return h
    return None


# ---------------------------------------------------------------------------
# 4. BILLING — the heart of GPUSetu
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
        "symbol": SYMBOL,
        "wall_seconds": round(wall, 1),
        "verified_seconds": verified,
        "billed_amount": round(verified * RATE_PER_SEC, 6),
        "time_based_amount": round(wall * RATE_PER_SEC, 6),   # what time-based billing would charge
    }


# ---------------------------------------------------------------------------
# 5. SETTLEMENT ON CHAIN (runs in the background so the dashboard never freezes)
# ---------------------------------------------------------------------------
def settle_in_background(job_id, verified_seconds):
    job = jobs[job_id]

    def work():
        try:
            tx = chain.settle(job_id, verified_seconds)
            paid = round(min(verified_seconds * RATE_PER_SEC, LOCK_AMOUNT), 6)
            job["chain"].update(status="settled", settle_tx=tx, settle_url=chain.tx_url(tx),
                                paid_to_host=paid, refunded_to_buyer=round(LOCK_AMOUNT - paid, 6))
        except Exception as e:
            job["chain"].update(status="failed", error=f"Settlement failed: {e}")

    job["chain"]["status"] = "settling"
    threading.Thread(target=work, daemon=True).start()


# ---------------------------------------------------------------------------
# 6. WEB SERVER
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


@app.get("/api/chain")
def chain_info():
    """Is on-chain payment on, and what are the wallet balances right now?"""
    if chain is None:
        return {"enabled": False, "symbol": SYMBOL, "error": chain_error}
    try:
        host_wallet = chain.address("host")
        return {
            "enabled": True,
            "symbol": SYMBOL,
            "contract": chain.cfg["contract_address"],
            "rate_per_sec": RATE_PER_SEC,
            "lock_amount": LOCK_AMOUNT,
            "buyer_balance": round(chain.balance("buyer"), 6),
            "host_balance": round(chain.balance("host"), 6),
            "host_stake": round(chain.host_stake(host_wallet), 6),
        }
    except Exception as e:
        return {"enabled": True, "symbol": SYMBOL, "error": f"Could not read the chain: {e}"}


@app.post("/api/jobs")
def submit_job(req: JobRequest):
    """MATCHING: pick the first idle host, lock payment, then start the job."""
    for h in HOSTS:
        with match_guard:
            if h["host_id"] in reserved or host_status(h)["status"] != "idle":
                continue
            reserved.add(h["host_id"])     # claim it before the slow payment step
        try:
            return start_on_host(h, req)
        finally:
            reserved.discard(h["host_id"])

    raise HTTPException(503, "Every host is busy or offline. The cloud fallback arrives in Phase 5.")


def start_on_host(h, req):
    """Lock payment, then start the job on host h (which is already reserved for us)."""
    job_id = uuid.uuid4().hex[:8]
    chain_info_for_job = {"status": "off"}

    # Step 1: lock the buyer's payment in escrow BEFORE any GPU work starts.
    if chain is not None:
        try:
            lock_tx = chain.lock_payment(job_id, h["wallet"], LOCK_AMOUNT)
        except Exception as e:
            raise HTTPException(502, f"Could not lock payment on MST testnet: {e}")
        chain_info_for_job = {"status": "locked", "locked_amount": LOCK_AMOUNT,
                              "lock_tx": lock_tx, "lock_url": chain.tx_url(lock_tx)}

    # Step 2: start the job on the host.
    try:
        started = host_call(h, "/run-job", "POST", {"job_type": req.job_type})
    except Exception as e:
        if chain is not None:                # job never ran -> give the buyer everything back
            jobs[job_id] = {"job_id": job_id, "chain": chain_info_for_job}
            settle_in_background(job_id, 0)
        raise HTTPException(502, f"Payment was locked but the host did not start the job ({e}). "
                                 "The payment is being refunded.")

    jobs[job_id] = {
        "job_id": job_id,
        "job_type": req.job_type,
        "host_id": h["host_id"],
        "host_job_id": started["job_id"],
        "source": "peer",             # Phase 5 adds "cloud" for the fallback
        "submitted_at": round(time.time(), 2),
        "chain": chain_info_for_job,
    }
    return jobs[job_id]


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str):
    job = jobs.get(job_id)
    if not job or "host_id" not in job:
        raise HTTPException(404, "No such job")
    host = find_host(job["host_id"])
    try:
        host_job = host_call(host, f"/job/{job['host_job_id']}")
    except Exception:
        raise HTTPException(502, "Lost contact with the host running this job.")

    bill = make_bill(host_job)
    job["status"] = host_job["status"]      # remember the latest state for the history list
    job["bill"] = bill

    # Step 3: once the job is over, settle on chain exactly once.
    if host_job["status"] != "running":
        with settle_guard:
            if job["chain"].get("status") == "locked":
                settle_in_background(job_id, bill["verified_seconds"])

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
    return sorted([j for j in jobs.values() if "host_id" in j],
                  key=lambda j: j["submitted_at"], reverse=True)
