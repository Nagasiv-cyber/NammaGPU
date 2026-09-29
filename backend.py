"""
GPUSetu Marketplace Backend  —  runs on the Aspire Lite

Start it with:
    python -m uvicorn backend:app --host 0.0.0.0 --port 9000
Dashboard:  http://127.0.0.1:9000

Every job goes through:
  1. PRICING AGENT   quotes a per-second price from recent demand
  2. MATCHING        first idle, staked peer host  ->  else CLOUD FALLBACK
  3. ESCROW          buyer's payment locked on MST before the GPU starts   (peer jobs)
  4. METERING        GPU telemetry every second; only busy seconds count
  5. ANOMALY AGENT   host's claim vs. telemetry  ->  SETTLE honestly, or SLASH a cheater
"""

import hashlib
import json
import os
import threading
import time
import uuid
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------------------------
# The ALG's address depends on the network: 192.168.137.1 on the ALG's own hotspot,
# something else on a phone hotspot. Override it at startup without editing code:
#     set HOST_URL=http://192.168.43.25:8000      (Command Prompt)
#     $env:HOST_URL = "http://192.168.43.25:8000"  (PowerShell)
HOSTS = [
    {"host_id": "acer-alg-rtx3050-6gb", "url": os.environ.get("HOST_URL", "http://192.168.137.1:8000")},
]
print(f"Host daemon expected at {HOSTS[0]['url']}")

BASE_RATE = 0.001         # normal price per VERIFIED GPU-second
SYMBOL = "MSTC"
BUSY_THRESHOLD = 20       # >= 20% GPU utilization in a second = real work
TIMEOUT = 3               # seconds to wait for a host before calling it offline

# Anomaly agent: slashing is for LYING, not for slow starts.
# Billing already pays only verified seconds, so idle startup time (5 s natively, 10-15 s
# inside a Docker sandbox) costs the host, not the buyer. Fraud = claiming work that the
# telemetry shows mostly never happened.
FRAUD_MIN_VERIFIED_SHARE = 0.5   # telemetry must back at least half of the claimed time
FRAUD_MIN_CLAIM = 10             # ignore tiny jobs, too short to judge

# Pricing agent: demand = job requests in the last 10 minutes.
DEMAND_WINDOW = 600
PRICE_TIERS = [           # (at least this many recent requests, multiplier, label)
    (6, 1.5, "high demand"),
    (3, 1.25, "rising demand"),
    (0, 1.0, "normal demand"),
]

# Cloud fallback (SIMULATED): what a Vast.ai / RunPod API call would give us.
CLOUD_PROVIDER = "Vast.ai-style cloud (simulated)"
CLOUD_SPINUP = 3          # seconds to "start an instance"
CLOUD_RUN = 15            # seconds the simulated job runs
CLOUD_WHOLESALE_SHARE = 0.75   # we pay the provider 75% of the buyer's price; 25% is our margin

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"      # buyer's copies of delivered files: results/<job_id>/
jobs = {}                               # the marketplace's job book
request_log = deque(maxlen=500)         # times of recent job requests (for the pricing agent)
settle_guard = threading.Lock()         # a job is settled or slashed exactly once
delivery_guard = threading.Lock()       # a job's files are delivered exactly once
match_guard = threading.Lock()          # two requests can't grab the same host at once
reserved = set()                        # hosts promised to a job still being set up

# ---------------------------------------------------------------------------
# 2. BLOCKCHAIN (optional)
# ---------------------------------------------------------------------------
chain = None
chain_error = None
LOCK_AMOUNT = 0
MIN_STAKE = 0
onchain_rate = None

try:
    from chain import Chain, load_config
    _cfg = load_config()
    if _cfg and _cfg.get("enabled") and _cfg.get("contract_address"):
        chain = Chain(_cfg)
        BASE_RATE = _cfg.get("rate_per_sec", BASE_RATE)
        onchain_rate = chain.rate_per_sec()
        MIN_STAKE = chain.min_stake()
        SYMBOL = _cfg.get("symbol", "tMSTC")
        LOCK_AMOUNT = _cfg["lock_amount"]
        for h in HOSTS:                      # demo: the one host uses the "host" test wallet
            h.setdefault("wallet", chain.address("host"))
        print(f"Chain ON: contract {_cfg['contract_address']}, on-chain rate {onchain_rate} {SYMBOL}/sec")
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
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(host["url"] + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def host_status(host):
    """Ask a host 'are you alive and free?' One retry, so a single slow reply isn't 'offline'."""
    last_error = None
    for _ in range(2):
        try:
            info = host_call(host, "/")
            return {"host_id": host["host_id"], "url": host["url"],
                    "gpu": info.get("gpu"), "status": info.get("status"),
                    "pause_reason": info.get("pause_reason"),
                    "sandbox": info.get("sandbox", "none")}
        except Exception as e:
            last_error = e
    print(f"Host {host['host_id']} unreachable: {last_error}")
    return {"host_id": host["host_id"], "url": host["url"], "gpu": None, "status": "offline"}


def host_deposit_status(host):
    """'ok' | 'slashed' (no deposit: may not take jobs) | 'unknown' (couldn't reach MST to check)."""
    if chain is None:
        return "ok"
    try:
        return "ok" if chain.host_stake(host["wallet"]) >= MIN_STAKE else "slashed"
    except Exception as e:
        print(f"Could not read {host['host_id']}'s deposit from MST testnet: {e}")
        return "unknown"


def find_host(host_id):
    return next((h for h in HOSTS if h["host_id"] == host_id), None)


# ---------------------------------------------------------------------------
# 4. PRICING AGENT — rule-based surge pricing
# ---------------------------------------------------------------------------
def pricing_agent():
    now = time.time()
    recent = sum(1 for t in request_log if now - t < DEMAND_WINDOW)
    for threshold, multiplier, label in PRICE_TIERS:
        if recent >= threshold:
            rate = round(BASE_RATE * multiplier, 6)
            return {"rate_per_sec": rate, "multiplier": multiplier, "recent_requests": recent,
                    "reason": f"{recent} job requests in the last 10 min: {label} ({multiplier}x base price)"}


def apply_price_on_chain(rate):
    """Change the contract's price only when the tier changes (each change is a transaction)."""
    global onchain_rate
    if chain is not None and rate != onchain_rate:
        chain.set_rate(rate)
        onchain_rate = rate


# ---------------------------------------------------------------------------
# 5. BILLING + ANOMALY AGENT
# ---------------------------------------------------------------------------
def make_bill(host_job, rate):
    """Charge only for seconds where the GPU was measurably working."""
    samples = host_job.get("samples", [])
    verified = sum(1 for s in samples if s["gpu_util"] >= BUSY_THRESHOLD)
    if host_job.get("status") != "running" and "busy_seconds" in host_job:
        verified = host_job["busy_seconds"]
    if host_job.get("status") == "running":
        wall = (samples[-1]["time"] - host_job["started_at"]) if samples else 0
    else:
        wall = host_job.get("runtime_s", 0)
    verified = min(verified, int(wall))      # never bill more seconds than the job ran
    return {
        "rate_per_sec": rate,
        "symbol": SYMBOL,
        "wall_seconds": round(wall, 1),
        "verified_seconds": verified,
        "billed_amount": round(verified * rate, 6),
        "time_based_amount": round(wall * rate, 6),
    }


def anomaly_agent(host_job, bill):
    """Compare what the host CLAIMS against what the GPU telemetry SHOWS."""
    claimed = host_job.get("claimed_seconds", int(bill["wall_seconds"]))
    verified = bill["verified_seconds"]
    share = verified / claimed if claimed else 1.0
    if claimed >= FRAUD_MIN_CLAIM and share < FRAUD_MIN_VERIFIED_SHARE:
        return {"verdict": "fraud", "claimed_seconds": claimed, "verified_seconds": verified,
                "reason": f"Host claimed {claimed} s of GPU work, but the telemetry backs only "
                          f"{verified} s ({share:.0%}). Below the {FRAUD_MIN_VERIFIED_SHARE:.0%} "
                          f"minimum, so the claim is treated as false."}
    return {"verdict": "honest", "claimed_seconds": claimed, "verified_seconds": verified,
            "reason": f"Telemetry backs {verified} s of the {claimed} s claimed ({share:.0%}). "
                      f"The other {claimed - verified} s were startup, which is never billed."}


# ---------------------------------------------------------------------------
# 6. ON-CHAIN SETTLEMENT (background, so the dashboard never freezes)
# ---------------------------------------------------------------------------
def settle_in_background(job_id, verified_seconds):
    job = jobs[job_id]
    rate = job.get("rate", BASE_RATE)

    def work():
        try:
            tx = chain.settle(job_id, verified_seconds)
            paid = round(min(verified_seconds * rate, LOCK_AMOUNT), 6)
            job["chain"].update(status="settled", settle_tx=tx, settle_url=chain.tx_url(tx),
                                paid_to_host=paid, refunded_to_buyer=round(LOCK_AMOUNT - paid, 6))
        except Exception as e:
            job["chain"].update(status="failed", error=f"Settlement failed: {e}")

    job["chain"]["status"] = "settling"
    threading.Thread(target=work, daemon=True).start()


def slash_in_background(job_id, reason):
    job = jobs[job_id]

    def work():
        try:
            penalty = chain.host_stake(chain.address("host"))
            tx = chain.slash(job_id, reason[:200])
            job["chain"].update(status="slashed", slash_tx=tx, slash_url=chain.tx_url(tx),
                                paid_to_host=0, refunded_to_buyer=LOCK_AMOUNT,
                                penalty_to_buyer=round(penalty, 6))
        except Exception as e:
            job["chain"].update(status="failed", error=f"Slash failed: {e}")

    job["chain"]["status"] = "slashing"
    threading.Thread(target=work, daemon=True).start()


# ---------------------------------------------------------------------------
# 6b. DELIVERY: copy the job's files from the host and verify their fingerprints
# ---------------------------------------------------------------------------
def deliver_in_background(job_id, host, host_job_id, artifacts):
    job = jobs[job_id]
    job["delivery"] = {"status": "copying", "files": []}

    def work():
        folder = RESULTS_DIR / job_id
        folder.mkdir(parents=True, exist_ok=True)
        files, all_ok = [], True
        for a in artifacts:
            try:
                url = f"{host['url']}/job/{host_job_id}/artifact/{a['name']}"
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = r.read()
                (folder / a["name"]).write_bytes(data)
                fingerprint = hashlib.sha256(data).hexdigest()
                ok = fingerprint == a["sha256"]
                all_ok &= ok
                files.append({"name": a["name"], "size_bytes": len(data), "sha256": fingerprint,
                              "verified": ok, "url": f"/api/jobs/{job_id}/files/{a['name']}"})
            except Exception as e:
                all_ok = False
                files.append({"name": a["name"], "error": str(e)})
        job["delivery"] = {"status": "delivered" if all_ok else "failed", "files": files,
                           "folder": str(folder)}

    threading.Thread(target=work, daemon=True).start()


# ---------------------------------------------------------------------------
# 7. STARTING JOBS: peer host, or cloud fallback
# ---------------------------------------------------------------------------
def start_on_host(h, req, quote):
    """Lock payment, then start the job on host h (already reserved for us)."""
    job_id = uuid.uuid4().hex[:8]
    chain_info = {"status": "off"}

    if chain is not None:
        try:
            apply_price_on_chain(quote["rate_per_sec"])
            lock_tx = chain.lock_payment(job_id, h["wallet"], LOCK_AMOUNT)
        except Exception as e:
            raise HTTPException(502, f"Could not lock payment on MST testnet: {e}")
        chain_info = {"status": "locked", "locked_amount": LOCK_AMOUNT,
                      "lock_tx": lock_tx, "lock_url": chain.tx_url(lock_tx)}

    try:
        started = host_call(h, "/run-job", "POST", {"job_type": req.job_type})
    except Exception as e:
        if chain is not None:                    # job never ran -> buyer gets everything back
            jobs[job_id] = {"job_id": job_id, "chain": chain_info, "rate": quote["rate_per_sec"]}
            settle_in_background(job_id, 0)
        raise HTTPException(502, f"Payment was locked but the host did not start the job ({e}). "
                                 "The payment is being refunded.")

    jobs[job_id] = {
        "job_id": job_id, "job_type": req.job_type, "host_id": h["host_id"],
        "host_job_id": started["job_id"], "source": "peer",
        "submitted_at": round(time.time(), 2), "rate": quote["rate_per_sec"],
        "pricing": quote, "chain": chain_info,
    }
    return jobs[job_id]


def start_on_cloud(req, quote, why):
    """SIMULATED cloud fallback: stands in for a Vast.ai / RunPod API call."""
    job_id = uuid.uuid4().hex[:8]
    jobs[job_id] = {
        "job_id": job_id, "job_type": req.job_type, "host_id": CLOUD_PROVIDER,
        "source": "cloud", "fallback_reason": why,
        "submitted_at": round(time.time(), 2), "rate": quote["rate_per_sec"],
        "pricing": quote, "chain": {"status": "off"},
    }
    return jobs[job_id]


def cloud_detail(job):
    elapsed = time.time() - job["submitted_at"]
    running = elapsed < CLOUD_SPINUP + CLOUD_RUN
    billed = 0 if elapsed < CLOUD_SPINUP else int(min(elapsed - CLOUD_SPINUP, CLOUD_RUN))
    rate = job["rate"]
    job["status"] = "running" if running else "finished"
    job["bill"] = {"rate_per_sec": rate, "symbol": SYMBOL, "wall_seconds": round(min(elapsed, CLOUD_SPINUP + CLOUD_RUN), 1),
                   "verified_seconds": billed, "billed_amount": round(billed * rate, 6),
                   "time_based_amount": round(billed * rate, 6),
                   "provider_cost": round(billed * rate * CLOUD_WHOLESALE_SHARE, 6),
                   "margin": round(billed * rate * (1 - CLOUD_WHOLESALE_SHARE), 6)}
    return {**job, "samples": [], "log_tail": [
        f"Fallback: {job['fallback_reason']}",
        f"Provider: {CLOUD_PROVIDER}. Instance started in {CLOUD_SPINUP} s.",
        "Billed by the provider's per-second meter (no peer telemetry for cloud jobs).",
    ] + (["DONE: cloud session finished"] if not running else [])}


# ---------------------------------------------------------------------------
# 8. WEB SERVER
# ---------------------------------------------------------------------------
app = FastAPI(title="NammaGPU Marketplace")

# The host portal runs on the host's own laptop and reads earnings from here.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class JobRequest(BaseModel):
    job_type: str = "mnist"      # "fake" = fraud demo (cheating host)


@app.get("/")
def product_app():
    """The NammaGPU product: home, rent, share your GPU, how we verify."""
    return FileResponse(BASE_DIR / "static" / "app.html")


@app.get("/console")
def demo_console():
    """The original single-screen demo console (backup for live demos)."""
    return FileResponse(BASE_DIR / "static" / "dashboard.html")


@app.get("/api/hosts")
def list_hosts():
    return [host_status(h) for h in HOSTS]


@app.get("/api/live")
def live():
    for h in HOSTS:
        try:
            reading = host_call(h, "/stats")
            reading["host_id"] = h["host_id"]
            return reading
        except Exception:
            continue
    raise HTTPException(503, "No host reachable. Check the hotspot and that the host daemon is running.")


@app.get("/api/pricing")
def pricing():
    return {**pricing_agent(), "symbol": SYMBOL, "base_rate": BASE_RATE}


@app.get("/api/chain")
def chain_info():
    if chain is None:
        return {"enabled": False, "symbol": SYMBOL, "error": chain_error}
    try:
        host_wallet = chain.address("host")
        return {
            "enabled": True, "symbol": SYMBOL,
            "contract": chain.cfg["contract_address"],
            "explorer_tx_url": chain.cfg.get("explorer_tx_url", ""),
            "rate_per_sec": onchain_rate, "lock_amount": LOCK_AMOUNT, "min_stake": MIN_STAKE,
            "buyer_balance": round(chain.balance("buyer"), 6),
            "host_balance": round(chain.balance("host"), 6),
            "host_stake": round(chain.host_stake(host_wallet), 6),
        }
    except Exception as e:
        return {"enabled": True, "symbol": SYMBOL, "error": f"Could not read the chain: {e}"}


@app.post("/api/jobs")
def submit_job(req: JobRequest):
    """PRICING -> MATCHING -> (peer + escrow) or (cloud fallback)."""
    quote = pricing_agent()
    request_log.append(time.time())

    reasons = []
    for h in HOSTS:
        with match_guard:
            hs = host_status(h)
            status = "reserved" if h["host_id"] in reserved else hs["status"]
            if status == "paused":
                status = f"paused ({hs.get('pause_reason') or 'by its owner'})"
            if status == "idle":
                deposit = host_deposit_status(h)
                if deposit == "slashed":
                    status = "suspended (deposit slashed, must re-stake)"
                elif deposit == "unknown":
                    status = "unverified (MST testnet unreachable, so its deposit can't be checked)"
            if status != "idle":
                reasons.append(f"{h['host_id']} is {status}")
                continue
            reserved.add(h["host_id"])           # claim it before the slow payment step
        try:
            return start_on_host(h, req, quote)
        finally:
            reserved.discard(h["host_id"])

    if req.job_type == "fake":
        raise HTTPException(503, "The fraud demo needs the peer host: " + "; ".join(reasons))
    return start_on_cloud(req, quote, "; ".join(reasons) or "no peer hosts registered")


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str):
    job = jobs.get(job_id)
    if not job or "host_id" not in job:
        raise HTTPException(404, "No such job")
    if job["source"] == "cloud":
        return cloud_detail(job)

    host = find_host(job["host_id"])
    try:
        host_job = host_call(host, f"/job/{job['host_job_id']}")
    except Exception:
        raise HTTPException(502, "Lost contact with the host running this job.")

    bill = make_bill(host_job, job["rate"])
    job["status"] = host_job["status"]
    job["bill"] = bill

    if host_job["status"] != "running":
        job.setdefault("audit", anomaly_agent(host_job, bill))
        with delivery_guard:
            if "delivery" not in job:
                if job["audit"]["verdict"] == "fraud":
                    job["delivery"] = {"status": "withheld", "files": []}   # a cheater's output isn't trusted
                elif host_job.get("artifacts"):
                    deliver_in_background(job_id, host, job["host_job_id"], host_job["artifacts"])
                else:
                    job["delivery"] = {"status": "none", "files": []}
        if job["audit"]["verdict"] == "fraud":
            bill["billed_amount"] = 0                # a caught cheater earns nothing
        with settle_guard:
            if job["chain"].get("status") == "locked":
                if job["audit"]["verdict"] == "fraud":
                    slash_in_background(job_id, job["audit"]["reason"])
                else:
                    settle_in_background(job_id, bill["verified_seconds"])

    return {
        **job,
        "status": host_job["status"],
        "samples": host_job.get("samples", []),
        "peak_util": host_job.get("peak_util"),
        "peak_mem_mb": host_job.get("peak_mem_mb"),
        "sandbox": host_job.get("sandbox", "none"),
        "log_tail": host_job.get("log_tail", []),
        "bill": bill,
    }


@app.get("/api/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str):
    """Download a delivered file (only files that were delivered and verified)."""
    job = jobs.get(job_id) or {}
    delivered = {f["name"] for f in job.get("delivery", {}).get("files", []) if f.get("verified")}
    if name not in delivered:
        raise HTTPException(404, "No such delivered file")
    return FileResponse(RESULTS_DIR / job_id / name, filename=name)


@app.get("/api/host/summary")
def host_summary():
    """Everything the host portal shows about money and reputation."""
    peer = [j for j in jobs.values() if j.get("source") == "peer" and j.get("status") not in (None, "running")]
    paid = [j for j in peer if j.get("chain", {}).get("status") == "settled"
            or (j.get("chain", {}).get("status") == "off" and (j.get("audit") or {}).get("verdict") == "honest")]
    today = time.strftime("%Y-%m-%d")
    earned = lambda js: round(sum((j.get("chain", {}).get("paid_to_host")
                                   if j.get("chain", {}).get("status") == "settled"
                                   else j.get("bill", {}).get("billed_amount", 0)) or 0 for j in js), 6)
    summary = {
        "symbol": SYMBOL,
        "earned_total": earned(paid),
        "earned_today": earned([j for j in paid if time.strftime("%Y-%m-%d", time.localtime(j["submitted_at"])) == today]),
        "jobs_honest": sum(1 for j in peer if (j.get("audit") or {}).get("verdict") == "honest"),
        "jobs_slashed": sum(1 for j in peer if (j.get("audit") or {}).get("verdict") == "fraud"),
        "recent": [{"job_id": j["job_id"], "verdict": (j.get("audit") or {}).get("verdict"),
                    "verified_seconds": j.get("bill", {}).get("verified_seconds"),
                    "earned": (j.get("chain", {}).get("paid_to_host") if j.get("chain", {}).get("status") == "settled"
                               else (0 if (j.get("audit") or {}).get("verdict") == "fraud" else j.get("bill", {}).get("billed_amount"))),
                    "submitted_at": j["submitted_at"]}
                   for j in sorted(peer, key=lambda j: j["submitted_at"], reverse=True)[:10]],
        "chain": chain is not None,
    }
    if chain is not None:
        try:
            wallet = chain.address("host")
            summary.update(wallet=wallet, wallet_balance=round(chain.balance("host"), 6),
                           deposit=round(chain.host_stake(wallet), 6), min_deposit=MIN_STAKE)
        except Exception as e:
            summary["chain_error"] = f"Could not read MST testnet: {e}"
    return summary


@app.post("/api/host/restake")
def host_restake():
    """DEMO ONLY: the marketplace holds the host's test key, so it can put up the deposit.
    In production the host signs this with their own wallet."""
    if chain is None:
        raise HTTPException(400, "The blockchain is off in this setup.")
    wallet = chain.address("host")
    current = chain.host_stake(wallet)
    if current >= MIN_STAKE:
        return {"status": "already_staked", "deposit": current}
    try:
        tx = chain.register_host(MIN_STAKE)
    except Exception as e:
        raise HTTPException(502, f"Could not put up the deposit on MST testnet: {e}")
    return {"status": "staked", "deposit": MIN_STAKE, "tx": tx, "tx_url": chain.tx_url(tx)}


@app.get("/api/jobs")
def job_history():
    for j in jobs.values():
        if j.get("source") == "cloud" and j.get("status") != "finished":
            cloud_detail(j)                      # bring finished cloud jobs up to date
    return sorted([j for j in jobs.values() if "host_id" in j],
                  key=lambda j: j["submitted_at"], reverse=True)
