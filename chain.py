"""
GPUSetu chain layer — lets the marketplace talk to the GPUSetuEscrow contract.

Every blockchain action here is a signed transaction:
  deploy / register_host / lock_payment / settle / slash
"""

import json
import threading
from pathlib import Path

from web3 import Web3

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "chain_config.json"
ARTIFACT_PATH = BASE_DIR / "contracts" / "GPUSetuEscrow.json"


def load_config():
    if not CONFIG_PATH.exists():
        return None
    return json.loads(CONFIG_PATH.read_text())


def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def job_key(job_id: str) -> bytes:
    """The contract identifies jobs by a 32-byte fingerprint of our job id."""
    return Web3.keccak(text=job_id)


class Chain:
    def __init__(self, cfg, w3=None):
        self.cfg = cfg
        self.w3 = w3 or Web3(Web3.HTTPProvider(cfg["rpc_url"], request_kwargs={"timeout": 30}))
        artifact = json.loads(ARTIFACT_PATH.read_text())
        self.abi = artifact["abi"]
        self.bytecode = artifact["bytecode"]
        # One wallet per role. "account" = a key that can sign transactions.
        self.accounts = {role: self.w3.eth.account.from_key(key) for role, key in cfg["keys"].items()}
        self.contract = None
        if cfg.get("contract_address"):
            self.contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(cfg["contract_address"]), abi=self.abi)
        self._send_lock = threading.Lock()     # one transaction at a time -> no nonce clashes

    # ------------------------------------------------------------ basics
    def address(self, role):
        return self.accounts[role].address

    def to_wei(self, amount):
        return self.w3.to_wei(amount, "ether")

    def from_wei(self, wei):
        return float(self.w3.from_wei(wei, "ether"))

    def balance(self, role):
        return self.from_wei(self.w3.eth.get_balance(self.address(role)))

    def tx_url(self, tx_hash):
        return self.cfg.get("explorer_tx_url", "") + tx_hash if tx_hash else None

    def _send(self, role, call, value_wei=0):
        """Build, sign, send one transaction and wait until it is in a block."""
        acct = self.accounts[role]
        with self._send_lock:
            tx = call.build_transaction({
                "from": acct.address,
                "value": value_wei,
                "nonce": self.w3.eth.get_transaction_count(acct.address, "pending"),
                "chainId": self.cfg["chain_id"],
                "gasPrice": self.w3.eth.gas_price,
            })
            signed = acct.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        hex_hash = self.w3.to_hex(tx_hash)
        if receipt.status != 1:
            raise RuntimeError(f"Transaction failed on chain: {hex_hash}")
        return hex_hash, receipt

    # ------------------------------------------------------------ setup
    def deploy(self, rate_per_sec, min_stake):
        factory = self.w3.eth.contract(abi=self.abi, bytecode=self.bytecode)
        tx_hash, receipt = self._send(
            "operator", factory.constructor(self.to_wei(rate_per_sec), self.to_wei(min_stake)))
        self.contract = self.w3.eth.contract(address=receipt.contractAddress, abi=self.abi)
        return receipt.contractAddress, tx_hash

    def register_host(self, stake):
        return self._send("host", self.contract.functions.registerHost(), self.to_wei(stake))[0]

    # ------------------------------------------------------------ jobs
    def lock_payment(self, job_id, host_address, amount):
        return self._send(
            "buyer",
            self.contract.functions.lockPayment(job_key(job_id), Web3.to_checksum_address(host_address)),
            self.to_wei(amount))[0]

    def settle(self, job_id, verified_seconds):
        return self._send("operator", self.contract.functions.settle(job_key(job_id), int(verified_seconds)))[0]

    def slash(self, job_id, reason):
        return self._send("operator", self.contract.functions.slash(job_key(job_id), reason))[0]

    # ------------------------------------------------------------ reading
    def rate_per_sec(self):
        return self.from_wei(self.contract.functions.ratePerSecond().call())

    def host_stake(self, host_address):
        return self.from_wei(self.contract.functions.hostStake(Web3.to_checksum_address(host_address)).call())

    def min_stake(self):
        return self.from_wei(self.contract.functions.minStake().call())
