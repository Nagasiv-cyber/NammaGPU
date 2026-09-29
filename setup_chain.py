"""
GPUSetu chain setup — run these ONCE, in order, on the Aspire Lite:

    python setup_chain.py check           # can we reach MST? do the wallets have test tokens?
    python setup_chain.py deploy          # put the contract on MST testnet
    python setup_chain.py register-host   # host locks its security deposit
    python setup_chain.py status          # show balances, stake and rate any time

Also useful:
    python setup_chain.py withdraw-stake  # host takes its deposit back (e.g. before redeploying)
"""

import sys

from chain import Chain, load_config, save_config


def main():
    cfg = load_config()
    if cfg is None:
        sys.exit("chain_config.json not found. Copy chain_config.example.json to chain_config.json and fill it in.")
    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    chain = Chain(cfg)
    sym = cfg.get("symbol", "tMSTC")

    if command == "check":
        if not chain.w3.is_connected():
            sys.exit(f"Cannot reach {cfg['rpc_url']}. Check the internet connection.")
        real_id = chain.w3.eth.chain_id
        print(f"Connected. Chain ID {real_id}, latest block {chain.w3.eth.block_number}")
        if real_id != cfg["chain_id"]:
            sys.exit(f"Chain ID mismatch: config says {cfg['chain_id']}, network says {real_id}. Fix chain_config.json.")
        for role in ("operator", "buyer", "host"):
            print(f"  {role:9} {chain.address(role)}  {chain.balance(role):.4f} {sym}")
        print("Every wallet needs test tokens from the faucet before the next step.")

    elif command == "deploy":
        address, tx = chain.deploy(cfg["rate_per_sec"], cfg["min_stake"])
        cfg["contract_address"] = address
        save_config(cfg)
        print(f"Contract deployed at {address}")
        print(f"Transaction: {chain.tx_url(tx)}")
        print("Saved the address into chain_config.json.")

    elif command == "register-host":
        need_contract(chain)
        tx = chain.register_host(cfg["min_stake"])
        print(f"Host staked {cfg['min_stake']} {sym}. Transaction: {chain.tx_url(tx)}")

    elif command == "withdraw-stake":
        need_contract(chain)
        tx = chain.withdraw_stake()
        print("Host had no deposit in this contract." if tx is None
              else f"Host deposit returned. Transaction: {chain.tx_url(tx)}")

    elif command == "status":
        need_contract(chain)
        host = chain.address("host")
        print(f"Contract     {cfg['contract_address']}")
        print(f"Rate         {chain.rate_per_sec()} {sym} per verified second")
        print(f"Host stake   {chain.host_stake(host)} {sym}  (minimum {chain.min_stake()})")
        for role in ("operator", "buyer", "host"):
            print(f"{role:12} {chain.balance(role):.6f} {sym}")

    else:
        sys.exit(__doc__)


def need_contract(chain):
    if chain.contract is None:
        sys.exit("No contract yet. Run:  python setup_chain.py deploy")


if __name__ == "__main__":
    main()
