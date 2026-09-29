// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/*
    GPUSetu Escrow — the "vending machine" for GPU payments.

    Who does what:
      * HOST     locks a security deposit (stake) before renting out its GPU.
      * BUYER    locks payment for a job into escrow before the job runs.
      * OPERATOR (the marketplace backend) reads the GPU telemetry and reports
                 how many seconds the GPU really worked. The contract then:
                   - pays the host  verifiedSeconds x rate
                   - refunds the buyer everything else
                 If the telemetry shows cheating, the operator can SLASH:
                   - buyer gets a full refund PLUS the host's deposit.
      * Anyone   can trigger a refund to the buyer if a job is never settled
                 within 1 hour, so buyer money can never be stuck forever.

    Prices: the operator's pricing agent may change ratePerSecond with demand,
    but every job keeps the rate that applied when its payment was locked.

    Deliberately simple: no external libraries, compile with EVM version "paris".
*/
contract GPUSetuEscrow {
    address public operator;
    uint256 public ratePerSecond;               // price per verified GPU-second, in wei
    uint256 public minStake;                    // deposit a host must keep locked
    uint256 public constant REFUND_DELAY = 1 hours;

    struct Job {
        address buyer;
        address host;
        uint256 amount;       // payment locked by the buyer
        uint256 rate;         // price per verified second, fixed at booking time
        uint64 createdAt;
        bool closed;
    }

    mapping(address => uint256) public hostStake;
    mapping(address => uint256) public activeJobs;   // a host can't withdraw its stake mid-job
    mapping(bytes32 => Job) public jobs;

    event HostRegistered(address indexed host, uint256 totalStake);
    event StakeWithdrawn(address indexed host, uint256 amount);
    event PaymentLocked(bytes32 indexed jobId, address indexed buyer, address indexed host, uint256 amount, uint256 rate);
    event Settled(bytes32 indexed jobId, uint256 verifiedSeconds, uint256 paidToHost, uint256 refundedToBuyer);
    event Slashed(bytes32 indexed jobId, address indexed host, uint256 penalty, string reason);
    event Refunded(bytes32 indexed jobId, uint256 amount);
    event RateChanged(uint256 ratePerSecond);

    modifier onlyOperator() {
        require(msg.sender == operator, "only operator");
        _;
    }

    constructor(uint256 _ratePerSecond, uint256 _minStake) {
        operator = msg.sender;
        ratePerSecond = _ratePerSecond;
        minStake = _minStake;
    }

    // ---------------- hosts ----------------

    function registerHost() external payable {
        hostStake[msg.sender] += msg.value;
        require(hostStake[msg.sender] >= minStake, "stake below minimum");
        emit HostRegistered(msg.sender, hostStake[msg.sender]);
    }

    function withdrawStake(uint256 amount) external {
        require(activeJobs[msg.sender] == 0, "host has active jobs");
        require(amount <= hostStake[msg.sender], "not enough stake");
        hostStake[msg.sender] -= amount;
        _send(msg.sender, amount);
        emit StakeWithdrawn(msg.sender, amount);
    }

    // ---------------- buyers ----------------

    function lockPayment(bytes32 jobId, address host) external payable {
        require(jobs[jobId].buyer == address(0), "job id already used");
        require(hostStake[host] >= minStake, "host not staked");
        require(msg.value > 0, "no payment sent");
        jobs[jobId] = Job(msg.sender, host, msg.value, ratePerSecond, uint64(block.timestamp), false);
        activeJobs[host] += 1;
        emit PaymentLocked(jobId, msg.sender, host, msg.value, ratePerSecond);
    }

    function refundExpired(bytes32 jobId) external {
        Job storage j = _openJob(jobId);
        require(block.timestamp >= j.createdAt + REFUND_DELAY, "refund not available yet");
        _close(j);
        _send(j.buyer, j.amount);
        emit Refunded(jobId, j.amount);
    }

    // ---------------- operator ----------------

    function settle(bytes32 jobId, uint256 verifiedSeconds) external onlyOperator {
        Job storage j = _openJob(jobId);
        uint256 pay = verifiedSeconds * j.rate;     // the price agreed when the job was booked
        if (pay > j.amount) pay = j.amount;          // never pay more than was locked
        uint256 refund = j.amount - pay;
        _close(j);
        _send(j.host, pay);
        _send(j.buyer, refund);
        emit Settled(jobId, verifiedSeconds, pay, refund);
    }

    function slash(bytes32 jobId, string calldata reason) external onlyOperator {
        Job storage j = _openJob(jobId);
        uint256 penalty = hostStake[j.host];
        hostStake[j.host] = 0;
        _close(j);
        _send(j.buyer, j.amount + penalty);
        emit Slashed(jobId, j.host, penalty, reason);
    }

    function setRate(uint256 _ratePerSecond) external onlyOperator {
        ratePerSecond = _ratePerSecond;
        emit RateChanged(_ratePerSecond);
    }

    // ---------------- helpers ----------------

    function _openJob(bytes32 jobId) internal view returns (Job storage j) {
        j = jobs[jobId];
        require(j.buyer != address(0), "no such job");
        require(!j.closed, "job already closed");
    }

    function _close(Job storage j) internal {
        j.closed = true;                    // mark closed BEFORE sending money (blocks re-entry tricks)
        activeJobs[j.host] -= 1;
    }

    function _send(address to, uint256 amount) internal {
        if (amount == 0) return;
        (bool ok, ) = payable(to).call{value: amount}("");
        require(ok, "transfer failed");
    }
}
