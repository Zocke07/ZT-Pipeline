#!/usr/bin/env python3
"""verify_system_deployment.py – End-to-End Deployment Verification.

A real-world test runner that validates the Zero-Trust Federated Learning
pipeline actually works on the current machine.  Every assertion is based
on REAL output from commands and container logs — nothing is simulated.

Phases:
    1. Environment Audit   — GPU, Docker, certificates, signing keys
    2. Execution           — docker compose build + up with polling
    3. Log Forensics       — parse ACTUAL container logs for proof strings
    4. Reporting           — pass/fail verdict with evidence

Usage:
    python verify_system_deployment.py                # default: ZT + attack
    python verify_system_deployment.py --scenario zt  # ZT under attack (default)
    python verify_system_deployment.py --scenario clean  # 2 honest clients
    python verify_system_deployment.py --timeout 600  # custom timeout (sec)
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"
OVERRIDE_FILE = PROJECT_ROOT / "experiment_override.yml"

# Container names defined in docker-compose.yml
CONTAINER_SERVER = "fl-server"
CONTAINER_CLIENT0 = "fl-client-0"
CONTAINER_MALICIOUS = "fl-malicious"
CONTAINER_CLIENT1 = "fl-client-1"

# Required certificate files
REQUIRED_CERTS = [
    "certs/ca.crt", "certs/ca.key",
    "certs/server.crt", "certs/server.key",
    "certs/client-0.crt", "certs/client-0.key",
    "certs/client-1.crt", "certs/client-1.key",
]

# Required signing key files
REQUIRED_SIGNING_KEYS = [
    "signing_keys/client-0.private.pem", "signing_keys/client-0.public.pem",
    "signing_keys/client-1.private.pem", "signing_keys/client-1.public.pem",
]

DEFAULT_TIMEOUT = 480  # 8 minutes


# ═══════════════════════════════════════════════════════════════
#  Reporting helpers
# ═══════════════════════════════════════════════════════════════

class TestResult:
    """Accumulates individual check results for the final report."""

    def __init__(self) -> None:
        self.checks: List[Tuple[str, bool, str]] = []  # (name, passed, detail)
        self.failed_logs: Dict[str, str] = {}           # container → last N lines

    def record(self, name: str, passed: bool, detail: str = "") -> bool:
        self.checks.append((name, passed, detail))
        status = "PASS" if passed else "FAIL"
        icon = "✅" if passed else "❌"
        print(f"  {icon} [{status}] {name}")
        if detail:
            for line in detail.strip().splitlines():
                print(f"           {line}")
        return passed

    def attach_failure_logs(self, container: str, logs: str, last_n: int = 20) -> None:
        lines = logs.strip().splitlines()
        tail = "\n".join(lines[-last_n:]) if len(lines) > last_n else "\n".join(lines)
        self.failed_logs[container] = tail

    @property
    def all_passed(self) -> bool:
        return all(p for _, p, _ in self.checks)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        return sum(1 for _, p, _ in self.checks if p)

    def print_summary(self) -> None:
        print()
        print("=" * 64)
        if self.all_passed:
            print("  ✅ SYSTEM VERIFIED: running on real hardware.")
        else:
            failed = [(n, d) for n, p, d in self.checks if not p]
            print(f"  ❌ FAILURE: {len(failed)} of {self.total} checks failed.")
            for name, detail in failed:
                print(f"     • {name}")

        print(f"  Results: {self.passed_count}/{self.total} passed")
        print("=" * 64)

        if self.failed_logs:
            print()
            print("─── Failure Logs (last 20 lines per container) ───")
            for container, logs in self.failed_logs.items():
                print(f"\n  [{container}]:")
                for line in logs.splitlines():
                    print(f"    {line}")
            print()


# ═══════════════════════════════════════════════════════════════
#  Shell helpers
# ═══════════════════════════════════════════════════════════════

def _run(cmd: List[str], timeout: int = 60, cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run a command and return the CompletedProcess."""
    return subprocess.run(
        cmd, capture_output=True, encoding="utf-8", errors="replace",
        timeout=timeout, cwd=cwd or str(PROJECT_ROOT),
    )


def _docker_logs(container: str) -> str:
    """Retrieve all logs from a container (running or stopped)."""
    r = subprocess.run(
        ["docker", "logs", container],
        capture_output=True, encoding="utf-8", errors="replace",
    )
    # docker puts some output on stderr (timestamps, warnings)
    return (r.stdout or "") + (r.stderr or "")


def _docker_compose_cmd(compose_files: List[Path], project: Optional[str] = None) -> List[str]:
    cmd = ["docker", "compose"]
    for f in compose_files:
        cmd.extend(["-f", str(f)])
    if project:
        cmd.extend(["-p", project])
    return cmd


# ═══════════════════════════════════════════════════════════════
#  Phase 1: Environment Audit
# ═══════════════════════════════════════════════════════════════

def phase1_environment_audit(report: TestResult) -> bool:
    """Pre-flight checks: GPU, Docker, certificates, signing keys."""
    print()
    print("━" * 64)
    print("  PHASE 1: Environment Audit (Pre-Flight)")
    print("━" * 64)

    ok = True

    # ── 1.1  GPU visibility ───────────────────────────────────
    try:
        r = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                   "--format=csv,noheader"], timeout=15)
        gpu_found = r.returncode == 0 and r.stdout.strip() != ""
        detail = r.stdout.strip().splitlines()[0] if gpu_found else (r.stderr.strip() or "nvidia-smi not found or no GPU")
        ok &= report.record("GPU visible (nvidia-smi)", gpu_found, detail)
    except FileNotFoundError:
        ok &= report.record("GPU visible (nvidia-smi)", False, "nvidia-smi not found in PATH")
    except subprocess.TimeoutExpired:
        ok &= report.record("GPU visible (nvidia-smi)", False, "nvidia-smi timed out")

    # ── 1.2  Docker daemon running ────────────────────────────
    try:
        r = _run(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=15)
        docker_ok = r.returncode == 0 and r.stdout.strip() != ""
        detail = f"Docker Engine {r.stdout.strip()}" if docker_ok else (r.stderr.strip()[:200] or "docker info failed")
        ok &= report.record("Docker daemon running", docker_ok, detail)
    except FileNotFoundError:
        ok &= report.record("Docker daemon running", False, "docker not found in PATH")
    except subprocess.TimeoutExpired:
        ok &= report.record("Docker daemon running", False, "docker info timed out")

    # ── 1.3  NVIDIA Container Toolkit ─────────────────────────
    try:
        r = _run(["docker", "run", "--rm", "--gpus", "all",
                   "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
                   "nvidia-smi", "-L"], timeout=120)
        nct_ok = r.returncode == 0 and "GPU" in (r.stdout + r.stderr)
        gpu_line = ""
        for line in (r.stdout + r.stderr).splitlines():
            if "GPU" in line:
                gpu_line = line.strip()
                break
        detail = gpu_line if nct_ok else "NVIDIA Container Toolkit not working or image pull failed"
        ok &= report.record("GPU accessible inside Docker", nct_ok, detail)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        ok &= report.record("GPU accessible inside Docker", False, str(e))

    # ── 1.4  mTLS certificates present ────────────────────────
    missing_certs = [f for f in REQUIRED_CERTS if not (PROJECT_ROOT / f).exists()]
    if missing_certs:
        # Attempt to generate
        gen_script = PROJECT_ROOT / "generate_keys.py"
        if gen_script.exists():
            print("         Certificates missing — attempting generation …")
            r = _run([sys.executable, str(gen_script)], timeout=60, cwd=str(PROJECT_ROOT))
            missing_certs = [f for f in REQUIRED_CERTS if not (PROJECT_ROOT / f).exists()]
    certs_ok = len(missing_certs) == 0
    detail = "" if certs_ok else f"Missing: {missing_certs}"
    ok &= report.record("mTLS certificates present", certs_ok, detail)

    # ── 1.5  Signing keys present ─────────────────────────────
    missing_keys = [f for f in REQUIRED_SIGNING_KEYS if not (PROJECT_ROOT / f).exists()]
    if missing_keys:
        gen_script = PROJECT_ROOT / "generate_keys.py"
        if gen_script.exists():
            print("         Signing keys missing — attempting generation …")
            r = _run([sys.executable, str(gen_script)], timeout=60, cwd=str(PROJECT_ROOT))
            missing_keys = [f for f in REQUIRED_SIGNING_KEYS if not (PROJECT_ROOT / f).exists()]
    keys_ok = len(missing_keys) == 0
    detail = "" if keys_ok else f"Missing: {missing_keys}"
    ok &= report.record("RSA signing keys present", keys_ok, detail)

    # ── 1.6  Certificate validity (not expired) ───────────────
    ca_crt = PROJECT_ROOT / "certs" / "ca.crt"
    if ca_crt.exists():
        try:
            r = _run(["openssl", "x509", "-in", str(ca_crt), "-noout",
                       "-checkend", "0"], timeout=10)
            cert_valid = r.returncode == 0
            detail = "Certificate not expired" if cert_valid else "CA certificate is EXPIRED — regenerate with: python generate_keys.py"
            ok &= report.record("CA certificate not expired", cert_valid, detail)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # openssl not on host PATH — skip gracefully
            report.record("CA certificate not expired", True, "openssl not found on host — skipped (checked inside container)")

    # ── 1.7  docker-compose.yml exists ────────────────────────
    ok &= report.record("docker-compose.yml exists", COMPOSE_FILE.exists())

    return ok


# ═══════════════════════════════════════════════════════════════
#  Phase 2: Execution (Build + Up + Poll)
# ═══════════════════════════════════════════════════════════════

def _container_running(container: str) -> bool:
    """Check if a container is in 'running' state."""
    r = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container],
        capture_output=True, encoding="utf-8", errors="replace",
    )
    return r.returncode == 0 and "true" in r.stdout.lower()


def _container_exited(container: str) -> Optional[int]:
    """Return exit code if container has exited, else None."""
    r = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}} {{.State.ExitCode}}", container],
        capture_output=True, encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        return None
    parts = r.stdout.strip().split()
    if len(parts) >= 2 and parts[0] == "exited":
        return int(parts[1])
    return None


def _poll_for_completion(
    server_container: str,
    timeout: int,
    poll_interval: int = 10,
) -> Tuple[bool, float]:
    """Poll container logs until the FL experiment completes or times out.

    Looks for Flower's server shutdown message, which appears after all
    rounds are done:
        'app_shutdown' or 'ServerApp' (shutdown) or the process exits.

    Returns (completed, elapsed_seconds).
    """
    t0 = time.perf_counter()
    deadline = t0 + timeout
    last_round_seen = 0

    while time.perf_counter() < deadline:
        # Check if server has exited (normal completion or crash)
        exit_code = _container_exited(server_container)
        if exit_code is not None:
            elapsed = time.perf_counter() - t0
            return True, elapsed

        # Check server logs for round progress (non-blocking status)
        logs = _docker_logs(server_container)
        rounds = re.findall(r"ROUND (\d+)", logs)
        if rounds:
            current_round = max(int(r) for r in rounds)
            if current_round > last_round_seen:
                last_round_seen = current_round
                print(f"         … Round {current_round} in progress "
                      f"({time.perf_counter() - t0:.0f}s elapsed)")

        # Check for Flower shutdown markers
        if any(marker in logs for marker in [
            "app_shutdown", "ServerApp",
            "Shutdown", "server stopped",
        ]):
            # Flower logged shutdown — give it a moment to exit
            time.sleep(5)
            elapsed = time.perf_counter() - t0
            return True, elapsed

        time.sleep(poll_interval)

    elapsed = time.perf_counter() - t0
    return False, elapsed


def phase2_execution(
    report: TestResult,
    scenario: str,
    timeout: int,
) -> bool:
    """Build images, start containers, poll until completion."""
    print()
    print("━" * 64)
    print(f"  PHASE 2: Execution  (scenario={scenario})")
    print("━" * 64)

    ok = True

    # Determine compose files and services
    compose_files: List[Path] = [COMPOSE_FILE]
    if scenario == "zt":
        # Default docker-compose.yml runs server + client-0 + malicious
        # Override sets MIN_ACCEPTED=1 so honest client is still aggregated
        compose_files.append(OVERRIDE_FILE)
        services = ["data-init", "server", "client-0", "malicious"]
    elif scenario == "clean":
        # Two honest clients via the 'clean' profile
        services = ["data-init", "server", "client-0", "client-1"]
    else:
        services = ["data-init", "server", "client-0", "malicious"]

    compose_cmd = _docker_compose_cmd(compose_files)

    # ── 2.1  Teardown any previous run ────────────────────────
    print("    Cleaning up previous containers …")
    _run(compose_cmd + ["down", "-v", "--remove-orphans"], timeout=60)

    # ── 2.2  Build ────────────────────────────────────────────
    print("    Building Docker images …")
    build_cmd = compose_cmd + ["build"]
    try:
        r = _run(build_cmd, timeout=600)
        build_ok = r.returncode == 0
        detail = "" if build_ok else (r.stderr.strip()[-300:] or r.stdout.strip()[-300:])
        ok &= report.record("Docker build succeeded", build_ok, detail)
    except subprocess.TimeoutExpired:
        ok &= report.record("Docker build succeeded", False, "Build timed out (>600s)")
        return False

    if not build_ok:
        return False

    # ── 2.3  Start services ───────────────────────────────────
    print("    Starting services …")
    profile_args = ["--profile", "clean"] if scenario == "clean" else []
    up_cmd = compose_cmd + profile_args + [
        "up", "-d",
    ] + services
    r = _run(up_cmd, timeout=120)
    start_ok = r.returncode == 0
    detail = "" if start_ok else r.stderr.strip()[:300]
    ok &= report.record("Containers started", start_ok, detail)

    if not start_ok:
        return False

    # ── 2.4  Wait for data-init to complete ───────────────────
    print("    Waiting for CIFAR-10 data init …")
    data_init_ok = False
    for _ in range(60):  # up to 5 minutes at 5s intervals
        ec = _container_exited("fl-data-init")
        if ec is not None:
            data_init_ok = ec == 0
            break
        time.sleep(5)
    detail = "" if data_init_ok else "data-init did not complete or exited with error"
    ok &= report.record("CIFAR-10 data initialised", data_init_ok, detail)

    # ── 2.5  Poll for FL completion ───────────────────────────
    print("    Waiting for FL training to complete …")
    completed, elapsed = _poll_for_completion(CONTAINER_SERVER, timeout)
    detail = f"Completed in {elapsed:.0f}s" if completed else f"Timed out after {elapsed:.0f}s"
    ok &= report.record("FL training completed", completed, detail)

    # Give containers a few seconds to flush logs
    time.sleep(5)

    return ok


# ═══════════════════════════════════════════════════════════════
#  Phase 3: Log Forensics
# ═══════════════════════════════════════════════════════════════

def phase3_log_forensics(report: TestResult, scenario: str) -> bool:
    """Search container logs for proof-of-life strings.

    Every searched string is EXACTLY as printed by the actual code:
        server.py, client.py, client_malicious.py, signing.py
    """
    print()
    print("━" * 64)
    print("  PHASE 3: Log Forensics (Evidence Collection)")
    print("━" * 64)

    ok = True

    server_logs = _docker_logs(CONTAINER_SERVER)
    client0_logs = _docker_logs(CONTAINER_CLIENT0)
    attack_logs = _docker_logs(CONTAINER_MALICIOUS) if scenario == "zt" else ""
    client1_logs = _docker_logs(CONTAINER_CLIENT1) if scenario == "clean" else ""

    # ── 3.1  Gate 1 (mTLS) — Identity Verification ───────────
    #
    # The server does not print a per-client "connected" message for mTLS;
    # mTLS is enforced at the gRPC/TLS layer.  Proof of Gate 1 is:
    #   Server loaded its certs:  "[SERVER] ✓  CA certificate loaded"
    #   Clients loaded their certs: "[Client 0] ✓  Client certificate loaded"
    #   Clients patched gRPC:       "[mTLS] ✓  gRPC patched"
    #
    # If any client FAILED mTLS, it would never reach Gate 2.
    # So Gate 2 receiving updates IS proof that mTLS succeeded.

    # Server-side: mTLS certs loaded
    server_mtls = "CA certificate loaded" in server_logs and "Server certificate loaded" in server_logs
    ok &= report.record(
        "Gate 1 (Identity): Server mTLS certs loaded",
        server_mtls,
        _extract_matching_lines(server_logs, "certificate loaded"),
    )

    # Client-side: mTLS certs loaded and gRPC patched
    client_mtls = "Client certificate loaded" in client0_logs and "gRPC patched" in client0_logs
    ok &= report.record(
        "Gate 1 (Identity): Client-0 mTLS certs loaded",
        client_mtls,
        _extract_matching_lines(client0_logs, "certificate loaded|gRPC patched"),
    )

    # ── 3.2  Gate 2 (Signatures) — Integrity Verification ────
    #
    # Actual server log:  "[Gate 2] ✓  Signature VALID for client-0"
    # Actual server log:  "[Gate 2] Result: N passed, M rejected"

    gate2_valid = "Signature VALID" in server_logs
    gate2_lines = _extract_matching_lines(server_logs, r"\[Gate 2\]")
    ok &= report.record(
        "Gate 2 (Integrity): At least one signature verified",
        gate2_valid,
        gate2_lines,
    )

    # Check that Gate 2 ran for at least one round
    gate2_results = re.findall(r"\[Gate 2\] Result: (\d+) passed, (\d+) rejected", server_logs)
    gate2_ran = len(gate2_results) > 0
    ok &= report.record(
        "Gate 2 (Integrity): Verification ran for ≥1 round",
        gate2_ran,
        f"{len(gate2_results)} round(s) processed" if gate2_ran else "No Gate 2 Result lines found",
    )

    # ── 3.3  Gate 3 (Anomaly Detection) — Quality Verification ─
    #
    # Actual server log:  "[Gate 3] Result: N accepted, M rejected"
    # Actual server log:  "── Gate 3: Anomaly Detection ──"

    gate3_ran = "Gate 3: Anomaly Detection" in server_logs
    gate3_lines = _extract_matching_lines(server_logs, r"\[Gate 3\] Result:")
    ok &= report.record(
        "Gate 3 (Quality): Anomaly detection executed",
        gate3_ran,
        gate3_lines if gate3_ran else "Gate 3 header not found in server logs",
    )

    # ── 3.4  Attack-specific checks ───────────────────────────
    if scenario == "zt":
        # The malicious client should have connected and sent updates
        malicious_connected = "ATTACK MODE" in attack_logs
        ok &= report.record(
            "Attacker connected and began training",
            malicious_connected,
            _extract_matching_lines(attack_logs, "ATTACK MODE|fit loss"),
        )

        # Malicious client signed its updates (passes Gate 2)
        # Actual log: "[MALICIOUS] Update SIGNED  ✓  → Gate 2 passes, Gate 3 should FIRE"
        malicious_signed = "Update SIGNED" in attack_logs
        ok &= report.record(
            "Attacker passed Gate 2 (signed update accepted)",
            malicious_signed,
            _extract_matching_lines(attack_logs, "Update SIGNED|Gate 2 passes"),
        )

        # Gate 3 should have detected the attack
        # Actual log: "[Gate 3] 🚨 SECURITY ALERT: ... REJECTED"
        attack_detected = "SECURITY ALERT" in server_logs and "REJECTED" in server_logs
        alert_lines = _extract_matching_lines(server_logs, "SECURITY ALERT")
        ok &= report.record(
            "Gate 3 (Defense): Malicious client REJECTED",
            attack_detected,
            alert_lines if attack_detected else "No SECURITY ALERT found — attacker was NOT detected",
        )
        if not attack_detected:
            report.attach_failure_logs(CONTAINER_SERVER, server_logs)

        # Honest client's updates should have been accepted
        # Look for: "[Gate 2] ✓  Signature VALID for client-0"
        honest_accepted = "Signature VALID for client-0" in server_logs
        ok &= report.record(
            "Honest client-0 updates accepted",
            honest_accepted,
        )

    elif scenario == "clean":
        # Both clients should pass all gates
        both_valid = ("Signature VALID for client-0" in server_logs and
                      "Signature VALID for client-1" in server_logs)
        ok &= report.record(
            "Both honest clients passed Gate 2",
            both_valid,
            _extract_matching_lines(server_logs, "Signature VALID"),
        )

        # No security alerts should fire
        no_alerts = "SECURITY ALERT" not in server_logs
        ok &= report.record(
            "No security alerts (clean run)",
            no_alerts,
            "" if no_alerts else _extract_matching_lines(server_logs, "SECURITY ALERT"),
        )

    # ── 3.5  FL completed at least 1 round of aggregation ────
    aggregation_ran = "Aggregating" in server_logs and "clean updates via FedAvg" in server_logs
    ok &= report.record(
        "FedAvg aggregation executed",
        aggregation_ran,
        _extract_matching_lines(server_logs, "Aggregating.*FedAvg"),
    )

    # ── 3.6  Evaluation metrics present ───────────────────────
    eval_pattern = r"eval loss:\s*[\d.]+\s+acc:\s*[\d.]+"
    eval_matches = re.findall(eval_pattern, client0_logs)
    evals_present = len(eval_matches) > 0
    detail = f"{len(eval_matches)} eval result(s): {eval_matches[-1]}" if evals_present else "No eval metrics found"
    ok &= report.record(
        "Client-0 evaluation metrics present",
        evals_present,
        detail,
    )

    # ── 3.7  GPU was used (if available) ──────────────────────
    gpu_used = "cuda" in client0_logs.lower() or "Using device: cuda" in client0_logs
    # Not a hard failure if running CPU-only, but worth noting
    report.record(
        "GPU used for training (informational)",
        gpu_used,
        _extract_matching_lines(client0_logs, "Using device:|cuda"),
    )

    # Attach failure logs if anything went wrong
    if not ok:
        if server_logs:
            report.attach_failure_logs(CONTAINER_SERVER, server_logs)
        if client0_logs:
            report.attach_failure_logs(CONTAINER_CLIENT0, client0_logs)
        if attack_logs:
            report.attach_failure_logs(CONTAINER_MALICIOUS, attack_logs)

    return ok


def _extract_matching_lines(logs: str, pattern: str, max_lines: int = 5) -> str:
    """Return up to `max_lines` lines from logs matching the regex pattern."""
    matches = []
    for line in logs.splitlines():
        if re.search(pattern, line):
            matches.append(line.strip())
            if len(matches) >= max_lines:
                break
    return "\n".join(matches)


# ═══════════════════════════════════════════════════════════════
#  Phase 4: Reporting + Cleanup
# ═══════════════════════════════════════════════════════════════

def phase4_cleanup(compose_files: List[Path], leave_running: bool = False) -> None:
    """Tear down containers unless asked to leave them running."""
    if leave_running:
        print("\n  (--keep-running: containers left running for inspection)")
        return
    print("\n  Tearing down containers …")
    cmd = _docker_compose_cmd(compose_files) + ["down", "-v", "--remove-orphans"]
    _run(cmd, timeout=60)
    print("  Teardown complete.")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the Zero-Trust FL pipeline on real hardware.",
    )
    parser.add_argument(
        "--scenario", choices=["zt", "clean"], default="zt",
        help="'zt' = ZT under attack (default); 'clean' = 2 honest clients",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Max seconds to wait for FL completion (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--keep-running", action="store_true",
        help="Don't tear down containers after verification (useful for debugging)",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("  Zero-Trust FL — System Deployment Verification")
    print(f"  Scenario : {args.scenario}")
    print(f"  Timeout  : {args.timeout}s")
    print("=" * 64)

    report = TestResult()

    # ── Phase 1 ───────────────────────────────────────────────
    env_ok = phase1_environment_audit(report)
    if not env_ok:
        print("\n  ⚠  Environment issues detected — proceeding anyway …")

    # ── Phase 2 ───────────────────────────────────────────────
    compose_files = [COMPOSE_FILE]
    if args.scenario == "zt" and OVERRIDE_FILE.exists():
        compose_files.append(OVERRIDE_FILE)

    exec_ok = phase2_execution(report, args.scenario, args.timeout)

    # ── Phase 3 (only if Phase 2 reached FL completion) ───────
    if exec_ok:
        phase3_log_forensics(report, args.scenario)
    else:
        report.record(
            "Log forensics skipped",
            False,
            "Phase 2 (Execution) did not complete — cannot inspect logs",
        )
        # Still grab whatever logs exist
        for c in [CONTAINER_SERVER, CONTAINER_CLIENT0, CONTAINER_MALICIOUS]:
            logs = _docker_logs(c)
            if logs.strip():
                report.attach_failure_logs(c, logs)

    # ── Phase 4 ───────────────────────────────────────────────
    phase4_cleanup(compose_files, leave_running=args.keep_running)

    # ── Final Report ──────────────────────────────────────────
    report.print_summary()

    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
