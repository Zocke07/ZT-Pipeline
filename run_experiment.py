#!/usr/bin/env python3
"""run_experiment.py – Comparative Federated Learning Security Experiment.

Runs two Docker-based FL experiments sequentially and produces results.json
with REAL metrics parsed from container logs.  No results are fabricated.

Experiment A  "Baseline Under Attack"
    Insecure FL (no mTLS, no signatures, no anomaly detection)
    1 honest client  +  1 label-flip attacker
    Expected: attack succeeds, accuracy degrades

Experiment B  "Zero-Trust Under Attack"
    Full ZT pipeline (mTLS + RSA-PSS signatures + anomaly detection)
    1 honest client  +  1 label-flip attacker  (same attack params)
    Expected: attacker detected by Gate 3, accuracy preserved

Usage:
    python run_experiment.py                 # run both experiments
    python run_experiment.py --exp a         # only Experiment A
    python run_experiment.py --exp b         # only Experiment B
    python run_experiment.py --timeout 900   # custom timeout (sec)

Output:
    results.json   in the project root directory
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths  (all relative to the project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

BASELINE_ATTACK_COMPOSE = PROJECT_ROOT / "baseline_experiment" / "docker-compose-baseline-attack.yml"
ZT_BASE_COMPOSE = PROJECT_ROOT / "docker-compose.yml"
ZT_OVERRIDE_COMPOSE = PROJECT_ROOT / "experiment_override.yml"

DEFAULT_TIMEOUT = 600  # 10 minutes per experiment


# ═══════════════════════════════════════════════════════════════
#  Docker Compose helpers
# ═══════════════════════════════════════════════════════════════

def _compose_cmd(compose_files: List[Path], project: str) -> List[str]:
    """Build the base docker compose command with -f and -p flags."""
    cmd = ["docker", "compose"]
    for f in compose_files:
        cmd.extend(["-f", str(f)])
    cmd.extend(["-p", project])
    return cmd


def compose_build(compose_files: List[Path], project: str) -> bool:
    """Build images.  Returns True on success."""
    cmd = _compose_cmd(compose_files, project) + ["build"]
    print(f"    Building images …")
    result = subprocess.run(
        cmd, capture_output=True, encoding="utf-8", errors="replace",
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print(f"    BUILD FAILED (exit {result.returncode})")
        _print_truncated(result.stderr or result.stdout, label="build output")
        return False
    print(f"    Build OK")
    return True


def compose_up(
    compose_files: List[Path], project: str, timeout: int,
) -> Tuple[int, str, str]:
    """Run `docker compose up --abort-on-container-exit`.

    Returns (returncode, stdout, stderr).
    """
    cmd = _compose_cmd(compose_files, project) + [
        "up", "--abort-on-container-exit",
    ]
    print(f"    Running experiment (timeout={timeout}s) …")
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, capture_output=True, encoding="utf-8", errors="replace",
            timeout=timeout, cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - t0
        print(f"    TIMEOUT after {elapsed:.0f}s")
        # Attempt graceful teardown after timeout
        compose_down(compose_files, project)
        return -1, (e.stdout or ""), (e.stderr or "")
    elapsed = time.perf_counter() - t0
    print(f"    Finished in {elapsed:.0f}s (exit code {result.returncode})")
    return result.returncode, result.stdout, result.stderr


def compose_logs(
    compose_files: List[Path], project: str, service: str,
) -> str:
    """Retrieve logs for a specific service (works on stopped containers)."""
    cmd = _compose_cmd(compose_files, project) + [
        "logs", "--no-log-prefix", "--no-color", service,
    ]
    result = subprocess.run(
        cmd, capture_output=True, encoding="utf-8", errors="replace",
        cwd=str(PROJECT_ROOT),
    )
    # Some docker versions put logs into stderr
    return (result.stdout or "") + (result.stderr or "")


def compose_down(compose_files: List[Path], project: str) -> None:
    """Tear down containers, networks, and volumes."""
    cmd = _compose_cmd(compose_files, project) + [
        "down", "-v", "--remove-orphans",
    ]
    subprocess.run(
        cmd, capture_output=True, encoding="utf-8", errors="replace",
        cwd=str(PROJECT_ROOT),
    )
    print(f"    Teardown complete")


# ═══════════════════════════════════════════════════════════════
#  Log parsing – extract REAL metrics from stdout/stderr
# ═══════════════════════════════════════════════════════════════

def parse_eval_metrics(logs: str) -> List[Dict[str, float]]:
    """Extract per-round evaluation accuracy and loss from client logs.

    Looks for the pattern printed by evaluate():
        → eval loss: 1.2345  acc: 0.5678
    or:
        [BASELINE MALICIOUS] eval loss: 1.2345  acc: 0.5678
    or:
        [MALICIOUS] eval loss: 1.2345  acc: 0.5678

    Returns a list of dicts, one per occurrence (round), e.g.:
        [{"loss": 1.2345, "accuracy": 0.5678}, ...]
    """
    pattern = r"eval loss:\s*([\d.]+)\s+acc:\s*([\d.]+)"
    matches = re.findall(pattern, logs)
    return [{"loss": float(l), "accuracy": float(a)} for l, a in matches]


def parse_fit_metrics(logs: str) -> List[float]:
    """Extract per-round training loss from client logs.

    Looks for:  → fit loss: 1.2345
    Returns list of loss values.
    """
    pattern = r"fit loss:\s*([\d.]+)"
    return [float(l) for l in re.findall(pattern, logs)]


def parse_gate2_results(logs: str) -> List[Dict[str, int]]:
    """Extract Gate 2 (signature verification) results from ZT server logs.

    Pattern: [Gate 2] Result: 1 passed, 1 rejected
    """
    pattern = r"\[Gate 2\] Result:\s*(\d+)\s*passed,\s*(\d+)\s*rejected"
    return [{"passed": int(p), "rejected": int(r)} for p, r in re.findall(pattern, logs)]


def parse_gate3_results(logs: str) -> List[Dict[str, int]]:
    """Extract Gate 3 (anomaly detection) results from ZT server logs.

    Pattern: [Gate 3] Result: 2 accepted, 0 rejected
    """
    pattern = r"\[Gate 3\] Result:\s*(\d+)\s*accepted,\s*(\d+)\s*rejected"
    return [{"accepted": int(a), "rejected": int(r)} for a, r in re.findall(pattern, logs)]


def parse_security_alerts(logs: str) -> List[str]:
    """Extract Gate 3 security alert lines from ZT server logs.

    Pattern: [Gate 3] ... SECURITY ALERT: client-X REJECTED ...
    """
    pattern = r"SECURITY ALERT:.*?(client-\S+)\s+REJECTED"
    return re.findall(pattern, logs)


def parse_aggregation_skips(logs: str) -> int:
    """Count rounds where aggregation was skipped (too few accepted).

    Pattern: skipping aggregation to prevent single-client model domination
    or:      All updates anomalous – skipping aggregation
    or:      No valid updates – skipping aggregation
    """
    patterns = [
        r"skipping aggregation",
        r"All updates anomalous",
        r"No valid updates",
    ]
    count = 0
    for p in patterns:
        count += len(re.findall(p, logs, re.IGNORECASE))
    return count


# ═══════════════════════════════════════════════════════════════
#  Experiment runners
# ═══════════════════════════════════════════════════════════════

def run_experiment_a(timeout: int) -> Dict[str, Any]:
    """Experiment A: Baseline Under Attack.

    Runs the insecure baseline with 1 honest + 1 malicious client.
    Measures honest client accuracy (should degrade due to unfiltered poison).
    """
    print("\n" + "=" * 64)
    print("  EXPERIMENT A: Baseline (Insecure) Under Attack")
    print("  1 honest client  +  1 label-flip attacker")
    print("  Server: vanilla FedAvg, no security gates")
    print("=" * 64)

    compose_files = [BASELINE_ATTACK_COMPOSE]
    project = "exp-a-baseline"
    result: Dict[str, Any] = {
        "experiment": "A",
        "description": "Baseline (insecure) FL under label-flip attack",
        "security_gates": "NONE",
        "status": "not_started",
    }

    # ── Pre-flight check ──────────────────────────────────────
    if not BASELINE_ATTACK_COMPOSE.exists():
        result["status"] = "error"
        result["error"] = f"Compose file not found: {BASELINE_ATTACK_COMPOSE}"
        return result

    # ── Build ─────────────────────────────────────────────────
    if not compose_build(compose_files, project):
        result["status"] = "build_failed"
        compose_down(compose_files, project)
        return result

    # ── Run ───────────────────────────────────────────────────
    t_start = time.perf_counter()
    rc, stdout, stderr = compose_up(compose_files, project, timeout)
    wall_time = time.perf_counter() - t_start
    combined_output = stdout + stderr

    result["exit_code"] = rc
    result["wall_time_seconds"] = round(wall_time, 1)

    # ── Collect per-service logs ──────────────────────────────
    server_logs = compose_logs(compose_files, project, "server")
    honest_logs = compose_logs(compose_files, project, "client-0")
    attack_logs = compose_logs(compose_files, project, "malicious")

    # ── Parse metrics ─────────────────────────────────────────
    honest_eval = parse_eval_metrics(honest_logs)
    honest_fit = parse_fit_metrics(honest_logs)
    attack_eval = parse_eval_metrics(attack_logs)
    attack_fit = parse_fit_metrics(attack_logs)

    result["honest_client"] = {
        "per_round_eval": honest_eval,
        "per_round_fit_loss": honest_fit,
        "final_accuracy": honest_eval[-1]["accuracy"] if honest_eval else None,
        "final_loss": honest_eval[-1]["loss"] if honest_eval else None,
    }
    result["malicious_client"] = {
        "per_round_eval": attack_eval,
        "per_round_fit_loss": attack_fit,
        "attack_mode": "label_flip",
        "local_epochs": 2,
    }
    result["security_events"] = {
        "gate2_rejections": "N/A (no Gate 2)",
        "gate3_rejections": "N/A (no Gate 3)",
        "malicious_updates_aggregated": "ALL (no filtering)",
    }

    # Determine outcome
    if honest_eval:
        result["status"] = "completed"
    elif rc == -1:
        result["status"] = "timeout"
    else:
        result["status"] = "completed_no_metrics"
        result["note"] = "Experiment finished but no eval metrics were parsed from logs"

    # ── Teardown ──────────────────────────────────────────────
    compose_down(compose_files, project)
    return result


def run_experiment_b(timeout: int) -> Dict[str, Any]:
    """Experiment B: Zero-Trust Under Attack.

    Runs the ZT pipeline with 1 honest + 1 malicious client.
    MIN_ACCEPTED=1 (via override) so honest updates are still aggregated
    after the attacker is rejected by Gate 3.
    """
    print("\n" + "=" * 64)
    print("  EXPERIMENT B: Zero-Trust Pipeline Under Attack")
    print("  1 honest client  +  1 label-flip attacker")
    print("  Server: ZeroTrustFedAvg (Gate 1 + Gate 2 + Gate 3)")
    print("=" * 64)

    compose_files = [ZT_BASE_COMPOSE, ZT_OVERRIDE_COMPOSE]
    project = "exp-b-zt"
    result: Dict[str, Any] = {
        "experiment": "B",
        "description": "Zero-Trust FL under label-flip attack",
        "security_gates": "Gate 1 (mTLS) + Gate 2 (RSA-PSS) + Gate 3 (Anomaly)",
        "status": "not_started",
    }

    # ── Pre-flight checks ─────────────────────────────────────
    missing = []
    for f in compose_files:
        if not f.exists():
            missing.append(str(f))
    # Also check that certs and signing keys exist
    for required in [
        PROJECT_ROOT / "certs" / "ca.crt",
        PROJECT_ROOT / "certs" / "server.crt",
        PROJECT_ROOT / "certs" / "server.key",
        PROJECT_ROOT / "certs" / "client-0.crt",
        PROJECT_ROOT / "certs" / "client-0.key",
        PROJECT_ROOT / "certs" / "client-1.crt",
        PROJECT_ROOT / "certs" / "client-1.key",
        PROJECT_ROOT / "signing_keys" / "client-0.private.pem",
        PROJECT_ROOT / "signing_keys" / "client-0.public.pem",
        PROJECT_ROOT / "signing_keys" / "client-1.private.pem",
        PROJECT_ROOT / "signing_keys" / "client-1.public.pem",
    ]:
        if not required.exists():
            missing.append(str(required))

    if missing:
        result["status"] = "error"
        result["error"] = f"Required files missing: {missing}"
        return result

    # ── Build ─────────────────────────────────────────────────
    if not compose_build(compose_files, project):
        result["status"] = "build_failed"
        compose_down(compose_files, project)
        return result

    # ── Run ───────────────────────────────────────────────────
    t_start = time.perf_counter()
    rc, stdout, stderr = compose_up(compose_files, project, timeout)
    wall_time = time.perf_counter() - t_start
    combined_output = stdout + stderr

    result["exit_code"] = rc
    result["wall_time_seconds"] = round(wall_time, 1)

    # ── Collect per-service logs ──────────────────────────────
    server_logs = compose_logs(compose_files, project, "server")
    honest_logs = compose_logs(compose_files, project, "client-0")
    attack_logs = compose_logs(compose_files, project, "malicious")

    # ── Parse honest client metrics ───────────────────────────
    honest_eval = parse_eval_metrics(honest_logs)
    honest_fit = parse_fit_metrics(honest_logs)

    result["honest_client"] = {
        "per_round_eval": honest_eval,
        "per_round_fit_loss": honest_fit,
        "final_accuracy": honest_eval[-1]["accuracy"] if honest_eval else None,
        "final_loss": honest_eval[-1]["loss"] if honest_eval else None,
    }

    # ── Parse malicious client metrics ────────────────────────
    attack_eval = parse_eval_metrics(attack_logs)
    attack_fit = parse_fit_metrics(attack_logs)

    result["malicious_client"] = {
        "per_round_eval": attack_eval,
        "per_round_fit_loss": attack_fit,
        "attack_mode": "label_flip",
        "local_epochs": 2,
    }

    # ── Parse security gate results ───────────────────────────
    gate2 = parse_gate2_results(server_logs)
    gate3 = parse_gate3_results(server_logs)
    alerts = parse_security_alerts(server_logs)
    skips = parse_aggregation_skips(server_logs)

    total_gate2_rejected = sum(r["rejected"] for r in gate2) if gate2 else 0
    total_gate3_rejected = sum(r["rejected"] for r in gate3) if gate3 else 0
    attacker_detected = any("client-1" in a for a in alerts)

    result["security_events"] = {
        "gate2_per_round": gate2,
        "gate3_per_round": gate3,
        "total_gate2_rejections": total_gate2_rejected,
        "total_gate3_rejections": total_gate3_rejected,
        "security_alerts": alerts,
        "aggregation_rounds_skipped": skips,
        "attacker_detected": attacker_detected,
    }

    # Determine outcome
    if honest_eval:
        result["status"] = "completed"
    elif rc == -1:
        result["status"] = "timeout"
    else:
        result["status"] = "completed_no_metrics"
        result["note"] = "Experiment finished but no eval metrics were parsed from logs"

    # ── Teardown ──────────────────────────────────────────────
    compose_down(compose_files, project)
    return result


# ═══════════════════════════════════════════════════════════════
#  Comparison summary  (computed from real data only)
# ═══════════════════════════════════════════════════════════════

def build_comparison(exp_a: Dict, exp_b: Dict) -> Dict[str, Any]:
    """Build a side-by-side comparison from the two experiment results.

    Every value is derived from parsed metrics.  If a metric is missing,
    it is reported as null — never fabricated.
    """
    a_acc = exp_a.get("honest_client", {}).get("final_accuracy")
    b_acc = exp_b.get("honest_client", {}).get("final_accuracy")

    comparison: Dict[str, Any] = {
        "baseline_final_accuracy": a_acc,
        "zt_final_accuracy": b_acc,
        "accuracy_difference": round(b_acc - a_acc, 4) if (a_acc is not None and b_acc is not None) else None,
        "baseline_attack_blocked": False,
        "zt_attack_blocked": exp_b.get("security_events", {}).get("attacker_detected", False),
        "zt_total_gate2_rejections": exp_b.get("security_events", {}).get("total_gate2_rejections"),
        "zt_total_gate3_rejections": exp_b.get("security_events", {}).get("total_gate3_rejections"),
    }

    # Narrative conclusion derived from data
    conclusions = []
    if a_acc is not None and b_acc is not None:
        if b_acc > a_acc:
            conclusions.append(
                f"ZT accuracy ({b_acc:.4f}) > Baseline accuracy ({a_acc:.4f}): "
                f"Zero-Trust preserved model quality under attack."
            )
        elif b_acc == a_acc:
            conclusions.append("Both experiments yielded the same accuracy.")
        else:
            conclusions.append(
                f"Baseline accuracy ({a_acc:.4f}) > ZT accuracy ({b_acc:.4f}): "
                f"unexpected — investigate logs."
            )

    if exp_b.get("security_events", {}).get("attacker_detected"):
        conclusions.append(
            "Gate 3 detected and rejected the malicious client's updates."
        )
    else:
        conclusions.append(
            "Gate 3 did NOT flag the attacker — investigate logs and thresholds."
        )

    comparison["conclusions"] = conclusions
    return comparison


# ═══════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════

def _print_truncated(text: str, label: str = "output", max_lines: int = 30) -> None:
    """Print the last `max_lines` of text for debugging."""
    lines = text.strip().splitlines()
    if len(lines) > max_lines:
        print(f"    [{label}: showing last {max_lines} of {len(lines)} lines]")
        lines = lines[-max_lines:]
    for line in lines:
        print(f"      {line}")


def check_docker() -> bool:
    """Verify docker and docker compose are available."""
    try:
        r = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True, encoding="utf-8", errors="replace",
        )
        if r.returncode == 0:
            version_line = r.stdout.strip().splitlines()[0] if r.stdout.strip() else "unknown"
            print(f"  Docker Compose: {version_line}")
            return True
    except FileNotFoundError:
        pass
    print("  ERROR: 'docker compose' not found. Is Docker installed and in PATH?")
    return False


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run comparative FL security experiment (Baseline vs Zero-Trust).",
    )
    parser.add_argument(
        "--exp", choices=["a", "b", "both"], default="both",
        help="Which experiment(s) to run (default: both)",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Timeout per experiment in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--output", type=str, default="results.json",
        help="Output file path (default: results.json)",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("  Federated Learning Security — Comparative Experiment")
    print("=" * 64)
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"  Timestamp : {timestamp}")
    print(f"  Timeout   : {args.timeout}s per experiment")
    print(f"  Output    : {args.output}")

    if not check_docker():
        sys.exit(1)

    results: Dict[str, Any] = {
        "experiment_timestamp": timestamp,
        "timeout_seconds": args.timeout,
        "project_root": str(PROJECT_ROOT),
    }

    # ── Experiment A ──────────────────────────────────────────
    if args.exp in ("a", "both"):
        results["experiment_a_baseline_attack"] = run_experiment_a(args.timeout)
    else:
        results["experiment_a_baseline_attack"] = {"status": "skipped"}

    # ── Experiment B ──────────────────────────────────────────
    if args.exp in ("b", "both"):
        results["experiment_b_zt_attack"] = run_experiment_b(args.timeout)
    else:
        results["experiment_b_zt_attack"] = {"status": "skipped"}

    # ── Comparison ────────────────────────────────────────────
    exp_a = results.get("experiment_a_baseline_attack", {})
    exp_b = results.get("experiment_b_zt_attack", {})
    if exp_a.get("status") == "completed" and exp_b.get("status") == "completed":
        results["comparison"] = build_comparison(exp_a, exp_b)
    else:
        results["comparison"] = {
            "note": "Comparison unavailable — one or both experiments did not complete successfully.",
            "experiment_a_status": exp_a.get("status"),
            "experiment_b_status": exp_b.get("status"),
        }

    # ── Write results ─────────────────────────────────────────
    output_path = PROJECT_ROOT / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 64)
    print(f"  RESULTS WRITTEN TO: {output_path}")
    print("=" * 64)

    # Print quick summary
    if "comparison" in results and "conclusions" in results["comparison"]:
        print("\n  Summary:")
        for c in results["comparison"]["conclusions"]:
            print(f"    - {c}")

    a_acc = exp_a.get("honest_client", {}).get("final_accuracy")
    b_acc = exp_b.get("honest_client", {}).get("final_accuracy")
    if a_acc is not None:
        print(f"\n  Exp A (Baseline) final accuracy : {a_acc:.4f}")
    if b_acc is not None:
        print(f"  Exp B (ZT)       final accuracy : {b_acc:.4f}")

    print()


if __name__ == "__main__":
    main()
