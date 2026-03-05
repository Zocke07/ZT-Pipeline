#!/usr/bin/env python3
"""Docker-orchestrated experiment runner for the ZT-Pipeline thesis.

Generates credentials, builds Docker images, runs multi-seed experiments
across gate-ablation configurations using ``docker compose``, and produces
structured JSON results, CSV summaries, and matplotlib plots.

Usage examples
--------------
# Quick 2-seed smoke test (ZT, all gates, 2 clients, 1 attacker)
python run_experiments.py --seeds 42 123 --num-rounds 5 --num-clients 2

# Full thesis configuration matrix (3 seeds x 4 gate configs)
python run_experiments.py --preset thesis

# Custom: 5 clients, 2 attackers, noise attack, Gate 1+2 only
python run_experiments.py --seeds 42 123 456 \
    --num-clients 5 --num-attackers 2 \
    --attack-type noise --noise-scale 5.0 \
    --enable-gate1 --enable-gate2 --no-gate3

See ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from tracking.mlflow_logger import ExperimentTracker, NullTracker

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """All parameters for one experiment (may have multiple seeds)."""
    label: str = "experiment"
    seeds: List[int] = field(default_factory=lambda: [42])
    num_rounds: int = 20
    num_clients: int = 2
    num_attackers: int = 1
    attack_type: str = "label_flip"
    noise_scale: float = 5.0
    poison_scale: float = 10.0
    source_label: int = 5
    target_label: int = 3
    enable_gate1: bool = True
    enable_gate2: bool = True
    enable_gate3: bool = True
    z_threshold: float = 2.0
    min_accepted: int = 1
    local_epochs_malicious: int = 2
    timeout: int = 1800          # seconds per seed run
    gpu: bool = True
    # Phase 2 additions
    aggregation_method: str = "fedavg"        # fedavg | krum | multi-krum
    dirichlet_alpha: Optional[float] = None   # None → IID; float → non-IID
    threshold_values: Optional[List[float]] = None  # for sweep mode


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
STAGING_ROOT = PROJECT_ROOT / "experiment_staging"
DATA_DIR     = PROJECT_ROOT / "data"

ZT_IMAGE_TAG       = "zt-fl-experiment"
BASELINE_IMAGE_TAG = "baseline-fl-experiment"


# ---------------------------------------------------------------------------
# Credential generation
# ---------------------------------------------------------------------------

def generate_credentials(num_clients: int, cert_dir: Path, key_dir: Path) -> None:
    """Generate mTLS certs + signing keys for *num_clients*."""
    from generate_keys import generate_all
    generate_all(cert_dir=cert_dir, key_dir=key_dir, num_clients=num_clients)


# ---------------------------------------------------------------------------
# Docker image building
# ---------------------------------------------------------------------------

def build_images(cfg: ExperimentConfig) -> None:
    """Build the required Docker image(s) once."""
    ctx = str(PROJECT_ROOT).replace("\\", "/")

    if cfg.enable_gate1:
        print(f"\n[build] Building ZT image  ({ZT_IMAGE_TAG}) ...")
        _run([
            "docker", "build",
            "-t", ZT_IMAGE_TAG,
            "-f", (PROJECT_ROOT / "Dockerfile").as_posix(),
            ctx,
        ])
    else:
        print(f"\n[build] Building Baseline image  ({BASELINE_IMAGE_TAG}) ...")
        _run([
            "docker", "build",
            "-t", BASELINE_IMAGE_TAG,
            "-f", (PROJECT_ROOT / "baseline_experiment" / "Dockerfile").as_posix(),
            ctx,
        ])

    print("[build] Image build complete.\n")


# ---------------------------------------------------------------------------
# Docker Compose YAML generation
# ---------------------------------------------------------------------------

def _p(path: Path) -> str:
    """Path -> Docker-friendly forward-slash string."""
    return str(path).replace("\\", "/")


def _gpu_block(gpu: bool) -> str:
    if not gpu:
        return ""
    return """\
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]"""


def generate_compose(
    cfg: ExperimentConfig,
    seed: int,
    run_dir: Path,
    cert_dir: Path,
    key_dir: Path,
) -> Path:
    """Generate a docker-compose.yml for a single seed run."""
    result_dir = run_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    use_zt = cfg.enable_gate1
    image = ZT_IMAGE_TAG if use_zt else BASELINE_IMAGE_TAG
    dockerfile = "Dockerfile" if use_zt else "baseline_experiment/Dockerfile"
    server_cmd = "python server.py" if use_zt else "python baseline_server.py"
    client_cmd = "python client.py" if use_zt else "python baseline_client.py"
    malicious_cmd = ("python client_malicious.py" if use_zt
                     else "python baseline_malicious_client.py")

    ctx = _p(PROJECT_ROOT)
    data = _p(DATA_DIR)
    certs = _p(cert_dir)
    keys = _p(key_dir)
    results = _p(result_dir)
    gpu = _gpu_block(cfg.gpu)

    num_honest = cfg.num_clients - cfg.num_attackers
    services: list = []

    # ---- data-init ----
    services.append(f"""\
  data-init:
    build:
      context: {ctx}
      dockerfile: {dockerfile}
    image: {image}
    command: >
      python -c "from torchvision import datasets, transforms;
      datasets.CIFAR10('/data', train=True,  download=True,
        transform=transforms.ToTensor());
      datasets.CIFAR10('/data', train=False, download=True,
        transform=transforms.ToTensor());
      print('CIFAR-10 ready.')"
    volumes:
      - {data}:/data
    restart: "no"
""")

    # ---- server ----
    srv_env_lines = [
        f"      - NUM_CLIENTS={cfg.num_clients}",
        f"      - MIN_CLIENTS={cfg.num_clients}",
        f"      - NUM_ROUNDS={cfg.num_rounds}",
        f"      - SEED={seed}",
        f"      - RESULTS_DIR=/results",
        f"      - AGGREGATION_METHOD={cfg.aggregation_method}",
        f"      - NUM_ATTACKERS={cfg.num_attackers}",
    ]
    srv_vol_lines = [
        f"      - {results}:/results",
    ]
    if use_zt:
        srv_env_lines += [
            f"      - CERT_DIR=/certs",
            f"      - SIGNING_KEY_DIR=/signing_keys",
            f"      - MIN_ACCEPTED={cfg.min_accepted}",
            f"      - ANOMALY_Z_THRESHOLD={cfg.z_threshold}",
            f"      - ENABLE_GATE2={'true' if cfg.enable_gate2 else 'false'}",
            f"      - ENABLE_GATE3={'true' if cfg.enable_gate3 else 'false'}",
        ]
        srv_vol_lines += [
            f"      - {certs}:/certs:ro",
        ]
        for i in range(cfg.num_clients):
            srv_vol_lines.append(
                f"      - {keys}/client-{i}.public.pem:"
                f"/signing_keys/client-{i}.public.pem:ro"
            )

    srv_env = "\n".join(srv_env_lines)
    srv_vol = "\n".join(srv_vol_lines)

    services.append(f"""\
  server:
    build:
      context: {ctx}
      dockerfile: {dockerfile}
    image: {image}
    command: {server_cmd}
    environment:
{srv_env}
    volumes:
{srv_vol}
{gpu}
""")

    # ---- honest clients ----
    for i in range(num_honest):
        cl_env_lines = [
            f"      - CLIENT_ID={i}",
            f"      - NUM_CLIENTS={cfg.num_clients}",
            f"      - SERVER_ADDRESS=server:8080",
            f"      - SEED={seed}",
        ]
        if cfg.dirichlet_alpha is not None:
            cl_env_lines.append(f"      - DIRICHLET_ALPHA={cfg.dirichlet_alpha}")
        cl_vol_lines = [
            f"      - {data}:/data",
        ]
        if use_zt:
            cl_env_lines += [
                f"      - CERT_DIR=/certs",
                f"      - SIGNING_KEY_DIR=/signing_keys",
            ]
            cl_vol_lines += [
                f"      - {certs}:/certs:ro",
                (f"      - {keys}/client-{i}.private.pem:"
                 f"/signing_keys/client-{i}.private.pem:ro"),
            ]
        cl_env = "\n".join(cl_env_lines)
        cl_vol = "\n".join(cl_vol_lines)

        services.append(f"""\
  client-{i}:
    build:
      context: {ctx}
      dockerfile: {dockerfile}
    image: {image}
    command: >
      sh -c "sleep 5 && {client_cmd}"
    environment:
{cl_env}
    volumes:
{cl_vol}
    depends_on:
      server:
        condition: service_started
      data-init:
        condition: service_completed_successfully
{gpu}
""")

    # ---- malicious clients ----
    for j in range(cfg.num_attackers):
        cid = num_honest + j
        ml_env_lines = [
            f"      - CLIENT_ID={cid}",
            f"      - NUM_CLIENTS={cfg.num_clients}",
            f"      - SERVER_ADDRESS=server:8080",
            f"      - SEED={seed}",
            f"      - ATTACK_MODE={cfg.attack_type}",
            f"      - LOCAL_EPOCHS={cfg.local_epochs_malicious}",
        ]
        if cfg.attack_type == "noise":
            ml_env_lines.append(f"      - NOISE_SCALE={cfg.noise_scale}")
        if cfg.attack_type == "scale":
            ml_env_lines.append(f"      - POISON_SCALE={cfg.poison_scale}")
        if cfg.attack_type == "targeted":
            ml_env_lines.append(f"      - SOURCE_LABEL={cfg.source_label}")
            ml_env_lines.append(f"      - TARGET_LABEL={cfg.target_label}")
        if cfg.dirichlet_alpha is not None:
            ml_env_lines.append(f"      - DIRICHLET_ALPHA={cfg.dirichlet_alpha}")

        ml_vol_lines = [
            f"      - {data}:/data",
        ]
        if use_zt:
            ml_env_lines += [
                f"      - CERT_DIR=/certs",
                f"      - SIGNING_KEY_DIR=/signing_keys",
            ]
            ml_vol_lines += [
                f"      - {certs}:/certs:ro",
                (f"      - {keys}/client-{cid}.private.pem:"
                 f"/signing_keys/client-{cid}.private.pem:ro"),
            ]
        ml_env = "\n".join(ml_env_lines)
        ml_vol = "\n".join(ml_vol_lines)

        services.append(f"""\
  malicious-{j}:
    build:
      context: {ctx}
      dockerfile: {dockerfile}
    image: {image}
    command: >
      sh -c "sleep 12 && {malicious_cmd}"
    environment:
{ml_env}
    volumes:
{ml_vol}
    depends_on:
      server:
        condition: service_started
      data-init:
        condition: service_completed_successfully
{gpu}
""")

    # ---- assemble ----
    yaml_content = "services:\n" + "\n".join(services)
    compose_path = run_dir / "docker-compose.generated.yml"
    compose_path.write_text(yaml_content, encoding="utf-8")
    return compose_path


# ---------------------------------------------------------------------------
# Running & teardown
# ---------------------------------------------------------------------------

def _run(cmd: list, timeout: int | None = None, capture: bool = False, **kw):
    """Run a subprocess, print on failure."""
    try:
        return subprocess.run(
            cmd, check=True, timeout=timeout,
            capture_output=capture, text=True, **kw,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Command failed: {' '.join(cmd)}")
        if exc.stdout:
            print(exc.stdout[-2000:])
        if exc.stderr:
            print(exc.stderr[-2000:])
        raise
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Command timed out after {timeout}s: {' '.join(cmd)}")
        raise


def run_seed(
    cfg: ExperimentConfig,
    seed: int,
    compose_path: Path,
    project_name: str,
) -> Dict[str, Any]:
    """Run docker compose up for one seed, collect metrics."""
    print(f"\n{'='*60}")
    print(f"  SEED {seed} -- {cfg.label}")
    print(f"{'='*60}")

    compose_str = str(compose_path).replace("\\", "/")

    try:
        # Start all services in detached mode (background).
        # Using --abort-on-container-exit would kill everything when
        # data-init exits after downloading CIFAR-10, so we use -d
        # and explicitly wait for the server to finish.
        _run(
            ["docker", "compose",
             "-f", compose_str,
             "-p", project_name,
             "up", "--build", "-d"],
            timeout=300,  # build + start timeout
        )

        # Wait for the server container to exit (it finishes after all
        # FL rounds complete, then writes metrics.json and exits).
        server_container = f"{project_name}-server-1"
        print(f"[run] Waiting for {server_container} to finish ...")
        result = subprocess.run(
            ["docker", "wait", server_container],
            timeout=cfg.timeout,
            capture_output=True, text=True,
        )
        exit_code = result.stdout.strip()
        print(f"[run] Server exited with code {exit_code}")

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"[WARN] Seed {seed}: {exc}")
    finally:
        # Always tear down containers (keep data)
        try:
            # Collect server logs before teardown for debugging
            try:
                logs = subprocess.run(
                    ["docker", "compose",
                     "-f", compose_str,
                     "-p", project_name,
                     "logs", "server", "--tail", "30"],
                    capture_output=True, text=True, timeout=30,
                )
                if logs.stdout:
                    print(f"[logs] Server tail:\n{logs.stdout[-1500:]}")
            except Exception:
                pass

            _run(
                ["docker", "compose",
                 "-f", compose_str,
                 "-p", project_name,
                 "down", "--remove-orphans"],
                timeout=120,
            )
        except Exception:
            pass

    # ---- read metrics ----
    metrics_path = compose_path.parent / "results" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        print(f"[OK] Seed {seed}: read {len(metrics.get('round_metrics', []))} "
              f"rounds of metrics")
        return metrics
    else:
        print(f"[WARN] Seed {seed}: {metrics_path} not found")
        return {"seed": seed, "error": "metrics_not_found", "round_metrics": []}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_seeds(all_runs: List[Dict], cfg: Optional[ExperimentConfig] = None) -> Dict[str, Any]:
    """Compute mean +/- std of accuracy and loss across seeds.

    Also computes formal security evaluation metrics when attacker info
    is available:
      - TPR  (True Positive Rate / Detection Rate)
      - FPR  (False Positive Rate)
      - Precision, Recall
    """
    import numpy as np

    # Derive attacker / honest client IDs from config
    num_attackers = cfg.num_attackers if cfg else 0
    num_clients = cfg.num_clients if cfg else 0
    num_honest = num_clients - num_attackers
    attacker_cids = set(str(i) for i in range(num_honest, num_clients))
    honest_cids = set(str(i) for i in range(num_honest))

    round_data: Dict[int, Dict[str, list]] = {}
    for run in all_runs:
        for rm in run.get("round_metrics", []):
            r = rm["round"]
            if r not in round_data:
                round_data[r] = {"accuracy": [], "loss": [],
                                 "gate3_rejected": [],
                                 "gate3_rejected_cids": [],
                                 "aggregation_skipped": []}
            if "global_accuracy" in rm:
                round_data[r]["accuracy"].append(rm["global_accuracy"])
            if "global_loss" in rm:
                round_data[r]["loss"].append(rm["global_loss"])
            round_data[r]["gate3_rejected"].append(rm.get("gate3_rejected", 0))
            round_data[r]["gate3_rejected_cids"].append(
                rm.get("gate3_rejected_cids", []))
            round_data[r]["aggregation_skipped"].append(
                1 if rm.get("aggregation_skipped") else 0)

    summary_rounds = []
    all_tp, all_fp, all_fn, all_tn = 0, 0, 0, 0

    for r in sorted(round_data.keys()):
        d = round_data[r]
        entry: Dict[str, Any] = {"round": r}
        if d["accuracy"]:
            entry["accuracy_mean"] = float(np.mean(d["accuracy"]))
            entry["accuracy_std"] = float(np.std(d["accuracy"]))
        if d["loss"]:
            entry["loss_mean"] = float(np.mean(d["loss"]))
            entry["loss_std"] = float(np.std(d["loss"]))
        entry["gate3_rejected_mean"] = float(np.mean(d["gate3_rejected"]))
        entry["agg_skipped_frac"] = float(np.mean(d["aggregation_skipped"]))

        # Per-round detection metrics (averaged over seeds)
        if attacker_cids and d["gate3_rejected_cids"]:
            round_tp, round_fp, round_fn, round_tn = [], [], [], []
            for rej_cids in d["gate3_rejected_cids"]:
                rej_set = set(str(c) for c in rej_cids)
                tp = len(rej_set & attacker_cids)
                fp = len(rej_set & honest_cids)
                fn = len(attacker_cids - rej_set)
                tn = len(honest_cids - rej_set)
                round_tp.append(tp)
                round_fp.append(fp)
                round_fn.append(fn)
                round_tn.append(tn)
            entry["tp_mean"] = float(np.mean(round_tp))
            entry["fp_mean"] = float(np.mean(round_fp))
            entry["fn_mean"] = float(np.mean(round_fn))
            entry["tn_mean"] = float(np.mean(round_tn))
            all_tp += sum(round_tp)
            all_fp += sum(round_fp)
            all_fn += sum(round_fn)
            all_tn += sum(round_tn)

        summary_rounds.append(entry)

    final_accs = []
    for run in all_runs:
        rms = run.get("round_metrics", [])
        if rms:
            last = rms[-1]
            if "global_accuracy" in last:
                final_accs.append(last["global_accuracy"])

    result = {
        "num_seeds": len(all_runs),
        "final_accuracy_mean": float(np.mean(final_accs)) if final_accs else None,
        "final_accuracy_std": float(np.std(final_accs)) if final_accs else None,
        "round_summary": summary_rounds,
    }

    # Aggregate detection metrics over all rounds × seeds
    if all_tp + all_fp + all_fn + all_tn > 0:
        tpr = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
        fpr = all_fp / (all_fp + all_tn) if (all_fp + all_tn) > 0 else 0.0
        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
        recall = tpr  # same definition
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)
        result["detection_metrics"] = {
            "total_tp": all_tp,
            "total_fp": all_fp,
            "total_fn": all_fn,
            "total_tn": all_tn,
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        }
        print(f"[metrics] Detection: TPR={tpr:.4f}  FPR={fpr:.4f}  "
              f"Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")

    return result


# ---------------------------------------------------------------------------
# Output: CSV + plots
# ---------------------------------------------------------------------------

def save_csv(summary: Dict, out_dir: Path) -> Path:
    csv_path = out_dir / "summary.csv"
    rows = summary.get("round_summary", [])
    if not rows:
        return csv_path
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[output] CSV written: {csv_path}")
    return csv_path


def save_plots(
    all_runs: List[Dict],
    summary: Dict,
    out_dir: Path,
    label: str,
) -> None:
    """Generate accuracy & loss plots with per-seed traces + mean."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[output] matplotlib not installed -- skipping plots")
        return

    rounds_summary = summary.get("round_summary", [])
    if not rounds_summary:
        return

    xs = [r["round"] for r in rounds_summary]

    # -- Accuracy --
    fig, ax = plt.subplots(figsize=(10, 6))
    for run in all_runs:
        rms = run.get("round_metrics", [])
        seed = run.get("seed", "?")
        r_xs = [rm["round"] for rm in rms if "global_accuracy" in rm]
        r_ys = [rm["global_accuracy"] for rm in rms if "global_accuracy" in rm]
        if r_xs:
            ax.plot(r_xs, r_ys, alpha=0.3, linewidth=1, label=f"seed {seed}")
    means = [r.get("accuracy_mean") for r in rounds_summary]
    stds = [r.get("accuracy_std", 0) for r in rounds_summary]
    if any(m is not None for m in means):
        valid = [(x, m, s) for x, m, s in zip(xs, means, stds) if m is not None]
        if valid:
            vx, vm, vs = zip(*valid)
            vm, vs = np.array(vm), np.array(vs)
            ax.plot(vx, vm, "k-", linewidth=2, label="mean")
            ax.fill_between(vx, vm - vs, vm + vs, alpha=0.15, color="black")
    ax.set_xlabel("Round")
    ax.set_ylabel("Global Accuracy")
    ax.set_title(f"Accuracy -- {label}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy.png", dpi=150)
    plt.close(fig)

    # -- Loss --
    fig, ax = plt.subplots(figsize=(10, 6))
    for run in all_runs:
        rms = run.get("round_metrics", [])
        seed = run.get("seed", "?")
        r_xs = [rm["round"] for rm in rms if "global_loss" in rm]
        r_ys = [rm["global_loss"] for rm in rms if "global_loss" in rm]
        if r_xs:
            ax.plot(r_xs, r_ys, alpha=0.3, linewidth=1, label=f"seed {seed}")
    means = [r.get("loss_mean") for r in rounds_summary]
    stds = [r.get("loss_std", 0) for r in rounds_summary]
    if any(m is not None for m in means):
        valid = [(x, m, s) for x, m, s in zip(xs, means, stds) if m is not None]
        if valid:
            vx, vm, vs = zip(*valid)
            vm, vs = np.array(vm), np.array(vs)
            ax.plot(vx, vm, "k-", linewidth=2, label="mean")
            ax.fill_between(vx, vm - vs, vm + vs, alpha=0.15, color="black")
    ax.set_xlabel("Round")
    ax.set_ylabel("Global Loss")
    ax.set_title(f"Loss -- {label}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss.png", dpi=150)
    plt.close(fig)

    print(f"[output] Plots saved in {out_dir}/")

    # -- Detection metrics per round (TPR/FPR) --
    tps = [r.get("tp_mean") for r in rounds_summary]
    fps = [r.get("fp_mean") for r in rounds_summary]
    if any(t is not None for t in tps):
        fns = [r.get("fn_mean", 0) for r in rounds_summary]
        tns = [r.get("tn_mean", 0) for r in rounds_summary]
        tpr_vals, fpr_vals = [], []
        for tp, fp, fn, tn in zip(tps, fps, fns, tns):
            if tp is None:
                tpr_vals.append(None)
                fpr_vals.append(None)
            else:
                tpr_vals.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                fpr_vals.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        fig, ax = plt.subplots(figsize=(10, 5))
        valid_tpr = [(x, v) for x, v in zip(xs, tpr_vals) if v is not None]
        valid_fpr = [(x, v) for x, v in zip(xs, fpr_vals) if v is not None]
        if valid_tpr:
            ax.plot(*zip(*valid_tpr), "g-o", linewidth=2, label="TPR (Detection Rate)")
        if valid_fpr:
            ax.plot(*zip(*valid_fpr), "r-s", linewidth=2, label="FPR (False Alarm)")
        ax.set_xlabel("Round")
        ax.set_ylabel("Rate")
        ax.set_title(f"Detection Performance -- {label}")
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "detection_metrics.png", dpi=150)
        plt.close(fig)
        print(f"[output] Detection plot saved: {out_dir / 'detection_metrics.png'}")


def save_detection_csv(summary: Dict, out_dir: Path) -> None:
    """Save detection metrics (TPR/FPR/Precision/Recall) to CSV."""
    dm = summary.get("detection_metrics")
    if not dm:
        return
    csv_path = out_dir / "detection_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dm.keys()))
        w.writeheader()
        w.writerow(dm)
    print(f"[output] Detection CSV: {csv_path}")


def save_convergence_comparison(
    all_summaries: List[Dict],
    output_root: Path,
) -> None:
    """Plot accuracy convergence curves for multiple configs on one chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#d35f5f", "#e8a838", "#56b4e9", "#009e73",
              "#cc79a7", "#0072b2", "#d55e00", "#999999"]

    for idx, s in enumerate(all_summaries):
        rs = s.get("round_summary", [])
        if not rs:
            continue
        label = s.get("label", f"config-{idx}")
        xs = [r["round"] for r in rs]
        means = [r.get("accuracy_mean") for r in rs]
        stds = [r.get("accuracy_std", 0) for r in rs]
        valid = [(x, m, sd) for x, m, sd in zip(xs, means, stds) if m is not None]
        if not valid:
            continue
        vx, vm, vs = zip(*valid)
        vm, vs = np.array(vm), np.array(vs)
        c = colors[idx % len(colors)]
        ax.plot(vx, vm, "-o", color=c, linewidth=2, markersize=3,
                label=label.replace("_", " "))
        ax.fill_between(vx, vm - vs, vm + vs, alpha=0.1, color=c)

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Global Accuracy", fontsize=12)
    ax.set_title("Convergence Comparison Across Configurations", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(output_root / "convergence_comparison.png", dpi=150)
    plt.close(fig)
    print(f"[output] Convergence comparison: {output_root / 'convergence_comparison.png'}")


def save_threshold_sweep_plots(
    sweep_results: List[Dict],
    output_root: Path,
) -> None:
    """Plot accuracy and detection metrics vs Z-score threshold."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    thresholds = [r["z_threshold"] for r in sweep_results]
    accs = [r.get("final_accuracy_mean", 0) or 0 for r in sweep_results]
    tprs, fprs = [], []
    for r in sweep_results:
        dm = r.get("detection_metrics", {})
        tprs.append(dm.get("tpr", 0))
        fprs.append(dm.get("fpr", 0))

    # -- Accuracy vs Threshold --
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(thresholds, accs, "b-o", linewidth=2, label="Final Accuracy")
    ax1.set_xlabel("Z-Score Threshold")
    ax1.set_ylabel("Final Accuracy", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, tprs, "g-^", linewidth=2, label="TPR")
    ax2.plot(thresholds, fprs, "r-s", linewidth=2, label="FPR")
    ax2.set_ylabel("Rate", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim(-0.05, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.suptitle("Gate 3 Threshold Sensitivity Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_root / "threshold_sweep.png", dpi=150)
    plt.close(fig)
    print(f"[output] Threshold sweep plot: {output_root / 'threshold_sweep.png'}")

    # Save sweep CSV
    csv_path = output_root / "threshold_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["z_threshold", "accuracy", "tpr", "fpr"])
        w.writeheader()
        for t, a, tp, fp in zip(thresholds, accs, tprs, fprs):
            w.writerow({"z_threshold": t, "accuracy": a, "tpr": tp, "fpr": fp})
    print(f"[output] Threshold sweep CSV: {csv_path}")


# ---------------------------------------------------------------------------
# Thesis preset: full configuration matrix
# ---------------------------------------------------------------------------

THESIS_CONFIGS: List[ExperimentConfig] = [
    # ── Phase 1: Gate ablation (FedAvg) ──
    # 1) Baseline -- no gates at all
    ExperimentConfig(
        label="baseline_no_gates",
        enable_gate1=False, enable_gate2=False, enable_gate3=False,
    ),
    # 2) Gate 1 only  (mTLS, no sig verification, no anomaly detection)
    ExperimentConfig(
        label="gate1_only",
        enable_gate1=True, enable_gate2=False, enable_gate3=False,
    ),
    # 3) Gate 1 + Gate 2  (mTLS + signatures, no anomaly detection)
    ExperimentConfig(
        label="gate1_gate2",
        enable_gate1=True, enable_gate2=True, enable_gate3=False,
    ),
    # 4) Full Zero-Trust  (mTLS + signatures + anomaly detection)
    ExperimentConfig(
        label="full_zt",
        enable_gate1=True, enable_gate2=True, enable_gate3=True,
    ),

    # ── Phase 2: Aggregation method comparison ──
    # 5) Baseline + Krum (no gates, Krum aggregation)
    ExperimentConfig(
        label="baseline_krum",
        enable_gate1=False, enable_gate2=False, enable_gate3=False,
        aggregation_method="krum",
    ),
    # 6) Baseline + Multi-Krum
    ExperimentConfig(
        label="baseline_multikrum",
        enable_gate1=False, enable_gate2=False, enable_gate3=False,
        aggregation_method="multi-krum",
    ),
    # 7) Full ZT + Krum
    ExperimentConfig(
        label="full_zt_krum",
        enable_gate1=True, enable_gate2=True, enable_gate3=True,
        aggregation_method="krum",
    ),
    # 8) Full ZT + Multi-Krum
    ExperimentConfig(
        label="full_zt_multikrum",
        enable_gate1=True, enable_gate2=True, enable_gate3=True,
        aggregation_method="multi-krum",
    ),
]

# Separate non-IID configs (appended when --dirichlet-alpha is given)
NONIID_CONFIGS: List[ExperimentConfig] = [
    ExperimentConfig(
        label="baseline_noniid",
        enable_gate1=False, enable_gate2=False, enable_gate3=False,
    ),
    ExperimentConfig(
        label="full_zt_noniid",
        enable_gate1=True, enable_gate2=True, enable_gate3=True,
    ),
    ExperimentConfig(
        label="full_zt_krum_noniid",
        enable_gate1=True, enable_gate2=True, enable_gate3=True,
        aggregation_method="krum",
    ),
]


# ---------------------------------------------------------------------------
# MLflow logging helpers (called from orchestrator functions)
# ---------------------------------------------------------------------------

def _log_seed_run(
    tracker: Any,
    cfg: ExperimentConfig,
    seed: int,
    metrics: Dict,
    run_dir: Path,
) -> None:
    """Create a nested MLflow run for one seed and log its metrics."""
    tracker.start_run(run_name=f"seed_{seed}", nested=True,
                      config_dict={"seed": seed})
    try:
        # Derive attacker / honest IDs for detection-rate computation
        num_honest = cfg.num_clients - cfg.num_attackers
        attacker_cids = set(str(i) for i in range(num_honest, cfg.num_clients))
        honest_cids = set(str(i) for i in range(num_honest))

        for rm in metrics.get("round_metrics", []):
            rm_metrics: Dict[str, Any] = {}
            if "global_accuracy" in rm:
                rm_metrics["global_accuracy"] = rm["global_accuracy"]
            if "global_loss" in rm:
                rm_metrics["global_loss"] = rm["global_loss"]
            rm_metrics["accepted_clients"] = rm.get("gate3_accepted", 0)
            rm_metrics["rejected_clients"] = rm.get("gate3_rejected", 0)

            # Per-round detection rate / FPR
            rej_cids = set(str(c) for c in rm.get("gate3_rejected_cids", []))
            if attacker_cids:
                tp = len(rej_cids & attacker_cids)
                fp = len(rej_cids & honest_cids)
                rm_metrics["detection_rate"] = (
                    tp / len(attacker_cids) if attacker_cids else 0
                )
                rm_metrics["false_positive_rate"] = (
                    fp / len(honest_cids) if honest_cids else 0
                )

            tracker.log_metrics(rm_metrics, step=rm["round"])

        # Final accuracy for this seed
        rms = metrics.get("round_metrics", [])
        if rms and "global_accuracy" in rms[-1]:
            tracker.log_metrics({"final_accuracy": rms[-1]["global_accuracy"]})

        # Log model .pth artifacts
        models_dir = run_dir / "results" / "models"
        if models_dir.exists():
            for pth in sorted(models_dir.glob("round_*.pth")):
                tracker.log_artifact(str(pth))
    finally:
        tracker.end_run()  # always close seed run


def _log_summary_metrics(
    tracker: Any,
    cfg: ExperimentConfig,
    summary: Dict,
    out_dir: Path,
    staging: Path,
) -> None:
    """Log aggregate summary metrics and artifacts to the parent MLflow run."""
    final_metrics: Dict[str, Any] = {
        "final_accuracy": summary.get("final_accuracy_mean"),
        "final_accuracy_std": summary.get("final_accuracy_std"),
    }
    # Mean accuracy last 5 rounds
    rs = summary.get("round_summary", [])
    if len(rs) >= 5:
        last5 = [r.get("accuracy_mean", 0) for r in rs[-5:]
                 if r.get("accuracy_mean") is not None]
        if last5:
            final_metrics["mean_accuracy_last_5_rounds"] = (
                sum(last5) / len(last5)
            )

    # Detection metrics
    dm = summary.get("detection_metrics", {})
    if dm:
        final_metrics["overall_detection_rate"] = dm.get("tpr")
        final_metrics["overall_false_positive_rate"] = dm.get("fpr")
        final_metrics["overall_precision"] = dm.get("precision")
        final_metrics["overall_recall"] = dm.get("recall")
        final_metrics["overall_f1"] = dm.get("f1_score")

    tracker.log_metrics(final_metrics)

    # Log output artifacts (CSVs, plots, summary JSON)
    for p in out_dir.glob("*.csv"):
        tracker.log_artifact(str(p))
    for p in out_dir.glob("*.png"):
        tracker.log_artifact(str(p))
    summary_json = out_dir / "summary.json"
    if summary_json.exists():
        tracker.log_artifact(str(summary_json))

    # Log final model via mlflow.pytorch
    last_seed = cfg.seeds[-1] if cfg.seeds else None
    if last_seed is not None:
        final_model_path = (
            staging / f"seed_{last_seed}" / "results" / "models"
            / f"round_{cfg.num_rounds}.pth"
        )
        if final_model_path.exists():
            try:
                import torch
                from model import CifarCNN
                model = CifarCNN()
                model.load_state_dict(
                    torch.load(final_model_path, map_location="cpu",
                               weights_only=True)
                )
                tracker.log_model(model, "final_model")
            except Exception as exc:
                print(f"[mlflow] Could not log final PyTorch model: {exc}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_single_experiment(
    cfg: ExperimentConfig,
    output_root: Path,
    tracker: Optional[Any] = None,
) -> Dict:
    """Run one experiment (all seeds) and return aggregated results."""
    if tracker is None:
        tracker = NullTracker()

    out_dir = output_root / cfg.label
    out_dir.mkdir(parents=True, exist_ok=True)

    staging = STAGING_ROOT / cfg.label
    staging.mkdir(parents=True, exist_ok=True)

    # -- credentials (ZT only) --
    cert_dir = staging / "certs"
    key_dir = staging / "signing_keys"
    if cfg.enable_gate1:
        generate_credentials(cfg.num_clients, cert_dir, key_dir)
    else:
        cert_dir.mkdir(parents=True, exist_ok=True)
        key_dir.mkdir(parents=True, exist_ok=True)

    # -- Build Docker image --
    build_images(cfg)

    # -- MLflow: start parent run for this config --
    tracker.start_run(
        run_name=cfg.label,
        config_dict={
            "num_clients": cfg.num_clients,
            "num_attackers": cfg.num_attackers,
            "attack_type": cfg.attack_type,
            "aggregation_method": cfg.aggregation_method,
            "dirichlet_alpha": cfg.dirichlet_alpha,
            "anomaly_threshold": cfg.z_threshold,
            "rounds": cfg.num_rounds,
            "gate1_enabled": cfg.enable_gate1,
            "gate2_enabled": cfg.enable_gate2,
            "gate3_enabled": cfg.enable_gate3,
            "seeds": cfg.seeds,
            "min_accepted": cfg.min_accepted,
        },
    )
    tracker.log_system_info()
    tracker.log_reproducibility_info()

    try:
        # -- Run each seed --
        all_runs: List[Dict] = []
        for seed in cfg.seeds:
            run_dir = staging / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            compose_path = generate_compose(cfg, seed, run_dir, cert_dir, key_dir)
            project_name = f"exp-{cfg.label}-s{seed}".replace("_", "-")

            metrics = run_seed(cfg, seed, compose_path, project_name)
            # Enrich with experiment metadata
            metrics["seed"] = seed
            metrics["attack_type"] = cfg.attack_type
            metrics["enable_gate1"] = cfg.enable_gate1
            metrics["enable_gate2"] = cfg.enable_gate2
            metrics["enable_gate3"] = cfg.enable_gate3
            all_runs.append(metrics)

            # -- MLflow: nested seed run --
            _log_seed_run(tracker, cfg, seed, metrics, run_dir)

        # -- Aggregate --
        summary = aggregate_seeds(all_runs, cfg)
        summary["label"] = cfg.label
        summary["config"] = asdict(cfg)

        # -- Save outputs --
        with open(out_dir / "all_runs.json", "w", encoding="utf-8") as f:
            json.dump(all_runs, f, indent=2, ensure_ascii=False, default=str)
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        save_csv(summary, out_dir)
        save_detection_csv(summary, out_dir)
        save_plots(all_runs, summary, out_dir, cfg.label)

        # -- MLflow: log final summary metrics to parent run --
        _log_summary_metrics(tracker, cfg, summary, out_dir, staging)

        final = summary.get("final_accuracy_mean")
        final_std = summary.get("final_accuracy_std")
        if final is not None:
            print(f"\n[{cfg.label}] Final accuracy: {final:.4f} +/- {final_std:.4f}")
        else:
            print(f"\n[{cfg.label}] No accuracy data collected")

        return summary

    finally:
        tracker.end_run()  # always close parent run


def run_comparison(
    configs: List[ExperimentConfig],
    output_root: Path,
    tracker: Optional[Any] = None,
) -> None:
    """Run multiple experiments and produce a comparison summary."""
    if tracker is None:
        tracker = NullTracker()

    all_summaries = []
    for cfg in configs:
        s = run_single_experiment(cfg, output_root, tracker=tracker)
        all_summaries.append(s)

    # -- Comparison CSV --
    csv_path = output_root / "comparison.csv"
    rows = []
    for s in all_summaries:
        c = s.get("config", {})
        dm = s.get("detection_metrics", {})
        rows.append({
            "label": s["label"],
            "gate1": c.get("enable_gate1"),
            "gate2": c.get("enable_gate2"),
            "gate3": c.get("enable_gate3"),
            "aggregation": c.get("aggregation_method", "fedavg"),
            "dirichlet_alpha": c.get("dirichlet_alpha"),
            "seeds": s["num_seeds"],
            "final_acc_mean": s.get("final_accuracy_mean"),
            "final_acc_std": s.get("final_accuracy_std"),
            "tpr": dm.get("tpr"),
            "fpr": dm.get("fpr"),
            "precision": dm.get("precision"),
            "recall": dm.get("recall"),
            "f1": dm.get("f1_score"),
        })
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[comparison] Written to {csv_path}")

    # -- Comparison bar chart --
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = [s["label"].replace("_", "\n") for s in all_summaries]
        means = [s.get("final_accuracy_mean", 0) or 0 for s in all_summaries]
        stds = [s.get("final_accuracy_std", 0) or 0 for s in all_summaries]

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
        colors = ["#d35f5f", "#e8a838", "#56b4e9", "#009e73",
                  "#cc79a7", "#0072b2", "#d55e00", "#999999"]
        bar_colors = [colors[i % len(colors)] for i in range(len(labels))]
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Final Accuracy")
        ax.set_title("Experiment Comparison: Final Accuracy")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(output_root / "comparison.png", dpi=150)
        plt.close(fig)
        print(f"[comparison] Plot saved: {output_root / 'comparison.png'}")
    except ImportError:
        pass

    # -- Convergence comparison --
    save_convergence_comparison(all_summaries, output_root)

    # -- Print table --
    print(f"\n{'='*90}")
    print("  EXPERIMENT COMPARISON SUMMARY")
    print(f"{'='*90}")
    hdr = (f"{'Label':<25} {'G1':>3} {'G2':>3} {'G3':>3} {'Agg':>10} "
           f"{'Acc Mean':>9} {'Acc Std':>8} {'TPR':>6} {'FPR':>6} {'F1':>6}")
    print(hdr)
    print("-" * 90)
    for r in rows:
        g1 = "ON" if r["gate1"] else "OFF"
        g2 = "ON" if r["gate2"] else "OFF"
        g3 = "ON" if r["gate3"] else "OFF"
        agg = r.get("aggregation", "fedavg")
        m = f"{r['final_acc_mean']:.4f}" if r["final_acc_mean"] else "N/A"
        s = f"{r['final_acc_std']:.4f}" if r["final_acc_std"] else "N/A"
        tp = f"{r['tpr']:.4f}" if r.get("tpr") is not None else "N/A"
        fp = f"{r['fpr']:.4f}" if r.get("fpr") is not None else "N/A"
        f1 = f"{r['f1']:.4f}" if r.get("f1") is not None else "N/A"
        print(f"{r['label']:<25} {g1:>3} {g2:>3} {g3:>3} {agg:>10} "
              f"{m:>9} {s:>8} {tp:>6} {fp:>6} {f1:>6}")
    print(f"{'='*90}\n")

    # -- MLflow: log comparison-level artifacts --
    tracker.start_run(run_name="comparison_summary")
    try:
        for p in output_root.glob("comparison*"):
            tracker.log_artifact(str(p))
        conv = output_root / "convergence_comparison.png"
        if conv.exists():
            tracker.log_artifact(str(conv))
    finally:
        tracker.end_run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Docker-orchestrated FL experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--preset", choices=["thesis", "smoke", "threshold-sweep"],
                    help="Predefined experiment matrix")

    # Seeds / rounds
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--num-rounds", type=int, default=20)

    # Client topology
    ap.add_argument("--num-clients", type=int, default=2)
    ap.add_argument("--num-attackers", type=int, default=1)

    # Attack
    ap.add_argument("--attack-type", default="label_flip",
                    choices=["label_flip", "targeted", "noise", "scale"])
    ap.add_argument("--noise-scale", type=float, default=5.0)
    ap.add_argument("--poison-scale", type=float, default=10.0)
    ap.add_argument("--source-label", type=int, default=5)
    ap.add_argument("--target-label", type=int, default=3)

    # Gate toggles
    ap.add_argument("--enable-gate1", action="store_true", default=True)
    ap.add_argument("--no-gate1", dest="enable_gate1", action="store_false")
    ap.add_argument("--enable-gate2", action="store_true", default=True)
    ap.add_argument("--no-gate2", dest="enable_gate2", action="store_false")
    ap.add_argument("--enable-gate3", action="store_true", default=True)
    ap.add_argument("--no-gate3", dest="enable_gate3", action="store_false")

    # Tuning
    ap.add_argument("--z-threshold", type=float, default=2.0)
    ap.add_argument("--min-accepted", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800,
                    help="Timeout per seed in seconds")

    # Phase 2 additions
    ap.add_argument("--aggregation-method", default="fedavg",
                    choices=["fedavg", "krum", "multi-krum"],
                    help="Aggregation strategy")
    ap.add_argument("--dirichlet-alpha", type=float, default=None,
                    help="Dirichlet concentration for non-IID partition "
                         "(omit for IID)")
    ap.add_argument("--threshold-values", nargs="+", type=float,
                    default=None,
                    help="Z-threshold values for sweep mode")

    # GPU
    ap.add_argument("--no-gpu", dest="gpu", action="store_false", default=True)

    # MLflow
    ap.add_argument("--no-mlflow", dest="mlflow", action="store_false",
                    default=True,
                    help="Disable MLflow tracking")
    ap.add_argument("--mlflow-experiment", type=str, default=None,
                    help="MLflow experiment name "
                         "(default: auto-set per preset)")
    ap.add_argument("--mlflow-tracking-uri", type=str,
                    default="./mlruns",
                    help="MLflow tracking URI (default: ./mlruns)")

    # Output
    ap.add_argument("--output-dir", type=Path,
                    default=PROJECT_ROOT / "experiment_results")

    return ap.parse_args()


def _apply_common_args(cfg: ExperimentConfig, args) -> None:
    """Apply shared CLI arguments to an ExperimentConfig."""
    if args.seeds != [42]:
        cfg.seeds = args.seeds
    elif len(cfg.seeds) <= 1:
        cfg.seeds = [42, 123, 456]
    cfg.num_rounds = args.num_rounds
    cfg.num_clients = args.num_clients
    cfg.num_attackers = args.num_attackers
    cfg.attack_type = args.attack_type
    cfg.timeout = args.timeout
    cfg.gpu = args.gpu
    if args.dirichlet_alpha is not None:
        cfg.dirichlet_alpha = args.dirichlet_alpha


def run_threshold_sweep(
    base_cfg: ExperimentConfig,
    thresholds: List[float],
    output_root: Path,
    tracker: Optional[Any] = None,
) -> None:
    """Run the full ZT pipeline at multiple Z-score thresholds.

    Produces a threshold-vs-accuracy/TPR/FPR sensitivity plot.
    """
    if tracker is None:
        tracker = NullTracker()

    sweep_dir = output_root / "threshold_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    sweep_results = []
    for z in thresholds:
        from dataclasses import replace as dc_replace
        cfg = dc_replace(base_cfg,
                         label=f"zt_z{z:.1f}".replace(".", "p"),
                         z_threshold=z)
        summary = run_single_experiment(cfg, sweep_dir, tracker=tracker)
        summary["z_threshold"] = z
        sweep_results.append(summary)

    save_threshold_sweep_plots(sweep_results, output_root)

    # Save full sweep JSON
    with open(output_root / "threshold_sweep_results.json", "w",
              encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2, default=str)
    print(f"[sweep] Results saved to {output_root}")

    # -- MLflow: log sweep-level artifacts --
    tracker.start_run(run_name="threshold_sweep_summary")
    try:
        for p in output_root.glob("threshold_sweep*"):
            tracker.log_artifact(str(p))
    finally:
        tracker.end_run()


def main() -> None:
    args = parse_args()
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # -- Create MLflow tracker (or no-op) --
    if args.mlflow:
        exp_name = args.mlflow_experiment
        if exp_name is None:
            preset = args.preset or "custom"
            exp_name = f"ZT-Pipeline / {preset.title().replace('-', ' ')}"
        tracker: Any = ExperimentTracker(
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=exp_name,
        )
        print(f"[mlflow] Tracking URI: {args.mlflow_tracking_uri}")
        print(f"[mlflow] Experiment:   {exp_name}")
    else:
        tracker = NullTracker()

    if args.preset == "thesis":
        # Run the full N-config x N-seed matrix
        configs = list(THESIS_CONFIGS)
        if args.dirichlet_alpha is not None:
            configs.extend(NONIID_CONFIGS)
        for cfg in configs:
            _apply_common_args(cfg, args)
        run_comparison(configs, output_root, tracker=tracker)

    elif args.preset == "threshold-sweep":
        thresholds = args.threshold_values or [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        base_cfg = ExperimentConfig(
            label="sweep_base",
            enable_gate1=True, enable_gate2=True, enable_gate3=True,
            aggregation_method=args.aggregation_method,
        )
        _apply_common_args(base_cfg, args)
        run_threshold_sweep(base_cfg, thresholds, output_root, tracker=tracker)

    elif args.preset == "smoke":
        # Quick 2-seed test with all gates
        cfg = ExperimentConfig(
            label="smoke_test",
            seeds=[42, 123],
            num_rounds=3,
            num_clients=2,
            num_attackers=1,
            timeout=args.timeout,
            gpu=args.gpu,
            aggregation_method=args.aggregation_method,
            dirichlet_alpha=args.dirichlet_alpha,
        )
        run_single_experiment(cfg, output_root, tracker=tracker)

    else:
        # Single custom experiment
        cfg = ExperimentConfig(
            label=("zt" if args.enable_gate1 else "baseline") + "_custom",
            seeds=args.seeds,
            num_rounds=args.num_rounds,
            num_clients=args.num_clients,
            num_attackers=args.num_attackers,
            attack_type=args.attack_type,
            noise_scale=args.noise_scale,
            poison_scale=args.poison_scale,
            source_label=args.source_label,
            target_label=args.target_label,
            enable_gate1=args.enable_gate1,
            enable_gate2=args.enable_gate2,
            enable_gate3=args.enable_gate3,
            z_threshold=args.z_threshold,
            min_accepted=args.min_accepted,
            timeout=args.timeout,
            gpu=args.gpu,
            aggregation_method=args.aggregation_method,
            dirichlet_alpha=args.dirichlet_alpha,
        )
        run_single_experiment(cfg, output_root, tracker=tracker)

    print("\n[DONE] All experiments finished.")


if __name__ == "__main__":
    main()
