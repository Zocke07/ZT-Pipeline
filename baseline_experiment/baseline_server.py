"""Baseline (Insecure) Flower Server – Standard FedAvg on CIFAR-10.

This is the CONTROL GROUP for the thesis experiment.  It represents how
federated learning is typically deployed in tutorials and many real-world
prototypes: **no mTLS, no digital signatures, no anomaly detection**.

Differences from the Zero-Trust server (server.py):
  ✗  Gate 1 (mTLS)             — DISABLED.  Plain insecure gRPC.
  ✗  Gate 2 (Signatures)       — DISABLED.  No signing verification.
  ✗  Gate 3 (Anomaly Detection) — DISABLED.  All updates accepted blindly.

The aggregation strategy is unmodified FedAvg: every client update that
arrives is averaged into the global model, regardless of content.

Purpose: Prove that without Zero-Trust gates, a single malicious client
can degrade the global model with no resistance.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from server_utils import (
    MetricsCollector,
    krum_aggregate,
    save_round_model,
    weighted_eval_metrics,
)
from training import set_seed

# ---------------------------------------------------------------------------
# Configuration (mirrors the ZT server for fair comparison)
# ---------------------------------------------------------------------------
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", "2"))
NUM_ROUNDS = int(os.environ.get("NUM_ROUNDS", "20"))
MIN_CLIENTS = int(os.environ.get("MIN_CLIENTS", str(NUM_CLIENTS)))

# Reproducibility
SEED = int(os.environ.get("SEED", "-1"))

# Structured results output
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results"))

# Aggregation method: fedavg | krum | multi-krum
AGGREGATION_METHOD = os.environ.get("AGGREGATION_METHOD", "fedavg").lower()
NUM_ATTACKERS = int(os.environ.get("NUM_ATTACKERS", "0"))


# ---------------------------------------------------------------------------
# Baseline FedAvg with metrics collection
# ---------------------------------------------------------------------------
class BaselineMetricsFedAvg(FedAvg):
    """Standard FedAvg (or Krum/Multi-Krum) that records per-round metrics."""

    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        aggregation_method: str = "fedavg",
        num_attackers: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._metrics = metrics_collector
        self.aggregation_method = aggregation_method
        self.num_attackers = num_attackers

    def aggregate_fit(self, server_round, results, failures):
        method = self.aggregation_method
        print(f"\n── Aggregating {len(results)} updates via {method} ──")

        if method in ("krum", "multi-krum") and results:
            agg_weights, total_ex = krum_aggregate(
                results,
                num_byzantine=self.num_attackers,
                multi=(method == "multi-krum"),
            )
            agg_params = ndarrays_to_parameters(agg_weights)
            agg = (agg_params, {})
            save_round_model(
                list(parameters_to_ndarrays(agg_params)),
                server_round, RESULTS_DIR, prefix="BASELINE",
            )
        else:
            agg = super().aggregate_fit(server_round, results, failures)
            if agg is not None and agg[0] is not None:
                save_round_model(
                    parameters_to_ndarrays(agg[0]),
                    server_round, RESULTS_DIR, prefix="BASELINE",
                )

        if self._metrics:
            n = len(results) if results else 0
            self._metrics.record_fit(
                server_round,
                gate2_passed=n,
                gate2_rejected=0,
                gate3_accepted=n,
                gate3_rejected=0,
                gate3_rejected_cids=[],
                aggregation_skipped=False,
                aggregation_method=method,
                num_clients_reporting=n,
            )
        return agg

    def aggregate_evaluate(self, server_round, results, failures):
        result = super().aggregate_evaluate(server_round, results, failures)
        if result is not None and self._metrics:
            loss, mets = result
            if loss is not None:
                acc = mets.get("accuracy")
                self._metrics.record_eval(
                    server_round, loss, acc if acc is not None else 0.0,
                )
        return result


def main() -> None:
    # ── Reproducibility ────────────────────────────────────────────────
    if SEED >= 0:
        set_seed(SEED)
        print(f"[BASELINE] Seed set to {SEED}")

    print("=" * 60)
    print("  BASELINE (INSECURE) FEDERATED LEARNING SERVER")
    print("  No mTLS  |  No Signatures  |  No Anomaly Detection")
    print("=" * 60)
    print(f"  Clients expected : {NUM_CLIENTS}")
    print(f"  Rounds           : {NUM_ROUNDS}")
    print(f"  Aggregation      : {AGGREGATION_METHOD}")
    print(f"  Security         : NONE")
    print("=" * 60)

    # ── Metrics collector ──────────────────────────────────────────────
    mc = MetricsCollector()

    strategy = BaselineMetricsFedAvg(
        metrics_collector=mc,
        aggregation_method=AGGREGATION_METHOD,
        num_attackers=NUM_ATTACKERS,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=MIN_CLIENTS,
        min_evaluate_clients=MIN_CLIENTS,
        min_available_clients=MIN_CLIENTS,
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
        evaluate_metrics_aggregation_fn=weighted_eval_metrics,
    )

    # No `certificates` parameter → gRPC runs in plaintext / insecure mode
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    # ── Save structured metrics ────────────────────────────────────────
    mc.save(
        RESULTS_DIR / "metrics.json",
        metadata={
            "seed": SEED,
            "num_rounds": NUM_ROUNDS,
            "num_clients": NUM_CLIENTS,
            "enable_gate1": False,
            "enable_gate2": False,
            "enable_gate3": False,
            "z_threshold": 0.0,
            "aggregation_method": AGGREGATION_METHOD,
            "num_attackers": NUM_ATTACKERS,
        },
    )


if __name__ == "__main__":
    main()
