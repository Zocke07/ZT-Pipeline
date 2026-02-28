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

import flwr as fl
from flwr.server.strategy import FedAvg

# ---------------------------------------------------------------------------
# Configuration (mirrors the ZT server for fair comparison)
# ---------------------------------------------------------------------------
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", "2"))
NUM_ROUNDS = int(os.environ.get("NUM_ROUNDS", "3"))
MIN_CLIENTS = int(os.environ.get("MIN_CLIENTS", str(NUM_CLIENTS)))


def main() -> None:
    print("=" * 60)
    print("  BASELINE (INSECURE) FEDERATED LEARNING SERVER")
    print("  No mTLS  |  No Signatures  |  No Anomaly Detection")
    print("=" * 60)
    print(f"  Clients expected : {NUM_CLIENTS}")
    print(f"  Rounds           : {NUM_ROUNDS}")
    print(f"  Strategy         : FedAvg (unmodified)")
    print(f"  Security         : NONE")
    print("=" * 60)

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=MIN_CLIENTS,
        min_evaluate_clients=MIN_CLIENTS,
        min_available_clients=MIN_CLIENTS,
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
    )

    # No `certificates` parameter → gRPC runs in plaintext / insecure mode
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
