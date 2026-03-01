"""Baseline (Insecure) Flower Client – Standard CIFAR-10 Training.

This is the CONTROL GROUP client.  It connects to the Flower server
over plain insecure gRPC and sends model updates with **no cryptographic
protections whatsoever**.

Differences from the Zero-Trust client (client.py):
  ✗  Gate 1 (mTLS)        — DISABLED.  Uses ``insecure=True``.
  ✗  Gate 2 (Signatures)  — DISABLED.  No signing keys loaded or used.
  ✗  No deny-by-default   — Client starts regardless of missing certs/keys.

Everything else is identical to the secure client:
  - Same CifarCNN model architecture
  - Same CIFAR-10 IID partitioning
  - Same hyperparameters (batch_size=64, lr=0.001, 1 local epoch)
  - Same AMP mixed-precision, TF32, torch.compile optimisations

Purpose: Provide a fair accuracy/loss comparison where the ONLY variable
is the presence or absence of Zero-Trust security mechanisms.
"""

import os
import warnings
from typing import List

import flwr as fl
from torch.utils.data import DataLoader

from data_utils import get_cifar10, partition_data
from training import (
    BATCH_SIZE,
    DEVICE,
    create_model,
    create_scaler,
    evaluate,
    get_parameters,
    set_parameters,
    train_one_epoch,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Flower client (NO signing, NO mTLS)
# ---------------------------------------------------------------------------
class BaselineCifarClient(fl.client.NumPyClient):
    """Standard (insecure) Flower NumPyClient.

    Identical to the ZT CifarClient except:
      - No signing key loaded
      - ``fit()`` returns plain metrics (no signature, no client_id binding)
    """

    def __init__(self, client_id: int, num_clients: int) -> None:
        self.client_id = client_id
        self.model = create_model(compile_model=True)
        self.scaler = create_scaler()

        use_pin = DEVICE.type == "cuda"
        train_set, test_set = get_cifar10()
        self.train_loader = DataLoader(
            partition_data(train_set, num_clients, client_id),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
            pin_memory=use_pin, persistent_workers=True,
        )
        self.test_loader = DataLoader(
            test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
            pin_memory=use_pin, persistent_workers=True,
        )

        print(f"[Baseline Client {client_id}] Device: {DEVICE}  |  "
              f"Train samples: {len(self.train_loader.dataset)}")

    # -- Flower interface ---------------------------------------------------
    def get_parameters(self, config) -> List:
        return get_parameters(self.model)

    def set_parameters(self, parameters: List) -> None:
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loss = train_one_epoch(self.model, self.train_loader, self.scaler)
        updated_params = self.get_parameters(config={})

        # ── NO SIGNING — updates sent in plaintext ──
        metrics: dict = {"client_id": float(self.client_id)}
        print(f"  → fit loss: {loss:.4f}  [UNSIGNED, insecure]")
        return updated_params, len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = evaluate(self.model, self.test_loader)
        print(f"  → eval loss: {loss:.4f}  acc: {accuracy:.4f}")
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}


# ---------------------------------------------------------------------------
# Main – connect INSECURELY to Flower server
# ---------------------------------------------------------------------------
def main() -> None:
    client_id = int(os.environ.get("CLIENT_ID", "0"))
    num_clients = int(os.environ.get("NUM_CLIENTS", "2"))
    server_addr = os.environ.get("SERVER_ADDRESS", "server:8080")

    print(f"[Baseline Client {client_id}] Connecting INSECURELY to {server_addr}")
    print(f"[Baseline Client {client_id}] No mTLS  |  No Signatures")

    client = BaselineCifarClient(client_id, num_clients)

    # insecure=True  → plain gRPC, no TLS at all
    fl.client.start_client(
        server_address=server_addr,
        client=client.to_client(),
        insecure=True,
    )


if __name__ == "__main__":
    main()
