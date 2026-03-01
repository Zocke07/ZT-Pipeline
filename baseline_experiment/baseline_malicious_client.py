"""Baseline (Insecure) Malicious Client – Label Flipping Attack.

This is the ATTACKER for Experiment A.  It connects to the insecure
baseline server and sends poisoned model updates.

Because the baseline has NO security gates:
  ✗  No mTLS           → anyone can connect
  ✗  No signatures     → updates are not verified
  ✗  No anomaly det.   → all updates are blindly aggregated

The poisoned update is accepted every round, causing the global model
accuracy to degrade significantly.

Attack mode: Global label flip  (label → 9 − label).
  - Identical to the ZT attacker (client_malicious.py) for fair comparison.
  - The ONLY difference: no signing, no mTLS.
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
    evaluate,
    get_parameters,
    set_parameters,
    train_one_epoch_with_label_transform,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration (mirrors client_malicious.py for fair comparison)
# ---------------------------------------------------------------------------
ATTACK_MODE = os.environ.get("ATTACK_MODE", "label_flip")
NUM_CLASSES = 10
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", "2"))


# ---------------------------------------------------------------------------
# Attack: Label Flipping
# ---------------------------------------------------------------------------
def _flip_labels(labels):
    """Invert every label: class i -> class (NUM_CLASSES - 1 - i)."""
    return (NUM_CLASSES - 1) - labels


# ---------------------------------------------------------------------------
# Malicious Flower Client (INSECURE – no signing, no mTLS)
# ---------------------------------------------------------------------------
class BaselineMaliciousClient(fl.client.NumPyClient):
    """Insecure malicious client for the baseline experiment.

    Identical attack logic to the ZT MaliciousClient, but:
      - No signing key (updates sent unsigned)
      - No mTLS certificates (connects insecurely)
      - The baseline server accepts everything → attack succeeds
    """

    def __init__(self, client_id: int, num_clients: int) -> None:
        self.client_id = client_id
        self.model = create_model(compile_model=False)

        train_set, test_set = get_cifar10()
        self.train_loader = DataLoader(
            partition_data(train_set, num_clients, client_id),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        )
        self.test_loader = DataLoader(
            test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )

        print(f"[BASELINE MALICIOUS {client_id}] ATTACK MODE: {ATTACK_MODE}")
        print(f"[BASELINE MALICIOUS {client_id}] No signing  |  No mTLS  |  "
              f"Updates will be BLINDLY ACCEPTED")
        print(f"[BASELINE MALICIOUS {client_id}] Device: {DEVICE}  |  "
              f"Train samples: {len(self.train_loader.dataset)}")

    def get_parameters(self, config) -> List:
        return get_parameters(self.model)

    def set_parameters(self, parameters: List) -> None:
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        for _ in range(LOCAL_EPOCHS):
            loss = train_one_epoch_with_label_transform(
                self.model, self.train_loader, _flip_labels,
            )
        poisoned_params = self.get_parameters(config={})

        print(f"  [BASELINE MALICIOUS] fit loss (flipped labels): {loss:.4f}  "
              f"({LOCAL_EPOCHS} epoch(s))  [UNSIGNED, ACCEPTED BLINDLY]")

        metrics: dict = {"client_id": float(self.client_id)}
        return poisoned_params, len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = evaluate(self.model, self.test_loader)
        print(f"  [BASELINE MALICIOUS] eval loss: {loss:.4f}  acc: {accuracy:.4f}")
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}


# ---------------------------------------------------------------------------
# Main – connect INSECURELY
# ---------------------------------------------------------------------------
def main() -> None:
    client_id = int(os.environ.get("CLIENT_ID", "1"))
    num_clients = int(os.environ.get("NUM_CLIENTS", "2"))
    server_addr = os.environ.get("SERVER_ADDRESS", "server:8080")

    print(f"[BASELINE MALICIOUS {client_id}] Connecting INSECURELY to {server_addr}")

    client = BaselineMaliciousClient(client_id, num_clients)
    fl.client.start_client(
        server_address=server_addr,
        client=client.to_client(),
        insecure=True,
    )


if __name__ == "__main__":
    main()
