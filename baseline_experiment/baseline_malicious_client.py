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
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import CifarCNN

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration (mirrors client_malicious.py for fair comparison)
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

ATTACK_MODE = os.environ.get("ATTACK_MODE", "label_flip")
NUM_CLASSES = 10
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", "2"))


# ---------------------------------------------------------------------------
# Data helpers (identical to baseline_client.py)
# ---------------------------------------------------------------------------
def _get_cifar10(data_dir: str = "/data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train_set = datasets.CIFAR10(data_dir, train=True,  download=True, transform=transform)
    test_set  = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    return train_set, test_set


def partition_data(train_set, num_clients: int, client_id: int):
    total = len(train_set)
    shard_size = total // num_clients
    start = client_id * shard_size
    return Subset(train_set, list(range(start, start + shard_size)))


# ---------------------------------------------------------------------------
# Attack: Label Flipping
# ---------------------------------------------------------------------------
def _flip_labels(labels: torch.Tensor) -> torch.Tensor:
    """Invert every label: class i -> class (NUM_CLASSES - 1 - i)."""
    return (NUM_CLASSES - 1) - labels


def train_with_label_flipping(model: nn.Module, loader: DataLoader) -> float:
    """Train one epoch on FLIPPED labels. Returns average loss."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = _flip_labels(labels).to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


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
        self.model = CifarCNN().to(DEVICE)

        train_set, test_set = _get_cifar10()
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
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List) -> None:
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.array(v))
             for k, v in zip(self.model.state_dict().keys(), parameters)}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        for _ in range(LOCAL_EPOCHS):
            loss = train_with_label_flipping(self.model, self.train_loader)
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
