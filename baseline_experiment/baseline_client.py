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
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# model.py is copied into /app at build time (same CifarCNN as the ZT pipeline)
from model import CifarCNN

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Hyperparameters (identical to the ZT client for fair comparison)
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
LEARNING_RATE = 0.001
LOCAL_EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPU optimisation: enable TF32 on Ampere+ GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# Data helpers (identical to client.py)
# ---------------------------------------------------------------------------
def _get_cifar10(data_dir: str = "/data"):
    """Download CIFAR-10 and return (train_set, test_set)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train_set = datasets.CIFAR10(data_dir, train=True,  download=True, transform=transform)
    test_set  = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    return train_set, test_set


def partition_data(train_set, num_clients: int, client_id: int):
    """Simple IID partition: split training set into equal shards."""
    total = len(train_set)
    shard_size = total // num_clients
    start = client_id * shard_size
    end = start + shard_size
    return Subset(train_set, list(range(start, end)))


# ---------------------------------------------------------------------------
# Train / evaluate helpers (identical to client.py)
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module, loader: DataLoader, scaler: torch.amp.GradScaler,
) -> float:
    """Train for one epoch with AMP mixed precision, return average loss."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    """Evaluate model, return (loss, accuracy)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


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
        self.model = CifarCNN().to(DEVICE)
        if DEVICE.type == "cuda":
            self.model = torch.compile(self.model)

        self.scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

        use_pin = DEVICE.type == "cuda"
        train_set, test_set = _get_cifar10()
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
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List) -> None:
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.array(v))
             for k, v in zip(self.model.state_dict().keys(), parameters)}
        )
        self.model.load_state_dict(state_dict, strict=True)

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
    # No root_certificates → no certificate verification
    fl.client.start_client(
        server_address=server_addr,
        client=client.to_client(),
        insecure=True,
    )


if __name__ == "__main__":
    main()
