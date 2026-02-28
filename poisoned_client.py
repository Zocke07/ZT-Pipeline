"""Poisoned Flower Client – Simulates a malicious insider.

This client is authenticated (Gate 1 – has valid mTLS certs) and signs
its updates (Gate 2 – has a valid signing key).  However, instead of
training on real data, it sends **random Gaussian noise** scaled to be
much larger than legitimate updates.

Purpose: Prove that Gate 3 (anomaly detection) catches and rejects
poisoned model updates even from fully authenticated clients.

Usage:
    Set POISON_MODE via environment variable:
        "noise"  → replace all weights with random noise (default)
        "scale"  → train normally but multiply weights by 100×

    Set POISON_SCALE to control the noise magnitude (default: 100.0).
"""

import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import grpc
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import CifarCNN
from signing import sign_parameters, load_private_key

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SIGNING_KEY_DIR = Path(os.environ.get("SIGNING_KEY_DIR", "/signing_keys"))
CERT_DIR = Path(os.environ.get("CERT_DIR", "/certs"))

BATCH_SIZE = 64
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

POISON_MODE = os.environ.get("POISON_MODE", "noise")    # "noise" or "scale"
POISON_SCALE = float(os.environ.get("POISON_SCALE", "100.0"))


# ---------------------------------------------------------------------------
# Data helpers (same as client.py)
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
    end = start + shard_size
    return Subset(train_set, list(range(start, end)))


def train_one_epoch(model, loader):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader):
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
# Poisoned Flower Client
# ---------------------------------------------------------------------------
class PoisonedClient(fl.client.NumPyClient):
    """A malicious client that passes Gate 1 + Gate 2 but sends bad weights."""

    def __init__(self, client_id: int, num_clients: int) -> None:
        self.client_id = client_id
        self.model = CifarCNN().to(DEVICE)
        train_set, test_set = _get_cifar10()
        self.train_loader = DataLoader(
            partition_data(train_set, num_clients, client_id),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
        )
        self.test_loader = DataLoader(
            test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
        )

        # Load signing key (so Gate 2 passes)
        key_path = SIGNING_KEY_DIR / f"client-{client_id}.private.pem"
        if key_path.exists():
            self.signing_key = load_private_key(key_path)
            print(f"[POISON Client {client_id}] ✓  Signing key loaded")
        else:
            self.signing_key = None

        print(f"[POISON Client {client_id}] 🧪 Mode: {POISON_MODE}  "
              f"Scale: {POISON_SCALE}  Device: {DEVICE}")

    def get_parameters(self, config) -> List:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List) -> None:
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        if POISON_MODE == "scale":
            # Train normally, then scale weights to absurd magnitude
            train_one_epoch(self.model, self.train_loader)
            poisoned = [w * POISON_SCALE for w in self.get_parameters(config={})]
            print(f"  → [POISON] Trained normally then scaled weights by {POISON_SCALE}×")
        else:
            # Replace weights entirely with random Gaussian noise
            poisoned = [
                np.random.randn(*w.shape).astype(np.float32) * POISON_SCALE
                for w in self.get_parameters(config={})
            ]
            print(f"  → [POISON] Injected random noise (σ={POISON_SCALE})")

        # Sign the poisoned weights (Gate 2 will pass!)
        metrics: dict = {"client_id": float(self.client_id)}
        if self.signing_key is not None:
            sig = sign_parameters(poisoned, self.signing_key)
            metrics["signature"] = sig
            print(f"  → [POISON] Poisoned update SIGNED ✓ (will pass Gate 2)")

        return poisoned, len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = evaluate(self.model, self.test_loader)
        print(f"  → eval loss: {loss:.4f}  acc: {accuracy:.4f}")
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}


# ---------------------------------------------------------------------------
# mTLS (same as client.py)
# ---------------------------------------------------------------------------
def _load_client_certificates(client_id: int):
    ca_cert_path = CERT_DIR / "ca.crt"
    client_cert_path = CERT_DIR / f"client-{client_id}.crt"
    client_key_path = CERT_DIR / f"client-{client_id}.key"

    if not all(p.exists() for p in [ca_cert_path, client_cert_path, client_key_path]):
        return None

    return (
        ca_cert_path.read_bytes(),
        client_cert_path.read_bytes(),
        client_key_path.read_bytes(),
    )


def _patch_grpc_for_mtls(ca_cert, client_cert, client_key):
    _original_fn = grpc.ssl_channel_credentials

    def _mtls(root_certificates=None, private_key=None, certificate_chain=None):
        return _original_fn(
            root_certificates=root_certificates or ca_cert,
            private_key=client_key,
            certificate_chain=client_cert,
        )

    grpc.ssl_channel_credentials = _mtls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    client_id = int(os.environ.get("CLIENT_ID", "0"))
    num_clients = int(os.environ.get("NUM_CLIENTS", "2"))
    server_addr = os.environ.get("SERVER_ADDRESS", "server:8080")

    certs = _load_client_certificates(client_id)
    if certs is not None:
        ca_cert, client_cert, client_key = certs
        _patch_grpc_for_mtls(ca_cert, client_cert, client_key)
        root_certificates = ca_cert
    else:
        root_certificates = None

    client = PoisonedClient(client_id, num_clients)
    fl.client.start_client(
        server_address=server_addr,
        client=client.to_client(),
        root_certificates=root_certificates,
    )


if __name__ == "__main__":
    main()
