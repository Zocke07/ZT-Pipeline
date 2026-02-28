"""Malicious Flower Client – Label Flipping Attack (+ optional weight noise).

Simulates an adversarial participant in the federated learning pipeline.

Zero-Trust bypass strategy (what the attacker achieves):
  ✓ Gate 1 (mTLS)    – presents a valid, CA-signed client certificate.
  ✓ Gate 2 (Signing) – signs updates with its legitimate RSA private key.
  ✗ Detected by Gate 3 – the update delta is anti-correlated with honest
                          clients, triggering directional anomaly detection.

Attack modes (set via ATTACK_MODE env var):
  "label_flip" (default)
       During training, all labels are inverted:
           label → (NUM_CLASSES - 1 - label)
       i.e., class 0 ↔ class 9, class 1 ↔ class 8, etc.
       Effect: the local gradient points in the opposite direction to
       what honest training would produce.  The aggregated global model
       accuracy on the test set degrades toward ~10% (random chance)
       when the poisoned update is accepted.

  "targeted"
       Flips only a single source label to a target label:
           SOURCE_LABEL (env) → TARGET_LABEL (env)
       Subtler attack: harder to detect visually, still triggers
       directional detection.

  "noise"
       Adds large Gaussian noise directly to the post-training weights
       (simpler attack, already in poisoned_client.py — included here
       for comparison in the thesis).

Thesis note:
    This client deliberately passes cryptographic verification (Gates 1 & 2)
    to show that identity + integrity checks alone are insufficient.
    Gate 3 (behavioral / statistical detection) is required to complete the
    Zero-Trust pipeline.
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
CERT_DIR        = Path(os.environ.get("CERT_DIR", "/certs"))

BATCH_SIZE    = 64
LEARNING_RATE = 0.001
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPU optimisation: enable TF32 on Ampere+ GPUs (RTX 4000 Ada)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

ATTACK_MODE   = os.environ.get("ATTACK_MODE",   "label_flip")
NOISE_SCALE   = float(os.environ.get("NOISE_SCALE", "5.0"))
SOURCE_LABEL  = int(os.environ.get("SOURCE_LABEL", "0"))
TARGET_LABEL  = int(os.environ.get("TARGET_LABEL", "1"))
NUM_CLASSES   = 10
LOCAL_EPOCHS  = int(os.environ.get("LOCAL_EPOCHS",  "2"))


# ---------------------------------------------------------------------------
# Data helpers
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
    total      = len(train_set)
    shard_size = total // num_clients
    start      = client_id * shard_size
    return Subset(train_set, list(range(start, start + shard_size)))


# ---------------------------------------------------------------------------
# Attack: Label Flipping training loop
# ---------------------------------------------------------------------------

def _flip_labels_global(labels: torch.Tensor) -> torch.Tensor:
    """Invert every label: class i → class (NUM_CLASSES-1-i)."""
    return (NUM_CLASSES - 1) - labels


def _flip_labels_targeted(labels: torch.Tensor) -> torch.Tensor:
    """Flip only SOURCE_LABEL → TARGET_LABEL."""
    flipped = labels.clone()
    flipped[flipped == SOURCE_LABEL] = TARGET_LABEL
    return flipped


def train_with_label_flipping(
    model: nn.Module,
    loader: DataLoader,
    flip_fn,
) -> float:
    """Train for one epoch with flipped labels. Returns average loss."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = flip_fn(labels).to(DEVICE)          # ← ATTACK: swap labels
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def train_normal(model: nn.Module, loader: DataLoader) -> float:
    """Honest training (used in "noise" mode before adding noise)."""
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


def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct    += (outputs.argmax(dim=1) == labels).sum().item()
            total      += labels.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Malicious Flower Client
# ---------------------------------------------------------------------------

class MaliciousClient(fl.client.NumPyClient):
    """Authenticated malicious client: passes Gates 1+2, defeated by Gate 3.

    The client holds valid mTLS certificates and a legitimate signing key,
    so it appears to be a trusted participant.  However:
      - In "label_flip" mode it trains on *inverted* labels, causing its
        weight update delta to point away from the honest training direction.
      - In "targeted" mode it flips one source class to a target class.
      - In "noise"  mode it trains normally then corrupts weights with noise.
    """

    def __init__(self, client_id: int, num_clients: int) -> None:
        self.client_id = client_id
        self.model     = CifarCNN().to(DEVICE)

        train_set, test_set = _get_cifar10()
        self.train_loader = DataLoader(
            partition_data(train_set, num_clients, client_id),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        )
        self.test_loader = DataLoader(
            test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )

        # Gate 2: Load legitimate signing key (so signatures pass)
        key_path = SIGNING_KEY_DIR / f"client-{client_id}.private.pem"
        if key_path.exists():
            self.signing_key = load_private_key(key_path)
            print(f"[MALICIOUS {client_id}] ✓  Signing key loaded  → Gate 2 WILL pass")
        else:
            self.signing_key = None
            print(f"[MALICIOUS {client_id}] ⚠  No signing key")

        # Select attack function
        if ATTACK_MODE == "label_flip":
            self._flip_fn = _flip_labels_global
            desc = "GLOBAL label flip  (label → 9-label)"
        elif ATTACK_MODE == "targeted":
            self._flip_fn = _flip_labels_targeted
            desc = f"TARGETED flip  ({SOURCE_LABEL} → {TARGET_LABEL})"
        else:
            self._flip_fn = None   # noise mode
            desc = f"weight NOISE  (σ={NOISE_SCALE})"

        print(f"[MALICIOUS {client_id}] 🔴 ATTACK MODE: {ATTACK_MODE}  ({desc})")
        print(f"[MALICIOUS {client_id}] Device: {DEVICE}  |  "
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

        if ATTACK_MODE in ("label_flip", "targeted"):
            # ── Attack: train on flipped labels ─────────────────────
            for _ in range(LOCAL_EPOCHS):
                loss = train_with_label_flipping(
                    self.model, self.train_loader, self._flip_fn
                )
            poisoned_params = self.get_parameters(config={})
            print(f"  [MALICIOUS] fit loss (on flipped labels): {loss:.4f}  "
                  f"({LOCAL_EPOCHS} epoch(s)) ← anti-correlated update")
        else:
            # ── Attack: honest training + noise injection ────────────
            for _ in range(LOCAL_EPOCHS):
                loss = train_normal(self.model, self.train_loader)
            honest_params  = self.get_parameters(config={})
            poisoned_params = [
                w + np.random.randn(*w.shape).astype(np.float32) * NOISE_SCALE
                for w in honest_params
            ]
            print(f"  [MALICIOUS] honest loss: {loss:.4f}  "
                  f"→ added noise (σ={NOISE_SCALE})")

        # Gate 2: Sign the poisoned weights (signature will be VALID)
        server_round = int(config.get("server_round", 0))
        metrics: dict = {"client_id": float(self.client_id)}
        if self.signing_key is not None:
            sig = sign_parameters(poisoned_params, self.signing_key, server_round=server_round)
            metrics["signature"] = sig
            print(f"  [MALICIOUS] Update SIGNED  ✓  → Gate 2 passes, Gate 3 should FIRE")

        return poisoned_params, len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = evaluate(self.model, self.test_loader)
        print(f"  [MALICIOUS] eval loss: {loss:.4f}  acc: {accuracy:.4f}")
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}


# ---------------------------------------------------------------------------
# Gate 1: mTLS helpers (identical to client.py)
# ---------------------------------------------------------------------------

def _load_client_certificates(client_id: int) -> Optional[Tuple[bytes, bytes, bytes]]:
    ca_path   = CERT_DIR / "ca.crt"
    cert_path = CERT_DIR / f"client-{client_id}.crt"
    key_path  = CERT_DIR / f"client-{client_id}.key"

    if not all(p.exists() for p in [ca_path, cert_path, key_path]):
        print(f"[MALICIOUS {client_id}] ⚠  Certs missing – connecting WITHOUT mTLS")
        return None

    print(f"[MALICIOUS {client_id}] 🔒 mTLS certs loaded  → Gate 1 WILL pass")
    return ca_path.read_bytes(), cert_path.read_bytes(), key_path.read_bytes()


def _patch_grpc_for_mtls(ca_cert: bytes, client_cert: bytes, client_key: bytes) -> None:
    _orig = grpc.ssl_channel_credentials

    def _mtls(root_certificates=None, private_key=None, certificate_chain=None):
        return _orig(
            root_certificates=root_certificates or ca_cert,
            private_key=client_key,
            certificate_chain=client_cert,
        )

    grpc.ssl_channel_credentials = _mtls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client_id   = int(os.environ.get("CLIENT_ID",      "1"))
    num_clients = int(os.environ.get("NUM_CLIENTS",    "2"))
    server_addr =     os.environ.get("SERVER_ADDRESS", "server:8080")

    # Gate 1
    certs = _load_client_certificates(client_id)
    if certs is not None:
        ca_cert, client_cert, client_key = certs
        _patch_grpc_for_mtls(ca_cert, client_cert, client_key)
        root_certificates = ca_cert
    else:
        root_certificates = None

    client = MaliciousClient(client_id, num_clients)
    fl.client.start_client(
        server_address=server_addr,
        client=client.to_client(),
        root_certificates=root_certificates,
    )


if __name__ == "__main__":
    main()
