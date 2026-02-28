"""Flower Federated Learning Client – PyTorch CNN on CIFAR-10.

Each client:
  1. Downloads / loads CIFAR-10 (partitioned by client ID).
  2. Trains for 1 local epoch on GPU.
  3. Signs the updated weights with its RSA private key.
  4. Sends signed weights back to the server.

Zero-Trust Gate 1: mTLS – each client presents its unique certificate
to the server. The server verifies it against the trusted Root CA.

Zero-Trust Gate 2: Integrity – model updates are digitally signed with
RSA-PSS (SHA-256).  The server verifies the signature before aggregation.
"""

import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple

import grpc
import numpy as np
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
# Gate 2: Signing key path
# ---------------------------------------------------------------------------
SIGNING_KEY_DIR = Path(os.environ.get("SIGNING_KEY_DIR", "/signing_keys"))

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
LEARNING_RATE = 0.001
LOCAL_EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPU optimisation: enable TF32 on Ampere+ GPUs (RTX 4000 Ada)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# Data helpers
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
    """Simple IID partition: split training set into `num_clients` equal shards."""
    total = len(train_set)
    shard_size = total // num_clients
    start = client_id * shard_size
    end = start + shard_size
    indices = list(range(start, end))
    return Subset(train_set, indices)


# ---------------------------------------------------------------------------
# Train / evaluate helpers
# ---------------------------------------------------------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader, scaler: torch.amp.GradScaler) -> float:
    """Train for one epoch with AMP mixed precision, return average loss."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
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
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Flower client
# ---------------------------------------------------------------------------
class CifarClient(fl.client.NumPyClient):
    """Flower NumPyClient wrapping the PyTorch CNN.

    Gate 2 enhancement: ``fit()`` now signs the serialized weight update
    with the client's RSA private key and attaches the base64 signature
    + client_id in the metrics dict so the server can verify integrity.
    """

    def __init__(self, client_id: int, num_clients: int) -> None:
        self.client_id = client_id
        self.model = CifarCNN().to(DEVICE)
        # G4: torch.compile for PyTorch 2.x graph optimization
        if DEVICE.type == "cuda":
            self.model = torch.compile(self.model)

        # G2: AMP GradScaler for mixed-precision training
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

        # Gate 2: Load signing private key — MANDATORY (Zero Trust: deny by default)
        key_path = SIGNING_KEY_DIR / f"client-{client_id}.private.pem"
        if key_path.exists():
            self.signing_key = load_private_key(key_path)
            print(f"[Client {client_id}] ✓  Signing key loaded from {key_path}")
        else:
            raise RuntimeError(
                f"[Client {client_id}] ✗  FATAL: signing key not found at {key_path}. "
                f"Zero-Trust policy: unsigned clients MUST NOT participate."
            )

        print(f"[Client {client_id}] Using device: {DEVICE}  |  "
              f"Train samples: {len(self.train_loader.dataset)}")

    # -- Flower interface ---------------------------------------------------
    def get_parameters(self, config) -> List:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List) -> None:
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.array(v)) for k, v in zip(self.model.state_dict().keys(), parameters)}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loss = train_one_epoch(self.model, self.train_loader, self.scaler)
        updated_params = self.get_parameters(config={})

        # Gate 2: Sign the model update (mandatory — no fallback)
        server_round = int(config.get("server_round", 0))
        metrics: dict = {"client_id": float(self.client_id)}
        sig = sign_parameters(updated_params, self.signing_key, server_round=server_round)
        metrics["signature"] = sig
        print(f"  → fit loss: {loss:.4f}  [signed ✓, round={server_round}]")

        return updated_params, len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = evaluate(self.model, self.test_loader)
        print(f"  → eval loss: {loss:.4f}  acc: {accuracy:.4f}")
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}


# ---------------------------------------------------------------------------
# mTLS certificate loader
# ---------------------------------------------------------------------------
CERT_DIR = Path(os.environ.get("CERT_DIR", "/certs"))


def _load_client_certificates(client_id: int) -> Optional[Tuple[bytes, bytes, bytes]]:
    """Load mTLS certificates for this client.

    Returns (ca_cert, client_cert, client_key) or None if certs are missing.
    Flower uses these to:
      - Verify the server certificate against the CA  (ca.crt)
      - Present this client's certificate to server   (client-{id}.crt)
      - Decrypt with this client's private key         (client-{id}.key)
    """
    ca_cert_path = CERT_DIR / "ca.crt"
    client_cert_path = CERT_DIR / f"client-{client_id}.crt"
    client_key_path = CERT_DIR / f"client-{client_id}.key"

    if not all(p.exists() for p in [ca_cert_path, client_cert_path, client_key_path]):
        print(f"[Client {client_id}] ⚠  Certificates not found – connecting WITHOUT mTLS")
        return None

    print(f"[Client {client_id}] 🔒 Loading mTLS certificates from {CERT_DIR}")
    ca_cert = ca_cert_path.read_bytes()
    client_cert = client_cert_path.read_bytes()
    client_key = client_key_path.read_bytes()
    print(f"[Client {client_id}] ✓  CA certificate loaded")
    print(f"[Client {client_id}] ✓  Client certificate loaded")
    print(f"[Client {client_id}] ✓  Client private key loaded")
    return ca_cert, client_cert, client_key


def _patch_grpc_for_mtls(ca_cert: bytes, client_cert: bytes, client_key: bytes) -> None:
    """Monkey-patch gRPC channel credentials to enable full mutual TLS.

    Flower's ``start_client`` only passes ``root_certificates`` (the CA cert)
    to ``grpc.ssl_channel_credentials()``.  For **mutual** TLS the client must
    also present its own certificate + private key so the server can verify
    the client's identity.  This patch intercepts the credential-creation call
    and injects the client cert/key pair.
    """
    _original_fn = grpc.ssl_channel_credentials

    def _mtls_ssl_channel_credentials(
        root_certificates=None, private_key=None, certificate_chain=None
    ):
        return _original_fn(
            root_certificates=root_certificates or ca_cert,
            private_key=client_key,
            certificate_chain=client_cert,
        )

    grpc.ssl_channel_credentials = _mtls_ssl_channel_credentials
    print("[mTLS] ✓  gRPC patched – client certificate will be presented on connect")


# ---------------------------------------------------------------------------
# Main – connect to Flower server
# ---------------------------------------------------------------------------
def main() -> None:
    # CLIENT_ID and NUM_CLIENTS are injected via environment variables
    client_id = int(os.environ.get("CLIENT_ID", "0"))
    num_clients = int(os.environ.get("NUM_CLIENTS", "2"))
    server_addr = os.environ.get("SERVER_ADDRESS", "server:8080")

    # Gate 1: Load mTLS certificates — MANDATORY (Zero Trust: deny by default)
    certs = _load_client_certificates(client_id)
    if certs is None:
        raise RuntimeError(
            f"[Client {client_id}] \u2717  FATAL: mTLS certificates missing. "
            f"Zero-Trust policy: unauthenticated clients MUST NOT connect."
        )
    ca_cert, client_cert, client_key = certs
    _patch_grpc_for_mtls(ca_cert, client_cert, client_key)
    root_certificates = ca_cert

    client = CifarClient(client_id, num_clients)
    fl.client.start_client(
        server_address=server_addr,
        client=client.to_client(),
        root_certificates=root_certificates,
    )


if __name__ == "__main__":
    main()
