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
from pathlib import Path
from typing import List

import flwr as fl

from data_utils import make_dataloaders
from mtls import load_client_certificates, patch_grpc_for_mtls
from signing import sign_parameters, load_private_key
from training import (
    BATCH_SIZE,
    DEVICE,
    create_model,
    create_scaler,
    evaluate,
    get_parameters,
    set_parameters,
    set_seed,
    train_one_epoch,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration (from environment)
# ---------------------------------------------------------------------------
SIGNING_KEY_DIR = Path(os.environ.get("SIGNING_KEY_DIR", "/signing_keys"))
CERT_DIR = Path(os.environ.get("CERT_DIR", "/certs"))


# ---------------------------------------------------------------------------
# Flower client
# ---------------------------------------------------------------------------
class CifarClient(fl.client.NumPyClient):
    """Flower NumPyClient wrapping the PyTorch CNN.

    Gate 2 enhancement: ``fit()`` signs the serialized weight update
    with the client's RSA private key and attaches the base64 signature
    + client_id in the metrics dict so the server can verify integrity.
    """

    def __init__(self, client_id: int, num_clients: int) -> None:
        self.client_id = client_id
        self.model = create_model(compile_model=True)
        self.scaler = create_scaler()

        # Data loading
        _seed = int(os.environ.get("SEED", "0"))
        _alpha_str = os.environ.get("DIRICHLET_ALPHA", "")
        self.train_loader, self.test_loader = make_dataloaders(
            client_id, num_clients,
            dirichlet_alpha=float(_alpha_str) if _alpha_str else None,
            seed=_seed,
            batch_size=BATCH_SIZE,
            pin_memory=(DEVICE.type == "cuda"),
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
        return get_parameters(self.model)

    def set_parameters(self, parameters: List) -> None:
        set_parameters(self.model, parameters)

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
# Main – connect to Flower server
# ---------------------------------------------------------------------------
def main() -> None:
    client_id = int(os.environ.get("CLIENT_ID", "0"))
    num_clients = int(os.environ.get("NUM_CLIENTS", "2"))
    server_addr = os.environ.get("SERVER_ADDRESS", "server:8080")

    # Reproducibility: seed control
    seed = int(os.environ.get("SEED", "-1"))
    if seed >= 0:
        effective_seed = seed * 10000 + client_id
        set_seed(effective_seed)
        print(f"[Client {client_id}] Seed set: {effective_seed} (base={seed})")

    # Gate 1: Load mTLS certificates — MANDATORY (Zero Trust: deny by default)
    certs = load_client_certificates(CERT_DIR, client_id)
    if certs is None:
        raise RuntimeError(
            f"[Client {client_id}] ✗  FATAL: mTLS certificates missing. "
            f"Zero-Trust policy: unauthenticated clients MUST NOT connect."
        )
    ca_cert, client_cert, client_key = certs
    print(f"[Client {client_id}] 🔒 mTLS certificates loaded from {CERT_DIR}")
    print(f"[Client {client_id}] ✓  CA certificate loaded")
    print(f"[Client {client_id}] ✓  Client certificate loaded")
    print(f"[Client {client_id}] ✓  Client private key loaded")
    patch_grpc_for_mtls(ca_cert, client_cert, client_key)
    root_certificates = ca_cert

    client = CifarClient(client_id, num_clients)
    fl.client.start_client(
        server_address=server_addr,
        client=client.to_client(),
        root_certificates=root_certificates,
    )


if __name__ == "__main__":
    main()
