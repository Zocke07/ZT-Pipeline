"""Malicious Flower Client – Simulates adversarial participants.

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
       Effect: the local gradient points in the opposite direction to
       what honest training would produce.

  "targeted"
       Flips only a single source label to a target label:
           SOURCE_LABEL (env) → TARGET_LABEL (env)

  "noise"
       Adds large Gaussian noise directly to the post-training weights.

  "scale"
       Trains normally but multiplies all weights by POISON_SCALE
       (default: 100×), producing absurdly large update magnitudes.

Thesis note:
    This client deliberately passes cryptographic verification (Gates 1 & 2)
    to show that identity + integrity checks alone are insufficient.
    Gate 3 (behavioral / statistical detection) is required to complete the
    Zero-Trust pipeline.
"""

import os
import warnings
from pathlib import Path
from typing import List

import numpy as np
import flwr as fl
from torch.utils.data import DataLoader

from data_utils import get_cifar10, partition_data
from mtls import load_client_certificates, patch_grpc_for_mtls
from signing import sign_parameters, load_private_key
from training import (
    BATCH_SIZE,
    DEVICE,
    LEARNING_RATE,
    create_model,
    create_scaler,
    evaluate,
    get_parameters,
    set_parameters,
    train_one_epoch,
    train_one_epoch_with_label_transform,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SIGNING_KEY_DIR = Path(os.environ.get("SIGNING_KEY_DIR", "/signing_keys"))
CERT_DIR = Path(os.environ.get("CERT_DIR", "/certs"))

ATTACK_MODE = os.environ.get("ATTACK_MODE", "label_flip")
NOISE_SCALE = float(os.environ.get("NOISE_SCALE", "5.0"))
POISON_SCALE = float(os.environ.get("POISON_SCALE", "100.0"))
SOURCE_LABEL = int(os.environ.get("SOURCE_LABEL", "0"))
TARGET_LABEL = int(os.environ.get("TARGET_LABEL", "1"))
NUM_CLASSES = 10
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", "2"))


# ---------------------------------------------------------------------------
# Label-flip helpers
# ---------------------------------------------------------------------------

def _flip_labels_global(labels):
    """Invert every label: class i → class (NUM_CLASSES-1-i)."""
    return (NUM_CLASSES - 1) - labels


def _flip_labels_targeted(labels):
    """Flip only SOURCE_LABEL → TARGET_LABEL."""
    flipped = labels.clone()
    flipped[flipped == SOURCE_LABEL] = TARGET_LABEL
    return flipped


# ---------------------------------------------------------------------------
# Malicious Flower Client
# ---------------------------------------------------------------------------

class MaliciousClient(fl.client.NumPyClient):
    """Authenticated malicious client: passes Gates 1+2, defeated by Gate 3.

    The client holds valid mTLS certificates and a legitimate signing key,
    so it appears to be a trusted participant.  However:
      - In "label_flip" mode it trains on *inverted* labels.
      - In "targeted" mode it flips one source class to a target class.
      - In "noise"  mode it trains normally then corrupts weights with noise.
      - In "scale"  mode it trains normally then multiplies weights by a
        large factor (default 100×).
    """

    def __init__(self, client_id: int, num_clients: int) -> None:
        self.client_id = client_id
        self.model = create_model(compile_model=False)
        self.scaler = create_scaler()

        train_set, test_set = get_cifar10()
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
            self._flip_fn = None  # noise / scale mode
            if ATTACK_MODE == "scale":
                desc = f"weight SCALE  (×{POISON_SCALE})"
            else:
                desc = f"weight NOISE  (σ={NOISE_SCALE})"

        print(f"[MALICIOUS {client_id}] 🔴 ATTACK MODE: {ATTACK_MODE}  ({desc})")
        print(f"[MALICIOUS {client_id}] Device: {DEVICE}  |  "
              f"Train samples: {len(self.train_loader.dataset)}")

    # -- Flower interface ---------------------------------------------------

    def get_parameters(self, config) -> List:
        return get_parameters(self.model)

    def set_parameters(self, parameters: List) -> None:
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        if ATTACK_MODE in ("label_flip", "targeted"):
            # ── Attack: train on flipped labels ─────────────────────
            for _ in range(LOCAL_EPOCHS):
                loss = train_one_epoch_with_label_transform(
                    self.model, self.train_loader, self._flip_fn,
                )
            poisoned_params = self.get_parameters(config={})
            print(f"  [MALICIOUS] fit loss (on flipped labels): {loss:.4f}  "
                  f"({LOCAL_EPOCHS} epoch(s)) ← anti-correlated update")

        elif ATTACK_MODE == "scale":
            # ── Attack: honest training + scale weights ─────────────
            train_one_epoch(self.model, self.train_loader, self.scaler)
            poisoned_params = [w * POISON_SCALE for w in self.get_parameters(config={})]
            print(f"  [MALICIOUS] Trained normally then scaled weights by {POISON_SCALE}×")

        else:
            # ── Attack: honest training + noise injection ────────────
            for _ in range(LOCAL_EPOCHS):
                loss = train_one_epoch(self.model, self.train_loader, self.scaler)
            honest_params = self.get_parameters(config={})
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client_id = int(os.environ.get("CLIENT_ID", "1"))
    num_clients = int(os.environ.get("NUM_CLIENTS", "2"))
    server_addr = os.environ.get("SERVER_ADDRESS", "server:8080")

    # Gate 1
    certs = load_client_certificates(CERT_DIR, client_id)
    if certs is not None:
        ca_cert, client_cert, client_key = certs
        print(f"[MALICIOUS {client_id}] 🔒 mTLS certs loaded  → Gate 1 WILL pass")
        patch_grpc_for_mtls(ca_cert, client_cert, client_key)
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
