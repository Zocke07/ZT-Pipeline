"""Baseline (Insecure) Malicious Client – Attack Simulation.

This is the ATTACKER for the baseline experiment.  It connects to the
insecure baseline server and sends poisoned model updates.

Because the baseline has NO security gates:
  ✗  No mTLS           → anyone can connect
  ✗  No signatures     → updates are not verified
  ✗  No anomaly det.   → all updates are blindly aggregated

Attack modes (set via ATTACK_MODE env var):
  "label_flip" (default) – Invert every label: class i → class (9−i)
  "targeted"             – Flip only SOURCE_LABEL → TARGET_LABEL
  "noise"                – Train normally, then add Gaussian noise
  "scale"                – Train normally, then multiply weights by factor

Identical to the ZT attacker (client_malicious.py) for fair comparison.
The ONLY difference: no signing, no mTLS.
"""

import os
import warnings
from typing import List

import numpy as np
import flwr as fl

from data_utils import make_dataloaders, flip_labels_global, flip_labels_targeted
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
    train_one_epoch_with_label_transform,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration (mirrors client_malicious.py for fair comparison)
# ---------------------------------------------------------------------------
ATTACK_MODE = os.environ.get("ATTACK_MODE", "label_flip")
NUM_CLASSES = 10
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", "2"))
NOISE_SCALE = float(os.environ.get("NOISE_SCALE", "5.0"))
POISON_SCALE = float(os.environ.get("POISON_SCALE", "100.0"))
SOURCE_LABEL = int(os.environ.get("SOURCE_LABEL", "0"))
TARGET_LABEL = int(os.environ.get("TARGET_LABEL", "1"))



# ---------------------------------------------------------------------------
# Malicious Flower Client (INSECURE – no signing, no mTLS)
# ---------------------------------------------------------------------------
class BaselineMaliciousClient(fl.client.NumPyClient):
    """Insecure malicious client for the baseline experiment.

    Supports all 4 attack modes (same logic as ZT MaliciousClient):
      - label_flip: train on inverted labels
      - targeted:   flip source→target label
      - noise:      train normally, add Gaussian noise to weights
      - scale:      train normally, multiply weights by large factor

    No signing key, no mTLS → baseline server accepts everything.
    """

    def __init__(self, client_id: int, num_clients: int) -> None:
        self.client_id = client_id
        self.model = create_model(compile_model=False)
        self.scaler = create_scaler()

        # Data loading
        _seed = int(os.environ.get("SEED", "0"))
        _alpha_str = os.environ.get("DIRICHLET_ALPHA", "")
        self.train_loader, self.test_loader = make_dataloaders(
            client_id, num_clients,
            dirichlet_alpha=float(_alpha_str) if _alpha_str else None,
            seed=_seed,
            batch_size=BATCH_SIZE,
            num_workers=0,
        )

        # Select attack function
        if ATTACK_MODE == "label_flip":
            self._flip_fn = flip_labels_global
            desc = "GLOBAL label flip  (label → 9-label)"
        elif ATTACK_MODE == "targeted":
            self._flip_fn = lambda labels: flip_labels_targeted(
                labels, SOURCE_LABEL, TARGET_LABEL
            )
            desc = f"TARGETED flip  ({SOURCE_LABEL} → {TARGET_LABEL})"
        else:
            self._flip_fn = None
            if ATTACK_MODE == "scale":
                desc = f"weight SCALE  (×{POISON_SCALE})"
            else:
                desc = f"weight NOISE  (σ={NOISE_SCALE})"

        print(f"[BASELINE MALICIOUS {client_id}] ATTACK MODE: {ATTACK_MODE}  ({desc})")
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

        if ATTACK_MODE in ("label_flip", "targeted"):
            for _ in range(LOCAL_EPOCHS):
                loss = train_one_epoch_with_label_transform(
                    self.model, self.train_loader, self._flip_fn,
                )
            poisoned_params = self.get_parameters(config={})
            print(f"  [BASELINE MALICIOUS] fit loss (flipped labels): {loss:.4f}  "
                  f"({LOCAL_EPOCHS} epoch(s))  [UNSIGNED, ACCEPTED BLINDLY]")

        elif ATTACK_MODE == "scale":
            train_one_epoch(self.model, self.train_loader, self.scaler)
            poisoned_params = [w * POISON_SCALE for w in self.get_parameters(config={})]
            print(f"  [BASELINE MALICIOUS] Trained then scaled ×{POISON_SCALE}  "
                  f"[UNSIGNED, ACCEPTED BLINDLY]")

        else:  # noise
            for _ in range(LOCAL_EPOCHS):
                loss = train_one_epoch(self.model, self.train_loader, self.scaler)
            honest_params = self.get_parameters(config={})
            poisoned_params = [
                w + np.random.randn(*w.shape).astype(np.float32) * NOISE_SCALE
                for w in honest_params
            ]
            print(f"  [BASELINE MALICIOUS] honest loss: {loss:.4f}  "
                  f"→ added noise (σ={NOISE_SCALE})  [UNSIGNED, ACCEPTED BLINDLY]")

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

    # Reproducibility: seed control
    seed = int(os.environ.get("SEED", "-1"))
    if seed >= 0:
        effective_seed = seed * 10000 + client_id
        set_seed(effective_seed)
        print(f"[BASELINE MALICIOUS {client_id}] Seed set: {effective_seed} (base={seed})")

    print(f"[BASELINE MALICIOUS {client_id}] Connecting INSECURELY to {server_addr}")

    client = BaselineMaliciousClient(client_id, num_clients)
    fl.client.start_client(
        server_address=server_addr,
        client=client.to_client(),
        insecure=True,
    )


if __name__ == "__main__":
    main()
