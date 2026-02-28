"""Gate 2 – Model-update signing and verification utilities.

Uses RSA-PSS (SHA-256) via the ``cryptography`` library.

Workflow
--------
Client side:
    1.  Serialize the NumPy weight list to bytes  (``serialize_parameters``).
    2.  Sign with the client's RSA private key     (``sign_parameters``).
    3.  Attach the base64-encoded signature + client_id in the Flower
        ``fit()`` metrics dict.

Server side:
    1.  Look up the client's public key by ``client_id``.
    2.  Verify the signature against the received weights
        (``verify_signature``).
    3.  If invalid → reject the update before aggregation.
"""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
from typing import List, Optional

import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa, utils


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def serialize_parameters(parameters: List[np.ndarray]) -> bytes:
    """Deterministically serialize a list of NumPy arrays to bytes.

    Uses ``numpy.save`` in a BytesIO buffer so the exact same bytes are
    produced on both the client (signing) and server (verifying) side.
    """
    buf = io.BytesIO()
    for arr in parameters:
        np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _digest(data: bytes) -> bytes:
    """Compute SHA-256 digest (used for pre-hashing before RSA-PSS)."""
    return hashlib.sha256(data).digest()


# ---------------------------------------------------------------------------
# Key generation  (used by generate_signing_keys.py)
# ---------------------------------------------------------------------------

def generate_key_pair(key_dir: Path, name: str, key_size: int = 2048) -> None:
    """Generate an RSA key pair and write PEM files to *key_dir*.

    Produces:
        ``{name}.private.pem``  – PKCS8 private key (unencrypted)
        ``{name}.public.pem``   – SubjectPublicKeyInfo public key
    """
    key_dir.mkdir(parents=True, exist_ok=True)
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)

    (key_dir / f"{name}.private.pem").write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    (key_dir / f"{name}.public.pem").write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )


# ---------------------------------------------------------------------------
# Signing  (client side)
# ---------------------------------------------------------------------------

def load_private_key(pem_path: Path):
    """Load an RSA private key from a PEM file."""
    return serialization.load_pem_private_key(pem_path.read_bytes(), password=None)


def sign_parameters(parameters: List[np.ndarray], private_key) -> str:
    """Sign serialized parameters and return a base64-encoded signature.

    Steps:
        1. Serialize the weight arrays to bytes.
        2. Pre-hash with SHA-256 (keeps memory bounded for large models).
        3. Sign the digest with RSA-PSS + SHA-256.
        4. Return base64-encoded signature (safe for Flower metrics dict).
    """
    data = serialize_parameters(parameters)
    digest = _digest(data)
    signature = private_key.sign(
        digest,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        utils.Prehashed(hashes.SHA256()),
    )
    return base64.b64encode(signature).decode("ascii")


# ---------------------------------------------------------------------------
# Verification  (server side)
# ---------------------------------------------------------------------------

def load_public_key(pem_path: Path):
    """Load an RSA public key from a PEM file."""
    return serialization.load_pem_public_key(pem_path.read_bytes())


def verify_signature(
    parameters: List[np.ndarray],
    signature_b64: str,
    public_key,
) -> bool:
    """Verify an RSA-PSS signature over the serialized parameters.

    Returns ``True`` if the signature is valid, ``False`` otherwise.
    """
    data = serialize_parameters(parameters)
    digest = _digest(data)
    signature = base64.b64decode(signature_b64)
    try:
        public_key.verify(
            signature,
            digest,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            utils.Prehashed(hashes.SHA256()),
        )
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------

def load_client_public_keys(keys_dir: Path, num_clients: int) -> dict:
    """Load all client public keys into a dict keyed by client_id (str).

    Expected files: ``{keys_dir}/client-{id}.public.pem``
    """
    pub_keys: dict = {}
    for cid in range(num_clients):
        path = keys_dir / f"client-{cid}.public.pem"
        if path.exists():
            pub_keys[str(cid)] = load_public_key(path)
            print(f"[Gate 2] ✓  Loaded public key for client-{cid}")
        else:
            print(f"[Gate 2] ⚠  Public key missing for client-{cid}: {path}")
    return pub_keys
