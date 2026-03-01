"""Shared mTLS certificate loading and gRPC patching utilities.

Used by all Zero-Trust client variants (client.py, client_malicious.py)
that need to present client certificates during the TLS handshake.
"""

from pathlib import Path
from typing import Optional, Tuple

import grpc


def load_client_certificates(
    cert_dir: Path,
    client_id: int,
) -> Optional[Tuple[bytes, bytes, bytes]]:
    """Load mTLS certificates for a client.

    Returns ``(ca_cert, client_cert, client_key)`` or ``None`` if any
    required file is missing.

    Flower uses these to:
      - Verify the server certificate against the CA  (ca.crt)
      - Present this client's certificate to server   (client-{id}.crt)
      - Decrypt with this client's private key         (client-{id}.key)
    """
    ca_cert_path = cert_dir / "ca.crt"
    client_cert_path = cert_dir / f"client-{client_id}.crt"
    client_key_path = cert_dir / f"client-{client_id}.key"

    if not all(p.exists() for p in [ca_cert_path, client_cert_path, client_key_path]):
        return None

    ca_cert = ca_cert_path.read_bytes()
    client_cert = client_cert_path.read_bytes()
    client_key = client_key_path.read_bytes()
    return ca_cert, client_cert, client_key


def patch_grpc_for_mtls(
    ca_cert: bytes,
    client_cert: bytes,
    client_key: bytes,
) -> None:
    """Monkey-patch gRPC channel credentials to enable full mutual TLS.

    Flower's ``start_client`` only passes ``root_certificates`` (the CA cert)
    to ``grpc.ssl_channel_credentials()``.  For **mutual** TLS the client must
    also present its own certificate + private key so the server can verify
    the client's identity.  This patch intercepts the credential-creation call
    and injects the client cert/key pair.
    """
    _original_fn = grpc.ssl_channel_credentials

    def _mtls_ssl_channel_credentials(
        root_certificates=None, private_key=None, certificate_chain=None,
    ):
        return _original_fn(
            root_certificates=root_certificates or ca_cert,
            private_key=client_key,
            certificate_chain=client_cert,
        )

    grpc.ssl_channel_credentials = _mtls_ssl_channel_credentials
    print("[mTLS] ✓  gRPC patched – client certificate will be presented on connect")
