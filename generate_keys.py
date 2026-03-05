#!/usr/bin/env python3
"""Generate mTLS certificates and RSA signing keys for N FL clients.

Uses the ``cryptography`` library — no OpenSSL CLI dependency.
Can be imported by ``run_experiments.py`` or run standalone::

    python generate_keys.py --num-clients 5 --cert-dir certs --key-dir signing_keys
"""

from __future__ import annotations

import argparse
import ipaddress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


# ---------------------------------------------------------------------------
# mTLS certificate generation
# ---------------------------------------------------------------------------

def _write_key(path: Path, key) -> None:
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )


def _write_cert(path: Path, cert) -> None:
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def _make_name(cn: str) -> x509.Name:
    return x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Research"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ZT-Pipeline"),
        x509.NameAttribute(NameOID.COMMON_NAME, cn),
    ])


def _generate_signed_cert(
    cert_dir: Path,
    name: str,
    cn: str,
    san_dns: List[str],
    san_ips: List[str],
    ca_key,
    ca_cert,
    key_size: int,
    validity_days: int,
) -> None:
    """Issue a certificate signed by *ca_key*/*ca_cert*."""
    now = datetime.now(timezone.utc)
    key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    subject = _make_name(cn)

    san_entries: list = [x509.DNSName(d) for d in san_dns]
    san_entries += [x509.IPAddress(ipaddress.ip_address(ip)) for ip in san_ips]

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=validity_days))
        .add_extension(
            x509.SubjectAlternativeName(san_entries), critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([
                ExtendedKeyUsageOID.SERVER_AUTH,
                ExtendedKeyUsageOID.CLIENT_AUTH,
            ]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    _write_key(cert_dir / f"{name}.key", key)
    _write_cert(cert_dir / f"{name}.crt", cert)


def generate_mtls_certs(
    cert_dir: Path,
    num_clients: int,
    key_size: int = 4096,
    validity_days: int = 365,
) -> None:
    """Generate a CA, server cert, and *num_clients* client certificates."""
    cert_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)

    # 1. Root CA ---------------------------------------------------------
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    ca_name = _make_name("FL-Root-CA")
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=validity_days))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None), critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    _write_key(cert_dir / "ca.key", ca_key)
    _write_cert(cert_dir / "ca.crt", ca_cert)

    # 2. Server cert -----------------------------------------------------
    _generate_signed_cert(
        cert_dir, "server", "fl-server",
        san_dns=["server", "fl-server", "localhost"],
        san_ips=["127.0.0.1"],
        ca_key=ca_key, ca_cert=ca_cert,
        key_size=key_size, validity_days=validity_days,
    )

    # 3. Client certs ----------------------------------------------------
    for i in range(num_clients):
        name = f"client-{i}"
        san_dns = [name, f"fl-{name}", "localhost"]
        _generate_signed_cert(
            cert_dir, name, f"fl-{name}",
            san_dns=san_dns, san_ips=[],
            ca_key=ca_key, ca_cert=ca_cert,
            key_size=key_size, validity_days=validity_days,
        )

    print(f"[generate_keys] mTLS certs generated in {cert_dir}/ "
          f"(CA + server + {num_clients} clients)")


# ---------------------------------------------------------------------------
# RSA signing key generation
# ---------------------------------------------------------------------------

def generate_signing_keys(
    key_dir: Path,
    num_clients: int,
    key_size: int = 2048,
) -> None:
    """Generate RSA signing key-pairs (private + public) for *num_clients*."""
    key_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_clients):
        name = f"client-{i}"
        key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)

        (key_dir / f"{name}.private.pem").write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        (key_dir / f"{name}.public.pem").write_bytes(
            key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )

    print(f"[generate_keys] Signing key-pairs generated in {key_dir}/ "
          f"({num_clients} clients)")


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def generate_all(
    cert_dir: Path,
    key_dir: Path,
    num_clients: int,
    cert_key_size: int = 4096,
    signing_key_size: int = 2048,
) -> None:
    """Generate both mTLS certificates and signing keys."""
    generate_mtls_certs(cert_dir, num_clients, key_size=cert_key_size)
    generate_signing_keys(key_dir, num_clients, key_size=signing_key_size)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate FL experiment credentials")
    ap.add_argument("--num-clients", type=int, default=2)
    ap.add_argument("--cert-dir", type=Path, default=Path("certs"))
    ap.add_argument("--key-dir", type=Path, default=Path("signing_keys"))
    ap.add_argument("--cert-key-size", type=int, default=4096)
    ap.add_argument("--signing-key-size", type=int, default=2048)
    args = ap.parse_args()

    generate_all(
        cert_dir=args.cert_dir,
        key_dir=args.key_dir,
        num_clients=args.num_clients,
        cert_key_size=args.cert_key_size,
        signing_key_size=args.signing_key_size,
    )


if __name__ == "__main__":
    main()
