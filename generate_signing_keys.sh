#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  generate_signing_keys.sh – RSA key pairs for Gate 2 signing
#
#  Produces:
#    signing_keys/
#    ├── client-0.private.pem   (kept by client-0 only)
#    ├── client-0.public.pem    (shared with server)
#    ├── client-1.private.pem   (kept by client-1 only)
#    └── client-1.public.pem    (shared with server)
#
#  Key size: 2048-bit RSA (sufficient for signing model updates).
# ──────────────────────────────────────────────────────────────
set -euo pipefail

KEY_DIR="signing_keys"
KEY_SIZE=2048
NUM_CLIENTS="${1:-2}"

echo "==> Creating signing key directory: ${KEY_DIR}/"
rm -rf "${KEY_DIR}"
mkdir -p "${KEY_DIR}"

for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    NAME="client-${i}"
    echo "==> Generating RSA-${KEY_SIZE} key pair for ${NAME}..."

    # Generate private key (PKCS#8 PEM, unencrypted)
    openssl genpkey -algorithm RSA \
        -pkeyopt "rsa_keygen_bits:${KEY_SIZE}" \
        -out "${KEY_DIR}/${NAME}.private.pem" \
        2>/dev/null

    # Extract public key
    openssl pkey -in "${KEY_DIR}/${NAME}.private.pem" \
        -pubout \
        -out "${KEY_DIR}/${NAME}.public.pem" \
        2>/dev/null

    echo "    ✓ ${KEY_DIR}/${NAME}.private.pem"
    echo "    ✓ ${KEY_DIR}/${NAME}.public.pem"
done

echo ""
echo "✓ Signing keys generated in ${KEY_DIR}/:"
ls -la "${KEY_DIR}/"
echo ""
echo "Mount into containers via docker-compose:"
echo "  Server  → needs ALL public keys  (./signing_keys:/signing_keys:ro)"
echo "  Clients → need ONLY their own private key (mounted individually)"
