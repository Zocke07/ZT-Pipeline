#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  generate_certs.sh – Create a full mTLS PKI for the FL pipeline
#
#  Produces:
#    certs/
#    ├── ca.crt / ca.key              Root Certificate Authority
#    ├── server.crt / server.key      Server certificate (SAN: server, localhost)
#    ├── client-0.crt / client-0.key  Client 0 certificate
#    └── client-1.crt / client-1.key  Client 1 certificate
#
#  All certs are signed by the same CA → mutual trust.
#  Validity: 365 days.  Key size: 4096-bit RSA.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

CERT_DIR="certs"
DAYS=365
KEY_SIZE=4096

echo "==> Creating certificate directory: ${CERT_DIR}/"
rm -rf "${CERT_DIR}"
mkdir -p "${CERT_DIR}"

# ── 1. Root Certificate Authority ──────────────────────────
echo "==> Generating Root CA..."
openssl genrsa -out "${CERT_DIR}/ca.key" ${KEY_SIZE} 2>/dev/null
openssl req -new -x509 \
    -key "${CERT_DIR}/ca.key" \
    -out "${CERT_DIR}/ca.crt" \
    -days ${DAYS} \
    -subj "/C=US/ST=Research/O=ZT-Pipeline/CN=FL-Root-CA"

# ── Helper: generate a cert signed by the CA ───────────────
generate_cert() {
    local NAME="$1"   # e.g. "server", "client-0"
    local CN="$2"     # Common Name
    local SAN="$3"    # Subject Alternative Names (DNS/IP)

    echo "==> Generating certificate for: ${NAME} (CN=${CN})"

    # Private key
    openssl genrsa -out "${CERT_DIR}/${NAME}.key" ${KEY_SIZE} 2>/dev/null

    # Certificate Signing Request
    openssl req -new \
        -key "${CERT_DIR}/${NAME}.key" \
        -out "${CERT_DIR}/${NAME}.csr" \
        -subj "/C=US/ST=Research/O=ZT-Pipeline/CN=${CN}"

    # Extensions file for SAN (required for modern TLS)
    cat > "${CERT_DIR}/${NAME}.ext" <<EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth, clientAuth
subjectAltName = ${SAN}
EOF

    # Sign with CA
    openssl x509 -req \
        -in "${CERT_DIR}/${NAME}.csr" \
        -CA "${CERT_DIR}/ca.crt" \
        -CAkey "${CERT_DIR}/ca.key" \
        -CAcreateserial \
        -out "${CERT_DIR}/${NAME}.crt" \
        -days ${DAYS} \
        -extfile "${CERT_DIR}/${NAME}.ext" \
        2>/dev/null

    # Cleanup temporaries
    rm -f "${CERT_DIR}/${NAME}.csr" "${CERT_DIR}/${NAME}.ext"
}

# ── 2. Server Certificate ──────────────────────────────────
#    SAN includes the Docker Compose service name "server",
#    plus "localhost" and 127.0.0.1 for local testing.
generate_cert "server" "fl-server" \
    "DNS:server,DNS:fl-server,DNS:localhost,IP:127.0.0.1"

# ── 3. Client Certificates ─────────────────────────────────
generate_cert "client-0" "fl-client-0" \
    "DNS:client-0,DNS:fl-client-0,DNS:localhost"

generate_cert "client-1" "fl-client-1" \
    "DNS:client-1,DNS:fl-client-1,DNS:malicious,DNS:fl-malicious,DNS:localhost"

# ── Cleanup CA serial file ─────────────────────────────────
rm -f "${CERT_DIR}/ca.srl"

# ── Summary ─────────────────────────────────────────────────
echo ""
echo "✓ mTLS certificates generated in ${CERT_DIR}/:"
ls -la "${CERT_DIR}/"
echo ""
echo "CA:       ${CERT_DIR}/ca.crt"
echo "Server:   ${CERT_DIR}/server.crt  +  ${CERT_DIR}/server.key"
echo "Client-0: ${CERT_DIR}/client-0.crt  +  ${CERT_DIR}/client-0.key"
echo "Client-1: ${CERT_DIR}/client-1.crt  +  ${CERT_DIR}/client-1.key"
echo ""
echo "All certificates are signed by the Root CA (${CERT_DIR}/ca.crt)."
echo "Validity: ${DAYS} days from today."
