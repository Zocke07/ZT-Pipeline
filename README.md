# ZT-Pipeline

**Designing a Zero-Trust MLOps Pipeline for Secure Federated Edge Learning**

Master's thesis project simulating and evaluating a Zero-Trust security architecture applied to a Federated Learning (FL) system. The project compares an insecure baseline versus a hardened ZT pipeline, both subjected to adversarial attacks, to quantify the security guarantees each approach provides.

---

## Overview

Federated Learning allows multiple clients to collaboratively train a shared model without sharing raw data. However, the aggregation server remains vulnerable to poisoning attacks from compromised clients. This project implements and tests a **three-gate Zero-Trust pipeline** that authenticates, validates, and filters every model update before aggregation.

### The Three Gates

| Gate | Mechanism | Threat Mitigated |
|------|-----------|-----------------|
| **Gate 1 — Identity** | Mutual TLS (mTLS) | Unauthenticated / impersonating clients |
| **Gate 2 — Integrity** | RSA-PSS digital signatures | Tampered model updates |
| **Gate 3 — Quality** | Cosine similarity + Z-score anomaly detection | Model poisoning attacks |

---

## Project Structure

```
ZT-Pipeline/
├── server.py                   # ZT aggregation server (three-gate pipeline)
├── client.py                   # Honest FL client
├── client_malicious.py         # Adversarial client (label_flip / targeted / noise / scale)
├── model.py                    # CNN model (CIFAR-10)
├── signing.py                  # RSA-PSS sign & verify helpers
├── data_utils.py               # Shared CIFAR-10 loading and IID partitioning
├── training.py                 # Shared training, evaluation, and GPU configuration
├── mtls.py                     # Shared mTLS certificate loading and gRPC patching
├── run_experiment.py           # Orchestrates Experiment A vs. Experiment B
├── verify_system_deployment.py # Pre-flight deployment checks
├── generate_certs.sh           # Generates mTLS certificates (Gate 1 key material)
├── generate_signing_keys.sh    # Generates RSA signing keys (Gate 2 key material)
├── Dockerfile                  # Container image (PyTorch 2.5.1, CUDA 12.4)
├── docker-compose.yml          # Service orchestration (server + clients)
├── experiment_override.yml     # Docker Compose overrides for experiment runs
├── requirements.txt            # Python dependencies
└── baseline_experiment/        # Insecure control-group implementation
    ├── baseline_server.py
    ├── baseline_client.py
    ├── baseline_malicious_client.py
    ├── docker-compose-baseline.yml
    └── docker-compose-baseline-attack.yml
```

---

## Technology Stack

- **Federated Learning:** [Flower](https://flower.ai/) (flwr 1.13.1) with FedAvg strategy
- **Deep Learning:** PyTorch 2.5.1, torchvision 0.20.1, CUDA 12.4, AMP mixed precision
- **Dataset:** CIFAR-10, IID-partitioned across clients
- **Security:** `cryptography` library (RSA-PSS), gRPC with mTLS channel credentials
- **Infrastructure:** Docker, Docker Compose, NVIDIA Container Toolkit (GPU passthrough)
- **Host OS:** Windows 11 with WSL2 + Docker Desktop

---

## Quick Start

### Prerequisites

- Windows 11 with WSL2 enabled
- Docker Desktop (WSL2 backend)
- NVIDIA GPU with driver ≥ 535.x and NVIDIA Container Toolkit

### 1. Generate Cryptographic Material

Run these once in Git Bash or WSL2:

```bash
bash generate_certs.sh          # mTLS certificates → certs/
bash generate_signing_keys.sh   # RSA signing keys  → signing_keys/
```

### 2. Build the Container Image

```powershell
docker compose build
```

### 3. Run the Full Experiment

```powershell
# Experiment B: Zero-Trust pipeline under attack
docker compose up

# Or run both experiments (baseline + ZT) via the orchestrator
python run_experiment.py
```

---

## Experiments

| Experiment | Description |
|------------|-------------|
| **A — Baseline (insecure)** | Standard FL with no authentication or validation; adversarial clients can freely poison the global model |
| **B — ZT Pipeline (secure)** | All three gates active; adversarial updates are detected and rejected before aggregation |

Results are written to `results.json` and compared to quantify the accuracy degradation each approach suffers under attack.

---

## Attack Modes

The adversarial client (`client_malicious.py`) supports four attack modes, controlled via the `ATTACK_TYPE` environment variable:

| Mode | Description |
|------|-------------|
| `label_flip` | Relabels all training samples to a target class |
| `targeted` | Flips one specific source class to a target class |
| `noise` | Injects Gaussian noise into model weights |
| `scale` | Scales model updates by a large factor (`POISON_SCALE`, default 100×) |

---

## Documentation

| File | Contents |
|------|----------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Full architecture, design decisions, security rationale, and thesis defence Q&A |
| [USER_MANUAL.md](USER_MANUAL.md) | Step-by-step setup, experiment execution, expected outputs, and troubleshooting |
| [TESTING_LOG.md](TESTING_LOG.md) | Record of all test runs and observed results |
