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
├── server_utils.py             # Shared server utilities (metrics, Krum, model saving)
├── client.py                   # Honest FL client
├── client_malicious.py         # Adversarial client (label_flip / targeted / noise / scale)
├── model.py                    # CifarCNN model definition
├── signing.py                  # RSA-PSS sign & verify helpers
├── data_utils.py               # CIFAR-10 loading, partitioning, DataLoader factory
├── training.py                 # Training loop, evaluation, and GPU configuration
├── mtls.py                     # mTLS certificate loading and gRPC channel patching
├── generate_keys.py            # Generates mTLS certs and RSA signing keys
├── run_experiments.py          # Primary experiment orchestrator (multi-seed, multi-config)
├── run_experiment.py           # Single-run orchestrator (legacy; kept for reference)
├── verify_system_deployment.py # Pre-flight deployment checks
├── Dockerfile                  # Container image (PyTorch 2.5.1, CUDA 12.4)
├── docker-compose.yml          # Service orchestration (server + clients)
├── experiment_override.yml     # Docker Compose overrides for experiment parameters
├── requirements.txt            # Python dependencies
├── tracking/
│   └── mlflow_logger.py        # MLflow ExperimentTracker and NullTracker wrappers
└── baseline_experiment/        # Insecure control-group implementation
    ├── baseline_server.py
    ├── baseline_client.py
    ├── baseline_malicious_client.py
    ├── docker-compose-baseline.yml
    └── docker-compose-baseline-attack.yml
```

---

## Technology Stack

- **Federated Learning:** [Flower](https://flower.ai/) (flwr 1.13.1) with custom ZeroTrustFedAvg strategy
- **Deep Learning:** PyTorch 2.5.1, torchvision 0.20.1, CUDA 12.4, AMP mixed precision, `torch.compile`
- **Dataset:** CIFAR-10, Dirichlet-partitioned across clients (configurable α)
- **Security:** `cryptography` library (RSA-PSS), gRPC with mTLS channel credentials
- **Experiment tracking:** MLflow (local tracking server, auto-logged per run)
- **Infrastructure:** Docker, Docker Compose, NVIDIA Container Toolkit (GPU passthrough)
- **Host OS:** Windows 11 with WSL2 + Docker Desktop

---

## Quick Start

### Prerequisites

- Windows 11 with WSL2 enabled
- Docker Desktop (WSL2 backend)
- NVIDIA GPU with driver ≥ 535.x and NVIDIA Container Toolkit

### 1. Generate Cryptographic Material

Run once from the repository root:

```powershell
python generate_keys.py         # mTLS certs → certs/   RSA keys → signing_keys/
```

### 2. Build the Container Image

```powershell
docker compose build
```

### 3. Run the Experiment Suite

```powershell
# Recommended: full multi-seed, multi-config experiment run
python run_experiments.py --preset smoke

# Single ad-hoc run (ZT pipeline under label-flip attack)
docker compose up
```

Results land in `experiment_results/` and are tracked in MLflow (`mlruns/`).

---

## Experiments

`run_experiments.py` executes a matrix of configurations across multiple random seeds and produces aggregated detection metrics.

| Configuration | Gates active | Aggregation | Attack |
|--------------|-------------|-------------|--------|
| baseline_clean | none | FedAvg | none |
| baseline_attack | none | FedAvg | configurable |
| zt_no_gates | none | FedAvg | configurable |
| zt_gate1_only | Gate 1 | FedAvg | configurable |
| zt_gates12 | Gate 1+2 | FedAvg | configurable |
| zt_full | Gate 1+2+3 | FedAvg | configurable |
| zt_full_krum | Gate 1+2+3 | Krum | configurable |
| zt_full_multikrum | Gate 1+2+3 | Multi-Krum | configurable |

Final metrics (accuracy, detection rate, precision, recall, F1) are written to `experiment_results/<preset>/`.

---

## Attack Modes

The adversarial client (`client_malicious.py`) supports four attack modes, controlled via the `ATTACK_MODE` environment variable:

| `ATTACK_MODE` | Description |
|--------------|-------------|
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
