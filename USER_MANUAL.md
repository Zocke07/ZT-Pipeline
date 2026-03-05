# ZT-Pipeline — User Manual & Runbook

> **Designing a Zero-Trust MLOps Pipeline for Secure Federated Edge Learning**
>
> Master's Thesis Project — Step-by-step guide to set up, run, and verify
> the complete Zero-Trust Federated Learning simulation.
>
> **Audience:** Anyone who needs to reproduce every experiment — from a
> first-year student following along to a thesis examiner verifying claims.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation Guide](#2-installation-guide)
3. [Running the Simulation](#3-running-the-simulation)
4. [Expected Output & Verification](#4-expected-output--verification)
5. [Troubleshooting FAQ](#5-troubleshooting-faq)

---

## 1. Prerequisites

Before you begin, make sure every item on this checklist is in place.
If any item is missing, the simulation **will not work**.

### 1.1 Hardware Requirements

| Component | Minimum | Used in This Project |
|-----------|---------|----------------------|
| **NVIDIA GPU** | Any CUDA-capable GPU (Compute Capability ≥ 7.0) | **Development:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM, Ada Lovelace) · **Production/Lab:** NVIDIA RTX 4000 Ada Generation |
| RAM | 8 GB | 16 GB recommended |
| Disk | 10 GB free | Docker images (~6 GB) + CIFAR-10 dataset (~170 MB) + certificates |

> **Note on GPUs:** The project was developed and tested on an RTX 4050 Laptop GPU.
> For the final thesis experiments, an NVIDIA RTX 4000 Ada Generation (lab workstation)
> will be used. Both are Ada Lovelace architecture (Compute Capability 8.9) and fully
> compatible with the CUDA 12.4 container image. Any NVIDIA GPU from the Pascal
> generation (GTX 10-series) onwards should work, though Tensor Core optimisations
> (TF32, FP16 AMP) require Volta (V100) or newer.

### 1.2 Software Requirements

| Software | Required Version | Purpose | Installation Link |
|----------|-----------------|---------|-------------------|
| **Windows 11** | 22H2 or later | Host operating system | — |
| **WSL2** (Windows Subsystem for Linux) | Ubuntu 22.04 recommended | Linux kernel for Docker containers | [Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install) |
| **Docker Desktop** | 4.x with WSL2 backend | Container runtime (builds and runs simulation containers) | [Install Docker Desktop](https://docs.docker.com/desktop/install/windows-install/) |
| **NVIDIA GPU Driver** (Windows) | ≥ 535.x (Game Ready or Studio) | GPU access from WSL2 — **do NOT install a separate driver inside WSL2** | [NVIDIA Drivers](https://www.nvidia.com/drivers) |
| **NVIDIA Container Toolkit** | Latest stable | Allows Docker containers to access the GPU | [Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| **Git** | Any recent version | Clone the repository | [Git for Windows](https://git-scm.com/download/win) |
| **Git Bash** or **WSL2 bash** | Included with Git for Windows | Run `.sh` scripts (certificate & key generation) | Included with Git |
| **OpenSSL** | ≥ 1.1.1 | Generate mTLS certificates and RSA signing keys | Included in Git Bash and WSL2 |

> **Important:** You do **not** need Python, PyTorch, or CUDA installed on your host
> machine. Everything runs inside Docker containers that bundle their own Python 3,
> PyTorch 2.5.1=, and CUDA 12.4 runtime.

### 1.3 Pre-Flight Check — Verify Your GPU Setup

Before proceeding, run these commands to confirm your GPU is visible to Docker.
**If any check fails, do not continue** — see [Section 5 (Troubleshooting)](#5-troubleshooting-faq).

**Step 1 — Check GPU on host:**

```powershell
nvidia-smi
```

You should see your GPU listed (e.g., `NVIDIA GeForce RTX 4050 Laptop GPU`) with
a CUDA version ≥ 12.0.

**Step 2 — Check GPU inside Docker:**

```powershell
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

You should see the same GPU listed inside the container. If you get an error about
device drivers, see [FAQ Q1](#q1-docker-says-could-not-select-device-driver--with-capabilities-gpu).

**Step 3 — Check Docker runtime:**

```powershell
docker info | Select-String "Runtimes"
```

Output must include `nvidia`:

```
 Runtimes: io.containerd.runc.v2 nvidia runc
```

✅ If all three checks pass, your environment is ready.

---

## 2. Installation Guide

### 2.1 Clone the Repository

```powershell
# Pick a working directory (example)
cd D:\Z\Master\Code\Exp

# Clone
git clone <repository-url> ZT-Pipeline
cd ZT-Pipeline
```

After cloning, your directory should look like this:

```
ZT-Pipeline/
├── model.py
├── server.py
├── client.py
├── client_malicious.py
├── signing.py
├── data_utils.py
├── training.py
├── mtls.py
├── generate_keys.py
├── server_utils.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── ...
```

### 2.2 Generate the Zero-Trust Cryptographic Material

This is the **most critical step**. Without certificates and signing keys, the
entire security pipeline refuses to start (deny-by-default design).

From the repository root, run:

```powershell
python generate_keys.py
```

**What this does:**

1. Creates a **Root Certificate Authority** (`ca.crt` / `ca.key`) — the trust anchor.
2. Generates a **server certificate** (`server.crt` / `server.key`) signed by the CA.
3. Generates **per-client certificates** (`client-0.crt`, `client-1.crt`)
   signed by the same CA, with SANs matching Docker Compose service names.
4. Generates a **2048-bit RSA key pair** for each client.
   Private keys (`*.private.pem`) stay with each client; public keys go to the server.

**Expected output:**

```
✓ mTLS certificates generated in certs/:
CA:       certs/ca.crt
Server:   certs/server.crt  +  certs/server.key
Client-0: certs/client-0.crt  +  certs/client-0.key
Client-1: certs/client-1.crt  +  certs/client-1.key

✓ Signing keys generated in signing_keys/:
    signing_keys/client-0.private.pem
    signing_keys/client-0.public.pem
    signing_keys/client-1.private.pem
    signing_keys/client-1.public.pem
```

**Verify** (optional):

```bash
openssl verify -CAfile certs/ca.crt certs/server.crt
# Expected: certs/server.crt: OK

openssl verify -CAfile certs/ca.crt certs/client-0.crt
# Expected: certs/client-0.crt: OK
```

After both scripts, your `certs/` and `signing_keys/` directories should be populated:

```
certs/
├── ca.crt          ← Root CA certificate (trust anchor)
├── ca.key          ← Root CA private key (NEVER mounted into containers)
├── server.crt      ← Server identity certificate
├── server.key      ← Server private key
├── client-0.crt    ← Client 0 identity certificate
├── client-0.key    ← Client 0 TLS private key
├── client-1.crt    ← Client 1 identity certificate (also used by malicious client)
└── client-1.key    ← Client 1 TLS private key

signing_keys/
├── client-0.private.pem   ← Client 0 signing key (stays with client-0 only)
├── client-0.public.pem    ← Client 0 public key (shared with server)
├── client-1.private.pem   ← Client 1 signing key (stays with client-1 only)
└── client-1.public.pem    ← Client 1 public key (shared with server)
```

> **Security note:** The CA private key (`ca.key`) and each client's private keys
> are **never** mounted into containers other than their own. This is the
> **least-privilege** principle enforced at the Docker volume mount level.

### 2.3 Build the Docker Image

Back in **PowerShell** (your normal terminal):

```powershell
cd D:\Z\Master\Code\Exp\ZT-Pipeline

docker compose build
```

This builds a single Docker image based on `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
with Python 3, PyTorch 2.5.1, Flower 1.13.1, and all project code.

**Expected output** (first build takes 5–10 minutes; subsequent builds are cached):

```
[+] Building 3/3
 ✔ server    Built
 ✔ client-0  Built
 ✔ malicious Built
```

> **Tip:** If the build fails with network errors, check your internet connection.
> PyTorch wheels are ~800 MB and are downloaded from `download.pytorch.org`.

### 2.4 Pre-Download the CIFAR-10 Dataset

The `data-init` service downloads CIFAR-10 into a shared Docker volume so that
clients don't race to download it simultaneously:

```powershell
docker compose up data-init
```

**Expected output:**

```
fl-data-init  | Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
fl-data-init  | Files already downloaded and verified
fl-data-init  | CIFAR-10 ready.
fl-data-init exited with code 0
```

You only need to run this once. The dataset persists in the `cifar-data` Docker
volume until you explicitly remove it with `docker compose down -v`.

---

## 3. Running the Simulation

### Scenario A: Normal Operation (2 Honest Clients — Clean Baseline)

This scenario runs the full Zero-Trust pipeline with **two honest clients**.
No attacks. All three security gates should pass, and the model should converge
to approximately **65–70% accuracy** on CIFAR-10 within 3 rounds.

```powershell
docker compose --profile clean up --abort-on-container-exit server client-0 client-1
```

**What happens:**

1. The **server** starts and loads mTLS certificates and signing public keys.
2. **Client-0** (5-second delay) and **Client-1** (5-second delay) connect via mTLS.
3. Each round:
   - Server broadcasts the global model.
   - Both clients train 1 local epoch on their CIFAR-10 partition.
   - Both clients sign their weight updates with RSA-PSS.
   - Server verifies signatures (**Gate 2**) — both pass.
   - Server runs anomaly detection (**Gate 3**) — both pass (correlated updates).
   - Server aggregates via FedAvg.
4. After 3 rounds, all containers exit.

**What to look for in the logs:**

> `[Insert Screenshot: Server terminal showing "Gate 2 Result: 2 passed, 0 rejected" and "Gate 3 Result: 2 accepted, 0 rejected" for each round, with accuracy increasing from ~50% to ~70%]`

```
fl-server | [Gate 2] ✓  Signature VALID for client-0  (round 1)
fl-server | [Gate 2] ✓  Signature VALID for client-1  (round 1)
fl-server | [Gate 2] Result: 2 passed, 0 rejected
fl-server | [Gate 3] Result: 2 accepted, 0 rejected
fl-server | ── Aggregating 2 clean updates via FedAvg ──
```

**Clean up after the run:**

```powershell
docker compose down        # Stops containers (keeps cifar-data volume)
docker compose down -v     # Stops containers AND removes volumes (full clean)
```

---

### Scenario B: Attack Simulation (1 Honest + 1 Malicious — Label-Flip Attack)

This is the **primary thesis experiment**. It demonstrates that a malicious client
with **valid credentials** (passes Gates 1 and 2) is still caught by the
**behavioral anomaly detection** (Gate 3).

```powershell
docker compose up --abort-on-container-exit
```

> This starts the **default** services: `server`, `client-0` (honest), and
> `malicious` (label-flip attacker with `CLIENT_ID=1`).

**What happens:**

1. The **malicious client** (`client_malicious.py`) trains with **inverted labels**
   (class 0 ↔ 9, class 1 ↔ 8, etc.), producing gradients that point in the
   **opposite direction** to honest training.
2. It holds a **valid mTLS certificate** → Gate 1 passes.
3. It **signs its poisoned weights** with a legitimate RSA key → Gate 2 passes.
4. The server computes the **cosine similarity** between the two clients'
   weight deltas:
   - **Round 1:** No previous global model exists → Z-score fallback (both accepted).
   - **Round 2:** Cosine similarity is barely positive (~0.02) → both accepted.
   - **Round 3:** Cosine similarity turns **negative** (~−0.11) → **malicious client REJECTED**.
5. After rejection, only 1 update survives. The `MIN_ACCEPTED=2` guard prevents
   aggregation to avoid single-client model domination.

**What to look for in the logs:**

> `[Insert Screenshot: Server terminal showing "🚨 SECURITY ALERT: anti-correlated deltas" and "client-1 REJECTED" in round 3]`

```
# The malicious client announces itself:
fl-malicious | [MALICIOUS 1] 🔴 ATTACK MODE: label_flip

# Round 3 — Gate 3 catches the attacker:
fl-server | [Gate 3] cos_sim(client-1 Δ, client-0 Δ) = -0.1109
fl-server | [Gate 3] 🚨 SECURITY ALERT: anti-correlated deltas (cos_sim=-0.1109 < 0)
fl-server | [Gate 3] 🚨 SECURITY ALERT: client-1 REJECTED (largest delta norm 10.150310)
fl-server | [Gate 3] Result: 1 accepted, 1 rejected ['1']
fl-server | [Zero-Trust] Only 1 update(s) survived the pipeline (minimum required: 2)
           – skipping aggregation to prevent single-client model domination.
```

---

### Scenario C: Tampered Signing Keys (Demonstrating Gate 2)

This simulation proves that swapping private keys causes **Gate 2 to reject
all updates** — the server detects that signatures don't match the expected
public keys.

**Step 1 — Swap the signing keys:**

```powershell
# Back up originals
Copy-Item signing_keys\client-0.private.pem signing_keys\client-0.private.pem.bak
Copy-Item signing_keys\client-1.private.pem signing_keys\client-1.private.pem.bak

# Swap: client-0 gets client-1's key and vice versa
Copy-Item signing_keys\client-1.private.pem signing_keys\client-0.private.pem
Copy-Item signing_keys\client-0.private.pem.bak signing_keys\client-1.private.pem
```

**Step 2 — Run (using the clean profile with 2 honest clients):**

```powershell
docker compose --profile clean up --abort-on-container-exit server client-0 client-1
```

**What to look for:**

> `[Insert Screenshot: Server logs showing "✗ REJECTED tampered update from client_id=1" and "No valid updates this round – skipping aggregation"]`

```
fl-server | WARNING: [Gate 2] ✗ REJECTED tampered update from client_id=1
fl-server | [Gate 2] Round 1: 0 accepted, 1 rejected
fl-server | WARNING: [Gate 2] No valid updates this round – skipping aggregation
```

**Step 3 — Restore the original keys (important!):**

```powershell
Copy-Item signing_keys\client-0.private.pem.bak signing_keys\client-0.private.pem -Force
Copy-Item signing_keys\client-1.private.pem.bak signing_keys\client-1.private.pem -Force
Remove-Item signing_keys\*.bak
```

---

### Quick Reference — All Run Commands

| Scenario | Command | Expected Result |
|----------|---------|-----------------|
| **A. Clean baseline** (2 honest) | `docker compose --profile clean up --abort-on-container-exit server client-0 client-1` | All gates pass, accuracy ~65–70% |
| **B. Attack simulation** (1 honest + 1 label-flip) | `docker compose up --abort-on-container-exit` | Attacker detected in round 3, accuracy recovers |
| **C. Swapped keys** (Gate 2 proof) | Swap keys → run clean profile → restore keys | Gate 2 rejects all updates |
| **Pre-download CIFAR-10** | `docker compose up data-init` | `CIFAR-10 ready.` |
| **Full cleanup** | `docker compose down -v --remove-orphans` | Removes all containers and volumes |
| **Monitor logs live** | `docker compose logs -f server` | Follow server output in real-time |

---

## 4. Expected Output & Verification

### 4.1 Success Logs — Clean Baseline (Scenario A)

When running with **2 honest clients**, you should see all three gates pass in every
round and accuracy steadily improving.

```
═══════════════════════════════════════════════════════
                     ROUND 1 SUMMARY
═══════════════════════════════════════════════════════

── Gate 1: mTLS ──
fl-server    | [SERVER] 🔒 Loading mTLS certificates from /certs
fl-server    | [SERVER] ✓  CA certificate loaded
fl-server    | [SERVER] ✓  Server certificate loaded
fl-server    | [SERVER] ✓  Server private key loaded
fl-server    | INFO:  Flower ECE: gRPC server running (3 rounds), SSL is enabled

fl-client-0  | [Client 0] 🔒 Loading mTLS certificates from /certs
fl-client-0  | [mTLS] ✓  gRPC patched – client certificate will be presented on connect

fl-client-1  | [Client 1] 🔒 Loading mTLS certificates from /certs
fl-client-1  | [mTLS] ✓  gRPC patched – client certificate will be presented on connect

── Gate 2: Signature Verification ──
fl-server    | [Gate 2] ✓  Signature VALID for client-0  (round 1)
fl-server    | [Gate 2] ✓  Signature VALID for client-1  (round 1)
fl-server    | [Gate 2] Result: 2 passed, 0 rejected

── Gate 3: Anomaly Detection ──
fl-server    | [Gate 3] Round 1: no global reference yet – using weight-norm Z-score
fl-server    | [Gate 3] ✓  client-0: z=1.00  ACCEPTED
fl-server    | [Gate 3] ✓  client-1: z=1.00  ACCEPTED
fl-server    | [Gate 3] Result: 2 accepted, 0 rejected

── Aggregation ──
fl-server    | ── Aggregating 2 clean updates via FedAvg ──

═══════════════════════════════════════════════════════
                     ROUND 2 SUMMARY
═══════════════════════════════════════════════════════

── Gate 2 ──
fl-server    | [Gate 2] ✓  Signature VALID for client-0  (round 2)
fl-server    | [Gate 2] ✓  Signature VALID for client-1  (round 2)
fl-server    | [Gate 2] Result: 2 passed, 0 rejected

── Gate 3 ──
fl-server    | [Gate 3] Global reference available – using delta cosine similarity
fl-server    | [Gate 3] ✓  Correlated deltas – both clients accepted
fl-server    | [Gate 3] Result: 2 accepted, 0 rejected

═══════════════════════════════════════════════════════
                     ROUND 3 SUMMARY
═══════════════════════════════════════════════════════

── Gate 2 ──
fl-server    | [Gate 2] Result: 2 passed, 0 rejected

── Gate 3 ──
fl-server    | [Gate 3] ✓  Correlated deltas – both clients accepted
fl-server    | [Gate 3] Result: 2 accepted, 0 rejected

── Final ──
fl-server    | Run finished 3 round(s) of federated learning
fl-server exited with code 0
fl-client-0 exited with code 0
fl-client-1 exited with code 0
```

**Accuracy progression (typical values for clean baseline):**

| Round | Eval Loss | Eval Accuracy | Gate 2 | Gate 3 |
|-------|-----------|---------------|--------|--------|
| 1 | ~1.46 | ~51% | 2/2 ✓ | 2/2 ✓ |
| 2 | ~1.02 | ~64% | 2/2 ✓ | 2/2 ✓ |
| 3 | ~0.87 | **~68–70%** | 2/2 ✓ | 2/2 ✓ |

✅ **How to verify success:** Accuracy should increase each round and reach
**65–70%** by round 3. All gate results should show **0 rejected**.

---

### 4.2 Attack Detected Logs — Label-Flip Attack (Scenario B)

When running with **1 honest + 1 malicious** client, the logs show a dramatically
different pattern:

```
═══════════════════════════════════════════════════════
              ATTACK DETECTION (ROUND 3)
═══════════════════════════════════════════════════════

fl-malicious | [MALICIOUS 1] 🔴 ATTACK MODE: label_flip
fl-malicious | [MALICIOUS 1] Training with FLIPPED labels (global inversion)

── Gate 1 ── (Attacker passes — has valid cert)
fl-malicious | [MALICIOUS 1] 🔒 mTLS certs loaded  → Gate 1 WILL pass

── Gate 2 ── (Attacker passes — signs with legitimate key)
fl-server    | [Gate 2] ✓  Signature VALID for client-0  (round 3)
fl-server    | [Gate 2] ✓  Signature VALID for client-1  (round 3)
fl-server    | [Gate 2] Result: 2 passed, 0 rejected

── Gate 3 ── (Attacker CAUGHT — anti-correlated deltas)
fl-server    | [Gate 3] Global reference available – using delta cosine similarity
fl-server    | [Gate 3] Delta norms: {'1': '10.150310', '0': '7.188231'}
fl-server    | [Gate 3] cos_sim(client-1 Δ, client-0 Δ) = -0.1109
fl-server    | [Gate 3] 🚨 SECURITY ALERT: anti-correlated deltas (cos_sim=-0.1109 < 0)
fl-server    | [Gate 3] 🚨 SECURITY ALERT: client-1 REJECTED (largest delta norm 10.150310)
fl-server    | [Gate 3] Result: 1 accepted, 1 rejected ['1']

── Minimum Accepted Guard ──
fl-server    | [Zero-Trust] Only 1 update(s) survived the pipeline
               (minimum required: 2) – skipping aggregation to prevent
               single-client model domination.
```

**Accuracy progression (typical values for attack scenario):**

| Round | Eval Accuracy | Gate 3 Result | cos_sim | Notes |
|-------|---------------|---------------|---------|-------|
| 1 | ~23% | 2 accepted | — | Z-score fallback (no reference); attack poisons model |
| 2 | ~25% | 2 accepted | +0.017 | Barely positive; attack not yet detectable |
| 3 | ~25% | **1 rejected** | **−0.111** | Anti-correlated! Attacker caught |

✅ **How to verify the attack was caught:**

1. Look for `🚨 SECURITY ALERT` in the server logs.
2. The rejected client ID (`client-1`) matches the malicious container.
3. The `cos_sim` value is **negative** (anti-correlated deltas).
4. Accuracy in rounds 1–2 is **degraded** (~20–25%) compared to the clean
   baseline (~50–64%), proving the attack's impact.
5. In round 3, the `MIN_ACCEPTED` guard **skips aggregation** rather than
   using a single client's update.

---

### 4.3 How to Read the Accuracy Metrics

The server evaluates the global model against the full CIFAR-10 test set
(10,000 images) after each aggregation round.

| Metric | What It Means |
|--------|---------------|
| **Eval Loss** | Cross-entropy loss on the test set. Lower = better. |
| **Eval Accuracy** | Percentage of correctly classified test images. Higher = better. |
| **Round 1** | First aggregation — model just initialized, accuracy ~50% for clean, ~20% if poisoned. |
| **Round 3** | Final round — clean baseline should reach ~65–70%. Attack scenario stays low (~25%) until the attacker is rejected. |

**Key insight for the thesis:** The accuracy difference between Scenario A
(clean, ~70%) and Scenario B (attack, ~25%) quantifies the attack's damage.
Gate 3's detection in round 3 prevents further degradation.

---

## 5. Troubleshooting FAQ

### Q1: Docker says `could not select device driver "" with capabilities: [[gpu]]`

**Cause:** The NVIDIA Container Toolkit is not installed or not configured for Docker.

**Fix (run inside WSL2 Ubuntu):**

```bash
# 1. Add the NVIDIA Container Toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 3. Configure Docker to use the nvidia runtime
sudo nvidia-ctk runtime configure --runtime=docker

# 4. Restart Docker
sudo systemctl restart docker
# Or: restart Docker Desktop from the Windows system tray

# 5. Verify
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

### Q2: `nvidia-container-cli: initialization error: driver not loaded`

**Cause:** The NVIDIA Windows host driver is not being forwarded into WSL2.

**Fix:**

1. **Update your NVIDIA Windows driver** to the latest version from
   [nvidia.com/drivers](https://www.nvidia.com/drivers).
2. **Do NOT install a separate CUDA/GPU driver inside WSL2** — WSL2 automatically
   forwards the Windows host driver via `/usr/lib/wsl/lib/`.
3. Update WSL2 itself:
   ```powershell
   wsl --update
   wsl --shutdown
   ```
   Then reopen your WSL2 terminal.
4. Verify the driver library exists inside WSL2:
   ```bash
   ls -la /usr/lib/wsl/lib/libcuda.so*
   # Should show: libcuda.so.1 → libcuda.so.1.1
   ```

---

### Q3: Why did the SSL/mTLS handshake fail?

**Symptoms:** Client logs show a gRPC error like this:

```
grpc._channel._InactiveRpcError: <_InactiveRpcError ... StatusCode.UNAVAILABLE
  "failed to connect to all addresses"
  ... SSL handshake failed
```

**Common causes and fixes:**

| Cause | Fix |
|-------|-----|
| Certificates not generated | Run `python generate_keys.py` |
| Certificates expired (older than 365 days) | Re-run `python generate_keys.py` to regenerate |
| Certs generated but not mounted | Check `docker-compose.yml` has `./certs:/certs:ro` for each service |
| Server started before certs exist | Ensure `certs/` directory is populated before `docker compose up` |
| SAN mismatch | Certificates must include `DNS:server` for the Docker Compose service name. Re-run `python generate_keys.py`. |

**Verify certificates are valid:**

```bash
# Check expiry
openssl x509 -in certs/server.crt -noout -dates

# Check SAN entries
openssl x509 -in certs/server.crt -noout -ext subjectAltName

# Verify chain of trust
openssl verify -CAfile certs/ca.crt certs/server.crt
openssl verify -CAfile certs/ca.crt certs/client-0.crt
```

---

### Q4: Why is my GPU not found? (`Using device: cpu` instead of `cuda`)

**Symptoms:** Client logs show `Using device: cpu` and training is very slow.

**Checklist:**

1. **Is `nvidia-smi` working inside Docker?** (See [Section 1.3](#13-pre-flight-check--verify-your-gpu-setup))

2. **Does `docker-compose.yml` have the GPU reservation?** Each service must include:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [gpu]
   ```

3. **Is PyTorch built with CUDA support?** Test inside a container:
   ```powershell
   docker compose run --rm server python -c "import torch; print(torch.cuda.is_available())"
   ```
   Expected: `True`. If `False`, the PyTorch wheel was installed without CUDA.
   Rebuild: `docker compose build --no-cache`

---

### Q5: `CUDA out of memory` during training

**Cause:** The model + batch of data exceeds your GPU's VRAM.

**Fixes (least to most drastic):**

1. **Reduce batch size** — Edit `client.py` and `client_malicious.py`:
   ```python
   BATCH_SIZE = 32    # Default is 64; halving approximately halves VRAM usage
   ```
   Then rebuild: `docker compose build`

2. **Disable `torch.compile`** — In `client.py`, comment out the line:
   ```python
   # self.model = torch.compile(self.model)
   ```
   `torch.compile` creates additional GPU memory overhead for the compiled graph.

3. **Close other GPU applications** — Check `nvidia-smi` for other processes using VRAM.

---

### Q6: `RuntimeError: Failed to find C compiler` on first training run

**Cause:** `torch.compile()` uses the Triton JIT compiler, which requires `gcc` inside
the container.

**Fix:** This should already be handled in the Dockerfile (`gcc` is included). If not,
add it:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev gcc && \
```

Then rebuild: `docker compose build --no-cache`

---

### Q7: `RuntimeError: FATAL: mTLS certificates missing` / `signing key not found`

**Cause:** The server or client cannot find its cryptographic material.

**Fix:**

1. Make sure you ran the key generation script:
   ```powershell
   python generate_keys.py
   ```

2. Verify the files exist:
   ```powershell
   Get-ChildItem certs\
   Get-ChildItem signing_keys\
   ```

3. This is **by design** — the system enforces a **deny-by-default** posture. If
   certificates or signing keys are missing, the process crashes immediately rather
   than running insecurely.

---

### Q8: Containers exit immediately with code 1

**Diagnosis:**

```powershell
# Check which container failed
docker compose ps -a

# Read the failing container's logs
docker compose logs <service-name>
```

**Common exit code meanings:**

| Code | Meaning |
|------|---------|
| `0` | Clean exit (normal) |
| `1` | Python exception (check logs for traceback) |
| `137` | OOM killed (SIGKILL) — increase Docker memory limit |
| `139` | Segmentation fault — rare; usually a CUDA/driver mismatch |

---

### Q9: Transient `TSI_DATA_CORRUPTED` / `BAD_RECORD_MAC` gRPC errors

**Cause:** A known gRPC/BoringSSL race condition when multiple clients send large
TLS-encrypted payloads simultaneously.

**Fix:** The staggered start times in `docker-compose.yml` (`sleep 5` for honest clients,
`sleep 12` for the malicious client) mitigate this. If it still occurs:

- Simply re-run the simulation. It's a transient issue that resolves on retry.
- If persistent, increase the sleep delay for the second client.

---

### Q10: How do I completely reset and start fresh?

```powershell
# Stop all containers and remove volumes
docker compose down -v --remove-orphans

# Regenerate all cryptographic material
python generate_keys.py

# Rebuild from scratch (no cache)
docker compose build --no-cache

# Pre-download dataset
docker compose up data-init

# Run
docker compose up --abort-on-container-exit
```

---

> **End of User Manual.**
>
> For architecture details, see [`ARCHITECTURE.md`](ARCHITECTURE.md).
> For phase-by-phase test results, see `TESTING_LOG.md`.
