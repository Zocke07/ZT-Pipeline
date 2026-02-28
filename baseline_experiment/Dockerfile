# ──────────────────────────────────────────────────────────────
#  Baseline (Insecure) FL – Dockerfile
#
#  Identical runtime to the ZT-Pipeline image (same CUDA base,
#  same Python packages, same model) so that any performance
#  difference is attributable ONLY to the security mechanisms.
#
#  Key difference: no signing.py copied, no cert infrastructure.
# ──────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies (gcc needed by torch.compile / Triton JIT)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev gcc && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages (same versions as ZT-Pipeline for fair comparison)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt

# Copy ONLY the baseline scripts + shared model (no signing.py, no certs)
COPY model.py ./
COPY baseline_experiment/baseline_server.py ./
COPY baseline_experiment/baseline_client.py ./
COPY baseline_experiment/baseline_malicious_client.py ./

CMD ["python", "--version"]
