# ──────────────────────────────────────────────────────────────
#  Base image: NVIDIA CUDA 12.4 + cuDNN 9 on Ubuntu 22.04
#  Provides GPU acceleration for PyTorch inside Docker.
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

# Install Python packages (PyTorch w/ CUDA 12.4, Flower, torchvision)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY model.py server.py client.py signing.py client_malicious.py server_utils.py ./
COPY data_utils.py training.py mtls.py ./

# Default: print usage
CMD ["python", "--version"]
