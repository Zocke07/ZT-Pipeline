"""Shared training, evaluation, and model parameter utilities.

Provides GPU configuration, hyperparameters, and common train/eval
functions used across all client variants.
"""

import os
import random
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import CifarCNN

# ---------------------------------------------------------------------------
# GPU configuration  (applied once at import time)
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------------
# Hyperparameters (shared across honest and baseline clients)
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
LEARNING_RATE = 0.001
LOCAL_EPOCHS = 1


# ---------------------------------------------------------------------------
# Reproducibility – seed control
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility.

    Seeds:  Python ``random``, NumPy, PyTorch CPU, PyTorch CUDA (all devices).
    Enables deterministic CuDNN and disables benchmark mode.

    Known sources of non-determinism that **cannot** be fully controlled:
      - Certain CUDA atomicAdd operations in backward passes
      - ``torch.nn.functional.interpolate`` with some modes
      - Sparse-dense matrix multiplications on CUDA
      - Non-deterministic reduction order in multi-threaded DataLoader workers
    See: https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Model parameter helpers
# ---------------------------------------------------------------------------

def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Load a list of NumPy arrays into the model's state dict."""
    state_dict = OrderedDict(
        {k: torch.from_numpy(np.array(v))
         for k, v in zip(model.state_dict().keys(), parameters)}
    )
    model.load_state_dict(state_dict, strict=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    scaler: torch.amp.GradScaler,
    lr: float = LEARNING_RATE,
) -> float:
    """Train for one epoch with AMP mixed precision. Returns average loss."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def train_one_epoch_with_label_transform(
    model: nn.Module,
    loader: DataLoader,
    label_fn,
    lr: float = LEARNING_RATE,
) -> float:
    """Train for one epoch, applying *label_fn* to every batch's labels.

    Used by malicious clients to implement label-flip attacks.
    No AMP — keeps the attack training path simple and deterministic.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = label_fn(labels).to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    """Evaluate the model. Returns (loss, accuracy)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_model(compile_model: bool = True) -> nn.Module:
    """Instantiate CifarCNN on DEVICE, optionally with torch.compile."""
    model = CifarCNN().to(DEVICE)
    if compile_model and DEVICE.type == "cuda":
        model = torch.compile(model)
    return model


def create_scaler() -> torch.amp.GradScaler:
    """Create an AMP GradScaler (enabled only on CUDA)."""
    return torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
