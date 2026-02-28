"""Shared CNN model definition for CIFAR-10 Federated Learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CifarCNN(nn.Module):
    """Simple CNN for CIFAR-10 (3×32×32 → 10 classes).

    Architecture:
        Conv(3→32)→ReLU→Pool → Conv(32→64)→ReLU→Pool →
        Conv(64→64)→ReLU → FC(1024→64)→ReLU → FC(64→10)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # 32×32 → 16×16
        x = self.pool(F.relu(self.conv2(x)))   # 16×16 → 8×8
        x = F.relu(self.conv3(x))              # 8×8 (no pool)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
