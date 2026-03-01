"""Shared CIFAR-10 data loading and partitioning utilities.

Used by all client variants (honest, malicious, baseline) to ensure
identical data handling across experiments.
"""

from torch.utils.data import Subset
from torchvision import datasets, transforms


# CIFAR-10 channel-wise mean and std (pre-computed over the training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10(data_dir: str = "/data"):
    """Download CIFAR-10 and return (train_set, test_set).

    Uses standard normalisation so every client works with the same
    pre-processing pipeline.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    return train_set, test_set


def partition_data(train_set, num_clients: int, client_id: int) -> Subset:
    """Simple IID partition: split training set into ``num_clients`` equal shards."""
    total = len(train_set)
    shard_size = total // num_clients
    start = client_id * shard_size
    end = start + shard_size
    indices = list(range(start, end))
    return Subset(train_set, indices)
