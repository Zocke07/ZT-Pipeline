"""Shared CIFAR-10 data loading and partitioning utilities.

Used by all client variants (honest, malicious, baseline) to ensure
identical data handling across experiments.

Supports both IID (equal-shard) and non-IID (Dirichlet) partitioning.

Public API
----------
get_cifar10          – load / download CIFAR-10 dataset
partition_data       – return the shard for one client
make_dataloaders     – convenience factory: train + test DataLoaders
flip_labels_global   – invert every label (0..9 → 9..0)
flip_labels_targeted – flip one source class to a target class
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
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


def partition_data(
    train_set,
    num_clients: int,
    client_id: int,
    *,
    dirichlet_alpha: float | None = None,
    seed: int = 0,
) -> Subset:
    """Partition *train_set* for one client.

    Parameters
    ----------
    dirichlet_alpha : float or None
        If ``None`` (default), perform simple IID equal-shard partitioning.
        Otherwise, use a symmetric Dirichlet distribution with concentration
        parameter *alpha* to create heterogeneous (non-IID) partitions.
        Smaller alpha → more heterogeneous.  Typical values: 0.1, 0.5, 1.0.
    seed : int
        Base seed for reproducible Dirichlet sampling.  All clients must use
        the **same** seed so that the global partition is consistent; each
        client then selects its own shard from the result.

    Returns
    -------
    Subset
        The data shard for *client_id*.
    """
    if dirichlet_alpha is None:
        # ── IID partition (equal shards) ───────────────────────────
        total = len(train_set)
        shard_size = total // num_clients
        start = client_id * shard_size
        end = start + shard_size
        indices = list(range(start, end))
        return Subset(train_set, indices)

    # ── Non-IID Dirichlet partition ────────────────────────────────
    return _dirichlet_partition(train_set, num_clients, client_id,
                                dirichlet_alpha, seed)


def _dirichlet_partition(
    dataset,
    num_clients: int,
    client_id: int,
    alpha: float,
    seed: int,
) -> Subset:
    """Latent Dirichlet Allocation–style partition over label distribution.

    Algorithm (Hsu et al., 2019 — "Measuring the Effects of Non-IID Data"):
      1.  Group all indices by class label.
      2.  For each class, draw a multinomial proportion vector from
          ``Dir(alpha, ..., alpha)`` with *num_clients* entries.
      3.  Assign each sample to a client according to those proportions.

    This guarantees that every sample is assigned exactly once and that all
    clients use the same seeded RNG for determinism.
    """
    targets = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = int(targets.max()) + 1

    rng = np.random.default_rng(seed)

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(targets == c)[0]
        # Draw proportions from Dirichlet
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # Convert proportions to sample counts (whole numbers)
        proportions = proportions / proportions.sum()
        counts = (proportions * len(class_idx)).astype(int)
        # Distribute remainder to largest-remainder clients
        remainder = len(class_idx) - counts.sum()
        fractional = (proportions * len(class_idx)) - counts
        top_k = np.argsort(fractional)[::-1][:remainder]
        counts[top_k] += 1

        # Shuffle and assign
        rng.shuffle(class_idx)
        offset = 0
        for k in range(num_clients):
            client_indices[k].extend(class_idx[offset:offset + counts[k]].tolist())
            offset += counts[k]

    indices = client_indices[client_id]
    print(f"[data] Client {client_id}: {len(indices)} samples "
          f"(Dirichlet α={alpha})")

    # Log class distribution for this client
    if indices:
        client_targets = targets[indices]
        dist = {int(c): int((client_targets == c).sum()) for c in range(num_classes)}
        print(f"[data] Client {client_id} class distribution: {dist}")

    return Subset(dataset, indices)


# ---------------------------------------------------------------------------
# DataLoader factory (shared boilerplate across all client variants)
# ---------------------------------------------------------------------------

def make_dataloaders(
    client_id: int,
    num_clients: int,
    *,
    dirichlet_alpha: float | None = None,
    seed: int = 0,
    batch_size: int = 64,
    pin_memory: bool = False,
    num_workers: int = 2,
    persistent_workers: bool = True,
    data_dir: str = "/data",
) -> tuple[DataLoader, DataLoader]:
    """Create train and test :class:`~torch.utils.data.DataLoader` objects.

    Returns ``(train_loader, test_loader)`` ready to pass to a client.

    Parameters
    ----------
    client_id         : zero-based client index
    num_clients       : total number of FL clients
    dirichlet_alpha   : Dirichlet α for non-IID split; ``None`` → IID
    seed              : RNG seed for deterministic partitioning
    batch_size        : samples per mini-batch
    pin_memory        : set True when training on CUDA
    num_workers       : DataLoader worker count (0 for malicious clients
                        to keep the attack path simple)
    persistent_workers: keep worker processes alive between epochs
                        (ignored when num_workers == 0)
    data_dir          : path to CIFAR-10 root directory
    """
    train_set, test_set = get_cifar10(data_dir)
    extra = {"persistent_workers": True} if num_workers > 0 and persistent_workers else {}

    train_loader = DataLoader(
        partition_data(train_set, num_clients, client_id,
                       dirichlet_alpha=dirichlet_alpha, seed=seed),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, **extra,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, **extra,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Label-flip helpers (shared by ZT and baseline malicious clients)
# ---------------------------------------------------------------------------

def flip_labels_global(labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """Invert every label: class *i* → class (*num_classes* − 1 − *i*)."""
    return (num_classes - 1) - labels


def flip_labels_targeted(
    labels: torch.Tensor,
    source_label: int,
    target_label: int,
) -> torch.Tensor:
    """Flip only *source_label* → *target_label*, leaving others unchanged."""
    flipped = labels.clone()
    flipped[flipped == source_label] = target_label
    return flipped
