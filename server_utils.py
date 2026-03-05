"""Shared server utilities for ZT-Pipeline FL servers.

Contains logic used by both the Zero-Trust server (server.py) and the
baseline insecure server (baseline_experiment/baseline_server.py):

  - JSON serialization helper
  - Weighted evaluation metric aggregation
  - Krum / Multi-Krum aggregation
  - Per-round model checkpoint saving
  - Structured per-round metrics collection
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flwr.common import FitRes, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

def json_safe(obj: Any) -> Any:
    """JSON serializer fallback for NumPy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# Flower evaluation metric aggregation
# ---------------------------------------------------------------------------

def weighted_eval_metrics(metrics_list: List[Tuple[int, Dict]]) -> Dict:
    """Weighted average of client evaluation metrics (Flower callback)."""
    total = sum(n for n, _ in metrics_list)
    if total == 0:
        return {}
    acc = sum(n * m.get("accuracy", 0.0) for n, m in metrics_list) / total
    return {"accuracy": acc}


# ---------------------------------------------------------------------------
# Krum / Multi-Krum aggregation
# ---------------------------------------------------------------------------

def krum_scores(
    weights_list: List[List[np.ndarray]],
    num_byzantine: int,
) -> List[float]:
    """Compute Krum scores: for each client, the sum of distances to its
    closest (n − f − 2) neighbours, where n = len(weights_list) and
    f = num_byzantine.  Lower score = more representative = better.

    Reference: Blanchard et al., "Machine Learning with Adversaries:
    Byzantine Tolerant Gradient Descent" (NeurIPS 2017).
    """
    n = len(weights_list)
    flat = [np.concatenate([w.ravel() for w in ws]) for ws in weights_list]

    # Pairwise squared L2 distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(flat[i] - flat[j]) ** 2)
            dists[i][j] = d
            dists[j][i] = d

    k = max(1, n - num_byzantine - 2)
    scores = []
    for i in range(n):
        sorted_dists = np.sort(dists[i])         # index 0 is self (distance 0)
        scores.append(float(np.sum(sorted_dists[1:k + 1])))
    return scores


def krum_aggregate(
    results: List[Tuple[ClientProxy, FitRes]],
    num_byzantine: int,
    multi: bool = False,
) -> Tuple[List[np.ndarray], int]:
    """Select the best client(s) via (Multi-)Krum and return averaged weights.

    Parameters
    ----------
    results      : accepted (ClientProxy, FitRes) pairs
    num_byzantine: estimated number of Byzantine clients in *results*
    multi        : if True, average the top (n − f) clients (Multi-Krum);
                   if False, use only the single best client (Krum)

    Returns
    -------
    (aggregated_weights, total_examples)
    """
    n = len(results)
    weights_list = [parameters_to_ndarrays(fr.parameters) for _, fr in results]
    examples = [fr.num_examples for _, fr in results]

    scores = krum_scores(weights_list, num_byzantine)

    for idx, (_, fr) in enumerate(results):
        cid = fr.metrics.get("client_id", "?")
        label = int(cid) if isinstance(cid, float) else cid
        print(f"[Krum] client-{label}: score={scores[idx]:.4f}")

    if multi:
        m = max(1, n - num_byzantine)
        selected_idx = np.argsort(scores)[:m]
        print(f"[Multi-Krum] Selected {m}/{n} clients (indices: {selected_idx.tolist()})")
    else:
        best = int(np.argmin(scores))
        selected_idx = [best]
        cid = results[best][1].metrics.get("client_id", "?")
        label = int(cid) if isinstance(cid, float) else cid
        print(f"[Krum] Selected client-{label} (score={scores[best]:.4f})")

    selected_weights = [weights_list[i] for i in selected_idx]
    selected_examples = [examples[i] for i in selected_idx]
    total = sum(selected_examples)

    if total == 0:
        agg = [np.mean(arrs, axis=0) for arrs in zip(*selected_weights)]
        return agg, sum(examples)

    agg = [
        sum(
            w[layer_i] * (ex / total)
            for w, ex in zip(selected_weights, selected_examples)
        )
        for layer_i in range(len(selected_weights[0]))
    ]
    return agg, total


# ---------------------------------------------------------------------------
# Per-round model checkpoint
# ---------------------------------------------------------------------------

def save_round_model(
    params: List[np.ndarray],
    server_round: int,
    results_dir: Path,
    prefix: str = "SERVER",
) -> None:
    """Save the aggregated global model as a ``.pth`` checkpoint.

    Creates ``results_dir/models/round_{server_round}.pth`` so the host
    orchestrator can log it as an MLflow artifact after the container exits.
    """
    try:
        import torch
        from model import CifarCNN
        model_dir = results_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model = CifarCNN()
        state_keys = list(model.state_dict().keys())
        state_dict = {k: torch.tensor(v) for k, v in zip(state_keys, params)}
        model.load_state_dict(state_dict, strict=True)
        path = model_dir / f"round_{server_round}.pth"
        torch.save(model.state_dict(), path)
    except Exception as exc:
        print(f"[{prefix}] Warning: could not save model checkpoint: {exc}")


# ---------------------------------------------------------------------------
# Structured per-round metrics collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Collects per-round structured metrics and serializes them to JSON.

    Both ZT and baseline servers record to this class; the resulting
    ``metrics.json`` format is identical so the orchestrator/MLflow logger
    can process both without special-casing.

    ``record_fit`` accepts **kwargs so callers can pass whatever gate/
    aggregation data is available; missing keys default to safe values.
    """

    def __init__(self) -> None:
        self._rounds: Dict[int, Dict[str, Any]] = {}

    def record_fit(self, server_round: int, **kwargs: Any) -> None:
        """Record fit-phase data for *server_round*."""
        if server_round not in self._rounds:
            self._rounds[server_round] = {"round": server_round}
        self._rounds[server_round].update(kwargs)

    def record_eval(self, server_round: int, loss: float, accuracy: float) -> None:
        """Record evaluation results for *server_round*."""
        if server_round not in self._rounds:
            self._rounds[server_round] = {"round": server_round}
        self._rounds[server_round]["global_loss"] = loss
        self._rounds[server_round]["global_accuracy"] = accuracy

    def to_list(self) -> List[Dict]:
        return [self._rounds[r] for r in sorted(self._rounds.keys())]

    def save(self, path: Path, metadata: Dict) -> None:
        """Write the full metrics payload to *path* as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**metadata, "round_metrics": self.to_list()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=json_safe)
        print(f"[metrics] Structured metrics written to {path}")
