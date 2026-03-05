"""Flower Federated Learning Server – FedAvg on CIFAR-10.

Starts a Flower gRPC server on port 8080, runs 3 FL rounds,
requiring a minimum of 2 clients per round.

Zero-Trust Gate 1: mTLS – server requires valid client certificates
signed by the trusted Root CA before allowing participation.

Zero-Trust Gate 2: Integrity – the server verifies RSA-PSS digital
signatures on every model update before aggregation.  Unsigned or
tampered updates are rejected.

Zero-Trust Gate 3: Quality – anomaly detection on weight updates.
Updates whose L2 norm deviates beyond a configurable Z-score threshold
from the population mean are flagged as potentially poisoned and rejected.
"""

import os
from logging import WARNING
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.common.logger import log

from server_utils import (
    MetricsCollector,
    krum_aggregate,
    save_round_model,
    weighted_eval_metrics,
)
from signing import verify_signature, load_client_public_keys
from training import set_seed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CERT_DIR = Path(os.environ.get("CERT_DIR", "/certs"))
SIGNING_KEY_DIR = Path(os.environ.get("SIGNING_KEY_DIR", "/signing_keys"))
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", "2"))
NUM_ROUNDS = int(os.environ.get("NUM_ROUNDS", "20"))

# Gate ablation toggles  (server-side enforcement)
ENABLE_GATE2 = os.environ.get("ENABLE_GATE2", "true").lower() == "true"
ENABLE_GATE3 = os.environ.get("ENABLE_GATE3", "true").lower() == "true"

# Reproducibility
SEED = int(os.environ.get("SEED", "-1"))

# Structured results output
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results"))

# Gate 3 tunables
ANOMALY_Z_THRESHOLD = float(os.environ.get("ANOMALY_Z_THRESHOLD", "2.0"))

# Aggregation method: fedavg | krum | multi-krum
AGGREGATION_METHOD = os.environ.get("AGGREGATION_METHOD", "fedavg").lower()


def _load_certificates() -> Tuple[bytes, bytes, bytes]:
    """Load mTLS certificates for the Flower gRPC server.

    Zero-Trust: the server MUST NOT start without valid certificates.
    """
    ca_cert_path = CERT_DIR / "ca.crt"
    server_cert_path = CERT_DIR / "server.crt"
    server_key_path = CERT_DIR / "server.key"

    missing = [p for p in [ca_cert_path, server_cert_path, server_key_path] if not p.exists()]
    if missing:
        raise RuntimeError(
            f"[SERVER] \u2717  FATAL: mTLS certificates missing: {missing}. "
            f"Zero-Trust policy: server MUST NOT start without mTLS."
        )

    print(f"[SERVER] 🔒 Loading mTLS certificates from {CERT_DIR}")
    ca_cert = ca_cert_path.read_bytes()
    server_cert = server_cert_path.read_bytes()
    server_key = server_key_path.read_bytes()
    print("[SERVER] ✓  CA certificate loaded")
    print("[SERVER] ✓  Server certificate loaded")
    print("[SERVER] ✓  Server private key loaded")
    return ca_cert, server_cert, server_key


# ---------------------------------------------------------------------------
# Gate 3 helper – L2-norm anomaly detection
# ---------------------------------------------------------------------------

def _compute_update_norm(ndarrays: List[np.ndarray]) -> float:
    """Compute the L2 (Euclidean) norm of a flattened weight vector."""
    flat = np.concatenate([a.ravel() for a in ndarrays])
    return float(np.linalg.norm(flat))


def _compute_cosine_similarity(a: List[np.ndarray], b: List[np.ndarray]) -> float:
    """Compute cosine similarity between two flattened weight vectors."""
    flat_a = np.concatenate([x.ravel() for x in a])
    flat_b = np.concatenate([x.ravel() for x in b])
    dot = np.dot(flat_a, flat_b)
    denom = np.linalg.norm(flat_a) * np.linalg.norm(flat_b)
    if denom == 0:
        return 0.0
    return float(dot / denom)


def _filter_anomalous_updates(
    candidates: List[Tuple[str, List[np.ndarray], "ClientProxy", "FitRes"]],
    z_threshold: float,
    global_params: Optional[List[np.ndarray]] = None,
) -> Tuple[
    List[Tuple["ClientProxy", "FitRes"]],
    List[str],
]:
    """Gate 3 anomaly detection: directional (delta-cosine) or Z-score fallback.

    Two strategies depending on whether the previous global model is available:

    **Directional / delta-cosine detection** (used when global_params is set):
        1. Compute delta_i = w_i − global_params  (the update direction).
        2. For exactly 2 clients:
               cos_sim = cosine_similarity(delta_0, delta_1)
               If cos_sim < 0 the two deltas are anti-correlated, meaning
               one client is training in the wrong direction (label flip,
               gradient inversion, etc.)  → reject the client whose delta
               has the larger L2 norm (the stronger attacker).
        3. For n ≥ 3 clients:
               Compute each client's mean cosine similarity against all
               others; reject any client below a correlation floor of −0.1.

    **Norm Z-score detection** (fallback on round 1, global_params is None):
        Classic Z-score on L2 norms of raw weights; rejects if z > threshold.
        Cannot distinguish direction on the first round.

    Returns (accepted, rejected_cids).
    """
    if not candidates:
        return [], []

    cids    = [c[0] for c in candidates]
    weights = [c[1] for c in candidates]
    proxies = [c[2] for c in candidates]
    fitress = [c[3] for c in candidates]

    accepted: List[Tuple[ClientProxy, FitRes]] = []
    rejected_cids: List[str] = []

    # ── Compute deltas (update direction) when global model is known ──────
    if global_params is not None:
        deltas = [
            [u - g for u, g in zip(w, global_params)]
            for w in weights
        ]
        norms = np.array([_compute_update_norm(d) for d in deltas])
        print(f"[Gate 3] Delta norms: "
              f"{dict(zip(cids, [f'{n:.6f}' for n in norms]))}")
    else:
        deltas = None
        norms  = np.array([_compute_update_norm(w) for w in weights])
        print(f"[Gate 3] Weight norms (round 1, no global ref): "
              f"{dict(zip(cids, [f'{n:.2f}' for n in norms]))}")

    # ── Case A: exactly 2 clients + deltas available → directional check ─
    if deltas is not None and len(candidates) == 2:
        cos_sim = _compute_cosine_similarity(deltas[0], deltas[1])
        print(f"[Gate 3] cos_sim(client-{cids[0]} Δ, client-{cids[1]} Δ) "
              f"= {cos_sim:.4f}")
        if cos_sim < 0.0:
            # Anti-correlated updates: reject the stronger outlier
            outlier_idx = int(np.argmax(norms))
            outlier_cid = cids[outlier_idx]
            print(f"[Gate 3] 🚨 SECURITY ALERT: anti-correlated deltas "
                  f"(cos_sim={cos_sim:.4f} < 0)")
            print(f"[Gate 3] 🚨 SECURITY ALERT: client-{outlier_cid} REJECTED "
                  f"(largest delta norm {norms[outlier_idx]:.6f})")
            for i in range(len(candidates)):
                if cids[i] == outlier_cid:
                    rejected_cids.append(cids[i])
                else:
                    accepted.append((proxies[i], fitress[i]))
        else:
            print(f"[Gate 3] ✓  Correlated deltas (cos_sim={cos_sim:.4f}) "
                  f"– both clients accepted")
            accepted = list(zip(proxies, fitress))
        return accepted, rejected_cids

    # ── Case B: n ≥ 3 clients → per-client mean cosine similarity ────────
    if deltas is not None:
        vecs = deltas
        print(f"[Gate 3] Checking pairwise delta cosine similarities")
        for i, cid in enumerate(cids):
            others = [vecs[j] for j in range(len(vecs)) if j != i]
            mean_cos = float(np.mean([
                _compute_cosine_similarity(vecs[i], o) for o in others
            ]))
            print(f"[Gate 3] client-{cid}: mean_cos_sim={mean_cos:.4f}")
            if mean_cos < -0.1:
                print(f"[Gate 3] 🚨 SECURITY ALERT: client-{cid} REJECTED "
                      f"(mean_cos_sim={mean_cos:.4f} < -0.1)")
                rejected_cids.append(cid)
            else:
                print(f"[Gate 3] ✓  client-{cid}: mean_cos_sim={mean_cos:.4f} – ACCEPTED")
                accepted.append((proxies[i], fitress[i]))
        return accepted, rejected_cids

    # ── Case C: round 1 fallback – Z-score on raw weight norms ───────────
    mean_norm = float(np.mean(norms))
    std_norm  = float(np.std(norms))
    print(f"[Gate 3] Mean norm: {mean_norm:.2f}  |  Std: {std_norm:.2f}  "
          f"|  Z-threshold: {z_threshold}")

    for i, cid in enumerate(cids):
        z_score    = abs(norms[i] - mean_norm) / std_norm if std_norm > 1e-8 else 0.0
        is_outlier = z_score > z_threshold

        if is_outlier and len(weights) >= 2:
            others   = [weights[j] for j in range(len(weights)) if j != i]
            centroid = [np.mean(arrs, axis=0) for arrs in zip(*others)]
            cos_sim  = _compute_cosine_similarity(weights[i], centroid)
            print(f"[Gate 3] client-{cid}: z={z_score:.2f}, "
                  f"cos_sim(vs centroid)={cos_sim:.4f}")
            if cos_sim > 0.5:
                print(f"[Gate 3] ⚠  client-{cid}: high Z-score but cosine OK "
                      f"– ACCEPTED with warning")
                is_outlier = False

        if is_outlier:
            print(f"[Gate 3] 🚨 SECURITY ALERT: client-{cid} rejected as ANOMALOUS "
                  f"(z={z_score:.2f}, norm={norms[i]:.2f})")
            rejected_cids.append(cid)
        else:
            if z_score > 0:
                print(f"[Gate 3] ✓  client-{cid}: z={z_score:.2f}, "
                      f"norm={norms[i]:.2f} – ACCEPTED")
            accepted.append((proxies[i], fitress[i]))

    return accepted, rejected_cids


# ---------------------------------------------------------------------------
# Identity binding: mTLS CN → client_id cross-reference
# ---------------------------------------------------------------------------

def _load_client_cn_map(cert_dir: Path, num_clients: int) -> Dict[str, str]:
    """Load client certificates and extract Common Name for each client_id.

    Builds a mapping: { "0": "fl-client-0", "1": "fl-client-1", ... }
    Used to cross-reference the self-reported client_id in fit() metrics
    against the identity embedded in the mTLS certificate.
    """
    try:
        from cryptography import x509
    except ImportError:
        print("[Identity] cryptography library not available – skipping CN map")
        return {}

    cn_map: Dict[str, str] = {}
    for i in range(num_clients):
        cert_path = cert_dir / f"client-{i}.crt"
        if cert_path.exists():
            cert_data = cert_path.read_bytes()
            cert = x509.load_pem_x509_certificate(cert_data)
            cn = cert.subject.get_attributes_for_oid(
                x509.oid.NameOID.COMMON_NAME
            )[0].value
            cn_map[str(i)] = cn
            print(f"[Identity] client-{i} cert CN: {cn}")
        else:
            print(f"[Identity] ⚠ cert not found: {cert_path}")
    return cn_map


def _verify_identity_binding(
    client_id: str,
    cn_map: Dict[str, str],
) -> bool:
    """Verify that a self-reported client_id has a matching certificate CN.

    Returns True if binding is valid or if cn_map is empty (baseline mode).
    """
    if not cn_map:
        return True  # No certs loaded (baseline/ablation mode)

    if client_id not in cn_map:
        print(f"[Identity] 🚨 REJECT: client_id={client_id} has NO "
              f"registered certificate")
        return False

    # CN should match pattern fl-client-{id}
    expected_cn = f"fl-client-{client_id}"
    actual_cn = cn_map[client_id]
    if actual_cn != expected_cn:
        print(f"[Identity] 🚨 REJECT: client_id={client_id} cert CN "
              f"'{actual_cn}' != expected '{expected_cn}'")
        return False

    return True


# ---------------------------------------------------------------------------
# Combined Strategy: Gate 2 (Signatures) + Gate 3 (Anomaly Detection)
# ---------------------------------------------------------------------------
class ZeroTrustFedAvg(FedAvg):
    """FedAvg secured with signature verification AND anomaly detection.

    Pipeline per round:
        Identity → verify self-reported client_id against cert CN
        Gate 2   → verify RSA-PSS signature on each update
        Gate 3   → statistical outlier detection on verified updates
        Aggregate → FedAvg, Krum, or Multi-Krum on clean updates

    Gate 2 and Gate 3 can be independently disabled via constructor flags
    to support ablation studies.
    """

    def __init__(
        self,
        public_keys: dict,
        z_threshold: float = 2.0,
        min_accepted: int = 1,
        enable_gate2: bool = True,
        enable_gate3: bool = True,
        metrics_collector: Optional[MetricsCollector] = None,
        aggregation_method: str = "fedavg",
        num_attackers: int = 0,
        cn_map: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.public_keys = public_keys
        self.z_threshold = z_threshold
        self.min_accepted = min_accepted
        self.enable_gate2 = enable_gate2
        self.enable_gate3 = enable_gate3
        self._metrics = metrics_collector
        self.aggregation_method = aggregation_method
        self.num_attackers = num_attackers
        self.cn_map = cn_map or {}
        # Tracks the most-recent aggregated global model for delta computation
        self._global_params: Optional[List[np.ndarray]] = None

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Gate 2 + Gate 3 pipeline with ablation toggles and metrics."""
        # Metric accumulators (always recorded via finally)
        g2_passed = 0
        g2_rejected_n = 0
        g3_accepted_n = 0
        g3_rejected_n = 0
        g3_rej_cids: list = []
        agg_skipped = True

        try:
            if not results:
                return None, {}

            print(f"\n{'='*60}")
            print(f"  ROUND {server_round} – Zero-Trust Aggregation Pipeline")
            print(f"{'='*60}")

            # ── Identity Binding: mTLS CN ↔ client_id ────────────────
            identity_ok = []
            identity_rejected = 0
            if self.cn_map:
                print(f"\n── Identity Binding: CN ↔ client_id ──")
                for client_proxy, fit_res in results:
                    metrics = fit_res.metrics
                    cid = str(int(metrics.get("client_id", -1)))
                    if _verify_identity_binding(cid, self.cn_map):
                        print(f"[Identity] ✓  client-{cid} identity bound")
                        identity_ok.append((client_proxy, fit_res))
                    else:
                        identity_rejected += 1
                print(f"[Identity] Result: {len(identity_ok)} bound, "
                      f"{identity_rejected} rejected")
            else:
                identity_ok = list(results)

            # ── Gate 2: Signature Verification ───────────────────────────
            verified = []
            gate2_rejected = 0

            if self.enable_gate2:
                print(f"\n── Gate 2: Signature Verification ──")
                for client_proxy, fit_res in identity_ok:
                    metrics = fit_res.metrics
                    cid = str(int(metrics.get("client_id", -1)))
                    sig = metrics.get("signature", "")

                    if cid not in self.public_keys:
                        log(WARNING, "[Gate 2] ✗ REJECTED unknown client_id=%s", cid)
                        gate2_rejected += 1
                        continue

                    if not sig:
                        log(WARNING, "[Gate 2] ✗ REJECTED unsigned update from client_id=%s", cid)
                        gate2_rejected += 1
                        continue

                    ndarrays = parameters_to_ndarrays(fit_res.parameters)
                    if verify_signature(ndarrays, sig, self.public_keys[cid], server_round=server_round):
                        print(f"[Gate 2] ✓  Signature VALID for client-{cid}")
                        verified.append((cid, ndarrays, client_proxy, fit_res))
                    else:
                        log(WARNING, "[Gate 2] ✗ REJECTED tampered update from client_id=%s", cid)
                        gate2_rejected += 1

                print(f"[Gate 2] Result: {len(verified)} passed, {gate2_rejected} rejected")
            else:
                # Gate 2 DISABLED – accept all without signature verification
                print(f"\n── Gate 2: DISABLED (ablation) ──")
                for client_proxy, fit_res in identity_ok:
                    metrics = fit_res.metrics
                    cid = str(int(metrics.get("client_id", -1)))
                    ndarrays = parameters_to_ndarrays(fit_res.parameters)
                    verified.append((cid, ndarrays, client_proxy, fit_res))
                print(f"[Gate 2] DISABLED – all {len(verified)} updates accepted")

            g2_passed = len(verified)
            g2_rejected_n = gate2_rejected

            if not verified:
                log(WARNING, "[Gate 2] No valid updates – skipping aggregation")
                return None, {}

            # ── Gate 3: Anomaly Detection ──────────────────────────────
            if self.enable_gate3:
                print(f"\n── Gate 3: Anomaly Detection ──")
                if self._global_params is None:
                    print("[Gate 3] Round 1: no global reference yet – using weight-norm Z-score")
                else:
                    print("[Gate 3] Global reference available – using delta cosine similarity")
                accepted, gate3_rejected_cids = _filter_anomalous_updates(
                    verified, self.z_threshold, self._global_params
                )

                gate3_rejected = len(gate3_rejected_cids)
                g3_rej_cids = list(gate3_rejected_cids)
                print(f"[Gate 3] Result: {len(accepted)} accepted, "
                      f"{gate3_rejected} rejected {gate3_rejected_cids if gate3_rejected else ''}")

                if not accepted:
                    log(WARNING, "[Gate 3] All updates anomalous – skipping aggregation")
                    return None, {}
            else:
                # Gate 3 DISABLED – accept all verified updates
                print(f"\n── Gate 3: DISABLED (ablation) ──")
                accepted = [(cp, fr) for (_, _, cp, fr) in verified]
                print(f"[Gate 3] DISABLED – all {len(accepted)} updates accepted")

            g3_accepted_n = len(accepted)
            g3_rejected_n = len(g3_rej_cids)

            # Minimum accepted clients guard
            if len(accepted) < self.min_accepted:
                log(
                    WARNING,
                    "[Zero-Trust] Only %d update(s) survived the pipeline "
                    "(minimum required: %d) – skipping aggregation.",
                    len(accepted),
                    self.min_accepted,
                )
                return None, {}

            # ── Aggregation (FedAvg / Krum / Multi-Krum) ──────────────
            method = self.aggregation_method
            print(f"\n── Aggregating {len(accepted)} clean updates via {method} ──")

            if method in ("krum", "multi-krum"):
                agg_weights, total_ex = krum_aggregate(
                    accepted,
                    num_byzantine=self.num_attackers,
                    multi=(method == "multi-krum"),
                )
                agg_params = ndarrays_to_parameters(agg_weights)
                agg_result = (agg_params, {})
            else:
                # Default: FedAvg
                agg_result = super().aggregate_fit(server_round, accepted, failures)

            if agg_result is not None and agg_result[0] is not None:
                self._global_params = parameters_to_ndarrays(agg_result[0])
                print(f"[{method}] Global model snapshot saved for round {server_round + 1}")
                save_round_model(self._global_params, server_round, RESULTS_DIR)
                agg_skipped = False

            return agg_result

        finally:
            if self._metrics:
                self._metrics.record_fit(
                    server_round,
                    gate2_passed=g2_passed,
                    gate2_rejected=g2_rejected_n,
                    gate3_accepted=g3_accepted_n,
                    gate3_rejected=g3_rejected_n,
                    gate3_rejected_cids=g3_rej_cids,
                    aggregation_skipped=agg_skipped,
                    aggregation_method=self.aggregation_method,
                    num_clients_reporting=len(results) if results else 0,
                )

    # Override aggregate_evaluate to capture per-round accuracy
    def aggregate_evaluate(self, server_round, results, failures):
        result = super().aggregate_evaluate(server_round, results, failures)
        if result is not None and self._metrics:
            loss, mets = result
            if loss is not None:
                acc = mets.get("accuracy")
                self._metrics.record_eval(
                    server_round, loss, acc if acc is not None else 0.0,
                )
        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ── Reproducibility ────────────────────────────────────────────────
    if SEED >= 0:
        set_seed(SEED)
        print(f"[SERVER] Seed set to {SEED}")

    # Gate 2: Load client public keys
    public_keys = load_client_public_keys(SIGNING_KEY_DIR, NUM_CLIENTS)

    # Gate 3: Configurable Z-score threshold
    print(f"[Gate 3] Anomaly Z-score threshold: {ANOMALY_Z_THRESHOLD}")
    print(f"[SERVER] Gate 2 (signatures): {'ENABLED' if ENABLE_GATE2 else 'DISABLED'}")
    print(f"[SERVER] Gate 3 (anomaly):    {'ENABLED' if ENABLE_GATE3 else 'DISABLED'}")
    print(f"[SERVER] Aggregation method:  {AGGREGATION_METHOD}")

    MIN_CLIENTS = int(os.environ.get("MIN_CLIENTS", str(NUM_CLIENTS)))
    MIN_ACCEPTED = int(os.environ.get("MIN_ACCEPTED", "2"))
    NUM_ATTACKERS = int(os.environ.get("NUM_ATTACKERS", "0"))

    # ── Identity binding: load client certificate CNs ──────────────────
    cn_map = _load_client_cn_map(CERT_DIR, NUM_CLIENTS) if ENABLE_GATE2 else {}

    # ── Metrics collector ──────────────────────────────────────────────
    mc = MetricsCollector()

    strategy = ZeroTrustFedAvg(
        public_keys=public_keys,
        z_threshold=ANOMALY_Z_THRESHOLD,
        min_accepted=MIN_ACCEPTED,
        enable_gate2=ENABLE_GATE2,
        enable_gate3=ENABLE_GATE3,
        metrics_collector=mc,
        aggregation_method=AGGREGATION_METHOD,
        num_attackers=NUM_ATTACKERS,
        cn_map=cn_map,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=MIN_CLIENTS,
        min_evaluate_clients=MIN_CLIENTS,
        min_available_clients=MIN_CLIENTS,
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
        evaluate_metrics_aggregation_fn=weighted_eval_metrics,
    )

    # Gate 1: Load mTLS certificates
    certificates = _load_certificates()

    print(f"[SERVER] Rounds: {NUM_ROUNDS}")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        certificates=certificates,
    )

    # ── Save structured metrics ────────────────────────────────────────
    mc.save(
        RESULTS_DIR / "metrics.json",
        metadata={
            "seed": SEED,
            "num_rounds": NUM_ROUNDS,
            "num_clients": NUM_CLIENTS,
            "enable_gate1": True,
            "enable_gate2": ENABLE_GATE2,
            "enable_gate3": ENABLE_GATE3,
            "z_threshold": ANOMALY_Z_THRESHOLD,
            "aggregation_method": AGGREGATION_METHOD,
            "num_attackers": NUM_ATTACKERS,
        },
    )


if __name__ == "__main__":
    main()
