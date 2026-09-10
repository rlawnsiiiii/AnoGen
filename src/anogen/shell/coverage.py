"""Coverage@τ, rare collision, gap, and GenIAS-style ARP / EDI on the locked embedding."""

from __future__ import annotations

import numpy as np

from anogen.shell.features import embed_windows
from anogen.shell.protocol import choose_tau


def min_distances(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """For each query row, L2 distance to the nearest gallery row."""
    q = np.asarray(query, dtype=np.float64)
    g = np.asarray(gallery, dtype=np.float64)
    if q.size == 0:
        return np.zeros(0, dtype=np.float64)
    if g.size == 0:
        return np.full(q.shape[0], np.inf)
    # (n_q, n_g)
    d2 = ((q[:, None, :] - g[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(d2.min(axis=1))


def coverage_at_tau(min_dists: np.ndarray, tau: float) -> float:
    d = np.asarray(min_dists, dtype=np.float64)
    if d.size == 0:
        return 0.0
    return float(np.mean(d <= tau))


def arp(min_dists: np.ndarray) -> float:
    """GenIAS ARP: 1 / (1 + mean min-distance from query to gallery)."""
    d = np.asarray(min_dists, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0
    return float(1.0 / (1.0 + np.mean(d)))


def edi_by_method(
    embs: dict[str, np.ndarray],
    *,
    n_regions: int = 16,
    seed: int = 0,
) -> dict[str, float]:
    """Shannon entropy of each gallery over k-means bins of the union (GenIAS EDI)."""
    from sklearn.cluster import KMeans

    names = [n for n, v in embs.items() if len(v)]
    if not names:
        return {}
    stacked = np.concatenate([np.asarray(embs[n], dtype=np.float64) for n in names], axis=0)
    k = max(2, min(int(n_regions), len(stacked)))
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(stacked)
    out: dict[str, float] = {}
    offset = 0
    for name in names:
        n = len(embs[name])
        lab = labels[offset : offset + n]
        offset += n
        counts = np.bincount(lab, minlength=k).astype(np.float64)
        p = counts / max(counts.sum(), 1.0)
        p = p[p > 0]
        out[name] = float(-(p * np.log(p)).sum())
    return out


def mean_pairwise_distance(emb: np.ndarray) -> float:
    x = np.asarray(emb, dtype=np.float64)
    if x.shape[0] < 2:
        return 0.0
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    n = x.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(np.sqrt(d2[iu]).mean())


def score_generator(
    real_anomaly: np.ndarray,
    real_rare: np.ndarray,
    generated: np.ndarray,
    *,
    tau: float | None = None,
    unguided_anomaly_dists: np.ndarray | None = None,
    unguided_target: float = 0.10,
) -> dict[str, float]:
    """Coverage of real anomaly vs rare windows by a generated gallery."""
    qa = embed_windows(real_anomaly)
    qr = embed_windows(real_rare)
    gal = embed_windows(generated)
    d_a = min_distances(qa, gal)
    d_r = min_distances(qr, gal)
    if tau is None:
        if unguided_anomaly_dists is None:
            raise ValueError("tau or unguided_anomaly_dists is required")
        tau = choose_tau(unguided_anomaly_dists, unguided_target)
    cov_a = coverage_at_tau(d_a, tau)
    cov_r = coverage_at_tau(d_r, tau)
    mean_a = float(np.mean(d_a)) if len(d_a) else float("inf")
    mean_r = float(np.mean(d_r)) if len(d_r) else float("inf")
    return {
        "tau": float(tau),
        "coverage_anomaly": cov_a,
        "coverage_rare": cov_r,
        "gap": cov_a - cov_r,
        "mean_min_d_anomaly": mean_a,
        "mean_min_d_rare": mean_r,
        "arp_anomaly": arp(d_a),
        "arp_rare": arp(d_r),
        "diversity": mean_pairwise_distance(gal),
        "n_generated": float(len(generated)),
        "n_anomaly": float(len(real_anomaly)),
        "n_rare": float(len(real_rare)),
    }
