"""Uncertainty and position-aware additions to the frozen protocol.

Nothing here changes ``feature_pack_v1``, τ, ARP, Coverage@τ or EDI as
defined. It adds:

* event-cluster bootstrap confidence intervals for ARP / Coverage@τ (L8):
  queries are resampled **by event**, because the 672 anomaly windows come
  from 22 events and windows of one event are not independent;
* an optional two-way bootstrap that also resamples the gallery, so the
  generator's own sampling noise enters the interval;
* paired differences between two galleries on the same queries (the
  comparison a reviewer will ask for: is 0.584 vs 0.565 real?);
* position diagnostics that ``feature_pack_v1`` cannot see (it is
  position-invariant by construction): where the dominant event sits, how
  uniform that is, and how far it is from the real anomalies' distribution.

All numpy; no torch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import wasserstein_distance


def distance_matrix(query: np.ndarray, gallery: np.ndarray, block: int = 512) -> np.ndarray:
    """Euclidean (n_q, n_g) distances, computed in blocks to bound memory."""
    q = np.asarray(query, dtype=np.float64)
    g = np.asarray(gallery, dtype=np.float64)
    out = np.empty((len(q), len(g)), dtype=np.float64)
    g2 = (g * g).sum(axis=1)
    for i in range(0, len(q), block):
        qb = q[i : i + block]
        d2 = (qb * qb).sum(axis=1)[:, None] + g2[None, :] - 2.0 * qb @ g.T
        out[i : i + block] = np.sqrt(np.maximum(d2, 0.0))
    return out


def _cluster_draw(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row indices of one cluster bootstrap draw (clusters sampled with replacement)."""
    uniq, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    pick = rng.integers(0, len(uniq), len(uniq))
    return np.concatenate([members[k] for k in pick])


def arp_coverage_ci(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    query_events: np.ndarray,
    *,
    tau: float,
    n_boot: int = 1000,
    seed: int = 0,
    resample_gallery: bool = False,
    level: float = 0.95,
) -> dict[str, Any]:
    """ARP and Coverage@τ with event-cluster bootstrap intervals.

    ``resample_gallery`` also draws the gallery rows with replacement in each
    replicate (two-way bootstrap). Point estimates are the plain protocol
    numbers, so they match ``score_generator`` exactly.
    """
    dm = distance_matrix(query_emb, gallery_emb)
    d = dm.min(axis=1)
    point = {"arp": float(1.0 / (1.0 + d.mean())), "coverage": float(np.mean(d <= tau))}
    rng = np.random.default_rng(int(seed))
    arps = np.empty(int(n_boot))
    covs = np.empty(int(n_boot))
    n_g = dm.shape[1]
    for b in range(int(n_boot)):
        rows = _cluster_draw(query_events, rng)
        if resample_gallery:
            cols = rng.integers(0, n_g, n_g)
            db = dm[np.ix_(rows, cols)].min(axis=1)
        else:
            db = d[rows]
        arps[b] = 1.0 / (1.0 + db.mean())
        covs[b] = np.mean(db <= tau)
    a = (1.0 - level) / 2.0
    return {
        **point,
        "arp_ci": [float(np.quantile(arps, a)), float(np.quantile(arps, 1 - a))],
        "coverage_ci": [float(np.quantile(covs, a)), float(np.quantile(covs, 1 - a))],
        "arp_se": float(arps.std(ddof=1)),
        "n_events": int(len(np.unique(np.asarray(query_events).astype(str)))),
        "n_queries": int(len(d)),
        "n_boot": int(n_boot),
        "two_way": bool(resample_gallery),
    }


def paired_arp_difference(
    query_emb: np.ndarray,
    gallery_a: np.ndarray,
    gallery_b: np.ndarray,
    query_events: np.ndarray,
    *,
    tau: float,
    n_boot: int = 2000,
    seed: int = 0,
    resample_gallery: bool = True,
    aligned: bool = False,
    level: float = 0.95,
) -> dict[str, Any]:
    """ARP(A) − ARP(B) and Coverage(A) − Coverage(B) on the same queries.

    ``aligned``: row i of A and of B come from the same donor and noise draw
    (fixsweep's common random numbers), so the same gallery columns are drawn
    for both; otherwise the two galleries are resampled independently.

    Paired over events (the same event draw for both galleries). With
    ``resample_gallery`` each gallery is also resampled independently.
    ``p_le_zero`` is the bootstrap fraction of replicates with diff <= 0
    (a one-sided bootstrap p-value for "A beats B").
    """
    da = distance_matrix(query_emb, gallery_a)
    db = distance_matrix(query_emb, gallery_b)
    ma, mb = da.min(axis=1), db.min(axis=1)

    def stats(xa: np.ndarray, xb: np.ndarray) -> tuple[float, float]:
        return (
            1.0 / (1.0 + xa.mean()) - 1.0 / (1.0 + xb.mean()),
            float(np.mean(xa <= tau) - np.mean(xb <= tau)),
        )

    point_arp, point_cov = stats(ma, mb)
    rng = np.random.default_rng(int(seed))
    arp_d = np.empty(int(n_boot))
    cov_d = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        rows = _cluster_draw(query_events, rng)
        if resample_gallery:
            ca = rng.integers(0, da.shape[1], da.shape[1])
            cb = ca if (aligned and da.shape[1] == db.shape[1]) else rng.integers(0, db.shape[1], db.shape[1])
            xa = da[np.ix_(rows, ca)].min(axis=1)
            xb = db[np.ix_(rows, cb)].min(axis=1)
        else:
            xa, xb = ma[rows], mb[rows]
        arp_d[b], cov_d[b] = stats(xa, xb)
    a = (1.0 - level) / 2.0
    return {
        "arp_diff": float(point_arp),
        "arp_diff_ci": [float(np.quantile(arp_d, a)), float(np.quantile(arp_d, 1 - a))],
        "arp_p_le_zero": float(np.mean(arp_d <= 0.0)),
        "coverage_diff": float(point_cov),
        "coverage_diff_ci": [float(np.quantile(cov_d, a)), float(np.quantile(cov_d, 1 - a))],
        "coverage_p_le_zero": float(np.mean(cov_d <= 0.0)),
        "n_boot": int(n_boot),
    }


# --------------------------------------------------------------------------
# Position diagnostics (feature_pack_v1 is blind to these)
# --------------------------------------------------------------------------


def peak_positions(x: np.ndarray, width: int = 9) -> np.ndarray:
    """Relative position of the largest |rolling-median residual| per window.

    Same definition as ``realism.diag_signed_peak`` (works on any units; the
    argmax is invariant to a per-channel affine map).
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    r = a - median_filter(a, size=(1, int(width)), mode="nearest")
    idx = np.abs(r).argmax(axis=1)
    return idx.astype(np.float64) / max(1, a.shape[1] - 1)


def position_report(
    x: np.ndarray,
    *,
    real_positions: np.ndarray | None = None,
    n_bins: int = 10,
    edge: float = 0.1,
) -> dict[str, float]:
    """Where the dominant excursion sits.

    peak_at_start / peak_at_end   fraction in the first / last ``edge`` (uniform = edge)
    edge_asymmetry                start − end (a causal backbone pushes this up)
    position_entropy              normalized histogram entropy (1 = uniform)
    position_w1_uniform           W1 to U(0, 1)
    position_w1_real              W1 to the real anomalies' positions, if given
    """
    pos = peak_positions(x)
    h, _ = np.histogram(np.clip(pos, 0.0, 1.0 - 1e-12), bins=int(n_bins), range=(0.0, 1.0))
    p = h / max(h.sum(), 1)
    nz = p[p > 0]
    out = {
        "peak_at_start": float(np.mean(pos < edge)),
        "peak_at_end": float(np.mean(pos > 1.0 - edge)),
        "edge_asymmetry": float(np.mean(pos < edge) - np.mean(pos > 1.0 - edge)),
        "position_entropy": float(-(nz * np.log(nz)).sum() / np.log(int(n_bins))),
        "position_w1_uniform": float(wasserstein_distance(pos, np.linspace(0.0, 1.0, 1001))),
    }
    if real_positions is not None and len(real_positions):
        out["position_w1_real"] = float(wasserstein_distance(pos, np.asarray(real_positions, dtype=np.float64)))
    return out


def envelope_frequency_report(u: np.ndarray, real_u: np.ndarray | None = None) -> dict[str, float]:
    """Envelope exits in channel-span units, split into frequency and magnitude.

    C1's documented failure is the *frequency* (57 % of windows vs 3.9 % real)
    with the right *magnitude*. Reporting the conditional magnitude separately
    keeps a fix that just shrinks everything from looking like a win.
    """
    def one(a: np.ndarray) -> dict[str, float]:
        a = np.asarray(a, dtype=np.float64)
        excess = np.maximum(-a, np.maximum(a - 1.0, 0.0)).max(axis=1)
        exit_ = excess > 0
        return {
            "exit_frac": float(exit_.mean()),
            "excess_p95_given_exit": float(np.quantile(excess[exit_], 0.95)) if exit_.any() else 0.0,
            "excess_max": float(excess.max()) if len(excess) else 0.0,
        }

    out = one(u)
    if real_u is not None:
        r = one(real_u)
        out["exit_frac_ratio"] = float(out["exit_frac"] / max(r["exit_frac"], 1e-12))
        out["real_exit_frac"] = r["exit_frac"]
    return out


def calibration_galleries(
    x_anom: np.ndarray,
    fold_a: np.ndarray,
    x_donor: np.ndarray,
    *,
    fold_id: int,
    rng: np.random.Generator,
    n: int | None = None,
) -> dict[str, np.ndarray]:
    """Reference galleries that bracket ARP / Coverage for one query fold.

    oracle_train_fold_anomalies  real anomaly windows from the *other* folds:
                                 what a generator that reproduced known faults
                                 exactly would score (event-OOF, so not trivial)
    donor_nominal                the untouched donors: what "no anomaly at all"
                                 scores. Any method within its CI of this row
                                 has not shown anomaly-specific realism.
    white_noise                  i.i.d. Gaussian at the donors' scale: a sanity
                                 floor for φ.
    """
    train = np.asarray(x_anom)[np.asarray(fold_a) != int(fold_id)]
    donors = np.asarray(x_donor)
    if n is not None and len(donors) > n:
        donors = donors[rng.choice(len(donors), size=int(n), replace=False)]
    noise = rng.normal(
        loc=donors.mean(axis=1, keepdims=True),
        scale=np.maximum(donors.std(axis=1, keepdims=True), 1e-8),
        size=donors.shape,
    )
    return {
        "oracle_train_fold_anomalies": train,
        "donor_nominal": donors,
        "white_noise": noise.astype(np.float32),
    }


# --------------------------------------------------------------------------
# Fidelity vs diversity, separated (Naeem et al. 2020; Kynkäänniemi et al. 2019)
# --------------------------------------------------------------------------


def prdc(real_emb: np.ndarray, gen_emb: np.ndarray, *, k: int = 5) -> dict[str, float]:
    """Precision, recall, density and coverage in the frozen φ space.

    ARP and EDI each mix fidelity with spread: a gallery that scatters widely
    gets closer to *some* real anomaly (ARP up) and fills more k-means bins
    (EDI up) whether or not its samples look like anomalies. These four
    numbers keep the two axes apart, all against the real anomaly set:

    precision  share of generated points inside some real point's k-NN ball
    recall     share of real points inside some generated point's k-NN ball
    density    mean number of real balls a generated point falls in, / k
    coverage   share of real points whose own k-NN ball contains a generated point

    Naeem et al. (ICML 2020) recommend density/coverage because precision and
    recall are fragile to outliers. k = 5 is their default.
    """
    r = np.asarray(real_emb, dtype=np.float64)
    g = np.asarray(gen_emb, dtype=np.float64)
    kk = int(min(k, len(r) - 1, len(g) - 1))
    if kk < 1:
        return {"precision": float("nan"), "recall": float("nan"), "density": float("nan"), "coverage": float("nan")}
    d_rr = distance_matrix(r, r)
    d_gg = distance_matrix(g, g)
    d_rg = distance_matrix(r, g)
    r_rad = np.sort(d_rr, axis=1)[:, kk]  # kk-th neighbour (index 0 is self)
    g_rad = np.sort(d_gg, axis=1)[:, kk]
    inside = d_rg <= r_rad[:, None]  # (n_r, n_g): gen j inside real i's ball
    return {
        "precision": float(inside.any(axis=0).mean()),
        "recall": float((d_rg <= g_rad[None, :]).any(axis=1).mean()),
        "density": float(inside.sum(axis=0).mean() / kk),
        "coverage": float((d_rg.min(axis=1) <= r_rad).mean()),
        "k": kk,
    }


def feature_attribution(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    names: list[str] | None = None,
) -> dict[str, float]:
    """Which φ components carry the query→nearest-gallery distance.

    For every query, take its nearest gallery point and split the squared
    distance by feature; report each feature's share of the total. If a
    gallery's ARP advantage comes from, say, the high-frequency band energy
    rather than from excursion shape, this is where it shows.
    """
    q = np.asarray(query_emb, dtype=np.float64)
    g = np.asarray(gallery_emb, dtype=np.float64)
    nn = distance_matrix(q, g).argmin(axis=1)
    sq = (q - g[nn]) ** 2
    share = sq.sum(axis=0) / max(sq.sum(), 1e-12)
    names = names or [f"f{i}" for i in range(q.shape[1])]
    return {n: float(v) for n, v in zip(names, share, strict=True)}


PHI_NAMES = [
    "z_mean",
    "z_std",
    "z_min",
    "z_max",
    "dz_mean",
    "dz_std",
    "peak_abs_z",
    "longest_excursion",
    "fft_band1",
    "fft_band2",
    "fft_band3",
    "fft_band4",
]


def query_keep_mask(
    channel_idx: np.ndarray,
    folds: np.ndarray,
    fold_tab: Any,
    *,
    fold_id: int,
    channels: list[str] | None = None,
) -> np.ndarray:
    """Torch-free twin of ``tune._keep_mask`` (same rule, same default names).

    Keep the windows of ``fold_id`` whose channel is not flagged ``below_min``
    for that fold. Default channel names follow ``tune._keep_mask``
    (channel_41 + index); pass ``channels`` for other panels.
    """
    dropped = set(
        fold_tab.loc[(fold_tab["fold"] == fold_id) & (fold_tab["below_min"]), "channel"].astype(str)
    )
    idx = np.asarray(channel_idx)
    names = np.array([channels[int(c)] if channels else f"channel_{41 + int(c)}" for c in idx])
    return (np.asarray(folds) == int(fold_id)) & np.array([n not in dropped for n in names], dtype=bool)
