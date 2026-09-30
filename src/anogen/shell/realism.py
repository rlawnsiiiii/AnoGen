"""Raw-window fidelity measures and anomaly-specific diagnostics.

This module is deliberately independent of ``feature_pack_v1``.  Published
distributional measures live in the first half; project-specific morphology
checks are prefixed ``diag_`` so they cannot be mistaken for standard realism
metrics.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import wasserstein_distance

from anogen.shell.scaler import ChannelMinMax

_EPS = 1e-12
INTERPRETABLE_FEATURES = ("mean", "std", "trend", "skewness", "kurtosis", "acf1")


def _windows(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64)
    if out.ndim == 1:
        out = out[None, :]
    if out.ndim != 2:
        raise ValueError(f"expected (N, W) windows, got shape {out.shape}")
    return out


def _channels(channel_idx: np.ndarray, n: int) -> np.ndarray:
    ch = np.asarray(channel_idx, dtype=np.int64).reshape(-1)
    if len(ch) != n:
        raise ValueError(f"channel_idx has {len(ch)} rows, expected {n}")
    return ch


def channel_span_normalize(
    x: np.ndarray,
    channel_idx: np.ndarray,
    scaler: ChannelMinMax,
) -> np.ndarray:
    """Map raw telemetry to channel training-envelope units without clipping."""
    a = _windows(x)
    ch = _channels(channel_idx, len(a))
    lo = scaler.lo[ch, None]
    span = (scaler.hi - scaler.lo)[ch, None]
    return (a - lo) / np.maximum(span, _EPS)


def _shared_channels(real_ch: np.ndarray, synth_ch: np.ndarray) -> list[int]:
    return sorted(set(np.asarray(real_ch, dtype=int)) & set(np.asarray(synth_ch, dtype=int)))


def marginal_w1(
    real: np.ndarray,
    synth: np.ndarray,
    real_ch: np.ndarray,
    synth_ch: np.ndarray,
    scaler: ChannelMinMax,
) -> dict[str, Any]:
    """Macro channel Wasserstein-1 on raw values in channel-span units."""
    r = channel_span_normalize(real, real_ch, scaler)
    g = channel_span_normalize(synth, synth_ch, scaler)
    rc = _channels(real_ch, len(r))
    gc = _channels(synth_ch, len(g))
    per: dict[str, float] = {}
    for c in _shared_channels(rc, gc):
        per[str(c)] = float(wasserstein_distance(r[rc == c].ravel(), g[gc == c].ravel()))
    return _macro_result(per)


def marginal_distribution_difference(
    real: np.ndarray,
    synth: np.ndarray,
    real_ch: np.ndarray,
    synth_ch: np.ndarray,
    scaler: ChannelMinMax,
    *,
    bins: int = 50,
    skip_first: bool = True,
) -> dict[str, Any]:
    """TSGBench MDD, following ``HistoLoss``/``calculate_mdd`` in TSGBench.

    For every channel and time step the real windows define ``bins`` equal-width
    edges spanning their own range, and a histogram *density*: count divided by
    the bin width and by the number of windows.  Generated values are binned on
    those same edges and divided by the generated count, so generated mass
    outside the real range contributes nothing, as in ``HistoLoss.compute``.
    The score is the mean absolute density difference over bins, time steps and
    channels.  TSGBench skips the first time step; ``skip_first`` reproduces it.

    The dropped out-of-range mass is reported separately because it is invisible
    to the score itself.
    """
    if int(bins) < 2:
        raise ValueError("bins must be >= 2")
    r = channel_span_normalize(real, real_ch, scaler)
    g = channel_span_normalize(synth, synth_ch, scaler)
    rc = _channels(real_ch, len(r))
    gc = _channels(synth_ch, len(g))
    start = 1 if skip_first else 0
    per: dict[str, float] = {}
    outside: dict[str, float] = {}
    for c in _shared_channels(rc, gc):
        rv = r[rc == c]
        gv = g[gc == c]
        if rv.shape[1] <= start:
            continue
        step_diff = np.empty(rv.shape[1] - start, dtype=np.float64)
        step_drop = np.empty(rv.shape[1] - start, dtype=np.float64)
        for j, t in enumerate(range(start, rv.shape[1])):
            rt, gt = rv[:, t], gv[:, t]
            lo, hi = float(rt.min()), float(rt.max())
            if hi == lo:
                hi = lo + 1e-5
            edges = np.linspace(lo, hi, int(bins) + 1)
            width = float(edges[1] - edges[0])
            rh = np.histogram(rt, bins=edges)[0] / width / float(len(rt))
            gh = np.histogram(gt, bins=edges)[0] / width / float(len(gt))
            step_diff[j] = float(np.mean(np.abs(rh - gh)))
            step_drop[j] = float(np.mean((gt < lo) | (gt > hi)))
        per[str(c)] = float(step_diff.mean())
        outside[str(c)] = float(step_drop.mean())
    result = _macro_result(per)
    result["out_of_real_range_fraction"] = _macro_result(outside)
    return result


def acf_rows(x: np.ndarray, max_lag: int = 64) -> np.ndarray:
    """Biased normalized ACF for every row, including lag zero."""
    a = _windows(x)
    if a.shape[1] == 0:
        return np.zeros((len(a), 0), dtype=np.float64)
    lag = min(max(0, int(max_lag)), a.shape[1] - 1)
    z = a - a.mean(axis=1, keepdims=True)
    denom = np.sum(z * z, axis=1)
    out = np.zeros((len(a), lag + 1), dtype=np.float64)
    out[:, 0] = 1.0
    valid = denom > _EPS
    for k in range(1, lag + 1):
        num = np.sum(z[:, :-k] * z[:, k:], axis=1)
        out[valid, k] = num[valid] / denom[valid]
    return out


def pooled_acf(x: np.ndarray, max_lag: int = 64) -> np.ndarray:
    """TSGBench ``acf_torch`` for one channel: mean and variance pooled.

    A single mean and biased variance are taken over windows and time, and the
    lag-``k`` product is averaged over all windows and all ``W - k`` valid
    offsets.  Returns lags ``0 .. min(max_lag, W) - 1``, so lag zero is one.
    """
    a = _windows(x)
    w = a.shape[1]
    lag = min(max(0, int(max_lag)), w)
    if len(a) == 0 or lag == 0:
        return np.zeros(lag, dtype=np.float64)
    z = a - a.mean()
    var = float(np.mean(z * z))
    out = np.zeros(lag, dtype=np.float64)
    out[0] = 1.0
    if var <= _EPS:
        return out
    for k in range(1, lag):
        out[k] = float(np.mean(z[:, k:] * z[:, :-k])) / var
    return out


def autocorrelation_difference(
    real: np.ndarray,
    synth: np.ndarray,
    real_ch: np.ndarray,
    synth_ch: np.ndarray,
    *,
    max_lag: int = 64,
) -> dict[str, Any]:
    """TSGBench ACD, following ``acf_torch``/``acf_diff``/``calculate_acd``.

    Per channel the pooled ACF vectors are compared by their Euclidean norm,
    then macro-averaged over channels.  The pooled ACF is invariant to the
    per-channel affine rescaling used elsewhere in this module, so the raw
    windows are used directly.
    """
    r, g = _windows(real), _windows(synth)
    rc, gc = _channels(real_ch, len(r)), _channels(synth_ch, len(g))
    per: dict[str, float] = {}
    curves: dict[str, dict[str, list[float]]] = {}
    for c in _shared_channels(rc, gc):
        ra = pooled_acf(r[rc == c], max_lag)
        ga = pooled_acf(g[gc == c], max_lag)
        n = min(len(ra), len(ga))
        per[str(c)] = float(np.sqrt(np.sum((ra[:n] - ga[:n]) ** 2)))
        curves[str(c)] = {"real": ra.tolist(), "synth": ga.tolist()}
    result = _macro_result(per)
    result["curves"] = curves
    return result


def dominant_fft_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dominant non-DC rFFT bin and sinusoidal amplitude for every window.

    The ``2 |X_k| / W`` scaling returns the amplitude ``A`` of a pure sinusoid
    ``A sin(2 pi k t / W)``, which is the quantity COSCI-GAN compares.  Ties in
    the periodogram resolve to the lowest bin.
    """
    a = _windows(x)
    if a.shape[1] < 3:
        return np.zeros(len(a)), np.zeros(len(a))
    centered = a - a.mean(axis=1, keepdims=True)
    amp = 2.0 * np.abs(np.fft.rfft(centered, axis=1)) / float(a.shape[1])
    amp = amp[:, 1:]
    idx = np.argmax(amp, axis=1)
    peak = amp[np.arange(len(a)), idx]
    return (idx + 1).astype(np.float64), peak.astype(np.float64)


def fft_average_wasserstein(
    real: np.ndarray,
    synth: np.ndarray,
    real_ch: np.ndarray,
    synth_ch: np.ndarray,
    scaler: ChannelMinMax,
) -> dict[str, Any]:
    """COSCI-GAN AWD: per-channel W1 of amplitudes, macro-averaged.

    Seyfi et al. define AWD as the channel average of the Wasserstein distance
    between real and generated amplitude distributions; their toy study knew the
    generating amplitude, so the extraction step is only described in Stenger et
    al.'s survey as "FFT determines the most likely period, then the amplitudes
    are compared".  ``dominant_fft_features`` is our frozen reading of that
    step.  Channels whose real amplitudes vanish are skipped rather than scored,
    which covers the survey's caveat about aperiodic series.

    ``frequency_w1`` is an extra project diagnostic, not part of AWD.
    """
    r = channel_span_normalize(real, real_ch, scaler)
    g = channel_span_normalize(synth, synth_ch, scaler)
    rc, gc = _channels(real_ch, len(r)), _channels(synth_ch, len(g))
    amp_per: dict[str, float] = {}
    freq_per: dict[str, float] = {}
    skipped: list[int] = []
    for c in _shared_channels(rc, gc):
        rf, ra = dominant_fft_features(r[rc == c])
        gf, ga = dominant_fft_features(g[gc == c])
        if len(ra) == 0 or float(np.max(ra)) <= _EPS:
            skipped.append(c)
            continue
        amp_per[str(c)] = float(wasserstein_distance(ra, ga))
        # Normalize frequency-bin distance by the number of non-DC bins.
        freq_per[str(c)] = float(
            wasserstein_distance(rf, gf) / max(1, r.shape[1] // 2)
        )
    result = _macro_result(amp_per)
    result["frequency_w1"] = _macro_result(freq_per)
    result["skipped_nonperiodic_channels"] = skipped
    return result


def mean_psd(x: np.ndarray) -> np.ndarray:
    """Mean one-sided periodogram after subtracting each window mean."""
    a = _windows(x)
    if len(a) == 0:
        return np.zeros(a.shape[1] // 2 + 1, dtype=np.float64)
    z = a - a.mean(axis=1, keepdims=True)
    return (np.abs(np.fft.rfft(z, axis=1)) ** 2 / max(1, a.shape[1])).mean(axis=0)


def interpretable_features(x: np.ndarray) -> np.ndarray:
    """Per-window statistics used by feature-distribution fidelity."""
    a = _windows(x)
    if len(a) == 0:
        return np.zeros((0, len(INTERPRETABLE_FEATURES)), dtype=np.float64)
    w = a.shape[1]
    t = np.linspace(-0.5, 0.5, w, dtype=np.float64)
    tc = t - t.mean()
    denom = float(np.sum(tc * tc))
    centered = a - a.mean(axis=1, keepdims=True)
    trend = (centered @ tc) / max(denom, _EPS)
    sd = a.std(axis=1, keepdims=True)
    z = np.divide(centered, sd, out=np.zeros_like(centered), where=sd > _EPS)
    sk = np.mean(z**3, axis=1)
    ku = np.mean(z**4, axis=1) - 3.0
    acf1 = acf_rows(a, 1)[:, 1] if w > 1 else np.zeros(len(a))
    return np.column_stack([a.mean(axis=1), a.std(axis=1), trend, sk, ku, acf1])


def feature_distribution_distance(
    real: np.ndarray,
    synth: np.ndarray,
    real_ch: np.ndarray,
    synth_ch: np.ndarray,
    scaler: ChannelMinMax,
) -> dict[str, Any]:
    """Per-feature W1 on channel-span-normalized windows."""
    r = channel_span_normalize(real, real_ch, scaler)
    g = channel_span_normalize(synth, synth_ch, scaler)
    rc, gc = _channels(real_ch, len(r)), _channels(synth_ch, len(g))
    feature_macro: dict[str, float] = {}
    per_channel: dict[str, dict[str, float]] = {}
    for c in _shared_channels(rc, gc):
        rf, gf = interpretable_features(r[rc == c]), interpretable_features(g[gc == c])
        row = {
            name: float(wasserstein_distance(rf[:, j], gf[:, j]))
            for j, name in enumerate(INTERPRETABLE_FEATURES)
        }
        per_channel[str(c)] = row
    for name in INTERPRETABLE_FEATURES:
        vals = [row[name] for row in per_channel.values()]
        feature_macro[name] = float(np.mean(vals)) if vals else float("nan")
    vals = list(feature_macro.values())
    return {
        "value": float(np.mean(vals)) if vals else float("nan"),
        "features": feature_macro,
        "per_channel": per_channel,
        "n_channels": len(per_channel),
    }


def grouped_c2st_split(
    real_groups: np.ndarray,
    synth_groups: np.ndarray,
    *,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Separate group splits for real and synthetic rows."""
    from sklearn.model_selection import GroupShuffleSplit

    rg = np.asarray(real_groups).reshape(-1)
    sg = np.asarray(synth_groups).reshape(-1)
    if len(np.unique(rg)) < 2 or len(np.unique(sg)) < 2:
        raise ValueError("C2ST requires at least two real and synthetic groups")
    splitter_r = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    splitter_s = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + 1)
    rtr, rte = next(splitter_r.split(np.zeros(len(rg)), groups=rg))
    str_, ste = next(splitter_s.split(np.zeros(len(sg)), groups=sg))
    return rtr, rte, str_, ste


def classifier_two_sample_test(
    real: np.ndarray,
    synth: np.ndarray,
    real_ch: np.ndarray,
    synth_ch: np.ndarray,
    real_groups: np.ndarray,
    synth_groups: np.ndarray,
    scaler: ChannelMinMax,
    *,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    test_size: float = 0.3,
    n_perm: int = 200,
) -> list[dict[str, float]]:
    """Balanced grouped C2ST with a frozen logistic classifier."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    r = channel_span_normalize(real, real_ch, scaler)
    g = channel_span_normalize(synth, synth_ch, scaler)
    rc, gc = _channels(real_ch, len(r)), _channels(synth_ch, len(g))
    rg, sg = np.asarray(real_groups).reshape(-1), np.asarray(synth_groups).reshape(-1)
    if len(rg) != len(r) or len(sg) != len(g):
        raise ValueError("C2ST group arrays must align with windows")
    rows: list[dict[str, float]] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        ri, gi = _balanced_channel_indices(rc, gc, rng)
        rtr, rte, str_, ste = grouped_c2st_split(
            rg[ri], sg[gi], test_size=float(test_size), seed=int(seed)
        )
        x_train = np.concatenate([r[ri][rtr], g[gi][str_]], axis=0)
        y_train = np.concatenate([np.ones(len(rtr)), np.zeros(len(str_))])
        x_test = np.concatenate([r[ri][rte], g[gi][ste]], axis=0)
        y_test = np.concatenate([np.ones(len(rte)), np.zeros(len(ste))])
        clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=int(seed))
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        acc = float(balanced_accuracy_score(y_test, pred))
        null = np.empty(max(0, int(n_perm)), dtype=np.float64)
        for i in range(len(null)):
            null[i] = balanced_accuracy_score(rng.permutation(y_test), pred)
        p = (
            float((1 + np.sum(null >= acc)) / (1 + len(null)))
            if len(null)
            else float("nan")
        )
        rows.append(
            {
                "seed": float(seed),
                "balanced_accuracy": acc,
                "discriminative_score": abs(acc - 0.5),
                "permutation_p": p,
                "n_train": float(len(y_train)),
                "n_test": float(len(y_test)),
            }
        )
    return rows


def diag_envelope(
    x: np.ndarray,
    channel_idx: np.ndarray,
    scaler: ChannelMinMax,
) -> dict[str, Any]:
    """Training-envelope violations in channel-span units."""
    u = channel_span_normalize(x, channel_idx, scaler)
    bad = (u < 0.0) | (u > 1.0)
    excess = np.maximum(-u, np.maximum(u - 1.0, 0.0))
    return {
        "sample_fraction": float(bad.mean()) if bad.size else 0.0,
        "window_fraction": float(bad.any(axis=1).mean()) if len(u) else 0.0,
        "max_excess": float(excess.max()) if excess.size else 0.0,
        "per_window_fraction": bad.mean(axis=1),
        "per_window_max_excess": excess.max(axis=1) if len(u) else np.zeros(0),
    }


def rolling_residual(x: np.ndarray, width: int = 9) -> np.ndarray:
    a = _windows(x)
    k = int(width)
    if k < 1 or k % 2 == 0:
        raise ValueError("rolling median width must be a positive odd integer")
    return a - median_filter(a, size=(1, k), mode="nearest")


def diag_signed_peak(x: np.ndarray, width: int = 9) -> dict[str, np.ndarray]:
    residual = rolling_residual(x, width)
    if residual.shape[1] == 0:
        return {"severity": np.zeros(len(residual)), "position": np.zeros(len(residual))}
    idx = np.abs(residual).argmax(axis=1)
    severity = residual[np.arange(len(residual)), idx]
    denom = max(1, residual.shape[1] - 1)
    return {"severity": severity, "position": idx.astype(np.float64) / denom}


def diag_cusum(x: np.ndarray, *, min_segment: int = 1) -> dict[str, np.ndarray]:
    """Maximum variance-normalized mean-change contrast and its split."""
    a = _windows(x)
    n, w = a.shape
    m = max(1, int(min_segment))
    if w < 2 * m:
        return {"contrast": np.zeros(n), "position": np.zeros(n)}
    cs = np.cumsum(a, axis=1)
    total = cs[:, -1]
    best_abs = np.full(n, -np.inf)
    best_signed = np.zeros(n)
    best_s = np.full(n, m)
    for s in range(m, w - m + 1):
        left = cs[:, s - 1] / float(s)
        right = (total - cs[:, s - 1]) / float(w - s)
        val = np.sqrt(s * (w - s) / float(w)) * (left - right)
        take = np.abs(val) > best_abs
        best_abs[take] = np.abs(val[take])
        best_signed[take] = val[take]
        best_s[take] = s
    return {"contrast": best_signed, "position": best_s.astype(np.float64) / float(w)}


def diag_persistence(x: np.ndarray) -> np.ndarray:
    a = _windows(x)
    q = max(1, a.shape[1] // 4)
    return a[:, -q:].mean(axis=1) - a[:, :q].mean(axis=1)


def diag_excursion_count(
    x: np.ndarray,
    *,
    width: int = 9,
    mad_scale: float = 4.0,
) -> np.ndarray:
    residual = rolling_residual(x, width)
    med = np.median(residual, axis=1, keepdims=True)
    mad = np.median(np.abs(residual - med), axis=1, keepdims=True)
    threshold = np.maximum(float(mad_scale) * mad, _EPS)
    mask = np.abs(residual - med) > threshold
    starts = mask.copy()
    if mask.shape[1] > 1:
        starts[:, 1:] &= ~mask[:, :-1]
    return starts.sum(axis=1).astype(np.int64)


def diag_first_difference(x: np.ndarray) -> dict[str, np.ndarray]:
    a = _windows(x)
    if a.shape[1] < 2:
        z = np.zeros(len(a))
        return {"std": z, "p99": z, "p999": z}
    d = np.abs(np.diff(a, axis=1))
    return {
        "std": np.diff(a, axis=1).std(axis=1),
        "p99": np.quantile(d, 0.99, axis=1),
        "p999": np.quantile(d, 0.999, axis=1),
    }


def _balanced_channel_indices(
    real_ch: np.ndarray,
    synth_ch: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    ri: list[int] = []
    gi: list[int] = []
    for c in _shared_channels(real_ch, synth_ch):
        r = np.flatnonzero(real_ch == c)
        g = np.flatnonzero(synth_ch == c)
        n = min(len(r), len(g))
        if n:
            ri.extend(rng.choice(r, size=n, replace=False).tolist())
            gi.extend(rng.choice(g, size=n, replace=False).tolist())
    if not ri:
        raise ValueError("real and synthetic sets share no populated channels")
    return np.asarray(ri, dtype=int), np.asarray(gi, dtype=int)


def _macro_result(per_channel: dict[str, float]) -> dict[str, Any]:
    vals = list(per_channel.values())
    return {
        "value": float(np.mean(vals)) if vals else float("nan"),
        "per_channel": per_channel,
        "n_channels": len(per_channel),
    }
