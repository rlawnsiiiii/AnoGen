"""Gallery diagnostics for the testbed, reusing the repo's realism measures."""

from __future__ import annotations

from typing import Any

import numpy as np

from anogen.shell.realism import (
    diag_cusum,
    diag_first_difference,
    diag_persistence,
    diag_signed_peak,
)


def envelope_units(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - lo) / max(hi - lo, 1e-12)


def position_profile(position: np.ndarray, n_bins: int = 10) -> np.ndarray:
    h, _ = np.histogram(np.clip(position, 0, 1 - 1e-9), bins=n_bins, range=(0.0, 1.0))
    return h / max(h.sum(), 1)


def position_entropy(position: np.ndarray, n_bins: int = 10) -> float:
    """Normalized entropy of the peak-position histogram (1 = uniform)."""
    p = position_profile(position, n_bins)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(n_bins))


def gallery_report(x: np.ndarray, lo: float, hi: float, *, median_width: int = 9) -> dict[str, Any]:
    u = envelope_units(x, lo, hi)
    bad = (u < 0) | (u > 1)
    excess = np.maximum(-u, np.maximum(u - 1, 0))
    peak = diag_signed_peak(u, width=median_width)
    cus = diag_cusum(u, min_segment=8)
    diff = diag_first_difference(u)
    pos = peak["position"]
    return {
        "env_window_frac": float(bad.any(axis=1).mean()),
        "env_max_excess": float(excess.max()),
        "env_mean_excess_given_exit": float(excess.max(axis=1)[bad.any(axis=1)].mean()) if bad.any() else 0.0,
        "cusum_mean": float(np.mean(cus["contrast"])),
        "cusum_abs_mean": float(np.mean(np.abs(cus["contrast"]))),
        "cusum_pos_std": float(np.std(cus["position"])),
        "persistence_abs_mean": float(np.mean(np.abs(diag_persistence(u)))),
        "diff_p999": float(np.mean(diff["p999"])),
        "peak_at_start": float(np.mean(pos < 0.1)),
        "peak_at_end": float(np.mean(pos > 0.9)),
        "peak_pos_entropy": position_entropy(pos),
        "peak_abs_severity": float(np.mean(np.abs(peak["severity"]))),
    }


def edge_stats(x: np.ndarray, ref: np.ndarray, edge: int = 16) -> dict[str, float]:
    """Per-position std of a gallery vs a reference set, first/middle/last ``edge`` bins.

    Ratio 1 means the marginal spread at that position is right.
    """
    sx, sr = np.std(x, axis=0), np.std(ref, axis=0)
    w = x.shape[1]
    mid = slice(w // 2 - edge // 2, w // 2 + edge // 2)
    d1x = np.diff(x, axis=1).std(axis=0)
    d1r = np.diff(ref, axis=1).std(axis=0)
    return {
        "std_ratio_start": float(sx[:edge].mean() / sr[:edge].mean()),
        "std_ratio_mid": float(sx[mid].mean() / sr[mid].mean()),
        "std_ratio_end": float(sx[-edge:].mean() / sr[-edge:].mean()),
        "dstd_ratio_start": float(d1x[:edge].mean() / d1r[:edge].mean()),
        "dstd_ratio_mid": float(d1x[mid].mean() / d1r[mid].mean()),
        "dstd_ratio_end": float(d1x[-edge:].mean() / d1r[-edge:].mean()),
    }
