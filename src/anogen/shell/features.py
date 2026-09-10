"""Locked evaluation embedding. Do not change without bumping the protocol."""

from __future__ import annotations

import numpy as np

FEATURE_NAME = "feature_pack_v1"
FEATURE_DIM = 12


def embed_windows(x: np.ndarray) -> np.ndarray:
    """Fixed 12-D morphology vector for each univariate window.

    z-path moments, first-difference moments, peak |z|, excursion length,
    and four rFFT log-energy bands. No learned parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.size == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float64)
    n, w = x.shape
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z = (x - mu) / sd
    d = np.diff(z, axis=1)
    absz = np.abs(z)
    peak = absz.max(axis=1)
    high = absz > 1.0
    excursion = np.zeros(n, dtype=np.float64)
    for i in range(n):
        excursion[i] = _longest_run(high[i]) / float(w)

    spec = np.abs(np.fft.rfft(z, axis=1)) ** 2
    n_bins = spec.shape[1]
    edges = np.linspace(0, n_bins, 5, dtype=int)
    bands = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        energy = spec[:, a:b].mean(axis=1) if b > a else np.zeros(n)
        bands.append(np.log(energy + 1e-8))

    feats = np.column_stack(
        [
            z.mean(axis=1),
            z.std(axis=1),
            z.min(axis=1),
            z.max(axis=1),
            d.mean(axis=1),
            d.std(axis=1),
            peak,
            excursion,
            *bands,
        ]
    )
    if feats.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"{FEATURE_NAME} dim {feats.shape[1]} != {FEATURE_DIM}")
    return feats.astype(np.float64)


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for bit in mask:
        if bit:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best
