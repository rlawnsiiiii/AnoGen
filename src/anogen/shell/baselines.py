"""Cheap generators that share the S0 window API. GenIAS arrives in S3."""

from __future__ import annotations

import numpy as np

TAXONOMY_FAMILIES: tuple[str, ...] = (
    "point",
    "contextual",
    "shapelet",
    "seasonal",
    "trend",
)


def posthoc_inject(
    nominal: np.ndarray,
    *,
    rng: np.random.Generator,
    families: tuple[str, ...] = ("step", "pulse", "ramp", "scale"),
    severity: tuple[float, float] = (0.5, 2.0),
) -> np.ndarray:
    """Add a handcrafted excursion to each nominal window."""
    x = np.asarray(nominal, dtype=np.float64).copy()
    if x.ndim == 1:
        x = x[None, :]
    n, w = x.shape
    out = np.empty_like(x)
    for i in range(n):
        fam = families[int(rng.integers(0, len(families)))]
        amp = float(rng.uniform(*severity))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        sd = _std(x[i])
        a = sign * amp * sd
        y = x[i].copy()
        if fam == "step":
            t0 = int(rng.integers(w // 8, max(w // 8 + 1, 7 * w // 8)))
            y[t0:] += a
        elif fam == "pulse":
            t0 = int(rng.integers(0, max(w - 8, 1)))
            dur = int(rng.integers(4, max(5, w // 8)))
            y[t0 : t0 + dur] += a
        elif fam == "ramp":
            t0 = int(rng.integers(0, max(w // 2, 1)))
            t1 = int(rng.integers(t0 + 4, w + 1))
            y[t0:t1] += np.linspace(0.0, a, t1 - t0)
        else:
            y = y * (1.0 + sign * float(rng.uniform(*severity)) * 0.25)
        out[i] = y
    return out.astype(np.float32)


def cutaddpaste_inject(
    nominal: np.ndarray,
    *,
    rng: np.random.Generator,
    width_frac: tuple[float, float] = (0.08, 0.35),
    trend_scale: float = 0.5,
    jitter_sigma: float = 0.05,
    n_paste: int = 1,
    channel_idx: np.ndarray | None = None,
) -> np.ndarray:
    """CutAddPaste (KDD 2024) synthesis: paste a patch from a different donor.

    Cut a contiguous segment from another window (same channel when
    ``channel_idx`` is given), add a linear trend plus Gaussian jitter,
    paste at a random offset. Pastes **real telemetry**, so ARP in φ can
    score high almost for free — same structural advantage as
    posthoc_inject.
    """
    x = np.asarray(nominal, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    n, w = x.shape
    lo, hi = float(width_frac[0]), float(width_frac[1])
    lo = min(max(lo, 1.0 / float(w)), 1.0)
    hi = min(max(hi, lo), 1.0)
    n_paste = max(1, int(n_paste))
    out = x.copy()
    for i in range(n):
        y = out[i]
        sd = _std(y)
        for _ in range(n_paste):
            j = _other_index(i, n, rng, channel_idx=channel_idx)
            L = int(rng.integers(max(4, int(round(lo * w))), max(5, int(round(hi * w)) + 1)))
            L = min(L, w)
            c0 = int(rng.integers(0, max(w - L + 1, 1)))
            p0 = int(rng.integers(0, max(w - L + 1, 1)))
            if n == 1 and c0 == p0 and w - L > 0:
                p0 = (c0 + max(1, L // 2)) % (w - L + 1)
            patch = x[j, c0 : c0 + L].copy()
            slope = float(rng.uniform(-trend_scale, trend_scale)) * sd
            jitter = rng.normal(0.0, jitter_sigma * sd, size=L)
            y[p0 : p0 + L] = patch + np.linspace(0.0, slope, L) + jitter
        out[i] = y
    return out.astype(np.float32)


def taxonomy_inject(
    nominal: np.ndarray,
    *,
    rng: np.random.Generator,
    families: tuple[str, ...] = TAXONOMY_FAMILIES,
    severity: tuple[float, float] = (0.5, 2.0),
    return_labels: bool = False,
    channel_idx: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Lai et al. 2021 (NeurIPS D&B) synthetic outlier taxonomy.

    Families: point, contextual, shapelet, seasonal, trend. Does not replace
    posthoc_inject (locked S4 row). Shapelet pastes a reversed segment from a
    different donor; the other families edit the host window in place.
    """
    x = np.asarray(nominal, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    n, w = x.shape
    if not families:
        raise ValueError("taxonomy_inject needs at least one family")
    unknown = [f for f in families if f not in TAXONOMY_FAMILIES]
    if unknown:
        raise ValueError(f"unknown taxonomy families {unknown}")
    out = np.empty_like(x)
    labels = np.empty(n, dtype=object)
    for i in range(n):
        fam = families[int(rng.integers(0, len(families)))]
        labels[i] = fam
        sev = float(rng.uniform(*severity))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        sd = _std(x[i])
        y = x[i].copy()
        if fam == "point":
            n_pts = int(rng.integers(1, 4))
            idx = rng.choice(w, size=min(n_pts, w), replace=False)
            y[idx] += sign * sev * sd * float(rng.uniform(2.0, 5.0))
        elif fam == "contextual":
            k = max(4, w // 16)
            t = int(rng.integers(k, max(k + 1, w - k)))
            nb = np.concatenate([y[t - k : t], y[t + 1 : t + k + 1]])
            loc_sd = float(nb.std()) if len(nb) else sd
            loc_mu = float(nb.mean()) if len(nb) else float(y[t])
            loc_sd = max(loc_sd, 0.25 * sd, 1e-8)
            y[t] = loc_mu + sign * sev * loc_sd * float(rng.uniform(3.0, 6.0))
        elif fam == "shapelet":
            L = int(rng.integers(max(8, w // 16), max(9, w // 4)))
            L = min(L, w)
            j = _other_index(i, n, rng, channel_idx=channel_idx)
            c0 = int(rng.integers(0, max(w - L + 1, 1)))
            p0 = int(rng.integers(0, max(w - L + 1, 1)))
            if n == 1 and c0 == p0 and w - L > 0:
                p0 = (c0 + max(1, L // 2)) % (w - L + 1)
            patch = x[j, c0 : c0 + L][::-1].copy()
            patch = patch - patch.mean() + y[p0 : p0 + L].mean()
            y[p0 : p0 + L] = patch
        elif fam == "seasonal":
            t0, t1 = _span(w, rng, min_len=max(16, w // 8))
            amp = sign * sev * sd
            period = float(rng.uniform(12.0, max(13.0, w / 4.0)))
            tt = np.arange(t1 - t0, dtype=np.float64)
            y[t0:t1] += amp * np.sin(2.0 * np.pi * tt / period)
        else:  # trend
            t0, t1 = _span(w, rng, min_len=max(8, w // 8))
            y[t0:t1] += np.linspace(0.0, sign * sev * sd, t1 - t0)
        out[i] = y
    gen = out.astype(np.float32)
    if return_labels:
        return gen, labels.astype(str)
    return gen


def _std(y: np.ndarray) -> float:
    sd = float(np.std(y))
    return sd if sd >= 1e-8 else 1.0


def _other_index(
    i: int,
    n: int,
    rng: np.random.Generator,
    channel_idx: np.ndarray | None = None,
) -> int:
    if n <= 1:
        return i
    if channel_idx is not None:
        ch = np.asarray(channel_idx)
        pool = np.flatnonzero(ch == ch[i])
        pool = pool[pool != i]
        if len(pool) == 0:
            return i
        return int(rng.choice(pool))
    j = int(rng.integers(0, n - 1))
    return j if j < i else j + 1


def _span(w: int, rng: np.random.Generator, *, min_len: int) -> tuple[int, int]:
    min_len = min(max(4, min_len), w)
    t0 = int(rng.integers(0, max(w - min_len + 1, 1)))
    t1 = int(rng.integers(t0 + min_len, w + 1))
    return t0, t1
