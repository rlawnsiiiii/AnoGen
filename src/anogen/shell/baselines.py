"""Cheap generators that share the S0 window API. GenIAS arrives in S3."""

from __future__ import annotations

import numpy as np


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
        sd = float(np.std(x[i]))
        if sd < 1e-8:
            sd = 1.0
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
