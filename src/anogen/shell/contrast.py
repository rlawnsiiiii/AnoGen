"""Parametric morphology targets: a type-specific energy that does not need prototypes.

The proto term ‖e(x̂₀) − a_i‖² asks for "this one labelled window". With 6
level-shift windows from a single event (all in fold 0) that is the same
target for ~64 samples each, at the same position. The level-shift slice
then collapses to one stereotyped early transient (NEXT_STEPS L1).

A *contrast* energy asks for the defining property of the type instead:

    d(x) = Σ_t w[t] x[t]        f = (d(x) − δ)²

with w a fixed linear contrast placed at a sampled position p and δ an
amplitude sampled from the train-fold examples of that type. Two contrasts:

    step    w = mean over [p, end) − mean over [start, p)    persistent level change
            (``step_span`` m limits both sides to ±m bins; the default None
            uses the whole window, which a local bump cannot satisfy)
    spike   w = mean over [p−h, p+h] − mean of the flanks   short excursion

Position is sampled per sample (uniform over the interior), so there is no
position collapse. The amplitude comes from the real examples (few-shot in
the same sense as the proto term), and the diffusion prior supplies the
texture. The energy is quadratic in x, so ∇f = 2 (d − δ) w is exact and cheap.

Everything here is numpy; ``steer.guided_ddim`` only evaluates (x·w − δ)².
"""

from __future__ import annotations

import numpy as np

CONTRAST_KINDS = ("step", "spike")


def contrast_weights(
    kind: str,
    positions: np.ndarray,
    width: int,
    *,
    step_span: int | None = None,
    spike_half: int = 1,
    flank_gap: int = 3,
    flank_len: int = 8,
) -> np.ndarray:
    """(N, width) contrast vectors, one per sampled position (bins)."""
    pos = np.asarray(positions, dtype=np.int64).reshape(-1)
    w = np.zeros((len(pos), int(width)), dtype=np.float64)
    for i, p in enumerate(pos):
        if kind == "step":
            span = int(step_span) if step_span else int(width)
            lo, hi = max(0, p - span), min(width, p + span)
            if p - lo < 1 or hi - p < 1:
                raise ValueError(f"step position {p} leaves an empty side")
            w[i, p:hi] += 1.0 / (hi - p)
            w[i, lo:p] -= 1.0 / (p - lo)
        elif kind == "spike":
            c0, c1 = max(0, p - spike_half), min(width, p + spike_half + 1)
            w[i, c0:c1] += 1.0 / (c1 - c0)
            left = np.arange(max(0, c0 - flank_gap - flank_len), max(0, c0 - flank_gap))
            right = np.arange(min(width, c1 + flank_gap), min(width, c1 + flank_gap + flank_len))
            flank = np.concatenate([left, right])
            if len(flank) == 0:
                raise ValueError(f"spike position {p} has no flank")
            w[i, flank] -= 1.0 / len(flank)
        else:
            raise ValueError(f"unknown contrast kind {kind!r} (use {CONTRAST_KINDS})")
    return w


def measure_contrast(kind: str, x: np.ndarray, **kw: int) -> tuple[np.ndarray, np.ndarray]:
    """Best contrast value and its position for each window (exhaustive scan).

    step: the split with the largest |mean(right) − mean(left)| (whole window by
    default; splits closer than W/16 to an edge are not considered);
    spike: the bin with the largest |centre − flanks|. Used to read the
    amplitude distribution off real examples and to score generated windows.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    n, w = a.shape
    if kind == "step":
        cand = np.arange(max(8, w // 16), w - max(8, w // 16))
    elif kind == "spike":
        m = int(kw.get("flank_gap", 3)) + int(kw.get("flank_len", 8)) + int(kw.get("spike_half", 1))
        cand = np.arange(m, w - m)
    else:
        raise ValueError(kind)
    wts = contrast_weights(kind, cand, w, **kw)  # (C, W)
    vals = a @ wts.T  # (N, C)
    best = np.abs(vals).argmax(axis=1)
    return vals[np.arange(n), best], cand[best].astype(np.float64) / max(w - 1, 1)


def sample_targets(
    kind: str,
    n: int,
    width: int,
    real_amplitudes: np.ndarray,
    *,
    rng: np.random.Generator,
    margin: float = 0.1,
    **kw: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions (uniform on the interior), amplitudes (resampled from the real
    examples, sign kept), and the matching contrast weights."""
    amps = np.asarray(real_amplitudes, dtype=np.float64).reshape(-1)
    if amps.size == 0:
        raise ValueError("need at least one real amplitude")
    if kind == "step":
        guard = max(8, width // 16)
    else:
        guard = int(kw.get("flank_gap", 3)) + int(kw.get("flank_len", 8)) + int(kw.get("spike_half", 1))
    lo = max(guard, int(margin * width))
    hi = min(width - guard, int((1.0 - margin) * width))
    pos = rng.integers(lo, hi, size=int(n))
    delta = rng.choice(amps, size=int(n), replace=True)
    return pos, delta, contrast_weights(kind, pos, width, **kw)


def contrast_mask(
    kind: str,
    positions: np.ndarray,
    width: int,
    *,
    dilate: int = 8,
    spike_half: int = 1,
    flank_gap: int = 3,
) -> np.ndarray:
    """(N, width) bool: the bins to *generate* for a masked (RePaint) edit.

    step   [p − dilate, end): the shifted tail, starting ``dilate`` bins early so
           the denoiser makes the transition instead of the mask edge;
    spike  [p − h − gap, p + h + gap]: the excursion and the gap to its flanks.
           ``dilate`` is not applied: the gap already gives the transition room,
           and the flanks must stay donor so the contrast is measured against
           real context (and cannot be met by pushing the flanks).
    The mask is also the bin-level label of the generated anomaly (for a step
    it starts ``dilate`` bins before the requested onset).
    """
    pos = np.asarray(positions, dtype=np.int64).reshape(-1)
    idx = np.arange(int(width))[None, :]
    if kind == "step":
        lo = np.maximum(pos - int(dilate), 0)[:, None]
        return idx >= lo
    if kind == "spike":
        r = int(spike_half) + int(flank_gap)
        return (idx >= (pos - r)[:, None]) & (idx <= (pos + r)[:, None])
    raise ValueError(f"unknown contrast kind {kind!r} (use {CONTRAST_KINDS})")
