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

CONTRAST_KINDS = ("step", "spike", "step_head")


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
        elif kind == "step_head":
            # mirror of "step": the *head* is the shifted side, so a positive
            # amplitude lifts a short early stretch and leaves the tail alone
            span = int(step_span) if step_span else int(width)
            lo, hi = max(0, p - span), min(width, p + span)
            if p - lo < 1 or hi - p < 1:
                raise ValueError(f"step position {p} leaves an empty side")
            w[i, lo:p] += 1.0 / (p - lo)
            w[i, p:hi] -= 1.0 / (hi - p)
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
    if kind in ("step", "step_head"):
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
    pos_range: tuple[float, float] | None = None,
    **kw: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions (uniform on the interior), amplitudes (resampled from the real
    examples, sign kept), and the matching contrast weights.

    ``pos_range`` (lo, hi) as fractions of the width narrows where the position
    may fall; None keeps the frozen symmetric ``margin`` interior. For a step the
    position also sets *which side is short*: the projection lifts the before-side
    by (d − δ)/(1 + n_before/n_after), so a position at 0.05 W puts almost the
    whole of a negative amplitude on a short early stretch and leaves the rest of
    the window alone -- the morphology every real Mission 1 level shift has.
    """
    amps = np.asarray(real_amplitudes, dtype=np.float64).reshape(-1)
    if amps.size == 0:
        raise ValueError("need at least one real amplitude")
    if kind in ("step", "step_head"):
        guard = max(8, width // 16)
    else:
        guard = int(kw.get("flank_gap", 3)) + int(kw.get("flank_len", 8)) + int(kw.get("spike_half", 1))
    f_lo, f_hi = (margin, 1.0 - margin) if pos_range is None else pos_range
    lo = max(guard, int(f_lo * width))
    hi = min(width - guard, int(f_hi * width))
    if hi <= lo:
        raise ValueError(f"pos_range {(f_lo, f_hi)} leaves no room past the guard of {guard}")
    pos = rng.integers(lo, hi, size=int(n))
    delta = rng.choice(amps, size=int(n), replace=True)
    return pos, delta, contrast_weights(kind, pos, width, **kw)


def mirror_rows(
    weights: np.ndarray,
    mask: np.ndarray | None,
    frac: float,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Time-reverse a random ``frac`` of the contrasts, so the shifted side varies.

    Reversing w turns ``step_head`` at split p into its mirror: the shifted side
    becomes the *tail*, at split W − p, with the amplitude sign unchanged. Both
    halves therefore lift a short end of the window, which is the regime that
    keeps the texture (ESA_RUN_2026-10-06 §9.3) -- it is only the long side that
    the nominal score refuses to hold. Without this every level shift puts its
    step on the same side, which an augmented detector would learn.
    """
    w = np.array(weights, dtype=np.float64)
    rows = rng.random(len(w)) < float(frac)
    w[rows] = w[rows][:, ::-1]
    m = None if mask is None else np.array(mask)
    if m is not None:
        m[rows] = m[rows][:, ::-1]
    return w, m, rows


def assign_donors_by_room(
    alloc: np.ndarray,
    level: np.ndarray,
    channel: np.ndarray,
    wanted: dict[str, float],
) -> np.ndarray:
    """Permute kind labels within each channel so each kind gets a parent with room.

    ``fit_to_headroom`` repairs an amplitude that the parent cannot hold; this
    removes the need for the repair by handing the amplitude a parent that can.
    Mission 1 needs it: every real level shift drops from the top of the channel
    span (pre-level 0.835-0.920) to the bottom, while the nominal donors are at
    0.099 median / 0.123 p95 -- a uniform draw therefore gives the level-shift
    slice a parent with 0.03 of downward room, although parents at 0.79-0.94 do
    exist in the tail (ESA_RUN_2026-10-06 §4a).

    ``wanted[kind]`` is the kind's typical signed amplitude; kinds wanting the
    most negative shift are dealt the highest parents. Per-channel counts per
    kind are untouched, so the channel stratification of the allocation and
    every downstream channel-balanced metric are unaffected.
    """
    out = np.asarray(alloc).astype(str).copy()
    lv = np.asarray(level, dtype=np.float64)
    if len(out) != len(lv) or len(out) != len(channel):
        raise ValueError(f"shapes {out.shape}, {lv.shape}, {np.shape(channel)} do not line up")
    for c in np.unique(np.asarray(channel)):
        slots = np.flatnonzero(np.asarray(channel) == c)
        here = out[slots]
        # most negative amplitude first, so it is dealt the highest parents;
        # kinds with no contrast take what is left, in their original order
        order = sorted({k for k in here if k in wanted}, key=lambda k: wanted[k])
        by_level = slots[np.argsort(-lv[slots], kind="stable")]
        taken = 0
        for k in order:
            n = int((here == k).sum())
            out[by_level[taken : taken + n]] = k
            taken += n
        rest = [k for k in here if k not in wanted]
        out[by_level[taken:]] = rest
    return out


def fit_to_headroom(
    delta: np.ndarray,
    parent_span: np.ndarray,
    weights: np.ndarray,
    *,
    rails: tuple[float, float] = (0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Flip a sampled amplitude into the direction where the parent has room.

    A target of δ moves the positive support of ``weights`` by about δ, so the
    sign decides which rail the planted anomaly runs into. Mission-1's nominal
    donors sit a median of 0.033 above their channel floor and 0.841 below the
    ceiling, while five of the six real level-shift amplitudes are negative:
    resampling the sign blind puts almost every generated level shift outside
    the training envelope by arithmetic (ESA_RUN_2026-10-06 §4a), and real
    level-shift windows leave it 1 in 6 times.

    The sign is flipped only when the sampled direction cannot hold the sampled
    size and the other direction holds more of it, so a donor pool with room on
    both sides keeps the measured sign distribution. The magnitude is never
    touched — shrinking excursions is the hashfix failure.

    ``parent_span`` and ``rails`` are in channel-span units (the S0 training
    envelope is [0, 1]); ``delta`` is in the denoiser's units and only has its
    sign changed, so the two need no common scale. Returns the adjusted delta
    and the bool mask of rows that were flipped.
    """
    d = np.asarray(delta, dtype=np.float64).reshape(-1)
    u = np.asarray(parent_span, dtype=np.float64)
    seg = np.asarray(weights) > 0
    if u.shape != seg.shape or len(d) != len(u):
        raise ValueError(f"shapes {d.shape}, {u.shape}, {seg.shape} do not line up")
    if not seg.any(axis=1).all():
        raise ValueError("every contrast needs a positive support")
    up = rails[1] - np.where(seg, u, -np.inf).max(axis=1)
    down = np.where(seg, u, np.inf).min(axis=1) - rails[0]
    room_now = np.where(d >= 0, up, down)
    room_other = np.where(d >= 0, down, up)
    flip = (np.abs(d) > room_now) & (room_other > room_now)
    return np.where(flip, -d, d), flip


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
    step_head
           [0, p + dilate]: the mirror, for a shifted head;
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
    if kind == "step_head":
        return idx <= np.minimum(pos + int(dilate), int(width) - 1)[:, None]
    if kind == "spike":
        r = int(spike_half) + int(flank_gap)
        return (idx >= (pos - r)[:, None]) & (idx <= (pos + r)[:, None])
    raise ValueError(f"unknown contrast kind {kind!r} (use {CONTRAST_KINDS})")
