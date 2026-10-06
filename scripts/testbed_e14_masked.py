"""E14: anomalies as masked inpainting (RePaint-style) instead of whole-window edits.

Each sample gets an explicit anomaly mask at a uniformly drawn position. Bins
outside the mask are replaced at every step by the donor noised to the current
level (RePaint), so the context is the real donor; only the masked bins are
generated, and the anomaly objective acts on them only (here a level shift
via the exact step-contrast projection of shell/contrast.py: mean of the
masked tail minus mean of the 64 bins before it = delta).

Compared with the whole-window step contrast of E6 (fixed sampler), on:
position error against the *requested* position (a mask is a label), amplitude,
seam sharpness (largest first difference within +-4 bins of the onset, real
level shifts ramp over 6 bins), texture inside the generated region, and how
much of the donor outside the mask is changed.
"""
import json
import sys
from pathlib import Path

import numpy as np

from anogen.testbed.gauss import GPChannel, LinearDenoiser, Schedule, level_shift
from anogen.testbed.metrics import gallery_report

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed")
W, N, CTX = 512, 256, 64
ch, sch = GPChannel(), Schedule()
rng = np.random.default_rng(0)
train = ch.sample(4096, W, rng)
lo, hi = float(train.min()), float(train.max())
donor = ch.sample(N, W, rng)
pos = rng.integers(80, 460, N)
delta = rng.choice([-1, 1], N) * rng.uniform(0.2, 0.3, N)
real = level_shift(donor, pos, delta)  # same donors: an ideal injected shift
idx = np.arange(W)[None, :]
tail = idx >= pos[:, None]                                  # anomaly = the shifted tail
ctx = (idx < pos[:, None]) & (idx >= pos[:, None] - CTX)    # reference segment before it
w_full = tail / tail.sum(1, keepdims=True) - ctx / ctx.sum(1, keepdims=True)
mask = tail


def contrast_edit(x0h: np.ndarray, only_mask: bool) -> np.ndarray:
    w_apply = np.where(mask, w_full, 0.0) if only_mask else w_full
    d = (x0h * w_full).sum(1) - delta
    return (d / (w_apply * w_apply).sum(1))[:, None] * w_apply


def run(den, *, nu: float, steps: int, masked: bool, resample: int = 1, seed: int = 1, dilate: int = 0) -> np.ndarray:
    global mask
    mask = idx >= (pos[:, None] - dilate)  # generated bins (dilated before the onset)
    r = np.random.default_rng(seed)
    t0 = int(round(nu * (sch.n_times - 1)))
    ab0 = sch.alpha_bar[t0]
    xt = np.sqrt(ab0) * donor + np.sqrt(1 - ab0) * r.standard_normal(donor.shape)
    times = np.linspace(t0, 0, steps, dtype=int)
    for i, t in enumerate(times):
        t = int(t)
        last = i + 1 == len(times)
        for k in range(1 if last else resample):
            ab = float(sch.alpha_bar[t])
            if masked:  # RePaint: known bins = donor at this noise level
                known = np.sqrt(ab) * donor + np.sqrt(1 - ab) * r.standard_normal(donor.shape)
                xt = np.where(mask, xt, known)
            x0h = den.x0_hat(xt, t)
            eps = (xt - np.sqrt(ab) * x0h) / np.sqrt(1 - ab)
            x0e = x0h - contrast_edit(x0h, masked)
            if last:
                return np.where(mask, x0e, donor) if masked else x0e
            abp = float(sch.alpha_bar[int(times[i + 1])])
            x_next = np.sqrt(abp) * x0e + np.sqrt(1 - abp) * eps
            if k + 1 < resample and abp > ab:
                a = ab / abp
                xt = np.sqrt(a) * x_next + np.sqrt(1 - a) * r.standard_normal(xt.shape)
            else:
                xt = x_next
    return xt


def local_contrast(x: np.ndarray, at: np.ndarray, half: int = 64) -> np.ndarray:
    return np.array([x[i, a: a + half].mean() - x[i, max(a - half, 0): a].mean() for i, a in enumerate(at)])


def report(x: np.ndarray) -> dict:
    amp = local_contrast(x, pos)
    # localization: best +-32-bin split within +-96 bins of the requested onset
    loc = []
    for i in range(len(x)):
        cand = np.arange(max(pos[i] - 96, 32), min(pos[i] + 96, W - 32))
        v = np.array([x[i, c: c + 32].mean() - x[i, c - 32: c].mean() for c in cand]) * np.sign(delta[i])
        loc.append(cand[v.argmax()] - pos[i])
    loc = np.array(loc)
    d = np.abs(np.diff(x, axis=1))
    seam = np.array([d[i, max(pos[i] - 5, 0): pos[i] + 4].max() for i in range(len(x))])
    inside = np.array([np.diff(x[i, pos[i] + 8:]).std() for i in range(len(x))])
    ref_inside = np.array([np.diff(donor[i, pos[i] + 8:]).std() for i in range(len(x))])
    outside = np.array([np.abs(x[i, : pos[i] - CTX] - donor[i, : pos[i] - CTX]).mean() if pos[i] > CTX else 0.0 for i in range(len(x))])
    g = gallery_report(x, lo, hi)
    return {"onset_abs_err_median": float(np.median(np.abs(loc))),
            "amp_at_requested_over_target": float(np.median(amp / delta)),
            "seam_max_diff_median": float(np.median(seam)),
            "texture_inside_ratio": float(np.mean(inside / ref_inside)),
            "outside_change_mean": float(outside.mean()),
            "diff_p999": g["diff_p999"], "peak_at_start": g["peak_at_start"], "peak_at_end": g["peak_at_end"]}


res = {"real level shifts (ramp 6 bins)": report(real)}
den = LinearDenoiser(ch, W, sch, "bidir")
for label, kw in (("whole-window contrast, nu 0.2 (E6 style)", dict(nu=0.2, steps=50, masked=False)),
                  ("masked, nu 0.2", dict(nu=0.2, steps=50, masked=True)),
                  ("masked, nu 0.2, resample 3", dict(nu=0.2, steps=50, masked=True, resample=3)),
                  ("masked + 8-bin dilation, nu 0.2", dict(nu=0.2, steps=50, masked=True, dilate=8)),
                  ("masked + 8-bin dilation, nu 0.5, resample 3", dict(nu=0.5, steps=50, masked=True, dilate=8, resample=3)),
                  ("masked, nu 0.5", dict(nu=0.5, steps=50, masked=True)),
                  ("masked, nu 0.5, resample 3", dict(nu=0.5, steps=50, masked=True, resample=3)),
                  ("masked, from t_max", dict(nu=1.0, steps=50, masked=True)),
                  ("masked, from t_max, resample 3", dict(nu=1.0, steps=50, masked=True, resample=3))):
    res[label] = report(run(den, **kw))
for k, r in res.items():
    print(f"{k:46s} onset err {r['onset_abs_err_median']:5.1f} bins | amp/target {r['amp_at_requested_over_target']:.2f} | seam {r['seam_max_diff_median']:.3f} | "
          f"texture in {r['texture_inside_ratio']:.2f} | outside change {r['outside_change_mean']:.3f} | start {r['peak_at_start']:.2f} end {r['peak_at_end']:.2f}", flush=True)
(OUT / "e14_masked.json").write_text(json.dumps(res, indent=1))
