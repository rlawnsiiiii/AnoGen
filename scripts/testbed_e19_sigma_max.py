"""E19: how large must σ_max be for a from-noise start? Closed form, exact denoiser.

Deterministic DDIM with the exact (affine) posterior-mean denoiser maps
x_T ~ N(0, I) affinely to x_0, so the generated mean and covariance are exact.
We report, for geometric schedules with increasing σ_max (and the frozen linear
one), the level bias (generated mean − true mean, in within-window sd) and the
variance / texture ratios, for the min/max-style data (mean 0.5, as the repo
trains) and for centred data (mean removed, as a z-score scaler would give).
``texture.schedule_bounds`` predicts the σ_max needed from the spectrum.
"""
import json, sys
from pathlib import Path
import numpy as np
from anogen.shell.texture import schedule_bounds
from anogen.testbed.gauss import GeometricSchedule, GPChannel, LinearDenoiser, Schedule

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed")
W, STEPS = 256, 50
I, D = np.eye(W), np.diff(np.eye(W), axis=0)


def from_noise(ch, sch):
    den = LinearDenoiser(ch, W, sch, "bidir")
    ab_arr = sch.alpha_bar
    times = np.linspace(len(ab_arr) - 1, 0, STEPS, dtype=int)
    M, q = I.copy(), np.zeros(W)
    mu = ch.mean * np.ones(W)
    for i, t in enumerate(times):
        ab, A = ab_arr[t], den.matrix(int(t))
        # x0h = mu + A (x - sqrt(ab) mu) = A x + (mu - sqrt(ab) A mu)
        Ax, cx = A, mu - np.sqrt(ab) * A @ mu
        if i + 1 == len(times):
            M, q = Ax @ M, Ax @ q + cx
            break
        abp = ab_arr[times[i + 1]]
        k = np.sqrt((1 - abp) / (1 - ab))
        P = np.sqrt(abp) * Ax + k * (I - np.sqrt(ab) * Ax)
        c = np.sqrt(abp) * cx - k * np.sqrt(ab) * cx
        M, q = P @ M, P @ q + c
    S = ch.cov(W)
    C = M @ M.T
    sd = np.sqrt(np.mean(np.diag(S)))
    return {"level_bias_in_sd": float(np.mean(q - mu) / sd), "var_ratio": float(np.trace(C) / np.trace(S)),
            "diff_var_ratio": float(np.trace(D @ C @ D.T) / np.trace(D @ S @ D.T))}


res = {}
for label, mean in (("min/max-style data (mean 0.5)", 0.5), ("centred data (mean 0)", 0.0)):
    ch = GPChannel(mean=mean)
    x = ch.sample(2000, W, np.random.default_rng(0))
    b = schedule_bounds(x, centered=(mean == 0.0))
    res[label] = {"predicted": b}
    print(f"{label}: predicted sigma_max {b['sigma_max']:.1f} (without DC {b['sigma_max_without_dc']:.1f}), sigma_min {b['sigma_min']:.2g}", flush=True)
    for name, sch in [("linear (frozen, sigma_max 2.56)", Schedule())] + [
        (f"geometric sigma_max {s:g}", GeometricSchedule(sigma_min=1e-3, sigma_max=s)) for s in (2.56, 10.0, 30.0, 100.0)
    ]:
        r = from_noise(ch, sch)
        res[label][name] = r
        print(f"  {name:34s} level bias {r['level_bias_in_sd']:+.2f} sd  var {r['var_ratio']:.2f}  texture {r['diff_var_ratio']:.2f}", flush=True)
(OUT / "e19_sigma_max.json").write_text(json.dumps(res, indent=1))
