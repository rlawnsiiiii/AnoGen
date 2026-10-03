"""E10: how much telemetry texture can the noise schedule reproduce, with *exact* denoisers?

Deterministic DDIM with an exact Gaussian denoiser is a linear map x_0 = M x_T,
so the generated covariance is M Cov(x_T) M^T in closed form. We report the
generated / true ratio of the total variance and of the first-difference
variance (texture), for the frozen linear schedule and for log-spaced noise.
"""
import json
from pathlib import Path
import numpy as np
from anogen.testbed.gauss import GPChannel, LinearDenoiser, Schedule

n = 128
I, D = np.eye(n), np.diff(np.eye(n), axis=0)


def ratios(ch, ab_arr, steps):
    class _S:
        alpha_bar = ab_arr
    den = LinearDenoiser(ch, n, Schedule(), "bidir")
    den.schedule = _S()
    S = ch.cov(n)
    T = len(ab_arr)
    times = np.unique(np.linspace(T - 1, 0, steps).astype(int))[::-1]
    M = I.copy()
    for i, t in enumerate(times):
        ab, A = ab_arr[t], den.matrix(t)
        if i + 1 == len(times):
            M = A @ M
            break
        abp = ab_arr[times[i + 1]]
        M = (np.sqrt(abp) * A + np.sqrt(1 - abp) * (I - np.sqrt(ab) * A) / np.sqrt(1 - ab)) @ M
    C = M @ (ab_arr[-1] * S + (1 - ab_arr[-1]) * I) @ M.T
    return float(np.trace(C) / np.trace(S)), float(np.trace(D @ C @ D.T) / np.trace(D @ S @ D.T))


res = {}
schedules = {"linear (frozen; sigma 0.010..2.56)": Schedule().alpha_bar}
for smin, smax in ((1e-3, 2.56), (1e-3, 10.0), (1e-3, 80.0)):
    sig = np.geomspace(smin, smax, 200)
    schedules[f"geometric sigma {smin:g}..{smax:g}"] = 1 / (1 + sig**2)
for texture in (0.02, 0.05):
    ch = GPChannel(ar_sigma=texture)
    for name, ab in schedules.items():
        for steps in (20, 50, 200):
            v, d = ratios(ch, ab, steps)
            res[f"texture sd {texture} | {name} | {steps} steps"] = {"var_ratio": v, "diff_var_ratio": d}
            print(f"texture sd {texture} | {name:36s} | {steps:3d} steps: variance {v:.2f}, first-difference variance {d:.2f}", flush=True)
Path("results/testbed/e10_schedule.json").write_text(json.dumps(res, indent=1))
