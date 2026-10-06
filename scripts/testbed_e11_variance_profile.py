"""E11: where does a causal denoiser put the error? Closed-form per-bin profile.

Deterministic DDIM with an exact linear denoiser is a linear map x_0 = M x_start,
so the generated covariance is C = M Cov(x_start) M^T in closed form. For each
denoiser we report, per bin i,

  var ratio   C[i,i] / Sigma[i,i]                (amplitude the sampler produces)
  diff ratio  Var(x[i+1]-x[i]) generated / true  (texture)
  bias in E[x0] is zero for all of them (zero-mean linear maps), so any
  "dominant excursion at the start" must come from excess variance there.

and also the expected squared error of the edit against its own donor for the
nu=0.2 edit (how far the regenerated window moves from the parent, per bin).
"""
import json
import sys
from pathlib import Path

import numpy as np

from anogen.testbed.gauss import GPChannel, LinearDenoiser, Schedule

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed")
W = 512
ch, sch = GPChannel(), Schedule()
S = ch.cov(W)
I = np.eye(W)
D = np.diff(I, axis=0)


def ddim_map(den: LinearDenoiser, t_start: int, steps: int) -> np.ndarray:
    times = np.linspace(t_start, 0, steps, dtype=int)
    M = I.copy()
    ab_arr = sch.alpha_bar
    for i, t in enumerate(times):
        ab, A = ab_arr[t], den.matrix(int(t))
        if i + 1 == len(times):
            return A @ M
        abp = ab_arr[times[i + 1]]
        step = np.sqrt(abp) * A + np.sqrt(1 - abp) * (I - np.sqrt(ab) * A) / np.sqrt(1 - ab)
        M = step @ M
    return M


def profile(v: np.ndarray) -> dict[str, float]:
    n = len(v)
    return {"first16": float(v[:16].mean()), "first10pct": float(v[: n // 10].mean()),
            "middle": float(v[n // 2 - 16 : n // 2 + 16].mean()),
            "last10pct": float(v[-(n // 10):].mean()), "last16": float(v[-16:].mean())}


res = {}
for kind in ("bidir", "causal", "flip_ramp"):
    den = LinearDenoiser(ch, W, sch, kind)
    for nu, steps in ((1.0, 20), (0.2, 50)):
        t0 = int(round(nu * (sch.n_times - 1)))
        ab0 = sch.alpha_bar[t0]
        # x_start = sqrt(ab0) x0 + sqrt(1-ab0) e, x0 ~ Sigma  (a donor edit)
        M = ddim_map(den, t0, steps)
        Cs = ab0 * S + (1 - ab0) * I
        C = M @ Cs @ M.T
        var_ratio = np.diag(C) / np.diag(S)
        diff_ratio = np.diag(D @ C @ D.T) / np.diag(D @ S @ D.T)
        # E||x_gen - x0||^2 per bin: x_gen = M(sqrt(ab0) x0 + sqrt(1-ab0) e)
        A0 = np.sqrt(ab0) * M - I
        dev = np.diag(A0 @ S @ A0.T + (1 - ab0) * M @ M.T) / np.diag(S)
        key = f"{kind} nu={nu}"
        res[key] = {"var_ratio": profile(var_ratio), "diff_var_ratio": profile(diff_ratio),
                    "deviation_from_donor": profile(dev),
                    "var_ratio_curve": var_ratio[::8].round(4).tolist()}
        print(f"{key:18s} var ratio first16 {res[key]['var_ratio']['first16']:.2f} mid {res[key]['var_ratio']['middle']:.2f} "
              f"last16 {res[key]['var_ratio']['last16']:.2f} | diff ratio first16 {res[key]['diff_var_ratio']['first16']:.2f} "
              f"mid {res[key]['diff_var_ratio']['middle']:.2f} last16 {res[key]['diff_var_ratio']['last16']:.2f} | "
              f"dev first16 {res[key]['deviation_from_donor']['first16']:.2f} mid {res[key]['deviation_from_donor']['middle']:.2f}", flush=True)
(OUT / "e11_variance_profile.json").write_text(json.dumps(res, indent=1))
