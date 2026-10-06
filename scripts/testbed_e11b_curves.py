"""E11b: per-bin first-difference variance (generated / true) across the window, nu = 0.2 edit, closed form.

Curves for the exact causal, bidirectional and flip-ramp denoisers on the frozen
linear schedule, and the bidirectional one on the log-spaced schedule at the
same noise level. Saved every 4 bins for plotting.
"""
import json, sys
from pathlib import Path
import numpy as np
from anogen.testbed.gauss import GeometricSchedule, GPChannel, LinearDenoiser, Schedule

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed")
W = 512
ch = GPChannel(); S = ch.cov(W); I = np.eye(W); D = np.diff(I, axis=0)
lin, geo = Schedule(), GeometricSchedule()

def curve(den, sch, t0, steps=50):
    times = np.linspace(t0, 0, steps, dtype=int); ab_arr = sch.alpha_bar; M = I.copy()
    for i, t in enumerate(times):
        ab, A = ab_arr[t], den.matrix(int(t))
        if i + 1 == len(times):
            M = A @ M; break
        abp = ab_arr[times[i + 1]]
        M = (np.sqrt(abp) * A + np.sqrt(1 - abp) * (I - np.sqrt(ab) * A) / np.sqrt(1 - ab)) @ M
    ab0 = ab_arr[t0]
    C = M @ (ab0 * S + (1 - ab0) * I) @ M.T
    return np.diag(D @ C @ D.T) / np.diag(D @ S @ D.T)

t_lin = 40
t_geo = int(np.argmin(np.abs(geo.alpha_bar - lin.alpha_bar[t_lin])))
res = {"bins": list(range(0, W - 1, 4)), "t_geo": t_geo}
for name, kind, sch, t0 in (("causal, linear schedule", "causal", lin, t_lin), ("bidirectional, linear schedule", "bidir", lin, t_lin),
                            ("flip ramp, linear schedule", "flip_ramp", lin, t_lin), ("bidirectional, log-spaced schedule", "bidir", geo, t_geo)):
    c = curve(LinearDenoiser(ch, W, sch, kind), sch, t0)
    res[name] = c[::4].round(4).tolist()
    print(name, "bin0 %.2f first16 %.2f mid %.2f last16 %.2f" % (c[0], c[:16].mean(), c[240:272].mean(), c[-16:].mean()), flush=True)
(OUT / "e11b_curves.json").write_text(json.dumps(res, indent=1))
