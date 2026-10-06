"""E13: can x̂₀-space guidance be made smooth with trained nets? Self-recurrence vs edit smoothing.

E8 found that with trained nets x̂₀-space guidance makes edges *sharper*
(diff p99.9 0.083 -> 0.128, real 0.050): the raw encoder gradient is applied
to x̂₀ with nothing to harmonize it. Two standard remedies:

* self-recurrence / time travel (Bansal et al. 2023; Yu et al. 2023; TFG):
  after each guided step, re-noise x_{t'} back to x_t with fresh noise and
  redo the step, k times in total, so the denoiser re-projects the edit;
* smoothing the edit (a Gaussian low-pass on the x̂₀ edit), which mimics the
  low-pass the x-space VJP applied without its causal smear.

Level-shift slice as in E2/E8: shell band 0.3 + prototype 1.0, nu = 0.2.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from anogen.testbed.gauss import BlockEncoder, GPChannel, Schedule, Shell, _unit, level_shift, proto_energy
from anogen.testbed.metrics import gallery_report
from anogen.testbed.npnet import NetDenoiser

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed")
W, N = 512, 128
ch, sch = GPChannel(), Schedule()
rng = np.random.default_rng(0)
train = ch.sample(4096, W, rng)
lo, hi = float(train.min()), float(train.max())
nets = {k: pickle.load(open(OUT / "nets" / f"{k}.pkl", "rb"))["net"] for k in ("causal", "bidir")}


class FlipNet(NetDenoiser):
    def x0_hat(self, xt, t):
        ab = float(self.schedule.alpha_bar[int(t)])
        z = xt - np.sqrt(ab) * self.mean
        tt = np.full(len(xt), int(t))
        e_f = self.net.forward(z, tt)
        e_b = self.net.forward(z[:, ::-1].copy(), tt)[:, ::-1]
        w = np.linspace(0.0, 1.0, z.shape[1])[None, :]
        return (xt - np.sqrt(1.0 - ab) * (w * e_f + (1 - w) * e_b)) / np.sqrt(ab)


enc = BlockEncoder(W, 8)
shell = Shell.from_nominal(enc, ch.sample(512, W, rng), ch.sample(512, W, rng))
donor = ch.sample(N, W, rng)
protos = enc.encode(level_shift(ch.sample(6, W, rng), np.array([90, 150, 210, 270, 330, 400]), np.full(6, 0.25)))
pidx = np.random.default_rng(17).integers(0, 6, N)


def edit_of(x0h: np.ndarray, smooth: float) -> np.ndarray:
    g = 0.3 * _unit(shell.band(x0h)[1], 1.0, True) + 1.0 * _unit(proto_energy(enc, x0h, protos[pidx])[1], 1.0, True)
    return gaussian_filter1d(g, smooth, axis=1, mode="reflect") if smooth > 0 else g


def sample(den, *, recur: int = 1, smooth: float = 0.0, final: float = 0.25, nu: float = 0.2, steps: int = 50, seed: int = 1):
    r = np.random.default_rng(seed)
    t0 = int(round(nu * (sch.n_times - 1)))
    ab0 = sch.alpha_bar[t0]
    xt = np.sqrt(ab0) * donor + np.sqrt(1 - ab0) * r.standard_normal(donor.shape)
    times = np.linspace(t0, 0, steps, dtype=int)
    for i, t in enumerate(times):
        t = int(t)
        last = i + 1 == len(times)
        for k in range(1 if last else recur):
            ab = float(sch.alpha_bar[t])
            x0h = den.x0_hat(xt, t)
            eps = (xt - np.sqrt(ab) * x0h) / np.sqrt(1 - ab)
            e = edit_of(x0h, smooth)
            if last:
                return x0h - final * e
            abp = float(sch.alpha_bar[int(times[i + 1])])
            x_next = np.sqrt(abp) * (x0h - e) + np.sqrt(1 - abp) * eps
            if k + 1 < recur and abp > ab:  # time travel: back to t with fresh noise
                a = ab / abp
                xt = np.sqrt(a) * x_next + np.sqrt(1 - a) * r.standard_normal(xt.shape)
            else:
                xt = x_next
    return xt


dens = {"bidir net": NetDenoiser(nets["bidir"], sch, W, ch.mean),
        "causal net + flip ramp": FlipNet(nets["causal"], sch, W, ch.mean)}
real = level_shift(ch.sample(N, W, rng), rng.integers(40, 470, N), np.full(N, 0.25))
res = {"real level shifts": gallery_report(real, lo, hi)}
print(f"real: diff p99.9 {res['real level shifts']['diff_p999']:.3f}", flush=True)
for dname, den in dens.items():
    for label, kw in (("x0, final 0.25", {}), ("x0, final 0", dict(final=0.0)),
                      ("x0, recurrence k=2", dict(recur=2)), ("x0, recurrence k=4", dict(recur=4)),
                      ("x0, edit low-pass sd 4", dict(smooth=4.0)), ("x0, edit low-pass sd 8", dict(smooth=8.0)),
                      ("x0, recurrence k=2 + low-pass sd 4", dict(recur=2, smooth=4.0))):
        x = sample(den, **kw)
        r = gallery_report(x, lo, hi)
        r["proto_dist"] = float(np.linalg.norm(enc.encode(x) - protos[pidx], axis=1).mean())
        res[f"{dname} | {label}"] = r
        print(f"{dname:24s} | {label:36s} diff p99.9 {r['diff_p999']:.3f} start {r['peak_at_start']:.2f} "
              f"end {r['peak_at_end']:.2f} proto dist {r['proto_dist']:.2f}", flush=True)
        (OUT / "e13_recurrence.json").write_text(json.dumps(res, indent=1))
