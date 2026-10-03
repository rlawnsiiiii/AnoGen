"""E10b: does a *trained* net on the geometric schedule keep telemetry texture?

E10 showed with exact denoisers that the frozen linear schedule (sigma_min 0.01)
reproduces only 36-68 % of the first-difference variance and that log-spaced
sigma down to 1e-3 keeps 89-98 %. Here the same comparison runs with the two
trained bidirectional numpy nets (E8a): ``bidir`` (linear) and ``bidir_geo``.

Edits are matched by *noise level*, not by step index: nu=0.2 on the linear
schedule is t=40; the geometric run starts at the step with the closest
alpha_bar. Every run uses 32 bins of real context on both sides (E8c).

Reported per run: variance ratio, first-difference variance ratio, high-band
(upper half of rFFT bins) log-power shift, peak@start / peak@end.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from anogen.testbed.gauss import GeometricSchedule, GPChannel, Schedule, guided_ddim
from anogen.testbed.metrics import gallery_report
from anogen.testbed.npnet import NetDenoiser

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed")
W, B, N = 512, 32, 192
ch = GPChannel()
rng = np.random.default_rng(0)
train = ch.sample(4096, W, rng)
lo, hi = float(train.min()), float(train.max())
lin, geo = Schedule(), GeometricSchedule()
nets = {k: pickle.load(open(OUT / "nets" / f"{k}.pkl", "rb"))["net"] for k in ("bidir", "bidir_geo")}


def texture(x: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    d, dr = np.diff(x, axis=1), np.diff(ref, axis=1)
    p = np.abs(np.fft.rfft(x - x.mean(axis=1, keepdims=True), axis=1)) ** 2
    pr = np.abs(np.fft.rfft(ref - ref.mean(axis=1, keepdims=True), axis=1)) ** 2
    hb = slice(p.shape[1] // 2, None)
    return {
        "var_ratio": float(x.var(axis=1).mean() / ref.var(axis=1).mean()),
        "diff_var_ratio": float(d.var(axis=1).mean() / dr.var(axis=1).mean()),
        "hf_logpower_shift": float(np.log10(p[:, hb].mean() / pr[:, hb].mean())),
    }


def matched_nu(sched, t_lin: int) -> float:
    t = int(np.argmin(np.abs(sched.alpha_bar - lin.alpha_bar[t_lin])))
    return t / (len(sched.alpha_bar) - 1)


long = ch.sample(N, W + 2 * B, rng)
ref = long[:, B:-B]
res = {}
for name, sched in (("bidir net, linear schedule", lin), ("bidir net, geometric schedule", geo)):
    net = nets["bidir" if sched is lin else "bidir_geo"]
    den = NetDenoiser(net, sched, W + 2 * B, ch.mean)
    runs = [("from noise, 20 steps", dict(start_from_noise=True, ddim_steps=20)),
            ("from noise, 50 steps", dict(start_from_noise=True, ddim_steps=50)),
            ("edit at nu=0.2 noise level, 50 steps", dict(nu=matched_nu(sched, 40), ddim_steps=50))]
    for label, kw in runs:
        x = guided_ddim(den, long, [], rng=np.random.default_rng(1), crop=B, crop_end=B, **kw)
        r = {**texture(x, ref), **{k: gallery_report(x, lo, hi)[k] for k in ("peak_at_start", "peak_at_end", "diff_p999", "env_window_frac")}}
        r["ref_diff_p999"] = gallery_report(ref, lo, hi)["diff_p999"]
        res[f"{name} | {label}"] = r
        print(f"{name:32s} | {label:38s} var {r['var_ratio']:.2f} diff-var {r['diff_var_ratio']:.2f} "
              f"HF shift {r['hf_logpower_shift']:+.2f} start {r['peak_at_start']:.2f} end {r['peak_at_end']:.2f}", flush=True)
        (OUT / "e10b_trained_geo.json").write_text(json.dumps(res, indent=1))
