"""E16: v-prediction vs ε-prediction for a trained net on the log-spaced schedule.

E10b found that a small ε-trained net on the geometric schedule (σ 1e-3..10)
under-produces amplitude from noise (variance 0.71–0.78 of the truth), while
on the ν = 0.2 edit both schedules were fine. v-prediction (Salimans & Ho
2022) is the usual remedy for wide, near-zero-terminal-SNR schedules. Same net
size, steps and data; 32 bins of real context on each side; matched noise
levels for the edit (as E10b).
"""
import json, pickle, sys
from pathlib import Path
import numpy as np
from anogen.testbed.gauss import GeometricSchedule, GPChannel, Schedule, guided_ddim
from anogen.testbed.metrics import gallery_report
from anogen.testbed.npnet import NetDenoiser

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed")
W, B, N = 512, 32, 192
ch = GPChannel()
rng = np.random.default_rng(0)
train = ch.sample(4096, W, rng); lo, hi = float(train.min()), float(train.max())
lin, geo = Schedule(), GeometricSchedule()
long = ch.sample(N, W + 2 * B, rng)
ref = long[:, B:-B]


def texture(x):
    d, dr = np.diff(x, axis=1), np.diff(ref, axis=1)
    return {"var_ratio": float(x.var(axis=1).mean() / ref.var(axis=1).mean()),
            "diff_var_ratio": float(d.var(axis=1).mean() / dr.var(axis=1).mean()),
            "level_bias_in_sd": float((x.mean() - ref.mean()) / ref.std(axis=1).mean())}


def matched_nu(sched, t_lin=40):
    t = int(np.argmin(np.abs(sched.alpha_bar - lin.alpha_bar[t_lin])))
    return t / (len(sched.alpha_bar) - 1)


res = {}
for name, key, sched, param in (("ε-net, linear schedule", "bidir", lin, "eps"),
                                ("ε-net, log-spaced schedule", "bidir_geo", geo, "eps"),
                                ("v-net, log-spaced schedule", "bidir_geo_v", geo, "v")):
    p = OUT / "nets" / f"{key}.pkl"
    if not p.is_file():
        continue
    den = NetDenoiser(pickle.load(open(p, "rb"))["net"], sched, W + 2 * B, ch.mean, param=param)
    for label, kw in (("from noise, 20 steps", dict(start_from_noise=True, ddim_steps=20)),
                      ("from noise, 50 steps", dict(start_from_noise=True, ddim_steps=50)),
                      ("edit at the ν = 0.2 noise level, 50 steps", dict(nu=matched_nu(sched), ddim_steps=50))):
        x = guided_ddim(den, long, [], rng=np.random.default_rng(1), crop=B, crop_end=B, **kw)
        r = {**texture(x), **{k: gallery_report(x, lo, hi)[k] for k in ("peak_at_start", "peak_at_end")}}
        res[f"{name} | {label}"] = r
        print(f"{name:28s} | {label:42s} var {r['var_ratio']:.2f} texture {r['diff_var_ratio']:.2f} "
              f"level {r['level_bias_in_sd']:+.2f} sd start {r['peak_at_start']:.2f} end {r['peak_at_end']:.2f}", flush=True)
        (OUT / "e16_vpred.json").write_text(json.dumps(res, indent=1))
