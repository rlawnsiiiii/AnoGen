"""E6: level shifts from a parametric step contrast instead of 6 shared prototypes."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance
from anogen.testbed.gauss import (GPChannel, Schedule, LinearDenoiser, BlockEncoder, Shell,
                                  GuideTerm, guided_ddim, proto_energy, level_shift)
from anogen.testbed.metrics import gallery_report
from anogen.shell.contrast import measure_contrast, sample_targets
from anogen.shell.realism import diag_cusum, diag_persistence

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed"); OUT.mkdir(parents=True, exist_ok=True)
W, N = 512, 384
ch, sch = GPChannel(), Schedule()
rng = np.random.default_rng(0)
train = ch.sample(4096, W, rng); lo, hi = float(train.min()), float(train.max())
enc = BlockEncoder(W, 8)
shell = Shell.from_nominal(enc, ch.sample(512, W, rng), ch.sample(512, W, rng))
donor = ch.sample(N, W, rng)
ls_pos = np.array([90, 150, 210, 270, 330, 400]); size = 0.25
protos_x = level_shift(ch.sample(6, W, rng), ls_pos, np.full(6, size))
protos = enc.encode(protos_x); pidx = np.random.default_rng(17).integers(0, 6, N)
real = level_shift(ch.sample(N, W, rng), rng.integers(40, 470, N), np.full(N, size))
real_amp, real_pos = measure_contrast("step", real)
few_amp, _ = measure_contrast("step", protos_x)           # the 6 "labelled" examples
pos, delta, wts = sample_targets("step", N, W, few_amp, rng=np.random.default_rng(3))

def contrast_fn(x):
    """Projection edit (d - delta) w / |w|^2, as steer.guided_ddim applies it."""
    d = (x * wts).sum(axis=1)
    return (d - delta) ** 2, ((d - delta) / (wts * wts).sum(axis=1))[:, None] * wts

def report(x):
    amp, p = measure_contrast("step", x)
    u = (x - lo) / (hi - lo)
    r = gallery_report(x, lo, hi)
    r.update({"step_amp_w1_to_real": float(wasserstein_distance(amp, real_amp)),
              "step_amp_mean": float(np.mean(amp)), "step_pos_std": float(np.std(p)),
              "step_pos_w1_to_real": float(wasserstein_distance(p, real_pos)),
              "shelf_rate": float(np.mean(np.abs(diag_persistence(u)) > 0.5 * size / (hi - lo)))})
    return r

rows = {"real level shifts": report(real)}
samplers = {"repo (causal, x, final 1)": dict(kind="causal", space="x", final=1.0),
            "fixed (flip_ramp, x0, final 0.25)": dict(kind="flip_ramp", space="x0", final=0.25)}
for sname, sk in samplers.items():
    den = LinearDenoiser(ch, W, sch, sk["kind"])
    for tname, extra in (("proto (6 shared)", [GuideTerm(1.0, lambda x: proto_energy(enc, x, protos[pidx]))]),
                         ("step contrast", [GuideTerm(1.0, contrast_fn, project=True)]),
                         ("proto + step contrast", [GuideTerm(1.0, lambda x: proto_energy(enc, x, protos[pidx])), GuideTerm(1.0, contrast_fn, project=True)])):
        terms = [GuideTerm(0.3, lambda x: shell.band(x))] + extra
        x = guided_ddim(den, donor, terms, rng=np.random.default_rng(1), space=sk["space"], final_scale=sk["final"])
        rows[f"{sname} | {tname}"] = report(x)
print("| sampler | target | shelf rate | step amp mean (real %.3f) | amp W1 | step pos sd (real %.3f) | pos W1 | diff p99.9 | start | end |" % (np.mean(real_amp), np.std(real_pos)))
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for k, r in rows.items():
    s, t = (k.split(" | ") + [""])[:2]
    print(f"| {s} | {t} | {r['shelf_rate']:.2f} | {r['step_amp_mean']:.3f} | {r['step_amp_w1_to_real']:.3f} | {r['step_pos_std']:.3f} | "
          f"{r['step_pos_w1_to_real']:.3f} | {r['diff_p999']:.3f} | {r['peak_at_start']:.2f} | {r['peak_at_end']:.2f} |")
(OUT / "e6_contrast.json").write_text(json.dumps(rows, indent=1))
