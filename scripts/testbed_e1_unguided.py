"""E1: does a causal denoiser alone create start-of-window artifacts?

Unguided DDIM with exact posterior-mean denoisers (no steering).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from anogen.testbed.gauss import GPChannel, Schedule, LinearDenoiser, guided_ddim
from anogen.testbed.metrics import gallery_report, edge_stats

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed")
OUT.mkdir(parents=True, exist_ok=True)
W, N = 512, 512
B = int(sys.argv[2]) if len(sys.argv) > 2 else 128
ch, sch = GPChannel(), Schedule()
rng = np.random.default_rng(0)
train = ch.sample(4096, W, rng)
lo, hi = float(train.min()), float(train.max())
long = ch.sample(N, W + B, rng)          # donors with real preceding context
donor, donor_long = long[:, B:], long
ref = ch.sample(N, W, rng)               # held-out truth for marginal checks

# x0_hat error profile at the edit time t*=40 (nu=0.2)
prof = {}
for kind in ("bidir", "causal", "anticausal", "flip_avg", "flip_ramp"):
    den = LinearDenoiser(ch, W, sch, kind)
    t = 40; ab = sch.alpha_bar[t]
    xt = np.sqrt(ab) * ref + np.sqrt(1 - ab) * rng.standard_normal(ref.shape)
    err = ((den.x0_hat(xt, t) - ref) ** 2).mean(axis=0)
    prof[kind] = {"mse_first16": float(err[:16].mean()), "mse_mid": float(err[248:264].mean()),
                  "mse_last16": float(err[-16:].mean()), "mse_bin0": float(err[0])}
print(json.dumps(prof, indent=1))

rows = {}
for nu, steps in ((1.0, 20), (0.2, 50)):
    for kind in ("bidir", "causal", "flip_avg", "flip_ramp", "causal+burnin"):
        t0 = time.time()
        r = np.random.default_rng(1)
        if kind == "causal+burnin":
            den = LinearDenoiser(ch, W + B, sch, "causal")
            x = guided_ddim(den, donor_long, [], rng=r, nu=nu, ddim_steps=steps, crop=B)
        else:
            den = LinearDenoiser(ch, W, sch, kind)
            x = guided_ddim(den, donor, [], rng=r, nu=nu, ddim_steps=steps)
        rep = {**gallery_report(x, lo, hi), **edge_stats(x, ref)}
        rows[f"nu{nu}_{kind}"] = rep
        print(f"nu={nu} {kind:14s} peak@start={rep['peak_at_start']:.3f} peak@end={rep['peak_at_end']:.3f} "
              f"std s/m/e={rep['std_ratio_start']:.2f}/{rep['std_ratio_mid']:.2f}/{rep['std_ratio_end']:.2f} "
              f"dstd s/m/e={rep['dstd_ratio_start']:.2f}/{rep['dstd_ratio_mid']:.2f}/{rep['dstd_ratio_end']:.2f} "
              f"env={rep['env_window_frac']:.3f} ({time.time()-t0:.0f}s)", flush=True)
real = {**gallery_report(ref, lo, hi), **edge_stats(ref, ch.sample(N, W, rng))}
rows["real_nominal"] = real
print("real", {k: round(v, 3) for k, v in real.items()})
(OUT / f"e1_unguided_B{B}.json").write_text(json.dumps({"x0hat_error_t40": prof, "galleries": rows}, indent=1))
