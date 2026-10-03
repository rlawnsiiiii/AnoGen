"""E3: band-only and spike-proto slices; Jacobian mass profile.

(a) Where does a uniform guidance gradient land after the denoiser VJP?
(b) Band-only (lam=0.3) and Point/Global needle slices under repo vs fixed samplers.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from anogen.testbed.gauss import (GPChannel, Schedule, LinearDenoiser, BlockEncoder, Shell,
                                  GuideTerm, guided_ddim, proto_energy, spike)
from anogen.testbed.metrics import gallery_report

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed"); OUT.mkdir(parents=True, exist_ok=True)
W, N = 512, 384
ch, sch = GPChannel(), Schedule()
rng = np.random.default_rng(0)
train = ch.sample(4096, W, rng); lo, hi = float(train.min()), float(train.max())
res = {"jacobian_mass_first10pct": {}}
# (a) VJP of a flat gradient and of a single-bin gradient at the window centre
for kind in ("bidir", "causal", "flip_ramp"):
    den = LinearDenoiser(ch, W, sch, kind)
    for t in (40, 20, 5):
        A = den.matrix(t)
        g_flat = np.ones((1, W)) @ A
        g_mid = np.zeros((1, W)); g_mid[0, W // 2] = 1.0; g_mid = g_mid @ A
        m = lambda g: float((g[0, : W // 10] ** 2).sum() / (g ** 2).sum())
        c = lambda g: float((np.abs(g[0]) * np.arange(W)).sum() / np.abs(g[0]).sum() / W)
        res["jacobian_mass_first10pct"][f"{kind}_t{t}"] = {"flat_first10": m(g_flat), "mid_centroid": c(g_mid), "mid_first10": m(g_mid)}
        print(kind, t, res["jacobian_mass_first10pct"][f"{kind}_t{t}"], flush=True)

enc = BlockEncoder(W, 8)
shell = Shell.from_nominal(enc, ch.sample(512, W, rng), ch.sample(512, W, rng))
donor = ch.sample(N, W, rng)
spk = spike(ch.sample(18, W, rng), rng.integers(30, 480, 18), rng.choice([-1, 1], 18) * rng.uniform(0.35, 0.6, 18))
protos = enc.encode(spk); pidx = np.random.default_rng(17).integers(0, 18, N)
real_spk = spike(ch.sample(N, W, rng), rng.integers(10, 500, N), rng.choice([-1, 1], N) * rng.uniform(0.35, 0.6, N))
res["real_spike"] = gallery_report(real_spk, lo, hi); res["real_nominal"] = gallery_report(donor, lo, hi)
cfgs = [("repo x final1", dict(kind="causal", space="x", final=1.0)),
        ("x final0", dict(kind="causal", space="x", final=0.0)),
        ("x0", dict(kind="causal", space="x0")),
        ("flip_ramp x0", dict(kind="flip_ramp", space="x0")),
        ("bidir x final1", dict(kind="bidir", space="x", final=1.0)),
        ("bidir x0", dict(kind="bidir", space="x0"))]
for slice_name, lam_anom in (("band_only", 0.0), ("needles", 1.0)):
    for name, kw in cfgs:
        den = LinearDenoiser(ch, W, sch, kw["kind"])
        P = protos[pidx]
        terms = [GuideTerm(0.3, lambda x: shell.band(x))]
        if lam_anom:
            terms.append(GuideTerm(lam_anom, lambda x: proto_energy(enc, x, P)))
        x = guided_ddim(den, donor, terms, rng=np.random.default_rng(1), space=kw["space"], final_scale=kw.get("final", 1.0))
        r = gallery_report(x, lo, hi)
        h, _ = shell.band(x); r["occupancy"] = float(np.mean(np.abs(h - shell.q_q) <= shell.delta))
        res[f"{slice_name}: {name}"] = r
for k, r in res.items():
    if k.startswith("jac"): continue
    print(f"{k:28s} env={r['env_window_frac']:.2f} max={r['env_max_excess']:.2f} dp999={r['diff_p999']:.3f} "
          f"pk@s={r['peak_at_start']:.2f} pk@e={r['peak_at_end']:.2f} H={r['peak_pos_entropy']:.2f} occ={r.get('occupancy', float('nan')):.2f}")
(OUT / "e3_slices.json").write_text(json.dumps(res, indent=1))
