"""E17b: does decaying guidance buy target reach at the cost of diversity?

Same level-shift slice as E17 (exact flip-ramp denoiser and the trained
bidirectional net, x̂₀-space, final 0.25). For decay 0 and 1, with and without
two recurrences, and for a halved prototype weight, report the prototype
distance (target reach) and the spread of samples that share a prototype
(mean pairwise L2 within a prototype group / the same for their donors).
"""
import json, pickle, sys
from pathlib import Path
import numpy as np
from anogen.testbed.gauss import (BlockEncoder, GPChannel, GuideTerm, LinearDenoiser, Schedule, Shell, guided_ddim,
                                  level_shift, proto_energy)
from anogen.testbed.metrics import gallery_report
from anogen.testbed.npnet import NetDenoiser

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed")
W, N = 512, 128
ch, sch = GPChannel(), Schedule()
rng = np.random.default_rng(0)
train = ch.sample(4096, W, rng); lo, hi = float(train.min()), float(train.max())
enc = BlockEncoder(W, 8)
shell = Shell.from_nominal(enc, ch.sample(512, W, rng), ch.sample(512, W, rng))
donor = ch.sample(N, W, rng)
protos = enc.encode(level_shift(ch.sample(6, W, rng), np.array([90, 150, 210, 270, 330, 400]), np.full(6, 0.25)))
pidx = np.random.default_rng(17).integers(0, 6, N)


def spread(x):
    vals = []
    for k in range(6):
        g = x[pidx == k]
        if len(g) > 1:
            d = np.linalg.norm(g[:, None, :] - g[None, :, :], axis=2)
            vals.append(d[np.triu_indices(len(g), 1)].mean())
    return float(np.mean(vals))


base_spread = spread(donor)
net = pickle.load(open(OUT / "nets" / "bidir.pkl", "rb"))["net"]
dens = {"exact flip ramp": LinearDenoiser(ch, W, sch, "flip_ramp"), "trained bidirectional net": NetDenoiser(net, sch, W, ch.mean)}
res = {"donor spread": base_spread}
for dname, den in dens.items():
    for lam_p in (1.0, 0.5):
        terms = [GuideTerm(0.3, lambda x: shell.band(x)), GuideTerm(lam_p, lambda x: proto_energy(enc, x, protos[pidx]))]
        for decay in (0.0, 1.0):
            for recur in (1, 2):
                x = guided_ddim(den, donor, terms, rng=np.random.default_rng(1), space="x0", final_scale=0.25,
                                guidance_decay=decay, n_recur=recur)
                r = gallery_report(x, lo, hi)
                row = {"proto_dist": float(np.linalg.norm(enc.encode(x) - protos[pidx], axis=1).mean()),
                       "spread_vs_donors": spread(x) / base_spread, "diff_p999": r["diff_p999"]}
                key = f"{dname} | proto weight {lam_p}, decay {decay}, recur {recur}"
                res[key] = row
                print(f"{key:62s} proto {row['proto_dist']:.3f} spread {row['spread_vs_donors']:.2f} diff p99.9 {row['diff_p999']:.3f}", flush=True)
                (OUT / "e17b_decay_diversity.json").write_text(json.dumps(res, indent=1))
