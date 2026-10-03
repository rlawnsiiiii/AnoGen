"""E7: what does DDIM regeneration do to phi = feature_pack_v1?

phi z-scores each window, so its first two components are constant (0 and 1),
and it is not standardized across features. Which components dominate the
distances, and which ones move when a nominal window is regenerated?
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from anogen.testbed.gauss import GPChannel, Schedule, LinearDenoiser, guided_ddim
from anogen.shell.features import embed_windows
from anogen.shell.evaluation import PHI_NAMES, distance_matrix

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed"); OUT.mkdir(parents=True, exist_ok=True)
W, N = 512, 512
ch, sch = GPChannel(), Schedule()
rng = np.random.default_rng(0)
donor = ch.sample(N, W, rng)
other = ch.sample(N, W, rng)   # independent real nominal set
phi_d, phi_o = embed_windows(donor), embed_windows(other)
res = {"phi_sd_real": dict(zip(PHI_NAMES, phi_d.std(axis=0).round(4).tolist())),
       "share_of_pairwise_sq_distance": {}}
diff = phi_d[:, None, :] - phi_o[None, :256, :]
share = (diff ** 2).mean(axis=(0, 1)); share = share / share.sum()
res["share_of_pairwise_sq_distance"] = dict(zip(PHI_NAMES, share.round(4).tolist()))
print("phi sd over real nominal windows:", {k: v for k, v in res["phi_sd_real"].items()})
print("share of pairwise squared phi distance:", {k: v for k, v in res["share_of_pairwise_sq_distance"].items() if v > 0.01})
def arp(q, g):
    return float(1.0 / (1.0 + distance_matrix(q, g).min(axis=1).mean()))
rows = {}
for kind in ("bidir", "causal"):
    den = LinearDenoiser(ch, W, sch, kind)
    for nu, steps in ((1.0, 20), (1.0, 100), (0.2, 50)):
        x = guided_ddim(den, donor, [], rng=np.random.default_rng(1), nu=nu, ddim_steps=steps)
        phi_g = embed_windows(x)
        shift = (phi_g.mean(axis=0) - phi_d.mean(axis=0)) / np.maximum(phi_d.std(axis=0), 1e-9)
        rows[f"{kind} nu={nu} steps={steps}"] = {
            "arp_real_queries": arp(phi_o, phi_g),
            "mean_shift_in_real_sd": dict(zip(PHI_NAMES, shift.round(2).tolist())),
        }
rows["donor (real nominal)"] = {"arp_real_queries": arp(phi_o, phi_d)}
for k, r in rows.items():
    top = sorted((r.get("mean_shift_in_real_sd") or {}).items(), key=lambda kv: -abs(kv[1]))[:3]
    print(f"{k:28s} ARP(real nominal queries) {r['arp_real_queries']:.3f}  largest phi shifts: {top}")
res["galleries"] = rows
(OUT / "e7_phi.json").write_text(json.dumps(res, indent=1))
