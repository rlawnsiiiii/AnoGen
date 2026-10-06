"""E17: guidance strength that decays with the noise level (``guidance_decay``).

The repo adds a unit-norm kick of the same size at every step, including the
last, low-noise steps whose edits are never denoised again. Here the kick is
weighted by (σ_t/σ_start)^p, normalized to mean 1 over the chain (the same
total budget), so p = 0 is the frozen behaviour and p = 1 makes each edit
proportional to the noise still to be removed.

Level-shift slice as E2/E13 (shell band 0.3 + prototype 1.0, ν = 0.2, x̂₀-space),
with exact denoisers and with the trained numpy nets, alone and with two
self-recurrence passes.
"""
import json
import pickle
import sys
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
train = ch.sample(4096, W, rng)
lo, hi = float(train.min()), float(train.max())
enc = BlockEncoder(W, 8)
shell = Shell.from_nominal(enc, ch.sample(512, W, rng), ch.sample(512, W, rng))
donor = ch.sample(N, W, rng)
protos = enc.encode(level_shift(ch.sample(6, W, rng), np.array([90, 150, 210, 270, 330, 400]), np.full(6, 0.25)))
pidx = np.random.default_rng(17).integers(0, 6, N)
real = level_shift(ch.sample(N, W, rng), rng.integers(40, 470, N), np.full(N, 0.25))
terms = [GuideTerm(0.3, lambda x: shell.band(x)), GuideTerm(1.0, lambda x: proto_energy(enc, x, protos[pidx]))]
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


dens = {
    "exact flip ramp": LinearDenoiser(ch, W, sch, "flip_ramp"),
    "exact bidirectional": LinearDenoiser(ch, W, sch, "bidir"),
    "trained bidirectional net": NetDenoiser(nets["bidir"], sch, W, ch.mean),
    "trained causal net + flip ramp": FlipNet(nets["causal"], sch, W, ch.mean),
}
res = {"real level shifts": gallery_report(real, lo, hi)}
print(f"real: diff p99.9 {res['real level shifts']['diff_p999']:.3f}", flush=True)
for dname, den in dens.items():
    for final in (1.0, 0.25):
        for decay in (0.0, 0.5, 1.0, 2.0):
            for recur in (1, 2):
                if recur == 2 and final == 1.0:
                    continue
                x = guided_ddim(den, donor, terms, rng=np.random.default_rng(1), space="x0", final_scale=final,
                                guidance_decay=decay, n_recur=recur)
                r = gallery_report(x, lo, hi)
                r["proto_dist"] = float(np.linalg.norm(enc.encode(x) - protos[pidx], axis=1).mean())
                key = f"{dname} | final {final}, decay {decay}, recur {recur}"
                res[key] = r
                print(f"{key:66s} diff p99.9 {r['diff_p999']:.3f} start {r['peak_at_start']:.2f} end {r['peak_at_end']:.2f} "
                      f"proto {r['proto_dist']:.3f}", flush=True)
                (OUT / "e17_guidance_decay.json").write_text(json.dumps(res, indent=1))
