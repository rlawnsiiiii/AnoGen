"""E8b: does a *trained* causal ε-net show the start bias the exact estimators predict?"""
import json, pickle, sys
from pathlib import Path
import numpy as np
from anogen.testbed.gauss import (GPChannel, Schedule, BlockEncoder, Shell, GuideTerm, guided_ddim,
                                  proto_energy, level_shift)
from anogen.testbed.npnet import NetDenoiser
from anogen.testbed.metrics import gallery_report

OUT = Path("results/testbed"); W, N = 512, 256
ch, sch = GPChannel(), Schedule()
rng = np.random.default_rng(0)
train = ch.sample(4096, W, rng); lo, hi = float(train.min()), float(train.max())
nets = {k: pickle.load(open(OUT / "nets" / f"{k}.pkl", "rb"))["net"] for k in ("causal", "bidir")}


class FlipNet(NetDenoiser):
    def x0_hat(self, xt, t):
        ab = float(self.schedule.alpha_bar[int(t)])
        z = xt - np.sqrt(ab) * self.mean
        tt = np.full(len(xt), int(t))
        e_f = self.net.forward(z, tt)
        e_b = self.net.forward(z[:, ::-1].copy(), tt)[:, ::-1]
        w = np.linspace(0.0, 1.0, z.shape[1])[None, :]
        eps = w * e_f + (1 - w) * e_b
        return (xt - np.sqrt(1.0 - ab) * eps) / np.sqrt(ab)

    def vjp(self, g, t):
        raise NotImplementedError("use guidance space x0 with the flip ensemble")


dens = {"causal net": NetDenoiser(nets["causal"], sch, W, ch.mean),
        "bidir net": NetDenoiser(nets["bidir"], sch, W, ch.mean),
        "causal net + flip ramp": FlipNet(nets["causal"], sch, W, ch.mean)}
res = {"x0_err_t40": {}, "unguided": {}, "vjp_centroid_t40": {}, "level_shift": {}}
ref = ch.sample(N, W, rng)
for name, den in dens.items():
    ab = sch.alpha_bar[40]
    xt = np.sqrt(ab) * ref + np.sqrt(1 - ab) * rng.standard_normal(ref.shape)
    err = ((den.x0_hat(xt, 40) - ref) ** 2).mean(axis=0)
    res["x0_err_t40"][name] = {"first16": float(err[:16].mean()), "mid": float(err[248:264].mean()), "last16": float(err[-16:].mean())}
    if "flip" not in name:
        g = np.zeros((32, W)); g[:, W // 2] = 1.0
        den.x0_hat(xt[:32], 40)
        v = np.abs(den.vjp(g, 40)).mean(axis=0)
        res["vjp_centroid_t40"][name] = float((v * np.arange(W)).sum() / v.sum() / W)
    for nu, steps in ((1.0, 20), (0.2, 50)):
        x = guided_ddim(den, ref, [], rng=np.random.default_rng(1), nu=nu, ddim_steps=steps)
        res["unguided"][f"{name} nu={nu}"] = gallery_report(x, lo, hi)
    print(name, res["x0_err_t40"][name], res["vjp_centroid_t40"].get(name), {k: (round(v['peak_at_start'], 3), round(v['peak_at_end'], 3)) for k, v in res["unguided"].items() if k.startswith(name)}, flush=True)

enc = BlockEncoder(W, 8)
shell = Shell.from_nominal(enc, ch.sample(512, W, rng), ch.sample(512, W, rng))
protos = enc.encode(level_shift(ch.sample(6, W, rng), np.array([90, 150, 210, 270, 330, 400]), np.full(6, 0.25)))
pidx = np.random.default_rng(17).integers(0, 6, N)
for name, den, space, final in (("causal net", dens["causal net"], "x", 1.0),
                                ("causal net", dens["causal net"], "x0", 0.25),
                                ("causal net + flip ramp", dens["causal net + flip ramp"], "x0", 0.25),
                                ("bidir net", dens["bidir net"], "x", 1.0),
                                ("bidir net", dens["bidir net"], "x0", 0.25)):
    terms = [GuideTerm(0.3, lambda x: shell.band(x)), GuideTerm(1.0, lambda x: proto_energy(enc, x, protos[pidx]))]
    x = guided_ddim(den, ref, terms, rng=np.random.default_rng(1), space=space, final_scale=final)
    r = gallery_report(x, lo, hi)
    res["level_shift"][f"{name}, {space}-space, final {final}"] = r
    print(f"level shift | {name}, {space}, final {final}: start {r['peak_at_start']:.2f} end {r['peak_at_end']:.2f} diff p99.9 {r['diff_p999']:.3f}", flush=True)
(OUT / "e8_trained_nets.json").write_text(json.dumps(res, indent=1))
