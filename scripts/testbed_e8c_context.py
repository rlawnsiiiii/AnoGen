"""E8c: trained nets + real context on both sides of the window (generate W+2B, keep the middle W)."""
import json, pickle
from pathlib import Path
import numpy as np
from anogen.testbed.gauss import GPChannel, Schedule, guided_ddim
from anogen.testbed.npnet import NetDenoiser
from anogen.testbed.metrics import gallery_report

OUT = Path("results/testbed"); W, N = 512, 256
ch, sch = GPChannel(), Schedule()
rng = np.random.default_rng(0)
train = ch.sample(4096, W, rng); lo, hi = float(train.min()), float(train.max())
nets = {k: pickle.load(open(OUT / "nets" / f"{k}.pkl", "rb"))["net"] for k in ("causal", "bidir")}
res = {}
for B in (0, 32, 64, 128):
    long = ch.sample(N, W + 2 * B, rng)
    for kind in ("causal", "bidir"):
        for side in (("start",) if kind == "causal" else ()) + ("both",):
            pre, post = B, (B if side == "both" else 0)
            if B == 0 and side == "start":
                continue
            x_in = long[:, B - pre : (W + 2 * B) - (B - post)] if B else long
            den = NetDenoiser(nets[kind], sch, x_in.shape[1], ch.mean)
            for nu, steps in ((1.0, 20), (0.2, 50)):
                x = guided_ddim(den, x_in, [], rng=np.random.default_rng(1), nu=nu, ddim_steps=steps, crop=pre, crop_end=post)
                r = gallery_report(x, lo, hi)
                key = f"{kind} net, context {B} bins ({side if B else 'none'}), nu={nu}"
                res[key] = r
                print(f"{key:52s} start {r['peak_at_start']:.3f} end {r['peak_at_end']:.3f}", flush=True)
(OUT / "e8c_context.json").write_text(json.dumps(res, indent=1))
