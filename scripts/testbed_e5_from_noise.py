"""E5: from-noise start. The linear schedule keeps sqrt(ab_T) ~ 0.36 of the signal,
so x_T ~ N(0, I) is off the training marginal. Compare with a marginal-matched start."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from anogen.testbed.gauss import GPChannel, Schedule, LinearDenoiser, guided_ddim
from anogen.testbed.metrics import edge_stats

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed"); OUT.mkdir(parents=True, exist_ok=True)
W, N = 512, 512
sch = Schedule()
print("sqrt(alpha_bar_T) =", float(np.sqrt(sch.alpha_bar[-1])))
res = {}
for mean in (0.5, 0.8):
    ch = GPChannel(mean=mean)
    rng = np.random.default_rng(0)
    ref = ch.sample(N, W, rng)
    sd = float(np.sqrt(ch.acov(1)[0]))
    for kind in ("bidir", "causal"):
        den = LinearDenoiser(ch, W, sch, kind)
        for init, ni in (("N(0,I)", None), ("matched", (mean, sd))):
            x = guided_ddim(den, ref, [], rng=np.random.default_rng(1), start_from_noise=True, ddim_steps=50, noise_init=ni)
            r = {"level_mean": float(x.mean()), "level_bias_in_sd": float((x.mean() - mean) / sd),
                 "window_mean_std_ratio": float(x.mean(axis=1).std() / ref.mean(axis=1).std()),
                 **edge_stats(x, ref)}
            res[f"mean{mean}_{kind}_{init}"] = r
            print(f"data mean {mean} {kind:6s} {init:8s} gen mean {r['level_mean']:.3f} (bias {r['level_bias_in_sd']:+.2f} sd) "
                  f"std s/m/e {r['std_ratio_start']:.2f}/{r['std_ratio_mid']:.2f}/{r['std_ratio_end']:.2f}", flush=True)
(OUT / "e5_from_noise.json").write_text(json.dumps(res, indent=1))
