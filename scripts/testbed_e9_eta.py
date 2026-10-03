"""E9: does stochastic DDIM (eta > 0) fix the high-frequency texture shift of deterministic DDIM?"""
import json, sys
from pathlib import Path
import numpy as np
from anogen.testbed.gauss import GPChannel, Schedule, LinearDenoiser, guided_ddim
from anogen.shell.features import embed_windows
from anogen.shell.evaluation import PHI_NAMES
from anogen.testbed.metrics import edge_stats

OUT = Path("results/testbed"); W, N = 512, 512
ch, sch = GPChannel(), Schedule()
rng = np.random.default_rng(0)
donor, other = ch.sample(N, W, rng), ch.sample(N, W, rng)
phi_d = embed_windows(other)
res = {}
den = LinearDenoiser(ch, W, sch, "bidir")
for nu, steps in ((1.0, 20), (1.0, 50), (0.2, 50)):
    for eta in (0.0, 0.5, 1.0):
        x = guided_ddim(den, donor, [], rng=np.random.default_rng(1), nu=nu, ddim_steps=steps, eta=eta)
        sh = (embed_windows(x).mean(0) - phi_d.mean(0)) / np.maximum(phi_d.std(0), 1e-9)
        e = edge_stats(x, other)
        res[f"nu={nu} steps={steps} eta={eta}"] = {"band_shift": dict(zip(PHI_NAMES[8:], sh[8:].round(2).tolist())),
                                                   "dstd_ratio_mid": e["dstd_ratio_mid"], "std_ratio_mid": e["std_ratio_mid"]}
        print(f"nu={nu} steps={steps} eta={eta}: HF band shifts {sh[9:].round(2)}  diff-std ratio {e['dstd_ratio_mid']:.2f}  std ratio {e['std_ratio_mid']:.2f}", flush=True)
(OUT / "e9_eta.json").write_text(json.dumps(res, indent=1))
