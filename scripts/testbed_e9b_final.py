"""E9b: the texture deficit is the final Tweedie mean at sigma_min = 0.01."""
import json
from pathlib import Path
import numpy as np
from anogen.testbed.gauss import GPChannel, Schedule, LinearDenoiser, guided_ddim
from anogen.shell.features import embed_windows
from anogen.shell.evaluation import PHI_NAMES
from anogen.testbed.metrics import edge_stats

W, N = 512, 512
sch = Schedule()
print("sigma_min (t=0) =", float(np.sqrt(1 - sch.alpha_bar[0])))
res = {}
for ar_sigma in (0.02, 0.05):
    ch = GPChannel(ar_sigma=ar_sigma)
    rng = np.random.default_rng(0)
    donor, other = ch.sample(N, W, rng), ch.sample(N, W, rng)
    phi_o = embed_windows(other)
    for kind in ("bidir", "causal"):
        den = LinearDenoiser(ch, W, sch, kind)
        for fd in (True, False):
            x = guided_ddim(den, donor, [], rng=np.random.default_rng(1), nu=0.2, ddim_steps=50, final_denoise=fd)
            sh = (embed_windows(x).mean(0) - phi_o.mean(0)) / np.maximum(phi_o.std(0), 1e-9)
            e = edge_stats(x, other)
            key = f"ar_sigma={ar_sigma} {kind} final_denoise={fd}"
            res[key] = {"hf_band_shift": sh[9:].round(2).tolist(), "dstd_ratio_mid": e["dstd_ratio_mid"]}
            print(f"{key:42s} HF band shift {sh[9:].round(2)}  diff-std ratio {e['dstd_ratio_mid']:.2f}", flush=True)
Path("results/testbed/e9b_final.json").write_text(json.dumps(res, indent=1))
