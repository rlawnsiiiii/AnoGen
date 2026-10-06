"""E12: does feature_pack_v1 reward smoothing that contains no anomaly at all?

Galleries are nominal donors with *no* anomaly content, low-pass filtered at
increasing strength (Gaussian kernel, sd in bins), or with extra white texture.
Queries are synthetic anomaly windows (level shifts + spikes + both) on fresh
nominal windows. If ARP in frozen phi rises with smoothing alone, ARP rewards
the texture loss of the sampler (E10, E11), not anomaly realism.

Also scored: an oracle gallery (fresh anomalies from the same generator), phi
standardized per feature with the two constant dims dropped, and phi without
the three high-frequency bands.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from anogen.shell.coverage import arp, min_distances
from anogen.shell.features import embed_windows
from anogen.testbed.gauss import GPChannel, level_shift, spike

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed")
W, N = 512, 512
ch = GPChannel()
rng = np.random.default_rng(0)


def anomalies(n: int, r: np.random.Generator) -> np.ndarray:
    base = ch.sample(n, W, r)
    k = r.integers(0, 3, n)
    sz = r.choice([-1, 1], n) * r.uniform(0.15, 0.35, n)
    out = base.copy()
    ls = k == 0
    out[ls] = level_shift(base[ls], r.integers(40, 470, ls.sum()), sz[ls])
    sp = k == 1
    out[sp] = spike(base[sp], r.integers(10, 500, sp.sum()), 2.5 * sz[sp], width=3)
    both = k == 2
    tmp = level_shift(base[both], r.integers(40, 470, both.sum()), sz[both])
    out[both] = spike(tmp, r.integers(10, 500, both.sum()), 2.5 * sz[both], width=3)
    return out


queries = anomalies(256, np.random.default_rng(1))
donors = ch.sample(N, W, rng)
nominal_ref = ch.sample(N, W, np.random.default_rng(5))
phi_ref = embed_windows(nominal_ref)
keep = phi_ref.std(axis=0) > 1e-9
mu, sd = phi_ref[:, keep].mean(axis=0), phi_ref[:, keep].std(axis=0)
NO_HF = np.array([True] * 9 + [False] * 3)


def scores(gal: np.ndarray) -> dict[str, float]:
    pq, pg = embed_windows(queries), embed_windows(gal)
    std_q, std_g = (pq[:, keep] - mu) / sd, (pg[:, keep] - mu) / sd
    return {
        "arp_frozen_phi": arp(min_distances(pq, pg)),
        "arp_standardized_phi": arp(min_distances(std_q, std_g)),
        "arp_phi_without_hf_bands": arp(min_distances(pq[:, NO_HF], pg[:, NO_HF])),
        "diff_sd_ratio": float(np.diff(gal, axis=1).std() / np.diff(donors, axis=1).std()),
    }


res = {}
for s in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
    g = gaussian_filter1d(donors, s, axis=1, mode="reflect") if s > 0 else donors
    res[f"donors, low-pass sd {s:g} bins"] = scores(g)
for a in (0.01, 0.02):
    res[f"donors + white texture sd {a:g}"] = scores(donors + a * rng.standard_normal(donors.shape))
res["oracle: fresh anomalies, same generator"] = scores(anomalies(N, np.random.default_rng(7)))
res["oracle, low-pass sd 1.5 bins"] = scores(gaussian_filter1d(anomalies(N, np.random.default_rng(7)), 1.5, axis=1))
for k, r in res.items():
    print(f"{k:42s} ARP frozen {r['arp_frozen_phi']:.3f} | standardized {r['arp_standardized_phi']:.3f} | "
          f"no HF {r['arp_phi_without_hf_bands']:.3f} | diff sd x{r['diff_sd_ratio']:.2f}", flush=True)
(OUT / "e12_phi_smoothing.json").write_text(json.dumps(res, indent=1))
