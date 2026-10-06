"""E18: which two-sample test and which feature space can see realistic defects?

"Real anomalies" are event-clustered synthetic anomalies (64 events x 4 windows;
spikes, level shifts, drifts) on the testbed channel. Each gallery differs from
them in one controlled way, and three classifier two-sample tests try to tell
them apart (grouped 5-fold CV, real rows grouped by event):

  raw   logistic regression on the raw windows (what realism.classifier_two_sample_test does)
  phi   logistic regression on standardized feature_pack_v1
  rocket ridge on ROCKET features (shell/rocket.py)

AUC 0.5 = indistinguishable. A useful test stays near 0.5 for the null gallery
and rises for every defect. ARP is also reported in frozen φ and in the ROCKET
PCA space, to see which one rewards a defect-free gallery.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from anogen.shell.coverage import arp, min_distances
from anogen.shell.features import embed_windows
from anogen.shell.rocket import RocketSpace, c2st_auc
from anogen.testbed.gauss import GPChannel, level_shift, spike

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed")
W, E, K = 512, 64, 4
ch = GPChannel()


def events(seed: int, *, amp_scale: float = 1.0, early: bool = False, smooth: float = 0.0, none: bool = False):
    r = np.random.default_rng(seed)
    xs, groups = [], []
    for e in range(E):
        kind = r.integers(0, 3)
        amp = r.choice([-1, 1]) * r.uniform(0.15, 0.35) * amp_scale
        pos = int(r.integers(10, 60)) if early else int(r.integers(60, 450))
        base = ch.sample(K, W, r)
        jit = np.clip(pos + r.integers(-10, 11, K), 5, W - 70)
        if none:
            x = base
        elif kind == 0:
            x = spike(base, jit, np.full(K, 2.0 * amp), width=3)
        elif kind == 1:
            x = level_shift(base, jit, np.full(K, amp))
        else:
            x = base.copy()
            for i, p in enumerate(jit):
                x[i, p:p + 64] += amp * np.linspace(0, 1, 64)
                x[i, p + 64:] += amp
        if smooth > 0:
            x = gaussian_filter1d(x, smooth, axis=1, mode="reflect")
        xs.append(x)
        groups += [f"{seed}-{e}"] * K
    return np.concatenate(xs), np.asarray(groups)


real, real_g = events(1)
galleries = {
    "null: same generator, new events": events(2),
    "nominal windows, no anomaly": events(3, none=True),
    "texture smoothed (low-pass sd 1.5 bins)": events(4, smooth=1.5),
    "amplitude x1.6": events(5, amp_scale=1.6),
    "all events in the first 10 % of the window": events(6, early=True),
}
ref = ch.sample(512, W, np.random.default_rng(9))
space = RocketSpace(n_kernels=500, n_components=64, seed=0).fit(ref)
phi_ref = embed_windows(ref)
keep = phi_ref.std(axis=0) > 1e-9
mu, sd = phi_ref[:, keep].mean(axis=0), phi_ref[:, keep].std(axis=0)


def phi_std(x):
    return (embed_windows(x)[:, keep] - mu) / sd


f_real = {"raw": real, "phi": phi_std(real), "rocket": space.features(real)}
e_real = {"phi": embed_windows(real), "rocket": space.embed(real)}
res = {}
for name, (x, g) in galleries.items():
    row = {}
    for feat, fn, clf in (("raw", lambda a: a, "logistic"), ("phi", phi_std, "logistic"), ("rocket", space.features, "ridge")):
        aucs = [c2st_auc(f_real[feat], fn(x), real_g, g, seed=s, classifier=clf)["auc"] for s in (0, 1, 2)]
        row[f"c2st_auc_{feat}"] = float(np.mean(aucs))
    row["arp_phi"] = arp(min_distances(e_real["phi"], embed_windows(x)))
    row["arp_rocket"] = arp(min_distances(e_real["rocket"], space.embed(x)))
    res[name] = row
    print(f"{name:44s} C2ST AUC raw {row['c2st_auc_raw']:.2f} phi {row['c2st_auc_phi']:.2f} rocket {row['c2st_auc_rocket']:.2f} | "
          f"ARP phi {row['arp_phi']:.3f} rocket {row['arp_rocket']:.3f}", flush=True)
(OUT / "e18_metrics.json").write_text(json.dumps(res, indent=1))
