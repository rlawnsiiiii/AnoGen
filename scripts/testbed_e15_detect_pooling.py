"""E15: how to pool the per-bin denoising error into a window score (diffdetect).

diffdetect averages the ε-error over all W bins. A 3-bin spike changes 3 of
512 bins, so the mean dilutes it ~170-fold. Compared here on the trained
bidirectional numpy net (E8), same draws for every pooling:

  mean        current diffdetect score
  max16       max over bins of the 16-bin moving average of the per-bin error
  top16       mean of the 16 largest per-bin errors
  msma        per-t mean errors as a feature vector, Mahalanobis distance to
              held-out nominal windows (multiscale score matching, Mahmood 2021)
  msma_max16  the same with the max16 statistic per t

Anomalies: spikes (width 3), level shifts, and a slow drift, each at a uniform
position. AUROC against fresh nominal windows.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import roc_auc_score

from anogen.testbed.gauss import GPChannel, Schedule, level_shift, spike

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed")
W, N = 512, 256
ch, sch = GPChannel(), Schedule()
rng = np.random.default_rng(0)
net = pickle.load(open(OUT / "nets" / "bidir.pkl", "rb"))["net"]
TS = tuple(int(t) for t in (sys.argv[2].split(",") if len(sys.argv) > 2 else (10, 20, 40)))


def per_bin_error(x: np.ndarray, seed: int, draws: int = 4) -> np.ndarray:
    """(len(TS), N, W) mean squared ε-error per bin, per t."""
    r = np.random.default_rng(seed)
    out = np.zeros((len(TS), len(x), x.shape[1]))
    for k, t in enumerate(TS):
        ab = sch.alpha_bar[t]
        for _ in range(draws):
            e = r.standard_normal(x.shape)
            xt = np.sqrt(ab) * (x - ch.mean) + np.sqrt(1 - ab) * e
            out[k] += (net.forward(xt, np.full(len(x), t)) - e) ** 2
    return out / draws


def pools(err: np.ndarray) -> dict[str, np.ndarray]:
    mean_t = err.mean(axis=2)                                   # (T, N)
    max_t = uniform_filter1d(err, 16, axis=2).max(axis=2)        # (T, N)
    top = np.sort(err.mean(axis=0), axis=1)[:, -16:].mean(axis=1)
    return {"mean": mean_t.mean(0), "max16": max_t.mean(0), "top16": top, "_mean_t": mean_t, "_max_t": max_t}


def mahal(feat: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mu = ref.mean(axis=1, keepdims=True)
    cov = np.cov(ref) + 1e-9 * np.eye(len(ref))
    d = feat - mu
    return np.einsum("in,ij,jn->n", d, np.linalg.inv(cov), d)


ref_nom = ch.sample(N, W, rng)
test_nom = ch.sample(N, W, rng)
base = ch.sample(3 * N, W, rng)
sign = rng.choice([-1, 1], 3 * N)
anoms = {
    "spike 0.15, 3 bins": spike(base[:N], rng.integers(3, W - 6, N), 0.15 * sign[:N], width=3),
    "level shift 0.10": level_shift(base[N:2 * N], rng.integers(40, 470, N), 0.10 * sign[N:2 * N]),
    "drift 0.15 over 128 bins": None,
}
drift = base[2 * N:].copy()
for i, p in enumerate(rng.integers(0, W - 128, N)):
    drift[i, p:p + 128] += 0.15 * sign[2 * N + i] * np.linspace(0, 1, 128)
    drift[i, p + 128:] += 0.15 * sign[2 * N + i]
anoms["drift 0.15 over 128 bins"] = drift

p_ref = pools(per_bin_error(ref_nom, 1))
p_nom = pools(per_bin_error(test_nom, 2))
res = {}
for name, xa in anoms.items():
    p_an = pools(per_bin_error(xa, 3))
    y = np.r_[np.zeros(N), np.ones(N)]
    row = {}
    for key in ("mean", "max16", "top16"):
        row[key] = float(roc_auc_score(y, np.r_[p_nom[key], p_an[key]]))
    row["msma"] = float(roc_auc_score(y, np.r_[mahal(p_nom["_mean_t"], p_ref["_mean_t"]), mahal(p_an["_mean_t"], p_ref["_mean_t"])]))
    row["msma_max16"] = float(roc_auc_score(y, np.r_[mahal(p_nom["_max_t"], p_ref["_max_t"]), mahal(p_an["_max_t"], p_ref["_max_t"])]))
    res[name] = row
    print(f"{name:26s} " + "  ".join(f"{k} {v:.3f}" for k, v in row.items()), flush=True)
(OUT / ("e15_detect_pooling.json" if TS == (10, 20, 40) else f"e15_detect_pooling_t{'-'.join(map(str, TS))}.json")).write_text(json.dumps(res, indent=1))
