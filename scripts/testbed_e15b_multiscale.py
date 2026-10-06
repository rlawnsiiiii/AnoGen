"""E15b: one score for every anomaly kind: per-t top-16 error, z-scored on held-out nominals, max over t."""
import json, pickle, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from anogen.testbed.gauss import GPChannel, Schedule, level_shift, spike

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed")
W, N = 512, 256
ch, sch = GPChannel(), Schedule(); rng = np.random.default_rng(0)
net = pickle.load(open(OUT / "nets" / "bidir.pkl", "rb"))["net"]
TS = (10, 20, 40, 60, 100, 150)

def top16_per_t(x, seed, draws=4):
    r = np.random.default_rng(seed); out = np.zeros((len(TS), len(x)))
    for k, t in enumerate(TS):
        ab = sch.alpha_bar[t]; acc = np.zeros(x.shape)
        for _ in range(draws):
            e = r.standard_normal(x.shape)
            acc += (net.forward(np.sqrt(ab) * (x - ch.mean) + np.sqrt(1 - ab) * e, np.full(len(x), t)) - e) ** 2
        out[k] = np.sort(acc / draws, axis=1)[:, -16:].mean(axis=1)
    return out

ref, nom = ch.sample(N, W, rng), ch.sample(N, W, rng)
base = ch.sample(3 * N, W, rng); sign = rng.choice([-1, 1], 3 * N)
anoms = {"spike 0.15, 3 bins": spike(base[:N], rng.integers(3, W - 6, N), 0.15 * sign[:N], width=3),
         "level shift 0.10": level_shift(base[N:2 * N], rng.integers(40, 470, N), 0.10 * sign[N:2 * N])}
drift = base[2 * N:].copy()
for i, p in enumerate(rng.integers(0, W - 128, N)):
    drift[i, p:p + 128] += 0.15 * sign[2 * N + i] * np.linspace(0, 1, 128); drift[i, p + 128:] += 0.15 * sign[2 * N + i]
anoms["drift 0.15 over 128 bins"] = drift
s_ref = top16_per_t(ref, 1); mu, sd = s_ref.mean(1, keepdims=True), s_ref.std(1, keepdims=True)
z_nom = (top16_per_t(nom, 2) - mu) / sd
res = {}
for name, xa in anoms.items():
    z_an = (top16_per_t(xa, 3) - mu) / sd
    y = np.r_[np.zeros(N), np.ones(N)]
    row = {f"t={t}": float(roc_auc_score(y, np.r_[z_nom[k], z_an[k]])) for k, t in enumerate(TS)}
    row["z-max over t"] = float(roc_auc_score(y, np.r_[z_nom.max(0), z_an.max(0)]))
    row["z-mean over t"] = float(roc_auc_score(y, np.r_[z_nom.mean(0), z_an.mean(0)]))
    res[name] = row; print(name, {k: round(v, 3) for k, v in row.items()}, flush=True)
(OUT / "e15b_multiscale.json").write_text(json.dumps(res, indent=1))
