"""Can the hand-made contrast template be beaten by a learned one?

w is a template placed at a sampled position, so the learnable object is the
template, not a 512-bin vector: a whole-window linear classifier would bake in
the position of the training events and undo the position fix.

Candidates, all forced to sum to zero so they stay level-blind like the hand one:
  hand       the template in shell/contrast.py
  matched    mean of sign-aligned anomaly patches (optimal under white noise)
  whitened   Sigma^-1 mu with Sigma the nominal patch covariance (coloured noise)
  logistic   a linear classifier on the patch, which is the "classifier weights"
             idea applied where it is shift-invariant

Scored two ways on held-out events. AUC saturates on this panel, so the ranking
metric is the detection margin: median anomaly score / p99.9 of nominal scores.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/extras/SSD/AI/AnoGen/src")
from anogen.config import load_config, panel_path  # noqa: E402
from anogen.shell.contrast import contrast_weights, measure_contrast  # noqa: E402
from anogen.shell.data import load_panel  # noqa: E402
from anogen.shell.events import types_from_cfg  # noqa: E402
from anogen.shell.morphology import kinds_for_windows  # noqa: E402
from anogen.shell.scaler import load_minmax  # noqa: E402

W = 512
cfg = load_config("/mnt/extras/SSD/AI/AnoGen/configs/shell_mission1.yaml")
root = Path(cfg["_repo_root"])
s0 = root / cfg["s0_dir"]
sc = load_minmax(s0 / "minmax_scaler.npz")
lab = np.load(s0 / "labeled_arrays.npz")
meta = pd.read_csv(s0 / "labeled_windows.csv")
anom = meta[meta["kind"] == "anomaly"].reset_index(drop=True)
x_a = np.asarray(lab["anomaly"], dtype=np.float64)
kl = kinds_for_windows(x_a, anom, types_from_cfg(cfg))
ev_a = anom["event_id"].astype(str).to_numpy()
xs_a = sc.transform(x_a, np.asarray(lab["anomaly_channel"]))

splits = cfg.get("splits") or {}
panel = load_panel(
    panel_path(cfg),
    channels=list(cfg.get("channels") or []),
    official_train_end=splits.get("official_train_end", "2007-01-01"),
    allow_test_telemetry=False,
    bin_seconds=int(cfg.get("bin_seconds", 30)),
)
nom = pd.read_csv(s0 / "nominal_index.csv")
rng = np.random.default_rng(0)
pick = rng.choice(len(nom), 6000, replace=False)
st, cn = nom["start"].to_numpy(np.int64)[pick], nom["channel_idx"].to_numpy(np.int64)[pick]
xs_n = sc.transform(
    np.stack([panel.Y[s : s + W, c] for s, c in zip(st, cn, strict=True)]).astype(np.float64), cn
)


def patches(x, centres, half, align=True):
    out = []
    for w_, p in zip(x, centres, strict=True):
        p = int(np.clip(p, half, W - half - 1))
        q = w_[p - half : p + half + 1]
        q = q - np.median(q)
        if align:
            # measure_contrast returns the best 3-bin window, so the true peak can
            # sit a bin off centre: re-centre on it, then flip down-excursions up
            c = int(np.clip(p - half + int(np.abs(q).argmax()), half, W - half - 1))
            q = w_[c - half : c + half + 1]
            q = q - np.median(q)
            if q[half] < 0:
                q = -q
        out.append(q)
    return np.asarray(out)


def scores(x, tmpl, half):
    t = np.asarray(tmpl, dtype=np.float64)
    t = t - t.mean()
    t = t / (np.linalg.norm(t) + 1e-12)
    win = np.lib.stride_tricks.sliding_window_view(x, len(t), axis=1)
    return np.abs(win @ t).max(axis=1)


def auc(pos, neg):
    gt = (pos[:, None] > neg[None, :]).mean()
    return float(gt + 0.5 * (pos[:, None] == neg[None, :]).mean())


def logistic(P, N, iters=400, lr=0.5, l2=1e-2):
    X = np.vstack([P, N])
    y = np.r_[np.ones(len(P)), np.zeros(len(N))]
    sw = np.r_[np.full(len(P), len(N) / len(P)), np.ones(len(N))]  # balance the classes
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        g = X.T @ (sw * (p - y)) / sw.sum() + l2 * w
        w -= lr * g
    return w


SPEC = {"real ESA Point / Global": ("spike", 12), "real level shift": ("step_head", 32)}
for kind, (ckind, half) in SPEC.items():
    m = kl == kind
    events = np.unique(ev_a[m])
    _, pf = measure_contrast(ckind, xs_a[m])
    centres = np.round(pf * (W - 1)).astype(int)
    P = patches(xs_a[m], centres, half)
    N = patches(xs_n, rng.integers(half, W - half - 1, len(xs_n)), half, align=False)
    hand = contrast_weights(ckind, np.array([W // 2]), W)[0][W // 2 - half : W // 2 + half + 1]

    print(f"\n=== {kind}: {m.sum()} windows from {len(events)} event(s), patch {2*half+1} bins ===")
    if len(events) < 2:
        print("  held-out evaluation impossible: a single event, so any learned template")
        print("  is fitted and tested on the same anomaly. Reporting in-sample only.")
    cov = np.cov(N, rowvar=False)
    cov = cov + 1e-3 * np.eye(len(cov)) * np.trace(cov) / len(cov)

    def fit(name, Ptr):
        if name == "hand":
            return hand
        if name == "matched":
            return Ptr.mean(axis=0)
        if name == "whitened":
            return np.linalg.solve(cov, Ptr.mean(axis=0))
        return logistic(Ptr, N)

    for name in ("hand", "matched", "whitened", "logistic"):
        a_s, margins = [], []
        folds = events if len(events) > 1 else [None]
        for e in folds:
            tr = np.ones(int(m.sum()), bool) if e is None else (ev_a[m] != e)
            te = np.ones(int(m.sum()), bool) if e is None else (ev_a[m] == e)
            t = fit(name, P[tr])
            sp, sn = scores(xs_a[m][te], t, half), scores(xs_n, t, half)
            a_s.append(auc(sp, sn))
            margins.append(float(np.median(sp) / np.percentile(sn, 99.9)))
        cos = float(t @ hand / (np.linalg.norm(t) * np.linalg.norm(hand) + 1e-12))
        print(f"  {name:9s} cos to hand {cos:+.3f}   AUC {np.mean(a_s):.4f}   "
              f"margin {np.mean(margins):6.2f}  per fold {np.round(margins, 2)}")

# --- what do the learned templates actually look like? ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

kind, (ckind, half) = "real ESA Point / Global", ("spike", 12)
m = kl == kind
_, pf = measure_contrast(ckind, xs_a[m])
P = patches(xs_a[m], np.round(pf * (W - 1)).astype(int), half)
N = patches(xs_n, rng.integers(half, W - half - 1, len(xs_n)), half, align=False)
cov = np.cov(N, rowvar=False)
cov = cov + 1e-3 * np.eye(len(cov)) * np.trace(cov) / len(cov)
hand = contrast_weights(ckind, np.array([W // 2]), W)[0][W // 2 - half : W // 2 + half + 1]
cands = {
    "hand": hand,
    "matched": P.mean(axis=0),
    "whitened": np.linalg.solve(cov, P.mean(axis=0)),
    "logistic": logistic(P, N),
}
fig, ax = plt.subplots(1, 5, figsize=(16, 2.6), sharex=True)
b = np.arange(-half, half + 1)
ax[0].plot(b, (P / np.abs(P).max(axis=1, keepdims=True)).T, lw=0.7, color="#9b1c1c", alpha=0.5)
ax[0].set_title("the 18 real spike patches\n(sign-aligned, scaled)", fontsize=9)
for a_, (n, t) in zip(ax[1:], cands.items()):
    t = t - t.mean()
    t = t / (np.abs(t).max() + 1e-12)
    if t[half] < 0:
        t = -t
    a_.bar(b, t, width=0.85, color=["#c0392b" if v > 0 else "#1b4f9c" for v in t])
    a_.set_title(f"{n}", fontsize=9)
for a_ in ax:
    a_.axhline(0, lw=0.5, color="0.5")
    a_.tick_params(labelsize=7)
    a_.set_xlabel("bin from centre", fontsize=7)
    for s in ("top", "right"):
        a_.spines[s].set_visible(False)
fig.suptitle("Point / Global: the real excursions and four candidate templates (normalized)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig("/mnt/extras/SSD/AI/AnoGen/docs/template_candidates_v2.png", dpi=150)
print("/mnt/extras/SSD/AI/AnoGen/docs/template_candidates_v2.png")
