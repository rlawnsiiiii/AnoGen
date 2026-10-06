"""Random-convolution features (ROCKET / MiniRocket style), numpy only.

A second, learned-free feature space for the realism protocol, next to the
frozen 12-D ``feature_pack_v1`` (φ). φ z-scores every window and is dominated
by its high-frequency log-band energies (docs/TESTBED.md, E7, E12); random
dilated convolutions cover shapes at many scales and are the standard strong
baseline for time-series classification (Dempster et al., ROCKET, DMKD 2020;
MiniRocket, KDD 2021).

Kernels follow ROCKET: length ∈ {7, 9, 11}, weights N(0, 1) mean-centred,
dilation ⌊2^u⌋ with u ~ U(0, log2((W−1)/(L−1))), zero padding with
probability 1/2. Biases follow MiniRocket: a random quantile (U(0.1, 0.9)) of
the kernel's output on a random reference window, so the proportion of
positive values (ppv) is informative on this data's scale. Two features per
kernel: ppv and max.

Inputs are expected in channel-span units (``realism.channel_span_normalize``),
so level and amplitude are kept (unlike φ). ``RocketSpace`` standardizes the
features on a reference set (the real nominal donors) and projects them onto
its principal components for distance-based scores (ARP, PRDC).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RocketTransform:
    n_kernels: int = 500
    seed: int = 0
    lengths: tuple[int, ...] = (7, 9, 11)
    kernels: list[tuple[np.ndarray, int, int, float]] = field(default_factory=list, init=False)
    width: int = field(default=0, init=False)

    def fit(self, x_ref: np.ndarray) -> RocketTransform:
        x = np.asarray(x_ref, dtype=np.float64)
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("x_ref must be a non-empty (N, W) array")
        rng = np.random.default_rng(int(self.seed))
        self.width = int(x.shape[1])
        self.kernels = []
        for _ in range(int(self.n_kernels)):
            length = int(rng.choice(self.lengths))
            w = rng.standard_normal(length)
            w -= w.mean()
            a = np.log2(max((self.width - 1) / (length - 1), 1.0))
            dil = int(np.floor(2.0 ** rng.uniform(0.0, a)))
            pad = ((length - 1) * dil) // 2 if rng.random() < 0.5 else 0
            ref = x[int(rng.integers(len(x)))][None, :]
            out = _conv(ref, w, dil, pad)[0]
            bias = -float(np.quantile(out, rng.uniform(0.1, 0.9)))
            self.kernels.append((w, dil, pad, bias))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """(N, W) -> (N, 2·n_kernels): [ppv, max] per kernel."""
        if not self.kernels:
            raise RuntimeError("call fit() first")
        a = np.asarray(x, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != self.width:
            raise ValueError(f"expected (N, {self.width}) windows, got {a.shape}")
        feats = np.empty((len(a), 2 * len(self.kernels)), dtype=np.float64)
        for k, (w, dil, pad, bias) in enumerate(self.kernels):
            out = _conv(a, w, dil, pad) + bias
            feats[:, 2 * k] = (out > 0).mean(axis=1)
            feats[:, 2 * k + 1] = out.max(axis=1)
        return feats


def _conv(x: np.ndarray, w: np.ndarray, dil: int, pad: int) -> np.ndarray:
    """Dilated 'valid' convolution (cross-correlation) of each row with w."""
    xp = np.pad(x, ((0, 0), (pad, pad))) if pad else x
    span = (len(w) - 1) * dil
    n_out = xp.shape[1] - span
    if n_out <= 0:
        raise ValueError("kernel span exceeds the window")
    out = np.zeros((len(xp), n_out), dtype=np.float64)
    for j, wj in enumerate(w):
        out += wj * xp[:, j * dil : j * dil + n_out]
    return out


@dataclass
class RocketSpace:
    """ROCKET features standardized on a reference set, plus a PCA projection."""

    n_kernels: int = 500
    n_components: int = 64
    seed: int = 0
    transform_: RocketTransform | None = field(default=None, init=False)
    mean_: np.ndarray | None = field(default=None, init=False)
    sd_: np.ndarray | None = field(default=None, init=False)
    components_: np.ndarray | None = field(default=None, init=False)

    def fit(self, x_ref: np.ndarray) -> RocketSpace:
        self.transform_ = RocketTransform(n_kernels=self.n_kernels, seed=self.seed).fit(x_ref)
        f = self.transform_.transform(x_ref)
        self.mean_ = f.mean(axis=0)
        self.sd_ = np.where(f.std(axis=0) > 1e-9, f.std(axis=0), 1.0)
        z = (f - self.mean_) / self.sd_
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        self.components_ = vt[: min(int(self.n_components), len(vt))]
        return self

    def features(self, x: np.ndarray) -> np.ndarray:
        """Standardized full feature vector (for classifiers)."""
        if self.transform_ is None:
            raise RuntimeError("call fit() first")
        return (self.transform_.transform(x) - self.mean_) / self.sd_

    def embed(self, x: np.ndarray) -> np.ndarray:
        """PCA coordinates (for nearest-neighbour scores such as ARP)."""
        return self.features(x) @ self.components_.T


def c2st_auc(
    feat_real: np.ndarray,
    feat_synth: np.ndarray,
    real_groups: np.ndarray,
    synth_groups: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 0,
    classifier: str = "ridge",
) -> dict[str, float]:
    """Grouped, cross-validated classifier two-sample test on given features.

    Real rows are split by their group (event) so that crops of one event
    never sit on both sides; synthetic rows by theirs (e.g. donor index).
    Returns out-of-fold AUC and balanced accuracy (0.5 = indistinguishable).
    ``classifier``: ``ridge`` (RidgeClassifierCV, ROCKET's default) or
    ``logistic``.
    """
    from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    fr, fs = np.asarray(feat_real, dtype=np.float64), np.asarray(feat_synth, dtype=np.float64)
    x = np.concatenate([fr, fs], axis=0)
    y = np.r_[np.ones(len(fr)), np.zeros(len(fs))]
    rg = np.asarray(real_groups).astype(str)
    sg = np.asarray(synth_groups).astype(str)
    groups = np.r_[np.char.add("r:", rg), np.char.add("s:", sg)]
    k = int(min(n_splits, len(np.unique(rg)), len(np.unique(sg))))
    if k < 2:
        raise ValueError("need at least two real and two synthetic groups")
    # shuffle group order deterministically so folds do not follow input order
    rng = np.random.default_rng(int(seed))
    uniq = np.unique(groups)
    relabel = dict(zip(uniq, rng.permutation(len(uniq)), strict=True))
    g_num = np.array([relabel[g] for g in groups])
    score = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(x, y, groups=g_num):
        if len(np.unique(y[tr])) < 2:
            continue
        if classifier == "ridge":
            clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight="balanced")
        elif classifier == "logistic":
            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        else:
            raise ValueError(classifier)
        clf.fit(x[tr], y[tr])
        score[te] = clf.decision_function(x[te])
    auc = float(roc_auc_score(y, score))
    return {
        "auc": auc,
        "balanced_accuracy": float(balanced_accuracy_score(y, score > 0)),
        "n_real": float(len(fr)),
        "n_synth": float(len(fs)),
        "n_folds": float(k),
    }


def rocket_c2st(
    space: RocketSpace,
    u_real: np.ndarray,
    u_synth: np.ndarray,
    real_ch: np.ndarray,
    synth_ch: np.ndarray,
    real_groups: np.ndarray,
    synth_groups: np.ndarray,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    feat_real: np.ndarray | None = None,
) -> dict[str, float]:
    """Channel-balanced, event-grouped ROCKET C2ST (inputs in channel-span units).

    Each seed subsamples both sets to the same count per shared channel (as
    ``realism.classifier_two_sample_test`` does), so the classifier cannot win
    on channel composition alone, then runs ``c2st_auc`` with a ridge
    classifier. ``feat_real`` lets callers reuse the real features across
    galleries. Returns the mean and spread of the out-of-fold AUC.
    """
    from anogen.shell.realism import _balanced_channel_indices

    fr = space.features(u_real) if feat_real is None else np.asarray(feat_real)
    fs = space.features(u_synth)
    rc, sc = np.asarray(real_ch, dtype=int), np.asarray(synth_ch, dtype=int)
    rg, sg = np.asarray(real_groups), np.asarray(synth_groups)
    aucs, bals = [], []
    for s in seeds:
        ri, si = _balanced_channel_indices(rc, sc, np.random.default_rng(int(s)))
        r = c2st_auc(fr[ri], fs[si], rg[ri], sg[si], seed=int(s), classifier="ridge")
        aucs.append(r["auc"])
        bals.append(r["balanced_accuracy"])
    return {
        "auc": float(np.mean(aucs)),
        "auc_min": float(np.min(aucs)),
        "auc_max": float(np.max(aucs)),
        "balanced_accuracy": float(np.mean(bals)),
        "n_seeds": float(len(aucs)),
    }
