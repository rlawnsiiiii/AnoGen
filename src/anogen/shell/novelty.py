"""Novelty filter for generated positives (numpy + scikit-learn, no torch).

A generated "anomaly" that a nominal-only model cannot tell from nominal
windows is a mislabelled negative. At a strict false-alarm budget such
positives cost recall: in the testbed (docs/TESTBED.md, E20) a gallery of
regenerated donors with no objective (the no-steer twin) more than halves the
recall of a detector trained on it, and dropping every generated window that
lies inside the nominal set restores it exactly.

``KnnNovelty`` is that test. It embeds windows with random convolutions
(``rocket.RocketSpace``, fitted on the nominal reference windows), scores each
window by its mean distance to the k nearest reference windows, and sets the
threshold at the q-quantile of the same score on held-out nominal windows. A
generated window is kept if it scores above the threshold, i.e. if a
nominal-only detector would flag it at a false-alarm rate of 1 − q. The share
kept is itself a useful number for a gallery: it is the fraction of generated
anomalies that are anomalous at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class KnnNovelty:
    k: int = 5
    q: float = 0.99
    n_kernels: int = 300
    n_components: int = 64
    seed: int = 0
    space_: Any = field(default=None, init=False, repr=False)
    nn_: Any = field(default=None, init=False, repr=False)
    threshold_: float = field(default=float("nan"), init=False)

    def fit(self, x_ref: np.ndarray, x_cal: np.ndarray) -> KnnNovelty:
        """x_ref: nominal windows the score is measured against; x_cal: other
        nominal windows (never in x_ref) that set the threshold."""
        from sklearn.neighbors import NearestNeighbors

        from anogen.shell.rocket import RocketSpace

        x_ref = np.asarray(x_ref, dtype=np.float64)
        x_cal = np.asarray(x_cal, dtype=np.float64)
        if len(x_ref) <= self.k or len(x_cal) == 0:
            raise ValueError("KnnNovelty needs more than k reference windows and at least one calibration window")
        self.space_ = RocketSpace(
            n_kernels=int(self.n_kernels), n_components=int(self.n_components), seed=int(self.seed)
        ).fit(x_ref)
        self.nn_ = NearestNeighbors(n_neighbors=int(self.k)).fit(self.space_.embed(x_ref))
        self.threshold_ = float(np.quantile(self.score(x_cal), float(self.q)))
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.nn_ is None:
            raise RuntimeError("call fit() first")
        d, _ = self.nn_.kneighbors(self.space_.embed(np.asarray(x, dtype=np.float64)))
        return d.mean(axis=1)

    def keep(self, x: np.ndarray) -> np.ndarray:
        """Boolean mask: windows outside the nominal set (score above the threshold)."""
        x = np.asarray(x)
        if len(x) == 0:
            return np.zeros(0, dtype=bool)
        return self.score(x) > self.threshold_


def novelty_from_cfg(cfg: dict[str, Any] | None, *, seed: int = 0) -> KnnNovelty | None:
    """``augdetect.synth_filter`` → an unfitted filter, or None when off.

    Accepted: ``{kind: rocket_knn, q: 0.99, k: 5, kernels: 300, components: 64}``
    (every key but ``kind`` optional); ``null`` / missing / ``kind: none`` = off.
    """
    if not cfg:
        return None
    kind = str(cfg.get("kind", "rocket_knn")).lower()
    if kind in {"none", "off", "false"}:
        return None
    if kind != "rocket_knn":
        raise ValueError(f"unknown synth_filter kind {kind!r} (use rocket_knn)")
    return KnnNovelty(
        k=int(cfg.get("k", 5)),
        q=float(cfg.get("q", 0.99)),
        n_kernels=int(cfg.get("kernels", 300)),
        n_components=int(cfg.get("components", 64)),
        seed=int(cfg.get("seed", seed)),
    )
