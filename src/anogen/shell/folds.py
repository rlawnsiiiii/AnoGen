"""Event-level out-of-fold assignment. All windows of an event share a fold."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold


def assign_event_folds(
    labeled: pd.DataFrame,
    *,
    n_folds: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Assign folds to development labeled windows via unique event_id."""
    out = labeled.copy()
    out["fold"] = -1
    if len(out) == 0:
        return out
    events = (
        out.groupby("event_id", sort=True)
        .agg(is_anomaly=("is_anomaly", "max"))
        .reset_index()
    )
    ids = events["event_id"].to_numpy()
    y = events["is_anomaly"].astype(int).to_numpy()
    n = len(ids)
    folds = np.full(n, -1, dtype=int)
    if n < 2:
        folds[:] = 0
    else:
        n_splits = min(int(n_folds), n)
        stratify = y.min() != y.max() and int(y.sum()) >= n_splits and int((1 - y).sum()) >= n_splits
        splitter: StratifiedKFold | KFold
        if stratify:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            split_y = y
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            split_y = None
        dummy = np.zeros(n)
        for fold, (_, test_idx) in enumerate(splitter.split(dummy, split_y)):
            folds[test_idx] = fold
    ev_fold = dict(zip(ids.tolist(), folds.tolist(), strict=True))
    out["fold"] = out["event_id"].map(ev_fold).fillna(-1).astype(int)
    return out


def channel_fold_counts(labeled: pd.DataFrame, *, min_anomaly_windows: int) -> pd.DataFrame:
    """Per-channel, per-fold anomaly window counts and drop flags."""
    if labeled is None or len(labeled) == 0:
        return pd.DataFrame(
            columns=["channel", "fold", "n_anomaly_windows", "n_rare_windows", "below_min"]
        )
    rows = []
    channels = sorted(labeled["channel"].unique())
    folds = sorted(int(f) for f in labeled["fold"].unique() if int(f) >= 0)
    for ch in channels:
        for fold in folds:
            part = labeled[(labeled["channel"] == ch) & (labeled["fold"] == fold)]
            n_a = int(part["is_anomaly"].sum())
            n_r = int(part["is_rare"].sum())
            rows.append(
                {
                    "channel": ch,
                    "fold": fold,
                    "n_anomaly_windows": n_a,
                    "n_rare_windows": n_r,
                    "below_min": n_a < int(min_anomaly_windows),
                }
            )
    return pd.DataFrame(rows)


def keep_fold_mask(
    channel_idx: np.ndarray,
    folds: np.ndarray,
    tab: pd.DataFrame,
    *,
    fold_id: int,
    channels: list[str],
) -> np.ndarray:
    """Keep windows in ``fold_id`` whose channel is not below-min in that fold.

    Uses the actual panel channel names. Do not assume Mission 1 ``channel_41+c``.
    """
    dropped = set(
        tab.loc[(tab["fold"] == fold_id) & (tab["below_min"]), "channel"].astype(str)
    )
    names = np.array([channels[int(c)] for c in np.asarray(channel_idx)])
    return (np.asarray(folds) == int(fold_id)) & np.array([n not in dropped for n in names])
