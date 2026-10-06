"""Dev-only anomaly detector: scores, not accuracies. Does not edit S5 train_classifier.

Windows are min-max scaled with the S0 ChannelMinMax (level shifts keep DC).
Do not per-window z-score.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from anogen.shell.diffusion import torch_available

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:  # pragma: no cover
    average_precision_score = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]


def threshold_at_far(neg_scores: np.ndarray, far: float = 0.01) -> float:
    """Score threshold with empirical false-alarm rate ≈ far on negatives."""
    neg = np.asarray(neg_scores, dtype=np.float64)
    neg = neg[np.isfinite(neg)]
    if neg.size == 0:
        return 0.0
    far = float(np.clip(far, 0.0, 1.0))
    if far <= 0.0:
        return float(np.max(neg) + 1e-12)
    if far >= 1.0:
        return float(np.min(neg) - 1e-12)
    return float(np.quantile(neg, 1.0 - far))


def event_hits(
    scores: np.ndarray,
    event_id: np.ndarray,
    threshold: float,
) -> dict[str, bool]:
    """An event is detected if any of its windows scores at or above threshold."""
    out: dict[str, bool] = {}
    by: dict[str, list[float]] = defaultdict(list)
    for s, ev in zip(np.asarray(scores, dtype=np.float64), np.asarray(event_id).astype(str), strict=True):
        by[ev].append(float(s))
    for ev, vals in by.items():
        out[ev] = bool(max(vals) >= threshold)
    return out


def event_recall(hits: dict[str, bool]) -> float:
    if not hits:
        return float("nan")
    return float(np.mean(list(hits.values())))


def bootstrap_event_recall(
    hits: dict[str, bool],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    ids = list(hits)
    if not ids or int(n_boot) <= 0:
        return {"median": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": len(ids)}
    vals = np.asarray([1.0 if hits[i] else 0.0 for i in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    stats = np.empty(int(n_boot), dtype=np.float64)
    n = len(vals)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        stats[b] = float(vals[idx].mean())
    return {
        "median": float(np.median(stats)),
        "lo": float(np.quantile(stats, 0.025)),
        "hi": float(np.quantile(stats, 0.975)),
        "point": float(vals.mean()),
        "n": int(n),
        "n_boot": int(n_boot),
    }


def fit_detector(
    x_pos: np.ndarray,
    x_neg: np.ndarray,
    *,
    seed: int = 0,
    steps: int = 400,
    hidden: int = 32,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str | None = None,
    shift_aug: int = 0,
    real_frac: float | None = None,
    n_real: int = 0,
) -> Any:
    """BCE, balanced batches. Returns a trained AnomalyClassifier.

    ``shift_aug`` > 0 translates every training window (both classes) by an
    independent U{-s..s} offset per batch: reflect-pad by s, then crop W at a
    random offset. Unlike a circular roll there is no wrap seam, which would
    turn a level shift's Δ into a fake step at the seam. 0 = unchanged.

    ``real_frac`` (with ``n_real`` > 0): the first ``n_real`` rows of ``x_pos``
    are real anomalies, the rest synthetic; every positive half-batch then
    draws round(real_frac · n_pos) rows from the real ones and the rest from
    the synthetic ones, instead of sampling all positives uniformly (where 256
    synthetic windows drown ~15 real ones). 0.5 gives both groups the same
    weight. None (default) keeps the uniform draw, bit-identical to before.
    """
    from anogen.shell.adapters import AnomalyClassifier
    import torch
    import torch.nn.functional as F

    if len(x_pos) < 1 or len(x_neg) < 1:
        raise ValueError("fit_detector needs at least one positive and one negative")
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    clf = AnomalyClassifier(hidden=int(hidden)).to(device_t)
    opt = torch.optim.Adam(clf.parameters(), lr=float(lr))
    xa = torch.from_numpy(np.asarray(x_pos, dtype=np.float32)).unsqueeze(1)
    xn = torch.from_numpy(np.asarray(x_neg, dtype=np.float32)).unsqueeze(1)
    n_pos = max(1, int(batch_size) // 2)
    n_neg = max(1, int(batch_size) - n_pos)
    split = real_frac is not None and 0 < int(n_real) < len(xa)
    if split:
        n_pos_real = min(max(int(round(float(real_frac) * n_pos)), 0), n_pos)
    clf.train()
    for _ in range(int(steps)):
        if split:
            ia = torch.cat(
                [
                    torch.randint(0, int(n_real), (n_pos_real,)),
                    torch.randint(int(n_real), len(xa), (n_pos - n_pos_real,)),
                ]
            )
        else:
            ia = torch.randint(0, len(xa), (n_pos,))
        inn = torch.randint(0, len(xn), (n_neg,))
        xb = torch.cat([xa[ia], xn[inn]], dim=0).to(device_t)
        if int(shift_aug) > 0:
            s_aug = min(int(shift_aug), xb.size(-1) - 1)
            padded = F.pad(xb, (s_aug, s_aug), mode="reflect")
            offs = torch.randint(0, 2 * s_aug + 1, (xb.size(0),))
            xb = torch.stack([padded[i, :, int(o) : int(o) + xb.size(-1)] for i, o in enumerate(offs)])
        yb = torch.cat(
            [
                torch.ones(n_pos, device=device_t),
                torch.zeros(n_neg, device=device_t),
            ]
        )
        loss = F.binary_cross_entropy_with_logits(clf(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    clf.eval()
    return clf


def predict_logits(
    clf: Any,
    x: np.ndarray,
    *,
    device: str | None = None,
    bsz: int = 256,
) -> np.ndarray:
    import torch

    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return np.zeros(0, dtype=np.float64)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = np.empty(len(x), dtype=np.float64)
    clf.eval()
    with torch.no_grad():
        for i in range(0, len(x), int(bsz)):
            sl = torch.from_numpy(x[i : i + int(bsz)]).unsqueeze(1).to(device_t)
            out[i : i + int(bsz)] = clf(sl).detach().cpu().numpy()
    return out


def eval_detector(
    clf: Any,
    *,
    x_anom: np.ndarray,
    x_nom: np.ndarray,
    x_rare: np.ndarray,
    event_id: np.ndarray,
    kind: np.ndarray,
    far: float = 0.01,
    device: str | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Logits plus AP, AUROC, recall@FAR, event recall, Rare-Event FAR, per-kind recall."""
    s_a = predict_logits(clf, x_anom, device=device)
    s_n = predict_logits(clf, x_nom, device=device)
    s_r = predict_logits(clf, x_rare, device=device) if len(x_rare) else np.zeros(0)
    t = threshold_at_far(s_n, far)
    actual_far = float(np.mean(s_n >= t)) if len(s_n) else float("nan")
    win_rec = float(np.mean(s_a >= t)) if len(s_a) else float("nan")
    hits = event_hits(s_a, event_id, t)
    rare_far = float(np.mean(s_r >= t)) if len(s_r) else float("nan")
    kind = np.asarray(kind).astype(str)
    per_kind: dict[str, float] = {}
    for k in sorted(set(kind.tolist())):
        m = kind == k
        per_kind[k] = float(np.mean(s_a[m] >= t)) if int(m.sum()) else float("nan")
    y = np.concatenate([np.ones(len(s_a)), np.zeros(len(s_n))])
    s = np.concatenate([s_a, s_n])
    ap = auroc = float("nan")
    if average_precision_score is not None and len(s_a) and len(s_n):
        ap = float(average_precision_score(y, s))
        if len(set(y.tolist())) > 1:
            auroc = float(roc_auc_score(y, s))
    return {
        "threshold": t,
        "target_far": float(far),
        "nominal_far": actual_far,
        "window_recall": win_rec,
        "event_recall": event_recall(hits),
        "event_hits": hits,
        "rare_far": rare_far,
        "ap": ap,
        "auroc": auroc,
        "per_kind_recall": per_kind,
        "n_anomaly": int(len(s_a)),
        "n_nominal": int(len(s_n)),
        "n_rare": int(len(s_r)),
        "n_events": int(len(hits)),
        "bootstrap": bootstrap_event_recall(hits, n_boot=n_boot, seed=seed),
    }


def paired_event_bootstrap(
    hits_a: dict[int, dict[str, bool]],
    hits_b: dict[int, dict[str, bool]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Event recall(A) − recall(B), paired by seed and event, bootstrap over events.

    Each event's hit rate is averaged over the seeds both arms share; events
    are then resampled with replacement. ``p_le_zero`` is the share of
    replicates with no improvement (one-sided bootstrap p for "A > B").
    """
    seeds = sorted(set(hits_a) & set(hits_b))
    events = sorted(set.intersection(*[set(hits_a[s]) & set(hits_b[s]) for s in seeds])) if seeds else []
    if not events:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_le_zero": float("nan"), "n_events": 0}
    a = np.array([[float(hits_a[s][e]) for s in seeds] for e in events]).mean(axis=1)
    b = np.array([[float(hits_b[s][e]) for s in seeds] for e in events]).mean(axis=1)
    d = a - b
    rng = np.random.default_rng(int(seed))
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(int(n_boot))])
    return {
        "diff": float(d.mean()),
        "lo": float(np.quantile(boot, 0.025)),
        "hi": float(np.quantile(boot, 0.975)),
        "p_le_zero": float(np.mean(boot <= 0.0)),
        "n_events": int(len(events)),
        "n_seeds": int(len(seeds)),
    }


def needs_torch() -> bool:
    return bool(torch_available())


def eval_scores(
    s_anom: np.ndarray,
    s_nom: np.ndarray,
    s_rare: np.ndarray,
    *,
    event_id: np.ndarray,
    kind: np.ndarray,
    far: float = 0.01,
) -> dict[str, Any]:
    """Same metrics as ``eval_detector`` for precomputed scores (higher = more anomalous)."""
    s_a = np.asarray(s_anom, dtype=np.float64)
    s_n = np.asarray(s_nom, dtype=np.float64)
    s_r = np.asarray(s_rare, dtype=np.float64)
    t = threshold_at_far(s_n, far)
    hits = event_hits(s_a, event_id, t)
    kind = np.asarray(kind).astype(str)
    per_kind = {k: float(np.mean(s_a[kind == k] >= t)) for k in sorted(set(kind.tolist()))}
    y = np.concatenate([np.ones(len(s_a)), np.zeros(len(s_n))])
    s = np.concatenate([s_a, s_n])
    ap = auroc = float("nan")
    if average_precision_score is not None and len(s_a) and len(s_n):
        ap = float(average_precision_score(y, s))
        auroc = float(roc_auc_score(y, s))
    return {
        "threshold": t,
        "nominal_far": float(np.mean(s_n >= t)) if len(s_n) else float("nan"),
        "window_recall": float(np.mean(s_a >= t)) if len(s_a) else float("nan"),
        "event_recall": event_recall(hits),
        "event_hits": hits,
        "rare_far": float(np.mean(s_r >= t)) if len(s_r) else float("nan"),
        "ap": ap,
        "auroc": auroc,
        "per_kind_recall": per_kind,
    }


def rank_fuse(*scores: np.ndarray, reference: list[np.ndarray] | None = None) -> np.ndarray:
    """Average of per-score empirical CDF values (rank fusion).

    Each score is mapped through the empirical CDF of its own ``reference``
    sample (e.g. the scores of training nominals), so detectors on different
    scales are combined without fitting anything on test data. Without a
    reference the score's own ranks are used.
    """
    out = np.zeros(len(np.asarray(scores[0])), dtype=np.float64)
    for i, s in enumerate(scores):
        s = np.asarray(s, dtype=np.float64)
        ref = np.sort(np.asarray(reference[i] if reference is not None else s, dtype=np.float64))
        out += np.searchsorted(ref, s, side="right") / max(len(ref), 1)
    return out / len(scores)
