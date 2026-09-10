"""Univariate window index tables for the shell track."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anogen.shell.data import Panel, SealedTelemetryAccessError
from anogen.shell.events import naive_utc


def time_to_index(grid: np.ndarray, ts, *, side: str = "left") -> int:
    t = np.datetime64(naive_utc(pd.Timestamp(ts)), "ns")
    return int(np.searchsorted(grid, t, side=side))


def window_finite(Y: np.ndarray, counts: np.ndarray, start: int, ch: int, width: int) -> bool:
    sl = slice(start, start + width)
    return bool(np.isfinite(Y[sl, ch]).all() and (counts[sl, ch] > 0).all())


TRAIN_COLS = [
    "event_id",
    "channel",
    "channel_idx",
    "start",
    "kind",
    "is_anomaly",
    "is_rare",
    "split",
    "placement",
]


def occupancy_mask(
    panel: Panel,
    events: pd.DataFrame,
    *,
    guard_bins: int,
    occupy_rares: bool = False,
) -> np.ndarray:
    """True on bins that must not be everyday-nominal samples.

    Anomaly spans (plus a guard) are always occupied. Rare Events are valid
    ops: leave them off the mask when they are part of the train pool
    (``occupy_rares=False``). Set ``occupy_rares=True`` for the old
    everyday-only corpus.
    """
    occ = np.zeros(panel.T, dtype=bool)
    for _, ev in events.iterrows():
        if bool(ev["is_anomaly"]):
            pass
        elif occupy_rares and bool(ev["is_rare"]):
            pass
        else:
            continue
        if bool(ev.get("in_test_pool", False)) and not panel.allow_test:
            continue
        lo = time_to_index(panel.grid, ev["start"], side="left")
        hi = time_to_index(panel.grid, ev["end"], side="right")
        if hi <= lo:
            hi = lo + 1
        lo = max(0, lo - int(guard_bins))
        hi = min(panel.T, hi + int(guard_bins))
        occ[lo:hi] = True
    return occ


def _even_starts(lo: int, hi: int, width: int, max_crops: int) -> list[int]:
    span = hi - lo
    if span < width:
        return []
    n_possible = 1 + (span - width) // width
    if n_possible <= max_crops:
        return [lo + i * width for i in range(n_possible)]
    if max_crops == 1:
        return [lo]
    step = (span - width) / (max_crops - 1)
    return [int(round(lo + i * step)) for i in range(max_crops)]


def extract_labeled_windows(
    panel: Panel,
    pairs: pd.DataFrame,
    *,
    width: int,
    max_crops_per_event: int,
    allow_test_telemetry: bool,
) -> pd.DataFrame:
    """Windows for (event, channel) pairs. Short events are centered in W."""
    rows: list[dict[str, Any]] = []
    sealed_skipped = 0
    for _, rec in pairs.iterrows():
        if bool(rec.get("in_test_pool", False)) and not allow_test_telemetry:
            sealed_skipped += 1
            continue
        ch = str(rec["channel"])
        try:
            cidx = panel.channel_index(ch)
        except ValueError:
            continue
        lo = time_to_index(panel.grid, rec["start"], side="left")
        hi = time_to_index(panel.grid, rec["end"], side="right")
        if hi <= lo:
            hi = min(panel.T, lo + 1)
        lo = min(max(lo, 0), panel.T)
        hi = min(max(hi, 0), panel.T)
        span = hi - lo
        if span >= width:
            starts = _even_starts(lo, hi, width, max_crops_per_event)
            placement = "crop"
        else:
            mid = (lo + hi) // 2
            start = int(mid - width // 2)
            start = max(0, min(start, panel.T - width))
            starts = [start] if start >= 0 and start + width <= panel.T else []
            placement = "center"
        for start in starts:
            try:
                panel.assert_span_allowed(start, width)
            except SealedTelemetryAccessError:
                sealed_skipped += 1
                continue
            except IndexError:
                continue
            if not window_finite(panel.Y, panel.counts, start, cidx, width):
                continue
            rows.append(
                {
                    "event_id": rec["event_id"],
                    "channel": ch,
                    "channel_idx": cidx,
                    "start": start,
                    "kind": "anomaly" if rec["is_anomaly"] else ("rare" if rec["is_rare"] else "other"),
                    "is_anomaly": bool(rec["is_anomaly"]),
                    "is_rare": bool(rec["is_rare"]),
                    "split": rec.get("split", ""),
                    "placement": placement,
                    "label_lo": lo,
                    "label_hi": hi,
                    "esa_class": rec.get("esa_class", ""),
                    "esa_subclass": rec.get("esa_subclass", ""),
                    "esa_dimensionality": rec.get("esa_dimensionality", ""),
                    "esa_locality": rec.get("esa_locality", ""),
                    "esa_length": rec.get("esa_length", ""),
                }
            )
    cols = [
        "event_id",
        "channel",
        "channel_idx",
        "start",
        "kind",
        "is_anomaly",
        "is_rare",
        "split",
        "placement",
        "label_lo",
        "label_hi",
        "esa_class",
        "esa_subclass",
        "esa_dimensionality",
        "esa_locality",
        "esa_length",
    ]
    out = pd.DataFrame(rows, columns=cols)
    out.attrs["sealed_skipped"] = sealed_skipped
    return out


def sample_nominal(
    panel: Panel,
    occ: np.ndarray,
    *,
    width: int,
    n_per_channel: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Rejection-sample nominal windows that miss occupied bins and NaNs."""
    cols = [
        "event_id",
        "channel",
        "channel_idx",
        "start",
        "kind",
        "is_anomaly",
        "is_rare",
        "split",
        "placement",
    ]
    rows: list[dict[str, Any]] = []
    max_tries = max(n_per_channel * 80, 200)
    limit = panel.T - width + 1
    if limit <= 0:
        return pd.DataFrame(rows, columns=cols)
    for cidx, ch in enumerate(panel.channels):
        seen: set[int] = set()
        tries = 0
        while len(seen) < n_per_channel and tries < max_tries:
            tries += 1
            start = int(rng.integers(0, limit))
            if start in seen:
                continue
            if occ[start : start + width].any():
                continue
            try:
                panel.assert_span_allowed(start, width)
            except (SealedTelemetryAccessError, IndexError):
                continue
            if not window_finite(panel.Y, panel.counts, start, cidx, width):
                continue
            seen.add(start)
            rows.append(
                {
                    "event_id": "",
                    "channel": ch,
                    "channel_idx": cidx,
                    "start": start,
                    "kind": "nominal",
                    "is_anomaly": False,
                    "is_rare": False,
                    "split": "nominal",
                    "placement": "sample",
                }
            )
    return pd.DataFrame(rows, columns=cols)


def materialize(panel: Panel, table: pd.DataFrame, width: int) -> np.ndarray:
    """Stack univariate windows. Empty table → (0, W)."""
    if table is None or len(table) == 0:
        return np.zeros((0, width), dtype=np.float32)
    x = np.empty((len(table), width), dtype=np.float32)
    starts = table["start"].to_numpy()
    chs = table["channel_idx"].to_numpy()
    for i, (s, c) in enumerate(zip(starts, chs, strict=True)):
        panel.assert_span_allowed(int(s), width)
        x[i] = panel.Y[int(s) : int(s) + width, int(c)]
    return x


def compose_train_index(
    nominal: pd.DataFrame,
    rare: pd.DataFrame,
    *,
    upsample: int = 1,
) -> pd.DataFrame:
    """Everyday nominals plus rare windows. ``upsample`` copies rares (later)."""
    parts = [nominal.loc[:, [c for c in TRAIN_COLS if c in nominal.columns]].copy()]
    if rare is not None and len(rare):
        cols = [c for c in TRAIN_COLS if c in rare.columns]
        r = rare.loc[:, cols].copy()
        r["kind"] = "rare_train"
        r["is_rare"] = True
        r["is_anomaly"] = False
        copies = max(1, int(upsample))
        parts.extend(r for _ in range(copies))
    return pd.concat(parts, ignore_index=True)


def load_train_index(s0: Path) -> pd.DataFrame:
    """Prefer train_index.csv (nominals + rares). Fall back to everyday-only."""
    train = Path(s0) / "train_index.csv"
    if train.is_file():
        return pd.read_csv(train)
    return pd.read_csv(Path(s0) / "nominal_index.csv")
