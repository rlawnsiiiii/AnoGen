"""Event table and official ESA splits. Label metadata only — no telemetry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ESA_TYPE_COLS = (
    "esa_class",
    "esa_subclass",
    "esa_dimensionality",
    "esa_locality",
    "esa_length",
)


def load_anomaly_types(root: str | Path) -> pd.DataFrame:
    """Read official ``anomaly_types.csv`` (ID, Category, Length, Locality, …)."""
    path = Path(root) / "anomaly_types.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def types_from_cfg(cfg: dict) -> pd.DataFrame:
    """``anomaly_types.csv`` under the configured ESA mission folder."""
    from anogen.config import data_root

    return load_anomaly_types(data_root(cfg))


def try_types_from_cfg(cfg: dict) -> pd.DataFrame | None:
    """Same as ``types_from_cfg``, or ``None`` if the CSV is missing."""
    try:
        return types_from_cfg(cfg)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None


def _first_str(g: pd.DataFrame, col: str, default: str = "") -> str:
    if col not in g.columns:
        return default
    v = g[col].iloc[0]
    try:
        if v is None or pd.isna(v):
            return default
    except (TypeError, ValueError):
        if v is None:
            return default
    return str(v).strip()


def naive_utc(values):
    """Strip timezone, converting to UTC first so instants stay aligned."""
    if isinstance(values, pd.Timestamp):
        t = pd.Timestamp(values)
        if t.tzinfo is not None:
            return t.tz_convert("UTC").tz_localize(None)
        return t
    if isinstance(values, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(values)
        if idx.tz is not None:
            return idx.tz_convert("UTC").tz_localize(None)
        return idx
    s = pd.to_datetime(values)
    tz = getattr(s.dtype, "tz", None)
    if tz is not None:
        return s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s


def build_event_table(labels: pd.DataFrame, anomaly_types: pd.DataFrame) -> pd.DataFrame:
    """One row per event ID with category, span, and annotated channels."""
    types = anomaly_types.copy()
    if "ID" not in types.columns:
        raise ValueError("anomaly_types.csv must have an ID column")
    merged = labels.merge(types, on="ID", how="left", suffixes=("", "_type"))
    if "Category" not in merged.columns:
        raise ValueError("anomaly_types.csv must have Category")

    rows = []
    for event_id, g in merged.groupby("ID", sort=False):
        start = naive_utc(pd.Timestamp(g["StartTime"].min()))
        end = naive_utc(pd.Timestamp(g["EndTime"].max()))
        cat = g["Category"].iloc[0]
        channels = tuple(sorted(g["Channel"].astype(str).unique()))
        rows.append(
            {
                "event_id": event_id,
                "category": cat,
                "start": start,
                "end": end,
                "n_channels": len(channels),
                "channels": channels,
                "is_anomaly": cat == "Anomaly",
                "is_rare": cat == "Rare Event",
                "esa_class": _first_str(g, "Class"),
                "esa_subclass": _first_str(g, "Subclass"),
                "esa_dimensionality": _first_str(g, "Dimensionality"),
                "esa_locality": _first_str(g, "Locality"),
                "esa_length": _first_str(g, "Length"),
            }
        )
    events = pd.DataFrame(rows)
    return events.sort_values("start").reset_index(drop=True)


def assign_splits(
    events: pd.DataFrame,
    *,
    path_train_end: pd.Timestamp | str,
    official_train_end: pd.Timestamp | str,
) -> pd.DataFrame:
    """Add split: path_train, pretest, test, or boundary."""
    path_train_end = naive_utc(pd.Timestamp(path_train_end))
    official_train_end = naive_utc(pd.Timestamp(official_train_end))
    out = events.copy()
    start = naive_utc(pd.to_datetime(out["start"]))
    end = naive_utc(pd.to_datetime(out["end"]))
    out["start"] = start
    out["end"] = end
    is_pretest = end < official_train_end
    is_test = start > official_train_end
    split = np.full(len(out), "boundary", dtype=object)
    split[is_pretest.to_numpy()] = "pretest"
    split[is_test.to_numpy()] = "test"
    in_path = is_pretest & (end < path_train_end)
    split[in_path.to_numpy()] = "path_train"
    out["split"] = split
    out["in_pretest_pool"] = is_pretest.to_numpy()
    out["in_test_pool"] = is_test.to_numpy()
    out["in_path_train"] = in_path.to_numpy()
    return out


def panel_pairs(events: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """One row per (event_id, channel) that the event annotates on the panel."""
    panel = set(channels)
    rows = []
    for _, ev in events.iterrows():
        for ch in ev["channels"]:
            if ch not in panel:
                continue
            rows.append(
                {
                    "event_id": ev["event_id"],
                    "channel": ch,
                    "category": ev["category"],
                    "is_anomaly": bool(ev["is_anomaly"]),
                    "is_rare": bool(ev["is_rare"]),
                    "start": ev["start"],
                    "end": ev["end"],
                    "split": ev.get("split", ""),
                    "in_pretest_pool": bool(ev.get("in_pretest_pool", False)),
                    "in_test_pool": bool(ev.get("in_test_pool", False)),
                    "esa_class": ev.get("esa_class", ""),
                    "esa_subclass": ev.get("esa_subclass", ""),
                    "esa_dimensionality": ev.get("esa_dimensionality", ""),
                    "esa_locality": ev.get("esa_locality", ""),
                    "esa_length": ev.get("esa_length", ""),
                }
            )
    return pd.DataFrame(rows)
