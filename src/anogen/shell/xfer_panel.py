"""Build or slice a sealed-safe panel for transfer eval. Never reads past the cut."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anogen.shell.events import naive_utc


def slice_panel(
    src: Path,
    src_channels: list[str],
    keep: list[str],
    dest: Path,
) -> dict[str, Any]:
    """Copy selected columns from an existing Y/counts/grid npz."""
    missing = [ch for ch in keep if ch not in src_channels]
    if missing:
        raise ValueError(f"channels not in source panel: {missing}")
    blob = np.load(src, allow_pickle=False)
    idx = [src_channels.index(ch) for ch in keep]
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        Y=np.asarray(blob["Y"])[:, idx],
        counts=np.asarray(blob["counts"])[:, idx],
        grid=np.asarray(blob["grid"]).astype("datetime64[ns]"),
    )
    return {"n_bins": int(blob["Y"].shape[0]), "k": len(keep), "source": str(src)}


def bin_from_pickles(
    data_root: Path,
    channels: list[str],
    *,
    official_train_end: str,
    bin_seconds: int,
    dest: Path,
) -> dict[str, Any]:
    """Mean-bin analog pickles up to (not including) official_train_end.

    Two passes so only one raw series sits in memory at a time.
    """
    cut = naive_utc(pd.Timestamp(official_train_end))
    starts: list[pd.Timestamp] = []
    for ch in channels:
        print(f"xfer panel: scan {ch}", flush=True)
        s = _read_esa_channel(data_root / "channels" / f"{ch}.zip", ch)
        s = s.loc[s.index < cut]
        if s.empty:
            raise RuntimeError(f"{ch} has no samples before {cut}")
        starts.append(s.index.min())
        del s
    t0 = min(starts).floor(f"{int(bin_seconds)}s")
    grid = pd.date_range(t0, cut, freq=f"{int(bin_seconds)}s", inclusive="left")
    Y = np.full((len(grid), len(channels)), np.nan, dtype=np.float32)
    counts = np.zeros((len(grid), len(channels)), dtype=np.int16)
    for c, ch in enumerate(channels):
        print(f"xfer panel: bin {ch}", flush=True)
        s = _read_esa_channel(data_root / "channels" / f"{ch}.zip", ch)
        s = s.loc[s.index < cut]
        binned = s.resample(f"{int(bin_seconds)}s", origin=t0).mean()
        cnt = s.resample(f"{int(bin_seconds)}s", origin=t0).count()
        Y[:, c] = binned.reindex(grid).to_numpy(dtype=np.float32)
        counts[:, c] = cnt.reindex(grid).fillna(0).to_numpy(dtype=np.int16)
        del s
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        Y=Y,
        counts=counts,
        grid=grid.to_numpy(dtype="datetime64[ns]"),
    )
    return {
        "n_bins": int(len(grid)),
        "k": len(channels),
        "origin": str(t0),
        "bin_seconds": int(bin_seconds),
        "source": str(data_root),
    }


def write_panel_meta(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _read_esa_channel(path: Path, name: str) -> pd.Series:
    path = Path(path)
    if not path.is_file():
        alt = path.with_suffix(".pkl")
        if alt.is_file():
            path = alt
        else:
            raise FileNotFoundError(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            inner = zf.namelist()[0]
            with zf.open(inner) as fh:
                obj = pd.read_pickle(fh)
    else:
        obj = pd.read_pickle(path)
    if isinstance(obj, pd.Series):
        s = obj
    else:
        col = name if name in obj.columns else obj.columns[0]
        s = obj[col]
    s.index = naive_utc(pd.DatetimeIndex(s.index))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.astype("float64")
