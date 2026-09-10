"""Transfer-eval helpers: named keep-mask and sealed panel build."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from anogen.shell.folds import keep_fold_mask
from anogen.shell.xfer_panel import bin_from_pickles, slice_panel


def test_keep_fold_mask_uses_actual_channel_names() -> None:
    tab = pd.DataFrame(
        {
            "channel": ["channel_9", "channel_15", "channel_9"],
            "fold": [0, 0, 1],
            "n_anomaly_windows": [1, 8, 8],
            "n_rare_windows": [0, 0, 0],
            "below_min": [True, False, False],
        }
    )
    channels = ["channel_9", "channel_15"]
    ch_idx = np.array([0, 1, 0, 1])
    folds = np.array([0, 0, 1, 0])
    keep = keep_fold_mask(ch_idx, folds, tab, fold_id=0, channels=channels)
    # drop channel_9 fold 0; keep channel_15 fold 0 (both windows)
    assert keep.tolist() == [False, True, False, True]


def test_slice_panel_selects_named_columns(tmp_path: Path) -> None:
    src = tmp_path / "src.npz"
    T = 12
    grid = np.arange("2001-01-01", "2001-01-13", dtype="datetime64[D]").astype("datetime64[ns]")
    Y = np.arange(T * 3, dtype=np.float32).reshape(T, 3)
    counts = np.ones((T, 3), dtype=np.int16)
    np.savez_compressed(src, Y=Y, counts=counts, grid=grid)
    dest = tmp_path / "out.npz"
    info = slice_panel(src, ["channel_a", "channel_b", "channel_c"], ["channel_c", "channel_a"], dest)
    blob = np.load(dest)
    assert info["k"] == 2
    np.testing.assert_array_equal(blob["Y"][:, 0], Y[:, 2])
    np.testing.assert_array_equal(blob["Y"][:, 1], Y[:, 0])


def test_bin_from_pickles_stops_before_official_cut(tmp_path: Path) -> None:
    root = tmp_path / "ESA-MissionX"
    ch_dir = root / "channels"
    ch_dir.mkdir(parents=True)
    idx = pd.date_range("2001-01-01", periods=20, freq="30s")
    values = np.linspace(0.0, 1.0, 20)
    values[-3:] = 99.0
    _write_zip_series(ch_dir / "channel_9.zip", "channel_9", idx, values)
    dest = tmp_path / "panel.npz"
    info = bin_from_pickles(
        root,
        ["channel_9"],
        official_train_end="2001-01-01 00:07:00",
        bin_seconds=30,
        dest=dest,
    )
    blob = np.load(dest)
    grid = blob["grid"].astype("datetime64[ns]")
    cut = np.datetime64("2001-01-01T00:07:00")
    assert info["k"] == 1
    assert bool((grid < cut).all())
    assert float(np.nanmax(blob["Y"])) < 90.0


def _write_zip_series(path: Path, name: str, idx: pd.DatetimeIndex, values: np.ndarray) -> None:
    df = pd.DataFrame({name: values}, index=idx)
    inner = path.with_suffix(".pkl")
    df.to_pickle(inner)
    with zipfile.ZipFile(path, "w") as zf:
        zf.write(inner, arcname=inner.name)
    inner.unlink()
