"""Plot helpers write PNGs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from anogen.shell.plots import (
    save_compare_rows,
    save_family_grid,
    save_named_grid,
    save_same_parent,
    save_strip,
)


def test_plot_helpers_write_files(tmp_path: Path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 32)).astype(np.float32)
    save_family_grid(
        tmp_path / "fam.png",
        {"anomaly": x, "nominal": x},
        bin_seconds=30,
        rng=rng,
        n_cols=3,
    )
    save_strip(
        tmp_path / "strip.png",
        x,
        title="t",
        color="#111111",
        bin_seconds=30,
        idx=np.arange(4),
    )
    save_same_parent(
        tmp_path / "same.png",
        x[:2],
        {"genias": x[2:4]},
        bin_seconds=30,
    )
    assert (tmp_path / "fam.png").is_file()
    assert (tmp_path / "strip.png").is_file()
    assert (tmp_path / "same.png").is_file()
    save_compare_rows(
        tmp_path / "cmp.png",
        {"real anomaly": x[:3], "genias": x[3:6]},
        bin_seconds=30,
        n_cols=3,
    )
    assert (tmp_path / "cmp.png").is_file()
    save_named_grid(
        tmp_path / "grid.png",
        {("real", "shift"): x[0], ("gen", "shift"): x[1]},
        row_names=["real", "gen"],
        col_names=["shift"],
        bin_seconds=30,
        share_col_ylim=True,
    )
    assert (tmp_path / "grid.png").is_file()
