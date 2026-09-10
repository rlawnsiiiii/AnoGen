"""Panel and label I/O with a sealed-test telemetry guard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from anogen.shell.events import naive_utc


class SealedTelemetryAccessError(RuntimeError):
    """Raised when shell code would read sealed-test channel values."""


@dataclass
class Panel:
    Y: np.ndarray
    counts: np.ndarray
    grid: np.ndarray
    channels: list[str]
    bin_seconds: int
    official_train_end: np.datetime64
    allow_test: bool

    @property
    def T(self) -> int:
        return int(self.Y.shape[0])

    @property
    def k(self) -> int:
        return int(self.Y.shape[1])

    def channel_index(self, name: str) -> int:
        return self.channels.index(name)

    def assert_span_allowed(self, start: int, width: int) -> None:
        if start < 0 or start + width > self.T:
            raise IndexError(f"window [{start}, {start + width}) outside panel T={self.T}")
        if self.allow_test:
            return
        last = self.grid[start + width - 1]
        if last >= self.official_train_end:
            raise SealedTelemetryAccessError(
                "Refusing a window that reaches official_train_end="
                f"{self.official_train_end}. Set allow_test_telemetry=true only "
                "for the sealed test."
            )


def load_metadata(root: Path) -> dict[str, pd.DataFrame]:
    root = Path(root)
    if not (root / "labels.csv").exists():
        raise FileNotFoundError(f"No labels.csv under {root}")
    labels = pd.read_csv(root / "labels.csv", parse_dates=["StartTime", "EndTime"])
    labels["StartTime"] = naive_utc(labels["StartTime"])
    labels["EndTime"] = naive_utc(labels["EndTime"])
    types = pd.read_csv(root / "anomaly_types.csv")
    return {"labels": labels, "anomaly_types": types, "base": root}


def load_panel(
    path: Path,
    *,
    channels: list[str],
    official_train_end: str | pd.Timestamp,
    allow_test_telemetry: bool = False,
    bin_seconds: int = 30,
) -> Panel:
    """Load panel_light.npz and crop any bins at or after the official cut."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    blob = np.load(path, allow_pickle=False)
    Y = np.asarray(blob["Y"])
    counts = np.asarray(blob["counts"])
    grid = np.asarray(blob["grid"]).astype("datetime64[ns]")
    if Y.ndim != 2 or Y.shape[1] != len(channels):
        raise ValueError(f"panel Y shape {Y.shape} does not match {len(channels)} channels")
    if len(grid) != Y.shape[0] or counts.shape != Y.shape:
        raise ValueError("panel Y, counts, and grid are misaligned")

    cut = np.datetime64(naive_utc(pd.Timestamp(official_train_end)), "ns")
    if not allow_test_telemetry:
        keep = grid < cut
        if not bool(keep.all()):
            Y = Y[keep]
            counts = counts[keep]
            grid = grid[keep]
        if grid.size and grid[-1] >= cut:
            raise SealedTelemetryAccessError(
                f"Panel still contains bins at or after official_train_end={cut}"
            )
    return Panel(
        Y=Y,
        counts=counts,
        grid=grid,
        channels=list(channels),
        bin_seconds=int(bin_seconds),
        official_train_end=cut,
        allow_test=bool(allow_test_telemetry),
    )
