"""Per-channel min-max on clean windows. Fit once on the train pool, invert after sampling.

Does not re-scale the noised x_t. Matches TS generation practice (CSDI z-score /
min-max, TSDiff MeanScaler): the diffusion prior is N(0, I), so x_0 should be O(1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_EPS = 1e-6


@dataclass
class ChannelMinMax:
    lo: np.ndarray
    hi: np.ndarray
    feature_range: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        self.lo = np.asarray(self.lo, dtype=np.float64).reshape(-1)
        self.hi = np.asarray(self.hi, dtype=np.float64).reshape(-1)
        if self.lo.shape != self.hi.shape:
            raise ValueError("lo and hi must have the same shape")
        span = np.maximum(self.hi - self.lo, _EPS)
        self.hi = self.lo + span
        a, b = self.feature_range
        self.feature_range = (float(a), float(b))

    @property
    def n_channels(self) -> int:
        return int(self.lo.size)

    def transform(self, x: np.ndarray, channel_idx: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        ch = np.asarray(channel_idx, dtype=np.int64)
        lo = self.lo[ch][:, None].astype(np.float32)
        span = (self.hi[ch] - self.lo[ch])[:, None].astype(np.float32)
        z = (x - lo) / span
        a, b = self.feature_range
        return (z * (b - a) + a).astype(np.float32)

    def inverse(self, y: np.ndarray, channel_idx: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32)
        ch = np.asarray(channel_idx, dtype=np.int64)
        a, b = self.feature_range
        z = (y - a) / (b - a)
        lo = self.lo[ch][:, None].astype(np.float32)
        span = (self.hi[ch] - self.lo[ch])[:, None].astype(np.float32)
        return (z * span + lo).astype(np.float32)

    def transform_torch(self, x: Any, channel_idx: Any) -> Any:
        import torch

        a, b = self.feature_range
        lo = torch.as_tensor(self.lo, device=x.device, dtype=x.dtype)[channel_idx]
        hi = torch.as_tensor(self.hi, device=x.device, dtype=x.dtype)[channel_idx]
        while lo.ndim < x.ndim:
            lo = lo.unsqueeze(-1)
            hi = hi.unsqueeze(-1)
        z = (x - lo) / (hi - lo)
        return z * (b - a) + a

    def inverse_torch(self, y: Any, channel_idx: Any) -> Any:
        import torch

        a, b = self.feature_range
        lo = torch.as_tensor(self.lo, device=y.device, dtype=y.dtype)[channel_idx]
        hi = torch.as_tensor(self.hi, device=y.device, dtype=y.dtype)[channel_idx]
        while lo.ndim < y.ndim:
            lo = lo.unsqueeze(-1)
            hi = hi.unsqueeze(-1)
        z = (y - a) / (b - a)
        return z * (hi - lo) + lo

    def save(self, path: Path) -> None:
        a, b = self.feature_range
        np.savez_compressed(
            path,
            lo=self.lo.astype(np.float64),
            hi=self.hi.astype(np.float64),
            feature_range=np.asarray([a, b], dtype=np.float64),
        )

    def to_ckpt(self) -> dict[str, Any]:
        a, b = self.feature_range
        return {
            "scaler_kind": "minmax",
            "scaler_lo": self.lo.astype(np.float64),
            "scaler_hi": self.hi.astype(np.float64),
            "scaler_range": [a, b],
        }

    def summary(self) -> dict[str, Any]:
        a, b = self.feature_range
        return {
            "kind": "minmax",
            "feature_range": [a, b],
            "n_channels": self.n_channels,
            "lo": [float(v) for v in self.lo],
            "hi": [float(v) for v in self.hi],
        }


def fit_channel_minmax(
    x: np.ndarray,
    channel_idx: np.ndarray,
    n_channels: int,
    *,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> ChannelMinMax:
    x = np.asarray(x, dtype=np.float64)
    ch = np.asarray(channel_idx, dtype=np.int64)
    lo = np.zeros(n_channels, dtype=np.float64)
    hi = np.ones(n_channels, dtype=np.float64)
    for c in range(n_channels):
        m = ch == c
        if not np.any(m):
            continue
        xc = x[m]
        lo[c] = float(np.min(xc))
        hi[c] = float(np.max(xc))
    return ChannelMinMax(lo=lo, hi=hi, feature_range=feature_range)


def load_minmax(path: Path) -> ChannelMinMax:
    blob = np.load(path)
    fr = blob["feature_range"]
    return ChannelMinMax(lo=blob["lo"], hi=blob["hi"], feature_range=(float(fr[0]), float(fr[1])))


def scaler_from_ckpt(ckpt: dict[str, Any] | None) -> ChannelMinMax | None:
    if not ckpt or ckpt.get("scaler_kind") != "minmax":
        return None
    lo, hi = ckpt.get("scaler_lo"), ckpt.get("scaler_hi")
    if lo is None or hi is None:
        return None
    rng = ckpt.get("scaler_range") or (0.0, 1.0)
    return ChannelMinMax(lo=np.asarray(lo), hi=np.asarray(hi), feature_range=(float(rng[0]), float(rng[1])))


def resolve_scaler(ckpt: dict[str, Any] | None = None, s0: Path | None = None) -> ChannelMinMax | None:
    """Prefer the scaler stored with the denoiser; fall back to S0's fit."""
    scaler = scaler_from_ckpt(ckpt)
    if scaler is not None:
        return scaler
    if s0 is not None:
        path = Path(s0) / "minmax_scaler.npz"
        if path.is_file():
            return load_minmax(path)
    return None
