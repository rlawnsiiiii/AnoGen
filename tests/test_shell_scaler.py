"""Per-channel min-max: invertibility and a two-step denoiser train."""

from __future__ import annotations

import numpy as np
import pytest

from anogen.shell.diffusion import torch_available
from anogen.shell.scaler import (
    ChannelMinMax,
    fit_channel_minmax,
    load_minmax,
    resolve_scaler,
    scaler_from_ckpt,
)


def test_minmax_numpy_roundtrip(tmp_path):
    x = np.array([[0.0, 1.0, 2.0], [10.0, 20.0, 30.0]], dtype=np.float32)
    ch = np.array([0, 1], dtype=np.int64)
    sc = fit_channel_minmax(x, ch, 2, feature_range=(0.0, 1.0))
    y = sc.transform(x, ch)
    assert np.allclose(y[0, 0], 0.0) and np.allclose(y[0, -1], 1.0)
    assert np.allclose(y[1, 0], 0.0) and np.allclose(y[1, -1], 1.0)
    xr = sc.inverse(y, ch)
    assert np.allclose(xr, x, atol=1e-5)
    sc.save(tmp_path / "minmax_scaler.npz")
    loaded = load_minmax(tmp_path / "minmax_scaler.npz")
    assert np.allclose(loaded.inverse(loaded.transform(x, ch), ch), x, atol=1e-5)
    ckpt = {"hidden": 8, **sc.to_ckpt()}
    from_ckpt = scaler_from_ckpt(ckpt)
    assert from_ckpt is not None
    assert np.allclose(from_ckpt.lo, sc.lo)
    assert resolve_scaler(ckpt) is not None
    assert resolve_scaler({}, tmp_path) is not None
    assert resolve_scaler(None) is None


def test_minmax_constant_channel_does_not_nan():
    x = np.ones((4, 8), dtype=np.float32) * 0.8
    ch = np.zeros(4, dtype=np.int64)
    sc = fit_channel_minmax(x, ch, 1)
    y = sc.transform(x, ch)
    assert np.isfinite(y).all()
    assert np.allclose(sc.inverse(y, ch), x, atol=1e-5)


@pytest.mark.skipif(not torch_available(), reason="torch extra not installed")
def test_minmax_torch_roundtrip_and_two_step_train():
    import torch

    from anogen.shell.diffusion import train_denoiser, unguided_from_nominal

    rng = np.random.default_rng(2)
    x = 0.7 + 0.05 * rng.normal(size=(16, 64)).astype(np.float32)
    ch = rng.integers(0, 2, size=16)
    sc = fit_channel_minmax(x, ch, 2, feature_range=(0.0, 1.0))
    xt = torch.from_numpy(x[:3])
    ct = torch.from_numpy(ch[:3])
    y = sc.transform_torch(xt, ct)
    xr = sc.inverse_torch(y, ct)
    assert torch.isfinite(y).all()
    assert torch.allclose(xr, xt, atol=1e-5)

    out = train_denoiser(
        x,
        ch,
        n_channels=2,
        hidden=16,
        n_times=8,
        steps=2,
        batch_size=8,
        seed=2,
        val_frac=0.25,
        device="cpu",
        scaler=sc,
    )
    samples = unguided_from_nominal(
        out["model"],
        out["schedule"],
        x[:2],
        ch[:2],
        nu=0.5,
        ddim_steps=3,
        device="cpu",
        scaler=sc,
    )
    assert samples.shape == (2, 64)
    assert np.isfinite(samples).all()
    assert scaler_from_ckpt(ChannelMinMax(lo=sc.lo, hi=sc.hi).to_ckpt()) is not None
