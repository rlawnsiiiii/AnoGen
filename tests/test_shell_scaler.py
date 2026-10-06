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


def test_fit_channel_standard_is_a_per_channel_zscore():
    import numpy as np

    from anogen.shell.scaler import fit_channel_standard, scaler_from_ckpt

    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(3.0, 0.5, (50, 32)), rng.normal(-1.0, 2.0, (50, 32))]
    ch = np.r_[np.zeros(50, int), np.ones(50, int)]
    sc = fit_channel_standard(x, ch, 2)
    z = sc.transform(x, ch)
    for c in (0, 1):
        assert abs(z[ch == c].mean()) < 1e-5 and abs(z[ch == c].std() - 1.0) < 1e-4
    assert np.allclose(sc.inverse(z, ch), x, atol=1e-4)
    back = scaler_from_ckpt(sc.to_ckpt())
    assert np.allclose(back.transform(x, ch), z)


def test_unit_scale_between_minmax_and_zscore():
    import numpy as np

    from anogen.shell.scaler import fit_channel_minmax, fit_channel_standard, unit_scale

    rng = np.random.default_rng(1)
    x = np.r_[rng.normal(0.0, 1.0, (40, 16)), rng.normal(5.0, 3.0, (40, 16))]
    ch = np.r_[np.zeros(40, int), np.ones(40, int)]
    mm = fit_channel_minmax(x, ch, 2)
    zs = fit_channel_standard(x, ch, 2)
    assert np.isclose(unit_scale(mm, mm), 1.0)
    # one minmax unit = span / sd z-units, per channel
    per = (mm.hi - mm.lo) / np.array([x[ch == c].std() for c in (0, 1)])
    assert np.isclose(unit_scale(zs, mm, [0]), per[0], rtol=1e-6)
    assert np.isclose(unit_scale(zs, mm), np.median(per), rtol=1e-6)
    assert unit_scale(None, mm) == 1.0
