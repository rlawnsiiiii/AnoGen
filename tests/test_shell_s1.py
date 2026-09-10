"""S1 denoiser smoke. Skips if torch is not installed."""

from __future__ import annotations

import numpy as np
import pytest

from anogen.shell.diffusion import torch_available

pytestmark = pytest.mark.skipif(not torch_available(), reason="torch extra not installed")


def test_unet_shapes_and_two_step_train():
    from anogen.shell.diffusion import UNet1D, count_params, train_denoiser, unguided_from_nominal

    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 64)).astype(np.float32)
    ch = rng.integers(0, 3, size=32)
    out = train_denoiser(
        x,
        ch,
        n_channels=3,
        hidden=16,
        n_times=10,
        steps=2,
        batch_size=8,
        seed=0,
        val_frac=0.25,
        device="cpu",
    )
    assert out["n_params"] == count_params(out["model"])
    assert out["val_mse"] >= 0.0
    samples = unguided_from_nominal(
        out["model"],
        out["schedule"],
        x[:4],
        ch[:4],
        nu=0.5,
        ddim_steps=4,
        device="cpu",
    )
    assert samples.shape == (4, 64)
    assert np.isfinite(samples).all()
    model = UNet1D(hidden=16, n_channels=3)
    assert count_params(model) > 1000


def test_tsdiff_backbone_two_step_train():
    from anogen.shell.diffusion import train_denoiser, unguided_from_nominal

    rng = np.random.default_rng(1)
    x = rng.normal(size=(16, 64)).astype(np.float32)
    ch = rng.integers(0, 3, size=16)
    out = train_denoiser(
        x,
        ch,
        n_channels=3,
        hidden=16,
        n_times=8,
        steps=2,
        batch_size=8,
        seed=1,
        val_frac=0.25,
        device="cpu",
        backbone="tsdiff",
        n_layers=2,
        d_state=8,
    )
    assert out["backbone"] == "tsdiff"
    assert out["n_params"] > 1000
    samples = unguided_from_nominal(
        out["model"],
        out["schedule"],
        x[:2],
        ch[:2],
        nu=0.5,
        ddim_steps=3,
        device="cpu",
    )
    assert samples.shape == (2, 64)
    assert np.isfinite(samples).all()
