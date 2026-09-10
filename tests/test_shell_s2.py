"""S2 shell energy and guided DDIM smoke."""

from __future__ import annotations

import numpy as np
import pytest

from anogen.shell.diffusion import torch_available

pytestmark = pytest.mark.skipif(not torch_available(), reason="torch extra not installed")


def test_encoder_and_guided_sample():
    from anogen.shell.diffusion import train_denoiser
    from anogen.shell.steer import guided_ddim, shell_from_nominal, train_encoder

    rng = np.random.default_rng(0)
    x = rng.normal(size=(48, 64)).astype(np.float32)
    ch = rng.integers(0, 2, size=48)
    enc = train_encoder(x, hidden=16, emb=8, steps=3, batch_size=16, device="cpu")
    shell = shell_from_nominal(
        enc["encoder"], x[:16], x[16:32], q=0.9, device="cpu"
    )
    assert np.isfinite(shell["Q_q"])
    assert shell["delta"] > 0
    den = train_denoiser(
        x,
        ch,
        n_channels=2,
        hidden=16,
        n_times=8,
        steps=2,
        batch_size=16,
        device="cpu",
        val_frac=0.25,
    )
    samples, h = guided_ddim(
        den["model"],
        enc["encoder"],
        x[:6],
        ch[:6],
        den["schedule"],
        ref=shell["ref"],
        Q_q=shell["Q_q"],
        tau=shell["tau"],
        nu=0.4,
        lam=0.1,
        c_max=1.0,
        ddim_steps=4,
        device="cpu",
        n_correct=2,
        ref_anom=shell["ref"],
        lam_anom=0.2,
        ref_rare=shell["ref"],
        lam_rare=0.2,
    )
    assert samples.shape == (6, 64)
    assert np.isfinite(samples).all()
    assert h.shape == (6,)
    prior, _ = guided_ddim(
        den["model"],
        enc["encoder"],
        np.zeros((4, 64), dtype=np.float32),
        ch[:4],
        den["schedule"],
        ref=shell["ref"],
        Q_q=shell["Q_q"],
        tau=shell["tau"],
        nu=1.0,
        lam=0.1,
        c_max=1.0,
        ddim_steps=4,
        device="cpu",
        start_from_noise=True,
    )
    assert prior.shape == (4, 64)
    assert np.isfinite(prior).all()
