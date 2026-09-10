"""Few-shot adapter and classifier smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

from anogen.shell.diffusion import torch_available

pytestmark = pytest.mark.skipif(not torch_available(), reason="torch extra not installed")


def test_adapter_and_classifier_two_steps():
    from anogen.shell.adapters import count_trainable, train_adapter, train_classifier
    from anogen.shell.diffusion import train_denoiser, unguided_from_nominal

    rng = np.random.default_rng(0)
    x_n = rng.normal(size=(20, 64)).astype(np.float32)
    x_a = x_n[:8] + 1.5
    ch = np.zeros(20, dtype=np.int64)
    den = train_denoiser(
        x_n,
        ch,
        n_channels=2,
        hidden=16,
        n_times=8,
        steps=2,
        batch_size=8,
        device="cpu",
        val_frac=0.25,
    )
    ad = train_adapter(
        den["model"],
        den["schedule"],
        x_a,
        ch[:8],
        x_n,
        ch,
        n_channels=2,
        hidden=8,
        steps=2,
        batch_size=8,
        device="cpu",
    )
    assert ad["n_adapter_params"] == count_trainable(ad["adapter"])
    assert ad["n_adapter_params"] > 0
    samples = unguided_from_nominal(
        ad["model"],
        den["schedule"],
        x_n[:4],
        ch[:4],
        nu=0.4,
        ddim_steps=3,
        device="cpu",
    )
    assert samples.shape == (4, 64)
    clf = train_classifier(x_a, x_n, hidden=8, steps=4, batch_size=8, device="cpu")
    assert 0.0 <= clf["acc_anomaly"] <= 1.0
    assert clf["n_rare"] == 0
    x_r = x_n[:6] + 0.4
    clf_r = train_classifier(
        x_a, x_n, x_rare=x_r, hidden=8, steps=4, batch_size=8, device="cpu"
    )
    assert clf_r["n_rare"] == 6
    assert 0.0 <= clf_r["acc_rare"] <= 1.0
    from anogen.shell.steer import guided_ddim, shell_from_nominal, train_encoder

    enc = train_encoder(x_n, hidden=16, emb=8, steps=2, batch_size=8, device="cpu")
    shell = shell_from_nominal(enc["encoder"], x_n[:8], x_n[8:16], q=0.9, device="cpu")
    prior, _ = guided_ddim(
        ad["model"],
        enc["encoder"],
        np.zeros((3, 64), dtype=np.float32),
        ch[:3],
        den["schedule"],
        ref=shell["ref"],
        Q_q=shell["Q_q"],
        tau=shell["tau"],
        nu=1.0,
        lam=0.0,
        c_max=1.0,
        ddim_steps=3,
        device="cpu",
        mode="off",
        start_from_noise=True,
    )
    assert prior.shape == (3, 64)
    assert np.isfinite(prior).all()
