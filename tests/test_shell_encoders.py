"""Temporal / SupCon encoder smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

from anogen.shell.diffusion import torch_available
from anogen.shell.encoders import all_encoder_names, auroc_scores, encoder_spec

pytestmark = pytest.mark.skipif(not torch_available(), reason="torch extra not installed")


def test_auroc_perfect_and_chance():
    assert auroc_scores([3.0, 4.0, 5.0], [0.0, 1.0, 2.0]) == pytest.approx(1.0)
    assert auroc_scores([0.0, 1.0, 2.0], [3.0, 4.0, 5.0]) == pytest.approx(0.0)
    mix = auroc_scores([0.0, 2.0], [1.0, 3.0])
    assert 0.0 < mix < 1.0


def test_specs_cover_pool_and_labels():
    names = all_encoder_names()
    assert "pool_recon" in names and "time_supcon" in names
    assert encoder_spec("pool_recon")["uses_labels"] is False
    assert encoder_spec("time_supcon")["uses_labels"] is True
    assert encoder_spec("time_recon")["pool"] == "time"


def test_time_encoder_flattens_and_trains():
    from anogen.shell.encoders import ShellEncoder, embed_shell, train_shell_encoder

    rng = np.random.default_rng(0)
    x = rng.normal(size=(24, 64)).astype(np.float32)
    xr = x[:6] + 0.2
    xa = x[:6] + 1.0
    out = train_shell_encoder(
        x,
        xr,
        xa,
        pool="time",
        loss="supcon",
        hidden=8,
        emb=8,
        time_emb=4,
        steps=3,
        per_class=4,
        device="cpu",
    )
    z = embed_shell(out["encoder"], x[:3], "cpu")
    assert z.ndim == 2
    assert z.shape[0] == 3
    assert z.shape[1] == 4 * (64 // 8)
    rec = out["encoder"].decode(__import__("torch").from_numpy(x[:2]).unsqueeze(1))
    assert rec.shape[-1] == 64

    pool = ShellEncoder(64, hidden=8, emb=8, pool="pool")
    zp = pool.encode(__import__("torch").randn(2, 1, 64))
    assert zp.shape == (2, 8)
