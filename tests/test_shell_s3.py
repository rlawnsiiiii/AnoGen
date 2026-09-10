"""S3 reference generators: GenIAS and post-hoc."""

from __future__ import annotations

import numpy as np
import pytest

from anogen.shell.baselines import posthoc_inject
from anogen.shell.diffusion import torch_available

pytestmark = pytest.mark.skipif(not torch_available(), reason="torch extra not installed")


def test_genias_inflates_and_posthoc_changes():
    from anogen.shell.genias import deviation_patch, genias_sample, train_genias

    rng = np.random.default_rng(0)
    x = rng.normal(size=(24, 64)).astype(np.float32)
    out = train_genias(x, hidden=16, latent=8, steps=2, batch_size=8, device="cpu")
    y = genias_sample(out["model"], x[:6], psi=2.5, device="cpu")
    assert y.shape == (6, 64)
    assert np.isfinite(y).all()
    z = posthoc_inject(x[:6], rng=rng)
    assert z.shape == (6, 64)
    assert not np.allclose(z, x[:6])
    parent = np.ones((2, 8), dtype=np.float32)
    parent[0, 3] = 2.0
    parent[1] = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    gen = parent.copy()
    gen[0, 3] = 5.0
    gen[1] = parent[1] + 0.01
    patched = deviation_patch(parent, gen, tau=0.2)
    assert patched[0, 3] == 5.0
    assert np.allclose(patched[1], parent[1])
    assert np.allclose(deviation_patch(parent, gen, tau=0.0), gen)
