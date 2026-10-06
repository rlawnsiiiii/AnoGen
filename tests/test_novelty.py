"""shell/novelty.KnnNovelty: keeps windows outside the nominal set (numpy only)."""

import numpy as np
import pytest

from anogen.shell.novelty import KnnNovelty, novelty_from_cfg


def _nominal(n, rng, w=128):
    t = np.arange(w)
    phase = rng.uniform(0, 2 * np.pi, (n, 1))
    return 0.5 + 0.1 * np.sin(2 * np.pi * t / 40 + phase) + 0.01 * rng.standard_normal((n, w))


def test_keeps_spikes_and_drops_regenerated_nominals():
    rng = np.random.default_rng(0)
    ref, cal, nominal_like = _nominal(300, rng), _nominal(200, rng), _nominal(60, rng)
    spikes = _nominal(60, rng)
    spikes[np.arange(60), rng.integers(10, 118, 60)] += 0.6
    nov = KnnNovelty(n_kernels=100, n_components=16, q=0.99).fit(ref, cal)
    keep_nom, keep_spk = nov.keep(nominal_like), nov.keep(spikes)
    assert keep_spk.mean() > 0.9
    assert keep_nom.mean() < 0.1
    assert nov.keep(np.zeros((0, 128))).shape == (0,)


def test_config_parsing():
    assert novelty_from_cfg(None) is None and novelty_from_cfg({"kind": "none"}) is None
    f = novelty_from_cfg({"kind": "rocket_knn", "q": 0.95, "kernels": 50})
    assert isinstance(f, KnnNovelty) and f.q == 0.95 and f.n_kernels == 50
    with pytest.raises(ValueError):
        novelty_from_cfg({"kind": "isolation_forest"})
    with pytest.raises(RuntimeError):
        KnnNovelty().score(np.zeros((1, 8)))
