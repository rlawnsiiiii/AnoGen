"""CutAddPaste / Lai taxonomy operators and genbase isolation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anogen.shell.baselines import (
    TAXONOMY_FAMILIES,
    cutaddpaste_inject,
    posthoc_inject,
    taxonomy_inject,
)


def test_cutaddpaste_pastes_other_donor():
    rng = np.random.default_rng(0)
    x = np.zeros((2, 64), dtype=np.float32)
    x[1, 8:40] = 7.0
    y = cutaddpaste_inject(
        x,
        rng=rng,
        width_frac=(0.4, 0.5),
        trend_scale=0.0,
        jitter_sigma=0.0,
        n_paste=1,
        channel_idx=np.array([0, 0]),
    )
    assert y.shape == (2, 64)
    assert y.dtype == np.float32
    assert np.isfinite(y).all()
    assert not np.allclose(y, x)
    assert float(y[0].max()) > 3.0


def test_cutaddpaste_single_window_and_dtype():
    rng = np.random.default_rng(1)
    x = np.linspace(0.0, 1.0, 48, dtype=np.float32)[None, :]
    y = cutaddpaste_inject(x, rng=rng)
    assert y.shape == (1, 48)
    assert y.dtype == np.float32
    assert np.isfinite(y).all()


def test_cutaddpaste_stays_on_channel():
    rng = np.random.default_rng(5)
    x = np.zeros((4, 32), dtype=np.float32)
    x[0] = 0.1
    x[1] = 0.1
    x[2:] = 9.0
    y = cutaddpaste_inject(
        x,
        rng=rng,
        width_frac=(0.4, 0.6),
        trend_scale=0.0,
        jitter_sigma=0.0,
        channel_idx=np.array([0, 0, 1, 1]),
    )
    assert float(y[0].max()) < 1.0
    assert float(y[1].max()) < 1.0


def test_taxonomy_families_change_series():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(40, 64)).astype(np.float32)
    y, labels = taxonomy_inject(x, rng=rng, return_labels=True)
    assert y.shape == x.shape
    assert y.dtype == np.float32
    assert np.isfinite(y).all()
    assert not np.allclose(y, x)
    assert set(labels.tolist()) <= set(TAXONOMY_FAMILIES)
    for fam in TAXONOMY_FAMILIES:
        sl = taxonomy_inject(
            x[:6],
            rng=np.random.default_rng(3),
            families=(fam,),
        )
        assert sl.shape == (6, 64)
        assert not np.allclose(sl, x[:6])


def test_taxonomy_rejects_unknown_family():
    with pytest.raises(ValueError, match="unknown taxonomy"):
        taxonomy_inject(np.zeros((2, 16)), rng=np.random.default_rng(0), families=("nope",))


def test_posthoc_untouched():
    rng = np.random.default_rng(4)
    x = np.linspace(0.0, 1.0, 32, dtype=np.float32)[None, :].repeat(5, axis=0)
    a = posthoc_inject(x, rng=np.random.default_rng(4))
    b = posthoc_inject(x, rng=np.random.default_rng(4))
    assert np.allclose(a, b)
    assert a.shape == (5, 32)


def test_genbase_isolation_and_cli():
    from anogen.cli import _PHASES
    from anogen.phases.genbase import assert_isolated

    assert "genbase" in _PHASES
    out = Path("/tmp/anogen_genbase_test")
    assert_isolated(out, Path("/tmp/shell_s3"), Path("/tmp/shell_s4"))
    with pytest.raises(RuntimeError, match="must not write"):
        assert_isolated(Path("/tmp/shell_s4"), Path("/tmp/shell_s3"), Path("/tmp/shell_s4"))
