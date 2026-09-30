"""Residual-copy editor: keep the donor carrier, slow hitch from DDIM."""

from __future__ import annotations

import numpy as np

from anogen.shell.editor import apply_editor, residual_copy, slow_component


def test_residual_copy_identity():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 64)).astype(np.float32)
    y = residual_copy(x, x, n_keep=3)
    assert y.shape == x.shape
    assert y.dtype == np.float32
    assert np.allclose(y, x, atol=1e-5)


def test_residual_copy_replaces_carrier():
    w = 512
    t = np.arange(w, dtype=np.float64)
    parent = 0.8 + 0.03 * np.sin(2 * np.pi * t / 100.0)
    gen = 0.8 + 0.03 * np.sin(2 * np.pi * t / 40.0)
    out = residual_copy(parent[None], gen[None], n_keep=3)[0]
    fast_p = parent - slow_component(parent, 3)
    fast_o = out - slow_component(out, 3)
    fast_g = gen - slow_component(gen, 3)
    c_out = float(np.corrcoef(fast_o, fast_p)[0, 1])
    c_gen = float(np.corrcoef(fast_g, fast_p)[0, 1])
    assert c_out > 0.95
    assert c_out > c_gen + 0.4


def test_editor_shift_keeps_slow_drop():
    w = 512
    t = np.linspace(0.0, 8.0 * np.pi, w)
    parent = 0.80 + 0.02 * np.sin(t)
    gen = np.concatenate([np.full(64, 1.00), np.full(w - 64, 0.78)])
    gen = gen + 0.02 * np.sin(2.3 * t)
    out = apply_editor(
        parent[None],
        gen[None],
        np.array(["real level shift"]),
        n_keep=3,
    )[0]
    assert float(out[:40].mean() - out[200:].mean()) > 0.08
    fast_p = parent - slow_component(parent, 3)
    fast_o = out - slow_component(out, 3)
    assert float(np.corrcoef(fast_o, fast_p)[0, 1]) > 0.9


def test_editor_keeps_needle():
    w = 128
    t = np.linspace(0.0, 8.0 * np.pi, w)
    parent = 0.80 + 0.02 * np.sin(t)
    gen = parent.copy()
    gen[40] = 0.25
    out = apply_editor(
        parent[None],
        gen[None],
        np.array(["real ESA Point / Global"]),
        patch_tau=0.2,
    )[0]
    assert float(out[40]) < 0.45
    assert abs(float(out[10] - parent[10])) < 0.02


def test_editor_quiet_stays_near_parent():
    rng = np.random.default_rng(1)
    w = 256
    t = np.linspace(0.0, 8.0 * np.pi, w)
    parent = 0.80 + 0.02 * np.sin(t)
    gen = parent + 0.015 * rng.normal(size=w)
    out = apply_editor(
        parent[None],
        gen[None],
        np.array(["real ESA local subsequence"]),
        n_keep=3,
    )[0]
    err_out = float(np.mean((out - parent) ** 2))
    err_gen = float(np.mean((gen - parent) ** 2))
    assert err_out < 0.4 * err_gen
