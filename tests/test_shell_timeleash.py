"""Time-domain donor leash (suggestion 2)."""

from __future__ import annotations

import numpy as np
import pytest

from anogen.shell.steer import parent_pull_np


def test_l2_is_identity():
    r = np.array([-0.2, -0.01, 0.0, 0.03, 0.4])
    assert np.allclose(parent_pull_np(r, kind="l2"), r)


def test_gm_saturates_large_residuals():
    r = np.array([0.01, 0.2, 0.5])
    g = parent_pull_np(r, kind="gm", delta=0.03)
    assert g[0] > 0.5 * r[0]
    assert abs(g[2]) < abs(g[1])
    assert abs(g[2]) < 0.05


def test_gm_small_matches_l2():
    r = np.array([-0.005, 0.004])
    g = parent_pull_np(r, kind="gm", delta=0.03)
    assert np.allclose(g, r, atol=5e-4)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        parent_pull_np(np.zeros(3), kind="nope")
