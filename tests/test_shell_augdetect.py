"""Detector scores and augdetect isolation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anogen.shell.detector import (
    bootstrap_event_recall,
    event_hits,
    event_recall,
    needs_torch,
    threshold_at_far,
)
from anogen.shell.diffusion import torch_available


def test_threshold_and_event_recall():
    neg = np.arange(100, dtype=np.float64)
    t = threshold_at_far(neg, 0.01)
    far = float(np.mean(neg >= t))
    assert 0.005 <= far <= 0.03
    scores = np.array([0.0, 2.0, 0.0, 0.5])
    ev = np.array(["a", "a", "b", "c"])
    hits = event_hits(scores, ev, 1.0)
    assert hits == {"a": True, "b": False, "c": False}
    assert abs(event_recall(hits) - 1.0 / 3.0) < 1e-12
    boot = bootstrap_event_recall(hits, n_boot=200, seed=0)
    assert boot["n"] == 3
    assert 0.0 <= boot["lo"] <= boot["median"] <= boot["hi"] <= 1.0


def test_augdetect_cli_and_isolation():
    from anogen.cli import _PHASES
    from anogen.phases.augdetect import DEFAULT_ARMS, assert_isolated

    assert "augdetect" in _PHASES
    assert "genbase" in _PHASES
    assert "real_only" in DEFAULT_ARMS
    assert "genfsdiff_c1" in DEFAULT_ARMS
    assert "c1_plus_real" in DEFAULT_ARMS
    assert_isolated(Path("/tmp/anogen_augdetect_test"), Path("/tmp/shell_s3"), Path("/tmp/shell_s4"))
    with pytest.raises(RuntimeError, match="must not write"):
        assert_isolated(Path("/tmp/shell_s3"), Path("/tmp/shell_s3"), Path("/tmp/shell_s4"))


@pytest.mark.skipif(not torch_available(), reason="torch extra not installed")
def test_fit_eval_separable():
    from anogen.shell.detector import eval_detector, fit_detector

    rng = np.random.default_rng(0)
    w = 64
    pos = rng.normal(size=(12, w)).astype(np.float32)
    pos[:, 20:28] += 4.0
    neg = rng.normal(size=(24, w)).astype(np.float32)
    clf = fit_detector(pos, neg, seed=0, steps=60, hidden=8, batch_size=8, device="cpu")
    ev = np.array(["e0"] * 4 + ["e1"] * 4 + ["e2"] * 4)
    kind = np.array(["k"] * 12)
    out = eval_detector(
        clf,
        x_anom=pos,
        x_nom=neg,
        x_rare=neg[:6],
        event_id=ev,
        kind=kind,
        far=0.05,
        device="cpu",
        n_boot=50,
        seed=0,
    )
    assert out["event_recall"] >= 0.5
    assert out["window_recall"] >= 0.5
    assert out["ap"] > 0.5
    assert "k" in out["per_kind_recall"]
    assert needs_torch()
