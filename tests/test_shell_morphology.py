"""Duration / waveform kinds for real anomaly windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from anogen.shell.morphology import (
    assign_anomaly_kinds,
    assign_morphology_kinds,
    kind_ref_indices,
    kind_stat,
    kinds_for_windows,
    nearest_generated,
    pick_kind_examples,
    rank_for_kind,
    window_shape_stats,
)


def test_assign_morphology_kinds_uses_span_and_shape():
    w = 32
    spike = np.zeros((1, w), dtype=np.float32)
    spike[0, 8] = 1.0
    subtle = 0.02 * np.sin(np.linspace(0, 6, w))[None, :].astype(np.float32)
    shift = np.zeros((1, w), dtype=np.float32)
    shift[0, w // 2 :] = 0.2
    medium = 0.03 * np.ones((1, w), dtype=np.float32)
    long_c = (0.03 + 0.002 * np.sin(np.linspace(0, 12 * np.pi, w)))[None, :].astype(np.float32)
    x = np.concatenate([spike, subtle, shift, medium, long_c], axis=0)
    span = np.array([5, 5, 2000, 500, 8000], dtype=np.int64)
    kinds = assign_morphology_kinds(x, span)
    assert list(kinds) == [
        "real short spike",
        "real short subtle",
        "real level shift",
        "real medium event",
        "real long campaign",
    ]


def test_assign_anomaly_kinds_point_not_relabeled_by_shift():
    x = np.zeros((1, 32), dtype=np.float32)
    x[0, 16:] = 0.5
    kinds = assign_anomaly_kinds(x, np.array(["Point"]), np.array(["Global"]))
    assert list(kinds) == ["real ESA Point / Global"]


def test_assign_anomaly_kinds_unknown_when_types_blank():
    x = np.zeros((1, 8), dtype=np.float32)
    kinds = assign_anomaly_kinds(x, np.array([""]), np.array([""]))
    assert list(kinds) == ["real ESA unknown"]


def test_kinds_for_windows_joins_anomaly_types():
    w = 32
    x = np.zeros((3, w), dtype=np.float32)
    x[2, w // 2 :] = 0.2
    windows = pd.DataFrame({"event_id": ["p", "s", "g"]})
    types = pd.DataFrame(
        {
            "ID": ["p", "s", "g"],
            "Length": ["Point", "Subsequence", "Subsequence"],
            "Locality": ["Global", "Local", "Global"],
        }
    )
    kinds = kinds_for_windows(x, windows, types)
    assert list(kinds) == [
        "real ESA Point / Global",
        "real ESA local subsequence",
        "real level shift",
    ]


def test_kinds_for_windows_uses_window_columns():
    w = 16
    x = np.zeros((2, w), dtype=np.float32)
    windows = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "esa_length": ["Point", "Subsequence"],
            "esa_locality": ["Local", "Local"],
        }
    )
    kinds = kinds_for_windows(x, windows, types=None)
    assert list(kinds) == ["real ESA Point / Local", "real ESA local subsequence"]


def test_assign_anomaly_kinds_uses_esa_length_locality():
    w = 32
    quiet = np.zeros((1, w), dtype=np.float32)
    shift = np.zeros((1, w), dtype=np.float32)
    shift[0, w // 2 :] = 0.2
    x = np.concatenate([quiet, quiet, quiet, shift], axis=0)
    length = np.array(["Point", "Point", "Subsequence", "Subsequence"], dtype=object)
    locality = np.array(["Local", "Global", "Local", "Global"], dtype=object)
    kinds = assign_anomaly_kinds(x, length, locality)
    assert list(kinds) == [
        "real ESA Point / Local",
        "real ESA Point / Global",
        "real ESA local subsequence",
        "real level shift",
    ]


def test_pick_kind_examples_prefers_distinct_events():
    x = np.zeros((6, 8), dtype=np.float32)
    x[:3, 2] = 1.0
    kinds = np.array(["real short spike"] * 6, dtype=object)
    ev = np.array(["a", "a", "b", "b", "c", "c"])
    picked = pick_kind_examples(x, kinds, ev, n=3)
    assert set(picked["real short spike"]["event_id"]) == {"a", "b", "c"}


def test_kind_ref_prefers_oof_then_leaks():
    kinds = np.array(
        ["real level shift", "real level shift", "real short spike", "real short spike"],
        dtype=object,
    )
    fold = np.array([0, 0, 1, 2])
    idx, leaked = kind_ref_indices(kinds, fold, "real level shift", query_fold=0)
    assert leaked and set(idx.tolist()) == {0, 1}
    idx, leaked = kind_ref_indices(kinds, fold, "real short spike", query_fold=0)
    assert not leaked and set(idx.tolist()) == {2, 3}


def test_kind_stat_ranks_shift_by_half_means():
    x = np.zeros((3, 8), dtype=np.float64)
    x[0, 4:] = 0.2
    x[1, 4:] = 0.05
    x[2, :] = 0.1
    amp, dmu = window_shape_stats(x)
    assert dmu[0] > dmu[1] > dmu[2]
    assert list(rank_for_kind(x, "real level shift", 2)) == [0, 1]
    assert kind_stat(x, "real short spike")[0] == amp[0]


def test_nearest_generated_returns_one_per_query():
    q = np.stack([np.zeros(16), np.ones(16)]).astype(np.float32)
    g = np.stack([0.1 * np.ones(16), np.linspace(0, 1, 16), np.zeros(16)]).astype(np.float32)
    match = nearest_generated(q, g)
    assert match.shape == (2, 16)
