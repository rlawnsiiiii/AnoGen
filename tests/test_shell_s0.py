"""S0 window protocol, sealed guard, and coverage helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anogen.shell.baselines import posthoc_inject
from anogen.shell.coverage import arp, coverage_at_tau, edi_by_method, min_distances, score_generator
from anogen.shell.data import Panel, SealedTelemetryAccessError, load_panel
from anogen.shell.events import assign_splits, build_event_table, panel_pairs
from anogen.shell.features import FEATURE_DIM, embed_windows
from anogen.shell.folds import assign_event_folds, channel_fold_counts
from anogen.shell.protocol import choose_tau, default_protocol, win_rule
from anogen.shell.steer import band_report
from anogen.shell.windows import (
    compose_train_index,
    extract_labeled_windows,
    occupancy_mask,
    sample_nominal,
    window_finite,
)


def _grid(n: int, start: str = "2001-01-01", step_s: int = 30) -> np.ndarray:
    t0 = np.datetime64(start, "ns")
    return t0 + np.arange(n) * np.timedelta64(step_s, "s")


def _panel(
    T: int = 12000,
    k: int = 2,
    *,
    official: str = "2001-01-04",
    allow_test: bool = False,
    start: str = "2001-01-01",
) -> Panel:
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(T, k)).astype(np.float32)
    counts = np.ones((T, k), dtype=np.int16)
    return Panel(
        Y=Y,
        counts=counts,
        grid=_grid(T, start=start),
        channels=[f"channel_{41 + i}" for i in range(k)],
        bin_seconds=30,
        official_train_end=np.datetime64(pd.Timestamp(official), "ns"),
        allow_test=allow_test,
    )


def _events_frame() -> pd.DataFrame:
    labels = pd.DataFrame(
        {
            "ID": ["a", "a", "b", "c", "d"],
            "Channel": [
                "channel_41",
                "channel_42",
                "channel_41",
                "channel_41",
                "channel_41",
            ],
            "StartTime": pd.to_datetime(
                [
                    "2001-01-01 12:00:00",
                    "2001-01-01 12:00:00",
                    "2001-01-01 04:00:00",
                    "2001-01-02 12:00:00",
                    "2001-04-01 00:00:00",
                ]
            ),
            "EndTime": pd.to_datetime(
                [
                    "2001-01-01 12:10:00",
                    "2001-01-01 12:10:00",
                    "2001-01-01 16:00:00",
                    "2001-01-02 12:05:00",
                    "2001-04-01 00:10:00",
                ]
            ),
        }
    )
    types = pd.DataFrame(
        {
            "ID": ["a", "b", "c", "d"],
            "Category": ["Anomaly", "Anomaly", "Rare Event", "Anomaly"],
        }
    )
    return assign_splits(
        build_event_table(labels, types),
        path_train_end="2001-01-02",
        official_train_end="2001-01-04",
    )


def test_event_table_persists_esa_type_columns():
    labels = pd.DataFrame(
        {
            "ID": ["a", "a"],
            "Channel": ["channel_41", "channel_42"],
            "StartTime": pd.to_datetime(["2001-01-01 12:00:00", "2001-01-01 12:00:00"]),
            "EndTime": pd.to_datetime(["2001-01-01 12:10:00", "2001-01-01 12:10:00"]),
        }
    )
    types = pd.DataFrame(
        {
            "ID": ["a"],
            "Category": ["Anomaly"],
            "Class": ["class_3"],
            "Subclass": ["subclass_2"],
            "Dimensionality": ["Multivariate"],
            "Locality": ["Global"],
            "Length": ["Point"],
        }
    )
    ev = build_event_table(labels, types)
    row = ev.iloc[0]
    assert row["esa_class"] == "class_3"
    assert row["esa_subclass"] == "subclass_2"
    assert row["esa_dimensionality"] == "Multivariate"
    assert row["esa_locality"] == "Global"
    assert row["esa_length"] == "Point"
    pairs = panel_pairs(ev, ["channel_41", "channel_42"])
    assert set(pairs["esa_length"]) == {"Point"}


def test_event_table_blank_esa_when_types_have_category_only():
    ev = _events_frame()
    assert ev["esa_length"].eq("").all()
    assert ev["esa_locality"].eq("").all()
    assert ev["esa_class"].eq("").all()
    panel = _panel()
    pairs = panel_pairs(ev, panel.channels)
    labeled = extract_labeled_windows(
        panel, pairs, width=512, max_crops_per_event=8, allow_test_telemetry=False
    )
    assert "esa_length" in labeled.columns
    assert labeled["esa_length"].eq("").all()


def test_event_table_and_splits_seal_test():
    ev = _events_frame()
    by_id = ev.set_index("event_id")["split"]
    assert by_id.loc["a"] == "path_train"
    assert by_id.loc["b"] == "path_train"
    assert by_id.loc["c"] == "pretest"
    assert by_id.loc["d"] == "test"
    assert not bool(ev.loc[ev["event_id"] == "d", "in_pretest_pool"].iloc[0])
    pairs = panel_pairs(ev, ["channel_41", "channel_42"])
    assert set(pairs.loc[pairs["event_id"] == "a", "channel"]) == {
        "channel_41",
        "channel_42",
    }


def test_short_event_is_centered():
    panel = _panel()
    ev = _events_frame()
    pairs = panel_pairs(ev, panel.channels)
    labeled = extract_labeled_windows(
        panel, pairs, width=512, max_crops_per_event=8, allow_test_telemetry=False
    )
    short = labeled[(labeled["event_id"] == "a") & (labeled["channel"] == "channel_41")]
    assert len(short) == 1
    assert short.iloc[0]["placement"] == "center"
    start = int(short.iloc[0]["start"])
    lo, hi = int(short.iloc[0]["label_lo"]), int(short.iloc[0]["label_hi"])
    mid = (lo + hi) // 2
    assert abs((start + 256) - mid) <= 1


def test_long_event_uses_nonoverlapping_crops():
    panel = _panel()
    ev = _events_frame()
    pairs = panel_pairs(ev, panel.channels)
    labeled = extract_labeled_windows(
        panel, pairs, width=512, max_crops_per_event=8, allow_test_telemetry=False
    )
    long = labeled[labeled["event_id"] == "b"].sort_values("start")
    assert len(long) >= 2
    assert (long["placement"] == "crop").all()
    starts = long["start"].to_numpy()
    assert np.all(np.diff(starts) >= 512)


def test_test_events_are_not_extracted():
    panel = _panel()
    ev = _events_frame()
    pairs = panel_pairs(ev, panel.channels)
    labeled = extract_labeled_windows(
        panel, pairs, width=512, max_crops_per_event=8, allow_test_telemetry=False
    )
    assert "d" not in set(labeled["event_id"])
    assert labeled.attrs["sealed_skipped"] >= 1


def test_nominal_misses_events_and_guard():
    panel = _panel()
    ev = _events_frame()
    occ = occupancy_mask(panel, ev, guard_bins=16, occupy_rares=True)
    rng = np.random.default_rng(1)
    nom = sample_nominal(panel, occ, width=512, n_per_channel=40, rng=rng)
    assert len(nom) > 0
    for start, cidx in zip(nom["start"], nom["channel_idx"], strict=True):
        assert not occ[int(start) : int(start) + 512].any()
        assert window_finite(panel.Y, panel.counts, int(start), int(cidx), 512)


def test_train_index_adds_rares_without_occupying_them():
    panel = _panel()
    ev = _events_frame()
    occ_everyday = occupancy_mask(panel, ev, guard_bins=16, occupy_rares=False)
    occ_all = occupancy_mask(panel, ev, guard_bins=16, occupy_rares=True)
    assert int(occ_everyday.sum()) < int(occ_all.sum())
    rng = np.random.default_rng(1)
    nom = sample_nominal(panel, occ_everyday, width=512, n_per_channel=20, rng=rng)
    pairs = panel_pairs(ev, panel.channels)
    labeled = extract_labeled_windows(
        panel, pairs, width=512, max_crops_per_event=8, allow_test_telemetry=False
    )
    rare = labeled[labeled["is_rare"]].reset_index(drop=True)
    train = compose_train_index(nom, rare, upsample=1)
    assert len(train) == len(nom) + len(rare)
    assert int(train["is_rare"].sum()) == len(rare)


def test_sealed_guard_rejects_post_cut_window():
    panel = _panel(
        T=800,
        start="2001-01-03T20:00:00",
        official="2001-01-04",
        allow_test=False,
    )
    assert panel.grid[-1] >= panel.official_train_end
    with pytest.raises(SealedTelemetryAccessError):
        panel.assert_span_allowed(panel.T - 512, 512)


def test_load_panel_crops_sealed_tail(tmp_path: Path):
    T = 200
    grid = _grid(T, start="2006-12-31T23:00:00")
    Y = np.zeros((T, 1), dtype=np.float32)
    counts = np.ones((T, 1), dtype=np.int16)
    path = tmp_path / "p.npz"
    np.savez(path, Y=Y, counts=counts, grid=grid)
    panel = load_panel(
        path,
        channels=["channel_41"],
        official_train_end="2007-01-01",
        allow_test_telemetry=False,
    )
    assert panel.T < T
    assert panel.grid[-1] < np.datetime64("2007-01-01", "ns")


def test_event_fold_is_shared_by_all_windows():
    panel = _panel()
    ev = _events_frame()
    pairs = panel_pairs(ev, panel.channels)
    labeled = extract_labeled_windows(
        panel, pairs, width=256, max_crops_per_event=8, allow_test_telemetry=False
    )
    labeled = assign_event_folds(labeled, n_folds=3, seed=0)
    for eid, g in labeled.groupby("event_id"):
        assert g["fold"].nunique() == 1
    counts = channel_fold_counts(labeled, min_anomaly_windows=5)
    assert set(counts.columns) >= {"channel", "fold", "n_anomaly_windows", "below_min"}


def test_locked_embedding_is_deterministic():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(7, 128)).astype(np.float32)
    a = embed_windows(x)
    b = embed_windows(x)
    assert a.shape == (7, FEATURE_DIM)
    np.testing.assert_allclose(a, b)


def test_band_report_and_label_roles():
    proto = default_protocol({"shell": {}})
    assert "anomaly" in proto["labels"]
    assert "rare" in proto["labels"]
    assert proto["train_includes_rares"] is True
    assert "everyday" in proto["labels"]["nominal"].lower()
    empty = band_report(np.zeros(0), Q_q=1.0, delta=0.2)
    assert empty["n"] == 0.0
    h = np.array([0.9, 1.0, 1.1, 2.0])
    rep = band_report(h, Q_q=1.0, delta=0.15)
    assert rep["n"] == 4.0
    assert 0.0 < rep["frac_in_band"] < 1.0
    assert rep["frac_above"] > 0.0


def test_tau_and_win_rule():
    dists = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    tau = choose_tau(dists, 0.4)
    assert 0.1 <= tau <= 0.4
    proto = default_protocol({"shell": {}})
    decision = win_rule(
        shell_gap=0.3,
        baseline_gaps={"genias": 0.1, "posthoc": 0.05},
        occupancy=0.9,
        diversity=1.0,
        unguided_diversity=1.5,
        protocol=proto,
    )
    assert decision["won"]
    lose = win_rule(
        shell_gap=0.05,
        baseline_gaps={"genias": 0.1, "posthoc": 0.05},
        occupancy=0.9,
        diversity=1.0,
        unguided_diversity=1.5,
        protocol=proto,
    )
    assert not lose["won"]


def test_coverage_gap_and_posthoc():
    rng = np.random.default_rng(2)
    real_a = rng.normal(size=(12, 64)) + 3.0
    real_r = rng.normal(size=(12, 64))
    gen = real_a + rng.normal(scale=0.05, size=real_a.shape)
    unguided = rng.normal(size=(12, 64))
    d_u = min_distances(embed_windows(real_a), embed_windows(unguided))
    scored = score_generator(
        real_a, real_r, gen, unguided_anomaly_dists=d_u, unguided_target=0.10
    )
    assert scored["gap"] >= 0.0
    assert 0.0 <= coverage_at_tau(d_u, scored["tau"]) <= 1.0
    injected = posthoc_inject(real_r, rng=rng)
    assert injected.shape == real_r.shape
    assert not np.allclose(injected, real_r)
    assert scored["arp_anomaly"] > scored["arp_rare"]
    assert 0.0 < scored["arp_anomaly"] <= 1.0


def test_arp_and_edi_order():
    close = np.array([0.1, 0.2, 0.15])
    far = np.array([2.0, 3.0, 2.5])
    assert arp(close) > arp(far)
    rng = np.random.default_rng(0)
    tight = rng.normal(size=(40, 4)) * 0.05
    spread = rng.normal(size=(40, 4))
    edi = edi_by_method({"tight": tight, "spread": spread}, n_regions=8, seed=0)
    assert edi["spread"] > edi["tight"]
