"""Raw-window realism measures and diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anogen.shell.realism import (
    acf_rows,
    autocorrelation_difference,
    channel_span_normalize,
    classifier_two_sample_test,
    diag_cusum,
    diag_envelope,
    diag_excursion_count,
    diag_persistence,
    diag_signed_peak,
    dominant_fft_features,
    feature_distribution_distance,
    fft_average_wasserstein,
    grouped_c2st_split,
    marginal_distribution_difference,
    marginal_w1,
    pooled_acf,
)
from anogen.shell.scaler import ChannelMinMax


def _scaler() -> ChannelMinMax:
    return ChannelMinMax(lo=np.array([0.0, 10.0]), hi=np.array([1.0, 20.0]))


def test_span_normalization_is_channel_affine_invariant():
    base = np.array([[0.1, 0.4, 0.8], [11.0, 14.0, 18.0]])
    got = channel_span_normalize(base, np.array([0, 1]), _scaler())
    assert np.allclose(got[0], got[1])


def test_identical_distributions_have_zero_published_distances():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.1, size=(16, 64))
    ch = np.zeros(len(x), dtype=int)
    scaler = _scaler()
    assert marginal_w1(x, x.copy(), ch, ch, scaler)["value"] == 0.0
    assert marginal_distribution_difference(x, x.copy(), ch, ch, scaler)["value"] == 0.0
    assert autocorrelation_difference(x, x.copy(), ch, ch)["value"] == 0.0
    assert feature_distribution_distance(x, x.copy(), ch, ch, scaler)["value"] == 0.0
    assert fft_average_wasserstein(x, x.copy(), ch, ch, scaler)["value"] == 0.0


def test_offset_changes_marginal_but_permutation_changes_acf():
    rng = np.random.default_rng(1)
    t = np.linspace(0.0, 8.0 * np.pi, 128)
    real = np.stack([0.4 + 0.1 * np.sin(t + p) for p in np.linspace(0, 1, 20)])
    ch = np.zeros(len(real), dtype=int)
    shifted = real + 0.2
    shuffled = np.stack([rng.permutation(row) for row in real])
    scaler = _scaler()
    assert marginal_w1(real, shifted, ch, ch, scaler)["value"] > 0.15
    # Permuting time leaves the pooled value marginal exactly unchanged.
    assert marginal_w1(real, shuffled, ch, ch, scaler)["value"] < 1e-12
    assert autocorrelation_difference(real, shuffled, ch, ch, max_lag=32)["value"] > 0.1


def test_fft_detects_frequency_and_amplitude_changes():
    t = np.arange(128)
    real = np.stack([0.5 + 0.1 * np.sin(2 * np.pi * 4 * t / 128 + p) for p in range(12)])
    synth = np.stack([0.5 + 0.2 * np.sin(2 * np.pi * 9 * t / 128 + p) for p in range(12)])
    ch = np.zeros(12, dtype=int)
    row = fft_average_wasserstein(real, synth, ch, ch, _scaler())
    assert row["value"] > 0.05
    assert row["frequency_w1"]["value"] > 0.01


def test_mdd_matches_tsgbench_density_definition():
    # Two windows of width three: the first time step is dropped, the remaining
    # two each put both real values in their own bin of a two-bin histogram.
    real = np.array([[9.0, 0.0, 0.0], [9.0, 1.0, 1.0]])
    synth = np.array([[9.0, 0.0, 2.0], [9.0, 1.0, 3.0]])
    ch = np.zeros(2, dtype=int)
    scaler = ChannelMinMax(lo=np.array([0.0]), hi=np.array([1.0]))
    row = marginal_distribution_difference(real, synth, ch, ch, scaler, bins=2)
    # t=1 agrees exactly; at t=2 both generated values land above the real
    # range, so TSGBench's histogram keeps no generated mass at all.  Real
    # density is 1 / 0.5 / 2 = 1.0 per bin, hence a mean absolute gap of 1.0.
    assert np.isclose(row["value"], 0.5)
    assert np.isclose(row["out_of_real_range_fraction"]["value"], 0.5)


def test_pooled_acf_recovers_cosine_and_ignores_channel_affine_scale():
    w = 256
    t = np.arange(w)
    x = np.stack([np.sin(2 * np.pi * 8 * t / w + p) for p in np.linspace(0, 2, 24)])
    got = pooled_acf(x, max_lag=33)
    assert got[0] == 1.0
    assert len(got) == 33
    assert np.isclose(got[32], 1.0, atol=0.05)
    assert np.isclose(got[16], -1.0, atol=0.05)
    assert np.allclose(pooled_acf(3.0 * x + 7.0, max_lag=33), got, atol=1e-9)


def test_acd_is_euclidean_norm_over_lags():
    w = 64
    t = np.arange(w)
    real = np.stack([np.sin(2 * np.pi * 4 * t / w + p) for p in np.linspace(0, 2, 16)])
    synth = np.stack([np.sin(2 * np.pi * 9 * t / w + p) for p in np.linspace(0, 2, 16)])
    ch = np.zeros(16, dtype=int)
    row = autocorrelation_difference(real, synth, ch, ch, max_lag=32)
    diff = pooled_acf(real, 32) - pooled_acf(synth, 32)
    assert np.isclose(row["value"], np.sqrt(np.sum(diff**2)))


def test_dominant_fft_amplitude_recovers_sinusoid_amplitude():
    w = 128
    t = np.arange(w)
    x = np.stack([a * np.sin(2 * np.pi * 5 * t / w) + 0.3 for a in (0.1, 0.25)])
    freq, amp = dominant_fft_features(x)
    assert np.allclose(freq, 5.0)
    assert np.allclose(amp, [0.1, 0.25], atol=1e-9)


def test_acf_rows_handles_constants():
    got = acf_rows(np.ones((2, 8)), max_lag=3)
    assert np.allclose(got[:, 0], 1.0)
    assert np.allclose(got[:, 1:], 0.0)


def test_envelope_reports_exact_fractions_and_excess():
    x = np.array([[0.0, 0.5, 1.2, -0.1], [0.2, 0.3, 0.4, 0.5]])
    row = diag_envelope(x, np.array([0, 0]), _scaler())
    assert row["sample_fraction"] == 0.25
    assert row["window_fraction"] == 0.5
    assert np.isclose(row["max_excess"], 0.2)


def test_signed_peak_polarity_and_position():
    x = np.zeros((2, 65))
    x[0, 20] = 3.0
    x[1, 45] = -4.0
    got = diag_signed_peak(x, width=9)
    assert got["severity"][0] > 0
    assert got["severity"][1] < 0
    assert np.isclose(got["position"][0], 20 / 64)
    assert np.isclose(got["position"][1], 45 / 64)


def test_cusum_localizes_step_and_persistence_rejects_bump():
    step = np.concatenate([np.ones(32), np.zeros(96)])
    bump = np.zeros(128)
    bump[48:64] = 1.0
    got = diag_cusum(np.stack([step, bump]), min_segment=4)
    assert abs(got["position"][0] - 0.25) < 0.02
    persistence = diag_persistence(np.stack([step, bump]))
    assert persistence[0] < -0.9
    assert abs(persistence[1]) < 1e-12


def test_excursion_count_counts_contiguous_events():
    x = np.zeros((1, 80))
    x[0, 10] = 3.0
    x[0, 30:33] = -2.0
    x[0, 60] = 4.0
    assert int(diag_excursion_count(x, width=9)[0]) == 3


def test_grouped_split_has_no_group_leakage():
    rg = np.repeat(np.arange(8), 2)
    sg = np.repeat(np.arange(10), 3)
    rtr, rte, str_, ste = grouped_c2st_split(rg, sg, test_size=0.3, seed=2)
    assert set(rg[rtr]).isdisjoint(set(rg[rte]))
    assert set(sg[str_]).isdisjoint(set(sg[ste]))


def test_c2st_separates_shifted_data():
    rng = np.random.default_rng(3)
    n, w = 80, 24
    real = rng.normal(0.2, 0.03, size=(n, w))
    synth = rng.normal(0.8, 0.03, size=(n, w))
    ch = np.zeros(n, dtype=int)
    groups = np.repeat(np.arange(n // 2), 2)
    rows = classifier_two_sample_test(
        real,
        synth,
        ch,
        ch,
        groups,
        groups + 100,
        _scaler(),
        seeds=(0, 1),
        n_perm=20,
    )
    assert min(r["balanced_accuracy"] for r in rows) > 0.95
    assert max(r["permutation_p"] for r in rows) < 0.1


def test_realism_phase_is_registered_and_isolated():
    from anogen.cli import _PHASES
    from anogen.phases.realism import assert_isolated

    assert "realism" in _PHASES
    assert_isolated(
        Path("/tmp/shell_realism"),
        Path("/tmp/shell_s0"),
        Path("/tmp/shell_s3"),
        Path("/tmp/shell_s4"),
    )
    with pytest.raises(RuntimeError, match="must not write"):
        assert_isolated(
            Path("/tmp/shell_s4"),
            Path("/tmp/shell_s0"),
            Path("/tmp/shell_s4"),
        )
