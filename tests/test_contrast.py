import numpy as np
import pytest

from anogen.shell.contrast import contrast_weights, measure_contrast, sample_targets


def test_step_contrast_reads_a_step():
    x = np.zeros((1, 256))
    x[0, 100:] = 0.3
    w = contrast_weights("step", np.array([100]), 256)
    assert abs(float((x @ w[0])[0]) - 0.3) < 1e-12
    amp, pos = measure_contrast("step", x)
    assert abs(amp[0] - 0.3) < 1e-12 and abs(pos[0] * 255 - 100) <= 1


def test_spike_contrast_reads_a_spike_and_ignores_level():
    x = np.full((1, 128), 0.7)
    x[0, 60:63] += 0.5
    w = contrast_weights("spike", np.array([61]), 128)
    assert abs(float((x @ w[0])[0]) - 0.5) < 1e-12
    assert abs(float(np.full(128, 3.0) @ w[0])) < 1e-12  # weights sum to zero
    amp, pos = measure_contrast("spike", x)
    assert abs(amp[0] - 0.5) < 1e-12 and abs(pos[0] * 127 - 61) <= 1


def test_sample_targets_spread_positions_and_resample_amplitudes():
    pos, delta, w = sample_targets("step", 2000, 512, np.array([0.1, -0.2]), rng=np.random.default_rng(0))
    assert set(np.round(delta, 6)) == {0.1, -0.2}
    assert pos.min() >= 51 and pos.max() < 461 and np.std(pos / 511) > 0.2
    assert w.shape == (2000, 512) and np.allclose(w.sum(axis=1), 0.0)


def test_bad_kind():
    with pytest.raises(ValueError):
        contrast_weights("ramp", np.array([10]), 64)


def test_contrast_mask_shapes_and_extent():
    from anogen.shell.contrast import contrast_mask

    m = contrast_mask("step", np.array([100, 300]), 512, dilate=8)
    assert m.shape == (2, 512) and m.dtype == bool
    assert not m[0, 91] and m[0, 92] and m[0, -1]
    s = contrast_mask("spike", np.array([200]), 512, dilate=4, spike_half=1, flank_gap=3)
    assert s[0].sum() == 2 * (1 + 3) + 1 and s[0, 200]
    # the flanks of the spike contrast stay outside the generated region
    w = contrast_weights("spike", np.array([200]), 512, spike_half=1, flank_gap=3)
    assert not np.any(s[0] & (w[0] < 0))
