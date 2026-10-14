import numpy as np
import pytest

from anogen.shell.contrast import (
    contrast_weights,
    fit_to_headroom,
    measure_contrast,
    sample_targets,
)


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


def test_fit_to_headroom_flips_only_what_does_not_fit():
    w = contrast_weights("step", np.array([256] * 4), 512)
    # rows 0-1 sit on the floor (0.03 below, 0.97 above), rows 2-3 in the middle
    u = np.concatenate([np.full((2, 512), 0.03), np.full((2, 512), 0.5)])
    delta = np.array([-0.8, 0.8, -0.3, 0.3])
    out, flipped = fit_to_headroom(delta, u, w)
    # the floor donor cannot hold -0.8 but can hold +0.8, so that one turns
    assert flipped.tolist() == [True, False, False, False]
    assert out.tolist() == [0.8, 0.8, -0.3, 0.3]
    # magnitudes are never touched: shrinking excursions is the hashfix failure
    assert np.allclose(np.abs(out), np.abs(delta))


def test_fit_to_headroom_keeps_the_sign_when_neither_side_fits():
    w = contrast_weights("step", np.array([256]), 512)
    out, flipped = fit_to_headroom(np.array([-4.0]), np.full((1, 512), 0.03), w)
    # +4 does not fit either, but up has more room, so it still turns
    assert flipped.tolist() == [True] and out.tolist() == [4.0]
    out2, flipped2 = fit_to_headroom(np.array([-4.0]), np.full((1, 512), 0.97), w)
    assert flipped2.tolist() == [False] and out2.tolist() == [-4.0]


def test_fit_to_headroom_rejects_mismatched_shapes():
    w = contrast_weights("step", np.array([256]), 512)
    with pytest.raises(ValueError):
        fit_to_headroom(np.array([0.1, 0.2]), np.zeros((1, 512)), w)


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


def test_step_head_is_the_mirror_of_step():
    """The head is the shifted side, so a positive amplitude lifts an early stretch."""
    w = contrast_weights("step_head", np.array([40]), 512)[0]
    assert np.isclose(w[:40].sum(), 1.0) and np.isclose(w[40:].sum(), -1.0)
    high_early = np.r_[np.full(40, 0.9), np.full(472, 0.1)]
    assert np.isclose(high_early @ w, 0.8)
    # ... and mirrors "step" exactly
    assert np.allclose(w, -contrast_weights("step", np.array([40]), 512)[0])


def test_step_head_mask_covers_the_head_and_locks_the_tail():
    from anogen.shell.contrast import contrast_mask

    m = contrast_mask("step_head", np.array([40]), 512, dilate=8)[0]
    assert m[:40].all() and m[48] and not m[49] and not m[-1]


def test_step_head_projection_lands_the_real_morphology():
    """One masked projection must lift the head and leave the tail on the donor."""
    from anogen.shell.contrast import contrast_mask

    pos, width = 40, 512
    w = contrast_weights("step_head", np.array([pos]), width)[0]
    m = contrast_mask("step_head", np.array([pos]), width, dilate=8)[0]
    donor = np.full(width, 0.1)
    wd = w * m
    y = donor - (donor @ w - 0.8) * wd / (wd * wd).sum()
    assert np.isclose(y @ w, 0.8)
    assert np.isclose(y[:pos].mean(), 0.9, atol=2e-3)
    assert np.allclose(y[pos + 9 :], 0.1)


def test_pos_range_narrows_the_sampled_position():
    from anogen.shell.contrast import sample_targets

    rng = np.random.default_rng(0)
    pos, _, _ = sample_targets("step_head", 200, 512, np.array([0.8]), rng=rng, pos_range=(0.0625, 0.18))
    # the frozen step guard of W/16 still applies underneath the requested range
    assert pos.min() >= 32 and pos.max() < int(0.18 * 512)
    with pytest.raises(ValueError):
        sample_targets("step", 4, 512, np.array([0.8]), rng=rng, pos_range=(0.01, 0.02))


def test_assign_donors_by_room_gives_the_drop_the_high_parents():
    from anogen.shell.contrast import assign_donors_by_room

    ch = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    level = np.array([0.1, 0.9, 0.2, 0.8, 0.5, 0.05, 0.95, 0.3])
    alloc = np.array(["ls", "pg", "pg", "ls", "pg", "ls", "ls", "pg"])
    out = assign_donors_by_room(alloc, level, ch, {"ls": -0.8, "pg": 0.9})
    for c in (0, 1):
        m = ch == c
        assert sorted(out[m]) == sorted(alloc[m])  # per-channel counts untouched
        assert level[m][out[m] == "ls"].min() >= level[m][out[m] == "pg"].max()


def test_assign_donors_by_room_rejects_mismatched_shapes():
    from anogen.shell.contrast import assign_donors_by_room

    with pytest.raises(ValueError):
        assign_donors_by_room(np.array(["a", "b"]), np.array([0.1]), np.array([0, 0]), {})
