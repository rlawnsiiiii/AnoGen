"""encscore recipes match locked S4 band and STEERING_MATH combined (id 14)."""

from anogen.phases.encscore import _BAND, _COMBINED


def test_band_is_locked_s4_f():
    assert _BAND["lam"] == 0.3
    assert _BAND["nu"] == 0.2
    assert _BAND["ddim_steps"] == 20
    assert _BAND["lam_anom"] == 0.0
    assert _BAND["normalize_grad"] is False


def test_combined_is_steering_math_id14():
    assert _COMBINED["lam"] == 0.3
    assert _COMBINED["lam_anom"] == 1.0
    assert _COMBINED["lam_rare"] == 1.0
    assert _COMBINED["ddim_steps"] == 50
    assert _COMBINED["normalize_grad"] is True
