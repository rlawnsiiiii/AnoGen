import numpy as np

from anogen.shell.texture import kept_fraction, sigma_min_for, texture_report, window_spectrum
from anogen.testbed.gauss import GPChannel, LinearDenoiser, Schedule


def test_white_noise_wiener_fraction():
    rng = np.random.default_rng(0)
    v = 0.02**2
    x = 0.5 + np.sqrt(v) * rng.standard_normal((4000, 256))
    spec = window_spectrum(x)
    assert np.allclose(spec[1:-1].mean(), v, rtol=0.05)
    for s in (0.01, 0.02, 0.05):
        assert abs(kept_fraction(spec, s, width=256) - v / (v + s * s)) < 0.02


def test_matches_exact_posterior_mean_on_gp():
    """Kept texture of E[x0 | x0 + σ ε] (exact Wiener smoother) vs the spectral formula."""
    ch, w, s = GPChannel(), 256, 0.01
    sig = ch.cov(w)
    A = sig @ np.linalg.inv(sig + s * s * np.eye(w))
    D = np.diff(np.eye(w), axis=0)
    C = A @ (sig + s * s * np.eye(w)) @ A.T  # Cov of the posterior mean
    exact = np.trace(D @ C @ D.T) / np.trace(D @ sig @ D.T)
    x = ch.sample(3000, w, np.random.default_rng(1))
    approx = kept_fraction(window_spectrum(x), s, width=w)
    assert abs(exact - approx) < 0.05


def test_report_and_sigma_rule():
    x = GPChannel().sample(500, 512, np.random.default_rng(2))
    r = texture_report(x, [0.01, 0.001])
    assert r["kept_texture_sigma_0.001"] > r["kept_texture_sigma_0.01"]
    s = sigma_min_for(window_spectrum(x), width=512, keep=0.95)
    assert kept_fraction(window_spectrum(x), s, width=512) >= 0.95
    _ = LinearDenoiser, Schedule  # testbed import smoke


def test_schedule_bounds_dc_dominates_uncentred_data():
    from anogen.shell.texture import component_power, schedule_bounds

    x = GPChannel(mean=0.5).sample(400, 256, np.random.default_rng(3))
    raw = schedule_bounds(x)
    cen = schedule_bounds(x - 0.5, centered=True)
    assert raw["dc_power"] > 50 * raw["spectrum_max"]  # W·0.25 = 64 vs harmonics ~1
    assert raw["sigma_max"] > 5 * cen["sigma_max"]
    assert abs(cen["sigma_max"] - cen["sigma_max_without_dc"]) < 1e-9 or cen["sigma_max"] >= cen["sigma_max_without_dc"]
    p = component_power(np.full((3, 64), 2.0))
    assert np.isclose(p["dc_power"], 64 * 4.0)
