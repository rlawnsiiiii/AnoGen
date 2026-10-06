"""How much telemetry texture a noise schedule can reproduce (numpy only).

The last DDIM step returns the posterior mean at the smallest noise level
σ_min. For a stationary Gaussian window with power spectrum S(f) (per-sample
variance units), the posterior mean keeps the fraction S/(S + σ²) of each
frequency's variance (a Wiener filter). Texture, i.e. the first-difference
variance, weights frequency f by |1 − e^{−iω}|² = 2 − 2 cos ω, so the kept
share of first-difference variance is

    Σ_f w(f) S(f) · S(f)/(S(f) + σ²)  /  Σ_f w(f) S(f).

This is an upper bound for any sampler that ends with a Tweedie step at σ_min
(a perfect denoiser, exact ODE). Use it to choose σ_min *before* retraining:
measure S(f) on the scaled training windows of each channel.

The other end of the schedule follows from the same spectrum. At σ_max a
component of power P keeps SNR = P / σ_max² in q(x_T | x₀); a sampler started
from N(0, I) is only on-distribution when that is ≈ 0 for the strongest
component, including the window mean (DC), whose power is W·mean² when the
data are not centred. ``schedule_bounds`` turns both into numbers.
"""

from __future__ import annotations

import numpy as np


def window_spectrum(x: np.ndarray) -> np.ndarray:
    """Mean Hann-tapered periodogram of mean-removed windows, per-sample variance units.

    White noise of variance v gives S(f) ≈ v at every frequency. The taper keeps
    strong low-frequency harmonics (orbit periods) from leaking into the
    high-frequency bins, which would overstate the texture that survives.
    """
    a = np.asarray(x, dtype=np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    h = np.hanning(a.shape[1])
    p = np.abs(np.fft.rfft(a * h, axis=1)) ** 2 / (h * h).sum()
    return p.mean(axis=0)


def kept_fraction(spec: np.ndarray, sigma: float, *, width: int, texture: bool = True) -> float:
    """Share of (first-difference, or total) variance kept by E[x0 | x0 + σ ε]."""
    s = np.asarray(spec, dtype=np.float64)
    omega = 2.0 * np.pi * np.arange(len(s)) / float(width)
    weight = (2.0 - 2.0 * np.cos(omega)) if texture else np.ones_like(s)
    # rfft bins other than DC and Nyquist stand for two complex bins
    mult = np.full(len(s), 2.0)
    mult[0] = 1.0
    if width % 2 == 0:
        mult[-1] = 1.0
    num = (mult * weight * s * s / (s + float(sigma) ** 2)).sum()
    den = (mult * weight * s).sum()
    return float(num / max(den, 1e-30))


def texture_report(x_scaled: np.ndarray, sigmas: list[float]) -> dict[str, float]:
    """Scale facts of one channel's scaled windows and the kept texture per σ_min."""
    a = np.asarray(x_scaled, dtype=np.float64)
    spec = window_spectrum(a)
    d = np.diff(a, axis=1)
    out = {
        "window_mean": float(a.mean()),
        "sigma_data_within_window": float(a.std(axis=1).mean()),
        "diff_sd": float(d.std()),
        "texture_sd_per_bin": float(d.std() / np.sqrt(2.0)),
    }
    for s in sigmas:
        out[f"kept_texture_sigma_{s:g}"] = kept_fraction(spec, s, width=a.shape[1])
        out[f"kept_variance_sigma_{s:g}"] = kept_fraction(spec, s, width=a.shape[1], texture=False)
    return out


def sigma_min_for(spec: np.ndarray, *, width: int, keep: float = 0.95) -> float:
    """Largest σ_min (log grid 1e-5..1) that keeps at least ``keep`` of the texture."""
    best = 1e-5
    for s in np.geomspace(1e-5, 1.0, 101):
        if kept_fraction(spec, float(s), width=width) >= keep:
            best = float(s)
    return best


def component_power(x_scaled: np.ndarray, *, center: float | None = None) -> dict[str, float]:
    """Largest per-frequency power of the windows, and the power of their mean (DC).

    ``center``: value subtracted before the DC power (None = 0, the data as the
    denoiser sees them; the channel mean = what a centred scaler would leave).
    Both in the per-sample-variance units of ``window_spectrum``.
    """
    a = np.asarray(x_scaled, dtype=np.float64)
    w = a.shape[1]
    c = 0.0 if center is None else float(center)
    dc = float(w * np.mean((a.mean(axis=1) - c) ** 2))
    spec = window_spectrum(a)
    return {"spectrum_max": float(spec[1:].max()), "dc_power": dc}


def schedule_bounds(
    x_scaled: np.ndarray,
    *,
    keep_texture: float = 0.95,
    terminal_snr: float = 0.01,
    centered: bool = False,
) -> dict[str, float]:
    """σ_min that keeps ``keep_texture`` of the texture and σ_max that leaves at
    most ``terminal_snr`` in the strongest component (DC included).

    ``centered=True`` assumes the denoiser sees the data with the channel mean
    removed (z-score scaler); otherwise the raw scaled values (min/max scaler),
    whose large mean usually dominates σ_max.
    """
    a = np.asarray(x_scaled, dtype=np.float64)
    spec = window_spectrum(a)
    p = component_power(a, center=float(a.mean()) if centered else None)
    p_max = max(p["spectrum_max"], p["dc_power"])
    return {
        "sigma_min": sigma_min_for(spec, width=a.shape[1], keep=keep_texture),
        "sigma_max": float(np.sqrt(p_max / float(terminal_snr))),
        "sigma_max_without_dc": float(np.sqrt(p["spectrum_max"] / float(terminal_snr))),
        **p,
    }
