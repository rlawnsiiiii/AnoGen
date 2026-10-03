"""Exact linear-Gaussian denoisers and a numpy mirror of ``guided_ddim``.

Why this exists
---------------
The TSDiff backbone in ``shell/tsdiff.py`` is *causal*: every S4-D layer is a
one-sided convolution (``y[t] = sum_{s<=t} k[t-s] u[s]``) and every other op is
pointwise in time, so ``eps_theta(x_t)[i]`` only sees ``x_t[0..i]``. For a
stationary Gaussian process the best any causal model can do is the causal
posterior mean ``E[x_0[i] | x_t[0..i]]``, which has a closed form. Comparing it
with the bidirectional posterior mean ``E[x_0 | x_t]`` isolates what the
causal restriction does to sampling and to guidance, with no training noise.

Everything here works in *scaled* units (the min-max space the denoiser is
trained in). Nothing imports torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.linalg import cho_factor, solve_triangular, toeplitz

# --------------------------------------------------------------------------
# Data: a stationary Gaussian "telemetry" channel
# --------------------------------------------------------------------------


@dataclass
class GPChannel:
    """Stationary Gaussian process in min-max scaled units.

    Random-phase harmonics (orbit period and overtones) plus AR(1) texture.
    ``C(tau) = sum_k s_k^2 cos(2 pi tau / P_k) + s_ar^2 rho^|tau|``.
    """

    mean: float = 0.5
    harmonics: tuple[tuple[float, float], ...] = ((200.0, 0.10), (100.0, 0.04), (66.7, 0.02))
    ar_rho: float = 0.9
    ar_sigma: float = 0.02
    jitter: float = 1e-6

    def acov(self, n: int) -> np.ndarray:
        tau = np.arange(n, dtype=np.float64)
        c = np.zeros(n)
        for period, sd in self.harmonics:
            c += sd**2 * np.cos(2.0 * np.pi * tau / float(period))
        c += self.ar_sigma**2 * self.ar_rho**tau
        c[0] += self.jitter
        return c

    def cov(self, n: int) -> np.ndarray:
        return toeplitz(self.acov(n))

    def sample(self, n_windows: int, width: int, rng: np.random.Generator) -> np.ndarray:
        chol = np.linalg.cholesky(self.cov(width))
        z = rng.standard_normal((width, n_windows))
        return (self.mean + chol @ z).T


# --------------------------------------------------------------------------
# Diffusion schedule (same linear beta as DiffusionSchedule.linear)
# --------------------------------------------------------------------------


@dataclass
class Schedule:
    n_times: int = 200
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    alpha_bar: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        betas = np.linspace(self.beta_start, self.beta_end, self.n_times)
        self.alpha_bar = np.cumprod(1.0 - betas)


class GeometricSchedule:
    """Log-spaced noise-to-signal ratios, twin of DiffusionSchedule.geometric."""

    def __init__(self, n_times: int = 200, sigma_min: float = 1e-3, sigma_max: float = 10.0):
        sig = np.geomspace(float(sigma_min), float(sigma_max), int(n_times))
        self.alpha_bar = 1.0 / (1.0 + sig**2)
        self.n_times = int(n_times)


# --------------------------------------------------------------------------
# Exact denoisers
# --------------------------------------------------------------------------

DENOISERS = ("bidir", "causal", "anticausal", "flip_avg", "flip_ramp")


class LinearDenoiser:
    """Posterior-mean denoiser ``x0_hat = mu + A_t (x_t - sqrt(ab_t) mu)``.

    kind:
      bidir       E[x0 | x_t]                      (what a denoiser should be)
      causal      E[x0[i] | x_t[0..i]]             (best possible causal model)
      anticausal  E[x0[i] | x_t[i..L-1]]           (causal model run on flip(x))
      flip_avg    0.5 causal + 0.5 anticausal       (zero-retrain test-time fix)
      flip_ramp   position-weighted blend, causal weight rising i/(L-1)
    """

    def __init__(self, channel: GPChannel, length: int, schedule: Schedule, kind: str = "bidir"):
        if kind not in DENOISERS:
            raise ValueError(f"unknown denoiser {kind}")
        self.channel = channel
        self.length = int(length)
        self.schedule = schedule
        self.kind = kind
        self.sigma = channel.cov(self.length)
        self._cache: dict[int, np.ndarray] = {}

    def matrix(self, t: int) -> np.ndarray:
        t = int(t)
        if t not in self._cache:
            self._cache[t] = self._build(t)
        return self._cache[t]

    def _build(self, t: int) -> np.ndarray:
        ab = float(self.schedule.alpha_bar[t])
        if self.kind == "bidir":
            s = ab * self.sigma + (1.0 - ab) * np.eye(self.length)
            c, low = cho_factor(s, lower=True)
            # A = sqrt(ab) Sigma S^{-1}; S symmetric so solve S A^T = sqrt(ab) Sigma
            from scipy.linalg import cho_solve

            return (cho_solve((c, low), np.sqrt(ab) * self.sigma)).T
        causal = _causal_matrix(self.sigma, ab)
        if self.kind == "causal":
            return causal
        j = np.eye(self.length)[::-1]
        anti = j @ causal @ j  # stationary + reversible: causal on flip(x)
        if self.kind == "anticausal":
            return anti
        if self.kind == "flip_avg":
            return 0.5 * (causal + anti)
        w = np.linspace(0.0, 1.0, self.length)[:, None]
        return w * causal + (1.0 - w) * anti

    def x0_hat(self, xt: np.ndarray, t: int) -> np.ndarray:
        """xt: (N, L) -> (N, L)."""
        ab = float(self.schedule.alpha_bar[int(t)])
        mu = self.channel.mean
        return mu + (xt - np.sqrt(ab) * mu) @ self.matrix(t).T

    def eps(self, xt: np.ndarray, t: int) -> np.ndarray:
        ab = float(self.schedule.alpha_bar[int(t)])
        return (xt - np.sqrt(ab) * self.x0_hat(xt, t)) / np.sqrt(1.0 - ab)

    def vjp(self, g_x0: np.ndarray, t: int) -> np.ndarray:
        """d f(x0_hat(x_t)) / d x_t  given  d f / d x0_hat  (rows are samples)."""
        return g_x0 @ self.matrix(t)


def _causal_matrix(sigma: np.ndarray, ab: float) -> np.ndarray:
    """A with A[i, j] = 0 for j > i and x0_hat[i] = E[x0[i] | y[0..i]].

    Innovations form: S = ab Sigma + (1-ab) I = L L^T, e = L^{-1} y spans
    y[0..i] with its first i+1 entries, Cov(x0, e) = sqrt(ab) Sigma L^{-T}.
    E[x0[i] | e[0..i]] = sum_{j<=i} Cov(x0[i], e_j) e_j  ->  A = tril(B) L^{-1}.
    """
    n = sigma.shape[0]
    s = ab * sigma + (1.0 - ab) * np.eye(n)
    low = np.linalg.cholesky(s)
    # B = sqrt(ab) Sigma L^{-T}  <=>  B^T = L^{-1} (sqrt(ab) Sigma)
    b = solve_triangular(low, np.sqrt(ab) * sigma, lower=True).T
    linv = solve_triangular(low, np.eye(n), lower=True)
    return np.tril(b) @ linv


# --------------------------------------------------------------------------
# Temporal encoder + energies (numpy mirror of time_* encoders / soft energy)
# --------------------------------------------------------------------------


@dataclass
class BlockEncoder:
    """e(x) = [block means, block stds] on ``n_blocks`` equal blocks.

    Keeps the temporal layout, like ``ShellEncoder(pool="time")`` (W/8 steps).
    """

    width: int = 512
    block: int = 8

    @property
    def n_blocks(self) -> int:
        return self.width // self.block

    def encode(self, x: np.ndarray) -> np.ndarray:
        b = x.reshape(len(x), self.n_blocks, self.block)
        return np.concatenate([b.mean(axis=2), b.std(axis=2)], axis=1)

    def vjp(self, x: np.ndarray, g_z: np.ndarray) -> np.ndarray:
        n = len(x)
        nb, bl = self.n_blocks, self.block
        b = x.reshape(n, nb, bl)
        m = b.mean(axis=2, keepdims=True)
        s = b.std(axis=2, keepdims=True)
        gm = g_z[:, :nb, None] / bl
        gs = g_z[:, nb:, None] * (b - m) / (bl * np.maximum(s, 1e-8))
        return (gm + gs).reshape(n, self.width)


def proto_energy(enc: BlockEncoder, x: np.ndarray, proto: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """h = ||e(x) - a||^2 per row, and dh/dx. ``proto`` is (N, D)."""
    z = enc.encode(x)
    diff = z - proto
    return (diff**2).sum(axis=1), enc.vjp(x, 2.0 * diff)


def soft_energy(z: np.ndarray, ref: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """h = -tau logsumexp(-||z - r||^2 / tau) and dh/dz (rows)."""
    d2 = ((z[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
    a = -d2 / tau
    amax = a.max(axis=1, keepdims=True)
    w = np.exp(a - amax)
    lse = amax[:, 0] + np.log(w.sum(axis=1))
    w = w / w.sum(axis=1, keepdims=True)
    h = -tau * lse
    grad = 2.0 * (z - w @ ref)  # sum_r w_r 2 (z - r)
    return h, grad


@dataclass
class Shell:
    """Quantile band on soft energy, as in ``shell_from_nominal``."""

    enc: BlockEncoder
    ref: np.ndarray
    tau: float
    q_q: float
    delta: float

    @classmethod
    def from_nominal(cls, enc: BlockEncoder, x_ref: np.ndarray, x_val: np.ndarray, q: float = 0.99) -> Shell:
        z_ref = enc.encode(x_ref)
        d2 = ((z_ref[:, None, :] - z_ref[None, :, :]) ** 2).sum(axis=2)
        off = d2[~np.eye(len(z_ref), dtype=bool)]
        tau = float(max(np.sqrt(np.median(off)), 1e-3))
        h, _ = soft_energy(enc.encode(x_val), z_ref, tau)
        iqr = float(np.subtract(*np.percentile(h, [75, 25])))
        return cls(enc, z_ref, tau, float(np.quantile(h, q)), max(0.25 * max(iqr, 1e-6), 1e-4))

    def band(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = self.enc.encode(x)
        h, gz = soft_energy(z, self.ref, self.tau)
        f = (h - self.q_q) ** 2
        return h, self.enc.vjp(x, (2.0 * (h - self.q_q))[:, None] * gz)


# --------------------------------------------------------------------------
# Sampler (mirror of steer.guided_ddim, plus the proposed variants)
# --------------------------------------------------------------------------

GUIDANCE_SPACES = ("x", "x0", "eps")


def _unit(g: np.ndarray, c_max: float, normalize: bool) -> np.ndarray:
    if normalize:
        n = np.linalg.norm(g, axis=1, keepdims=True)
        g = g / np.maximum(n, 1e-8)
    return np.clip(g, -c_max, c_max)


@dataclass
class GuideTerm:
    """One additive objective term on x0_hat[crop:] (scaled units).

    ``fn(x0_window) -> (value_per_row, grad_wrt_window)``; ``lam`` is its weight.
    """

    lam: float
    fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
    # project=True: fn returns (value, edit) and lam·edit is applied as is, in
    # x̂₀ coordinates (no unit normalization, no denoiser VJP), as
    # steer.guided_ddim does for the contrast term.
    project: bool = False


def guided_ddim(
    den: LinearDenoiser,
    x_start: np.ndarray,
    terms: list[GuideTerm],
    *,
    rng: np.random.Generator,
    nu: float = 0.2,
    ddim_steps: int = 50,
    c_max: float = 1.0,
    normalize_grad: bool = True,
    space: str = "x",
    final_scale: float = 1.0,
    start_from_noise: bool = False,
    crop: int = 0,
    crop_end: int = 0,
    t_window: tuple[float, float] = (0.0, 1.0),
    noise_init: tuple[float, float] | None = None,
    eta: float = 0.0,
    final_denoise: bool = True,
) -> np.ndarray:
    """Numpy mirror of ``steer.guided_ddim`` (eta = 0 by default).

    eta > 0 adds DDIM's stochastic term (eta = 1 ~ ancestral DDPM sampling):
    sigma = eta sqrt((1-ab')/(1-ab)) sqrt(1 - ab/ab'), and the eps coefficient
    becomes sqrt(1 - ab' - sigma^2).

    space="x"   repo behaviour: g = unit(J^T grad f) subtracted from x_{t'} at every
                step, and from the final x0_hat (scaled by ``final_scale``, which
                applies to the last edit in every space).
    space="x0"  Jacobian-free: g = unit(grad f) applied to x0_hat, then re-noised
                with the same eps (manifold-preserving guidance). No backprop
                through the denoiser, so no causal smear.
    space="eps" classifier-guidance form: eps' = eps + lam sqrt(1-ab) unit(J^T grad f);
                the induced x0 shift scales as (1-ab)/sqrt(ab) and vanishes at t=0.

    ``crop`` > 0 runs the chain on a longer sequence (burn-in prefix) and the
    objective only sees ``x0_hat[:, crop:]``; the prefix is dropped at the end.
    ``t_window`` restricts guidance to steps whose t / t_start lies inside it.
    """
    if space not in GUIDANCE_SPACES:
        raise ValueError(space)
    sch = den.schedule
    n_times = len(sch.alpha_bar)
    t_start = n_times - 1 if start_from_noise else int(round(nu * (n_times - 1)))
    x0 = np.asarray(x_start, dtype=np.float64)
    if x0.shape[1] != den.length:
        raise ValueError(f"donor length {x0.shape[1]} != denoiser length {den.length}")
    if start_from_noise:
        xt = rng.standard_normal(x0.shape)
        if noise_init is not None:  # marginal-matched start, as steer.noise_init_mean
            ab = sch.alpha_bar[t_start]
            m, sd = noise_init
            xt = np.sqrt(ab) * m + np.sqrt(1.0 - ab + ab * sd**2) * xt
    else:
        ab = sch.alpha_bar[t_start]
        xt = np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * rng.standard_normal(x0.shape)
    times = np.linspace(t_start, 0, max(1, int(ddim_steps)), dtype=int)

    def total_grad(x0h: np.ndarray, t: int, through_denoiser: bool, project: bool = False) -> np.ndarray:
        """Sum of the unit-normalized gradient terms, or (project=True) of the
        projection terms only (x̂₀ units, applied as is)."""
        tot = np.zeros_like(x0h)
        for term in terms:
            if term.lam == 0.0 or bool(term.project) != bool(project):
                continue
            end = x0h.shape[1] - crop_end
            _, g_win = term.fn(x0h[:, crop:end])
            g = np.zeros_like(x0h)
            g[:, crop:end] = g_win
            if term.project:
                tot += term.lam * g
                continue
            if through_denoiser:
                g = den.vjp(g, t)
            tot += term.lam * _unit(g, c_max, normalize_grad)
        return tot

    for i, t in enumerate(times):
        t = int(t)
        t_prev = int(times[i + 1]) if i + 1 < len(times) else -1
        ab = float(sch.alpha_bar[t])
        x0h = den.x0_hat(xt, t)
        eps = (xt - np.sqrt(ab) * x0h) / np.sqrt(1.0 - ab)
        frac = t / max(t_start, 1)
        active = t_window[0] <= frac <= t_window[1]
        edit = np.zeros_like(xt)
        proj = np.zeros_like(xt)
        if active and terms:
            proj = total_grad(x0h, t, False, project=True)
            if space == "x":
                edit = total_grad(x0h, t, True)
            elif space == "x0":
                edit = total_grad(x0h, t, False)
            else:  # eps: fold the x_t-gradient into the noise estimate
                g = total_grad(x0h, t, True)
                eps = eps + np.sqrt(1.0 - ab) * g
                x0h = (xt - np.sqrt(1.0 - ab) * eps) / np.sqrt(ab)
        if t_prev < 0:
            edit = edit + proj
            if final_denoise:
                xt = x0h - final_scale * edit
            else:
                # Keep the last state's residual noise instead of the Tweedie
                # mean: x_t / sqrt(ab) = x0_hat + sqrt(1-ab)/sqrt(ab) eps_hat.
                xt = (xt / np.sqrt(ab)) - final_scale * edit
            break
        abp = float(sch.alpha_bar[t_prev])
        sig = 0.0
        if eta > 0.0 and t_prev != t:
            sig = float(eta) * np.sqrt((1.0 - abp) / (1.0 - ab)) * np.sqrt(max(1.0 - ab / abp, 0.0))
        c_eps = np.sqrt(max(1.0 - abp - sig**2, 0.0))
        noise = sig * rng.standard_normal(xt.shape) if sig > 0 else 0.0
        if space == "x0":
            xt = np.sqrt(abp) * (x0h - edit - proj) + c_eps * eps + noise
        else:
            # as steer.guided_ddim: projection edits are in x̂₀ units, so they are
            # scaled by sqrt(ab') when subtracted from x_{t'}
            xt = np.sqrt(abp) * x0h + c_eps * eps - edit - np.sqrt(abp) * proj + noise
    return xt[:, crop : xt.shape[1] - crop_end]


# --------------------------------------------------------------------------
# Synthetic anomalies (prototype sources)
# --------------------------------------------------------------------------


def level_shift(x: np.ndarray, pos: np.ndarray, size: np.ndarray, ramp: int = 6) -> np.ndarray:
    """Persistent step of ``size`` at ``pos`` (short linear ramp, no overshoot)."""
    out = np.array(x, dtype=np.float64, copy=True)
    w = out.shape[1]
    t = np.arange(w)
    for i in range(len(out)):
        out[i] += size[i] * np.clip((t - pos[i]) / max(ramp, 1), 0.0, 1.0)
    return out


def spike(x: np.ndarray, pos: np.ndarray, size: np.ndarray, width: int = 3) -> np.ndarray:
    out = np.array(x, dtype=np.float64, copy=True)
    for i in range(len(out)):
        p = int(pos[i])
        out[i, p : p + width] += size[i]
    return out
