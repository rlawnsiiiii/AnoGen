"""Univariate GenIAS: TCN-VAE, inflate latent variance, decode.

The previous decoder added ``in_conv(x)`` and broadcast a *global* z. That
lets the net copy the waveform at ψ=1, and at ψ>1 it can only add a constant
plus TCN padding spikes. A working GenIAS decode must paint the series from z.
z is a short sequence (about W/16), not one vector.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from anogen.shell.diffusion import _require_torch, torch_available

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


if torch_available():

    class TCNBlock(nn.Module):
        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            self.dilation = int(dilation)
            self.conv = nn.Conv1d(channels, channels, 3, padding=0, dilation=self.dilation)
            self.norm = nn.GroupNorm(min(8, channels), channels)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            pad = self.dilation
            if x.size(-1) <= pad:
                y = F.pad(x, (pad, pad), mode="replicate")
            else:
                y = F.pad(x, (pad, pad), mode="reflect")
            return x + F.silu(self.norm(self.conv(y)))

    class TCNAE(nn.Module):
        def __init__(self, hidden: int = 48, latent: int = 16, n_layers: int = 4) -> None:
            super().__init__()
            self.in_conv = nn.Conv1d(1, hidden, 3, padding=1)
            self.enc_blocks = nn.ModuleList([TCNBlock(hidden, 2**i) for i in range(n_layers)])
            self.to_mu = nn.Conv1d(hidden, latent, 1)
            self.to_logv = nn.Conv1d(hidden, latent, 1)
            self.from_z = nn.Conv1d(latent, hidden, 1)
            self.mid_blocks = nn.ModuleList([TCNBlock(hidden, 2**i) for i in range(n_layers)])
            self.out_blocks = nn.ModuleList([TCNBlock(hidden, 2**i) for i in range(n_layers)])
            self.out = nn.Conv1d(hidden, 1, 3, padding=1)

        def _bottleneck_len(self, length: int) -> int:
            return max(4, int(length) // 8)

        def encode(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            h = self.in_conv(x)
            for b in self.enc_blocks:
                h = b(h)
            h = F.adaptive_avg_pool1d(h, self._bottleneck_len(x.size(-1)))
            return self.to_mu(h), self.to_logv(h)

        def decode(self, z: "torch.Tensor", length: int) -> "torch.Tensor":
            h = self.from_z(z)
            mid = max(4, int(length) // 4)
            h = F.interpolate(h, size=mid, mode="linear", align_corners=False)
            for b in self.mid_blocks:
                h = b(h)
            h = F.interpolate(h, size=int(length), mode="linear", align_corners=False)
            for b in self.out_blocks:
                h = b(h)
            return self.out(h)

        def forward(
            self, x: "torch.Tensor", psi: float = 1.0
        ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            # Per-window z-score so the loss is the oscillation, not the ~0.8 DC.
            mean = x.mean(dim=-1, keepdim=True)
            scale = x.std(dim=-1, keepdim=True).clamp_min(1e-3)
            xn = (x - mean) / scale
            mu, logv = self.encode(xn)
            std = torch.exp(0.5 * logv.clamp(-8.0, 8.0))
            eps = torch.randn_like(std)
            z = mu + float(psi) * std * eps
            rec_n = self.decode(z, x.size(-1))
            rec = rec_n * scale + mean
            return rec, mu, logv

        def forward_pair(
            self, x: "torch.Tensor", psi: "torch.Tensor | float"
        ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """Normal and anomalous decodes from one shared ε (GenIAS Eq. 1).

            x̂ from z = μ + σ⊙ε, x̃ from z̃ = μ + ψ·(σ⊙ε). Both are returned in the
            per-window whitened units the losses are computed in, alongside the
            whitened input; ``psi`` may be a learnable tensor.
            """
            mean = x.mean(dim=-1, keepdim=True)
            scale = x.std(dim=-1, keepdim=True).clamp_min(1e-3)
            xn = (x - mean) / scale
            mu, logv = self.encode(xn)
            std = torch.exp(0.5 * logv.clamp(-8.0, 8.0))
            noise = std * torch.randn_like(std)
            rec_hat = self.decode(mu + noise, x.size(-1))
            rec_tilde = self.decode(mu + psi * noise, x.size(-1))
            return xn, rec_hat, rec_tilde, torch.stack([mu, logv])

else:  # pragma: no cover

    class TCNAE:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


def _shape_scores(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Centered R² and mean per-window Pearson correlation."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    num = (ac * bc).sum(axis=1)
    den = np.sqrt((ac * ac).sum(axis=1) * (bc * bc).sum(axis=1)).clip(min=1e-12)
    corr = float(np.nanmean(num / den))
    var = float(np.mean(ac * ac))
    mse = float(np.mean((ac - bc) ** 2))
    return 1.0 - mse / max(var, 1e-12), corr


def train_genias(
    x: np.ndarray,
    *,
    hidden: int = 48,
    latent: int = 16,
    steps: int = 400,
    batch_size: int = 64,
    lr: float = 1e-3,
    kl_w: float = 0.01,
    val_frac: float = 0.1,
    seed: int = 0,
    device: str | None = None,
    faithful: bool = False,
    beta: float = 0.1,
    zeta: float = 0.1,
    sigma_prior: float = 0.5,
    delta_min: float = 0.1,
    delta_max: float = 0.2,
    psi_init: float = 2.0,
) -> dict[str, Any]:
    """Train the TCN-VAE.

    ``faithful=False`` (locked S3): reconstruction + standard KL only, ψ chosen
    at sampling time. This is a reconstructor, so ψ=2 barely leaves the parent
    (RMSE 8e-4 in docs/RESULTS.md) and every GenIAS row so far scores a
    near-copy of the donor.

    ``faithful=True`` follows the published objective (Darban et al. 2025,
    Eq. 2–6, UTS defaults): L = α L_recon + β L_perturb + ζ L_compKL with a
    *learned* ψ ≥ 1, the triplet-margin perturbation loss
    max(d(X,X̂) − d(X,X̃) + δ_min, 0) + max(d(X,X̃) − δ_max, 0) with d = MSE,
    and the compact KL against N(0, σ_prior²). γ (zero-perturbation) is 0 for
    univariate series, as in the paper. Distances are in per-window whitened
    units, the same units the reconstruction loss already uses here.
    """
    _require_torch()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected (N, W), got {x.shape}")
    rng = np.random.default_rng(seed)
    n = len(x)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac)) if n > 4 else max(1, n // 5)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    if len(train_idx) == 0:
        train_idx = val_idx

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TCNAE(hidden=hidden, latent=latent).to(device_t)
    # ψ = 1 + softplus(ρ) keeps the perturbation an inflation (ψ ≥ 1).
    psi_raw = torch.nn.Parameter(
        torch.tensor(float(np.log(np.expm1(max(float(psi_init) - 1.0, 1e-3)))), device=device_t)
    )
    params = list(model.parameters()) + ([psi_raw] if faithful else [])
    opt = torch.optim.Adam(params, lr=lr)
    hist_perturb: list[float] = []
    hist_psi: list[float] = []
    ds = TensorDataset(torch.from_numpy(x[train_idx]).unsqueeze(1))
    loader = DataLoader(ds, batch_size=min(batch_size, len(train_idx)), shuffle=True)
    x_val = torch.from_numpy(x[val_idx]).unsqueeze(1).to(device_t)

    model.train()
    step = 0
    last = float("nan")
    last_recon = float("nan")
    last_kl = float("nan")
    log_every = max(1, steps // 200)
    val_every = max(1, steps // 50)
    hist_step: list[int] = []
    hist_loss: list[float] = []
    hist_recon: list[float] = []
    hist_kl: list[float] = []
    hist_val_step: list[int] = []
    hist_val_recon: list[float] = []
    ema = float("nan")
    while step < steps:
        for (xb,) in loader:
            xb = xb.to(device_t)
            if faithful:
                psi_t = 1.0 + F.softplus(psi_raw)
                xn, rec_hat, rec_tilde, ml = model.forward_pair(xb, psi_t)
                mu, logv = ml[0], ml[1]
                recon = F.mse_loss(rec_hat, xn)
                d_hat = ((rec_hat - xn) ** 2).mean(dim=(1, 2))
                d_tilde = ((rec_tilde - xn) ** 2).mean(dim=(1, 2))
                perturb = (
                    F.relu(d_hat - d_tilde + float(delta_min)) + F.relu(d_tilde - float(delta_max))
                ).mean()
                sp2 = float(sigma_prior) ** 2
                kl = -0.5 * (
                    1 + logv - mu.pow(2) - logv.exp() / sp2 + 2.0 * float(np.log(sigma_prior))
                ).mean()
                loss = recon + float(beta) * perturb + float(zeta) * kl
            else:
                rec, mu, logv = model(xb, psi=1.0)
                mean = xb.mean(dim=-1, keepdim=True)
                scale = xb.std(dim=-1, keepdim=True).clamp_min(1e-3)
                recon = F.mse_loss((rec - mean) / scale, (xb - mean) / scale)
                kl = -0.5 * (1 + logv - mu.pow(2) - logv.exp()).mean()
                perturb = torch.zeros((), device=device_t)
                loss = recon + kl_w * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            last = float(loss.item())
            last_recon = float(recon.item())
            last_kl = float(kl.item())
            ema = last if not np.isfinite(ema) else (0.98 * ema + 0.02 * last)
            step += 1
            if step == 1 or step % log_every == 0 or step >= steps:
                hist_step.append(step)
                hist_loss.append(float(ema))
                hist_recon.append(last_recon)
                hist_kl.append(last_kl)
                hist_perturb.append(float(perturb.item()))
                hist_psi.append(float((1.0 + F.softplus(psi_raw)).item()))
            if step == 1 or step % val_every == 0 or step >= steps:
                model.eval()
                with torch.no_grad():
                    vr, _, _ = model(x_val, psi=1.0)
                    vm = x_val.mean(dim=-1, keepdim=True)
                    vs = x_val.std(dim=-1, keepdim=True).clamp_min(1e-3)
                    v_recon = float(F.mse_loss((vr - vm) / vs, (x_val - vm) / vs).item())
                hist_val_step.append(step)
                hist_val_recon.append(v_recon)
                model.train()
            if step >= steps:
                break
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(x[: min(256, len(x))]).unsqueeze(1).to(device_t)
        rec, _, _ = model(xb, psi=1.0)
        recon_mse = float(F.mse_loss(rec, xb).item())
        data_var = float(np.var(x[: min(256, len(x))]))
        recon_r2 = 1.0 - recon_mse / max(data_var, 1e-12)
        shape_r2, shape_corr = _shape_scores(
            xb.squeeze(1).cpu().numpy(), rec.squeeze(1).cpu().numpy()
        )
    return {
        "model": model,
        "device": str(device_t),
        "loss": last,
        "recon_mse": recon_mse,
        "kl": last_kl,
        "val_recon_mse": hist_val_recon[-1] if hist_val_recon else float("nan"),
        "data_var": data_var,
        "recon_r2": recon_r2,
        "shape_r2": shape_r2,
        "shape_corr": shape_corr,
        "steps": steps,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "kl_w": float(kl_w),
        "faithful": bool(faithful),
        "psi_learned": float((1.0 + F.softplus(psi_raw)).item()) if faithful else float("nan"),
        "loss_perturb": np.asarray(hist_perturb, dtype=np.float64),
        "psi_trace": np.asarray(hist_psi, dtype=np.float64),
        "loss_step": np.asarray(hist_step, dtype=np.int64),
        "loss_value": np.asarray(hist_loss, dtype=np.float64),
        "loss_recon": np.asarray(hist_recon, dtype=np.float64),
        "loss_kl": np.asarray(hist_kl, dtype=np.float64),
        "val_step": np.asarray(hist_val_step, dtype=np.int64),
        "val_recon": np.asarray(hist_val_recon, dtype=np.float64),
    }


def deviation_patch(
    parent: np.ndarray,
    generated: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Keep the decode only where it leaves the parent enough (GenIAS Alg. 2).

    Per timestep: use generated if (x − x̃)² > τ · (max(x) − min(x)), else keep x.
    A window with no such timestep stays fully nominal. ``tau <= 0`` is a no-op.
    """
    parent = np.asarray(parent, dtype=np.float32)
    generated = np.asarray(generated, dtype=np.float32)
    if parent.shape != generated.shape:
        raise ValueError(f"parent {parent.shape} vs generated {generated.shape}")
    if tau <= 0:
        return generated
    amp = (parent.max(axis=1) - parent.min(axis=1)).astype(np.float64)
    amp = np.maximum(amp, 1e-12)
    dev = (parent.astype(np.float64) - generated.astype(np.float64)) ** 2
    keep = dev > (float(tau) * amp[:, None])
    return np.where(keep, generated, parent).astype(np.float32)


def genias_patch_alg2(
    parent: np.ndarray,
    generated: np.ndarray,
    tau: float,
    *,
    units: str = "raw",
) -> tuple[np.ndarray, np.ndarray]:
    """GenIAS Algorithm 2 as published: per *dimension*, whole trajectory.

    For each dimension d: X̃_d is kept iff ‖X_d − X̃_d‖²₂ > τ·(max X_d − min X_d),
    otherwise the parent trajectory X_d is returned. For univariate windows
    this degenerates to "replace the whole window or keep the parent", which
    is stated here rather than hidden. ``deviation_patch`` (per timestep) is a
    different rule and stays as the editor's variant.

    The criterion compares a squared norm (units²) with an amplitude (units),
    so τ does not transfer across scalings. ``units="raw"`` applies it to the
    windows as given; ``units="whitened"`` applies it after per-window
    z-scoring of the parent (the units GenIAS trains in here), which makes the
    threshold scale-free. Returns (patched windows, replaced mask).
    """
    x = np.asarray(parent, dtype=np.float64)
    g = np.asarray(generated, dtype=np.float64)
    if x.shape != g.shape:
        raise ValueError(f"parent {x.shape} vs generated {g.shape}")
    if units == "whitened":
        mu = x.mean(axis=1, keepdims=True)
        sd = np.maximum(x.std(axis=1, keepdims=True), 1e-12)
        xs, gs = (x - mu) / sd, (g - mu) / sd
    elif units == "raw":
        xs, gs = x, g
    else:
        raise ValueError(f"units must be raw|whitened, got {units!r}")
    sq = ((xs - gs) ** 2).sum(axis=1)
    amp = xs.max(axis=1) - xs.min(axis=1)
    replace = sq > float(tau) * amp
    out = np.where(replace[:, None], g, x).astype(np.float32)
    return out, replace


def patch_stats(parent: np.ndarray, patched: np.ndarray) -> dict[str, float]:
    parent = np.asarray(parent, dtype=np.float64)
    patched = np.asarray(patched, dtype=np.float64)
    changed = np.abs(patched - parent) > 1e-8
    return {
        "frac_timesteps_patched": float(changed.mean()),
        "frac_windows_patched": float(changed.any(axis=1).mean()) if len(changed) else 0.0,
    }


@torch.no_grad() if torch_available() else (lambda f: f)
def genias_sample(
    model: Any,
    x_cond: np.ndarray,
    *,
    psi: float = 2.0,
    device: str | None = None,
    patch_tau: float = 0.0,
) -> np.ndarray:
    """Inflate latent variance, then optionally deviation-patch onto the parent."""
    _require_torch()
    device_t = torch.device(device or next(model.parameters()).device)
    x_np = np.asarray(x_cond, dtype=np.float32)
    xb = torch.from_numpy(x_np).unsqueeze(1).to(device_t)
    rec, _, _ = model(xb, psi=float(psi))
    y = rec.squeeze(1).cpu().numpy().astype(np.float32)
    if patch_tau > 0 and abs(float(psi) - 1.0) > 1e-9:
        y = deviation_patch(x_np, y, patch_tau)
    return y
