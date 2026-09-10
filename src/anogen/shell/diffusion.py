"""Univariate 1-D denoiser. Used by S1 (unguided) and later S2 (guided)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, TensorDataset

    _TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH = False


def torch_available() -> bool:
    return _TORCH


def _require_torch() -> None:
    if not _TORCH:
        raise ImportError("torch is required for S1. uv sync --extra neural")


def sinusoidal_embedding(t: "torch.Tensor", dim: int) -> "torch.Tensor":
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


if _TORCH:

    class ConvBlock(nn.Module):
        def __init__(self, cin: int, cout: int, time_dim: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(cin, cout, 3, padding=1)
            self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
            self.norm1 = nn.GroupNorm(min(8, cout), cout)
            self.norm2 = nn.GroupNorm(min(8, cout), cout)
            self.time = nn.Linear(time_dim, cout)
            self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

        def forward(self, x: "torch.Tensor", temb: "torch.Tensor") -> "torch.Tensor":
            h = self.norm1(self.conv1(x))
            h = h + self.time(temb).unsqueeze(-1)
            h = F.silu(h)
            h = F.silu(self.norm2(self.conv2(h)))
            return h + self.skip(x)

    class UNet1D(nn.Module):
        """Small U-Net. W must be divisible by 4."""

        def __init__(self, hidden: int = 48, n_channels: int = 6, time_dim: int = 128) -> None:
            super().__init__()
            self.time_dim = time_dim
            self.time_mlp = nn.Sequential(
                nn.Linear(time_dim, time_dim * 2),
                nn.SiLU(),
                nn.Linear(time_dim * 2, time_dim),
            )
            self.ch_emb = nn.Embedding(n_channels, hidden)
            h, h2, h3 = hidden, hidden * 2, hidden * 4
            self.inc = ConvBlock(1, h, time_dim)
            self.down1 = nn.Conv1d(h, h2, 4, stride=2, padding=1)
            self.b1 = ConvBlock(h2, h2, time_dim)
            self.down2 = nn.Conv1d(h2, h3, 4, stride=2, padding=1)
            self.bot = ConvBlock(h3, h3, time_dim)
            self.up2 = nn.ConvTranspose1d(h3, h2, 4, stride=2, padding=1)
            self.b2 = ConvBlock(h2 * 2, h2, time_dim)
            self.up1 = nn.ConvTranspose1d(h2, h, 4, stride=2, padding=1)
            self.b3 = ConvBlock(h * 2, h, time_dim)
            self.out = nn.Conv1d(h, 1, 1)

        def forward(
            self, x: "torch.Tensor", t: "torch.Tensor", channel_idx: "torch.Tensor"
        ) -> "torch.Tensor":
            # x: (B, 1, W)
            temb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
            h = self.inc(x, temb)
            h = h + self.ch_emb(channel_idx).unsqueeze(-1)
            d1 = self.b1(self.down1(h), temb)
            d2 = self.bot(self.down2(d1), temb)
            u2 = self.up2(d2)
            u2 = self.b2(torch.cat([u2, d1], dim=1), temb)
            u1 = self.up1(u2)
            u1 = self.b3(torch.cat([u1, h], dim=1), temb)
            return self.out(u1)

else:  # pragma: no cover

    class UNet1D:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


@dataclass
class DiffusionSchedule:
    betas: Any
    alphas: Any
    alpha_bar: Any

    @classmethod
    def linear(cls, n_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> DiffusionSchedule:
        _require_torch()
        betas = torch.linspace(beta_start, beta_end, n_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        return cls(betas=betas, alphas=alphas, alpha_bar=alpha_bar)

    def to(self, device: Any) -> DiffusionSchedule:
        return DiffusionSchedule(
            betas=self.betas.to(device),
            alphas=self.alphas.to(device),
            alpha_bar=self.alpha_bar.to(device),
        )


def q_sample(
    x0: "torch.Tensor",
    t: "torch.Tensor",
    schedule: DiffusionSchedule,
    noise: "torch.Tensor | None" = None,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    if noise is None:
        noise = torch.randn_like(x0)
    ab = schedule.alpha_bar[t].view(-1, 1, 1)
    xt = ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise
    return xt, noise


@torch.no_grad() if _TORCH else (lambda f: f)
def ddim_sample(
    model: Any,
    x: "torch.Tensor",
    channel_idx: "torch.Tensor",
    schedule: DiffusionSchedule,
    *,
    n_steps: int | None = None,
    t_start: int | None = None,
) -> "torch.Tensor":
    """Deterministic DDIM. ``x`` is already noised to ``t_start`` (or T-1)."""
    _require_torch()
    n_train = int(schedule.betas.numel())
    t_start = n_train - 1 if t_start is None else int(t_start)
    n_steps = n_steps or min(50, t_start + 1)
    times = np.linspace(t_start, 0, n_steps, dtype=int)
    xt = x
    for i, t in enumerate(times):
        t_prev = times[i + 1] if i + 1 < len(times) else -1
        t_batch = torch.full((xt.size(0),), int(t), device=xt.device, dtype=torch.long)
        eps = model(xt, t_batch, channel_idx)
        ab = schedule.alpha_bar[t]
        x0 = (xt - (1.0 - ab).sqrt() * eps) / ab.sqrt()
        if t_prev < 0:
            xt = x0
            break
        ab_prev = schedule.alpha_bar[t_prev]
        xt = ab_prev.sqrt() * x0 + (1.0 - ab_prev).sqrt() * eps
    return xt


def count_params(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def build_denoiser(
    *,
    backbone: str = "unet",
    hidden: int = 48,
    n_channels: int = 6,
    n_layers: int = 6,
    d_state: int = 64,
    time_dim: int = 128,
    state_dict: Any | None = None,
    device: Any | None = None,
) -> Any:
    """Construct UNet1D or the TSDiff S4-D score. Old checkpoints default to unet."""
    _require_torch()
    name = (backbone or "unet").lower()
    if name in {"tsdiff", "s4"}:
        from anogen.shell.tsdiff import TSDiffBackbone

        model = TSDiffBackbone(
            hidden=hidden,
            n_channels=n_channels,
            n_layers=n_layers,
            d_state=d_state,
            time_dim=time_dim,
        )
    else:
        model = UNet1D(hidden=hidden, n_channels=n_channels, time_dim=time_dim)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    if device is not None:
        model.to(device)
    return model


def denoiser_from_ckpt(ckpt: dict[str, Any], device: Any | None = None) -> Any:
    return build_denoiser(
        backbone=str(ckpt.get("backbone", "unet")),
        hidden=int(ckpt.get("hidden", 48)),
        n_channels=int(ckpt["n_channels"]),
        n_layers=int(ckpt.get("n_layers", 6)),
        d_state=int(ckpt.get("d_state", 64)),
        time_dim=int(ckpt.get("time_dim", 128)),
        state_dict=ckpt["state_dict"],
        device=device,
    )


def train_denoiser(
    x: np.ndarray,
    channel_idx: np.ndarray,
    *,
    n_channels: int,
    hidden: int = 48,
    n_times: int = 200,
    steps: int = 200,
    batch_size: int = 32,
    lr: float = 2e-4,
    seed: int = 0,
    val_frac: float = 0.1,
    device: str | None = None,
    backbone: str = "unet",
    n_layers: int = 6,
    d_state: int = 64,
    scaler: Any | None = None,
) -> dict[str, Any]:
    """Train ε-prediction. Returns model, schedule, and scalar logs."""
    _require_torch()
    if x.ndim != 2:
        raise ValueError(f"expected (N, W), got {x.shape}")
    if scaler is not None:
        x = scaler.transform(x, channel_idx)
    rng = np.random.default_rng(seed)
    n = len(x)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    if len(train_idx) == 0:
        train_idx = val_idx

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_denoiser(
        backbone=backbone,
        hidden=hidden,
        n_channels=n_channels,
        n_layers=n_layers,
        d_state=d_state,
    ).to(device_t)
    schedule = DiffusionSchedule.linear(n_times).to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def _loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        xt = torch.from_numpy(np.asarray(x[idx], dtype=np.float32)).unsqueeze(1)
        ct = torch.from_numpy(np.asarray(channel_idx[idx], dtype=np.int64))
        ds: Dataset = TensorDataset(xt, ct)
        return DataLoader(ds, batch_size=min(batch_size, len(idx)), shuffle=shuffle)

    train_loader = _loader(train_idx, True)
    val_loader = _loader(val_idx, False)
    model.train()
    step = 0
    last_loss = float("nan")
    log_every = max(1, steps // 200)
    val_every = max(1, steps // 50)
    hist_step: list[int] = []
    hist_loss: list[float] = []
    hist_val_step: list[int] = []
    hist_val: list[float] = []
    ema = float("nan")
    while step < steps:
        for xb, cb in train_loader:
            xb = xb.to(device_t)
            cb = cb.to(device_t)
            t = torch.randint(0, n_times, (xb.size(0),), device=device_t)
            xt, noise = q_sample(xb, t, schedule)
            pred = model(xt, t, cb)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_loss = float(loss.item())
            ema = last_loss if not np.isfinite(ema) else (0.98 * ema + 0.02 * last_loss)
            step += 1
            if step == 1 or step % log_every == 0 or step >= steps:
                hist_step.append(step)
                hist_loss.append(float(ema))
            if step == 1 or step % val_every == 0 or step >= steps:
                model.eval()
                vm = _val_mse(model, schedule, val_loader, device_t, n_times)
                hist_val_step.append(step)
                hist_val.append(vm)
                model.train()
            if step >= steps:
                break

    model.eval()
    val_mse = hist_val[-1] if hist_val else _val_mse(model, schedule, val_loader, device_t, n_times)
    loss_by_t = _val_mse_by_t(model, schedule, val_loader, device_t, n_times, n_bins=8)
    return {
        "model": model,
        "schedule": schedule,
        "device": str(device_t),
        "n_params": count_params(model),
        "backbone": (backbone or "unet").lower(),
        "n_layers": int(n_layers),
        "d_state": int(d_state),
        "steps": steps,
        "train_loss": last_loss,
        "train_loss_ema": float(ema),
        "val_mse": val_mse,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "loss_step": np.asarray(hist_step, dtype=np.int64),
        "loss_value": np.asarray(hist_loss, dtype=np.float64),
        "val_step": np.asarray(hist_val_step, dtype=np.int64),
        "val_value": np.asarray(hist_val, dtype=np.float64),
        "loss_by_t_lo": np.asarray(loss_by_t["lo"], dtype=np.int64),
        "loss_by_t_hi": np.asarray(loss_by_t["hi"], dtype=np.int64),
        "loss_by_t": np.asarray(loss_by_t["mse"], dtype=np.float64),
    }


@torch.no_grad() if _TORCH else (lambda f: f)
def _val_mse_by_t(
    model: Any,
    schedule: DiffusionSchedule,
    loader: Any,
    device: Any,
    n_times: int,
    n_bins: int = 8,
) -> dict[str, Any]:
    """Mean ε-MSE on the val set, bucketed by diffusion time."""
    _require_torch()
    edges = np.linspace(0, n_times, n_bins + 1, dtype=int)
    totals = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    for xb, cb in loader:
        xb = xb.to(device)
        cb = cb.to(device)
        t = torch.randint(0, n_times, (xb.size(0),), device=device)
        xt, noise = q_sample(xb, t, schedule)
        pred = model(xt, t, cb)
        per = F.mse_loss(pred, noise, reduction="none").mean(dim=(1, 2))
        tb = t.detach().cpu().numpy()
        pb = per.detach().cpu().numpy()
        bins = np.clip(np.searchsorted(edges, tb, side="right") - 1, 0, n_bins - 1)
        for i in range(n_bins):
            m = bins == i
            if m.any():
                totals[i] += float(pb[m].sum())
                counts[i] += float(m.sum())
    mse = np.divide(totals, np.maximum(counts, 1.0))
    return {"lo": edges[:-1], "hi": edges[1:], "mse": mse}


@torch.no_grad() if _TORCH else (lambda f: f)
def _val_mse(model: Any, schedule: DiffusionSchedule, loader: Any, device: Any, n_times: int) -> float:
    _require_torch()
    total = 0.0
    n = 0
    for xb, cb in loader:
        xb = xb.to(device)
        cb = cb.to(device)
        t = torch.randint(0, n_times, (xb.size(0),), device=device)
        xt, noise = q_sample(xb, t, schedule)
        pred = model(xt, t, cb)
        total += float(F.mse_loss(pred, noise, reduction="sum").item())
        n += xb.numel()
    return total / max(n, 1)


@torch.no_grad() if _TORCH else (lambda f: f)
def unguided_from_nominal(
    model: Any,
    schedule: DiffusionSchedule,
    x0: np.ndarray,
    channel_idx: np.ndarray,
    *,
    nu: float = 1.0,
    ddim_steps: int = 20,
    device: str | None = None,
    scaler: Any | None = None,
) -> np.ndarray:
    """Noise nominal windows to ``nu`` of the schedule, then DDIM back."""
    _require_torch()
    device_t = torch.device(device or next(model.parameters()).device)
    n_times = int(schedule.betas.numel())
    t_start = max(0, min(n_times - 1, int(round(nu * (n_times - 1)))))
    x_in = np.asarray(x0, dtype=np.float32)
    ch = np.asarray(channel_idx, dtype=np.int64)
    if scaler is not None:
        x_in = scaler.transform(x_in, ch)
    xt = torch.from_numpy(x_in).unsqueeze(1).to(device_t)
    cb = torch.from_numpy(ch).to(device_t)
    t = torch.full((xt.size(0),), t_start, device=device_t, dtype=torch.long)
    noised, _ = q_sample(xt, t, schedule.to(device_t) if schedule.betas.device != device_t else schedule)
    out = ddim_sample(
        model,
        noised,
        cb,
        schedule.to(device_t) if schedule.betas.device != device_t else schedule,
        n_steps=ddim_steps,
        t_start=t_start,
    )
    y = out.squeeze(1).cpu().numpy().astype(np.float32)
    if scaler is not None:
        y = scaler.inverse(y, ch)
    return y
