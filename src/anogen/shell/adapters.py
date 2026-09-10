"""Few-shot adapters and a fold-local anomaly classifier.

Matches the image recipe: the generator sees tail samples, and a
discriminator trained on those samples guides sampling (Sehwag 2022).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from anogen.shell.diffusion import (
    DiffusionSchedule,
    _require_torch,
    q_sample,
    sinusoidal_embedding,
    torch_available,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


if torch_available():

    class ResidualAdapter(nn.Module):
        """Small 1-D CNN residual added to a frozen ε-prediction.

        Not a LoRA factorization: four conv layers, time MLP, channel embed.
        Typical size is ~5–15% of the frozen TSDiff backbone.
        """

        def __init__(self, n_channels: int = 6, hidden: int = 64, time_dim: int = 128) -> None:
            super().__init__()
            self.time_dim = time_dim
            self.time_mlp = nn.Sequential(
                nn.Linear(time_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            self.ch_emb = nn.Embedding(n_channels, hidden)
            self.net = nn.Sequential(
                nn.Conv1d(1, hidden, 5, padding=2),
                nn.SiLU(),
                nn.Conv1d(hidden, hidden, 5, padding=2),
                nn.SiLU(),
                nn.Conv1d(hidden, hidden, 5, padding=2),
                nn.SiLU(),
                nn.Conv1d(hidden, 1, 5, padding=2),
            )

        def forward(
            self, x: "torch.Tensor", t: "torch.Tensor", channel_idx: "torch.Tensor"
        ) -> "torch.Tensor":
            temb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
            h = self.net[0](x)
            h = h + temb.unsqueeze(-1) + self.ch_emb(channel_idx).unsqueeze(-1)
            for layer in self.net[1:]:
                h = layer(h)
            return h

    class AdaptedDenoiser(nn.Module):
        def __init__(self, backbone: nn.Module, adapter: ResidualAdapter) -> None:
            super().__init__()
            self.backbone = backbone
            self.adapter = adapter
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        def forward(
            self, x: "torch.Tensor", t: "torch.Tensor", channel_idx: "torch.Tensor"
        ) -> "torch.Tensor":
            with torch.no_grad():
                base = self.backbone(x, t, channel_idx)
            return base + self.adapter(x, t, channel_idx)

    class AnomalyClassifier(nn.Module):
        """Fold-local P(anomaly | window). Sehwag-style sampling guide."""

        def __init__(self, hidden: int = 32) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, hidden, 7, stride=2, padding=3),
                nn.SiLU(),
                nn.Conv1d(hidden, hidden * 2, 5, stride=2, padding=2),
                nn.SiLU(),
                nn.Conv1d(hidden * 2, hidden * 2, 5, stride=2, padding=2),
                nn.SiLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(hidden * 2, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            h = self.net(x).squeeze(-1)
            return self.head(h).squeeze(-1)

else:  # pragma: no cover

    class ResidualAdapter:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()

    class AdaptedDenoiser:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()

    class AnomalyClassifier:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


def count_trainable(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_adapter(
    backbone: Any,
    schedule: DiffusionSchedule,
    x_anom: np.ndarray,
    ch_anom: np.ndarray,
    x_nom: np.ndarray,
    ch_nom: np.ndarray,
    *,
    n_channels: int,
    hidden: int = 64,
    steps: int = 800,
    batch_size: int = 32,
    lr: float = 1e-3,
    anom_frac: float = 0.5,
    device: str | None = None,
    scaler: Any | None = None,
) -> dict[str, Any]:
    """Denoising loss on train-fold anomalies mixed with matched nominals."""
    _require_torch()
    if len(x_anom) < 2:
        raise ValueError("need at least 2 train-fold anomaly windows for the adapter")
    device_t = torch.device(device or next(backbone.parameters()).device)
    adapter = ResidualAdapter(n_channels=n_channels, hidden=hidden).to(device_t)
    model = AdaptedDenoiser(backbone, adapter).to(device_t)
    opt = torch.optim.Adam(adapter.parameters(), lr=lr)
    xa_np = np.asarray(x_anom, dtype=np.float32)
    xn_np = np.asarray(x_nom, dtype=np.float32)
    if scaler is not None:
        xa_np = scaler.transform(xa_np, ch_anom)
        xn_np = scaler.transform(xn_np, ch_nom)
    xa = torch.from_numpy(xa_np).unsqueeze(1)
    ca = torch.from_numpy(np.asarray(ch_anom, dtype=np.int64))
    xn = torch.from_numpy(xn_np).unsqueeze(1)
    cn = torch.from_numpy(np.asarray(ch_nom, dtype=np.int64))
    n_times = int(schedule.betas.numel())
    sched = schedule.to(device_t) if schedule.betas.device != device_t else schedule
    last = float("nan")
    n_a, n_n = len(xa), len(xn)
    b_a = max(1, int(round(batch_size * anom_frac)))
    b_n = max(1, batch_size - b_a)
    log_every = max(1, steps // 200)
    hist_step: list[int] = []
    hist_loss: list[float] = []
    model.train()
    adapter.train()
    for step in range(1, steps + 1):
        ia = torch.randint(0, n_a, (b_a,))
        inn = torch.randint(0, n_n, (b_n,))
        xb = torch.cat([xa[ia], xn[inn]], dim=0).to(device_t)
        cb = torch.cat([ca[ia], cn[inn]], dim=0).to(device_t)
        t = torch.randint(0, n_times, (xb.size(0),), device=device_t)
        xt, noise = q_sample(xb, t, sched)
        pred = model(xt, t, cb)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.item())
        if step == 1 or step % log_every == 0 or step == steps:
            hist_step.append(step)
            hist_loss.append(last)
    model.eval()
    return {
        "model": model,
        "adapter": adapter,
        "loss": last,
        "n_adapter_params": count_trainable(adapter),
        "device": str(device_t),
        "loss_step": np.asarray(hist_step, dtype=np.int64),
        "loss_value": np.asarray(hist_loss, dtype=np.float64),
    }


def train_classifier(
    x_anom: np.ndarray,
    x_nom: np.ndarray,
    x_rare: np.ndarray | None = None,
    *,
    hidden: int = 32,
    steps: int = 400,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str | None = None,
) -> dict[str, Any]:
    """Binary P(anomaly | window). Rares, if given, are extra negatives.

    Train-fold rares only. Held-out rares stay the S4/S5 collision set.
    """
    _require_torch()
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    clf = AnomalyClassifier(hidden=hidden).to(device_t)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    xa = torch.from_numpy(np.asarray(x_anom, dtype=np.float32)).unsqueeze(1)
    xn = torch.from_numpy(np.asarray(x_nom, dtype=np.float32)).unsqueeze(1)
    use_rare = x_rare is not None and len(x_rare) > 0
    xr = (
        torch.from_numpy(np.asarray(x_rare, dtype=np.float32)).unsqueeze(1)
        if use_rare
        else None
    )
    last = float("nan")
    log_every = max(1, steps // 200)
    hist_step: list[int] = []
    hist_loss: list[float] = []
    clf.train()
    n_pos = max(1, batch_size // 2)
    n_neg = max(1, batch_size - n_pos)
    for step in range(1, steps + 1):
        ia = torch.randint(0, len(xa), (n_pos,))
        if use_rare:
            n_r = max(1, n_neg // 2)
            n_n = max(1, n_neg - n_r)
            inn = torch.randint(0, len(xn), (n_n,))
            ir = torch.randint(0, len(xr), (n_r,))
            x_neg = torch.cat([xn[inn], xr[ir]], dim=0)
        else:
            inn = torch.randint(0, len(xn), (n_neg,))
            x_neg = xn[inn]
        xb = torch.cat([xa[ia], x_neg], dim=0).to(device_t)
        yb = torch.cat(
            [
                torch.ones(len(ia), device=device_t),
                torch.zeros(x_neg.size(0), device=device_t),
            ]
        )
        logit = clf(xb)
        loss = F.binary_cross_entropy_with_logits(logit, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.item())
        if step == 1 or step % log_every == 0 or step == steps:
            hist_step.append(step)
            hist_loss.append(last)
    clf.eval()
    with torch.no_grad():
        acc_a = float((clf(xa.to(device_t)) > 0).float().mean())
        acc_n = float((clf(xn[: min(len(xn), 512)].to(device_t)) < 0).float().mean())
        acc_r = (
            float((clf(xr.to(device_t)) < 0).float().mean()) if use_rare else float("nan")
        )
    return {
        "classifier": clf,
        "loss": last,
        "acc_anomaly": acc_a,
        "acc_nominal": acc_n,
        "acc_rare": acc_r,
        "n_rare": int(len(xr)) if use_rare else 0,
        "device": str(device_t),
        "loss_step": np.asarray(hist_step, dtype=np.int64),
        "loss_value": np.asarray(hist_loss, dtype=np.float64),
    }
