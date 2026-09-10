"""Encoder ablations for steering: pooled vs temporal, recon vs SupCon.

Does not replace the locked S2 ``ConvEncoder``. Energy still uses ``soft_energy``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from anogen.shell.diffusion import _require_torch, torch_available
from anogen.shell.steer import soft_energy

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


PoolKind = Literal["pool", "time"]
LossKind = Literal["recon", "supcon", "both"]


if torch_available():

    class ShellEncoder(nn.Module):
        """1-D conv encoder. ``pool`` matches S2; ``time`` keeps W/8 feature map."""

        def __init__(
            self,
            width: int,
            *,
            hidden: int = 32,
            emb: int = 32,
            time_emb: int = 8,
            pool: PoolKind = "pool",
        ) -> None:
            super().__init__()
            if width % 8 != 0:
                raise ValueError(f"width must be divisible by 8, got {width}")
            self.width = width
            self.pool = pool
            self.hidden = hidden
            self.emb = emb
            self.time_emb = time_emb
            self.backbone = nn.Sequential(
                nn.Conv1d(1, hidden, 7, stride=2, padding=3),
                nn.SiLU(),
                nn.Conv1d(hidden, hidden * 2, 5, stride=2, padding=2),
                nn.SiLU(),
                nn.Conv1d(hidden * 2, hidden * 2, 5, stride=2, padding=2),
                nn.SiLU(),
            )
            c = hidden * 2
            t = width // 8
            if pool == "pool":
                self.gap = nn.AdaptiveAvgPool1d(1)
                self.proj = nn.Linear(c, emb)
                self.dec = nn.Sequential(
                    nn.Linear(emb, c * t),
                    nn.Unflatten(1, (c, t)),
                    nn.SiLU(),
                    nn.ConvTranspose1d(c, hidden, 4, stride=2, padding=1),
                    nn.SiLU(),
                    nn.ConvTranspose1d(hidden, hidden, 4, stride=2, padding=1),
                    nn.SiLU(),
                    nn.ConvTranspose1d(hidden, 1, 4, stride=2, padding=1),
                )
            else:
                self.gap = None
                self.proj = nn.Conv1d(c, time_emb, 1)
                self.dec = nn.Sequential(
                    nn.SiLU(),
                    nn.ConvTranspose1d(c, hidden, 4, stride=2, padding=1),
                    nn.SiLU(),
                    nn.ConvTranspose1d(hidden, hidden, 4, stride=2, padding=1),
                    nn.SiLU(),
                    nn.ConvTranspose1d(hidden, 1, 4, stride=2, padding=1),
                )

        def features(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.backbone(x)

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            feat = self.features(x)
            if self.pool == "pool":
                return self.proj(self.gap(feat).squeeze(-1))
            return self.proj(feat).flatten(1)

        def decode(self, x: "torch.Tensor") -> "torch.Tensor":
            feat = self.features(x)
            if self.pool == "pool":
                z = self.proj(self.gap(feat).squeeze(-1))
                rec = self.dec(z)
            else:
                rec = self.dec(feat)
            if rec.shape[-1] != x.shape[-1]:
                rec = F.interpolate(rec, size=x.shape[-1], mode="linear", align_corners=False)
            return rec

        def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            return self.encode(x), self.decode(x)

else:  # pragma: no cover

    class ShellEncoder:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


def encoder_spec(name: str) -> dict[str, Any]:
    table = {
        "pool_recon": {"pool": "pool", "loss": "recon", "uses_labels": False},
        "time_recon": {"pool": "time", "loss": "recon", "uses_labels": False},
        "pool_supcon": {"pool": "pool", "loss": "supcon", "uses_labels": True},
        "time_supcon": {"pool": "time", "loss": "supcon", "uses_labels": True},
        "time_both": {"pool": "time", "loss": "both", "uses_labels": True},
    }
    if name not in table:
        raise ValueError(f"unknown encoder {name}")
    return {"name": name, **table[name]}


def all_encoder_names() -> list[str]:
    return ["pool_recon", "time_recon", "pool_supcon", "time_supcon", "time_both"]


def supcon_loss(z: "torch.Tensor", y: "torch.Tensor", temperature: float = 0.1) -> "torch.Tensor":
    """Supervised contrastive (Khosla et al.): everyday / rare / anomaly."""
    z = F.normalize(z, dim=1)
    n = z.size(0)
    if n < 2:
        return z.new_zeros(())
    sim = z @ z.T / float(temperature)
    logits_mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    log_prob = sim - torch.log((torch.exp(sim) * logits_mask).sum(dim=1, keepdim=True).clamp(min=1e-8))
    pos = (y.unsqueeze(0) == y.unsqueeze(1)) & logits_mask
    pos_n = pos.sum(dim=1).clamp(min=1)
    loss = -(log_prob * pos).sum(dim=1) / pos_n
    has_pos = pos.sum(dim=1) > 0
    if not bool(has_pos.any()):
        return z.new_zeros(())
    return loss[has_pos].mean()


def train_shell_encoder(
    x_nom: np.ndarray,
    x_rare: np.ndarray | None = None,
    x_anom: np.ndarray | None = None,
    *,
    pool: PoolKind = "pool",
    loss: LossKind = "recon",
    hidden: int = 32,
    emb: int = 32,
    time_emb: int = 8,
    steps: int = 800,
    batch_size: int = 64,
    per_class: int = 32,
    lr: float = 1e-3,
    temperature: float = 0.1,
    recon_weight: float = 1.0,
    supcon_weight: float = 1.0,
    device: str | None = None,
) -> dict[str, Any]:
    _require_torch()
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    width = int(x_nom.shape[1])
    model = ShellEncoder(width, hidden=hidden, emb=emb, time_emb=time_emb, pool=pool).to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xn = torch.from_numpy(np.asarray(x_nom, dtype=np.float32))
    xr = torch.from_numpy(np.asarray(x_rare, dtype=np.float32)) if x_rare is not None and len(x_rare) else None
    xa = torch.from_numpy(np.asarray(x_anom, dtype=np.float32)) if x_anom is not None and len(x_anom) else None
    rng = np.random.default_rng(0)
    hist_step: list[int] = []
    hist_loss: list[float] = []
    hist_recon: list[float] = []
    hist_con: list[float] = []
    model.train()
    last = 0.0
    log_every = max(1, steps // 200)
    for step in range(1, steps + 1):
        if loss == "recon":
            idx = rng.choice(len(xn), size=min(batch_size, len(xn)), replace=False)
            xb = xn[idx].unsqueeze(1).to(device_t)
            rec = model.decode(xb)
            l_rec = F.mse_loss(rec, xb)
            l_con = xb.new_zeros(())
            total = recon_weight * l_rec
        else:
            chunks, ys = _balanced_batch(xn, xr, xa, per_class, rng, device_t)
            z = model.encode(chunks)
            l_con = supcon_loss(z, ys, temperature)
            if loss == "both":
                rec = model.decode(chunks)
                l_rec = F.mse_loss(rec, chunks)
            else:
                l_rec = chunks.new_zeros(())
            total = recon_weight * l_rec + supcon_weight * l_con
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        last = float(total.item())
        if step == 1 or step % log_every == 0 or step == steps:
            hist_step.append(step)
            hist_loss.append(last)
            hist_recon.append(float(l_rec.detach().item()))
            hist_con.append(float(l_con.detach().item()))
    model.eval()
    return {
        "encoder": model,
        "device": str(device_t),
        "loss": last,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "embed_dim": int(model.encode(xn[:1].unsqueeze(1).to(device_t)).shape[1]),
        "loss_step": np.asarray(hist_step, dtype=np.int64),
        "loss_value": np.asarray(hist_loss, dtype=np.float64),
        "loss_recon": np.asarray(hist_recon, dtype=np.float64),
        "loss_supcon": np.asarray(hist_con, dtype=np.float64),
    }


@torch.no_grad() if torch_available() else (lambda f: f)
def embed_shell(model: Any, x: np.ndarray, device: str | None = None) -> np.ndarray:
    _require_torch()
    device_t = torch.device(device or next(model.parameters()).device)
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(1).to(device_t)
    return model.encode(xt).cpu().numpy().astype(np.float64)


def h_of_windows(
    model: Any,
    x: np.ndarray,
    ref: "torch.Tensor",
    tau: float,
    device: str,
) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(0, dtype=np.float64)
    z = torch.from_numpy(embed_shell(model, x, device)).to(ref.device)
    return soft_energy(z, ref, tau).detach().cpu().numpy()


def auroc_scores(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC for 'higher score = more positive'. Mann–Whitney, no sklearn."""
    p = np.asarray(pos, dtype=np.float64).reshape(-1)
    n = np.asarray(neg, dtype=np.float64).reshape(-1)
    if p.size == 0 or n.size == 0:
        return float("nan")
    # P(pos > neg) + 0.5 P(tie)
    # vectorised via ranks
    allv = np.concatenate([p, n])
    ranks = allv.argsort().argsort().astype(np.float64) + 1.0
    rank_pos = ranks[: p.size].sum()
    u = rank_pos - p.size * (p.size + 1) / 2.0
    return float(u / (p.size * n.size))


def gradient_locality(
    model: Any,
    x: np.ndarray,
    ref: "torch.Tensor",
    tau: float,
    device: str,
    *,
    top_frac: float = 0.1,
) -> float:
    """Fraction of ‖∇_x h‖² mass on the top-``top_frac`` timesteps."""
    _require_torch()
    device_t = torch.device(device)
    model.eval()
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(1).to(device_t)
    xt.requires_grad_(True)
    z = model.encode(xt)
    h = soft_energy(z, ref.to(device_t), tau).mean()
    g = torch.autograd.grad(h, xt, allow_unused=True)[0]
    if g is None:
        return float("nan")
    g2 = (g.detach().squeeze(1) ** 2)
    mass = g2.sum(dim=1).clamp(min=1e-12)
    k = max(1, int(round(top_frac * g2.size(1))))
    top = torch.topk(g2, k, dim=1).values.sum(dim=1)
    return float((top / mass).mean().cpu())


def _balanced_batch(
    xn: "torch.Tensor",
    xr: "torch.Tensor | None",
    xa: "torch.Tensor | None",
    per_class: int,
    rng: np.random.Generator,
    device: "torch.device",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    parts: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for tensor, lab in ((xn, 0), (xr, 1), (xa, 2)):
        if tensor is None or len(tensor) == 0:
            continue
        n = min(int(per_class), len(tensor))
        idx = rng.choice(len(tensor), size=n, replace=len(tensor) < n)
        parts.append(tensor[idx].unsqueeze(1))
        ys.append(torch.full((n,), lab, dtype=torch.long))
    x = torch.cat(parts, dim=0).to(device)
    y = torch.cat(ys, dim=0).to(device)
    return x, y
