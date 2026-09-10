"""Quantile-band shell encoder and guided DDIM (S2)."""

from __future__ import annotations

from typing import Any

import math

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

    class ConvEncoder(nn.Module):
        """Reconstruction encoder locked by the S0 protocol."""

        def __init__(self, width: int, hidden: int = 32, emb: int = 32) -> None:
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv1d(1, hidden, 7, stride=2, padding=3),
                nn.SiLU(),
                nn.Conv1d(hidden, hidden * 2, 5, stride=2, padding=2),
                nn.SiLU(),
                nn.Conv1d(hidden * 2, hidden * 2, 5, stride=2, padding=2),
                nn.SiLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.proj = nn.Linear(hidden * 2, emb)
            self.dec = nn.Sequential(
                nn.Linear(emb, hidden * 2 * (width // 8)),
                nn.Unflatten(1, (hidden * 2, width // 8)),
                nn.SiLU(),
                nn.ConvTranspose1d(hidden * 2, hidden, 4, stride=2, padding=1),
                nn.SiLU(),
                nn.ConvTranspose1d(hidden, hidden, 4, stride=2, padding=1),
                nn.SiLU(),
                nn.ConvTranspose1d(hidden, 1, 4, stride=2, padding=1),
            )

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            z = self.enc(x).squeeze(-1)
            return self.proj(z)

        def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            z = self.encode(x)
            return z, self.dec(z)

else:  # pragma: no cover

    class ConvEncoder:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


def train_encoder(
    x: np.ndarray,
    *,
    hidden: int = 32,
    emb: int = 32,
    steps: int = 400,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    device: str | None = None,
) -> dict[str, Any]:
    _require_torch()
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    width = int(x.shape[1])
    if width % 8 != 0:
        raise ValueError(f"encoder width must be divisible by 8, got {width}")
    model = ConvEncoder(width, hidden=hidden, emb=emb).to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(1))
    loader = DataLoader(ds, batch_size=min(batch_size, len(x)), shuffle=True)
    model.train()
    step = 0
    last = float("nan")
    log_every = max(1, steps // 200)
    hist_step: list[int] = []
    hist_loss: list[float] = []
    while step < steps:
        for (xb,) in loader:
            xb = xb.to(device_t)
            _, rec = model(xb)
            if rec.shape[-1] != xb.shape[-1]:
                rec = F.interpolate(rec, size=xb.shape[-1], mode="linear", align_corners=False)
            loss = F.mse_loss(rec, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last = float(loss.item())
            step += 1
            if step == 1 or step % log_every == 0 or step >= steps:
                hist_step.append(step)
                hist_loss.append(last)
            if step >= steps:
                break
    model.eval()
    return {
        "encoder": model,
        "device": str(device_t),
        "recon_mse": last,
        "steps": steps,
        "loss_step": np.asarray(hist_step, dtype=np.int64),
        "loss_value": np.asarray(hist_loss, dtype=np.float64),
    }


@torch.no_grad() if torch_available() else (lambda f: f)
def embed_encoder(model: Any, x: np.ndarray, device: str | None = None) -> np.ndarray:
    _require_torch()
    device_t = torch.device(device or next(model.parameters()).device)
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(1).to(device_t)
    z = model.encode(xt)
    return z.cpu().numpy().astype(np.float64)


def soft_energy(
    z: "torch.Tensor",
    ref: "torch.Tensor",
    tau: "torch.Tensor | float",
) -> "torch.Tensor":
    """h_soft = -τ log ∑_r exp(−‖z − e(r)‖² / τ)."""
    d2 = ((z.unsqueeze(1) - ref.unsqueeze(0)) ** 2).sum(dim=-1)
    return -tau * torch.logsumexp(-d2 / tau, dim=1)


def nearest_energy(z: "torch.Tensor", ref: "torch.Tensor") -> "torch.Tensor":
    """h = min_i ‖z − a_i‖². τ → 0 limit of soft_energy. Close to one ref is enough."""
    d2 = ((z.unsqueeze(1) - ref.unsqueeze(0)) ** 2).sum(dim=-1)
    return d2.min(dim=1).values


def knn_energy(
    z: "torch.Tensor",
    ref: "torch.Tensor",
    tau: "torch.Tensor | float",
    k: int = 8,
) -> "torch.Tensor":
    """Soft energy on the k nearest refs only. No pull toward the global barycenter."""
    d2 = ((z.unsqueeze(1) - ref.unsqueeze(0)) ** 2).sum(dim=-1)
    kk = min(max(1, int(k)), int(ref.size(0)))
    near = d2.topk(kk, dim=1, largest=False).values
    return -tau * torch.logsumexp(-near / tau, dim=1)


def proto_energy(z: "torch.Tensor", proto: "torch.Tensor") -> "torch.Tensor":
    """h = ‖z − a_{i(b)}‖². One prototype embedding per batch row."""
    return ((z - proto) ** 2).sum(dim=-1)


def kind_balanced_soft_energy(
    z: "torch.Tensor",
    ref: "torch.Tensor",
    tau: "torch.Tensor | float",
    kind_ids: "torch.Tensor",
) -> "torch.Tensor":
    """Soft energy that weights kinds equally, not windows.

    ``h = −τ log Σ_k π_k mean_{i∈R_k} exp(−‖z−a_i‖²/τ)`` with ``π_k = 1/K``.
    Far away the barycenter is the mean of kind-means, not the campaign swamp.
    """
    ids = kind_ids.to(device=ref.device, dtype=torch.long)
    if ids.numel() != ref.size(0):
        raise ValueError(f"kind_ids length {ids.numel()} != n_ref {ref.size(0)}")
    d2 = ((z.unsqueeze(1) - ref.unsqueeze(0)) ** 2).sum(dim=-1)
    logs: list["torch.Tensor"] = []
    for k in torch.unique(ids).tolist():
        mask = ids == int(k)
        n_k = int(mask.sum())
        log_mean = torch.logsumexp(-d2[:, mask] / tau, dim=1) - math.log(max(n_k, 1))
        logs.append(log_mean)
    if not logs:
        return soft_energy(z, ref, tau)
    stacked = torch.stack(logs, dim=1)
    n_kind = stacked.size(1)
    return -tau * (torch.logsumexp(stacked, dim=1) + math.log(1.0 / n_kind))


def anom_energy(
    z: "torch.Tensor",
    ref: "torch.Tensor",
    tau: "torch.Tensor | float",
    *,
    kind: str = "soft",
    knn: int = 8,
    proto: "torch.Tensor | None" = None,
    kind_ids: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """h_anom under one of the H_ANOM.md / KIND_MIX.md strategies. ``soft`` is locked."""
    name = str(kind or "soft").lower()
    if name == "soft":
        return soft_energy(z, ref, tau)
    if name == "nearest":
        return nearest_energy(z, ref)
    if name == "knn":
        return knn_energy(z, ref, tau, k=knn)
    if name == "proto":
        if proto is None:
            raise ValueError("anom_energy='proto' needs a (B, D) proto tensor")
        return proto_energy(z, proto)
    if name in {"kind_soft", "kindsoft"}:
        if kind_ids is None:
            raise ValueError("anom_energy='kind_soft' needs kind_ids")
        return kind_balanced_soft_energy(z, ref, tau, kind_ids)
    raise ValueError(f"unknown anom_energy {kind!r} (use soft|nearest|knn|proto|kind_soft)")


def band_report(h: np.ndarray, Q_q: float, delta: float) -> dict[str, float]:
    """Where a labeled set sits relative to the frozen nominal band.

    Diagnostic only. Does not set Q_q and must not be used to retune λ/q/δ.
    """
    arr = np.asarray(h, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "frac_in_band": float("nan"),
            "frac_below": float("nan"),
            "frac_above": float("nan"),
        }
    in_band = np.abs(arr - float(Q_q)) <= float(delta)
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "frac_in_band": float(in_band.mean()),
        "frac_below": float((arr < float(Q_q) - float(delta)).mean()),
        "frac_above": float((arr > float(Q_q) + float(delta)).mean()),
    }


def shell_from_nominal(
    encoder: Any,
    x_ref: np.ndarray,
    x_val: np.ndarray,
    *,
    q: float = 0.99,
    tau: float | None = None,
    delta_scale: float = 0.25,
    device: str | None = None,
) -> dict[str, Any]:
    _require_torch()
    device_t = torch.device(device or next(encoder.parameters()).device)
    encoder.eval()
    z_ref = torch.from_numpy(embed_encoder(encoder, x_ref, str(device_t))).to(device_t)
    z_val = torch.from_numpy(embed_encoder(encoder, x_val, str(device_t))).to(device_t)
    if tau is None:
        # Median pairwise distance so the kernel is on a natural scale.
        with torch.no_grad():
            d2 = ((z_ref.unsqueeze(1) - z_ref.unsqueeze(0)) ** 2).sum(-1)
            n = z_ref.size(0)
            off = d2[~torch.eye(n, dtype=torch.bool, device=device_t)]
            tau = float(off.median().sqrt().clamp(min=1e-3).item())
    with torch.no_grad():
        h = soft_energy(z_val, z_ref, tau).cpu().numpy()
    qv = float(np.quantile(h, q))
    iqr = float(np.subtract(*np.percentile(h, [75, 25])))
    delta = max(float(delta_scale) * max(iqr, 1e-6), 1e-4)
    return {
        "Q_q": qv,
        "q": float(q),
        "tau": float(tau),
        "delta": delta,
        "h_val_mean": float(h.mean()),
        "h_val_std": float(h.std()),
        "ref": z_ref.detach(),
    }


def guided_ddim(
    model: Any,
    encoder: Any,
    x0: np.ndarray,
    channel_idx: np.ndarray,
    schedule: Any,
    *,
    ref: "torch.Tensor",
    Q_q: float,
    tau: float,
    nu: float,
    lam: float,
    c_max: float,
    ddim_steps: int = 20,
    device: str | None = None,
    mode: str = "band",
    classifier: Any | None = None,
    lam_cls: float = 0.0,
    normalize_grad: bool = False,
    scaler: Any | None = None,
    n_correct: int = 1,
    ref_anom: "torch.Tensor | None" = None,
    lam_anom: float = 0.0,
    ref_rare: "torch.Tensor | None" = None,
    lam_rare: float = 0.0,
    start_from_noise: bool = False,
    anom_energy_kind: str = "soft",
    anom_knn: int = 8,
    anom_proto_index: "torch.Tensor | None" = None,
    anom_kind_ids: "torch.Tensor | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Noise nominal windows to ν, DDIM with clipped constraint gradients.

    start_from_noise=True ignores the donor waveform: x_T ~ N(0, I)
    in scaled units (channel index still conditions the score).
    mode=band pulls toward Q_q. mode=push maximizes h_soft (no leash).
    classifier is a few-shot anomaly head; its logit is maximized (Sehwag).
    ref_anom / lam_anom attract toward train-fold anomaly embeddings
    (minimize h vs that set). anom_energy_kind selects the H_ANOM.md
    strategy (soft | nearest | knn | proto | kind_soft); default soft is locked S4.
    ref_rare / lam_rare still use full-set soft_energy (repel Rare Events).
    Both refs must be event-OOF.
    n_correct reapplies guidance at the same t before the DDIM advance —
    extra GD steps to land in the constraint, not extra diffusion times.
    Gradients are optionally unit-normalized per sample, as in Sehwag 2022.
    """
    _require_torch()
    from anogen.shell.diffusion import q_sample

    device_t = torch.device(device or next(model.parameters()).device)
    model.eval()
    encoder.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in encoder.parameters():
        p.requires_grad_(False)

    n_times = int(schedule.betas.numel())
    t_start = n_times - 1 if start_from_noise else max(0, min(n_times - 1, int(round(nu * (n_times - 1)))))
    sched = schedule.to(device_t) if schedule.betas.device != device_t else schedule
    x_np = np.asarray(x0, dtype=np.float32)
    ch_np = np.asarray(channel_idx, dtype=np.int64)
    if scaler is not None:
        x_np = scaler.transform(x_np, ch_np)
    xt0 = torch.from_numpy(x_np).unsqueeze(1).to(device_t)
    cb = torch.from_numpy(ch_np).to(device_t)
    if start_from_noise:
        xt = torch.randn_like(xt0)
    else:
        t0 = torch.full((xt0.size(0),), t_start, device=device_t, dtype=torch.long)
        xt, _ = q_sample(xt0, t0, sched)
    ref_t = ref.to(device_t)
    ref_a = ref_anom.to(device_t) if ref_anom is not None else None
    ref_r = ref_rare.to(device_t) if ref_rare is not None else None
    proto_a = None
    if ref_a is not None and str(anom_energy_kind).lower() == "proto":
        if anom_proto_index is None:
            raise ValueError("anom_energy_kind='proto' needs anom_proto_index")
        proto_a = ref_a[torch.as_tensor(anom_proto_index, device=device_t, dtype=torch.long)]
    kind_ids_t = None
    if ref_a is not None and str(anom_energy_kind).lower() in {"kind_soft", "kindsoft"}:
        if anom_kind_ids is None:
            raise ValueError("anom_energy_kind='kind_soft' needs anom_kind_ids")
        kind_ids_t = torch.as_tensor(anom_kind_ids, device=device_t, dtype=torch.long)
    n_correct = max(1, int(n_correct))
    times = np.linspace(t_start, 0, max(1, int(ddim_steps)), dtype=int)
    last_h = None
    for i, t in enumerate(times):
        t_prev = times[i + 1] if i + 1 < len(times) else -1
        for corr in range(n_correct):
            xt = xt.detach().requires_grad_(True)
            t_batch = torch.full((xt.size(0),), int(t), device=device_t, dtype=torch.long)
            eps = model(xt, t_batch, cb)
            ab = sched.alpha_bar[int(t)]
            x0_hat = (xt - (1.0 - ab).sqrt() * eps) / ab.sqrt()
            x0_enc = (
                scaler.inverse_torch(x0_hat.squeeze(1), cb).unsqueeze(1)
                if scaler is not None
                else x0_hat
            )
            z = encoder.encode(x0_enc)
            h = soft_energy(z, ref_t, tau)
            use_anom = ref_a is not None and lam_anom != 0.0
            use_rare = ref_r is not None and lam_rare != 0.0
            use_cls = classifier is not None and lam_cls != 0.0
            extras = int(use_anom) + int(use_rare) + int(use_cls)
            if mode == "off":
                total = torch.zeros_like(xt)
            else:
                err = (-h).mean() if mode == "push" else ((h - Q_q) ** 2).mean()
                grad = torch.autograd.grad(err, xt, retain_graph=extras > 0, allow_unused=True)[0]
                if grad is None:
                    grad = torch.zeros_like(xt)
                total = lam * _prep_grad(grad, c_max, normalize_grad)
            if use_anom:
                extras -= 1
                h_a = anom_energy(
                    z,
                    ref_a,
                    tau,
                    kind=anom_energy_kind,
                    knn=anom_knn,
                    proto=proto_a,
                    kind_ids=kind_ids_t,
                )
                g_a = torch.autograd.grad(h_a.mean(), xt, retain_graph=extras > 0)[0]
                total = total + lam_anom * _prep_grad(g_a, c_max, normalize_grad)
            if use_rare:
                extras -= 1
                h_r = soft_energy(z, ref_r, tau)
                g_r = torch.autograd.grad((-h_r).mean(), xt, retain_graph=extras > 0)[0]
                total = total + lam_rare * _prep_grad(g_r, c_max, normalize_grad)
            if use_cls:
                logit = classifier(x0_enc)
                g_cls = torch.autograd.grad((-logit).mean(), xt)[0]
                total = total + lam_cls * _prep_grad(g_cls, c_max, normalize_grad)
            last_h = h.detach()
            advance = corr == n_correct - 1
            if t_prev < 0 and advance:
                xt = (x0_hat - total).detach()
                break
            if not advance:
                xt = (ab.sqrt() * x0_hat + (1.0 - ab).sqrt() * eps - total).detach()
                continue
            ab_prev = sched.alpha_bar[int(t_prev)]
            xt = (ab_prev.sqrt() * x0_hat + (1.0 - ab_prev).sqrt() * eps - total).detach()
        if t_prev < 0:
            break
    # Occupancy uses the last predicted x0 energy (before the final subtract).
    samples = xt.squeeze(1).cpu().numpy().astype(np.float32)
    if scaler is not None:
        samples = scaler.inverse(samples, ch_np)
    h_np = last_h.cpu().numpy() if last_h is not None else np.zeros(len(samples))
    return samples, h_np


def chunked_guided_ddim(
    model: Any,
    encoder: Any,
    x0: np.ndarray,
    channel_idx: np.ndarray,
    schedule: Any,
    *,
    bsz: int,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """``guided_ddim`` in batches. Extra kwargs are forwarded as-is."""
    xs, hs = [], []
    x0 = np.asarray(x0)
    ch = np.asarray(channel_idx)
    step = max(1, int(bsz))
    proto_idx = kwargs.pop("anom_proto_index", None)
    if str(kwargs.get("anom_energy_kind", "soft")).lower() == "proto":
        ref_a = kwargs.get("ref_anom")
        if ref_a is None:
            raise ValueError("anom_energy_kind='proto' needs ref_anom")
        if proto_idx is None:
            gen = torch.Generator()
            gen.manual_seed(int(kwargs.pop("anom_proto_seed", 0)))
            proto_idx = torch.randint(0, int(ref_a.size(0)), (len(x0),), generator=gen)
        else:
            kwargs.pop("anom_proto_seed", None)
            proto_idx = torch.as_tensor(proto_idx, dtype=torch.long)
    else:
        kwargs.pop("anom_proto_seed", None)
    for i in range(0, len(x0), step):
        extra = dict(kwargs)
        if proto_idx is not None:
            extra["anom_proto_index"] = proto_idx[i : i + step]
        s, h = guided_ddim(model, encoder, x0[i : i + step], ch[i : i + step], schedule, **extra)
        xs.append(s)
        hs.append(h)
    return np.concatenate(xs, axis=0), np.concatenate(hs, axis=0)


def _prep_grad(grad: "torch.Tensor", c_max: float, normalize_grad: bool) -> "torch.Tensor":
    if normalize_grad:
        return _unit_grad(grad, c_max)
    return grad.clamp(-c_max, c_max)


def _unit_grad(grad: "torch.Tensor", c_max: float) -> "torch.Tensor":
    n = grad.reshape(grad.size(0), -1).norm(dim=1).clamp(min=1e-8).view(-1, 1, 1)
    return (grad / n).clamp(-c_max, c_max)
