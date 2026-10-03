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


GUIDANCE_SPACES = ("x", "x0")


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
    lam_parent: float = 0.0,
    parent_leash: str = "l2",
    parent_delta: float = 0.03,
    apply_final_grad: bool = True,
    final_grad_scale: float | None = None,
    guidance_space: str = "x",
    guidance_t_window: tuple[float, float] = (0.0, 1.0),
    burnin_prefix: np.ndarray | None = None,
    burnin_bins: int = 0,
    anom_proto_shift: "torch.Tensor | None" = None,
    lam_repel: float = 0.0,
    repel_project: bool = True,
    noise_init_mean: np.ndarray | None = None,
    noise_init_std: np.ndarray | None = None,
    contrast_weights: np.ndarray | None = None,
    contrast_target: np.ndarray | None = None,
    lam_contrast: float = 0.0,
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
    lam_parent > 0 subtracts a time-domain pull toward the donor (not unit-
    normalized). parent_leash=l2 is μ(x̂_0−x_par); gm saturates large residuals
    so a shelf or needle can survive.
    apply_final_grad=False returns the last Tweedie x̂_0 without subtracting
    ∇f (the kick is otherwise stamped onto the output with no further DDIM).
    Default True keeps locked S4 / parent50 behaviour.

    Additions (docs/POSITION_BIAS.md; all defaults reproduce the frozen runs):

    final_grad_scale   continuous version of apply_final_grad: the last edit is
                       multiplied by this (1.0 = locked, 0.0 = hashfix). When
                       None, apply_final_grad decides (1.0 / 0.0).
    guidance_space     "x"  (locked): ∇ is taken w.r.t. x_t, i.e. through the
                       denoiser Jacobian, and subtracted from x_{t'}.
                       "x0": ∇ is taken w.r.t. the Tweedie x̂_0 itself and the
                       edited x̂_0 is re-noised with the same ε (manifold-
                       preserving / Jacobian-free guidance). With a causal
                       backbone the x-space VJP smears every edit toward the
                       window start; x0-space has no such path, and it skips
                       the backward pass through the denoiser (faster).
    guidance_t_window  (lo, hi) on t / t_start; guidance is only applied on steps
                       inside it (e.g. (0, 0.5) = only the low-noise half).
    burnin_prefix      (N, B) raw-unit telemetry immediately *preceding* each
                       donor. The chain runs on B+W bins; every objective sees
                       only the last W bins of x̂_0 and the prefix is dropped
                       from the output. Gives a causal backbone real context at
                       the window start. With start_from_noise, use burnin_bins
                       instead (the prefix is then pure noise like the rest).
    anom_proto_shift   (N,) integer shift, in encoder time steps, applied to the
                       proto embedding (temporal encoders only; edge-replicate).
                       Moves the requested event in time so that samples
                       sharing one of very few prototypes (6 level shifts) do
                       not all aim at the same position.
    noise_init_mean    per-channel mean (and ``noise_init_std`` std) of the
                       *scaled* training windows. With start_from_noise the
                       chain then starts from N(√ᾱ_T m_c, (1−ᾱ_T+ᾱ_T s_c²) I),
                       the per-bin marginal of q(x_T), instead of N(0, I).
                       The linear schedule leaves √ᾱ_T ≈ 0.36 of the clean
                       signal at t = T−1 (SNR 0.15), so N(0, I) is an input
                       the denoiser never saw (Lin et al., WACV 2024); since
                       φ z-scores every window, the resulting level bias is
                       invisible to ARP / EDI. None keeps the locked N(0, I).
    contrast_weights   (N, W) linear contrasts from ``shell/contrast.py`` (step /
                       spike at a sampled position) with ``contrast_target``
                       (N,) amplitudes in *scaled* units; adds
                       lam_contrast · unit(∇ (x̂₀·w − δ)²). A prototype-free
                       type target with no position collapse.
    lam_repel          particle-guidance repulsion between samples of the same
                       call (Corso et al., ICLR 2024): RBF kernel on encoder
                       embeddings, median bandwidth. With repel_project the
                       component along the band gradient is removed so spread
                       happens along constant shell energy.
    """
    _require_torch()
    from anogen.shell.diffusion import q_sample

    space = str(guidance_space).lower()
    if space not in GUIDANCE_SPACES:
        raise ValueError(f"guidance_space must be one of {GUIDANCE_SPACES}, got {guidance_space!r}")
    if final_grad_scale is None:
        final_scale = 1.0 if apply_final_grad else 0.0
    else:
        final_scale = float(final_grad_scale)
    t_lo, t_hi = (float(v) for v in guidance_t_window)

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
    width = int(x_np.shape[1])
    if burnin_prefix is not None:
        pre = np.asarray(burnin_prefix, dtype=np.float32)
        if pre.ndim != 2 or len(pre) != len(x_np):
            raise ValueError(f"burnin_prefix must be (N, B) with N={len(x_np)}, got {pre.shape}")
        x_np = np.concatenate([pre, x_np], axis=1)
    elif int(burnin_bins) > 0:
        if not start_from_noise:
            raise ValueError("burnin_bins without a prefix needs start_from_noise=True; pass burnin_prefix")
        x_np = np.concatenate([np.repeat(x_np[:, :1], int(burnin_bins), axis=1), x_np], axis=1)
    crop = int(x_np.shape[1]) - width
    if scaler is not None:
        x_np = scaler.transform(x_np, ch_np)
    xt0 = torch.from_numpy(x_np).unsqueeze(1).to(device_t)
    cb = torch.from_numpy(ch_np).to(device_t)
    if start_from_noise:
        xt = torch.randn_like(xt0)
        if noise_init_mean is not None:
            ab_t = sched.alpha_bar[int(t_start)]
            m = torch.as_tensor(np.asarray(noise_init_mean, dtype=np.float32), device=device_t)[cb]
            sd = (
                torch.as_tensor(np.asarray(noise_init_std, dtype=np.float32), device=device_t)[cb]
                if noise_init_std is not None
                else torch.zeros_like(m)
            )
            xt = ab_t.sqrt() * m.view(-1, 1, 1) + (1.0 - ab_t + ab_t * sd.view(-1, 1, 1) ** 2).sqrt() * xt
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
        if anom_proto_shift is not None:
            proto_a = shift_time_embedding(
                proto_a,
                torch.as_tensor(anom_proto_shift, device=device_t, dtype=torch.long),
                _encoder_time_steps(encoder),
            )
    kind_ids_t = None
    if ref_a is not None and str(anom_energy_kind).lower() in {"kind_soft", "kindsoft"}:
        if anom_kind_ids is None:
            raise ValueError("anom_energy_kind='kind_soft' needs anom_kind_ids")
        kind_ids_t = torch.as_tensor(anom_kind_ids, device=device_t, dtype=torch.long)
    n_correct = max(1, int(n_correct))
    times = np.linspace(t_start, 0, max(1, int(ddim_steps)), dtype=int)
    use_anom = ref_a is not None and lam_anom != 0.0
    use_rare = ref_r is not None and lam_rare != 0.0
    use_cls = classifier is not None and lam_cls != 0.0
    use_repel = float(lam_repel) != 0.0 and xt0.size(0) > 1
    use_contrast = contrast_weights is not None and float(lam_contrast) != 0.0
    if use_contrast:
        cw = torch.as_tensor(np.asarray(contrast_weights, dtype=np.float32), device=device_t)
        ct = torch.as_tensor(np.asarray(contrast_target, dtype=np.float32), device=device_t)
        if cw.shape != (xt0.size(0), width) or ct.shape != (xt0.size(0),):
            raise ValueError(f"contrast weights {tuple(cw.shape)} / target {tuple(ct.shape)} do not match ({xt0.size(0)}, {width})")

    def objective_terms(x0_win: "torch.Tensor") -> tuple[list[tuple[float, "torch.Tensor", str]], "torch.Tensor"]:
        """(weight, scalar loss, name) per active term, and h_soft (for occupancy)."""
        x0_enc = (
            scaler.inverse_torch(x0_win.squeeze(1), cb).unsqueeze(1) if scaler is not None else x0_win
        )
        z = encoder.encode(x0_enc)
        h = soft_energy(z, ref_t, tau)
        terms: list[tuple[float, "torch.Tensor", str]] = []
        if mode != "off":
            err = (-h).mean() if mode == "push" else ((h - Q_q) ** 2).mean()
            terms.append((float(lam), err, "band"))
        if use_anom:
            h_a = anom_energy(
                z, ref_a, tau, kind=anom_energy_kind, knn=anom_knn, proto=proto_a, kind_ids=kind_ids_t
            )
            terms.append((float(lam_anom), h_a.mean(), "anom"))
        if use_rare:
            terms.append((float(lam_rare), (-soft_energy(z, ref_r, tau)).mean(), "rare"))
        if use_cls:
            terms.append((float(lam_cls), (-classifier(x0_enc)).mean(), "cls"))
        if use_repel:
            terms.append((float(lam_repel), _repulsion(z), "repel"))
        if use_contrast:
            d = (x0_win.squeeze(1) * cw).sum(dim=-1)
            terms.append((float(lam_contrast), ((d - ct) ** 2).mean(), "contrast"))
        return terms, h

    def guidance(target: "torch.Tensor", terms: list[tuple[float, "torch.Tensor", str]]) -> "torch.Tensor":
        """Σ λ_k prep(∇_target loss_k); repulsion optionally projected off ∇band."""
        total = torch.zeros_like(target)
        band_dir = None
        for k, (weight, loss, name) in enumerate(terms):
            grad = torch.autograd.grad(
                loss, target, retain_graph=k < len(terms) - 1, allow_unused=True
            )[0]
            if grad is None:
                grad = torch.zeros_like(target)
            if name == "band":
                band_dir = grad
            if name == "repel" and repel_project and band_dir is not None:
                grad = _project_out(grad, band_dir)
            total = total + weight * _prep_grad(grad, c_max, normalize_grad)
        return total

    last_h = None
    for i, t in enumerate(times):
        t_prev = times[i + 1] if i + 1 < len(times) else -1
        frac = float(t) / float(max(t_start, 1))
        active = t_lo <= frac <= t_hi
        for corr in range(n_correct):
            t_batch = torch.full((xt.size(0),), int(t), device=device_t, dtype=torch.long)
            ab = sched.alpha_bar[int(t)]
            if space == "x":
                xt = xt.detach().requires_grad_(True)
                eps = model(xt, t_batch, cb)
                x0_hat = (xt - (1.0 - ab).sqrt() * eps) / ab.sqrt()
                x0_win = x0_hat[..., crop:]
                terms, h = objective_terms(x0_win)
                total = guidance(xt, terms) if (active and terms) else torch.zeros_like(xt)
                if float(lam_parent) != 0.0 and not start_from_noise and active:
                    pull = _parent_pull(
                        x0_win - xt0[..., crop:], kind=parent_leash, delta=parent_delta
                    ).clamp(-c_max, c_max)
                    total = total + float(lam_parent) * F.pad(pull, (crop, 0))
                x0_next = x0_hat.detach()
                edit = total.detach()
                eps = eps.detach()
            else:  # x0-space: no backward pass through the denoiser
                with torch.no_grad():
                    eps = model(xt.detach(), t_batch, cb)
                    x0_hat = (xt.detach() - (1.0 - ab).sqrt() * eps) / ab.sqrt()
                leaf = x0_hat[..., crop:].detach().requires_grad_(True)
                terms, h = objective_terms(leaf)
                total = guidance(leaf, terms) if (active and terms) else torch.zeros_like(leaf)
                if float(lam_parent) != 0.0 and not start_from_noise and active:
                    total = total + float(lam_parent) * _parent_pull(
                        leaf.detach() - xt0[..., crop:], kind=parent_leash, delta=parent_delta
                    ).clamp(-c_max, c_max)
                x0_next = x0_hat
                edit = F.pad(total.detach(), (crop, 0))
            last_h = h.detach()
            advance = corr == n_correct - 1
            if t_prev < 0 and advance:
                xt = (x0_next - final_scale * edit).detach()
                break
            if space == "x":
                # Locked algebra: subtract the x_t-gradient from the DDIM output.
                a_next = ab if not advance else sched.alpha_bar[int(t_prev)]
                xt = (a_next.sqrt() * x0_next + (1.0 - a_next).sqrt() * eps - edit).detach()
            else:
                a_next = ab if not advance else sched.alpha_bar[int(t_prev)]
                xt = (a_next.sqrt() * (x0_next - edit) + (1.0 - a_next).sqrt() * eps).detach()
        if t_prev < 0:
            break
    # Occupancy uses the last predicted x0 energy (before the final subtract).
    samples = xt[..., crop:].squeeze(1).detach().cpu().numpy().astype(np.float32)
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
    """``guided_ddim`` in batches. Extra kwargs are forwarded as-is.

    Per-sample arrays (proto index, proto shift, burn-in prefix) are sliced with
    the batch. ``anom_proto_shift_max`` > 0 draws one integer shift per sample
    from U{-m..m} (seeded by ``anom_proto_shift_seed``).
    """
    xs, hs = [], []
    x0 = np.asarray(x0)
    ch = np.asarray(channel_idx)
    step = max(1, int(bsz))
    proto_idx = kwargs.pop("anom_proto_index", None)
    shift_max = int(kwargs.pop("anom_proto_shift_max", 0) or 0)
    shift_seed = int(kwargs.pop("anom_proto_shift_seed", 0) or 0)
    proto_shift = kwargs.pop("anom_proto_shift", None)
    prefix = kwargs.pop("burnin_prefix", None)
    c_w = kwargs.pop("contrast_weights", None)
    c_t = kwargs.pop("contrast_target", None)
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
        if proto_shift is None and shift_max > 0:
            proto_shift = torch.as_tensor(
                proto_shift_draws(len(x0), shift_max, seed=shift_seed), dtype=torch.long
            )
    else:
        kwargs.pop("anom_proto_seed", None)
        proto_shift = None
    if proto_shift is not None:
        proto_shift = torch.as_tensor(proto_shift, dtype=torch.long)
    for i in range(0, len(x0), step):
        extra = dict(kwargs)
        if proto_idx is not None:
            extra["anom_proto_index"] = proto_idx[i : i + step]
        if proto_shift is not None:
            extra["anom_proto_shift"] = proto_shift[i : i + step]
        if prefix is not None:
            extra["burnin_prefix"] = np.asarray(prefix)[i : i + step]
        if c_w is not None:
            extra["contrast_weights"] = np.asarray(c_w)[i : i + step]
            extra["contrast_target"] = np.asarray(c_t)[i : i + step]
        s, h = guided_ddim(model, encoder, x0[i : i + step], ch[i : i + step], schedule, **extra)
        xs.append(s)
        hs.append(h)
    return np.concatenate(xs, axis=0), np.concatenate(hs, axis=0)


def proto_shift_draws(n: int, shift_max: int, *, seed: int = 0) -> np.ndarray:
    """One integer shift per sample, uniform on {-m, ..., m}."""
    m = max(0, int(shift_max))
    rng = np.random.default_rng(int(seed))
    return rng.integers(-m, m + 1, size=int(n)).astype(np.int64)


def shift_time_embedding_np(z: np.ndarray, shifts: np.ndarray, n_steps: int) -> np.ndarray:
    """Numpy twin of ``shift_time_embedding`` (tests run without torch)."""
    z = np.asarray(z)
    n = len(z)
    c = z.shape[1] // int(n_steps)
    if c * int(n_steps) != z.shape[1]:
        raise ValueError(f"embedding dim {z.shape[1]} is not a multiple of {n_steps} time steps")
    zz = z.reshape(n, c, int(n_steps))
    src = np.clip(np.arange(int(n_steps))[None, :] - np.asarray(shifts).reshape(-1, 1), 0, int(n_steps) - 1)
    out = np.take_along_axis(zz, np.repeat(src[:, None, :], c, axis=1), axis=2)
    return out.reshape(n, -1)


def shift_time_embedding(z: "torch.Tensor", shifts: "torch.Tensor", n_steps: int) -> "torch.Tensor":
    """Translate a flattened (C × T) temporal embedding by ``shifts`` steps.

    ``ShellEncoder(pool="time").encode`` returns ``proj(feat).flatten(1)``, i.e.
    channel-major (index c·T + t). Positive shift moves content later in time;
    vacated steps repeat the edge value (no wrap-around, which would teleport
    the end of the window to its start).
    """
    n = z.size(0)
    c = z.size(1) // int(n_steps)
    if c * int(n_steps) != z.size(1):
        raise ValueError(f"embedding dim {z.size(1)} is not a multiple of {n_steps} time steps")
    zz = z.reshape(n, c, int(n_steps))
    t_idx = torch.arange(int(n_steps), device=z.device).unsqueeze(0)
    src = (t_idx - shifts.view(-1, 1).to(z.device)).clamp(0, int(n_steps) - 1)
    out = torch.gather(zz, 2, src.unsqueeze(1).expand(n, c, int(n_steps)))
    return out.reshape(n, -1)


def _encoder_time_steps(encoder: Any) -> int:
    if str(getattr(encoder, "pool", "pool")) != "time":
        raise ValueError("anom_proto_shift needs a temporal encoder (ShellEncoder pool='time')")
    return int(encoder.width) // 8


def _repulsion(z: "torch.Tensor") -> "torch.Tensor":
    """Mean pairwise RBF similarity; minimizing it pushes samples apart."""
    d2 = ((z.unsqueeze(1) - z.unsqueeze(0)) ** 2).sum(dim=-1)
    n = z.size(0)
    off = ~torch.eye(n, dtype=torch.bool, device=z.device)
    bw = d2.detach()[off].median().clamp(min=1e-8)
    return torch.exp(-d2 / bw)[off].mean()


def _project_out(grad: "torch.Tensor", direction: "torch.Tensor") -> "torch.Tensor":
    """Remove each row's component along ``direction`` (per sample)."""
    g = grad.reshape(grad.size(0), -1)
    d = direction.reshape(direction.size(0), -1)
    d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return (g - (g * d).sum(dim=1, keepdim=True) * d).view_as(grad)


def _parent_pull(residual: "torch.Tensor", *, kind: str, delta: float) -> "torch.Tensor":
    """Time-domain donor leash. Not unit-normalized (scale is window units / step)."""
    name = str(kind).lower()
    if name in {"gm", "geman", "cauchy"}:
        d = max(float(delta), 1e-8)
        return residual / (1.0 + (residual / d).pow(2))
    if name in {"l2", "mse", "sq"}:
        return residual
    raise ValueError(f"unknown parent_leash {kind}")


def parent_pull_np(residual: np.ndarray, *, kind: str = "l2", delta: float = 0.03) -> np.ndarray:
    """Numpy twin of ``_parent_pull`` (unit tests, no torch)."""
    r = np.asarray(residual, dtype=np.float64)
    name = str(kind).lower()
    if name in {"gm", "geman", "cauchy"}:
        d = max(float(delta), 1e-8)
        return r / (1.0 + (r / d) ** 2)
    if name in {"l2", "mse", "sq"}:
        return r
    raise ValueError(f"unknown parent_leash {kind}")


def _prep_grad(grad: "torch.Tensor", c_max: float, normalize_grad: bool) -> "torch.Tensor":
    if normalize_grad:
        return _unit_grad(grad, c_max)
    return grad.clamp(-c_max, c_max)


def _unit_grad(grad: "torch.Tensor", c_max: float) -> "torch.Tensor":
    n = grad.reshape(grad.size(0), -1).norm(dim=1).clamp(min=1e-8).view(-1, 1, 1)
    return (grad / n).clamp(-c_max, c_max)
