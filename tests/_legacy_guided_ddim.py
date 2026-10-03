"""Frozen copy of steer.guided_ddim at commit a24bf92 (before docs/POSITION_BIAS.md).

Used only by tests/test_steer_equivalence.py to prove that the refactored
sampler with default arguments reproduces the locked runs bit for bit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from anogen.shell.steer import _parent_pull, _prep_grad, anom_energy, soft_energy

def legacy_guided_ddim(
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
    """
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
            if float(lam_parent) != 0.0 and not start_from_noise:
                total = total + float(lam_parent) * _parent_pull(
                    x0_hat - xt0, kind=parent_leash, delta=parent_delta
                ).clamp(-c_max, c_max)
            last_h = h.detach()
            advance = corr == n_correct - 1
            if t_prev < 0 and advance:
                xt = (x0_hat - total).detach() if apply_final_grad else x0_hat.detach()
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


