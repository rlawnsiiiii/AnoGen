"""S3: GenIAS and post-hoc reference galleries on the S0 window API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anogen.config import REPO_ROOT, panel_path
from anogen.shell.data import load_panel
from anogen.shell.baselines import posthoc_inject
from anogen.shell.diffusion import torch_available
from anogen.shell.genias import genias_sample, patch_stats, train_genias
from anogen.shell.plots import save_loss_curves, save_same_parent
from anogen.shell.windows import load_train_index, materialize


def run_s3(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    out = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    out.mkdir(parents=True, exist_ok=True)

    if not torch_available():
        report = {"ok": False, "skipped": True, "reason": "torch is not installed"}
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    if not (s0 / "protocol.json").is_file() or not (s0 / "nominal_index.csv").is_file():
        report = {"ok": False, "skipped": True, "reason": f"S0 missing under {s0}"}
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    proto = json.loads((s0 / "protocol.json").read_text())
    width = int(proto["W"])
    n_gen = int(proto["N_generate"])
    channels = list(cfg.get("channels") or [])
    splits = cfg.get("splits") or {}
    panel = load_panel(
        panel_path(cfg),
        channels=channels,
        official_train_end=splits.get("official_train_end", "2007-01-01"),
        allow_test_telemetry=False,
        bin_seconds=int(cfg.get("bin_seconds", 30)),
    )
    # Everyday donors for the shared cond gallery. Train GenIAS on the full
    # valid pool (nominals + rares) when train_index.csv exists.
    nominal = pd.read_csv(s0 / "nominal_index.csv")
    train = load_train_index(s0)
    x_all = materialize(panel, train, width)
    x_nom = materialize(panel, nominal, width)
    ch_nom = nominal["channel_idx"].to_numpy(dtype=np.int64)

    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    starts, chs = [], []
    for cidx, _ch in enumerate(panel.channels):
        idx = np.where(ch_nom == cidx)[0]
        if len(idx) < n_gen:
            raise RuntimeError(f"not enough nominal windows for {panel.channels[cidx]}")
        pick = rng.choice(idx, size=n_gen, replace=False)
        starts.append(pick)
        chs.append(np.full(n_gen, cidx))
    cond_idx = np.concatenate(starts)
    cond_ch = np.concatenate(chs)
    x_cond = x_nom[cond_idx]

    gcfg = dict((cfg.get("shell") or {}).get("genias") or {})
    gia = train_genias(
        x_all,
        hidden=int(gcfg.get("hidden", 48)),
        latent=int(gcfg.get("latent", 16)),
        steps=int(gcfg.get("steps", 400)),
        batch_size=int(gcfg.get("batch_size", 64)),
        lr=float(gcfg.get("lr", 1e-3)),
        kl_w=float(gcfg.get("kl_w", 0.01)),
        seed=int(cfg.get("seed", 0)),
    )
    psi_main = float(gcfg.get("psi", 2.0))
    patch_tau = float(gcfg.get("patch_tau", 0.2))
    genias_x = genias_sample(
        gia["model"], x_cond, psi=psi_main, device=gia["device"], patch_tau=patch_tau
    )
    extras = {}
    for psi in gcfg.get("psi_extra") or [1.0, 1.5]:
        extras[f"genias_psi{psi}"] = genias_sample(
            gia["model"],
            x_cond,
            psi=float(psi),
            device=gia["device"],
            patch_tau=patch_tau,
        )
    post = posthoc_inject(x_cond, rng=rng)

    import torch

    np.savez_compressed(
        out / "galleries.npz",
        posthoc=post,
        genias=genias_x,
        channel_idx=cond_ch,
        cond=x_cond,
        **extras,
    )
    torch.save(
        {
            "state_dict": gia["model"].state_dict(),
            "hidden": int(gcfg.get("hidden", 48)),
            "latent": int(gcfg.get("latent", 16)),
            "steps": int(gcfg.get("steps", 400)),
            "arch": "spatial_tcn_vae",
        },
        out / "genias.pt",
    )
    loss_hist = {
        k: gia[k]
        for k in ("loss_step", "loss_value", "loss_recon", "loss_kl", "val_step", "val_recon")
        if k in gia
    }
    if loss_hist:
        np.savez_compressed(out / "loss_history.npz", **loss_hist)
        _write_s3_curves(out, loss_hist, parent=x_cond, recon=_psi1(extras), sample=genias_x)

    recon = _psi1(extras)
    recon_rmse = float(np.sqrt(np.mean((recon - x_cond) ** 2))) if recon is not None else float("nan")
    sample_rmse = float(np.sqrt(np.mean((genias_x - x_cond) ** 2)))
    report = {
        "ok": True,
        "skipped": False,
        "N_generate": n_gen,
        "n_total": int(len(x_cond)),
        "n_train_windows": int(len(x_all)),
        "genias_loss": gia["loss"],
        "genias_recon_mse": gia.get("recon_mse"),
        "genias_val_recon_mse": gia.get("val_recon_mse"),
        "genias_kl": gia.get("kl"),
        "genias_recon_r2": gia.get("recon_r2"),
        "genias_shape_r2": gia.get("shape_r2"),
        "genias_shape_corr": gia.get("shape_corr"),
        "genias_data_var": gia.get("data_var"),
        "genias_steps": gia.get("steps"),
        "psi": psi_main,
        "patch_tau": patch_tau,
        **{f"patch_{k}": v for k, v in patch_stats(x_cond, genias_x).items()},
        "recon_rmse_psi1": recon_rmse,
        "sample_rmse_psi2": sample_rmse,
        "psi2_moves_off_parent": bool(sample_rmse > 1.25 * recon_rmse) if np.isfinite(recon_rmse) else None,
        "finite_genias": bool(np.isfinite(genias_x).all()),
        "finite_posthoc": bool(np.isfinite(post).all()),
        "loss_dropped": bool(
            len(loss_hist.get("val_recon", [])) >= 2
            and float(loss_hist["val_recon"][-1]) < float(loss_hist["val_recon"][0])
        ),
        "note": "GenIAS and post-hoc are the references. Diffusion samples come from S1/S2/S4.",
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _psi1(extras: dict[str, np.ndarray]) -> np.ndarray | None:
    for key in extras:
        if key.startswith("genias_psi") and float(key.replace("genias_psi", "")) == 1.0:
            return extras[key]
    return extras.get("genias_psi1.0")


def _abs(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _write_s3_curves(
    out: Path,
    loss_hist: dict[str, np.ndarray],
    *,
    parent: np.ndarray,
    recon: np.ndarray | None,
    sample: np.ndarray,
) -> None:
    series = {}
    if "loss_step" in loss_hist:
        series["train EMA (recon+βKL)"] = (loss_hist["loss_step"], loss_hist["loss_value"])
        if "loss_recon" in loss_hist:
            series["train recon MSE"] = (loss_hist["loss_step"], loss_hist["loss_recon"])
    if "val_step" in loss_hist:
        series["val recon MSE"] = (loss_hist["val_step"], loss_hist["val_recon"])
    if series:
        save_loss_curves(
            out / "loss_curve.png",
            series,
            title="S3 GenIAS: per-window z-scored reconstruction + KL",
            ylabel="whitened MSE / loss",
        )
    kids = {}
    n = min(6, len(parent), len(sample))
    if recon is not None:
        kids["GenIAS ψ=1 (recon)"] = recon[:n]
    kids["GenIAS ψ=2"] = sample[:n]
    if n:
        save_same_parent(out / "recon_vs_parent.png", parent[:n], kids, bin_seconds=30)
