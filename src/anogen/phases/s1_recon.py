"""Unguided parent reconstruction sweep (λ = 0). Does not retrain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from anogen.config import REPO_ROOT, panel_path
from anogen.shell.data import load_panel
from anogen.shell.diffusion import (
    DiffusionSchedule,
    denoiser_from_ckpt,
    q_sample,
    torch_available,
    unguided_from_nominal,
)
from anogen.shell.plots import save_bar, save_same_parent
from anogen.shell.scaler import resolve_scaler
from anogen.shell.windows import load_train_index, materialize


NUS = (0.05, 0.2, 1.0)


def run_s1_recon(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s1 = _abs(cfg.get("s1_dir", root / "results/shell_s1"), root)
    plots = _abs(cfg.get("plots_dir", root / "results/shell_plots"), root)
    out = s1
    out.mkdir(parents=True, exist_ok=True)

    if not torch_available():
        report = {"ok": False, "skipped": True, "reason": "torch is not installed"}
        (out / "recon_sweep.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    ckpt_path = s1 / "denoiser.pt"
    if not ckpt_path.is_file() or not (s0 / "protocol.json").is_file():
        report = {"ok": False, "skipped": True, "reason": "Need S0 protocol and S1 denoiser.pt"}
        (out / "recon_sweep.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    proto = json.loads((s0 / "protocol.json").read_text())
    width = int(proto["W"])
    channels = list(cfg.get("channels") or [])
    splits = cfg.get("splits") or {}
    panel = load_panel(
        panel_path(cfg),
        channels=channels,
        official_train_end=splits.get("official_train_end", "2007-01-01"),
        allow_test_telemetry=False,
        bin_seconds=int(cfg.get("bin_seconds", 30)),
    )
    train = load_train_index(s0)
    x = materialize(panel, train, width)
    ch = train["channel_idx"].to_numpy(dtype=np.int64)

    dcfg = dict((cfg.get("shell") or {}).get("diffusion") or {})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = denoiser_from_ckpt(ckpt, device=device)
    scaler = resolve_scaler(ckpt, s0)
    model.eval()
    schedule = DiffusionSchedule.linear(int(ckpt["n_times"])).to(torch.device(device))
    n_times = int(ckpt["n_times"])
    ddim_steps = int(dcfg.get("ddim_steps", 20))

    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    n_show = min(int(dcfg.get("n_sample", 256)), len(x))
    pick = rng.choice(len(x), size=n_show, replace=False)
    parent = x[pick]
    parent_ch = ch[pick]

    rows = []
    recs: dict[str, np.ndarray] = {}
    for nu in NUS:
        rec = unguided_from_nominal(
            model,
            schedule,
            parent,
            parent_ch,
            nu=float(nu),
            ddim_steps=ddim_steps,
            device=device,
            scaler=scaler,
        )
        noised = _noised_only(
            parent,
            schedule,
            nu=float(nu),
            n_times=n_times,
            device=device,
            scaler=scaler,
            channel_idx=parent_ch,
        )
        recs[f"nu={nu}"] = rec
        ddim = _scores(parent, rec)
        ddim["rmse_vs_shuffled_parent"] = _shuffled_rmse(parent, rec, rng)
        rows.append(
            {
                "nu": float(nu),
                "t_start": int(round(float(nu) * (n_times - 1))),
                "ddim": ddim,
                "noised_only": _scores(parent, noised),
            }
        )

    n_plot = min(6, len(parent))
    save_same_parent(
        out / "recon_vs_parent.png",
        parent[:n_plot],
        {f"unguided ν={nu}": recs[f"nu={nu}"][:n_plot] for nu in NUS},
        bin_seconds=int(cfg.get("bin_seconds", 30)),
    )
    save_bar(
        out / "recon_rmse_by_nu.png",
        [f"ν={r['nu']}\nDDIM" for r in rows] + [f"ν={r['nu']}\nnoised" for r in rows],
        np.array([r["ddim"]["rmse"] for r in rows] + [r["noised_only"]["rmse"] for r in rows]),
        title="Unguided (λ=0) RMSE to parent vs noised-only baseline",
        ylabel="RMSE",
    )
    plots.mkdir(parents=True, exist_ok=True)
    (plots / "train_s1_recon_vs_parent.png").write_bytes((out / "recon_vs_parent.png").read_bytes())
    (plots / "train_s1_recon_rmse_by_nu.png").write_bytes((out / "recon_rmse_by_nu.png").read_bytes())

    learned = all(r["ddim"]["rmse"] < r["noised_only"]["rmse"] for r in rows if r["nu"] < 1.0)
    parent_pair = _shuffled_rmse(parent, parent, rng)
    report = {
        "ok": True,
        "skipped": False,
        "n_windows": int(n_show),
        "ddim_steps": ddim_steps,
        "n_times": n_times,
        "lambda": 0.0,
        "scaler": scaler.summary() if scaler is not None else {"kind": "none"},
        "parent_vs_shuffled_parent_rmse": parent_pair,
        "nus": rows,
        "beats_noised_baseline_at_small_nu": learned,
        "note": (
            "λ=0 DDIM from a noised parent. Small ν should return the parent. "
            "ν=1 is generation: RMSE to that parent can stay large even if the "
            "denoiser is trained. Compare DDIM to the noised-only column."
        ),
    }
    (out / "recon_sweep.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _scores(parent: np.ndarray, y: np.ndarray) -> dict[str, float]:
    err = np.asarray(y, dtype=np.float64) - np.asarray(parent, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(err**2)))
    med = float(np.median(np.sqrt(np.mean(err**2, axis=1))))
    pc = parent - parent.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    num = (pc * yc).sum(axis=1)
    den = np.sqrt((pc * pc).sum(axis=1) * (yc * yc).sum(axis=1)).clip(min=1e-12)
    return {
        "rmse": rmse,
        "median_window_rmse": med,
        "shape_corr": float(np.nanmean(num / den)),
    }


def _shuffled_rmse(parent: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    """RMSE after pairing each recon with a different parent (identity null)."""
    perm = rng.permutation(len(parent))
    if len(perm) > 1 and np.array_equal(perm, np.arange(len(parent))):
        perm = np.roll(perm, 1)
    err = np.asarray(y, dtype=np.float64) - np.asarray(parent[perm], dtype=np.float64)
    return float(np.sqrt(np.mean(err**2)))


def _noised_only(
    x0: np.ndarray,
    schedule: Any,
    *,
    nu: float,
    n_times: int,
    device: str,
    scaler: Any | None = None,
    channel_idx: np.ndarray | None = None,
) -> np.ndarray:
    t_start = max(0, min(n_times - 1, int(round(nu * (n_times - 1)))))
    x = np.asarray(x0, dtype=np.float32)
    ch = np.asarray(channel_idx, dtype=np.int64) if channel_idx is not None else None
    if scaler is not None:
        if ch is None:
            raise ValueError("channel_idx is required when a scaler is set")
        x = scaler.transform(x, ch)
    xt = torch.from_numpy(x).unsqueeze(1).to(device)
    t = torch.full((xt.size(0),), t_start, device=device, dtype=torch.long)
    noised, _ = q_sample(xt, t, schedule)
    y = noised.squeeze(1).cpu().numpy().astype(np.float32)
    if scaler is not None:
        y = scaler.inverse(y, ch)
    return y


def _abs(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p
