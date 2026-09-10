"""time_recon + combined f starting from x_T ~ N(0, I). No donor waveform."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from anogen.config import REPO_ROOT
from anogen.shell.diffusion import DiffusionSchedule, denoiser_from_ckpt, torch_available
from anogen.shell.encoders import ShellEncoder, embed_shell
from anogen.shell.events import try_types_from_cfg
from anogen.shell.morphology import (
    KIND_ORDER,
    diverse_idx,
    kinds_for_windows,
    pick_kind_examples,
)
from anogen.shell.plots import save_compare_rows, save_strip
from anogen.shell.scaler import resolve_scaler
from anogen.shell.steer import chunked_guided_ddim, shell_from_nominal


def run_plot_from_noise(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s1 = _abs(cfg.get("s1_dir", root / "results/shell_s1"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    enc_dir = _abs(cfg.get("enc_dir", root / "results/shell_enc"), root)
    plots = _abs(cfg.get("plots_dir", root / "results/shell_plots"), root)
    out = plots / "shell_tune"
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = enc_dir / "time_recon_fold0.pt"
    if not torch_available() or not ckpt_path.is_file() or not (s1 / "denoiser.pt").is_file():
        report = {"ok": False, "skipped": True, "reason": "Need torch, S1 denoiser, time_recon_fold0."}
        (out / "from_noise_summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    lab = np.load(s0 / "labeled_arrays.npz")
    gal = np.load(s3 / "galleries.npz")
    x_a = np.asarray(lab["anomaly"])
    x_r = np.asarray(lab["rare"])
    fold_a = np.asarray(lab["anomaly_fold"])
    fold_r = np.asarray(lab["rare_fold"])
    parent = np.asarray(gal["cond"])
    bin_seconds = int(cfg.get("bin_seconds", 30))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc = ShellEncoder(
        int(blob["width"]),
        hidden=int(blob["hidden"]),
        emb=int(blob["emb"]),
        time_emb=int(blob.get("time_emb", 8)),
        pool=str(blob.get("pool", "time")),
    )
    enc.load_state_dict(blob["state_dict"])
    enc.to(device_t)
    enc.eval()
    den = torch.load(s1 / "denoiser.pt", map_location=device, weights_only=False)
    model = denoiser_from_ckpt(den, device=device_t)
    scaler = resolve_scaler(den, s0)
    schedule = DiffusionSchedule.linear(int(den["n_times"])).to(device_t)
    scfg = dict((cfg.get("shell") or {}).get("steer") or {})
    width = int(blob["width"])
    n_ch = int(den.get("n_channels", 6))
    n = 128
    dummy = np.zeros((n, width), dtype=np.float32)
    ch = np.arange(n, dtype=np.int64) % n_ch

    n_ref = min(int(scfg.get("n_ref", 512)), len(parent))
    ref_idx = rng.choice(len(parent), size=n_ref, replace=False)
    val_idx = rng.choice(len(parent), size=min(256, len(parent)), replace=False)
    shell = shell_from_nominal(
        enc,
        parent[ref_idx],
        parent[val_idx],
        q=float(scfg.get("q", 0.99)),
        delta_scale=float(scfg.get("delta_scale", 0.25)),
        device=device,
    )
    ref = shell.pop("ref")
    ra = torch.from_numpy(embed_shell(enc, x_a[fold_a != 0][:256], device))
    rr = torch.from_numpy(embed_shell(enc, x_r[fold_r != 0][:256], device))
    samples, h = chunked_guided_ddim(
        model,
        enc,
        dummy,
        ch,
        schedule,
        bsz=int(scfg.get("n_sample", 64)),
        ref=ref,
        Q_q=float(shell["Q_q"]),
        tau=float(shell["tau"]),
        nu=1.0,
        lam=0.3,
        c_max=float(scfg.get("c_max", 1.0)),
        ddim_steps=50,
        device=device,
        scaler=scaler,
        normalize_grad=True,
        n_correct=1,
        ref_anom=ra,
        lam_anom=1.0,
        ref_rare=rr,
        lam_rare=1.0,
        start_from_noise=True,
    )
    np.savez_compressed(
        out / "time_recon_combined_from_noise.npz",
        x=samples,
        h=h,
        channel_idx=ch,
        Q_q=float(shell["Q_q"]),
        start="N(0,I)",
    )
    occ = float(np.mean(np.abs(h - float(shell["Q_q"])) <= float(shell["delta"])))
    written = _plot(out, samples, x_a, rng, bin_seconds, s0, try_types_from_cfg(cfg))
    report = {
        "ok": True,
        "skipped": False,
        "n": int(len(samples)),
        "occupancy": occ,
        "h_mean": float(h.mean()) if len(h) else float("nan"),
        "Q_q": float(shell["Q_q"]),
        "start": "N(0,I) in scaled units; no donor waveform",
        "recipe": "time_recon + combined f (id 14), 50 DDIM steps from t=T-1",
        "files": written,
        "dir": str(out),
    }
    (out / "from_noise_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _plot(
    out: Path,
    samples: np.ndarray,
    x_a: np.ndarray,
    rng: np.random.Generator,
    bin_seconds: int,
    s0: Path,
    types: pd.DataFrame | None = None,
) -> list[str]:
    written: list[str] = []
    idx = diverse_idx(samples, 8, rng)
    save_strip(
        out / "gen_time_recon_combined_from_noise.png",
        samples,
        title="time_recon + combined f  from x_T ~ N(0, I)",
        color="#1b9e77",
        bin_seconds=bin_seconds,
        idx=idx,
    )
    written.append("gen_time_recon_combined_from_noise.png")
    amp = samples.max(axis=1) - samples.min(axis=1)
    ext = np.argsort(amp)[::-1][:8]
    save_strip(
        out / "gen_time_recon_combined_from_noise_extreme.png",
        samples,
        title="time_recon + combined f  from noise (largest range)",
        color="#1b9e77",
        bin_seconds=bin_seconds,
        idx=ext,
    )
    written.append("gen_time_recon_combined_from_noise_extreme.png")
    rows: dict[str, np.ndarray] = {"from noise (diverse)": samples[idx[:5]] if len(idx) else samples[:5]}
    meta_path = s0 / "labeled_windows.csv"
    if meta_path.is_file() and len(x_a):
        anom = pd.read_csv(meta_path)
        anom = anom[anom["kind"] == "anomaly"].reset_index(drop=True)
        if len(anom) == len(x_a):
            try:
                kinds = kinds_for_windows(x_a, anom, types)
            except ValueError:
                kinds = None
            if kinds is not None:
                picked = pick_kind_examples(
                    x_a, kinds, anom["event_id"].to_numpy(), n=4, order=KIND_ORDER
                )
                for name, blob in picked.items():
                    rows[name] = blob["x"]
    else:
        rows["real anomaly (extreme)"] = x_a[np.argsort(x_a.max(1) - x_a.min(1))[::-1][:5]]
    save_compare_rows(
        out / "real_vs_time_recon_combined_from_noise.png",
        rows,
        bin_seconds=bin_seconds,
        n_cols=4,
        colors={"from noise (diverse)": "#1b9e77", **{k: "#b2182b" for k in rows if k != "from noise (diverse)"}},
        title="Real ESA-ADB kinds vs time_recon + combined f started from noise",
    )
    written.append("real_vs_time_recon_combined_from_noise.png")
    return written


def _abs(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p
