"""S2: reconstruction shell encoder, Q_q, guided DDIM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from anogen.config import REPO_ROOT, panel_path
from anogen.shell.data import load_panel
from anogen.shell.diffusion import DiffusionSchedule, denoiser_from_ckpt, torch_available
from anogen.shell.scaler import resolve_scaler
from anogen.shell.steer import guided_ddim, shell_from_nominal, train_encoder
from anogen.shell.windows import load_train_index, materialize


def run_s2(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = Path(cfg.get("s0_dir", root / "results/shell_s0"))
    s1 = Path(cfg.get("s1_dir", root / "results/shell_s1"))
    out = Path(cfg.get("s2_dir", root / "results/shell_s2"))
    if not s0.is_absolute():
        s0 = root / s0
    if not s1.is_absolute():
        s1 = root / s1
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    if not torch_available():
        report = {"ok": False, "skipped": True, "reason": "torch is not installed"}
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    if not (s0 / "train_index.csv").is_file() and not (s0 / "nominal_index.csv").is_file():
        report = {"ok": False, "skipped": True, "reason": f"S0 missing under {s0}"}
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    ckpt_path = s1 / "denoiser.pt"
    if not ckpt_path.is_file():
        report = {"ok": False, "skipped": True, "reason": f"S1 checkpoint missing: {ckpt_path}"}
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
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

    scfg = dict((cfg.get("shell") or {}).get("steer") or {})
    dcfg = dict((cfg.get("shell") or {}).get("diffusion") or {})
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    n = len(x)
    perm = rng.permutation(n)
    n_val = max(64, int(0.1 * n))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    enc_out = train_encoder(
        x[train_idx],
        hidden=int(scfg.get("enc_hidden", 32)),
        emb=int(scfg.get("emb", 32)),
        steps=int(scfg.get("enc_steps", 400)),
        batch_size=int(scfg.get("batch_size", 64)),
        lr=float(scfg.get("enc_lr", 1e-3)),
        seed=int(cfg.get("seed", 0)),
    )
    encoder = enc_out["encoder"]
    n_ref = min(int(scfg.get("n_ref", 512)), len(train_idx))
    ref_idx = rng.choice(train_idx, size=n_ref, replace=False)
    shell = shell_from_nominal(
        encoder,
        x[ref_idx],
        x[val_idx],
        q=float(scfg.get("q", 0.99)),
        delta_scale=float(scfg.get("delta_scale", 0.25)),
        device=enc_out["device"],
    )
    ref = shell.pop("ref")

    ckpt = torch.load(ckpt_path, map_location=enc_out["device"], weights_only=False)
    device_t = torch.device(enc_out["device"])
    model = denoiser_from_ckpt(ckpt, device=device_t)
    scaler = resolve_scaler(ckpt, s0)
    schedule = DiffusionSchedule.linear(int(ckpt["n_times"])).to(device_t)

    n_show = min(int(scfg.get("n_sample", 128)), len(val_idx))
    pick = val_idx[:n_show]
    samples, h = guided_ddim(
        model,
        encoder,
        x[pick],
        ch[pick],
        schedule,
        ref=ref,
        Q_q=float(shell["Q_q"]),
        tau=float(shell["tau"]),
        nu=float(scfg.get("nu", 0.2)),
        lam=float(scfg.get("lambda", 0.3)),
        c_max=float(scfg.get("c_max", 1.0)),
        ddim_steps=int(dcfg.get("ddim_steps", 20)),
        device=enc_out["device"],
        scaler=scaler,
    )
    finite = bool(np.isfinite(samples).all())
    occupancy = float(np.mean(np.abs(h - shell["Q_q"]) <= shell["delta"])) if finite else 0.0
    explode = bool((not finite) or np.nanmax(np.abs(samples)) > 50.0)

    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "width": width,
            "hidden": int(scfg.get("enc_hidden", 32)),
            "emb": int(scfg.get("emb", 32)),
            "Q_q": shell["Q_q"],
            "q": shell["q"],
            "tau": shell["tau"],
            "delta": shell["delta"],
        },
        out / "encoder.pt",
    )
    np.savez_compressed(out / "guided_samples.npz", x=samples, h=h, channel_idx=ch[pick])

    report = {
        "ok": finite and not explode,
        "skipped": False,
        "encoder": proto["shell_encoder"],
        "recon_mse": enc_out["recon_mse"],
        "Q_q": shell["Q_q"],
        "q": shell["q"],
        "tau": shell["tau"],
        "delta": shell["delta"],
        "occupancy": occupancy,
        "finite": finite,
        "explode": explode,
        "n_sample": int(len(samples)),
        "nu": float(scfg.get("nu", 0.2)),
        "lambda": float(scfg.get("lambda", 0.3)),
        "note": "S2 freezes the encoder and the shell. Do not retune after S4.",
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
