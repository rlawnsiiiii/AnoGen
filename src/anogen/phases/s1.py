"""S1: train a univariate denoiser on S0 nominal windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anogen.config import REPO_ROOT
from anogen.shell.coverage import min_distances
from anogen.shell.data import load_panel
from anogen.shell.diffusion import torch_available, train_denoiser, unguided_from_nominal
from anogen.shell.features import embed_windows
from anogen.shell.plots import save_bar, save_loss_curves, save_same_parent
from anogen.shell.scaler import fit_channel_minmax
from anogen.shell.windows import load_train_index, materialize


def run_s1(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = Path(cfg.get("s0_dir", root / "results/shell_s0"))
    if not s0.is_absolute():
        s0 = root / s0
    out = Path(cfg.get("s1_dir", root / "results/shell_s1"))
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    if not torch_available():
        report = {
            "ok": False,
            "skipped": True,
            "reason": "torch is not installed; uv sync --extra neural",
        }
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    proto_path = s0 / "protocol.json"
    has_index = (s0 / "train_index.csv").is_file() or (s0 / "nominal_index.csv").is_file()
    if not proto_path.is_file() or not has_index:
        report = {
            "ok": False,
            "skipped": True,
            "reason": f"S0 artifacts missing under {s0}. Run s0 first.",
        }
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    proto = json.loads(proto_path.read_text())
    width = int(proto["W"])
    channels = list(
        cfg.get("channels")
        or [
            "channel_41",
            "channel_42",
            "channel_43",
            "channel_44",
            "channel_45",
            "channel_46",
        ]
    )
    splits = cfg.get("splits") or {}
    from anogen.config import panel_path

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
    n_rare = int(train["is_rare"].sum()) if "is_rare" in train.columns else 0

    dcfg = dict((cfg.get("shell") or {}).get("diffusion") or {})
    scfg = dict((cfg.get("shell") or {}).get("scaler") or {})
    kind = str(scfg.get("kind", "minmax")).lower()
    scaler = None
    if kind == "minmax":
        fr = tuple(scfg.get("feature_range") or (0.0, 1.0))
        scaler = fit_channel_minmax(x, ch, panel.k, feature_range=(float(fr[0]), float(fr[1])))
        scaler.save(s0 / "minmax_scaler.npz")
    trained = train_denoiser(
        x,
        ch,
        n_channels=panel.k,
        hidden=int(dcfg.get("hidden", 48)),
        n_times=int(dcfg.get("n_times", 200)),
        steps=int(dcfg.get("steps", 2000)),
        batch_size=int(dcfg.get("batch_size", 64)),
        lr=float(dcfg.get("lr", 2e-4)),
        seed=int(cfg.get("seed", 0)),
        val_frac=float(dcfg.get("val_frac", 0.1)),
        backbone=str(dcfg.get("backbone", "unet")),
        n_layers=int(dcfg.get("n_layers", 6)),
        d_state=int(dcfg.get("d_state", 64)),
        scaler=scaler,
    )
    model = trained.pop("model")
    schedule = trained.pop("schedule")
    loss_hist = {
        k: trained.pop(k)
        for k in (
            "loss_step",
            "loss_value",
            "val_step",
            "val_value",
            "loss_by_t_lo",
            "loss_by_t_hi",
            "loss_by_t",
        )
        if k in trained
    }

    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    n_show = min(int(dcfg.get("n_sample", 128)), len(x))
    pick = rng.choice(len(x), size=n_show, replace=False)
    samples = unguided_from_nominal(
        model,
        schedule,
        x[pick],
        ch[pick],
        nu=float(dcfg.get("unguided_nu", 1.0)),
        ddim_steps=int(dcfg.get("ddim_steps", 20)),
        device=trained["device"],
        scaler=scaler,
    )
    real_e = embed_windows(x[pick])
    gen_e = embed_windows(samples)
    noise = rng.normal(loc=float(x[pick].mean()), scale=max(float(x[pick].std()), 1e-6), size=x[pick].shape)
    d_gen = float(min_distances(real_e, gen_e).mean())
    d_noise = float(min_distances(real_e, embed_windows(noise)).mean())

    import torch

    ckpt = {
        "state_dict": model.state_dict(),
        "backbone": str(dcfg.get("backbone", "unet")),
        "hidden": int(dcfg.get("hidden", 48)),
        "n_layers": int(dcfg.get("n_layers", 6)),
        "d_state": int(dcfg.get("d_state", 64)),
        "n_channels": panel.k,
        "n_times": int(dcfg.get("n_times", 200)),
        "W": width,
        "channels": channels,
    }
    if scaler is not None:
        ckpt.update(scaler.to_ckpt())
    torch.save(ckpt, out / "denoiser.pt")
    np.savez_compressed(out / "unguided_samples.npz", x=samples, channel_idx=ch[pick])
    if loss_hist:
        np.savez_compressed(out / "loss_history.npz", **loss_hist)
        _write_s1_curves(out, loss_hist, parent=x[pick], samples=samples)

    report = {
        "ok": True,
        "skipped": False,
        "s0_dir": str(s0),
        "W": width,
        **{k: (v.item() if isinstance(v, np.generic) else v) for k, v in trained.items()},
        "n_windows": int(len(x)),
        "n_nominal": int(len(x) - n_rare),
        "n_rare_in_train": n_rare,
        "sample_vs_nominal_l2": d_gen,
        "noise_vs_nominal_l2": d_noise,
        "closer_than_noise": d_gen < d_noise,
        "loss_dropped": bool(
            len(loss_hist.get("val_value", [])) >= 2
            and float(loss_hist["val_value"][-1]) < float(loss_hist["val_value"][0])
        ),
        "sample_mean": float(samples.mean()),
        "data_mean": float(x[pick].mean()),
        "sample_std": float(samples.std()),
        "data_std": float(x[pick].std()),
        "scaler": scaler.summary() if scaler is not None else {"kind": "none"},
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _write_s1_curves(out: Path, loss_hist: dict[str, np.ndarray], *, parent: np.ndarray, samples: np.ndarray) -> None:
    series = {}
    if "loss_step" in loss_hist:
        series["train EMA ε-MSE"] = (loss_hist["loss_step"], loss_hist["loss_value"])
    if "val_step" in loss_hist:
        series["val ε-MSE"] = (loss_hist["val_step"], loss_hist["val_value"])
    if series:
        save_loss_curves(out / "loss_curve.png", series, title="S1 denoiser: noise-prediction loss", ylabel="ε-MSE")
    if "loss_by_t" in loss_hist:
        lo = loss_hist["loss_by_t_lo"]
        hi = loss_hist["loss_by_t_hi"]
        labels = [f"t {int(a)}–{int(b) - 1}" for a, b in zip(lo, hi)]
        save_bar(
            out / "loss_by_t.png",
            labels,
            loss_hist["loss_by_t"],
            title="S1 val ε-MSE by diffusion time",
            ylabel="MSE",
        )
    n = min(6, len(parent), len(samples))
    if n:
        save_same_parent(
            out / "unguided_vs_parent.png",
            parent[:n],
            {"unguided DDIM (ν=1)": samples[:n]},
            bin_seconds=30,
        )
