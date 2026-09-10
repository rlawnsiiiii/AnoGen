"""Time-series plots for tune galleries and encoder + tune steering.

Writes ``results/shell_plots/shell_tune/``. Does not overwrite S2/S4.
"""

from __future__ import annotations

import json
import shutil
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
    nearest_generated,
    pick_kind_examples,
)
from anogen.shell.plots import save_compare_rows, save_named_grid, save_overlay, save_same_parent, save_strip
from anogen.shell.scaler import resolve_scaler
from anogen.shell.steer import chunked_guided_ddim, shell_from_nominal

# STEERING_MATH § Contrastive energies: f = λ(h_nom−Q_q)² + λ_anom h_anom + λ_rare (−h_rare)
# Grid id 14 is the only tune row that used all three terms together.
_RECIPES: dict[str, dict[str, Any]] = {
    "contrast": {
        "nu": 0.5,
        "lam": 0.0,
        "ddim_steps": 50,
        "normalize_grad": True,
        "n_correct": 1,
        "lam_anom": 3.0,
        "lam_rare": 3.0,
    },
    "occupancy": {
        "nu": 0.2,
        "lam": 1.5,
        "ddim_steps": 50,
        "normalize_grad": True,
        "n_correct": 4,
        "lam_anom": 0.0,
        "lam_rare": 0.0,
    },
    "band": {
        "nu": 0.2,
        "lam": 0.3,
        "ddim_steps": 20,
        "normalize_grad": False,
        "n_correct": 1,
        "lam_anom": 0.0,
        "lam_rare": 0.0,
    },
    "combined": {
        "nu": 0.2,
        "lam": 0.3,
        "ddim_steps": 50,
        "normalize_grad": True,
        "n_correct": 1,
        "lam_anom": 1.0,
        "lam_rare": 1.0,
    },
}


def run_plot_tune(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s1 = _abs(cfg.get("s1_dir", root / "results/shell_s1"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    tune = _abs(cfg.get("tune_dir", root / "results/shell_tune"), root)
    enc_dir = _abs(cfg.get("enc_dir", root / "results/shell_enc"), root)
    plots = _abs(cfg.get("plots_dir", root / "results/shell_plots"), root)
    out = plots / "shell_tune"
    out.mkdir(parents=True, exist_ok=True)
    bin_seconds = int(cfg.get("bin_seconds", 30))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    lab = np.load(s0 / "labeled_arrays.npz") if (s0 / "labeled_arrays.npz").is_file() else None
    gal = np.load(s3 / "galleries.npz") if (s3 / "galleries.npz").is_file() else None
    if lab is None or gal is None:
        report = {"ok": False, "skipped": True, "reason": "Need S0 labeled_arrays and S3 galleries."}
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    x_a = np.asarray(lab["anomaly"])
    x_r = np.asarray(lab["rare"])
    fold_a = np.asarray(lab["anomaly_fold"])
    fold_r = np.asarray(lab["rare_fold"])
    parent = np.asarray(gal["cond"])
    cond_ch = np.asarray(gal["channel_idx"])
    meta_path = s0 / "labeled_windows.csv"
    meta = pd.read_csv(meta_path) if meta_path.is_file() else None
    kinds = _real_kind_tables(x_a, meta, try_types_from_cfg(cfg))

    written = _plot_existing_tune(out, tune, parent, x_a, x_r, rng, bin_seconds)
    written += _plot_real_kinds(out, kinds, bin_seconds)
    arp_path = tune / "best_arp_anomaly.npz"
    if kinds and arp_path.is_file():
        written += _plot_kinds_vs_generated(
            out,
            "tune_search_best_arp",
            {"combined": np.asarray(np.load(arp_path)["x"])},
            kinds,
            rng,
            bin_seconds,
            png_name="real_kinds_vs_tune_search_best_arp.png",
        )
    flags: dict[str, bool] = {}
    galleries: dict[str, dict[str, np.ndarray]] = {}
    for variant in ("time_recon", "time_both"):
        enc_gal = _generate_encoder_tune(
            cfg,
            variant=variant,
            s0=s0,
            s1=s1,
            enc_dir=enc_dir,
            out=out,
            parent=parent,
            cond_ch=cond_ch,
            x_a=x_a,
            x_r=x_r,
            fold_a=fold_a,
            fold_r=fold_r,
        )
        flags[variant] = enc_gal is not None
        if enc_gal is not None:
            galleries[variant] = enc_gal
            written += _plot_encoder_tune(out, variant, enc_gal, x_a, x_r, rng, bin_seconds)
            written += _plot_kinds_vs_generated(out, variant, enc_gal, kinds, rng, bin_seconds)
    if galleries:
        written += _plot_kinds_vs_encoders(out, galleries, kinds, bin_seconds)

    (out / "INDEX.txt").write_text(
        "Generated time series from shell_tune and encoder + tune steering.\n"
        "combined = STEERING_MATH f = λ(h−Q_q)² + λ_anom h_anom + λ_rare (−h_rare) (tune id 14).\n"
        "real_*kinds* use ESA Length × Locality plus the level-shift overlay "
        "(docs/ESA_LABELS.md). Existing S0 CSVs join anomaly_types.csv on event_id.\n\n"
        + "\n".join(f"  {n}" for n in written)
        + "\n"
    )
    report = {"ok": True, "skipped": False, "dir": str(out), "files": written, **flags}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _plot_existing_tune(
    out: Path,
    tune: Path,
    parent: np.ndarray,
    x_a: np.ndarray,
    x_r: np.ndarray,
    rng: np.random.Generator,
    bin_seconds: int,
) -> list[str]:
    written: list[str] = []
    for src_name, dst_name in (
        ("arp_anomaly_bars.png", "search_arp_anomaly_bars.png"),
        ("gap_bars.png", "search_gap_bars.png"),
        ("occupancy_bars.png", "search_occupancy_bars.png"),
        ("overlay_best_arp.png", "overlay_best_arp_search.png"),
    ):
        src = tune / src_name
        if src.is_file():
            shutil.copy2(src, out / dst_name)
            written.append(dst_name)

    galleries: dict[str, np.ndarray] = {}
    colors = {
        "tune contrast (best ARP)": "#2166ac",
        "tune contrast (best gap)": "#1b9e77",
        "tune ZS (best occupancy)": "#e6ab02",
        "real anomaly": "#b2182b",
        "real rare": "#4d4d4d",
    }
    mapping = (
        ("best_arp_anomaly.npz", "tune contrast (best ARP)", "gen_tune_contrast_best_arp.png"),
        ("best_gap.npz", "tune contrast (best gap)", "gen_tune_contrast_best_gap.png"),
        ("best_occupancy.npz", "tune ZS (best occupancy)", "gen_tune_zs_best_occupancy.png"),
    )
    for fname, label, png in mapping:
        path = tune / fname
        if not path.is_file():
            continue
        x = np.asarray(np.load(path)["x"])
        galleries[label] = x
        idx = _extreme(x, 8)
        save_strip(out / png, x, title=label, color=colors[label], bin_seconds=bin_seconds, idx=idx)
        written.append(png)

    compare = {
        "real anomaly": _take(x_a, _extreme(x_a, 5)),
        "real rare": _take(x_r, rng.choice(len(x_r), size=min(5, len(x_r)), replace=False)),
    }
    for label, x in galleries.items():
        compare[label] = _take(x, _extreme(x, 5))
    save_compare_rows(
        out / "real_vs_tune_generated.png",
        compare,
        bin_seconds=bin_seconds,
        n_cols=5,
        colors=colors,
    )
    written.append("real_vs_tune_generated.png")
    return written


def _generate_encoder_tune(
    cfg: dict[str, Any],
    *,
    variant: str,
    s0: Path,
    s1: Path,
    enc_dir: Path,
    out: Path,
    parent: np.ndarray,
    cond_ch: np.ndarray,
    x_a: np.ndarray,
    x_r: np.ndarray,
    fold_a: np.ndarray,
    fold_r: np.ndarray,
    reuse: bool = True,
) -> dict[str, np.ndarray] | None:
    cache = out / f"{variant}_tune_galleries.npz"
    have: dict[str, np.ndarray] = {}
    if reuse and cache.is_file():
        z = np.load(cache)
        have = {k: np.asarray(z[k]) for k in z.files}
    need = [k for k in _RECIPES if k not in have]
    if not need and "parent" in have:
        return have
    ckpt_path = enc_dir / f"{variant}_fold0.pt"
    if not torch_available() or not ckpt_path.is_file() or not (s1 / "denoiser.pt").is_file():
        return have or None
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
    n = min(256, len(parent))
    x0 = have.get("parent", parent[:n])
    ch = have.get("channel_idx", cond_ch[:n])
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    n_ref = min(int(scfg.get("n_ref", 512)), len(x0))
    ref_idx = rng.choice(len(x0), size=n_ref, replace=False)
    val_idx = rng.choice(len(x0), size=min(256, len(x0)), replace=False)
    shell = shell_from_nominal(
        enc,
        x0[ref_idx],
        x0[val_idx],
        q=float(scfg.get("q", 0.99)),
        delta_scale=float(scfg.get("delta_scale", 0.25)),
        device=device,
    )
    ref = shell.pop("ref")
    ra = torch.from_numpy(embed_shell(enc, x_a[fold_a != 0][:256], device))
    rr = torch.from_numpy(embed_shell(enc, x_r[fold_r != 0][:256], device))
    bsz = int(scfg.get("n_sample", 64))
    common = dict(
        ref=ref,
        Q_q=float(shell["Q_q"]),
        tau=float(shell["tau"]),
        c_max=float(scfg.get("c_max", 1.0)),
        device=device,
        scaler=scaler,
        bsz=bsz,
    )
    have["parent"] = np.asarray(x0)
    have["channel_idx"] = np.asarray(ch)
    have["Q_q"] = np.asarray(float(shell["Q_q"]))
    for name in need:
        spec = dict(_RECIPES[name])
        use_contrast = float(spec["lam_anom"]) != 0.0 or float(spec["lam_rare"]) != 0.0
        x, _ = chunked_guided_ddim(
            model,
            enc,
            x0,
            ch,
            schedule,
            ref_anom=ra if use_contrast else None,
            ref_rare=rr if use_contrast else None,
            **spec,
            **common,
        )
        have[name] = x
    np.savez_compressed(cache, **have)
    return have


def _plot_encoder_tune(
    out: Path,
    variant: str,
    gal: dict[str, np.ndarray],
    x_a: np.ndarray,
    x_r: np.ndarray,
    rng: np.random.Generator,
    bin_seconds: int,
) -> list[str]:
    written: list[str] = []
    contrast_l = f"{variant} + tune contrast"
    occupancy_l = f"{variant} + tune occupancy"
    band_l = f"{variant} + S4 band"
    combined_l = f"{variant} + tune combined"
    colors = {
        contrast_l: "#2166ac",
        occupancy_l: "#e6ab02",
        band_l: "#d95f02",
        combined_l: "#1b9e77",
        "real anomaly": "#b2182b",
        "real rare": "#4d4d4d",
        "parent": "#4d4d4d",
    }
    mapping = (
        ("contrast", contrast_l, f"gen_{variant}_tune_contrast.png"),
        ("occupancy", occupancy_l, f"gen_{variant}_tune_occupancy.png"),
        ("band", band_l, f"gen_{variant}_s4_band.png"),
        ("combined", combined_l, f"gen_{variant}_tune_combined.png"),
    )
    for key, label, png in mapping:
        if key not in gal:
            continue
        idx = diverse_idx(gal[key], 8, rng) if key == "combined" else _extreme(gal[key], 8)
        save_strip(out / png, gal[key], title=label, color=colors[label], bin_seconds=bin_seconds, idx=idx)
        written.append(png)

    compare = {
        "real anomaly": _take(x_a, _extreme(x_a, 5)),
        "real rare": _take(x_r, rng.choice(len(x_r), size=min(5, len(x_r)), replace=False)),
    }
    for key, label, _png in mapping:
        if key in gal:
            compare[label] = _take(gal[key], _extreme(gal[key], 5) if key != "combined" else diverse_idx(gal[key], 5, rng))
    save_compare_rows(out / f"real_vs_{variant}_tune.png", compare, bin_seconds=bin_seconds, n_cols=5, colors=colors)
    written.append(f"real_vs_{variant}_tune.png")

    cols = list(range(min(4, len(gal["parent"]))))
    kids = {}
    if "band" in gal:
        kids[band_l] = gal["band"][cols]
    if "occupancy" in gal:
        kids[occupancy_l] = gal["occupancy"][cols]
    if "contrast" in gal:
        kids[contrast_l] = gal["contrast"][cols]
    if "combined" in gal:
        kids[combined_l] = gal["combined"][cols]
    save_same_parent(out / f"{variant}_same_parent_grid.png", gal["parent"][cols], kids, bin_seconds=bin_seconds)
    written.append(f"{variant}_same_parent_grid.png")
    overlay = dict(kids)
    overlay["real anomaly"] = x_a[cols] if len(x_a) > max(cols) else x_a[: len(cols)]
    save_overlay(out / f"{variant}_same_parent_overlay.png", gal["parent"][cols], overlay, bin_seconds=bin_seconds)
    written.append(f"{variant}_same_parent_overlay.png")
    return written


def _real_kind_tables(
    x_a: np.ndarray,
    meta: pd.DataFrame | None,
    types: pd.DataFrame | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    if meta is None or len(x_a) == 0:
        return {}
    anom = meta[meta["kind"] == "anomaly"].reset_index(drop=True)
    if len(anom) != len(x_a):
        return {}
    try:
        kinds = kinds_for_windows(x_a, anom, types)
    except ValueError:
        return {}
    return pick_kind_examples(
        x_a, kinds, anom["event_id"].to_numpy(), n=4, order=KIND_ORDER
    )


def _plot_real_kinds(out: Path, kinds: dict[str, dict[str, np.ndarray]], bin_seconds: int) -> list[str]:
    if not kinds:
        return []
    colors = {k: "#b2182b" for k in kinds}
    rows = {k: v["x"] for k, v in kinds.items()}
    save_compare_rows(
        out / "real_anomaly_kinds.png",
        rows,
        bin_seconds=bin_seconds,
        n_cols=4,
        colors=colors,
        title="Real ESA-ADB kinds (Length × Locality + level-shift overlay)",
    )
    return ["real_anomaly_kinds.png"]


def _plot_kinds_vs_generated(
    out: Path,
    variant: str,
    gal: dict[str, np.ndarray],
    kinds: dict[str, dict[str, np.ndarray]],
    rng: np.random.Generator,
    bin_seconds: int,
    png_name: str | None = None,
) -> list[str]:
    del rng
    if not kinds or "combined" not in gal:
        return []
    gen_rows: dict[str, np.ndarray] = {}
    for kind, blob in kinds.items():
        gen_rows[kind] = blob["x"]
        gen_rows[f"{variant} nearest to {kind}"] = nearest_generated(blob["x"], gal["combined"])
    colors = {k: ("#b2182b" if k.startswith("real ") else "#1b9e77") for k in gen_rows}
    png = png_name or f"real_kinds_vs_{variant}_combined.png"
    save_compare_rows(
        out / png,
        gen_rows,
        bin_seconds=bin_seconds,
        n_cols=4,
        colors=colors,
        title=f"Real ESA-ADB kinds vs nearest {variant} combined samples",
    )
    return [png]


def _plot_kinds_vs_encoders(
    out: Path,
    galleries: dict[str, dict[str, np.ndarray]],
    kinds: dict[str, dict[str, np.ndarray]],
    bin_seconds: int,
) -> list[str]:
    if not kinds:
        return []
    col_names = [k for k in KIND_ORDER if k in kinds]
    row_names = ["real ESA-ADB"]
    for variant in ("time_recon", "time_both"):
        if variant in galleries and "combined" in galleries[variant]:
            row_names.append(f"{variant} combined")
    cells: dict[tuple[str, str], np.ndarray] = {}
    titles: dict[tuple[str, str], str] = {}
    for col in col_names:
        real = kinds[col]["x"][0]
        cells[("real ESA-ADB", col)] = real
        titles[("real ESA-ADB", col)] = str(kinds[col]["event_id"][0])
        for variant in ("time_recon", "time_both"):
            row = f"{variant} combined"
            if row not in row_names:
                continue
            match = nearest_generated(kinds[col]["x"][:1], galleries[variant]["combined"])
            if len(match):
                cells[(row, col)] = match[0]
    colors = {
        "real ESA-ADB": "#b2182b",
        "time_recon combined": "#2166ac",
        "time_both combined": "#1b9e77",
    }
    save_named_grid(
        out / "real_kinds_vs_combined_encoders.png",
        cells,
        row_names=row_names,
        col_names=col_names,
        bin_seconds=bin_seconds,
        colors=colors,
        cell_titles=titles,
        title="ESA-ADB anomaly kinds vs combined (band + contrast) generation",
    )
    return ["real_kinds_vs_combined_encoders.png"]


def _extreme(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(0, dtype=int)
    n = min(n, len(x))
    score = x.max(axis=1) - x.min(axis=1)
    return np.argsort(score)[::-1][:n]


def _take(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    if len(idx) == 0:
        return x[:0]
    return x[np.asarray(idx, dtype=int)]


def _abs(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p
