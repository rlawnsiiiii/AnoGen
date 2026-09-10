"""Try the three H_ANOM.md strategies on locked Mission 1 channels 41–46.

Frozen S1 score + fold-0 time_recon / time_both. Combined f (tune id 14).
Does not overwrite results/shell_s4 or shell_tune. Isolated results/shell_hanom/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from anogen.config import REPO_ROOT
from anogen.shell.coverage import mean_pairwise_distance
from anogen.shell.diffusion import DiffusionSchedule, denoiser_from_ckpt, torch_available
from anogen.shell.encoders import ShellEncoder, embed_shell
from anogen.shell.features import embed_windows
from anogen.shell.events import try_types_from_cfg
from anogen.shell.morphology import KIND_ORDER, diverse_idx, nearest_generated
from anogen.shell.plots import save_compare_rows, save_named_grid, save_strip
from anogen.shell.scaler import resolve_scaler
from anogen.shell.steer import band_report, chunked_guided_ddim, shell_from_nominal
from anogen.phases.plot_tune import _plot_kinds_vs_generated, _real_kind_tables
from anogen.phases.tune import _score_gallery

_COMBINED = {
    "nu": 0.2,
    "lam": 0.3,
    "ddim_steps": 50,
    "normalize_grad": True,
    "n_correct": 1,
    "lam_anom": 1.0,
    "lam_rare": 1.0,
}
_NORARE = {**_COMBINED, "lam_rare": 0.0}

STRATEGIES = (
    "nearest",
    "knn",
    "proto",
)

_COLORS = {
    "nearest": "#2166ac",
    "knn": "#1b9e77",
    "proto": "#d95f02",
    "soft": "#7570b3",
}


def run_hanom(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s1 = _abs(cfg.get("s1_dir", root / "results/shell_s1"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    s4 = _abs(cfg.get("s4_dir", root / "results/shell_s4"), root)
    enc_dir = _abs(cfg.get("enc_dir", root / "results/shell_enc"), root)
    tune_plots = _abs(cfg.get("plots_dir", root / "results/shell_plots"), root) / "shell_tune"
    out = _abs(cfg.get("hanom_dir", root / "results/shell_hanom"), root)
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    if not torch_available():
        report = {"ok": False, "skipped": True, "reason": "torch is not installed"}
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    needed = {
        "S0": s0 / "labeled_arrays.npz",
        "S1": s1 / "denoiser.pt",
        "S3": s3 / "galleries.npz",
        "S4": s4 / "summary.json",
        "time_recon": enc_dir / "time_recon_fold0.pt",
        "time_both": enc_dir / "time_both_fold0.pt",
    }
    for label, path in needed.items():
        if not path.is_file():
            report = {"ok": False, "skipped": True, "reason": f"{label} missing: {path}"}
            (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            return report

    tau_cov = float(json.loads((s4 / "summary.json").read_text())["tau"])
    lab = np.load(s0 / "labeled_arrays.npz")
    gal = np.load(s3 / "galleries.npz")
    fold_tab = pd.read_csv(s0 / "channel_fold_counts.csv")
    x_a = np.asarray(lab["anomaly"])
    x_r = np.asarray(lab["rare"])
    fold_a = np.asarray(lab["anomaly_fold"])
    fold_r = np.asarray(lab["rare_fold"])
    ch_a = np.asarray(lab["anomaly_channel"])
    ch_r = np.asarray(lab["rare_channel"])
    parent = np.asarray(gal["cond"])
    cond_ch = np.asarray(gal["channel_idx"])
    meta = pd.read_csv(s0 / "labeled_windows.csv") if (s0 / "labeled_windows.csv").is_file() else None
    kinds = _real_kind_tables(x_a, meta, try_types_from_cfg(cfg))
    n = min(256, len(parent))
    x0 = parent[:n]
    ch = cond_ch[:n]
    bin_seconds = int(cfg.get("bin_seconds", 30))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    hcfg = dict((cfg.get("shell") or {}).get("hanom") or {})
    knn_k = int(hcfg.get("knn", 8))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    den = torch.load(s1 / "denoiser.pt", map_location=device, weights_only=False)
    model = denoiser_from_ckpt(den, device=device_t)
    scaler = resolve_scaler(den, s0)
    schedule = DiffusionSchedule.linear(int(den["n_times"])).to(device_t)
    scfg = dict((cfg.get("shell") or {}).get("steer") or {})
    bsz = int(scfg.get("n_sample", 64))

    score_kw = dict(
        x_a=x_a,
        x_r=x_r,
        fold_a=fold_a,
        fold_r=fold_r,
        ch_a=ch_a,
        ch_r=ch_r,
        fold_tab=fold_tab,
        tau=tau_cov,
    )

    families: dict[str, Any] = {}
    for fam, recipe, dest, drop_rare, label in (
        ("with_rare", _COMBINED, out, False, "combined f (with h_rare)"),
        ("norare", _NORARE, out / "norare", True, "band + h_anom only (no h_rare)"),
    ):
        dest.mkdir(parents=True, exist_ok=True)
        fam_plots = dest / "plots"
        fam_plots.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        methods: dict[str, Any] = {}
        galleries: dict[str, dict[str, np.ndarray]] = {}
        for variant in ("time_recon", "time_both"):
            enc, shell, ra, rr = _load(
                enc_dir / f"{variant}_fold0.pt",
                x0,
                x_a,
                x_r,
                fold_a,
                fold_r,
                scfg,
                device,
                int(cfg.get("seed", 0)),
            )
            common = dict(
                ref=shell["ref"],
                Q_q=float(shell["Q_q"]),
                tau=float(shell["tau"]),
                c_max=float(scfg.get("c_max", 1.0)),
                device=device,
                scaler=scaler,
                bsz=bsz,
                ref_anom=ra,
                ref_rare=None if drop_rare else rr,
                **recipe,
            )
            vgals: dict[str, np.ndarray] = {"parent": x0, "channel_idx": ch}
            soft_cache = tune_plots / f"{variant}_tune_galleries.npz"
            if soft_cache.is_file() and "combined" in np.load(soft_cache).files:
                vgals["soft"] = np.asarray(np.load(soft_cache)["combined"])[:n]
            for strat in STRATEGIES:
                cache = dest / f"{variant}_{strat}.npz"
                if cache.is_file() and not bool(hcfg.get("force", False)):
                    blob = np.load(cache)
                    x = np.asarray(blob["x"])
                    h = np.asarray(blob["h"])
                else:
                    print(f"hanom {fam} {variant} {strat} n={n}", flush=True)
                    x, h = chunked_guided_ddim(
                        model,
                        enc,
                        x0,
                        ch,
                        schedule,
                        anom_energy_kind=strat,
                        anom_knn=knn_k,
                        anom_proto_seed=int(cfg.get("seed", 0)),
                        **common,
                    )
                    np.savez_compressed(
                        cache,
                        x=x,
                        h=h,
                        channel_idx=ch,
                        Q_q=float(shell["Q_q"]),
                        delta=float(shell["delta"]),
                        strategy=np.asarray(strat),
                        drop_rare=np.asarray(drop_rare),
                    )
                vgals[strat] = x
                scored = _score_gallery(x, only_fold=0, **score_kw)
                scored["occupancy"] = float(np.mean(np.abs(h - shell["Q_q"]) <= shell["delta"]))
                scored["energy"] = band_report(h, float(shell["Q_q"]), float(shell["delta"]))
                scored["diversity"] = float(mean_pairwise_distance(embed_windows(x)))
                scored["kind_hits"] = _kind_hits(x)
                scored["encoder"] = variant
                scored["strategy"] = strat
                scored["drop_rare"] = drop_rare
                scored["n"] = int(len(x))
                scored["shot"] = "FS"
                scored["note"] = (
                    "256 S3 parents, fold-0 queries, frozen S4 τ. Not a new encscore freeze. "
                    + ("λ_rare=0." if drop_rare else "λ_rare=1.")
                )
                methods[f"{variant}_{strat}"] = scored
                written += _plot_one(
                    fam_plots, variant, strat, x, kinds, rng, bin_seconds, label=label
                )
            galleries[variant] = vgals
            written += _plot_variant_grid(
                fam_plots, variant, vgals, kinds, bin_seconds, label=label
            )
        written += _plot_strategy_grid(fam_plots, galleries, kinds, bin_seconds, label=label)
        (fam_plots / "INDEX.txt").write_text(
            f"h_anom strategies on Mission 1 41–46. {label}.\n"
            "nearest / knn / proto: docs/H_ANOM.md.\n\n"
            + "\n".join(f"  {n}" for n in written)
            + "\n"
        )
        families[fam] = {
            "methods": {k: _brief(v) for k, v in methods.items()},
            "plots": written,
            "dir": str(dest),
            "drop_rare": drop_rare,
            "f": (
                "λ(h_nom−Q_q)² + λ_anom h_anom"
                + ("" if drop_rare else " + λ_rare (−h_rare)")
            ),
        }

    report = {
        "ok": True,
        "skipped": False,
        "tau": tau_cov,
        "tau_source": "s4_frozen",
        "n": n,
        "knn": knn_k,
        "families": families,
        "methods": families["with_rare"]["methods"],
        "plots": families["with_rare"]["plots"],
        "dir": str(out),
        "note": (
            "Three h_anom strategies on frozen S1 / time_* fold0. "
            "with_rare = id 14 combined. norare = same with λ_rare=0. "
            "256 donors. Fold-0 Coverage@τ only. Does not overwrite shell_s4."
        ),
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    (out / "norare" / "summary.json").write_text(
        json.dumps(_jsonable(families["norare"]), indent=2) + "\n"
    )
    return report


def _load(
    ckpt_path: Path,
    parent: np.ndarray,
    x_a: np.ndarray,
    x_r: np.ndarray,
    fold_a: np.ndarray,
    fold_r: np.ndarray,
    scfg: dict[str, Any],
    device: str,
    seed: int,
) -> tuple[Any, dict[str, Any], Any, Any]:
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
    rng = np.random.default_rng(seed)
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
    ra = torch.from_numpy(embed_shell(enc, x_a[fold_a != 0][:256], device))
    rr = torch.from_numpy(embed_shell(enc, x_r[fold_r != 0][:256], device))
    return enc, shell, ra, rr


def _kind_hits(x: np.ndarray) -> dict[str, float]:
    """Cheap in-window kind rates. No event span (generated windows have none)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0:
        return {"frac_spike_amp": 0.0, "frac_shift": 0.0, "med_amp": 0.0, "med_dmu": 0.0}
    half = x.shape[1] // 2
    amp = x.max(axis=1) - x.min(axis=1)
    dmu = np.abs(x[:, :half].mean(axis=1) - x[:, half:].mean(axis=1))
    return {
        "frac_spike_amp": float(np.mean(amp >= 0.1)),
        "frac_shift": float(np.mean(dmu >= 0.01)),
        "med_amp": float(np.median(amp)),
        "med_dmu": float(np.median(dmu)),
    }


def _plot_one(
    out: Path,
    variant: str,
    strat: str,
    x: np.ndarray,
    kinds: dict[str, dict[str, np.ndarray]],
    rng: np.random.Generator,
    bin_seconds: int,
    label: str = "combined f",
) -> list[str]:
    written: list[str] = []
    color = _COLORS[strat]
    idx = diverse_idx(x, 8, rng)
    png = f"gen_{variant}_{strat}.png"
    save_strip(
        out / png,
        x,
        title=f"{variant} {label}, h_anom={strat}",
        color=color,
        bin_seconds=bin_seconds,
        idx=idx,
    )
    written.append(png)
    written += _plot_kinds_vs_generated(
        out,
        f"{variant}_{strat}",
        {"combined": x},
        kinds,
        rng,
        bin_seconds,
        png_name=f"real_kinds_vs_{variant}_{strat}.png",
    )
    return written


def _plot_variant_grid(
    out: Path,
    variant: str,
    vgals: dict[str, np.ndarray],
    kinds: dict[str, dict[str, np.ndarray]],
    bin_seconds: int,
    label: str = "combined f",
) -> list[str]:
    if not kinds:
        return []
    col_names = [k for k in KIND_ORDER if k in kinds]
    row_names = ["real ESA-ADB"]
    for strat in ("soft", *STRATEGIES):
        if strat in vgals:
            row_names.append(f"{variant} {strat}")
    cells: dict[tuple[str, str], np.ndarray] = {}
    for col in col_names:
        cells[("real ESA-ADB", col)] = kinds[col]["x"][0]
        for strat in ("soft", *STRATEGIES):
            row = f"{variant} {strat}"
            if row not in row_names:
                continue
            match = nearest_generated(kinds[col]["x"][:1], vgals[strat])
            if len(match):
                cells[(row, col)] = match[0]
    colors = {
        "real ESA-ADB": "#b2182b",
        f"{variant} soft": _COLORS["soft"],
        f"{variant} nearest": _COLORS["nearest"],
        f"{variant} knn": _COLORS["knn"],
        f"{variant} proto": _COLORS["proto"],
    }
    png = f"real_kinds_vs_{variant}_hanom.png"
    save_named_grid(
        out / png,
        cells,
        row_names=row_names,
        col_names=col_names,
        bin_seconds=bin_seconds,
        colors=colors,
        title=f"ESA-ADB kinds vs {variant} {label}, h_anom strategies",
    )
    return [png]


def _plot_strategy_grid(
    out: Path,
    galleries: dict[str, dict[str, np.ndarray]],
    kinds: dict[str, dict[str, np.ndarray]],
    bin_seconds: int,
    label: str = "combined f",
) -> list[str]:
    if not kinds or "time_both" not in galleries:
        return []
    vgals = galleries["time_both"]
    col_names = [k for k in KIND_ORDER if k in kinds]
    row_names = ["real ESA-ADB"]
    for strat in ("soft", *STRATEGIES):
        if strat in vgals:
            row_names.append(strat)
    cells: dict[tuple[str, str], np.ndarray] = {}
    for col in col_names:
        cells[("real ESA-ADB", col)] = kinds[col]["x"][0]
        for strat in ("soft", *STRATEGIES):
            if strat not in vgals:
                continue
            match = nearest_generated(kinds[col]["x"][:1], vgals[strat])
            if len(match):
                cells[(strat, col)] = match[0]
    colors = {"real ESA-ADB": "#b2182b", **_COLORS}
    png = "real_kinds_vs_strategies_time_both.png"
    save_named_grid(
        out / png,
        cells,
        row_names=row_names,
        col_names=col_names,
        bin_seconds=bin_seconds,
        colors=colors,
        title=f"ESA-ADB kinds vs time_both {label} (h_anom strategies)",
    )
    return [png]


def _brief(row: dict[str, Any]) -> dict[str, Any]:
    skip = {"energy"}
    return {k: v for k, v in row.items() if k not in skip}


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _abs(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p
