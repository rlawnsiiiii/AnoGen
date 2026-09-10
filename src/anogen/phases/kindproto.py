"""Kind-conditional proto: aim each gallery at one morphology kind.

Same proto energy as H_ANOM.md, different ref list (LEVEL_SHIFT.md idea A).
Recipes: locked parent ν=0.2 / 50 DDIM steps, and from-noise / 200 steps
(n_times=200, so 200 is one iterate per diffusion time).

Isolated results/shell_kindproto/. Does not overwrite shell_s4 or shell_hanom.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from anogen.config import REPO_ROOT
from anogen.phases.hanom import _abs, _brief, _jsonable, _kind_hits, _load
from anogen.phases.plot_tune import _real_kind_tables
from anogen.phases.tune import _keep_mask, _score_gallery
from anogen.shell.coverage import arp, coverage_at_tau, mean_pairwise_distance, min_distances
from anogen.shell.diffusion import DiffusionSchedule, denoiser_from_ckpt, torch_available
from anogen.shell.encoders import embed_shell
from anogen.shell.features import embed_windows
from anogen.shell.events import types_from_cfg
from anogen.shell.morphology import (
    KIND_ORDER,
    KIND_SLUG,
    kind_ref_indices,
    kind_stat,
    kinds_for_windows,
    rank_for_kind,
    window_shape_stats,
)
from anogen.shell.plots import save_compare_rows, save_named_grid, save_strip
from anogen.shell.scaler import resolve_scaler
from anogen.shell.steer import band_report, chunked_guided_ddim

_PARENT50 = {
    "nu": 0.2,
    "lam": 0.3,
    "ddim_steps": 50,
    "normalize_grad": True,
    "n_correct": 1,
    "lam_anom": 1.0,
    "lam_rare": 0.0,
    "start_from_noise": False,
}
_NOISE200 = {
    "nu": 1.0,
    "lam": 0.3,
    "ddim_steps": 200,
    "normalize_grad": True,
    "n_correct": 1,
    "lam_anom": 1.0,
    "lam_rare": 0.0,
    "start_from_noise": True,
}

RECIPES = {
    "parent50": _PARENT50,
    "noise200": _NOISE200,
}

_COLORS = {
    "real ESA-ADB": "#b2182b",
    "time_recon": "#2166ac",
    "time_both": "#1b9e77",
    "parent50": "#d95f02",
    "noise200": "#7570b3",
}


def run_kindproto(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s1 = _abs(cfg.get("s1_dir", root / "results/shell_s1"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    s4 = _abs(cfg.get("s4_dir", root / "results/shell_s4"), root)
    enc_dir = _abs(cfg.get("enc_dir", root / "results/shell_enc"), root)
    out = _abs(cfg.get("kindproto_dir", root / "results/shell_kindproto"), root)
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
        "meta": s0 / "labeled_windows.csv",
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
    meta = pd.read_csv(s0 / "labeled_windows.csv")
    x_a = np.asarray(lab["anomaly"])
    x_r = np.asarray(lab["rare"])
    fold_a = np.asarray(lab["anomaly_fold"])
    fold_r = np.asarray(lab["rare_fold"])
    ch_a = np.asarray(lab["anomaly_channel"])
    ch_r = np.asarray(lab["rare_channel"])
    parent = np.asarray(gal["cond"])
    cond_ch = np.asarray(gal["channel_idx"])
    anom = meta[meta["kind"] == "anomaly"].reset_index(drop=True)
    types = types_from_cfg(cfg)
    kind_lab = kinds_for_windows(x_a, anom, types)
    examples = _real_kind_tables(x_a, meta, types)
    kcfg = dict((cfg.get("shell") or {}).get("kindproto") or {})
    n = min(int(kcfg.get("n", 128)), len(parent))
    x0 = parent[:n]
    ch = cond_ch[:n]
    bin_seconds = int(cfg.get("bin_seconds", 30))
    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    den = torch.load(s1 / "denoiser.pt", map_location=device, weights_only=False)
    model = denoiser_from_ckpt(den, device=device_t)
    scaler = resolve_scaler(den, s0)
    schedule = DiffusionSchedule.linear(int(den["n_times"])).to(device_t)
    scfg = dict((cfg.get("shell") or {}).get("steer") or {})
    bsz = int(scfg.get("n_sample", 64))
    n_times = int(den["n_times"])

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

    methods: dict[str, Any] = {}
    galleries: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    written: list[str] = []
    ref_info: dict[str, Any] = {}

    for variant in ("time_recon", "time_both"):
        enc, shell, _ra, _rr = _load(
            enc_dir / f"{variant}_fold0.pt",
            x0,
            x_a,
            x_r,
            fold_a,
            fold_r,
            scfg,
            device,
            seed,
        )
        galleries.setdefault(variant, {})
        for rec_name, recipe in RECIPES.items():
            if rec_name == "noise200":
                recipe = {**recipe, "ddim_steps": int(kcfg.get("ddim_steps_long", 200))}
            vgals: dict[str, np.ndarray] = {}
            for kind in KIND_ORDER:
                slug = KIND_SLUG[kind]
                idx, leaked = kind_ref_indices(kind_lab, fold_a, kind, query_fold=0)
                ref_info[kind] = {
                    "n_all": int((kind_lab == kind).sum()),
                    "n_ref": int(len(idx)),
                    "n_oof": int(((kind_lab == kind) & (fold_a != 0)).sum()),
                    "leaked": leaked,
                }
                key = f"{variant}_{rec_name}_{slug}"
                if len(idx) == 0:
                    methods[key] = {
                        "ok": False,
                        "skipped": True,
                        "reason": f"no refs for {kind}",
                        "encoder": variant,
                        "recipe": rec_name,
                        "kind": kind,
                    }
                    continue
                cache = out / f"{key}.npz"
                if cache.is_file() and not bool(kcfg.get("force", False)):
                    blob = np.load(cache)
                    x = np.asarray(blob["x"])
                    h = np.asarray(blob["h"])
                else:
                    print(
                        f"kindproto {variant} {rec_name} {slug} n={n} refs={len(idx)} "
                        f"leaked={leaked} steps={recipe['ddim_steps']}",
                        flush=True,
                    )
                    ra = torch.from_numpy(embed_shell(enc, x_a[idx], device))
                    x, h = chunked_guided_ddim(
                        model,
                        enc,
                        x0,
                        ch,
                        schedule,
                        ref=shell["ref"],
                        Q_q=float(shell["Q_q"]),
                        tau=float(shell["tau"]),
                        c_max=float(scfg.get("c_max", 1.0)),
                        device=device,
                        scaler=scaler,
                        bsz=bsz,
                        ref_anom=ra,
                        anom_energy_kind="proto",
                        anom_proto_seed=seed + 100 * (1 + KIND_ORDER.index(kind)),
                        **recipe,
                    )
                    np.savez_compressed(
                        cache,
                        x=x,
                        h=h,
                        channel_idx=ch,
                        Q_q=float(shell["Q_q"]),
                        delta=float(shell["delta"]),
                        kind=np.asarray(kind),
                        recipe=np.asarray(rec_name),
                        leaked=np.asarray(leaked),
                        n_ref=np.asarray(len(idx)),
                    )
                vgals[kind] = x
                scored = _score_gallery(x, only_fold=0, **score_kw)
                scored["occupancy"] = float(np.mean(np.abs(h - shell["Q_q"]) <= shell["delta"]))
                scored["energy"] = band_report(h, float(shell["Q_q"]), float(shell["delta"]))
                scored["diversity"] = float(mean_pairwise_distance(embed_windows(x)))
                scored["kind_hits"] = _kind_hits(x)
                scored["coverage_kind"] = _coverage_kind(
                    x, x_a, fold_a, ch_a, kind_lab, kind, fold_tab, tau_cov
                )
                scored["encoder"] = variant
                scored["recipe"] = rec_name
                scored["kind"] = kind
                scored["n"] = int(len(x))
                scored["n_ref"] = int(len(idx))
                scored["leaked"] = leaked
                scored["ddim_steps"] = int(recipe["ddim_steps"])
                scored["start_from_noise"] = bool(recipe["start_from_noise"])
                scored["n_times"] = n_times
                scored["shot"] = "kind-leaky" if leaked else "FS-kind"
                scored["note"] = (
                    "Kind-conditional proto, fold-0 queries, frozen S4 τ. "
                    "Not a new encscore freeze. "
                    + (
                        "Refs include fold-0 windows of this kind (no OOF refs)."
                        if leaked
                        else "Refs are event-OOF windows of this kind."
                    )
                )
                methods[key] = scored
            galleries[variant][rec_name] = vgals
            written += _plot_recipe(plots, variant, rec_name, vgals, examples, rng, bin_seconds)

    written += _plot_overview(plots, galleries, examples, bin_seconds)
    written += _plot_level_shift(plots, galleries, examples, bin_seconds)
    (plots / "INDEX.txt").write_text(
        "Kind-conditional proto on Mission 1 41–46.\n"
        "Each gallery aims proto at one ESA Length × Locality kind "
        "(plus the level-shift overlay; docs/ESA_LABELS.md).\n"
        "parent50 = ν=0.2 / 50 DDIM steps. noise200 = x_T~N(0,I) / 200 steps.\n"
        "Level-shift / medium refs leak fold-0 (all those windows are fold 0).\n\n"
        + "\n".join(f"  {name}" for name in written)
        + "\n"
    )

    report = {
        "ok": True,
        "skipped": False,
        "tau": tau_cov,
        "tau_source": "s4_frozen",
        "n": n,
        "n_times": n_times,
        "ddim_steps_locked": 50,
        "ddim_steps_long": int(kcfg.get("ddim_steps_long", 200)),
        "refs": ref_info,
        "methods": {k: _brief(v) for k, v in methods.items()},
        "plots": written,
        "dir": str(out),
        "note": (
            "Kind-conditional proto (LEVEL_SHIFT idea A). "
            "256-or-less S3 parents, fold-0 Coverage@τ. Does not overwrite shell_s4."
        ),
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return report


def _coverage_kind(
    gallery: np.ndarray,
    x_a: np.ndarray,
    fold_a: np.ndarray,
    ch_a: np.ndarray,
    kind_lab: np.ndarray,
    kind: str,
    fold_tab: pd.DataFrame,
    tau: float,
    *,
    fold_id: int = 0,
) -> dict[str, float]:
    keep = _keep_mask(ch_a, fold_a, fold_tab, fold_id=fold_id) & (kind_lab == kind)
    q = x_a[keep]
    if len(q) == 0 or len(gallery) == 0:
        return {"n_query": 0.0, "coverage": float("nan"), "arp": float("nan")}
    d = min_distances(embed_windows(q), embed_windows(gallery))
    return {
        "n_query": float(len(q)),
        "coverage": coverage_at_tau(d, tau),
        "arp": arp(d),
    }


def _plot_recipe(
    out: Path,
    variant: str,
    rec_name: str,
    vgals: dict[str, np.ndarray],
    examples: dict[str, dict[str, np.ndarray]],
    rng: np.random.Generator,
    bin_seconds: int,
) -> list[str]:
    written: list[str] = []
    color = _COLORS.get(variant, "#333333")
    for kind, x in vgals.items():
        slug = KIND_SLUG[kind]
        idx = rank_for_kind(x, kind, 8)
        png = f"aimed_{variant}_{rec_name}_{slug}.png"
        save_strip(
            out / png,
            x,
            title=f"{variant} {rec_name} aimed at {kind}",
            color=color,
            bin_seconds=bin_seconds,
            idx=idx,
        )
        written.append(png)
        if kind not in examples:
            continue
        rows = {
            kind: examples[kind]["x"][:4],
            f"best {rec_name}": x[rank_for_kind(x, kind, 4)],
        }
        cmp = f"real_vs_aimed_{variant}_{rec_name}_{slug}.png"
        save_compare_rows(
            out / cmp,
            rows,
            bin_seconds=bin_seconds,
            n_cols=4,
            colors={kind: "#b2182b", f"best {rec_name}": color},
            title=f"Real {kind} vs best-stat {variant} {rec_name}",
        )
        written.append(cmp)

    col_names = [k for k in KIND_ORDER if k in vgals and k in examples]
    if not col_names:
        return written
    row_names = ["real ESA-ADB", f"{variant} best-stat", f"{variant} typical"]
    cells: dict[tuple[str, str], np.ndarray] = {}
    titles: dict[tuple[str, str], str] = {}
    for col in col_names:
        cells[("real ESA-ADB", col)] = examples[col]["x"][0]
        x = vgals[col]
        best = rank_for_kind(x, col, 1)
        mid = int(np.argsort(-kind_stat(x, col))[len(x) // 2])
        cells[(f"{variant} best-stat", col)] = x[int(best[0])]
        cells[(f"{variant} typical", col)] = x[mid]
        amp, dmu = window_shape_stats(x)
        titles[(f"{variant} best-stat", col)] = f"|Δμ|={dmu[int(best[0])]:.3f}"
        titles[(f"{variant} typical", col)] = f"|Δμ|={dmu[mid]:.3f}"
    png = f"real_vs_aimed_{variant}_{rec_name}.png"
    save_named_grid(
        out / png,
        cells,
        row_names=row_names,
        col_names=col_names,
        bin_seconds=bin_seconds,
        colors={
            "real ESA-ADB": "#b2182b",
            f"{variant} best-stat": color,
            f"{variant} typical": "#7570b3",
        },
        cell_titles=titles,
        title=f"Kind-conditional proto ({variant} {rec_name}). Best = kind rule, not φ.",
        share_col_ylim=True,
    )
    written.append(png)
    del rng
    return written


def _plot_overview(
    out: Path,
    galleries: dict[str, dict[str, dict[str, np.ndarray]]],
    examples: dict[str, dict[str, np.ndarray]],
    bin_seconds: int,
) -> list[str]:
    if "time_both" not in galleries:
        return []
    col_names = [k for k in KIND_ORDER if k in examples]
    row_names = ["real ESA-ADB"]
    cells: dict[tuple[str, str], np.ndarray] = {}
    titles: dict[tuple[str, str], str] = {}
    colors = {"real ESA-ADB": "#b2182b"}
    for col in col_names:
        cells[("real ESA-ADB", col)] = examples[col]["x"][0]
    for rec_name, color in (("parent50", "#d95f02"), ("noise200", "#7570b3")):
        vgals = galleries["time_both"].get(rec_name) or {}
        row = f"time_both {rec_name}"
        if not vgals:
            continue
        row_names.append(row)
        colors[row] = color
        for col in col_names:
            if col not in vgals:
                continue
            x = vgals[col]
            j = int(rank_for_kind(x, col, 1)[0])
            cells[(row, col)] = x[j]
            _, dmu = window_shape_stats(x)
            titles[(row, col)] = f"|Δμ|={dmu[j]:.3f}"
    png = "real_vs_aimed_time_both.png"
    save_named_grid(
        out / png,
        cells,
        row_names=row_names,
        col_names=col_names,
        bin_seconds=bin_seconds,
        colors=colors,
        cell_titles=titles,
        title="ESA-ADB kinds vs kind-conditional proto (time_both, best-stat)",
        share_col_ylim=True,
    )
    return [png]


def _plot_level_shift(
    out: Path,
    galleries: dict[str, dict[str, dict[str, np.ndarray]]],
    examples: dict[str, dict[str, np.ndarray]],
    bin_seconds: int,
) -> list[str]:
    kind = "real level shift"
    if kind not in examples:
        return []
    written: list[str] = []
    real = examples[kind]["x"]
    for variant, recs in galleries.items():
        for rec_name, vgals in recs.items():
            if kind not in vgals:
                continue
            x = vgals[kind]
            rows = {
                "real level shift": real[:4],
                "best |μ_L−μ_R|": x[rank_for_kind(x, kind, 4)],
            }
            png = f"level_shift_{variant}_{rec_name}.png"
            save_compare_rows(
                out / png,
                rows,
                bin_seconds=bin_seconds,
                n_cols=4,
                colors={"real level shift": "#b2182b", "best |μ_L−μ_R|": _COLORS.get(variant, "#333")},
                title=f"Level shift: real vs best |Δμ| ({variant} {rec_name})",
            )
            written.append(png)
    return written
