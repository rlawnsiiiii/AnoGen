"""Sweep λ (band) and λ_anom (kind proto) for ESA stratified / hybrid.

Isolated results/shell_kindmix_htune/. Does not overwrite shell_s4 or
results/shell_kindmix_esa/.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from anogen.config import REPO_ROOT
from anogen.phases.hanom import _abs, _brief, _jsonable, _kind_hits, _load
from anogen.phases.kindmix import (
    HYBRID_NEEDLES_PROTO_KINDS,
    HYBRID_PROTO_KINDS,
    _PARENT50,
    _coverage_by_kind,
    _coverage_slices,
    _kind_alloc_labels,
    _stratified,
)
from anogen.phases.plot_tune import _real_kind_tables
from anogen.phases.tune import _score_gallery
from anogen.shell.coverage import mean_pairwise_distance
from anogen.shell.diffusion import DiffusionSchedule, denoiser_from_ckpt, torch_available
from anogen.shell.encoders import embed_shell
from anogen.shell.events import types_from_cfg
from anogen.shell.features import embed_windows
from anogen.shell.morphology import (
    KIND_ORDER,
    KIND_SLUG,
    kind_ref_indices,
    kinds_for_windows,
    nearest_generated,
    rank_for_kind,
    window_shape_stats,
)
from anogen.shell.plots import save_compare_rows, save_named_grid
from anogen.shell.scaler import resolve_scaler
from anogen.shell.steer import band_report

# λ = band leash. λ_anom = pull to the chosen kind proto.
GRID = (
    ("l03_a10", 0.3, 1.0),  # locked parent50
    ("l03_a05", 0.3, 0.5),
    ("l03_a03", 0.3, 0.3),
    ("l10_a10", 1.0, 1.0),
    ("l10_a05", 1.0, 0.5),
    ("l15_a05", 1.5, 0.5),
)

RECIPES = ("stratified", "hybrid", "hybrid_needles")
_RECIPE_PROTO = {
    "stratified": None,
    "hybrid": HYBRID_PROTO_KINDS,
    "hybrid_needles": HYBRID_NEEDLES_PROTO_KINDS,
}

_ROW_COLORS = {
    "real ESA-ADB": "#b2182b",
    "l03_a10": "#d95f02",
    "l03_a05": "#e6ab02",
    "l03_a03": "#7570b3",
    "l10_a10": "#2166ac",
    "l10_a05": "#1b9e77",
    "l15_a05": "#4d4d4d",
    "stratified": "#d95f02",
    "hybrid": "#1b9e77",
    "hybrid_needles": "#7570b3",
}


def total_variation(x: np.ndarray) -> np.ndarray:
    """Mean |x[t+1] − x[t]| per window (roughness)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2:
        return np.zeros(len(x), dtype=np.float64)
    return np.mean(np.abs(np.diff(x, axis=1)), axis=1)


def run_kindmix_htune(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s1 = _abs(cfg.get("s1_dir", root / "results/shell_s1"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    s4 = _abs(cfg.get("s4_dir", root / "results/shell_s4"), root)
    enc_dir = _abs(cfg.get("enc_dir", root / "results/shell_enc"), root)
    out = _abs(cfg.get("kindmix_htune_dir", root / "results/shell_kindmix_htune"), root)
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
    hcfg = dict((cfg.get("shell") or {}).get("kindmix_htune") or {})
    n = min(int(hcfg.get("n", 128)), len(parent))
    recipes = tuple(hcfg.get("recipes") or RECIPES)
    for rec in recipes:
        if rec not in _RECIPE_PROTO:
            raise ValueError(f"unknown htune recipe {rec}")
    x0 = parent[:n]
    ch = cond_ch[:n]
    bin_seconds = int(cfg.get("bin_seconds", 30))
    seed = int(cfg.get("seed", 0))
    force = bool(hcfg.get("force", False))

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

    kind_idx: dict[str, np.ndarray] = {}
    ref_info: dict[str, Any] = {}
    for kind in KIND_ORDER:
        idx, leaked = kind_ref_indices(kind_lab, fold_a, kind, query_fold=0)
        kind_idx[kind] = idx
        ref_info[kind] = {
            "n_all": int((kind_lab == kind).sum()),
            "n_ref": int(len(idx)),
            "n_oof": int(((kind_lab == kind) & (fold_a != 0)).sum()),
            "leaked": leaked,
        }
    active = [k for k in KIND_ORDER if len(kind_idx[k]) > 0]
    real_tv = {
        k: float(np.median(total_variation(examples[k]["x"]))) for k in active if k in examples
    }

    methods: dict[str, Any] = {}
    galleries: dict[str, dict[str, np.ndarray]] = {}
    written: list[str] = []

    for enc_name in ("time_recon", "time_both"):
        enc, shell, _ra, _rr = _load(
            enc_dir / f"{enc_name}_fold0.pt",
            x0,
            x_a,
            x_r,
            fold_a,
            fold_r,
            scfg,
            device,
            seed,
        )
        kind_z = {
            kind: torch.from_numpy(embed_shell(enc, x_a[idx], device))
            for kind, idx in kind_idx.items()
            if len(idx)
        }
        vgals: dict[str, np.ndarray] = {}
        for recipe in recipes:
            for tag, lam, lam_anom in GRID:
                key = f"{enc_name}_{recipe}_{tag}"
                cache = out / f"{key}.npz"
                recipe_kw = {**_PARENT50, "lam": float(lam), "lam_anom": float(lam_anom)}
                if cache.is_file() and not force:
                    blob = np.load(cache)
                    x = np.asarray(blob["x"])
                    h = np.asarray(blob["h"])
                    alloc = (
                        np.asarray(blob["kind_alloc"], dtype=object)
                        if "kind_alloc" in blob.files
                        else _kind_alloc_labels(len(x), active)
                    )
                else:
                    print(
                        f"kindmixhtune {enc_name} {recipe} {tag} λ={lam} λ_anom={lam_anom} n={n}",
                        flush=True,
                    )
                    common = dict(
                        ref=shell["ref"],
                        Q_q=float(shell["Q_q"]),
                        tau=float(shell["tau"]),
                        c_max=float(scfg.get("c_max", 1.0)),
                        device=device,
                        scaler=scaler,
                        bsz=bsz,
                        **recipe_kw,
                    )
                    x, h = _stratified(
                        model,
                        enc,
                        x0,
                        ch,
                        schedule,
                        common,
                        kind_z,
                        seed,
                        proto_kinds=_RECIPE_PROTO[recipe],
                    )
                    alloc = _kind_alloc_labels(len(x), active)
                    np.savez_compressed(
                        cache,
                        x=x,
                        h=h,
                        channel_idx=ch,
                        kind_alloc=alloc.astype(str),
                        lam=np.asarray(lam),
                        lam_anom=np.asarray(lam_anom),
                        Q_q=float(shell["Q_q"]),
                        delta=float(shell["delta"]),
                    )
                vgals[f"{recipe}_{tag}"] = x
                scored = _score_gallery(x, only_fold=0, **score_kw)
                scored.update(_slice_metrics(x, alloc, examples))
                scored["occupancy"] = float(np.mean(np.abs(h - shell["Q_q"]) <= shell["delta"]))
                scored["energy"] = band_report(h, float(shell["Q_q"]), float(shell["delta"]))
                scored["diversity"] = float(mean_pairwise_distance(embed_windows(x)))
                scored["kind_hits"] = _kind_hits(x)
                scored["coverage_kind"] = _coverage_by_kind(
                    x, x_a, fold_a, ch_a, kind_lab, fold_tab, tau_cov, fold_id=0
                )
                scored["slice_kind"] = _coverage_slices(
                    x, alloc, x_a, fold_a, ch_a, kind_lab, fold_tab, tau_cov, fold_id=0
                )
                scored["encoder"] = enc_name
                scored["recipe"] = recipe
                scored["tag"] = tag
                scored["lam"] = float(lam)
                scored["lam_anom"] = float(lam_anom)
                scored["n"] = int(len(x))
                methods[key] = scored
        galleries[enc_name] = vgals
        for recipe in recipes:
            rec_gals = {t: vgals[f"{recipe}_{t}"] for t, _, _ in GRID if f"{recipe}_{t}" in vgals}
            written += _plot_hp_grid(plots, enc_name, recipe, rec_gals, examples, bin_seconds, mode="aimed")
            written += _plot_hp_grid(plots, enc_name, recipe, rec_gals, examples, bin_seconds, mode="nearest")
            written += _plot_hp_kind_rows(plots, enc_name, recipe, rec_gals, examples, bin_seconds)
        written += _plot_recipe_at_baseline(plots, enc_name, vgals, examples, bin_seconds)

    recommend = recommend_hparams(methods, real_tv)
    written += _plot_metric_bars(plots, methods)
    written += _plot_recommend(plots, galleries, examples, recommend, bin_seconds)
    _write_metrics_csv(out / "metrics.csv", methods)
    (plots / "INDEX.txt").write_text(
        "ESA stratified / hybrid λ / λ_anom sweep. Docs: docs/KIND_MIX.md §9.\n"
        "l03_a10 = locked parent50 (λ=0.3, λ_anom=1).\n"
        "hybrid = proto on level shift only; band on points + subsequences.\n"
        "hybrid_needles = proto on shift + Point/Global; band on subsequences.\n"
        "Aimed = kind-slice best-stat; nearest = nearest in φ from the whole gallery.\n\n"
        + "\n".join(f"  {name}" for name in written)
        + "\n"
    )
    report = {
        "ok": True,
        "skipped": False,
        "tau": tau_cov,
        "n": n,
        "grid": [{"tag": t, "lam": a, "lam_anom": b} for t, a, b in GRID],
        "recipes": list(recipes),
        "refs": ref_info,
        "real_tv": real_tv,
        "methods": {k: _brief(v) for k, v in methods.items()},
        "recommend": recommend,
        "plots": written,
        "dir": str(out),
        "note": (
            "Stratified / hybrid λ/λ_anom sweep, 128 S3 parents, fold-0. "
            "Does not overwrite shell_s4 or shell_kindmix_esa."
        ),
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return report


def _slice_metrics(
    x: np.ndarray,
    alloc: np.ndarray,
    examples: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    alloc = np.asarray(alloc)
    amp, dmu = window_shape_stats(x)
    tv = total_variation(x)
    out: dict[str, Any] = {
        "tv_mean": float(tv.mean()),
        "tv_median": float(np.median(tv)),
        "slices": {},
    }
    for kind in KIND_ORDER:
        m = alloc == kind
        if not np.any(m):
            continue
        rec: dict[str, float] = {
            "n": float(m.sum()),
            "frac_amp": float(np.mean(amp[m] >= 0.1)),
            "frac_shift": float(np.mean(dmu[m] >= 0.01)),
            "med_amp": float(np.median(amp[m])),
            "med_dmu": float(np.median(dmu[m])),
            "med_tv": float(np.median(tv[m])),
        }
        if kind in examples:
            rec["real_med_tv"] = float(np.median(total_variation(examples[kind]["x"])))
        out["slices"][kind] = rec
    return out


def recommend_hparams(methods: dict[str, Any], real_tv: dict[str, float]) -> dict[str, Any]:
    """Pick λ / λ_anom / recipe. Keep shelves + needles, then lowest subsequence TV."""
    local = "real ESA local subsequence"
    glob = "real ESA global subsequence"
    point = "real ESA Point / Global"
    shift = "real level shift"
    picks: dict[str, Any] = {}
    for enc in ("time_recon", "time_both"):
        rows = []
        for recipe in RECIPES:
            for tag, lam, lam_anom in GRID:
                m = methods.get(f"{enc}_{recipe}_{tag}") or {}
                if not m:
                    continue
                sl = m.get("slices") or {}
                p = sl.get(point) or {}
                s = sl.get(shift) or {}
                lo = sl.get(local) or {}
                g = sl.get(glob) or {}
                keep_shift = float(s.get("frac_shift", 0.0)) >= 0.80
                keep_point = float(p.get("frac_amp", 0.0)) >= 0.50
                tv_l = float(lo.get("med_tv", 1.0))
                tv_g = float(g.get("med_tv", 1.0))
                target_l = float(real_tv.get(local, tv_l) or tv_l)
                target_g = float(real_tv.get(glob, tv_g) or tv_g)
                excess = max(0.0, tv_l / max(target_l, 1e-8) - 1.0) + max(
                    0.0, tv_g / max(target_g, 1e-8) - 1.0
                )
                rows.append(
                    {
                        "recipe": recipe,
                        "tag": tag,
                        "lam": lam,
                        "lam_anom": lam_anom,
                        "keep_shift": keep_shift,
                        "keep_point": keep_point,
                        "frac_shift": float(s.get("frac_shift", 0.0)),
                        "frac_amp_point": float(p.get("frac_amp", 0.0)),
                        "tv_local": tv_l,
                        "tv_global": tv_g,
                        "occupancy": float(m.get("occupancy", 0.0)),
                        "arp_anomaly": float(m.get("arp_anomaly", 0.0)),
                        "tv_median": float(m.get("tv_median", 0.0)),
                        "excess_tv": excess,
                    }
                )
        viable = [r for r in rows if r["keep_shift"] and r["keep_point"]]
        pool = viable or rows
        best = min(
            pool,
            key=lambda r: (r["excess_tv"], -r["occupancy"], -r["arp_anomaly"]),
        )
        picks[enc] = {
            "recipe": best["recipe"],
            "tag": best["tag"],
            "lam": best["lam"],
            "lam_anom": best["lam_anom"],
            "viable_kept_kinds": bool(viable),
            "reason": (
                "Among settings that keep shift-slice P(|Δμ|≥0.01)≥0.8 and "
                "Point/Global P(amp≥0.1)≥0.5, pick the lowest excess roughness "
                "on local/global subsequence vs the real crops; then occupancy, ARP."
                if viable
                else "No setting kept both needles and shelves; lowest excess TV overall."
            ),
            "candidates": rows,
        }
    return picks


def _plot_hp_grid(
    out: Path,
    enc_name: str,
    recipe: str,
    vgals: dict[str, np.ndarray],
    examples: dict[str, dict[str, np.ndarray]],
    bin_seconds: int,
    *,
    mode: str,
) -> list[str]:
    if not examples:
        return []
    col_names = [k for k in KIND_ORDER if k in examples]
    row_names = ["real ESA-ADB"] + [t for t, _, _ in GRID if t in vgals]
    cells: dict[tuple[str, str], np.ndarray] = {}
    for col in col_names:
        cells[("real ESA-ADB", col)] = examples[col]["x"][0]
        for tag, _, _ in GRID:
            if tag not in vgals:
                continue
            x = vgals[tag]
            if mode == "aimed":
                active = [k for k in KIND_ORDER if k in examples]
                alloc = _kind_alloc_labels(len(x), active)
                sl = x[alloc == col]
                if len(sl):
                    cells[(tag, col)] = sl[int(rank_for_kind(sl, col, 1)[0])]
            else:
                match = nearest_generated(examples[col]["x"][:1], x)
                if len(match):
                    cells[(tag, col)] = match[0]
    png = f"htune_{enc_name}_{recipe}_{mode}.png"
    save_named_grid(
        out / png,
        cells,
        row_names=row_names,
        col_names=col_names,
        bin_seconds=bin_seconds,
        colors=_ROW_COLORS,
        title=f"{recipe} λ/λ_anom ({enc_name}, {mode})",
        share_col_ylim=True,
    )
    return [png]


def _plot_hp_kind_rows(
    out: Path,
    enc_name: str,
    recipe: str,
    vgals: dict[str, np.ndarray],
    examples: dict[str, dict[str, np.ndarray]],
    bin_seconds: int,
) -> list[str]:
    written: list[str] = []
    for kind in KIND_ORDER:
        if kind not in examples:
            continue
        rows: dict[str, np.ndarray] = {kind: examples[kind]["x"][:4]}
        for tag, _, _ in GRID:
            if tag not in vgals:
                continue
            rows[f"{tag} nearest"] = nearest_generated(examples[kind]["x"][:4], vgals[tag])
        slug = KIND_SLUG.get(kind, "kind")
        png = f"htune_{enc_name}_{recipe}_{slug}_nearest.png"
        save_compare_rows(
            out / png,
            rows,
            bin_seconds=bin_seconds,
            n_cols=4,
            colors={kind: "#b2182b", **{f"{t} nearest": _ROW_COLORS[t] for t, _, _ in GRID}},
            title=f"{kind}: real vs nearest {recipe} HPs ({enc_name})",
        )
        written.append(png)
    return written


def _plot_recipe_at_baseline(
    out: Path,
    enc_name: str,
    vgals: dict[str, np.ndarray],
    examples: dict[str, dict[str, np.ndarray]],
    bin_seconds: int,
) -> list[str]:
    """Aimed-slice: stratified vs hybrid vs hybrid_needles at locked parent50."""
    tag = "l03_a10"
    recs = [r for r in RECIPES if f"{r}_{tag}" in vgals]
    col_names = [k for k in KIND_ORDER if k in examples]
    if not recs or not col_names:
        return []
    row_names = ["real ESA-ADB"] + recs
    cells: dict[tuple[str, str], np.ndarray] = {}
    for col in col_names:
        cells[("real ESA-ADB", col)] = examples[col]["x"][0]
        for rec in recs:
            x = vgals[f"{rec}_{tag}"]
            active = [k for k in KIND_ORDER if k in examples]
            alloc = _kind_alloc_labels(len(x), active)
            sl = x[alloc == col]
            if len(sl):
                cells[(rec, col)] = sl[int(rank_for_kind(sl, col, 1)[0])]
    png = f"htune_{enc_name}_recipes_parent50_aimed.png"
    save_named_grid(
        out / png,
        cells,
        row_names=row_names,
        col_names=col_names,
        bin_seconds=bin_seconds,
        colors=_ROW_COLORS,
        title=f"Recipes at parent50 λ=0.3 λ_anom=1 ({enc_name}, aimed)",
        share_col_ylim=True,
    )
    return [png]


def _plot_recommend(
    out: Path,
    galleries: dict[str, dict[str, np.ndarray]],
    examples: dict[str, dict[str, np.ndarray]],
    recommend: dict[str, Any],
    bin_seconds: int,
) -> list[str]:
    written: list[str] = []
    col_names = [k for k in KIND_ORDER if k in examples]
    if not col_names:
        return written
    for enc_name in ("time_recon", "time_both"):
        pick = recommend.get(enc_name) or {}
        vgals = galleries.get(enc_name) or {}
        rec = pick.get("recipe")
        tag = pick.get("tag")
        if not rec or not tag:
            continue
        key = f"{rec}_{tag}"
        if key not in vgals:
            continue
        x = vgals[key]
        base = vgals.get("stratified_l03_a10")
        active = [k for k in KIND_ORDER if k in examples]
        alloc = _kind_alloc_labels(len(x), active)
        row_names = ["real ESA-ADB", f"recommend {rec} {tag}"]
        cells: dict[tuple[str, str], np.ndarray] = {}
        for col in col_names:
            cells[("real ESA-ADB", col)] = examples[col]["x"][0]
            sl = x[alloc == col]
            if len(sl):
                cells[(row_names[1], col)] = sl[int(rank_for_kind(sl, col, 1)[0])]
        if base is not None:
            row_names.append("stratified parent50")
            balloc = _kind_alloc_labels(len(base), active)
            for col in col_names:
                sl = base[balloc == col]
                if len(sl):
                    cells[("stratified parent50", col)] = sl[int(rank_for_kind(sl, col, 1)[0])]
        png = f"htune_{enc_name}_recommend_aimed.png"
        save_named_grid(
            out / png,
            cells,
            row_names=row_names,
            col_names=col_names,
            bin_seconds=bin_seconds,
            colors={
                "real ESA-ADB": "#b2182b",
                row_names[1]: "#1b9e77",
                "stratified parent50": "#d95f02",
            },
            title=f"Recommended {rec} {tag} vs parent50 stratified ({enc_name})",
            share_col_ylim=True,
        )
        written.append(png)
    return written


def _plot_metric_bars(out: Path, methods: dict[str, Any]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    tags = [t for t, _, _ in GRID]
    rec_colors = {"stratified": "#d95f02", "hybrid": "#1b9e77", "hybrid_needles": "#7570b3"}
    written: list[str] = []
    for enc in ("time_recon", "time_both"):
        fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.4))
        series = (
            ("occupancy", "Occupancy", None),
            ("arp_anomaly", "ARP anom", None),
            ("tv_median", "Median TV (all)", None),
            ("frac_shift", "Shift-slice P(|Δμ|≥0.01)", "real level shift"),
            ("frac_amp_point", "Point/Global P(amp≥0.1)", "real ESA Point / Global"),
            ("tv_subseq", "Median TV local+global", None),
        )
        for ax, (key, title, kind) in zip(axes.ravel(), series, strict=True):
            for recipe in RECIPES:
                ys = []
                for tag, _, _ in GRID:
                    m = methods.get(f"{enc}_{recipe}_{tag}") or {}
                    if key == "frac_shift":
                        ys.append(float(((m.get("slices") or {}).get(kind) or {}).get("frac_shift", float("nan"))))
                    elif key == "frac_amp_point":
                        ys.append(float(((m.get("slices") or {}).get(kind) or {}).get("frac_amp", float("nan"))))
                    elif key == "tv_subseq":
                        sl = m.get("slices") or {}
                        lo = float((sl.get("real ESA local subsequence") or {}).get("med_tv", float("nan")))
                        gl = float((sl.get("real ESA global subsequence") or {}).get("med_tv", float("nan")))
                        ys.append(float(np.nanmean([lo, gl])))
                    else:
                        ys.append(float(m.get(key, float("nan"))))
                ax.plot(range(len(tags)), ys, marker="o", label=recipe, color=rec_colors[recipe])
            ax.set_xticks(range(len(tags)))
            ax.set_xticklabels([t.replace("_", "\n") for t in tags], fontsize=7)
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"λ / λ_anom sweep ({enc}, 128 donors, fold 0)", fontsize=11)
        fig.tight_layout()
        png = f"htune_{enc}_metrics.png"
        fig.savefig(out / png, dpi=140)
        plt.close(fig)
        written.append(png)
    return written


def _write_metrics_csv(path: Path, methods: dict[str, Any]) -> None:
    point = "real ESA Point / Global"
    shift = "real level shift"
    local = "real ESA local subsequence"
    glob = "real ESA global subsequence"
    rows = []
    for key, m in methods.items():
        sl = m.get("slices") or {}
        rows.append(
            {
                "key": key,
                "encoder": m.get("encoder"),
                "recipe": m.get("recipe"),
                "tag": m.get("tag"),
                "lam": m.get("lam"),
                "lam_anom": m.get("lam_anom"),
                "occupancy": m.get("occupancy"),
                "arp_anomaly": m.get("arp_anomaly"),
                "diversity": m.get("diversity"),
                "tv_median": m.get("tv_median"),
                "frac_shift_slice": (sl.get(shift) or {}).get("frac_shift"),
                "frac_amp_point": (sl.get(point) or {}).get("frac_amp"),
                "tv_local": (sl.get(local) or {}).get("med_tv"),
                "tv_global": (sl.get(glob) or {}).get("med_tv"),
                "coverage_anomaly": m.get("coverage_anomaly"),
            }
        )
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
