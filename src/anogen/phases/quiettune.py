"""Narrow λ_anom on hybrid_needles subsequence slices (texture, not Locality).

Base recipe is locked hybrid_needles (λ=0.3, λ_anom=1 on shift + Point/Global).
The two quiet kinds currently run band (λ_anom=0). This sweep tries small proto
pulls so median TV / spectrum sit nearer the real crops.

Isolated results/shell_quiettune/. Does not overwrite shell_s4, kindmix_htune,
or any 1536 freeze. Matching roughness is not ESA Local vs Global.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from anogen.config import REPO_ROOT
from anogen.phases.hanom import _abs, _brief, _jsonable, _kind_hits, _load
from anogen.phases.kindmix import (
    HYBRID_NEEDLES_PROTO_KINDS,
    _PARENT50,
    _coverage_by_kind,
    _coverage_slices,
    _kind_alloc_labels,
    _stratified,
)
from anogen.phases.kindmixhtune import _slice_metrics, total_variation
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
)
from anogen.shell.plots import save_compare_rows, save_named_grid
from anogen.shell.scaler import resolve_scaler
from anogen.shell.steer import band_report

LOCAL = "real ESA local subsequence"
GLOBAL = "real ESA global subsequence"
POINT = "real ESA Point / Global"
SHIFT = "real level shift"
QUIET_KINDS = (LOCAL, GLOBAL)

DEFAULT_LAM_QUIET = (0.0, 0.05, 0.10, 0.15, 0.20)

_ROW_COLORS = {
    "real ESA-ADB": "#b2182b",
    "q00": "#4d4d4d",
    "q05": "#7570b3",
    "q10": "#1b9e77",
    "q15": "#d95f02",
    "q20": "#2166ac",
}


def hf_lf_ratio(x: np.ndarray) -> np.ndarray:
    """High / low rFFT energy from feature_pack_v1 log-bands (last four cols)."""
    feats = embed_windows(x)
    if len(feats) == 0:
        return np.zeros(0, dtype=np.float64)
    energy = np.exp(feats[:, -4:])
    low = energy[:, :2].mean(axis=1)
    high = energy[:, 2:].mean(axis=1)
    return high / np.maximum(low, 1e-8)


def two_sided_rel(value: float, target: float) -> float:
    """|value/target − 1|. Zero when value matches target."""
    t = float(target)
    if not np.isfinite(t) or abs(t) < 1e-12:
        return float("nan")
    return float(abs(float(value) / t - 1.0))


def needles_quiet_map(lam_local: float, lam_global: float) -> dict[str, float]:
    """hybrid_needles structured kinds at λ_anom=1, quiet kinds at the given λ."""
    out = {k: 1.0 for k in HYBRID_NEEDLES_PROTO_KINDS}
    out[LOCAL] = float(lam_local)
    out[GLOBAL] = float(lam_global)
    return out


def _tag(lam_local: float, lam_global: float) -> str:
    if abs(lam_local - lam_global) < 1e-12:
        return f"q{int(round(lam_local * 100)):02d}"
    return f"qL{int(round(lam_local * 100)):02d}_g{int(round(lam_global * 100)):02d}"


def _quiet_jobs(qcfg: dict[str, Any]) -> list[tuple[str, float, float]]:
    shared = [float(x) for x in (qcfg.get("lam_quiet") or DEFAULT_LAM_QUIET)]
    loc = qcfg.get("lam_local")
    glob = qcfg.get("lam_global")
    if loc is None and glob is None:
        return [(_tag(a, a), a, a) for a in shared]
    loc_l = [float(x) for x in (loc if loc is not None else shared)]
    glob_l = [float(x) for x in (glob if glob is not None else shared)]
    jobs = []
    for a in loc_l:
        for b in glob_l:
            jobs.append((_tag(a, b), a, b))
    return jobs


def run_quiettune(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s1 = _abs(cfg.get("s1_dir", root / "results/shell_s1"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    s4 = _abs(cfg.get("s4_dir", root / "results/shell_s4"), root)
    enc_dir = _abs(cfg.get("enc_dir", root / "results/shell_enc"), root)
    out = _abs(cfg.get("quiettune_dir", root / "results/shell_quiettune"), root)
    plots = out / "plots"
    docs_plots = root / "docs" / "quiettune"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    docs_plots.mkdir(parents=True, exist_ok=True)

    locked = {s3, s4}
    if out.resolve() in {p.resolve() for p in locked}:
        raise RuntimeError("quiettune must not write into shell_s3 or shell_s4")

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
    qcfg = dict((cfg.get("shell") or {}).get("quiettune") or {})
    n = min(int(qcfg.get("n", 128)), len(parent))
    jobs = _quiet_jobs(qcfg)
    x0 = parent[:n]
    ch = cond_ch[:n]
    bin_seconds = int(cfg.get("bin_seconds", 30))
    seed = int(cfg.get("seed", 0))
    force = bool(qcfg.get("force", False))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    den = torch.load(s1 / "denoiser.pt", map_location=device, weights_only=False)
    model = denoiser_from_ckpt(den, device=device_t)
    scaler = resolve_scaler(den, s0)
    schedule = DiffusionSchedule.from_ckpt(den).to(device_t)
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
    real_spec = {
        k: float(np.median(hf_lf_ratio(examples[k]["x"]))) for k in active if k in examples
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
        for tag, lam_l, lam_g in jobs:
            key = f"{enc_name}_{tag}"
            cache = out / f"{key}.npz"
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
                    f"quiettune {enc_name} {tag} λ_local={lam_l:g} λ_global={lam_g:g} n={n}",
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
                    **_PARENT50,
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
                    proto_kinds=needles_quiet_map(lam_l, lam_g),
                )
                alloc = _kind_alloc_labels(len(x), active)
                np.savez_compressed(
                    cache,
                    x=x,
                    h=h,
                    channel_idx=ch,
                    kind_alloc=alloc.astype(str),
                    lam_local=np.asarray(lam_l),
                    lam_global=np.asarray(lam_g),
                    Q_q=float(shell["Q_q"]),
                    delta=float(shell["delta"]),
                )
            vgals[tag] = x
            scored = _score_gallery(x, only_fold=0, **score_kw)
            scored.update(_slice_metrics(x, alloc, examples))
            _add_spec_slices(scored, x, alloc, examples)
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
            scored["tag"] = tag
            scored["lam_local"] = float(lam_l)
            scored["lam_global"] = float(lam_g)
            scored["n"] = int(len(x))
            methods[key] = scored
        galleries[enc_name] = vgals
        written += _plot_aimed_grid(plots, enc_name, vgals, examples, jobs, bin_seconds)
        written += _plot_kind_strips(plots, enc_name, vgals, examples, jobs, bin_seconds, mode="aimed")
        written += _plot_kind_strips(plots, enc_name, vgals, examples, jobs, bin_seconds, mode="nearest")

    recommend = recommend_quiet(methods, real_tv, real_spec)
    written += _plot_tv_curves(plots, methods, real_tv, jobs)
    written += _plot_recommend(plots, galleries, examples, recommend, bin_seconds)
    _write_metrics_csv(out / "metrics.csv", methods)
    (plots / "INDEX.txt").write_text(_index_text(written, jobs, recommend))
    for name in written + ["INDEX.txt"]:
        src = plots / name
        if src.is_file():
            shutil.copy2(src, docs_plots / name)
    if (out / "metrics.csv").is_file():
        shutil.copy2(out / "metrics.csv", docs_plots / "metrics.csv")

    report = {
        "ok": True,
        "skipped": False,
        "tau": tau_cov,
        "n": n,
        "jobs": [{"tag": t, "lam_local": a, "lam_global": b} for t, a, b in jobs],
        "refs": ref_info,
        "real_tv": real_tv,
        "real_spec": real_spec,
        "methods": {k: _brief(v) for k, v in methods.items()},
        "recommend": recommend,
        "plots": written,
        "dir": str(out),
        "docs_plots": str(docs_plots),
        "note": (
            "hybrid_needles with small proto λ_anom on local/global subsequence. "
            "Fold-0, 128 S3 parents. Texture match, not ESA Locality. "
            "Does not overwrite shell_s4 or kindmix_htune."
        ),
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    shutil.copy2(out / "summary.json", docs_plots / "summary.json")
    return report


def _add_spec_slices(
    scored: dict[str, Any],
    x: np.ndarray,
    alloc: np.ndarray,
    examples: dict[str, dict[str, np.ndarray]],
) -> None:
    alloc = np.asarray(alloc)
    ratio = hf_lf_ratio(x)
    sl = scored.setdefault("slices", {})
    for kind in KIND_ORDER:
        m = alloc == kind
        if not np.any(m):
            continue
        rec = sl.setdefault(kind, {})
        rec["med_spec"] = float(np.median(ratio[m]))
        if kind in examples:
            rec["real_med_spec"] = float(np.median(hf_lf_ratio(examples[kind]["x"])))


def recommend_quiet(
    methods: dict[str, Any],
    real_tv: dict[str, float],
    real_spec: dict[str, float],
) -> dict[str, Any]:
    """Keep needles/shelves, then two-sided TV then spectrum on quiet slices."""
    picks: dict[str, Any] = {}
    t_l = float(real_tv.get(LOCAL, float("nan")))
    t_g = float(real_tv.get(GLOBAL, float("nan")))
    s_l = float(real_spec.get(LOCAL, float("nan")))
    s_g = float(real_spec.get(GLOBAL, float("nan")))
    for enc in ("time_recon", "time_both"):
        rows = []
        for key, m in methods.items():
            if m.get("encoder") != enc:
                continue
            sl = m.get("slices") or {}
            p = sl.get(POINT) or {}
            sh = sl.get(SHIFT) or {}
            lo = sl.get(LOCAL) or {}
            g = sl.get(GLOBAL) or {}
            tv_l = float(lo.get("med_tv", float("nan")))
            tv_g = float(g.get("med_tv", float("nan")))
            sp_l = float(lo.get("med_spec", float("nan")))
            sp_g = float(g.get("med_spec", float("nan")))
            err_tv = two_sided_rel(tv_l, t_l) + two_sided_rel(tv_g, t_g)
            err_spec = two_sided_rel(sp_l, s_l) + two_sided_rel(sp_g, s_g)
            keep_shift = float(sh.get("frac_shift", 0.0)) >= 0.80
            keep_point = float(p.get("frac_amp", 0.0)) >= 0.50
            rows.append(
                {
                    "key": key,
                    "tag": m.get("tag"),
                    "lam_local": float(m.get("lam_local", 0.0)),
                    "lam_global": float(m.get("lam_global", 0.0)),
                    "keep_shift": keep_shift,
                    "keep_point": keep_point,
                    "frac_shift": float(sh.get("frac_shift", 0.0)),
                    "frac_amp_point": float(p.get("frac_amp", 0.0)),
                    "tv_local": tv_l,
                    "tv_global": tv_g,
                    "spec_local": sp_l,
                    "spec_global": sp_g,
                    "err_tv": err_tv,
                    "err_spec": err_spec,
                    "occupancy": float(m.get("occupancy", 0.0)),
                    "arp_anomaly": float(m.get("arp_anomaly", 0.0)),
                }
            )
        viable = [r for r in rows if r["keep_shift"] and r["keep_point"]]
        pool = viable or rows
        if not pool:
            picks[enc] = {"tag": None, "viable_kept_kinds": False, "candidates": []}
            continue
        best = min(
            pool,
            key=lambda r: (
                r["err_tv"] if np.isfinite(r["err_tv"]) else 1e9,
                r["err_spec"] if np.isfinite(r["err_spec"]) else 1e9,
                -r["occupancy"],
                -r["arp_anomaly"],
            ),
        )
        picks[enc] = {
            "tag": best["tag"],
            "lam_local": best["lam_local"],
            "lam_global": best["lam_global"],
            "viable_kept_kinds": bool(viable),
            "err_tv": best["err_tv"],
            "err_spec": best["err_spec"],
            "reason": (
                "Among settings with shift-slice P(|Δμ|≥0.01)≥0.8 and "
                "Point/Global P(amp≥0.1)≥0.5, pick lowest two-sided TV error "
                "on local/global subsequence vs real crops; then two-sided "
                "HF/LF spectrum error; then occupancy, ARP."
                if viable
                else "No setting kept both needles and shelves; lowest two-sided TV overall."
            ),
            "candidates": rows,
        }
    return picks


def _plot_aimed_grid(
    out: Path,
    enc_name: str,
    vgals: dict[str, np.ndarray],
    examples: dict[str, dict[str, np.ndarray]],
    jobs: list[tuple[str, float, float]],
    bin_seconds: int,
) -> list[str]:
    if not examples:
        return []
    col_names = [k for k in KIND_ORDER if k in examples]
    tags = [t for t, _, _ in jobs if t in vgals]
    row_names = ["real ESA-ADB"] + tags
    cells: dict[tuple[str, str], np.ndarray] = {}
    active = [k for k in KIND_ORDER if k in examples]
    for col in col_names:
        cells[("real ESA-ADB", col)] = examples[col]["x"][0]
        for tag in tags:
            x = vgals[tag]
            alloc = _kind_alloc_labels(len(x), active)
            sl = x[alloc == col]
            if len(sl):
                cells[(tag, col)] = sl[int(rank_for_kind(sl, col, 1)[0])]
    colors = {"real ESA-ADB": "#b2182b", **{t: _ROW_COLORS.get(t, "#2166ac") for t in tags}}
    png = f"quiet_{enc_name}_aimed.png"
    save_named_grid(
        out / png,
        cells,
        row_names=row_names,
        col_names=col_names,
        bin_seconds=bin_seconds,
        colors=colors,
        title=f"hybrid_needles quiet λ_anom ({enc_name}, aimed)",
        share_col_ylim=True,
    )
    return [png]


def _plot_kind_strips(
    out: Path,
    enc_name: str,
    vgals: dict[str, np.ndarray],
    examples: dict[str, dict[str, np.ndarray]],
    jobs: list[tuple[str, float, float]],
    bin_seconds: int,
    *,
    mode: str,
) -> list[str]:
    written: list[str] = []
    tags = [t for t, _, _ in jobs if t in vgals]
    active = [k for k in KIND_ORDER if k in examples]
    for kind in KIND_ORDER:
        if kind not in examples:
            continue
        rows: dict[str, np.ndarray] = {kind: examples[kind]["x"][:4]}
        colors = {kind: "#b2182b"}
        for tag in tags:
            x = vgals[tag]
            if mode == "aimed":
                alloc = _kind_alloc_labels(len(x), active)
                sl = x[alloc == kind]
                if not len(sl):
                    continue
                idx = rank_for_kind(sl, kind, min(4, len(sl)))
                rows[tag] = sl[idx]
            else:
                rows[tag] = nearest_generated(examples[kind]["x"][:4], x)
            colors[tag] = _ROW_COLORS.get(tag, "#2166ac")
        slug = KIND_SLUG.get(kind, "kind")
        png = f"quiet_{enc_name}_{slug}_{mode}4.png"
        save_compare_rows(
            out / png,
            rows,
            bin_seconds=bin_seconds,
            n_cols=4,
            colors=colors,
            title=f"{kind}: real vs quiet λ ({enc_name}, {mode})",
        )
        written.append(png)
    return written


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
    active = col_names
    for enc_name in ("time_recon", "time_both"):
        pick = recommend.get(enc_name) or {}
        vgals = galleries.get(enc_name) or {}
        tag = pick.get("tag")
        if not tag or tag not in vgals or "q00" not in vgals:
            continue
        x = vgals[tag]
        base = vgals["q00"]
        alloc = _kind_alloc_labels(len(x), active)
        balloc = _kind_alloc_labels(len(base), active)
        rec_row = f"pick {tag}"
        row_names = ["real ESA-ADB", "q00 (band quiet)", rec_row]
        cells: dict[tuple[str, str], np.ndarray] = {}
        for col in col_names:
            cells[("real ESA-ADB", col)] = examples[col]["x"][0]
            sl0 = base[balloc == col]
            sl = x[alloc == col]
            if len(sl0):
                cells[("q00 (band quiet)", col)] = sl0[int(rank_for_kind(sl0, col, 1)[0])]
            if len(sl):
                cells[(rec_row, col)] = sl[int(rank_for_kind(sl, col, 1)[0])]
        png = f"quiet_{enc_name}_recommend_aimed.png"
        save_named_grid(
            out / png,
            cells,
            row_names=row_names,
            col_names=col_names,
            bin_seconds=bin_seconds,
            colors={
                "real ESA-ADB": "#b2182b",
                "q00 (band quiet)": "#4d4d4d",
                rec_row: "#1b9e77",
            },
            title=f"quiettune pick vs band ({enc_name}, aimed)",
            share_col_ylim=True,
        )
        written.append(png)
    return written


def _plot_tv_curves(
    out: Path,
    methods: dict[str, Any],
    real_tv: dict[str, float],
    jobs: list[tuple[str, float, float]],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    shared = [(t, a, b) for t, a, b in jobs if abs(a - b) < 1e-12]
    if not shared:
        shared = list(jobs)
    written: list[str] = []
    for enc in ("time_recon", "time_both"):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
        xs = [a for _, a, _ in shared]
        for ax, kind, title in (
            (axes[0], LOCAL, "local subsequence"),
            (axes[1], GLOBAL, "global subsequence"),
        ):
            tvs = []
            for tag, _, _ in shared:
                m = methods.get(f"{enc}_{tag}") or {}
                tvs.append(float(((m.get("slices") or {}).get(kind) or {}).get("med_tv", float("nan"))))
            ax.plot(xs, tvs, marker="o", color="#2166ac", label="generated median TV")
            tgt = real_tv.get(kind)
            if tgt is not None and np.isfinite(tgt):
                ax.axhline(tgt, color="#b2182b", ls="--", label=f"real median TV ({tgt:.4f})")
            ax.set_xlabel("λ_anom on quiet slices")
            ax.set_ylabel("median TV")
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"Two-sided roughness target ({enc})", fontsize=11)
        fig.tight_layout()
        png = f"quiet_{enc}_tv.png"
        fig.savefig(out / png, dpi=140)
        plt.close(fig)
        written.append(png)
    return written


def _write_metrics_csv(path: Path, methods: dict[str, Any]) -> None:
    rows = []
    for key, m in methods.items():
        sl = m.get("slices") or {}
        rows.append(
            {
                "key": key,
                "encoder": m.get("encoder"),
                "tag": m.get("tag"),
                "lam_local": m.get("lam_local"),
                "lam_global": m.get("lam_global"),
                "occupancy": m.get("occupancy"),
                "arp_anomaly": m.get("arp_anomaly"),
                "diversity": m.get("diversity"),
                "coverage_anomaly": m.get("coverage_anomaly"),
                "frac_shift_slice": (sl.get(SHIFT) or {}).get("frac_shift"),
                "frac_amp_point": (sl.get(POINT) or {}).get("frac_amp"),
                "tv_local": (sl.get(LOCAL) or {}).get("med_tv"),
                "tv_global": (sl.get(GLOBAL) or {}).get("med_tv"),
                "spec_local": (sl.get(LOCAL) or {}).get("med_spec"),
                "spec_global": (sl.get(GLOBAL) or {}).get("med_spec"),
            }
        )
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _index_text(written: list[str], jobs: list[tuple[str, float, float]], recommend: dict[str, Any]) -> str:
    lines = [
        "quiettune: hybrid_needles + small proto λ_anom on local/global subsequence.",
        "q00 = current recipe (band on quiet kinds). Not ESA Locality.",
        "Aimed = best-stat inside the allocated kind slice.",
        "",
        "jobs:",
    ]
    for t, a, b in jobs:
        lines.append(f"  {t}: λ_local={a:g} λ_global={b:g}")
    lines.append("")
    for enc, pick in recommend.items():
        lines.append(
            f"  pick {enc}: {pick.get('tag')} "
            f"λ_local={pick.get('lam_local')} λ_global={pick.get('lam_global')} "
            f"err_tv={pick.get('err_tv')}"
        )
    lines.append("")
    lines.extend(f"  {name}" for name in written)
    lines.append("")
    return "\n".join(lines)
