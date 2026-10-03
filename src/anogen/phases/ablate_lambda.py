"""λ_shell = 0 and no-steer ablations on the locked S4 protocol.

Two sampling variants, parent50 otherwise (ν=0.2, 50 DDIM, unit ∇, λ_rare=0):

- protoonly: λ=0, λ_anom=1 — proto pull only (stratified = every slice;
  hybrid_needles = proto on shift + Point/Global, other slices unguided).
- nosteer: λ=0, λ_anom=0 — guided DDIM with a zero force.

Isolated results/shell_ablate_lambda/. Does not overwrite shell_s4,
kindmixscore, or kindmixhtune.
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
from anogen.phases.encscore import _abs, _brief, _jsonable, _load_encoder
from anogen.phases.kindmix import (
    HYBRID_NEEDLES_PROTO_KINDS,
    _PARENT50,
    _kind_alloc_labels,
)
from anogen.phases.kindmixscore import _edi_recipe_union, _stratified_oof
from anogen.phases.plot_tune import _real_kind_tables
from anogen.shell.diffusion import DiffusionSchedule, denoiser_from_ckpt, torch_available
from anogen.shell.events import types_from_cfg
from anogen.shell.morphology import (
    KIND_ORDER,
    KIND_SLUG,
    nearest_generated,
    rank_for_kind,
    kinds_for_windows,
)
from anogen.shell.plots import save_compare_rows, save_named_grid
from anogen.shell.scaler import resolve_scaler

PROTOONLY = {**_PARENT50, "lam": 0.0, "lam_anom": 1.0, "lam_rare": 0.0}
NOSTEER = {**_PARENT50, "lam": 0.0, "lam_anom": 0.0, "lam_rare": 0.0}

JOBS: tuple[tuple[str, frozenset[str] | None, dict[str, Any]], ...] = (
    ("stratified_protoonly", None, PROTOONLY),
    ("hybrid_needles_protoonly", HYBRID_NEEDLES_PROTO_KINDS, PROTOONLY),
    ("nosteer", frozenset(), NOSTEER),
)

_ROW_COLORS = {
    "real ESA-ADB": "#b2182b",
    "nosteer": "#4d4d4d",
    "stratified_protoonly": "#d95f02",
    "hybrid_needles_protoonly": "#7570b3",
}


def run_ablate_lambda(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s1 = _abs(cfg.get("s1_dir", root / "results/shell_s1"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    s4 = _abs(cfg.get("s4_dir", root / "results/shell_s4"), root)
    enc_dir = _abs(cfg.get("enc_dir", root / "results/shell_enc"), root)
    enc_score = _abs(cfg.get("enc_score_dir", root / "results/shell_enc_score"), root)
    out = _abs(cfg.get("ablate_lambda_dir", root / "results/shell_ablate_lambda"), root)
    plots = out / "plots"
    docs_plots = root / "docs" / "ablate_lambda"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    docs_plots.mkdir(parents=True, exist_ok=True)

    if not torch_available():
        report = {"ok": False, "skipped": True, "reason": "torch is not installed"}
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    needed = {
        "S0": s0 / "labeled_arrays.npz",
        "S1": s1 / "denoiser.pt",
        "S3": s3 / "galleries.npz",
        "S4": s4 / "summary.json",
        "meta": s0 / "labeled_windows.csv",
        "time_recon": enc_dir / "time_recon_fold0.pt",
        "time_both": enc_dir / "time_both_fold0.pt",
    }
    for label, path in needed.items():
        if not path.is_file():
            report = {"ok": False, "skipped": True, "reason": f"{label} missing: {path}"}
            (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            return report

    s4_sum = json.loads((s4 / "summary.json").read_text())
    tau = float(s4_sum["tau"])
    lab = np.load(s0 / "labeled_arrays.npz")
    gal = np.load(s3 / "galleries.npz")
    fold_tab = pd.read_csv(s0 / "channel_fold_counts.csv")
    meta = pd.read_csv(s0 / "labeled_windows.csv")
    x_a, x_r = np.asarray(lab["anomaly"]), np.asarray(lab["rare"])
    fold_a, fold_r = np.asarray(lab["anomaly_fold"]), np.asarray(lab["rare_fold"])
    ch_a, ch_r = np.asarray(lab["anomaly_channel"]), np.asarray(lab["rare_channel"])
    x_cond = np.asarray(gal["cond"])
    cond_ch = np.asarray(gal["channel_idx"])
    anom = meta[meta["kind"] == "anomaly"].reset_index(drop=True)
    if len(anom) != len(x_a):
        report = {
            "ok": False,
            "skipped": True,
            "reason": f"labeled anomaly rows {len(anom)} != arrays {len(x_a)}",
        }
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    types = types_from_cfg(cfg)
    kind_lab = kinds_for_windows(x_a, anom, types)
    examples = _real_kind_tables(x_a, meta, types)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    den = torch.load(s1 / "denoiser.pt", map_location=device, weights_only=False)
    model = denoiser_from_ckpt(den, device=device_t)
    scaler = resolve_scaler(den, s0)
    schedule = DiffusionSchedule.from_ckpt(den).to(device_t)
    scfg = dict((cfg.get("shell") or {}).get("steer") or {})
    bsz = int(scfg.get("n_sample", 128))
    seed = int(cfg.get("seed", 0))
    bin_seconds = int(cfg.get("bin_seconds", 30))
    score_kw = dict(
        x_a=x_a,
        x_r=x_r,
        fold_a=fold_a,
        fold_r=fold_r,
        ch_a=ch_a,
        ch_r=ch_r,
        fold_tab=fold_tab,
        tau=tau,
    )

    methods: dict[str, Any] = {}
    fold0: dict[str, np.ndarray] = {}
    for enc_name in ("time_recon", "time_both"):
        enc, shell = _load_encoder(
            enc_dir / f"{enc_name}_fold0.pt",
            x_cond,
            scfg,
            device,
            seed,
        )
        for recipe, proto_kinds, sample_kw in JOBS:
            print(
                f"ablate_lambda {enc_name} {recipe} λ={sample_kw['lam']} "
                f"λ_anom={sample_kw['lam_anom']} OOF n={len(x_cond)}×3",
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
                **sample_kw,
            )
            scored = _stratified_oof(
                model,
                enc,
                x_cond,
                cond_ch,
                schedule,
                common=common,
                shell=shell,
                device=device,
                kind_lab=kind_lab,
                x_a=x_a,
                fold_a=fold_a,
                score_kw=score_kw,
                out=out,
                variant=enc_name,
                recipe=recipe,
                proto_kinds=proto_kinds,
                seed=seed,
            )
            scored["encoder"] = enc_name
            scored["recipe"] = recipe
            scored["lam"] = float(sample_kw["lam"])
            scored["lam_anom"] = float(sample_kw["lam_anom"])
            scored["shot"] = "FS" if sample_kw["lam_anom"] else "ZS"
            methods[f"{enc_name}_{recipe}"] = scored
            fold0_path = out / f"{enc_name}_{recipe}_fold0.npz"
            if fold0_path.is_file():
                fold0[f"{enc_name}_{recipe}"] = np.asarray(np.load(fold0_path)["x"])

    locked = dict(s4_sum.get("methods") or {})
    for name in ("shell", "genias", "posthoc", "unguided"):
        if name in locked:
            methods[name] = dict(locked[name])

    recipes = tuple(j[0] for j in JOBS)
    edi_table = _edi_recipe_union(
        s4=s4, enc_out=enc_score, recipe_out=out, gal=gal, recipes=recipes
    )
    for name, val in edi_table.items():
        if name in methods:
            methods[name]["edi_table"] = val
            if any(str(name).endswith(f"_{r}") for r in recipes):
                methods[name]["edi"] = val

    written: list[str] = []
    for enc_name in ("time_recon", "time_both"):
        for recipe, _, _ in JOBS:
            key = f"{enc_name}_{recipe}"
            cache = out / f"{enc_name}_{recipe}_fold0.npz"
            if not cache.is_file():
                continue
            blob = np.load(cache)
            x = np.asarray(blob["x"])
            alloc = (
                np.asarray(blob["kind_alloc"], dtype=object)
                if "kind_alloc" in blob.files
                else _kind_alloc_labels(len(x), [k for k in KIND_ORDER if k in examples])
            )
            written += _plot_job(plots, enc_name, recipe, x, alloc, examples, bin_seconds)
        written += _plot_compare(
            plots,
            enc_name,
            {r: fold0[f"{enc_name}_{r}"] for r, _, _ in JOBS if f"{enc_name}_{r}" in fold0},
            examples,
            bin_seconds,
        )

    (plots / "INDEX.txt").write_text(
        "λ_shell=0 and no-steer ablations. Docs: docs/ABLATE_LAMBDA.md.\n"
        "protoonly: λ=0, λ_anom=1 (proto pull only).\n"
        "nosteer: λ=0, λ_anom=0 (zero force, ν=0.2 parent edit).\n"
        "Aimed = best-stat in the allocated kind slice (fold 0, 1536).\n"
        "Nearest = nearest in locked φ from the whole fold-0 gallery.\n\n"
        + "\n".join(f"  {name}" for name in written)
        + "\n"
    )
    for name in written:
        src = plots / name
        if src.is_file():
            shutil.copy2(src, docs_plots / name)
    shutil.copy2(plots / "INDEX.txt", docs_plots / "INDEX.txt")

    report = {
        "ok": True,
        "skipped": False,
        "tau": tau,
        "tau_source": "s4_frozen",
        "n_donors": int(len(x_cond)),
        "jobs": [
            {"recipe": r, "lam": float(kw["lam"]), "lam_anom": float(kw["lam_anom"])}
            for r, _, kw in JOBS
        ],
        "methods": {k: _brief(v) for k, v in methods.items()},
        "edi_table_union": {
            "k": 16,
            "n_per_gallery": 1536,
            "values": edi_table,
            "note": (
                "New k-means union (S4 + encscore + ablation fold-0). "
                "Do not mix this EDI with hybrid *, stratified ‡, or the 9-method table. "
                "ARP is comparable everywhere."
            ),
        },
        "folds": {k: v.get("folds") for k, v in methods.items() if v.get("folds") is not None},
        "plots": written,
        "plots_dir": str(plots),
        "docs_plots": str(docs_plots),
        "kind_scheme": "esa",
        "note": (
            "λ_shell=0 (proto only) and no-steer (λ=λ_anom=0) on frozen S4 τ / "
            "feature_pack_v1. 1536 S3 donors, 3-fold OOF. parent50 otherwise. "
            "Does not overwrite results/shell_s4 or kindmixscore caches."
        ),
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return report


def _plot_job(
    out: Path,
    enc_name: str,
    recipe: str,
    x: np.ndarray,
    alloc: np.ndarray,
    examples: dict[str, dict[str, np.ndarray]],
    bin_seconds: int,
) -> list[str]:
    if not examples:
        return []
    alloc = np.asarray(alloc)
    written: list[str] = []
    col_names = [k for k in KIND_ORDER if k in examples and np.any(alloc == k)]
    aimed: dict[tuple[str, str], np.ndarray] = {}
    nearest: dict[tuple[str, str], np.ndarray] = {}
    for kind in col_names:
        gen = x[alloc == kind]
        real = examples[kind]["x"]
        slug = KIND_SLUG.get(kind, "kind")
        aimed_png = f"ablate_{enc_name}_{recipe}_{slug}_aimed.png"
        save_compare_rows(
            out / aimed_png,
            {
                kind: real[:4],
                "aimed slice": gen[rank_for_kind(gen, kind, 4)] if len(gen) else real[:0],
            },
            bin_seconds=bin_seconds,
            n_cols=4,
            colors={kind: "#b2182b", "aimed slice": _ROW_COLORS.get(recipe, "#d95f02")},
            title=f"{enc_name} {recipe}: real {kind} vs aimed slice",
        )
        written.append(aimed_png)
        near_png = f"ablate_{enc_name}_{recipe}_{slug}_nearest.png"
        save_compare_rows(
            out / near_png,
            {kind: real[:4], "nearest in φ": nearest_generated(real[:4], x)},
            bin_seconds=bin_seconds,
            n_cols=4,
            colors={kind: "#b2182b", "nearest in φ": _ROW_COLORS.get(recipe, "#d95f02")},
            title=f"{enc_name} {recipe}: real {kind} vs nearest in φ",
        )
        written.append(near_png)
        aimed[("real ESA-ADB", kind)] = real[0]
        nearest[("real ESA-ADB", kind)] = real[0]
        if len(gen):
            aimed[("aimed slice", kind)] = gen[int(rank_for_kind(gen, kind, 1)[0])]
        match = nearest_generated(real[:1], x)
        if len(match):
            nearest[("nearest in φ", kind)] = match[0]
    if col_names:
        for mode, cells, row in (
            ("aimed", aimed, "aimed slice"),
            ("nearest", nearest, "nearest in φ"),
        ):
            png = f"ablate_{enc_name}_{recipe}_{mode}.png"
            save_named_grid(
                out / png,
                cells,
                row_names=["real ESA-ADB", row],
                col_names=col_names,
                bin_seconds=bin_seconds,
                colors={"real ESA-ADB": "#b2182b", row: _ROW_COLORS.get(recipe, "#d95f02")},
                title=f"{enc_name} {recipe} ({mode})",
                share_col_ylim=True,
            )
            written.append(png)
    return written


def _plot_compare(
    out: Path,
    enc_name: str,
    vgals: dict[str, np.ndarray],
    examples: dict[str, dict[str, np.ndarray]],
    bin_seconds: int,
) -> list[str]:
    if not examples or not vgals:
        return []
    col_names = [k for k in KIND_ORDER if k in examples]
    row_names = ["real ESA-ADB"] + [r for r, _, _ in JOBS if r in vgals]
    cells: dict[tuple[str, str], np.ndarray] = {}
    for col in col_names:
        cells[("real ESA-ADB", col)] = examples[col]["x"][0]
        for recipe in row_names[1:]:
            x = vgals[recipe]
            active = [k for k in KIND_ORDER if k in examples]
            alloc = _kind_alloc_labels(len(x), active)
            sl = x[alloc == col]
            if len(sl):
                cells[(recipe, col)] = sl[int(rank_for_kind(sl, col, 1)[0])]
    png = f"ablate_{enc_name}_compare_aimed.png"
    save_named_grid(
        out / png,
        cells,
        row_names=row_names,
        col_names=col_names,
        bin_seconds=bin_seconds,
        colors=_ROW_COLORS,
        title=f"No-steer vs proto-only ({enc_name}, aimed)",
        share_col_ylim=True,
    )
    return [png]
