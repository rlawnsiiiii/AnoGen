"""Score time_recon / time_both × stratified proto on the locked S4 protocol.

Same τ, φ, 1536 S3 donors, 3-fold OOF as encscore. Isolated
results/shell_kindmix_score_esa/. Does not overwrite shell_s4, shell_enc_score,
or the morphology freeze in results/shell_kindmix_score/.

Recipe is kindmix parent50 (λ=0.3, λ_anom=1, λ_rare=0, 50 DDIM, unit ∇)
with ESA Length × Locality plus the level-shift overlay (docs/ESA_LABELS.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from anogen.config import REPO_ROOT
from anogen.phases.encscore import _abs, _brief, _jsonable, _load_encoder
from anogen.phases.kindmix import (
    _PARENT50,
    _coverage_by_kind,
    _coverage_slices,
    _kind_alloc_labels,
    _stratified,
)
from anogen.phases.tune import _score_gallery
from anogen.shell.coverage import edi_by_method, mean_pairwise_distance
from anogen.shell.diffusion import DiffusionSchedule, denoiser_from_ckpt, torch_available
from anogen.shell.encoders import embed_shell
from anogen.shell.events import types_from_cfg
from anogen.shell.features import embed_windows
from anogen.shell.morphology import KIND_ORDER, KIND_SLUG, kind_ref_indices, kinds_for_windows
from anogen.shell.scaler import resolve_scaler
from anogen.shell.steer import band_report


def run_kindmixscore(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s1 = _abs(cfg.get("s1_dir", root / "results/shell_s1"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    s4 = _abs(cfg.get("s4_dir", root / "results/shell_s4"), root)
    enc_dir = _abs(cfg.get("enc_dir", root / "results/shell_enc"), root)
    enc_score = _abs(cfg.get("enc_score_dir", root / "results/shell_enc_score"), root)
    out = _abs(cfg.get("kindmix_score_dir", root / "results/shell_kindmix_score_esa"), root)
    out.mkdir(parents=True, exist_ok=True)

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    den = torch.load(s1 / "denoiser.pt", map_location=device, weights_only=False)
    model = denoiser_from_ckpt(den, device=device_t)
    scaler = resolve_scaler(den, s0)
    schedule = DiffusionSchedule.linear(int(den["n_times"])).to(device_t)
    scfg = dict((cfg.get("shell") or {}).get("steer") or {})
    bsz = int(scfg.get("n_sample", 128))
    seed = int(cfg.get("seed", 0))
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
    ref_info: dict[str, Any] = {}
    for variant in ("time_recon", "time_both"):
        enc, shell = _load_encoder(
            enc_dir / f"{variant}_fold0.pt",
            x_cond,
            scfg,
            device,
            seed,
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
        print(f"kindmixscore {variant} stratified OOF n={len(x_cond)}×3", flush=True)
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
            variant=variant,
            seed=seed,
        )
        scored["encoder"] = variant
        scored["recipe"] = "stratified"
        scored["shot"] = "FS"
        methods[f"{variant}_stratified"] = scored
        ref_info[variant] = scored.get("refs", {})

    locked = dict(s4_sum.get("methods") or {})
    for name in ("shell", "genias", "posthoc", "unguided"):
        if name in locked:
            methods[name] = dict(locked[name])

    edi_table = _edi_strat_union(s4=s4, enc_out=enc_score, strat_out=out, gal=gal)
    for name, val in edi_table.items():
        if name in methods:
            methods[name]["edi_table"] = val
            if str(name).endswith("_stratified") or str(name).endswith("_combined"):
                methods[name]["edi"] = val

    report = {
        "ok": True,
        "skipped": False,
        "tau": tau,
        "tau_source": "s4_frozen",
        "n_donors": int(len(x_cond)),
        "methods": {k: _brief(v) for k, v in methods.items()},
        "refs": ref_info,
        "edi_table_union": {
            "k": 16,
            "n_per_gallery": 1536,
            "values": edi_table,
            "note": (
                "GenIAS App. E.2 EDI on S4 + encscore + stratified fold-0 galleries. "
                "New partition — do not mix with locked S4 5-method, encscore 9-method, "
                "or S5-extended EDI."
            ),
        },
        "folds": {k: v.get("folds") for k, v in methods.items() if v.get("folds") is not None},
        "kind_scheme": "esa",
        "note": (
            "Stratified proto scored with frozen S4 τ and feature_pack_v1. "
            "1536 S3 donors, 3-fold OOF. parent50: λ=0.3, λ_anom=1, λ_rare=0, "
            "50 DDIM, unit ∇. ESA Length × Locality + level-shift overlay. "
            "Fold-0 encoder only. Does not overwrite results/shell_s4 "
            "or results/shell_kindmix_score/ (morphology freeze)."
        ),
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return report


def _stratified_oof(
    model: Any,
    enc: Any,
    x_cond: np.ndarray,
    cond_ch: np.ndarray,
    schedule: Any,
    *,
    common: dict[str, Any],
    shell: dict[str, Any],
    device: str,
    kind_lab: np.ndarray,
    x_a: np.ndarray,
    fold_a: np.ndarray,
    score_kw: dict[str, Any],
    out: Path,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    fold_rows = []
    chunks = []
    q_q, delta = float(shell["Q_q"]), float(shell["delta"])
    ch_a = score_kw["ch_a"]
    fold_tab = score_kw["fold_tab"]
    tau = float(score_kw["tau"])
    refs_by_fold: dict[str, Any] = {}
    for fold_id in sorted({int(f) for f in fold_a.tolist() if int(f) >= 0}):
        kind_z: dict[str, torch.Tensor] = {}
        fold_refs: dict[str, Any] = {}
        for kind in KIND_ORDER:
            idx, leaked = kind_ref_indices(kind_lab, fold_a, kind, query_fold=fold_id)
            fold_refs[kind] = {
                "n_all": int((kind_lab == kind).sum()),
                "n_ref": int(len(idx)),
                "n_oof": int(((kind_lab == kind) & (fold_a != fold_id)).sum()),
                "leaked": leaked,
            }
            if len(idx) == 0:
                continue
            kind_z[kind] = torch.from_numpy(embed_shell(enc, x_a[idx], device))
        refs_by_fold[str(fold_id)] = fold_refs
        active = [k for k in KIND_ORDER if k in kind_z]
        cache = out / f"{variant}_stratified_fold{fold_id}.npz"
        if cache.is_file():
            blob = np.load(cache)
            samples = np.asarray(blob["x"])
            h = np.asarray(blob["h"])
            alloc = (
                np.asarray(blob["kind_alloc"], dtype=object)
                if "kind_alloc" in blob.files
                else _kind_alloc_labels(len(samples), active)
            )
        else:
            samples, h = _stratified(
                model,
                enc,
                x_cond,
                cond_ch,
                schedule,
                common,
                kind_z,
                seed + 1000 * fold_id,
                kind_order=KIND_ORDER,
                kind_slug=KIND_SLUG,
            )
            alloc = _kind_alloc_labels(len(samples), active)
            np.savez_compressed(
                cache,
                x=samples,
                h=h,
                channel_idx=cond_ch,
                kind_alloc=alloc.astype(str),
            )
        chunks.append(samples)
        scored = _score_gallery(samples, only_fold=fold_id, **score_kw)
        scored["fold"] = fold_id
        scored["occupancy"] = float(np.mean(np.abs(h - q_q) <= delta))
        scored["energy"] = band_report(h, q_q, delta)
        scored["coverage_kind"] = _coverage_by_kind(
            samples, x_a, fold_a, ch_a, kind_lab, fold_tab, tau, fold_id=fold_id
        )
        scored["slice_kind"] = _coverage_slices(
            samples, alloc, x_a, fold_a, ch_a, kind_lab, fold_tab, tau, fold_id=fold_id
        )
        fold_rows.append(scored)
    gallery = np.concatenate(chunks, axis=0)
    np.savez_compressed(out / f"{variant}_stratified.npz", x=gallery)
    return {
        "coverage_anomaly": float(np.mean([r["coverage_anomaly"] for r in fold_rows])),
        "coverage_rare": float(np.mean([r["coverage_rare"] for r in fold_rows])),
        "gap": float(np.mean([r["gap"] for r in fold_rows])),
        "arp_anomaly": float(np.mean([r["arp_anomaly"] for r in fold_rows])),
        "arp_rare": float(np.mean([r["arp_rare"] for r in fold_rows])),
        "mean_min_d_anomaly": float(np.mean([r["mean_min_d_anomaly"] for r in fold_rows])),
        "mean_min_d_rare": float(np.mean([r["mean_min_d_rare"] for r in fold_rows])),
        "occupancy": float(np.mean([r["occupancy"] for r in fold_rows])),
        "diversity": mean_pairwise_distance(embed_windows(chunks[0])),
        "coverage_kind": _mean_kind_rows([r.get("coverage_kind") or {} for r in fold_rows]),
        "slice_kind": _mean_kind_rows([r.get("slice_kind") or {} for r in fold_rows]),
        "folds": fold_rows,
        "refs": refs_by_fold,
        "n": int(len(chunks[0])),
        "Q_q": q_q,
        "kind_scheme": "esa",
    }


def _mean_kind_rows(rows: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    keys = {k for row in rows for k in row}
    out: dict[str, dict[str, float]] = {}
    for kind in keys:
        metrics = [row[kind] for row in rows if kind in row]
        if not metrics:
            continue
        fields = {f for m in metrics for f in m}
        out[kind] = {
            f: float(np.nanmean([m.get(f, float("nan")) for m in metrics])) for f in fields
        }
    return out


def _edi_strat_union(
    *,
    s4: Path,
    enc_out: Path,
    strat_out: Path,
    gal: Any,
) -> dict[str, float]:
    s4g = np.load(s4 / "shell_gallery.npz")
    embs: dict[str, np.ndarray] = {
        "shell": embed_windows(np.asarray(s4g["x"])),
        "unguided": embed_windows(np.asarray(s4g["unguided"])),
        "genias": embed_windows(np.asarray(gal["genias"])),
        "posthoc": embed_windows(np.asarray(gal["posthoc"])),
    }
    if "genias_patched" in getattr(gal, "files", ()):
        embs["genias_patched"] = embed_windows(np.asarray(gal["genias_patched"]))
    for variant in ("time_recon", "time_both"):
        band = enc_out / f"{variant}_band.npz"
        comb = enc_out / f"{variant}_combined_fold0.npz"
        if band.is_file():
            embs[f"{variant}_band"] = embed_windows(np.asarray(np.load(band)["x"]))
        if comb.is_file():
            embs[f"{variant}_combined"] = embed_windows(np.asarray(np.load(comb)["x"]))
        strat = strat_out / f"{variant}_stratified_fold0.npz"
        if strat.is_file():
            embs[f"{variant}_stratified"] = embed_windows(np.asarray(np.load(strat)["x"]))
    return edi_by_method(embs)
