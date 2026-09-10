"""Score time_recon / time_both + band or combined f on the locked S4 protocol.

Same τ, φ, 1536 S3 donors as S4. Does not overwrite results/shell_s4.
Combined uses event-OOF anomaly/rare refs (few-shot). Band is zero-shot.
Fold-0 encoder weights only (later-fold ckpts were not saved).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from anogen.config import REPO_ROOT
from anogen.shell.coverage import edi_by_method, mean_pairwise_distance
from anogen.shell.diffusion import DiffusionSchedule, denoiser_from_ckpt, torch_available
from anogen.shell.encoders import ShellEncoder, embed_shell
from anogen.shell.features import embed_windows
from anogen.shell.scaler import resolve_scaler
from anogen.shell.steer import band_report, chunked_guided_ddim, shell_from_nominal
from anogen.phases.s4 import _score_methods
from anogen.phases.tune import _score_gallery

_BAND = {
    "nu": 0.2,
    "lam": 0.3,
    "ddim_steps": 20,
    "normalize_grad": False,
    "n_correct": 1,
    "lam_anom": 0.0,
    "lam_rare": 0.0,
}
_COMBINED = {
    "nu": 0.2,
    "lam": 0.3,
    "ddim_steps": 50,
    "normalize_grad": True,
    "n_correct": 1,
    "lam_anom": 1.0,
    "lam_rare": 1.0,
}


def run_encscore(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s1 = _abs(cfg.get("s1_dir", root / "results/shell_s1"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    s4 = _abs(cfg.get("s4_dir", root / "results/shell_s4"), root)
    enc_dir = _abs(cfg.get("enc_dir", root / "results/shell_enc"), root)
    out = _abs(cfg.get("enc_score_dir", root / "results/shell_enc_score"), root)
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
    x_a, x_r = np.asarray(lab["anomaly"]), np.asarray(lab["rare"])
    fold_a, fold_r = np.asarray(lab["anomaly_fold"]), np.asarray(lab["rare_fold"])
    ch_a, ch_r = np.asarray(lab["anomaly_channel"]), np.asarray(lab["rare_channel"])
    x_cond = np.asarray(gal["cond"])
    cond_ch = np.asarray(gal["channel_idx"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    den = torch.load(s1 / "denoiser.pt", map_location=device, weights_only=False)
    model = denoiser_from_ckpt(den, device=device_t)
    scaler = resolve_scaler(den, s0)
    schedule = DiffusionSchedule.linear(int(den["n_times"])).to(device_t)
    scfg = dict((cfg.get("shell") or {}).get("steer") or {})
    tcfg = dict((cfg.get("shell") or {}).get("tune") or {})
    bsz = int(scfg.get("n_sample", 128))
    n_ref = int(tcfg.get("n_ref_label", 256))
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
    for variant in ("time_recon", "time_both"):
        enc, shell = _load_encoder(
            enc_dir / f"{variant}_fold0.pt",
            x_cond,
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
        )
        print(f"encscore {variant} band n={len(x_cond)}", flush=True)
        band_x, band_h = chunked_guided_ddim(
            model, enc, x_cond, cond_ch, schedule, **_BAND, **common
        )
        np.savez_compressed(
            out / f"{variant}_band.npz",
            x=band_x,
            h=band_h,
            channel_idx=cond_ch,
            Q_q=float(shell["Q_q"]),
            delta=float(shell["delta"]),
        )
        band_scored = _score_methods({"g": band_x}, **score_kw)["g"]
        band_scored["occupancy"] = float(np.mean(np.abs(band_h - shell["Q_q"]) <= shell["delta"]))
        band_scored["energy"] = band_report(band_h, float(shell["Q_q"]), float(shell["delta"]))
        band_scored["Q_q"] = float(shell["Q_q"])
        band_scored["encoder"] = variant
        band_scored["recipe"] = "band"
        band_scored["n"] = int(len(band_x))
        band_scored["shot"] = "ZS"
        methods[f"{variant}_band"] = band_scored

        print(f"encscore {variant} combined OOF n={len(x_cond)}×3", flush=True)
        combined = _combined_oof(
            model,
            enc,
            x_cond,
            cond_ch,
            schedule,
            common=common,
            shell=shell,
            n_ref=n_ref,
            device=device,
            x_a=x_a,
            x_r=x_r,
            fold_a=fold_a,
            fold_r=fold_r,
            score_kw=score_kw,
            out=out,
            variant=variant,
        )
        combined["encoder"] = variant
        combined["recipe"] = "combined"
        combined["shot"] = "FS"
        methods[f"{variant}_combined"] = combined

    locked = dict(s4_sum.get("methods") or {})
    for name in ("shell", "genias", "posthoc", "unguided"):
        if name in locked:
            methods[name] = dict(locked[name])

    edi_table = _edi_table_union(s4=s4, enc_out=out, gal=gal)
    for name, val in edi_table.items():
        if name in methods:
            methods[name]["edi_table"] = val
            if str(name).endswith("_combined"):
                methods[name]["edi"] = val

    report = {
        "ok": True,
        "skipped": False,
        "tau": tau,
        "tau_source": "s4_frozen",
        "n_donors": int(len(x_cond)),
        "methods": {k: _brief(v) for k, v in methods.items()},
        "edi_table_union": {
            "k": 16,
            "n_per_gallery": 1536,
            "values": edi_table,
            "note": (
                "GenIAS App. E.2 EDI on the 9 galleries in PAPER_CANDIDATES §2. "
                "Combined uses fold 0 (1536), not the 3-fold stack. "
                "Do not mix with locked S4 5-method edi in results/shell_s4."
            ),
        },
        "folds": {k: v.get("folds") for k, v in methods.items() if v.get("folds") is not None},
        "note": (
            "Newer encoders scored with frozen S4 τ and feature_pack_v1. "
            "1536 S3 donors. Combined f uses event-OOF contrast refs (few-shot). "
            "Band is original S4 f (λ=0.3, ν=0.2, 20 steps). "
            "Fold-0 encoder only. Does not overwrite results/shell_s4. "
            "time_both encoder saw fold-1/2 anomalies at train time."
        ),
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return report


def _combined_oof(
    model: Any,
    enc: Any,
    x_cond: np.ndarray,
    cond_ch: np.ndarray,
    schedule: Any,
    *,
    common: dict[str, Any],
    shell: dict[str, Any],
    n_ref: int,
    device: str,
    x_a: np.ndarray,
    x_r: np.ndarray,
    fold_a: np.ndarray,
    fold_r: np.ndarray,
    score_kw: dict[str, Any],
    out: Path,
    variant: str,
    sample_kw: dict[str, Any] | None = None,
    tag: str = "combined",
) -> dict[str, Any]:
    fold_rows = []
    chunks = []
    q_q, delta = float(shell["Q_q"]), float(shell["delta"])
    ddim = {**_COMBINED, **(sample_kw or {})}
    for fold_id in sorted({int(f) for f in fold_a.tolist() if int(f) >= 0}):
        cache = out / f"{variant}_{tag}_fold{fold_id}.npz"
        if cache.is_file():
            blob = np.load(cache)
            samples = np.asarray(blob["x"])
            h = np.asarray(blob["h"])
        else:
            ra = torch.from_numpy(embed_shell(enc, x_a[fold_a != fold_id][:n_ref], device))
            rr = torch.from_numpy(embed_shell(enc, x_r[fold_r != fold_id][:n_ref], device))
            samples, h = chunked_guided_ddim(
                model,
                enc,
                x_cond,
                cond_ch,
                schedule,
                ref_anom=ra,
                ref_rare=rr,
                **ddim,
                **{k: v for k, v in common.items() if k != "bsz"},
                bsz=common["bsz"],
            )
            np.savez_compressed(cache, x=samples, h=h)
        chunks.append(samples)
        scored = _score_gallery(samples, only_fold=fold_id, **score_kw)
        scored["fold"] = fold_id
        scored["occupancy"] = float(np.mean(np.abs(h - q_q) <= delta))
        fold_rows.append(scored)
    gallery = np.concatenate(chunks, axis=0)
    np.savez_compressed(out / f"{variant}_{tag}.npz", x=gallery)
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
        "folds": fold_rows,
        "n": int(len(chunks[0])),
        "Q_q": q_q,
    }


def _edi_table_union(
    *,
    s4: Path,
    enc_out: Path,
    gal: Any,
) -> dict[str, float]:
    """GenIAS EDI on equal-sized galleries (combined = fold 0 only)."""
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
    return edi_by_method(embs)


def _load_encoder(
    ckpt_path: Path,
    parent: np.ndarray,
    scfg: dict[str, Any],
    device: str,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
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
    return enc, shell


def _brief(row: dict[str, Any]) -> dict[str, Any]:
    skip = {"folds", "energy"}
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
