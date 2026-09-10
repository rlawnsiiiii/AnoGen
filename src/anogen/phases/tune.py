"""OOF sampling-HP search. Does not overwrite the locked S4 table.

Two families:

* ``zs`` — more DDIM steps, stronger shell λ, extra corrections, optional push.
  Tests whether the sampler can land in |h − Q_q| ≤ δ.
* ``contrast`` — attract toward train-fold anomaly embeddings and repel
  train-fold Rare Events. That is the term that can tell faults from rares;
  the shell band cannot (S4 energy: rares sit on Q_q).

τ is the frozen S4 value. Selection is declared before the grid: occupancy
among ZS, ARP-anomaly among all finite, gap among all finite. Sealed test
stays closed.
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
from anogen.shell.coverage import score_generator
from anogen.shell.diffusion import DiffusionSchedule, denoiser_from_ckpt, torch_available
from anogen.shell.plots import save_bar, save_overlay
from anogen.shell.scaler import resolve_scaler
from anogen.shell.steer import ConvEncoder, band_report, chunked_guided_ddim, embed_encoder


def default_grids() -> list[dict[str, Any]]:
    """Curated OOF grid. Not a full Cartesian product."""
    zs: list[dict[str, Any]] = []
    for steps, lam, nu, n_correct, norm, mode in (
        (20, 0.3, 0.2, 1, False, "band"),  # locked S4
        (50, 0.3, 0.2, 1, False, "band"),
        (100, 0.3, 0.2, 1, False, "band"),
        (50, 0.3, 0.2, 4, False, "band"),
        (50, 1.5, 0.2, 1, True, "band"),
        (50, 4.0, 0.2, 1, True, "band"),
        (100, 1.5, 0.2, 1, True, "band"),
        (50, 1.5, 0.2, 4, True, "band"),
        (50, 1.5, 0.5, 1, True, "band"),
        (50, 1.5, 0.2, 1, True, "push"),
    ):
        zs.append(
            {
                "family": "zs",
                "ddim_steps": steps,
                "lam": lam,
                "nu": nu,
                "n_correct": n_correct,
                "normalize_grad": norm,
                "mode": mode,
                "lam_anom": 0.0,
                "lam_rare": 0.0,
            }
        )
    contrast: list[dict[str, Any]] = []
    for steps, nu, lam, lam_a, lam_r in (
        (50, 0.2, 0.0, 1.0, 1.0),
        (50, 0.2, 0.0, 3.0, 1.0),
        (50, 0.2, 0.0, 1.0, 3.0),
        (50, 0.2, 0.0, 3.0, 3.0),
        (50, 0.2, 0.3, 1.0, 1.0),
        (50, 0.5, 0.0, 3.0, 3.0),
        (80, 0.2, 0.0, 3.0, 3.0),
    ):
        contrast.append(
            {
                "family": "contrast",
                "ddim_steps": steps,
                "lam": lam,
                "nu": nu,
                "n_correct": 1,
                "normalize_grad": True,
                "mode": "band",
                "lam_anom": lam_a,
                "lam_rare": lam_r,
            }
        )
    return zs + contrast


def pick_winners(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    finite = [r for r in rows if r.get("finite") and not r.get("explode")]
    zs = [r for r in finite if r.get("family") == "zs"]
    return {
        "best_occupancy": max(zs, key=lambda r: r["occupancy"]) if zs else None,
        "best_arp_anomaly": max(finite, key=lambda r: r["arp_anomaly"]) if finite else None,
        "best_gap": max(finite, key=lambda r: r["gap"]) if finite else None,
    }


def run_tune(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s1 = _abs(cfg.get("s1_dir", root / "results/shell_s1"), root)
    s2 = _abs(cfg.get("s2_dir", root / "results/shell_s2"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    s4 = _abs(cfg.get("s4_dir", root / "results/shell_s4"), root)
    out = _abs(cfg.get("tune_dir", root / "results/shell_tune"), root)
    out.mkdir(parents=True, exist_ok=True)

    if not torch_available():
        report = {"ok": False, "skipped": True, "reason": "torch is not installed"}
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    needed = {
        "S0": s0 / "labeled_arrays.npz",
        "S1": s1 / "denoiser.pt",
        "S2": s2 / "encoder.pt",
        "S3": s3 / "galleries.npz",
        "S4": s4 / "summary.json",
    }
    for label, path in needed.items():
        if not path.is_file():
            report = {"ok": False, "skipped": True, "reason": f"{label} missing: {path}"}
            (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            return report

    tcfg = dict((cfg.get("shell") or {}).get("tune") or {})
    scfg = dict((cfg.get("shell") or {}).get("steer") or {})
    s4_sum = json.loads((s4 / "summary.json").read_text())
    tau = float(s4_sum["tau"])
    lab = np.load(s0 / "labeled_arrays.npz")
    gal = np.load(s3 / "galleries.npz")
    x_a, x_r = np.asarray(lab["anomaly"]), np.asarray(lab["rare"])
    fold_a, fold_r = np.asarray(lab["anomaly_fold"]), np.asarray(lab["rare_fold"])
    ch_a, ch_r = np.asarray(lab["anomaly_channel"]), np.asarray(lab["rare_channel"])
    fold_tab = pd.read_csv(s0 / "channel_fold_counts.csv")
    x_cond = np.asarray(gal["cond"])
    cond_ch = np.asarray(gal["channel_idx"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    den = torch.load(s1 / "denoiser.pt", map_location=device, weights_only=False)
    enc_blob = torch.load(s2 / "encoder.pt", map_location=device, weights_only=False)
    model = denoiser_from_ckpt(den, device=device_t)
    scaler = resolve_scaler(den, s0)
    encoder = ConvEncoder(
        int(enc_blob["width"]), hidden=int(enc_blob["hidden"]), emb=int(enc_blob["emb"])
    )
    encoder.load_state_dict(enc_blob["state_dict"])
    encoder.to(device_t)
    encoder.eval()
    schedule = DiffusionSchedule.linear(int(den["n_times"])).to(device_t)
    q_q = float(enc_blob["Q_q"])
    delta = float(enc_blob["delta"])
    tau_e = float(enc_blob["tau"])
    ref = torch.from_numpy(embed_encoder(encoder, x_cond[: min(512, len(x_cond))], device))

    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    n_cond = min(int(tcfg.get("n_cond", 192)), len(x_cond))
    pick = rng.choice(len(x_cond), size=n_cond, replace=False)
    x_s, ch_s = x_cond[pick], cond_ch[pick]
    n_ref = int(tcfg.get("n_ref_label", 256))
    bsz = int(tcfg.get("bsz", scfg.get("n_sample", 64)))
    c_max = float(scfg.get("c_max", 1.0))
    search_fold = 0
    ref_anom_s, ref_rare_s = _oof_refs(
        encoder, x_a, x_r, fold_a, fold_r, search_fold, n_ref, device
    )

    grids = list(tcfg.get("grid") or default_grids())
    rows: list[dict[str, Any]] = []
    for i, spec in enumerate(grids):
        row = _run_one(
            spec,
            model=model,
            encoder=encoder,
            schedule=schedule,
            scaler=scaler,
            x=x_s,
            ch=ch_s,
            ref=ref,
            Q_q=q_q,
            delta=delta,
            tau_e=tau_e,
            tau=tau,
            c_max=c_max,
            bsz=bsz,
            device=device,
            x_a=x_a,
            x_r=x_r,
            fold_a=fold_a,
            fold_r=fold_r,
            ch_a=ch_a,
            ch_r=ch_r,
            fold_tab=fold_tab,
            ref_anom=ref_anom_s,
            ref_rare=ref_rare_s,
            score_fold=search_fold,
        )
        row["id"] = i
        rows.append(row)
        print(
            f"tune {i + 1}/{len(grids)} {row['family']} steps={row['ddim_steps']} "
            f"λ={row['lam']} ν={row['nu']} occ={row.get('occupancy')} "
            f"ARP_a={row.get('arp_anomaly')} gap={row.get('gap')}",
            flush=True,
        )

    winners = pick_winners(rows)
    regen_n = min(int(tcfg.get("regen_n", 512)), len(x_cond))
    regen_idx = rng.choice(len(x_cond), size=regen_n, replace=False)
    regen: dict[str, Any] = {}
    for key, win in winners.items():
        if win is None:
            continue
        regen[key] = _regen(
            win,
            model=model,
            encoder=encoder,
            schedule=schedule,
            scaler=scaler,
            x_cond=x_cond[regen_idx],
            cond_ch=cond_ch[regen_idx],
            ref=ref,
            Q_q=q_q,
            delta=delta,
            tau_e=tau_e,
            tau=tau,
            c_max=c_max,
            bsz=bsz,
            device=device,
            x_a=x_a,
            x_r=x_r,
            fold_a=fold_a,
            fold_r=fold_r,
            ch_a=ch_a,
            ch_r=ch_r,
            fold_tab=fold_tab,
            n_ref=n_ref,
            out=out,
            name=key,
        )

    _write_csv(out / "sweep.csv", rows)
    _write_plots(out, rows, x_cond[regen_idx], x_a, int(cfg.get("bin_seconds", 30)))

    report = {
        "ok": True,
        "skipped": False,
        "tau": tau,
        "tau_source": "s4_frozen",
        "Q_q": q_q,
        "delta": delta,
        "n_cond_search": n_cond,
        "regen_n": regen_n,
        "search_fold_for_contrast": search_fold,
        "n_grid": len(rows),
        "rows": rows,
        "winners": {k: _winner_brief(v) for k, v in winners.items()},
        "regen": regen,
        "s4_baseline": {
            "occupancy": s4_sum.get("occupancy"),
            "shell": s4_sum.get("methods", {}).get("shell"),
        },
        "note": (
            "OOF sampling search after the locked S4 table. Does not retune "
            "S2 Q_q / δ or overwrite results/shell_s4. Contrast refs are "
            "train-fold only. Sealed test stays closed. Search coverage uses "
            f"{n_cond} donors; regen uses {regen_n}. Do not compare search "
            "Coverage@τ to the S4 1536-donor table."
        ),
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return report


def _run_one(
    spec: dict[str, Any],
    *,
    model: Any,
    encoder: Any,
    schedule: Any,
    scaler: Any,
    x: np.ndarray,
    ch: np.ndarray,
    ref: Any,
    Q_q: float,
    delta: float,
    tau_e: float,
    tau: float,
    c_max: float,
    bsz: int,
    device: str,
    x_a: np.ndarray,
    x_r: np.ndarray,
    fold_a: np.ndarray,
    fold_r: np.ndarray,
    ch_a: np.ndarray,
    ch_r: np.ndarray,
    fold_tab: pd.DataFrame,
    ref_anom: Any,
    ref_rare: Any,
    score_fold: int | None,
) -> dict[str, Any]:
    family = str(spec.get("family", "zs"))
    kwargs = _sample_kwargs(spec, ref, Q_q, tau_e, c_max, device, scaler)
    if family == "contrast":
        kwargs["ref_anom"] = ref_anom
        kwargs["ref_rare"] = ref_rare
        kwargs["lam_anom"] = float(spec["lam_anom"])
        kwargs["lam_rare"] = float(spec["lam_rare"])
    samples, h = chunked_guided_ddim(model, encoder, x, ch, schedule, bsz=bsz, **kwargs)
    finite = bool(np.isfinite(samples).all()) and bool(np.isfinite(h).all())
    explode = bool((not finite) or np.nanmax(np.abs(samples)) > 50.0)
    occupancy = float(np.mean(np.abs(h - Q_q) <= delta)) if finite else 0.0
    row: dict[str, Any] = {
        **{k: spec[k] for k in spec},
        "finite": finite,
        "explode": explode,
        "occupancy": occupancy,
        "h_mean": float(h.mean()) if finite else float("nan"),
        "h_std": float(h.std()) if finite else float("nan"),
        "energy": band_report(h, Q_q, delta) if finite else {},
    }
    if not finite or explode:
        row.update(
            {
                "coverage_anomaly": 0.0,
                "coverage_rare": 0.0,
                "gap": 0.0,
                "arp_anomaly": 0.0,
                "arp_rare": 0.0,
                "mean_min_d_anomaly": float("inf"),
                "mean_min_d_rare": float("inf"),
            }
        )
        return row
    scored = _score_gallery(
        samples,
        x_a=x_a,
        x_r=x_r,
        fold_a=fold_a,
        fold_r=fold_r,
        ch_a=ch_a,
        ch_r=ch_r,
        fold_tab=fold_tab,
        tau=tau,
        only_fold=score_fold,
    )
    row.update(scored)
    return row


def _regen(
    spec: dict[str, Any],
    *,
    model: Any,
    encoder: Any,
    schedule: Any,
    scaler: Any,
    x_cond: np.ndarray,
    cond_ch: np.ndarray,
    ref: Any,
    Q_q: float,
    delta: float,
    tau_e: float,
    tau: float,
    c_max: float,
    bsz: int,
    device: str,
    x_a: np.ndarray,
    x_r: np.ndarray,
    fold_a: np.ndarray,
    fold_r: np.ndarray,
    ch_a: np.ndarray,
    ch_r: np.ndarray,
    fold_tab: pd.DataFrame,
    n_ref: int,
    out: Path,
    name: str,
) -> dict[str, Any]:
    kwargs = _sample_kwargs(spec, ref, Q_q, tau_e, c_max, device, scaler)
    family = str(spec.get("family", "zs"))
    if family == "contrast":
        fold_rows = []
        chunks = []
        for fold_id in sorted({int(f) for f in fold_a.tolist() if int(f) >= 0}):
            ra, rr = _oof_refs(encoder, x_a, x_r, fold_a, fold_r, fold_id, n_ref, device)
            kwargs_f = {
                **kwargs,
                "ref_anom": ra,
                "ref_rare": rr,
                "lam_anom": float(spec["lam_anom"]),
                "lam_rare": float(spec["lam_rare"]),
            }
            samples, h = chunked_guided_ddim(
                model, encoder, x_cond, cond_ch, schedule, bsz=bsz, **kwargs_f
            )
            chunks.append(samples)
            scored = _score_gallery(
                samples,
                x_a=x_a,
                x_r=x_r,
                fold_a=fold_a,
                fold_r=fold_r,
                ch_a=ch_a,
                ch_r=ch_r,
                fold_tab=fold_tab,
                tau=tau,
                only_fold=fold_id,
            )
            scored["fold"] = fold_id
            scored["occupancy"] = float(np.mean(np.abs(h - Q_q) <= delta))
            fold_rows.append(scored)
        gallery = np.concatenate(chunks, axis=0)
        np.savez_compressed(out / f"{name}.npz", x=gallery, channel_idx=np.tile(cond_ch, 3))
        return {
            "family": family,
            "spec": _winner_brief(spec),
            "occupancy": float(np.mean([r["occupancy"] for r in fold_rows])),
            "coverage_anomaly": float(np.mean([r["coverage_anomaly"] for r in fold_rows])),
            "coverage_rare": float(np.mean([r["coverage_rare"] for r in fold_rows])),
            "gap": float(np.mean([r["gap"] for r in fold_rows])),
            "arp_anomaly": float(np.mean([r["arp_anomaly"] for r in fold_rows])),
            "arp_rare": float(np.mean([r["arp_rare"] for r in fold_rows])),
            "folds": fold_rows,
            "n": int(len(gallery)),
        }

    samples, h = chunked_guided_ddim(model, encoder, x_cond, cond_ch, schedule, bsz=bsz, **kwargs)
    np.savez_compressed(out / f"{name}.npz", x=samples, h=h, channel_idx=cond_ch)
    scored = _score_gallery(
        samples,
        x_a=x_a,
        x_r=x_r,
        fold_a=fold_a,
        fold_r=fold_r,
        ch_a=ch_a,
        ch_r=ch_r,
        fold_tab=fold_tab,
        tau=tau,
        only_fold=None,
    )
    scored["occupancy"] = float(np.mean(np.abs(h - Q_q) <= delta))
    scored["family"] = family
    scored["spec"] = _winner_brief(spec)
    scored["n"] = int(len(samples))
    scored["energy"] = band_report(h, Q_q, delta)
    return scored


def _sample_kwargs(
    spec: dict[str, Any],
    ref: Any,
    Q_q: float,
    tau_e: float,
    c_max: float,
    device: str,
    scaler: Any,
) -> dict[str, Any]:
    return {
        "ref": ref,
        "Q_q": Q_q,
        "tau": tau_e,
        "nu": float(spec["nu"]),
        "lam": float(spec["lam"]),
        "c_max": c_max,
        "ddim_steps": int(spec["ddim_steps"]),
        "device": device,
        "mode": str(spec.get("mode", "band")),
        "normalize_grad": bool(spec.get("normalize_grad", False)),
        "n_correct": int(spec.get("n_correct", 1)),
        "scaler": scaler,
    }


def _oof_refs(
    encoder: Any,
    x_a: np.ndarray,
    x_r: np.ndarray,
    fold_a: np.ndarray,
    fold_r: np.ndarray,
    fold_id: int,
    n_ref: int,
    device: str,
) -> tuple[Any, Any]:
    ia = np.where(fold_a != fold_id)[0]
    ir = np.where(fold_r != fold_id)[0]
    if len(ia) == 0:
        ia = np.arange(len(x_a))
    if len(ir) == 0:
        ir = np.arange(len(x_r))
    ia = ia[:n_ref]
    ir = ir[:n_ref]
    za = torch.from_numpy(embed_encoder(encoder, x_a[ia], device))
    zr = torch.from_numpy(embed_encoder(encoder, x_r[ir], device))
    return za, zr


def _score_gallery(
    gallery: np.ndarray,
    *,
    x_a: np.ndarray,
    x_r: np.ndarray,
    fold_a: np.ndarray,
    fold_r: np.ndarray,
    ch_a: np.ndarray,
    ch_r: np.ndarray,
    fold_tab: pd.DataFrame,
    tau: float,
    only_fold: int | None,
) -> dict[str, float]:
    folds = [only_fold] if only_fold is not None else sorted({int(f) for f in fold_a.tolist() if int(f) >= 0})
    rows = []
    for fold_id in folds:
        ka = _keep_mask(ch_a, fold_a, fold_tab, fold_id=fold_id)
        kr = _keep_mask(ch_r, fold_r, fold_tab, fold_id=fold_id)
        if int(ka.sum()) == 0:
            continue
        rare = x_r[kr] if int(kr.sum()) else np.zeros((0, x_a.shape[1]), dtype=x_a.dtype)
        rows.append(score_generator(x_a[ka], rare, gallery, tau=tau))
    if not rows:
        return {
            "coverage_anomaly": 0.0,
            "coverage_rare": 0.0,
            "gap": 0.0,
            "arp_anomaly": 0.0,
            "arp_rare": 0.0,
            "mean_min_d_anomaly": float("inf"),
            "mean_min_d_rare": float("inf"),
        }
    return {
        "coverage_anomaly": float(np.mean([r["coverage_anomaly"] for r in rows])),
        "coverage_rare": float(np.mean([r["coverage_rare"] for r in rows])),
        "gap": float(np.mean([r["gap"] for r in rows])),
        "arp_anomaly": float(np.mean([r["arp_anomaly"] for r in rows])),
        "arp_rare": float(np.mean([r["arp_rare"] for r in rows])),
        "mean_min_d_anomaly": float(np.mean([r["mean_min_d_anomaly"] for r in rows])),
        "mean_min_d_rare": float(np.mean([r["mean_min_d_rare"] for r in rows])),
    }


def _keep_mask(
    channel_idx: np.ndarray,
    folds: np.ndarray,
    tab: pd.DataFrame,
    *,
    fold_id: int,
) -> np.ndarray:
    dropped = set(tab.loc[(tab["fold"] == fold_id) & (tab["below_min"]), "channel"].astype(str))
    names = np.array([f"channel_{41 + int(c)}" for c in channel_idx])
    return (folds == fold_id) & np.array([n not in dropped for n in names])


def _winner_brief(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    keys = (
        "id",
        "family",
        "ddim_steps",
        "lam",
        "nu",
        "n_correct",
        "normalize_grad",
        "mode",
        "lam_anom",
        "lam_rare",
        "occupancy",
        "coverage_anomaly",
        "coverage_rare",
        "gap",
        "arp_anomaly",
        "arp_rare",
    )
    return {k: row[k] for k in keys if k in row}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "family",
        "ddim_steps",
        "lam",
        "nu",
        "n_correct",
        "normalize_grad",
        "mode",
        "lam_anom",
        "lam_rare",
        "finite",
        "explode",
        "occupancy",
        "h_mean",
        "coverage_anomaly",
        "coverage_rare",
        "gap",
        "arp_anomaly",
        "arp_rare",
        "mean_min_d_anomaly",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_plots(
    out: Path,
    rows: list[dict[str, Any]],
    parents: np.ndarray,
    anomalies: np.ndarray,
    bin_seconds: int,
) -> None:
    finite = [r for r in rows if r.get("finite") and not r.get("explode")]
    if not finite:
        return
    labels = [f"{r['family'][:1]}{r['id']}" for r in finite]
    save_bar(
        out / "occupancy_bars.png",
        labels,
        np.asarray([r["occupancy"] for r in finite]),
        title="Search occupancy on frozen Q_q ± δ",
        ylabel="occupancy",
    )
    save_bar(
        out / "arp_anomaly_bars.png",
        labels,
        np.asarray([r["arp_anomaly"] for r in finite]),
        title="Search ARP vs real anomalies (frozen τ, smaller gallery)",
        ylabel="ARP anomaly",
    )
    save_bar(
        out / "gap_bars.png",
        labels,
        np.asarray([r["gap"] for r in finite]),
        title="Search gap (Coverage anom − Coverage rare)",
        ylabel="gap",
    )
    path = out / "best_arp_anomaly.npz"
    if path.is_file() and len(parents) and len(anomalies):
        gen = np.load(path)["x"]
        n = min(4, len(parents), len(gen), len(anomalies))
        save_overlay(
            out / "overlay_best_arp.png",
            parents[:n],
            {"tuned": gen[:n], "real anomaly": anomalies[:n]},
            bin_seconds=bin_seconds,
        )


def _abs(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


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
