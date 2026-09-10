"""OOF encoder ablations. Does not overwrite S2/S4.

Compares pooled recon (S2-like) vs temporal latent vs 3-class SupCon
(everyday / rare / anomaly). Anomalies and rares in the contrastive loss
are train-fold only.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from anogen.config import REPO_ROOT, panel_path
from anogen.shell.coverage import score_generator
from anogen.shell.data import load_panel
from anogen.shell.diffusion import DiffusionSchedule, denoiser_from_ckpt, torch_available
from anogen.shell.encoders import (
    all_encoder_names,
    auroc_scores,
    embed_shell,
    encoder_spec,
    gradient_locality,
    h_of_windows,
    train_shell_encoder,
)
from anogen.shell.plots import save_bar, save_loss_curves
from anogen.shell.scaler import resolve_scaler
from anogen.shell.steer import chunked_guided_ddim, shell_from_nominal
from anogen.shell.windows import load_train_index, materialize


def run_enc(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s1 = _abs(cfg.get("s1_dir", root / "results/shell_s1"), root)
    s2 = _abs(cfg.get("s2_dir", root / "results/shell_s2"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    s4 = _abs(cfg.get("s4_dir", root / "results/shell_s4"), root)
    out = _abs(cfg.get("enc_dir", root / "results/shell_enc"), root)
    out.mkdir(parents=True, exist_ok=True)

    if not torch_available():
        report = {"ok": False, "skipped": True, "reason": "torch is not installed"}
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    if not (s0 / "labeled_arrays.npz").is_file() or not (s0 / "train_index.csv").is_file():
        report = {"ok": False, "skipped": True, "reason": f"S0 missing under {s0}"}
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    ecfg = dict((cfg.get("shell") or {}).get("enc_ablation") or {})
    scfg = dict((cfg.get("shell") or {}).get("steer") or {})
    dcfg = dict((cfg.get("shell") or {}).get("diffusion") or {})
    names = list(ecfg.get("variants") or all_encoder_names())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    width = int((cfg.get("shell") or {}).get("W", 512))

    channels = list(cfg.get("channels") or [])
    splits = cfg.get("splits") or {}
    panel = load_panel(
        panel_path(cfg),
        channels=channels,
        official_train_end=splits.get("official_train_end", "2007-01-01"),
        allow_test_telemetry=False,
        bin_seconds=int(cfg.get("bin_seconds", 30)),
    )
    train = load_train_index(s0)
    nom_tab = train.loc[~train["is_rare"].astype(bool)]
    x_nom = materialize(panel, nom_tab, width)
    lab = np.load(s0 / "labeled_arrays.npz")
    x_a, x_r = np.asarray(lab["anomaly"]), np.asarray(lab["rare"])
    fold_a, fold_r = np.asarray(lab["anomaly_fold"]), np.asarray(lab["rare_fold"])
    ch_a, ch_r = np.asarray(lab["anomaly_channel"]), np.asarray(lab["rare_channel"])
    fold_tab = pd.read_csv(s0 / "channel_fold_counts.csv")
    folds = sorted({int(f) for f in fold_a.tolist() if int(f) >= 0})

    gal = np.load(s3 / "galleries.npz") if (s3 / "galleries.npz").is_file() else None
    x_cond = np.asarray(gal["cond"]) if gal is not None else x_nom[:512]
    cond_ch = np.asarray(gal["channel_idx"]) if gal is not None else np.zeros(len(x_cond), dtype=np.int64)
    tau = None
    model_den = schedule = scaler = None
    if (s4 / "summary.json").is_file() and (s1 / "denoiser.pt").is_file():
        tau = float(json.loads((s4 / "summary.json").read_text())["tau"])
        den = torch.load(s1 / "denoiser.pt", map_location=device, weights_only=False)
        model_den = denoiser_from_ckpt(den, device=device)
        scaler = resolve_scaler(den, s0)
        schedule = DiffusionSchedule.linear(int(den["n_times"])).to(torch.device(device))

    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    n_ref = min(int(scfg.get("n_ref", 512)), len(x_nom))
    n_val = max(64, int(0.1 * len(x_nom)))
    perm = rng.permutation(len(x_nom))
    val_nom, train_nom_idx = perm[:n_val], perm[n_val:]
    x_nom_tr, x_nom_val = x_nom[train_nom_idx], x_nom[val_nom]
    n_steer = min(int(ecfg.get("n_steer", 192)), len(x_cond))
    steer_idx = rng.choice(len(x_cond), size=n_steer, replace=False)

    rows: list[dict[str, Any]] = []
    fold_rows: dict[str, list[dict[str, Any]]] = {n: [] for n in names}
    for name in names:
        spec = encoder_spec(name)
        print(f"enc {name} pool={spec['pool']} loss={spec['loss']}", flush=True)
        for fold_id in folds:
            xa_tr = x_a[fold_a != fold_id]
            xr_tr = x_r[fold_r != fold_id]
            steps = int(ecfg.get("steps_supcon" if spec["uses_labels"] else "steps_recon", 800))
            if spec["loss"] == "recon":
                x_fit = np.concatenate([x_nom_tr, xr_tr], axis=0) if len(xr_tr) else x_nom_tr
                x_rare_fit, x_anom_fit = None, None
            else:
                x_fit, x_rare_fit, x_anom_fit = x_nom_tr, xr_tr, xa_tr
            trained = train_shell_encoder(
                x_fit,
                x_rare_fit,
                x_anom_fit,
                pool=spec["pool"],
                loss=spec["loss"],
                hidden=int(scfg.get("enc_hidden", 32)),
                emb=int(scfg.get("emb", 32)),
                time_emb=int(ecfg.get("time_emb", 8)),
                steps=steps,
                batch_size=int(ecfg.get("batch_size", 64)),
                per_class=int(ecfg.get("per_class", 32)),
                lr=float(scfg.get("enc_lr", 1e-3)),
                temperature=float(ecfg.get("temperature", 0.1)),
                device=device,
            )
            enc = trained["encoder"]
            if fold_id == folds[0]:
                np.savez_compressed(
                    out / f"{name}_loss.npz",
                    loss_step=trained["loss_step"],
                    loss_value=trained["loss_value"],
                    loss_recon=trained["loss_recon"],
                    loss_supcon=trained["loss_supcon"],
                )
                torch.save(
                    {
                        "state_dict": enc.state_dict(),
                        "width": width,
                        "hidden": int(scfg.get("enc_hidden", 32)),
                        "emb": int(scfg.get("emb", 32)),
                        "time_emb": int(ecfg.get("time_emb", 8)),
                        "pool": spec["pool"],
                        "name": name,
                    },
                    out / f"{name}_fold{fold_id}.pt",
                )
            ref_idx = rng.choice(len(x_nom_tr), size=min(n_ref, len(x_nom_tr)), replace=False)
            shell = shell_from_nominal(
                enc,
                x_nom_tr[ref_idx],
                x_nom_val,
                q=float(scfg.get("q", 0.99)),
                delta_scale=float(scfg.get("delta_scale", 0.25)),
                device=device,
            )
            ref = shell.pop("ref")
            tau_e = float(shell["tau"])
            q_q = float(shell["Q_q"])
            delta = float(shell["delta"])
            ka = _keep_mask(ch_a, fold_a, fold_tab, fold_id=fold_id)
            kr = _keep_mask(ch_r, fold_r, fold_tab, fold_id=fold_id)
            xa_ev = x_a[ka]
            xr_ev = x_r[kr] if int(kr.sum()) else np.zeros((0, width), dtype=np.float32)
            h_n = h_of_windows(enc, x_nom_val, ref, tau_e, device)
            h_a = h_of_windows(enc, xa_ev, ref, tau_e, device)
            h_r = h_of_windows(enc, xr_ev, ref, tau_e, device)
            loc = gradient_locality(enc, x_nom_val[:32], ref, tau_e, device)
            scored = {
                "name": name,
                "fold": fold_id,
                "n_params": trained["n_params"],
                "embed_dim": trained["embed_dim"],
                "Q_q": q_q,
                "delta": delta,
                "h_nom_mean": float(h_n.mean()) if len(h_n) else float("nan"),
                "h_anom_mean": float(h_a.mean()) if len(h_a) else float("nan"),
                "h_rare_mean": float(h_r.mean()) if len(h_r) else float("nan"),
                "h_anom_std": float(h_a.std()) if len(h_a) else float("nan"),
                "h_rare_std": float(h_r.std()) if len(h_r) else float("nan"),
                "auroc_anom_vs_nom": auroc_scores(h_a, h_n),
                "auroc_anom_vs_rare": auroc_scores(h_a, h_r),
                "auroc_rare_vs_nom": auroc_scores(h_r, h_n),
                "locality": loc,
                "n_eval_anom": int(len(xa_ev)),
                "n_eval_rare": int(len(xr_ev)),
            }
            if model_den is not None and tau is not None and fold_id == folds[0]:
                samples, h_g = chunked_guided_ddim(
                    model_den,
                    enc,
                    x_cond[steer_idx],
                    cond_ch[steer_idx],
                    schedule,
                    bsz=int(scfg.get("n_sample", 64)),
                    ref=ref,
                    Q_q=q_q,
                    tau=tau_e,
                    nu=float(scfg.get("nu", 0.2)),
                    lam=float(scfg.get("lambda", 0.3)),
                    c_max=float(scfg.get("c_max", 1.0)),
                    ddim_steps=int(dcfg.get("ddim_steps", 20)),
                    device=device,
                    scaler=scaler,
                    normalize_grad=bool(spec["uses_labels"]),
                )
                finite = bool(np.isfinite(samples).all())
                gen = score_generator(xa_ev, xr_ev, samples, tau=tau) if finite else {}
                scored["steer_occupancy"] = (
                    float(np.mean(np.abs(h_g - q_q) <= delta)) if finite else 0.0
                )
                scored["steer_arp_anomaly"] = float(gen.get("arp_anomaly", 0.0))
                scored["steer_arp_rare"] = float(gen.get("arp_rare", 0.0))
                scored["steer_coverage_anomaly"] = float(gen.get("coverage_anomaly", 0.0))
                scored["steer_gap"] = float(gen.get("gap", 0.0))
                np.savez_compressed(out / f"{name}_steer.npz", x=samples, h=h_g)
            fold_rows[name].append(scored)
            print(
                f"  fold {fold_id} AUROC a/r={scored['auroc_anom_vs_rare']:.3f} "
                f"a/n={scored['auroc_anom_vs_nom']:.3f} loc={loc:.3f}",
                flush=True,
            )
        rows.append(_mean_fold(name, spec, fold_rows[name]))

    rank = sorted(rows, key=lambda r: (_nan0(r["auroc_anom_vs_rare"]), _nan0(r["auroc_anom_vs_nom"])), reverse=True)
    _write_csv(out / "metrics.csv", rows)
    _write_plots(out, rows, names)
    report = {
        "ok": True,
        "skipped": False,
        "variants": rows,
        "ranking": [r["name"] for r in rank],
        "most_promising": rank[0]["name"] if rank else None,
        "note": (
            "OOF encoder ablation. Does not overwrite results/shell_s2. "
            "SupCon sees train-fold anomalies/rares only. "
            "Primary rank: AUROC of h_soft (nominal refs) for anomaly vs rare. "
            "Steer metrics (if present) use frozen S4 τ and 192 donors — not the S4 table."
        ),
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return report


def _mean_fold(name: str, spec: dict[str, Any], folds: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "n_params",
        "embed_dim",
        "h_nom_mean",
        "h_anom_mean",
        "h_rare_mean",
        "h_anom_std",
        "h_rare_std",
        "auroc_anom_vs_nom",
        "auroc_anom_vs_rare",
        "auroc_rare_vs_nom",
        "locality",
        "steer_occupancy",
        "steer_arp_anomaly",
        "steer_arp_rare",
        "steer_coverage_anomaly",
        "steer_gap",
    ]
    out: dict[str, Any] = {"name": name, "pool": spec["pool"], "loss": spec["loss"], "n_folds": len(folds)}
    for k in keys:
        vals = [f[k] for f in folds if k in f and f[k] is not None and np.isfinite(f[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    out["folds"] = folds
    return out


def _write_plots(out: Path, rows: list[dict[str, Any]], names: list[str]) -> None:
    labels = [r["name"] for r in rows]
    for key, title, ylabel in (
        ("auroc_anom_vs_rare", "AUROC of h: anomaly vs rare (higher = separates faults)", "AUROC"),
        ("auroc_anom_vs_nom", "AUROC of h: anomaly vs everyday", "AUROC"),
        ("locality", "∇h mass on top 10% timesteps (temporal should be higher)", "fraction"),
        ("steer_arp_anomaly", "Steered ARP vs anomalies (192 donors, frozen τ)", "ARP"),
    ):
        vals = np.asarray([r.get(key, np.nan) for r in rows], dtype=np.float64)
        if np.isfinite(vals).any():
            save_bar(out / f"{key}.png", labels, np.nan_to_num(vals, nan=0.0), title=title, ylabel=ylabel)
    series = {}
    for name in names:
        path = out / f"{name}_loss.npz"
        if path.is_file():
            h = np.load(path)
            series[name] = (h["loss_step"], h["loss_value"])
    if series:
        save_loss_curves(
            out / "loss_curves.png",
            series,
            title="Encoder ablation train loss (fold 0)",
            ylabel="loss",
            log_y=True,
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "name",
        "pool",
        "loss",
        "n_params",
        "embed_dim",
        "auroc_anom_vs_rare",
        "auroc_anom_vs_nom",
        "auroc_rare_vs_nom",
        "h_anom_mean",
        "h_rare_mean",
        "h_nom_mean",
        "locality",
        "steer_occupancy",
        "steer_arp_anomaly",
        "steer_gap",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


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


def _nan0(v: Any) -> float:
    x = float(v)
    return x if np.isfinite(x) else -1.0


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
