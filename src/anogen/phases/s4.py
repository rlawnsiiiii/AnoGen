"""S4: OOF coverage vs rare collision. Freeze τ. Apply the S0 win rule."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from anogen.config import REPO_ROOT
from anogen.shell.coverage import edi_by_method, mean_pairwise_distance, min_distances, score_generator
from anogen.shell.diffusion import DiffusionSchedule, denoiser_from_ckpt, torch_available, unguided_from_nominal
from anogen.shell.features import embed_windows
from anogen.shell.protocol import choose_tau, win_rule
from anogen.shell.scaler import resolve_scaler
from anogen.shell.steer import (
    ConvEncoder,
    band_report,
    embed_encoder,
    guided_ddim,
    soft_energy,
)


def run_s4(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s1 = _abs(cfg.get("s1_dir", root / "results/shell_s1"), root)
    s2 = _abs(cfg.get("s2_dir", root / "results/shell_s2"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    out = _abs(cfg.get("s4_dir", root / "results/shell_s4"), root)
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
    }
    for label, path in needed.items():
        if not path.is_file():
            report = {"ok": False, "skipped": True, "reason": f"{label} missing: {path}"}
            (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            return report

    proto = json.loads((s0 / "protocol.json").read_text())
    lab = np.load(s0 / "labeled_arrays.npz")
    gal = np.load(s3 / "galleries.npz")
    x_a, x_r = lab["anomaly"], lab["rare"]
    fold_a, fold_r = lab["anomaly_fold"], lab["rare_fold"]
    ch_a, ch_r = lab["anomaly_channel"], lab["rare_channel"]
    fold_tab = pd.read_csv(s0 / "channel_fold_counts.csv")

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
    schedule = DiffusionSchedule.linear(int(den["n_times"])).to(device_t)
    ref = torch.from_numpy(
        embed_encoder(encoder, gal["cond"][: min(512, len(gal["cond"]))], device)
    )

    scfg = dict((cfg.get("shell") or {}).get("steer") or {})
    dcfg = dict((cfg.get("shell") or {}).get("diffusion") or {})
    x_cond = gal["cond"]
    cond_ch = gal["channel_idx"]
    shell_x = _chunked_guided(
        model,
        encoder,
        x_cond,
        cond_ch,
        schedule,
        ref=ref,
        Q_q=float(enc_blob["Q_q"]),
        tau_e=float(enc_blob["tau"]),
        nu=float(scfg.get("nu", 0.2)),
        lam=float(scfg.get("lambda", 0.3)),
        c_max=float(scfg.get("c_max", 1.0)),
        ddim_steps=int(dcfg.get("ddim_steps", 20)),
        device=device,
        mode="band",
        bsz=int(scfg.get("n_sample", 128)),
        scaler=scaler,
    )
    unguided = unguided_from_nominal(
        model,
        schedule,
        x_cond,
        cond_ch,
        nu=float(dcfg.get("unguided_nu", 1.0)),
        ddim_steps=int(dcfg.get("ddim_steps", 20)),
        device=device,
        scaler=scaler,
    )
    galleries = {
        "shell": shell_x,
        "genias": np.asarray(gal["genias"]),
        "posthoc": np.asarray(gal["posthoc"]),
        "unguided": unguided,
    }
    np.savez_compressed(out / "shell_gallery.npz", x=shell_x, unguided=unguided, channel_idx=cond_ch)

    q_q = float(enc_blob["Q_q"])
    delta = float(enc_blob["delta"])
    tau_e = float(enc_blob["tau"])
    h = _shell_h(encoder, shell_x, ref, tau_e, device)
    occupancy = float(np.mean(np.abs(h - q_q) <= delta))
    energy = {
        "generated_shell": band_report(h, q_q, delta),
        "real_anomaly": band_report(
            _shell_h(encoder, x_a, ref, tau_e, device) if len(x_a) else np.zeros(0),
            q_q,
            delta,
        ),
        "real_rare": band_report(
            _shell_h(encoder, x_r, ref, tau_e, device) if len(x_r) else np.zeros(0),
            q_q,
            delta,
        ),
        "note": (
            "h_soft of real Anomaly and Rare Event windows vs the frozen nominal "
            "Q_q. Diagnostic only. Do not retune lambda/q/delta from this."
        ),
    }

    keep0 = _keep_mask(ch_a, fold_a, fold_tab, fold_id=0)
    if int(keep0.sum()) == 0:
        keep0 = fold_a == 0
    d_cal = min_distances(embed_windows(x_a[keep0]), embed_windows(unguided))
    tau = choose_tau(d_cal, float(proto["tau"]["unguided_target"]))

    methods = _score_methods(
        galleries,
        x_a=x_a,
        x_r=x_r,
        fold_a=fold_a,
        fold_r=fold_r,
        ch_a=ch_a,
        ch_r=ch_r,
        fold_tab=fold_tab,
        tau=tau,
    )

    decision = win_rule(
        shell_gap=methods["shell"]["gap"],
        baseline_gaps={
            "genias": methods["genias"]["gap"],
            "posthoc": methods["posthoc"]["gap"],
        },
        occupancy=occupancy,
        diversity=methods["shell"]["diversity"],
        unguided_diversity=methods["unguided"]["diversity"],
        protocol=proto,
    )

    report = {
        "ok": True,
        "skipped": False,
        "tau": tau,
        "occupancy": occupancy,
        "Q_q": q_q,
        "delta": delta,
        "energy_diagnostic": energy,
        "methods": {k: {kk: vv for kk, vv in v.items() if kk != "folds"} for k, v in methods.items()},
        "folds": {k: v["folds"] for k, v in methods.items()},
        "win": decision,
        "note": (
            "tau frozen from unguided vs fold-0 anomalies. "
            "ARP/EDI use locked feature_pack_v1 (not Deep SVDD). "
            "Do not retune lambda/q/delta after this table. "
            "energy_diagnostic does not set Q_q."
        ),
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return report


def _abs(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _score_methods(
    galleries: dict[str, np.ndarray],
    *,
    x_a: np.ndarray,
    x_r: np.ndarray,
    fold_a: np.ndarray,
    fold_r: np.ndarray,
    ch_a: np.ndarray,
    ch_r: np.ndarray,
    fold_tab: pd.DataFrame,
    tau: float,
) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for name, gx in galleries.items():
        fold_rows = []
        for fold_id in sorted({int(f) for f in fold_a.tolist() if int(f) >= 0}):
            ka = _keep_mask(ch_a, fold_a, fold_tab, fold_id=fold_id)
            kr = _keep_mask(ch_r, fold_r, fold_tab, fold_id=fold_id)
            if int(ka.sum()) == 0:
                continue
            rare = x_r[kr] if int(kr.sum()) else np.zeros((0, x_a.shape[1]), dtype=x_a.dtype)
            scored = score_generator(x_a[ka], rare, gx, tau=tau)
            scored["fold"] = fold_id
            fold_rows.append(scored)
        methods[name] = {
            "coverage_anomaly": float(np.mean([r["coverage_anomaly"] for r in fold_rows])),
            "coverage_rare": float(np.mean([r["coverage_rare"] for r in fold_rows])),
            "gap": float(np.mean([r["gap"] for r in fold_rows])),
            "arp_anomaly": float(np.mean([r["arp_anomaly"] for r in fold_rows])),
            "arp_rare": float(np.mean([r["arp_rare"] for r in fold_rows])),
            "mean_min_d_anomaly": float(np.mean([r["mean_min_d_anomaly"] for r in fold_rows])),
            "mean_min_d_rare": float(np.mean([r["mean_min_d_rare"] for r in fold_rows])),
            "diversity": mean_pairwise_distance(embed_windows(gx)),
            "folds": fold_rows,
        }
    edi = edi_by_method({name: embed_windows(gx) for name, gx in galleries.items()})
    for name, val in edi.items():
        methods[name]["edi"] = val
    return methods


def _keep_mask(
    channel_idx: np.ndarray,
    folds: np.ndarray,
    tab: pd.DataFrame,
    *,
    fold_id: int,
) -> np.ndarray:
    dropped = set(
        tab.loc[(tab["fold"] == fold_id) & (tab["below_min"]), "channel"].astype(str)
    )
    names = np.array([f"channel_{41 + int(c)}" for c in channel_idx])
    return (folds == fold_id) & np.array([n not in dropped for n in names])


def _chunked_guided(
    model: Any,
    encoder: Any,
    x_cond: np.ndarray,
    cond_ch: np.ndarray,
    schedule: Any,
    *,
    ref: Any,
    Q_q: float,
    tau_e: float,
    nu: float,
    lam: float,
    c_max: float,
    ddim_steps: int,
    device: str,
    mode: str,
    bsz: int,
    scaler: Any | None = None,
) -> np.ndarray:
    chunks = []
    for i in range(0, len(x_cond), bsz):
        s, _h = guided_ddim(
            model,
            encoder,
            x_cond[i : i + bsz],
            cond_ch[i : i + bsz],
            schedule,
            ref=ref,
            Q_q=Q_q,
            tau=tau_e,
            nu=nu,
            lam=lam,
            c_max=c_max,
            ddim_steps=ddim_steps,
            device=device,
            mode=mode,
            scaler=scaler,
        )
        chunks.append(s)
    return np.concatenate(chunks, axis=0)


def _shell_h(encoder: Any, x: np.ndarray, ref: Any, tau: float, device: str) -> np.ndarray:
    device_t = torch.device(device)
    z = torch.from_numpy(embed_encoder(encoder, x, device)).to(device_t)
    h = soft_energy(z, ref.to(device_t), tau)
    return h.detach().cpu().numpy()


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
