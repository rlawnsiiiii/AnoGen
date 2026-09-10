"""S5: event-OOF few-shot adapters on the locked TSDiff score.

Does not overwrite S4. Does not retrain GenIAS. τ stays the frozen S4 value.

Recipes scored on the same 1536 S3 donors, 3-fold event-OOF:

* adapter_only — adapted ε, ν=0.2, no ∇f (S5 alone)
* adapter_cls — adapted ε + S2 band leash + classifier (documented S5)
* adapter + time_recon combined f
* adapter + time_both combined f

Plots (fold-0 adapter): ν=0.2 edits and x_T ~ N(0,I), same strip style as
results/shell_plots/shell_tune/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from anogen.config import REPO_ROOT
from anogen.phases.tune import _score_gallery
from anogen.shell.adapters import train_adapter, train_classifier
from anogen.shell.coverage import edi_by_method, mean_pairwise_distance
from anogen.shell.diffusion import (
    DiffusionSchedule,
    denoiser_from_ckpt,
    torch_available,
    unguided_from_nominal,
)
from anogen.shell.encoders import ShellEncoder, embed_shell
from anogen.shell.features import embed_windows
from anogen.shell.events import try_types_from_cfg
from anogen.shell.morphology import (
    KIND_ORDER,
    diverse_idx,
    kinds_for_windows,
    pick_kind_examples,
)
from anogen.shell.plots import save_compare_rows, save_strip
from anogen.shell.scaler import resolve_scaler
from anogen.shell.steer import (
    ConvEncoder,
    chunked_guided_ddim,
    embed_encoder,
    shell_from_nominal,
)

_COMBINED = {
    "nu": 0.2,
    "lam": 0.3,
    "ddim_steps": 50,
    "normalize_grad": True,
    "n_correct": 1,
    "lam_anom": 1.0,
    "lam_rare": 1.0,
}

_PLOT_COLORS = {
    "adapter_only": "#7570b3",
    "adapter_cls": "#e7298a",
    "adapter_time_recon_combined": "#1b9e77",
    "adapter_time_both_combined": "#66a61e",
}


def run_s5(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s1 = _abs(cfg.get("s1_dir", root / "results/shell_s1"), root)
    s2 = _abs(cfg.get("s2_dir", root / "results/shell_s2"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    s4 = _abs(cfg.get("s4_dir", root / "results/shell_s4"), root)
    enc_dir = _abs(cfg.get("enc_dir", root / "results/shell_enc"), root)
    enc_score = _abs(cfg.get("enc_score_dir", root / "results/shell_enc_score"), root)
    out = _abs(cfg.get("s5_dir", root / "results/shell_s5"), root)
    plots = _abs(cfg.get("plots_dir", root / "results/shell_plots"), root) / "shell_s5"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

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
        "time_recon": enc_dir / "time_recon_fold0.pt",
        "time_both": enc_dir / "time_both_fold0.pt",
    }
    for label, path in needed.items():
        if not path.is_file():
            report = {"ok": False, "skipped": True, "reason": f"{label} missing: {path}"}
            (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            return report

    proto = json.loads((s0 / "protocol.json").read_text())
    s4_sum = json.loads((s4 / "summary.json").read_text())
    tau = float(s4_sum["tau"])
    lab = np.load(s0 / "labeled_arrays.npz")
    gal = np.load(s3 / "galleries.npz")
    x_a = np.asarray(lab["anomaly"])
    x_r = np.asarray(lab["rare"])
    fold_a = np.asarray(lab["anomaly_fold"])
    fold_r = np.asarray(lab["rare_fold"])
    ch_a = np.asarray(lab["anomaly_channel"])
    ch_r = np.asarray(lab["rare_channel"])
    fold_tab = pd.read_csv(s0 / "channel_fold_counts.csv")
    x_cond = np.asarray(gal["cond"])
    cond_ch = np.asarray(gal["channel_idx"])
    bin_seconds = int(cfg.get("bin_seconds", 30))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    den = torch.load(s1 / "denoiser.pt", map_location=device, weights_only=False)
    backbone = denoiser_from_ckpt(den, device=device_t)
    backbone.eval()
    scaler = resolve_scaler(den, s0)
    schedule = DiffusionSchedule.linear(int(den["n_times"])).to(device_t)
    n_channels = int(den["n_channels"])
    width = int(den.get("W", x_cond.shape[1]))

    s2_enc, s2_shell = _load_s2(s2, x_cond, device)
    time_encs: dict[str, tuple[Any, dict[str, Any]]] = {}
    scfg = dict((cfg.get("shell") or {}).get("steer") or {})
    for name in ("time_recon", "time_both"):
        time_encs[name] = _load_time_encoder(
            enc_dir / f"{name}_fold0.pt",
            x_cond,
            scfg,
            device,
            int(cfg.get("seed", 0)),
        )

    acfg = dict((cfg.get("shell") or {}).get("adapter") or {})
    dcfg = dict((cfg.get("shell") or {}).get("diffusion") or {})
    tcfg = dict((cfg.get("shell") or {}).get("tune") or {})
    min_fs = int(proto.get("min_fs_windows", 8))
    bsz = int(scfg.get("n_sample", 128))
    n_ref = int(tcfg.get("n_ref_label", 256))
    edit_nu = float(scfg.get("nu", 0.2))
    ddim_s5 = int(dcfg.get("ddim_steps", 20))
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

    folds = sorted({int(f) for f in fold_a.tolist() if int(f) >= 0})
    recipe_folds: dict[str, list[dict[str, Any]]] = {
        "adapter_only": [],
        "adapter_cls": [],
        "adapter_time_recon_combined": [],
        "adapter_time_both_combined": [],
    }
    chunks: dict[str, list[np.ndarray]] = {k: [] for k in recipe_folds}
    adapter_logs: list[dict[str, Any]] = []
    fold0_models: dict[str, Any] | None = None

    for fold_id in folds:
        tr = fold_a != fold_id
        n_tr = int(tr.sum())
        if n_tr < min_fs:
            report = {
                "ok": False,
                "skipped": True,
                "reason": f"fold {fold_id} has only {n_tr} train anomaly windows (< {min_fs})",
            }
            (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            return report
        xa, ca = x_a[tr], ch_a[tr]
        xr = x_r[fold_r != fold_id]
        pick_n = rng.choice(len(x_cond), size=min(len(x_cond), max(4 * n_tr, 256)), replace=False)

        print(f"s5 fold {fold_id}: train adapter n_anom={n_tr}", flush=True)
        ad = train_adapter(
            backbone,
            schedule,
            xa,
            ca,
            x_cond[pick_n],
            cond_ch[pick_n],
            n_channels=n_channels,
            hidden=int(acfg.get("hidden", 64)),
            steps=int(acfg.get("steps", 800)),
            batch_size=int(acfg.get("batch_size", 32)),
            lr=float(acfg.get("lr", 1e-3)),
            anom_frac=float(acfg.get("anom_frac", 0.5)),
            device=device,
            scaler=scaler,
        )
        use_rare_neg = bool(acfg.get("rare_hard_neg", True)) and len(xr) > 0
        clf = train_classifier(
            xa,
            x_cond[pick_n],
            x_rare=xr if use_rare_neg else None,
            hidden=int(acfg.get("clf_hidden", 32)),
            steps=int(acfg.get("clf_steps", 400)),
            batch_size=int(acfg.get("batch_size", 32)),
            lr=float(acfg.get("lr", 1e-3)),
            device=device,
        )
        clf_ok = float(clf["acc_anomaly"]) >= 0.6 and float(clf["acc_nominal"]) >= 0.6
        if use_rare_neg and int(clf["n_rare"]) >= 4:
            clf_ok = clf_ok and float(clf["acc_rare"]) >= 0.5
        torch.save(
            {
                "state_dict": ad["adapter"].state_dict(),
                "hidden": int(acfg.get("hidden", 64)),
                "n_channels": n_channels,
                "fold": fold_id,
            },
            out / f"adapter_fold{fold_id}.pt",
        )
        adapter_logs.append(
            {
                "fold": fold_id,
                "n_train_anom": n_tr,
                "n_train_rare": int(len(xr)),
                "adapter_loss": ad["loss"],
                "n_adapter_params": ad["n_adapter_params"],
                "clf_acc_anomaly": clf["acc_anomaly"],
                "clf_acc_nominal": clf["acc_nominal"],
                "clf_acc_rare": clf["acc_rare"],
                "clf_used_for_guidance": clf_ok,
                "rare_hard_neg": use_rare_neg,
            }
        )
        if fold_id == folds[0]:
            fold0_models = {"adapted": ad["model"], "clf": clf["classifier"] if clf_ok else None}

        print(f"s5 fold {fold_id}: adapter_only ν={edit_nu}", flush=True)
        only = unguided_from_nominal(
            ad["model"],
            schedule,
            x_cond,
            cond_ch,
            nu=edit_nu,
            ddim_steps=ddim_s5,
            device=device,
            scaler=scaler,
        )
        print(f"s5 fold {fold_id}: adapter_cls", flush=True)
        guided, h_cls = chunked_guided_ddim(
            ad["model"],
            s2_enc,
            x_cond,
            cond_ch,
            schedule,
            bsz=bsz,
            ref=s2_shell["ref"],
            Q_q=float(s2_shell["Q_q"]),
            tau=float(s2_shell["tau"]),
            nu=edit_nu,
            lam=float(acfg.get("lambda_shell", 0.15)),
            c_max=float(scfg.get("c_max", 1.0)),
            ddim_steps=ddim_s5,
            device=device,
            mode="band",
            classifier=clf["classifier"] if clf_ok else None,
            lam_cls=float(acfg.get("lambda_cls", 0.8)) if clf_ok else 0.0,
            normalize_grad=True,
            scaler=scaler,
        )

        fold_gals = {
            "adapter_only": only,
            "adapter_cls": guided,
        }
        occ = {
            "adapter_only": float("nan"),
            "adapter_cls": float(np.mean(np.abs(h_cls - float(s2_shell["Q_q"])) <= float(s2_shell["delta"]))),
        }
        q_store = {
            "adapter_only": float(s2_shell["Q_q"]),
            "adapter_cls": float(s2_shell["Q_q"]),
        }

        for variant in ("time_recon", "time_both"):
            enc, shell = time_encs[variant]
            ra = torch.from_numpy(embed_shell(enc, x_a[fold_a != fold_id][:n_ref], device))
            rr = torch.from_numpy(embed_shell(enc, x_r[fold_r != fold_id][:n_ref], device))
            key = f"adapter_{variant}_combined"
            print(f"s5 fold {fold_id}: {key}", flush=True)
            samples, h = chunked_guided_ddim(
                ad["model"],
                enc,
                x_cond,
                cond_ch,
                schedule,
                bsz=bsz,
                ref=shell["ref"],
                Q_q=float(shell["Q_q"]),
                tau=float(shell["tau"]),
                c_max=float(scfg.get("c_max", 1.0)),
                device=device,
                scaler=scaler,
                ref_anom=ra,
                ref_rare=rr,
                **_COMBINED,
            )
            fold_gals[key] = samples
            occ[key] = float(np.mean(np.abs(h - float(shell["Q_q"])) <= float(shell["delta"])))
            q_store[key] = float(shell["Q_q"])

        np.savez_compressed(out / f"fold{fold_id}_galleries.npz", channel_idx=cond_ch, **fold_gals)
        for name, gx in fold_gals.items():
            scored = _score_gallery(gx, only_fold=fold_id, **score_kw)
            scored["fold"] = fold_id
            scored["occupancy"] = occ[name]
            scored["Q_q"] = q_store[name]
            recipe_folds[name].append(scored)
            chunks[name].append(gx)

    methods = {}
    for name, rows in recipe_folds.items():
        methods[name] = {
            "coverage_anomaly": float(np.mean([r["coverage_anomaly"] for r in rows])),
            "coverage_rare": float(np.mean([r["coverage_rare"] for r in rows])),
            "gap": float(np.mean([r["gap"] for r in rows])),
            "arp_anomaly": float(np.mean([r["arp_anomaly"] for r in rows])),
            "arp_rare": float(np.mean([r["arp_rare"] for r in rows])),
            "mean_min_d_anomaly": float(np.mean([r["mean_min_d_anomaly"] for r in rows])),
            "mean_min_d_rare": float(np.mean([r["mean_min_d_rare"] for r in rows])),
            "occupancy": float(np.nanmean([r["occupancy"] for r in rows])),
            "diversity": mean_pairwise_distance(embed_windows(chunks[name][0])),
            "n": int(len(chunks[name][0])),
            "Q_q": rows[0]["Q_q"],
            "shot": "FS",
        }

    noise_gals: dict[str, np.ndarray] = {}
    if fold0_models is not None:
        noise_gals = _from_noise(
            adapted=fold0_models["adapted"],
            clf=fold0_models["clf"],
            s2_enc=s2_enc,
            s2_shell=s2_shell,
            time_encs=time_encs,
            x_a=x_a,
            x_r=x_r,
            fold_a=fold_a,
            fold_r=fold_r,
            schedule=schedule,
            scaler=scaler,
            width=width,
            n_channels=n_channels,
            device=device,
            acfg=acfg,
            scfg=scfg,
            n_ref=n_ref,
            bsz=bsz,
            ddim_s5=ddim_s5,
            first_fold=folds[0],
        )
        for name, gx in noise_gals.items():
            np.savez_compressed(out / f"{name}_from_noise.npz", x=gx)

    written = _write_plots(
        plots,
        {k: chunks[k][0] for k in chunks if chunks[k]},
        noise_gals,
        x_a,
        rng,
        bin_seconds,
        s0,
        try_types_from_cfg(cfg),
    )

    edi = _edi_s5_union(s3=s3, s4=s4, enc_score=enc_score, s5=out, gal=gal)
    for name, val in edi.items():
        if name in methods:
            methods[name]["edi_table"] = val

    report = {
        "ok": True,
        "skipped": False,
        "tau": tau,
        "tau_source": "s4_frozen",
        "n_adapter_params": adapter_logs[0]["n_adapter_params"] if adapter_logs else 0,
        "n_backbone_params": int(sum(p.numel() for p in backbone.parameters())),
        "backbone": str(den.get("backbone", "tsdiff")),
        "nu": edit_nu,
        "n_donors": int(len(x_cond)),
        "adapter_logs": adapter_logs,
        "methods": methods,
        "folds": recipe_folds,
        "edi_table_union": {
            "k": 16,
            "n_per_gallery": 1536,
            "values": edi,
            "note": (
                "GenIAS EDI on S4 + encscore + S5 fold-0 galleries (1536 each). "
                "Not the locked S4 5-method EDI and not the encscore 9-method EDI."
            ),
        },
        "plots": written,
        "plots_dir": str(plots),
        "note": (
            "Event-OOF adapters on the locked min-max TSDiff score. "
            "No GenIAS retrain. Combined f uses event-OOF contrast refs. "
            "time_* encoders are fold-0 weights. Does not overwrite results/shell_s4. "
            "S6 stays sealed."
        ),
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return report


def _from_noise(
    *,
    adapted: Any,
    clf: Any,
    s2_enc: Any,
    s2_shell: dict[str, Any],
    time_encs: dict[str, tuple[Any, dict[str, Any]]],
    x_a: np.ndarray,
    x_r: np.ndarray,
    fold_a: np.ndarray,
    fold_r: np.ndarray,
    schedule: Any,
    scaler: Any,
    width: int,
    n_channels: int,
    device: str,
    acfg: dict[str, Any],
    scfg: dict[str, Any],
    n_ref: int,
    bsz: int,
    ddim_s5: int,
    first_fold: int,
) -> dict[str, np.ndarray]:
    n = 128
    dummy = np.zeros((n, width), dtype=np.float32)
    ch = np.arange(n, dtype=np.int64) % n_channels
    out: dict[str, np.ndarray] = {}
    print("s5 plots: from-noise adapter_only", flush=True)
    only, _ = chunked_guided_ddim(
        adapted,
        s2_enc,
        dummy,
        ch,
        schedule,
        bsz=bsz,
        ref=s2_shell["ref"],
        Q_q=float(s2_shell["Q_q"]),
        tau=float(s2_shell["tau"]),
        nu=1.0,
        lam=0.0,
        c_max=float(scfg.get("c_max", 1.0)),
        ddim_steps=ddim_s5,
        device=device,
        mode="off",
        scaler=scaler,
        start_from_noise=True,
    )
    out["adapter_only"] = only
    print("s5 plots: from-noise adapter_cls", flush=True)
    guided, _ = chunked_guided_ddim(
        adapted,
        s2_enc,
        dummy,
        ch,
        schedule,
        bsz=bsz,
        ref=s2_shell["ref"],
        Q_q=float(s2_shell["Q_q"]),
        tau=float(s2_shell["tau"]),
        nu=1.0,
        lam=float(acfg.get("lambda_shell", 0.15)),
        c_max=float(scfg.get("c_max", 1.0)),
        ddim_steps=ddim_s5,
        device=device,
        mode="band",
        classifier=clf,
        lam_cls=float(acfg.get("lambda_cls", 0.8)) if clf is not None else 0.0,
        normalize_grad=True,
        scaler=scaler,
        start_from_noise=True,
    )
    out["adapter_cls"] = guided
    for variant in ("time_recon", "time_both"):
        enc, shell = time_encs[variant]
        ra = torch.from_numpy(embed_shell(enc, x_a[fold_a != first_fold][:n_ref], device))
        rr = torch.from_numpy(embed_shell(enc, x_r[fold_r != first_fold][:n_ref], device))
        print(f"s5 plots: from-noise adapter_{variant}_combined", flush=True)
        samples, _ = chunked_guided_ddim(
            adapted,
            enc,
            dummy,
            ch,
            schedule,
            bsz=bsz,
            ref=shell["ref"],
            Q_q=float(shell["Q_q"]),
            tau=float(shell["tau"]),
            c_max=float(scfg.get("c_max", 1.0)),
            device=device,
            scaler=scaler,
            ref_anom=ra,
            ref_rare=rr,
            start_from_noise=True,
            **{**_COMBINED, "nu": 1.0},
        )
        out[f"adapter_{variant}_combined"] = samples
    return out


def _write_plots(
    plots: Path,
    edit: dict[str, np.ndarray],
    noise: dict[str, np.ndarray],
    x_a: np.ndarray,
    rng: np.random.Generator,
    bin_seconds: int,
    s0: Path,
    types: pd.DataFrame | None = None,
) -> list[str]:
    titles = {
        "adapter_only": "S5 adapter-only",
        "adapter_cls": "S5 adapter + classifier",
        "adapter_time_recon_combined": "S5 + time_recon + combined f",
        "adapter_time_both_combined": "S5 + time_both + combined f",
    }
    written: list[str] = []
    real_rows = _real_kind_rows(s0, x_a, types)
    for name, title in titles.items():
        color = _PLOT_COLORS.get(name, "#1b9e77")
        if name in edit:
            written.extend(
                _recipe_plots(plots, name, title, "ν=0.2", edit[name], real_rows, rng, bin_seconds, color)
            )
        if name in noise:
            written.extend(
                _recipe_plots(
                    plots,
                    f"{name}_from_noise",
                    title,
                    "x_T ~ N(0, I)",
                    noise[name],
                    real_rows,
                    rng,
                    bin_seconds,
                    color,
                )
            )
    return written


def _recipe_plots(
    plots: Path,
    stem: str,
    title: str,
    start: str,
    samples: np.ndarray,
    real_rows: dict[str, np.ndarray],
    rng: np.random.Generator,
    bin_seconds: int,
    color: str,
) -> list[str]:
    written: list[str] = []
    idx = diverse_idx(samples, 8, rng)
    save_strip(
        plots / f"gen_{stem}.png",
        samples,
        title=f"{title}  {start}",
        color=color,
        bin_seconds=bin_seconds,
        idx=idx,
    )
    written.append(f"gen_{stem}.png")
    amp = samples.max(axis=1) - samples.min(axis=1)
    ext = np.argsort(amp)[::-1][:8]
    save_strip(
        plots / f"gen_{stem}_extreme.png",
        samples,
        title=f"{title}  {start} (largest range)",
        color=color,
        bin_seconds=bin_seconds,
        idx=ext,
    )
    written.append(f"gen_{stem}_extreme.png")
    gen_label = f"generated ({start})"
    rows: dict[str, np.ndarray] = {gen_label: samples[idx[:4]] if len(idx) else samples[:4]}
    rows.update(real_rows)
    save_compare_rows(
        plots / f"real_vs_{stem}.png",
        rows,
        bin_seconds=bin_seconds,
        n_cols=4,
        colors={gen_label: color, **{k: "#b2182b" for k in real_rows}},
        title=f"Real ESA-ADB kinds vs {title} ({start})",
    )
    written.append(f"real_vs_{stem}.png")
    return written


def _real_kind_rows(
    s0: Path, x_a: np.ndarray, types: pd.DataFrame | None = None
) -> dict[str, np.ndarray]:
    meta_path = s0 / "labeled_windows.csv"
    if not meta_path.is_file() or not len(x_a):
        return {"real anomaly (extreme)": x_a[np.argsort(x_a.max(1) - x_a.min(1))[::-1][:4]]}
    anom = pd.read_csv(meta_path)
    anom = anom[anom["kind"] == "anomaly"].reset_index(drop=True)
    if len(anom) != len(x_a):
        return {"real anomaly (extreme)": x_a[np.argsort(x_a.max(1) - x_a.min(1))[::-1][:4]]}
    try:
        kinds = kinds_for_windows(x_a, anom, types)
    except ValueError:
        return {"real anomaly (extreme)": x_a[np.argsort(x_a.max(1) - x_a.min(1))[::-1][:4]]}
    picked = pick_kind_examples(
        x_a, kinds, anom["event_id"].to_numpy(), n=4, order=KIND_ORDER
    )
    return {name: blob["x"] for name, blob in picked.items()}


def _load_s2(s2: Path, parent: np.ndarray, device: str) -> tuple[Any, dict[str, Any]]:
    device_t = torch.device(device)
    blob = torch.load(s2 / "encoder.pt", map_location=device, weights_only=False)
    enc = ConvEncoder(int(blob["width"]), hidden=int(blob["hidden"]), emb=int(blob["emb"]))
    enc.load_state_dict(blob["state_dict"])
    enc.to(device_t)
    enc.eval()
    shell = {
        "ref": torch.from_numpy(embed_encoder(enc, parent[: min(512, len(parent))], device)),
        "Q_q": float(blob["Q_q"]),
        "tau": float(blob["tau"]),
        "delta": float(blob["delta"]),
    }
    return enc, shell


def _load_time_encoder(
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


def _edi_s5_union(
    *,
    s3: Path,
    s4: Path,
    enc_score: Path,
    s5: Path,
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
        band = enc_score / f"{variant}_band.npz"
        comb = enc_score / f"{variant}_combined_fold0.npz"
        if band.is_file():
            embs[f"{variant}_band"] = embed_windows(np.asarray(np.load(band)["x"]))
        if comb.is_file():
            embs[f"{variant}_combined"] = embed_windows(np.asarray(np.load(comb)["x"]))
    fold0 = s5 / "fold0_galleries.npz"
    if fold0.is_file():
        z = np.load(fold0)
        for key in (
            "adapter_only",
            "adapter_cls",
            "adapter_time_recon_combined",
            "adapter_time_both_combined",
        ):
            if key in z.files:
                embs[key] = embed_windows(np.asarray(z[key]))
    return edi_by_method(embs)


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
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj
