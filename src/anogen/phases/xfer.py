"""Transfer eval on other ESA channels / missions.

Retrains a compact denoiser + GenIAS + time_recon on each dataset.
Does not touch results/shell_s* or run S6. Per-dataset τ — not the locked 0.105.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anogen.config import REPO_ROOT, causaldiscovery_root, data_root, load_config, panel_path
from anogen.shell.baselines import posthoc_inject
from anogen.shell.coverage import edi_by_method, mean_pairwise_distance, min_distances, score_generator
from anogen.shell.data import load_panel
from anogen.shell.encoders import embed_shell, train_shell_encoder
from anogen.shell.features import embed_windows
from anogen.shell.folds import keep_fold_mask
from anogen.shell.protocol import choose_tau, win_rule
from anogen.shell.scaler import fit_channel_minmax, resolve_scaler
from anogen.shell.steer import band_report, chunked_guided_ddim, shell_from_nominal
from anogen.shell.windows import load_train_index, materialize
from anogen.shell.xfer_panel import bin_from_pickles, slice_panel, write_panel_meta

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

_METHODS = ("unguided", "posthoc", "genias", "time_recon_band", "time_recon_combined")


def run_xfer(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    listed = list((cfg.get("xfer") or {}).get("datasets") or [])
    if listed:
        children = []
        for rel in listed:
            child = load_config(root / rel if not Path(rel).is_absolute() else Path(rel))
            child["phase"] = "xfer"
            try:
                children.append(run_xfer_dataset(child))
            except Exception as exc:  # noqa: BLE001 — isolate datasets
                name = str((child.get("xfer") or {}).get("name") or rel)
                children.append({"ok": False, "skipped": False, "name": name, "reason": str(exc)})
        report = {
            "ok": all(r.get("ok") for r in children),
            "skipped": all(r.get("skipped") for r in children),
            "datasets": children,
            "note": (
                "Per-dataset τ and galleries. Do not mix with locked S4 "
                "(Mission 1 ch 41–46, τ=0.105, 1536 donors)."
            ),
        }
        out = root / "results" / "xfer"
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
        return report
    return run_xfer_dataset(cfg)


def run_xfer_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    from anogen.shell.diffusion import (
        DiffusionSchedule,
        denoiser_from_ckpt,
        torch_available,
    )

    root = Path(cfg.get("_repo_root", REPO_ROOT))
    xcfg = dict(cfg.get("xfer") or {})
    name = str(xcfg.get("name") or "dataset")
    force = bool(xcfg.get("force", False))
    ds = _abs(cfg.get("xfer_dir", root / "results" / "xfer" / name), root)
    ds.mkdir(parents=True, exist_ok=True)
    s0 = ds / "s0"
    s1 = ds / "s1"
    s3 = ds / "s3"
    enc_dir = ds / "enc"
    score_dir = ds / "score"
    plots = ds / "plots"
    for p in (s0, s1, s3, enc_dir, score_dir, plots):
        p.mkdir(parents=True, exist_ok=True)

    channels = list(cfg.get("channels") or [])
    if not channels:
        raise ValueError("xfer config must list channels")
    splits = cfg.get("splits") or {}
    official_end = splits.get("official_train_end", "2007-01-01")
    bin_seconds = int(cfg.get("bin_seconds", 30))
    local = dict(cfg)
    local["allow_test_telemetry"] = False
    local["output_dir"] = str(s0)
    local["s0_dir"] = str(s0)
    local["panel_npz"] = str(ds / "panel.npz")
    local["phase"] = "s0"

    if not torch_available():
        report = {"ok": False, "skipped": True, "name": name, "reason": "torch is not installed"}
        (ds / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    print(f"xfer {name}: panel", flush=True)
    panel_info = _ensure_panel(local, ds / "panel.npz", force=force)
    print(f"xfer {name}: s0 windows", flush=True)
    s0_rep = _ensure_s0(local, s0, force=force)
    if not s0_rep.get("ok"):
        (ds / "summary.json").write_text(json.dumps(_jsonable(s0_rep), indent=2) + "\n")
        return {**s0_rep, "name": name}

    proto = json.loads((s0 / "protocol.json").read_text())
    width = int(proto["W"])
    n_gen = int(proto["N_generate"])
    panel = load_panel(
        panel_path(local),
        channels=channels,
        official_train_end=official_end,
        allow_test_telemetry=False,
        bin_seconds=bin_seconds,
    )
    train = load_train_index(s0)
    nominal = pd.read_csv(s0 / "nominal_index.csv")
    lab = np.load(s0 / "labeled_arrays.npz")
    x_a = np.asarray(lab["anomaly"])
    x_r = np.asarray(lab["rare"])
    fold_a = np.asarray(lab["anomaly_fold"])
    fold_r = np.asarray(lab["rare_fold"])
    ch_a = np.asarray(lab["anomaly_channel"])
    ch_r = np.asarray(lab["rare_channel"])
    fold_tab = pd.read_csv(s0 / "channel_fold_counts.csv")

    print(f"xfer {name}: denoiser", flush=True)
    den_rep = _ensure_denoiser(local, s0, s1, panel, train, width, force=force)
    print(f"xfer {name}: genias + post-hoc", flush=True)
    gal_rep = _ensure_galleries(local, s0, s3, panel, train, nominal, width, n_gen, force=force)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    den = torch.load(s1 / "denoiser.pt", map_location=device, weights_only=False)
    model = denoiser_from_ckpt(den, device=device_t)
    scaler = resolve_scaler(den, s0)
    schedule = DiffusionSchedule.linear(int(den["n_times"])).to(device_t)
    gal = np.load(s3 / "galleries.npz")
    x_cond = np.asarray(gal["cond"])
    cond_ch = np.asarray(gal["channel_idx"])

    print(f"xfer {name}: time_recon encoder", flush=True)
    enc, shell = _ensure_encoder(local, s0, enc_dir, panel, train, width, x_cond, device, force=force)

    scfg = dict((local.get("shell") or {}).get("steer") or {})
    dcfg = dict((local.get("shell") or {}).get("diffusion") or {})
    tcfg = dict((local.get("shell") or {}).get("tune") or {})
    bsz = int(scfg.get("n_sample", 64))
    n_ref = int(tcfg.get("n_ref_label", 256))

    print(f"xfer {name}: unguided n={len(x_cond)}", flush=True)
    unguided = _ensure_unguided(
        score_dir,
        model,
        schedule,
        x_cond,
        cond_ch,
        nu=float(dcfg.get("unguided_nu", 1.0)),
        ddim_steps=int(dcfg.get("ddim_steps", 20)),
        device=device,
        scaler=scaler,
        force=force,
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
    print(f"xfer {name}: time_recon band", flush=True)
    band_x, band_h = _ensure_band(
        score_dir, model, enc, x_cond, cond_ch, schedule, common, force=force
    )
    print(f"xfer {name}: time_recon combined OOF", flush=True)
    comb_folds, comb0_x, comb0_h = _ensure_combined(
        score_dir,
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
        force=force,
    )

    keep0 = keep_fold_mask(ch_a, fold_a, fold_tab, fold_id=0, channels=channels)
    if int(keep0.sum()) == 0:
        keep0 = fold_a == 0
    if int(keep0.sum()) == 0:
        report = {
            "ok": False,
            "skipped": True,
            "name": name,
            "reason": "no fold-0 anomaly windows to choose tau",
        }
        (ds / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    target = float(proto["tau"]["unguided_target"])
    tau = choose_tau(min_distances(embed_windows(x_a[keep0]), embed_windows(unguided)), target)

    galleries = {
        "unguided": unguided,
        "posthoc": np.asarray(gal["posthoc"]),
        "genias": np.asarray(gal["genias"]),
        "time_recon_band": band_x,
    }
    methods = _score_methods(
        galleries,
        x_a=x_a,
        x_r=x_r,
        fold_a=fold_a,
        fold_r=fold_r,
        ch_a=ch_a,
        ch_r=ch_r,
        fold_tab=fold_tab,
        channels=channels,
        tau=tau,
    )
    methods["time_recon_band"]["occupancy"] = float(
        np.mean(np.abs(band_h - float(shell["Q_q"])) <= float(shell["delta"]))
    )
    methods["time_recon_band"]["energy"] = band_report(
        band_h, float(shell["Q_q"]), float(shell["delta"])
    )
    methods["time_recon_band"]["shot"] = "ZS"
    methods["time_recon_combined"] = _score_combined(
        comb_folds,
        x_a=x_a,
        x_r=x_r,
        fold_a=fold_a,
        fold_r=fold_r,
        ch_a=ch_a,
        ch_r=ch_r,
        fold_tab=fold_tab,
        channels=channels,
        tau=tau,
        q_q=float(shell["Q_q"]),
        delta=float(shell["delta"]),
        gallery0=comb0_x,
        h0=comb0_h,
    )

    edi = edi_by_method(
        {
            "unguided": embed_windows(unguided),
            "posthoc": embed_windows(np.asarray(gal["posthoc"])),
            "genias": embed_windows(np.asarray(gal["genias"])),
            "time_recon_band": embed_windows(band_x),
            "time_recon_combined": embed_windows(comb0_x),
        }
    )
    for key, val in edi.items():
        methods[key]["edi"] = val

    decision = win_rule(
        shell_gap=methods["time_recon_band"]["gap"],
        baseline_gaps={
            "genias": methods["genias"]["gap"],
            "posthoc": methods["posthoc"]["gap"],
        },
        occupancy=float(methods["time_recon_band"]["occupancy"]),
        diversity=methods["time_recon_band"]["diversity"],
        unguided_diversity=methods["unguided"]["diversity"],
        protocol=proto,
    )

    _write_plots(
        plots,
        parent=x_cond,
        unguided=unguided,
        posthoc=np.asarray(gal["posthoc"]),
        genias=np.asarray(gal["genias"]),
        band=band_x,
        combined=comb0_x,
        real_anom=x_a,
        methods=methods,
        bin_seconds=bin_seconds,
        name=name,
    )

    report = {
        "ok": True,
        "skipped": False,
        "name": name,
        "channels": channels,
        "bin_seconds": bin_seconds,
        "W": width,
        "n_donors": int(len(x_cond)),
        "N_generate": n_gen,
        "tau": tau,
        "tau_source": "unguided vs fold-0 anomalies on this dataset",
        "Q_q": float(shell["Q_q"]),
        "delta": float(shell["delta"]),
        "occupancy_band": methods["time_recon_band"]["occupancy"],
        "compact": {
            "denoiser_steps": den_rep.get("steps"),
            "genias_steps": gal_rep.get("steps"),
            "n_nominal_per_channel": proto.get("n_nominal_per_channel"),
            "note": "Reduced vs locked Mission 1 S4 (20k / 15k / 4096 / 256).",
        },
        "s0": {
            "viable": s0_rep.get("viable"),
            "n_anomaly_windows": s0_rep.get("n_anomaly_windows"),
            "n_anomaly_events_windowed": s0_rep.get("n_anomaly_events_windowed"),
            "n_rare_windows": s0_rep.get("n_rare_windows"),
            "n_nominal_windows": s0_rep.get("n_nominal_windows"),
        },
        "panel": panel_info,
        "methods": {k: _brief(v) for k, v in methods.items()},
        "folds": {k: v.get("folds") for k, v in methods.items() if v.get("folds") is not None},
        "win_band_vs_refs": decision,
        "note": (
            "Isolated tree results/xfer/. Do not overwrite results/shell_s*. "
            "Do not mix Coverage@τ with locked S4 (different τ, donors, channels). "
            "time_recon + band is ZS. time_recon + combined is few-shot (event-OOF refs). "
            "S5 and time_both are not run. EDI is this dataset's 5-method union only."
        ),
    }
    (score_dir / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    (ds / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    _write_methods_csv(ds / "methods.csv", methods, tau)
    return report


def _ensure_panel(cfg: dict[str, Any], dest: Path, *, force: bool) -> dict[str, Any]:
    spec = dict((cfg.get("xfer") or {}).get("panel") or {})
    channels = list(cfg["channels"])
    meta_path = dest.with_name("panel_meta.json")
    if dest.is_file() and not force and not spec.get("rebuild"):
        return {"ok": True, "reused": True, "path": str(dest)}
    kind = str(spec.get("kind", "existing"))
    if kind == "slice":
        src = Path(spec["src"])
        cd = causaldiscovery_root(cfg)
        if not src.is_absolute():
            src = cd / src
        meta_src = Path(spec["src_meta"])
        if not meta_src.is_absolute():
            meta_src = cd / meta_src
        src_channels = list(json.loads(meta_src.read_text())["channels"])
        info = slice_panel(src, src_channels, channels, dest)
    elif kind == "pickles":
        splits = cfg.get("splits") or {}
        info = bin_from_pickles(
            data_root(cfg),
            channels,
            official_train_end=str(splits.get("official_train_end", "2007-01-01")),
            bin_seconds=int(cfg.get("bin_seconds", 30)),
            dest=dest,
        )
    else:
        src = panel_path(cfg)
        if not src.is_file():
            raise FileNotFoundError(src)
        if src.resolve() != dest.resolve():
            dest.write_bytes(src.read_bytes())
        info = {"source": str(src), "copied": True}
    info.update({"ok": True, "reused": False, "path": str(dest), "channels": channels})
    write_panel_meta(meta_path, info)
    return info


def _ensure_s0(cfg: dict[str, Any], s0: Path, *, force: bool) -> dict[str, Any]:
    summary = s0 / "summary.json"
    if summary.is_file() and (s0 / "labeled_arrays.npz").is_file() and not force:
        return json.loads(summary.read_text())
    from anogen.phases.s0 import run_s0

    return run_s0(cfg)


def _ensure_denoiser(
    cfg: dict[str, Any],
    s0: Path,
    s1: Path,
    panel: Any,
    train: pd.DataFrame,
    width: int,
    *,
    force: bool,
) -> dict[str, Any]:
    from anogen.shell.diffusion import train_denoiser
    import torch

    ckpt = s1 / "denoiser.pt"
    if ckpt.is_file() and not force:
        return json.loads((s1 / "summary.json").read_text()) if (s1 / "summary.json").is_file() else {"reused": True}
    x = materialize(panel, train, width)
    ch = train["channel_idx"].to_numpy(dtype=np.int64)
    dcfg = dict((cfg.get("shell") or {}).get("diffusion") or {})
    scfg = dict((cfg.get("shell") or {}).get("scaler") or {})
    scaler = None
    if str(scfg.get("kind", "minmax")).lower() == "minmax":
        fr = tuple(scfg.get("feature_range") or (0.0, 1.0))
        scaler = fit_channel_minmax(x, ch, panel.k, feature_range=(float(fr[0]), float(fr[1])))
        scaler.save(s0 / "minmax_scaler.npz")
    trained = train_denoiser(
        x,
        ch,
        n_channels=panel.k,
        hidden=int(dcfg.get("hidden", 64)),
        n_times=int(dcfg.get("n_times", 200)),
        steps=int(dcfg.get("steps", 4000)),
        batch_size=int(dcfg.get("batch_size", 128)),
        lr=float(dcfg.get("lr", 2e-4)),
        seed=int(cfg.get("seed", 0)),
        val_frac=float(dcfg.get("val_frac", 0.1)),
        backbone=str(dcfg.get("backbone", "tsdiff")),
        n_layers=int(dcfg.get("n_layers", 6)),
        d_state=int(dcfg.get("d_state", 64)),
        scaler=scaler,
    )
    model = trained.pop("model")
    trained.pop("schedule", None)
    blob = {
        "state_dict": model.state_dict(),
        "backbone": str(dcfg.get("backbone", "tsdiff")),
        "hidden": int(dcfg.get("hidden", 64)),
        "n_layers": int(dcfg.get("n_layers", 6)),
        "d_state": int(dcfg.get("d_state", 64)),
        "n_channels": panel.k,
        "n_times": int(dcfg.get("n_times", 200)),
        "W": width,
        "channels": list(panel.channels),
    }
    if scaler is not None:
        blob.update(scaler.to_ckpt())
    torch.save(blob, ckpt)
    report = {
        "ok": True,
        "steps": int(dcfg.get("steps", 4000)),
        "n_windows": int(len(x)),
        "train_loss": trained.get("train_loss"),
        "val_mse": trained.get("val_mse"),
    }
    (s1 / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return report


def _ensure_galleries(
    cfg: dict[str, Any],
    s0: Path,
    s3: Path,
    panel: Any,
    train: pd.DataFrame,
    nominal: pd.DataFrame,
    width: int,
    n_gen: int,
    *,
    force: bool,
) -> dict[str, Any]:
    from anogen.shell.genias import genias_sample, train_genias
    import torch

    gal_path = s3 / "galleries.npz"
    if gal_path.is_file() and not force:
        return json.loads((s3 / "summary.json").read_text()) if (s3 / "summary.json").is_file() else {"reused": True}
    x_all = materialize(panel, train, width)
    x_nom = materialize(panel, nominal, width)
    ch_nom = nominal["channel_idx"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    starts, chs = [], []
    replaced: list[str] = []
    for cidx, ch_name in enumerate(panel.channels):
        idx = np.where(ch_nom == cidx)[0]
        if len(idx) == 0:
            raise RuntimeError(f"no nominal windows for {ch_name}")
        use_replace = len(idx) < n_gen
        if use_replace:
            replaced.append(f"{ch_name}:{len(idx)}")
        pick = rng.choice(idx, size=n_gen, replace=use_replace)
        starts.append(pick)
        chs.append(np.full(n_gen, cidx))
    cond_idx = np.concatenate(starts)
    cond_ch = np.concatenate(chs)
    x_cond = x_nom[cond_idx]
    gcfg = dict((cfg.get("shell") or {}).get("genias") or {})
    gia = train_genias(
        x_all,
        hidden=int(gcfg.get("hidden", 48)),
        latent=int(gcfg.get("latent", 16)),
        steps=int(gcfg.get("steps", 2500)),
        batch_size=int(gcfg.get("batch_size", 128)),
        lr=float(gcfg.get("lr", 1e-3)),
        kl_w=float(gcfg.get("kl_w", 0.01)),
        seed=int(cfg.get("seed", 0)),
    )
    psi_main = float(gcfg.get("psi", 2.0))
    patch_tau = float(gcfg.get("patch_tau", 0.2))
    genias_x = genias_sample(
        gia["model"], x_cond, psi=psi_main, device=gia["device"], patch_tau=patch_tau
    )
    extras = {}
    for psi in gcfg.get("psi_extra") or [1.0]:
        extras[f"genias_psi{psi}"] = genias_sample(
            gia["model"], x_cond, psi=float(psi), device=gia["device"], patch_tau=patch_tau
        )
    post = posthoc_inject(x_cond, rng=rng)
    np.savez_compressed(
        gal_path,
        posthoc=post,
        genias=genias_x,
        channel_idx=cond_ch,
        cond=x_cond,
        **extras,
    )
    torch.save(
        {
            "state_dict": gia["model"].state_dict(),
            "hidden": int(gcfg.get("hidden", 48)),
            "latent": int(gcfg.get("latent", 16)),
            "steps": int(gcfg.get("steps", 2500)),
            "arch": "spatial_tcn_vae",
        },
        s3 / "genias.pt",
    )
    report = {
        "ok": True,
        "steps": int(gcfg.get("steps", 2500)),
        "n_total": int(len(x_cond)),
        "genias_loss": gia.get("loss"),
        "psi": psi_main,
        "donor_replace": replaced,
    }
    (s3 / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return report


def _ensure_encoder(
    cfg: dict[str, Any],
    s0: Path,
    enc_dir: Path,
    panel: Any,
    train: pd.DataFrame,
    width: int,
    x_cond: np.ndarray,
    device: str,
    *,
    force: bool,
):
    import torch
    from anogen.shell.encoders import ShellEncoder

    ckpt = enc_dir / "time_recon.pt"
    scfg = dict((cfg.get("shell") or {}).get("steer") or {})
    ecfg = dict((cfg.get("shell") or {}).get("enc_ablation") or {})
    if ckpt.is_file() and not force:
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        enc = ShellEncoder(
            int(blob["width"]),
            hidden=int(blob["hidden"]),
            emb=int(blob["emb"]),
            time_emb=int(blob.get("time_emb", 8)),
            pool=str(blob.get("pool", "time")),
        )
        enc.load_state_dict(blob["state_dict"])
        enc.to(torch.device(device))
        enc.eval()
    else:
        nom_tab = train.loc[~train["is_rare"].astype(bool)]
        x_nom = materialize(panel, nom_tab, width)
        lab = np.load(s0 / "labeled_arrays.npz")
        x_rare = np.asarray(lab["rare"])
        x_fit = np.concatenate([x_nom, x_rare], axis=0) if len(x_rare) else x_nom
        trained = train_shell_encoder(
            x_fit,
            pool="time",
            loss="recon",
            hidden=int(scfg.get("enc_hidden", 32)),
            emb=int(scfg.get("emb", 32)),
            time_emb=int(ecfg.get("time_emb", 8)),
            steps=int(ecfg.get("steps_recon", 400)),
            batch_size=int(ecfg.get("batch_size", 64)),
            lr=float(scfg.get("enc_lr", 1e-3)),
            device=device,
        )
        enc = trained["encoder"]
        torch.save(
            {
                "state_dict": enc.state_dict(),
                "width": width,
                "hidden": int(scfg.get("enc_hidden", 32)),
                "emb": int(scfg.get("emb", 32)),
                "time_emb": int(ecfg.get("time_emb", 8)),
                "pool": "time",
                "name": "time_recon",
            },
            ckpt,
        )
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    n_ref = min(int(scfg.get("n_ref", 512)), len(x_cond))
    ref_idx = rng.choice(len(x_cond), size=n_ref, replace=False)
    val_idx = rng.choice(len(x_cond), size=min(256, len(x_cond)), replace=False)
    shell = shell_from_nominal(
        enc,
        x_cond[ref_idx],
        x_cond[val_idx],
        q=float(scfg.get("q", 0.99)),
        delta_scale=float(scfg.get("delta_scale", 0.25)),
        device=device,
    )
    return enc, shell


def _ensure_unguided(
    score_dir: Path,
    model: Any,
    schedule: Any,
    x_cond: np.ndarray,
    cond_ch: np.ndarray,
    *,
    nu: float,
    ddim_steps: int,
    device: str,
    scaler: Any,
    force: bool,
) -> np.ndarray:
    from anogen.shell.diffusion import unguided_from_nominal

    path = score_dir / "unguided.npz"
    if path.is_file() and not force:
        return np.asarray(np.load(path)["x"])
    x = unguided_from_nominal(
        model, schedule, x_cond, cond_ch, nu=nu, ddim_steps=ddim_steps, device=device, scaler=scaler
    )
    np.savez_compressed(path, x=x, channel_idx=cond_ch)
    return x


def _ensure_band(
    score_dir: Path,
    model: Any,
    enc: Any,
    x_cond: np.ndarray,
    cond_ch: np.ndarray,
    schedule: Any,
    common: dict[str, Any],
    *,
    force: bool,
) -> tuple[np.ndarray, np.ndarray]:
    path = score_dir / "time_recon_band.npz"
    if path.is_file() and not force:
        blob = np.load(path)
        return np.asarray(blob["x"]), np.asarray(blob["h"])
    x, h = chunked_guided_ddim(model, enc, x_cond, cond_ch, schedule, **_BAND, **common)
    np.savez_compressed(path, x=x, h=h, channel_idx=cond_ch)
    return x, h


def _ensure_combined(
    score_dir: Path,
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
    force: bool,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    import torch

    folds = sorted({int(f) for f in fold_a.tolist() if int(f) >= 0})
    rows = []
    x0 = h0 = None
    for fold_id in folds:
        path = score_dir / f"time_recon_combined_fold{fold_id}.npz"
        if path.is_file() and not force:
            blob = np.load(path)
            samples, h = np.asarray(blob["x"]), np.asarray(blob["h"])
        else:
            ia = x_a[fold_a != fold_id][:n_ref]
            ir = x_r[fold_r != fold_id][:n_ref]
            ra = torch.from_numpy(embed_shell(enc, ia, device)) if len(ia) else None
            rr = torch.from_numpy(embed_shell(enc, ir, device)) if len(ir) else None
            samples, h = chunked_guided_ddim(
                model,
                enc,
                x_cond,
                cond_ch,
                schedule,
                ref_anom=ra,
                ref_rare=rr,
                **_COMBINED,
                **{k: v for k, v in common.items() if k != "bsz"},
                bsz=common["bsz"],
            )
            np.savez_compressed(path, x=samples, h=h)
        rows.append({"fold": fold_id, "x": samples, "h": h})
        if fold_id == folds[0]:
            x0, h0 = samples, h
    assert x0 is not None and h0 is not None
    return rows, x0, h0


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
    channels: list[str],
    tau: float,
) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for name, gx in galleries.items():
        fold_rows = []
        for fold_id in sorted({int(f) for f in fold_a.tolist() if int(f) >= 0}):
            ka = keep_fold_mask(ch_a, fold_a, fold_tab, fold_id=fold_id, channels=channels)
            kr = keep_fold_mask(ch_r, fold_r, fold_tab, fold_id=fold_id, channels=channels)
            if int(ka.sum()) == 0:
                ka = fold_a == fold_id
            if int(ka.sum()) == 0:
                continue
            rare = x_r[kr] if int(kr.sum()) else np.zeros((0, x_a.shape[1]), dtype=x_a.dtype)
            scored = score_generator(x_a[ka], rare, gx, tau=tau)
            scored["fold"] = fold_id
            scored["n_query_anom"] = int(ka.sum())
            scored["n_query_rare"] = int(kr.sum())
            fold_rows.append(scored)
        methods[name] = _mean_folds(fold_rows, gx)
    return methods


def _score_combined(
    comb_folds: list[dict[str, Any]],
    *,
    x_a: np.ndarray,
    x_r: np.ndarray,
    fold_a: np.ndarray,
    fold_r: np.ndarray,
    ch_a: np.ndarray,
    ch_r: np.ndarray,
    fold_tab: pd.DataFrame,
    channels: list[str],
    tau: float,
    q_q: float,
    delta: float,
    gallery0: np.ndarray,
    h0: np.ndarray,
) -> dict[str, Any]:
    fold_rows = []
    for item in comb_folds:
        fold_id = int(item["fold"])
        ka = keep_fold_mask(ch_a, fold_a, fold_tab, fold_id=fold_id, channels=channels)
        kr = keep_fold_mask(ch_r, fold_r, fold_tab, fold_id=fold_id, channels=channels)
        if int(ka.sum()) == 0:
            ka = fold_a == fold_id
        if int(ka.sum()) == 0:
            continue
        rare = x_r[kr] if int(kr.sum()) else np.zeros((0, x_a.shape[1]), dtype=x_a.dtype)
        scored = score_generator(x_a[ka], rare, item["x"], tau=tau)
        scored["fold"] = fold_id
        scored["occupancy"] = float(np.mean(np.abs(item["h"] - q_q) <= delta))
        scored["n_query_anom"] = int(ka.sum())
        scored["n_query_rare"] = int(kr.sum())
        fold_rows.append(scored)
    out = _mean_folds(fold_rows, gallery0)
    out["occupancy"] = float(np.mean([r["occupancy"] for r in fold_rows])) if fold_rows else float("nan")
    out["energy"] = band_report(h0, q_q, delta)
    out["shot"] = "FS"
    return out


def _mean_folds(fold_rows: list[dict[str, Any]], gallery: np.ndarray) -> dict[str, Any]:
    if not fold_rows:
        return {
            "coverage_anomaly": 0.0,
            "coverage_rare": 0.0,
            "gap": 0.0,
            "arp_anomaly": 0.0,
            "arp_rare": 0.0,
            "mean_min_d_anomaly": float("inf"),
            "mean_min_d_rare": float("inf"),
            "diversity": mean_pairwise_distance(embed_windows(gallery)),
            "folds": [],
            "n": int(len(gallery)),
        }
    keys = (
        "coverage_anomaly",
        "coverage_rare",
        "gap",
        "arp_anomaly",
        "arp_rare",
        "mean_min_d_anomaly",
        "mean_min_d_rare",
    )
    out = {k: float(np.mean([r[k] for r in fold_rows])) for k in keys}
    out["diversity"] = mean_pairwise_distance(embed_windows(gallery))
    out["folds"] = fold_rows
    out["n"] = int(len(gallery))
    return out


def _write_plots(
    plots: Path,
    *,
    parent: np.ndarray,
    unguided: np.ndarray,
    posthoc: np.ndarray,
    genias: np.ndarray,
    band: np.ndarray,
    combined: np.ndarray,
    real_anom: np.ndarray,
    methods: dict[str, Any],
    bin_seconds: int,
    name: str,
) -> None:
    from anogen.shell.plots import save_bar, save_compare_rows, save_same_parent

    n = min(6, len(parent), len(unguided), len(posthoc), len(genias), len(band), len(combined))
    if n:
        save_same_parent(
            plots / "same_parent.png",
            parent[:n],
            {
                "unguided ν=1": unguided[:n],
                "post-hoc": posthoc[:n],
                "GenIAS ψ=2": genias[:n],
                "time_recon band": band[:n],
                "time_recon combined": combined[:n],
            },
            bin_seconds=bin_seconds,
        )
    rows = {
        "real anomaly": real_anom[:8] if len(real_anom) else parent[:1],
        "unguided": unguided[:8],
        "post-hoc": posthoc[:8],
        "GenIAS ψ=2": genias[:8],
        "time_recon band": band[:8],
        "time_recon combined": combined[:8],
    }
    save_compare_rows(
        plots / "real_vs_generated.png",
        rows,
        bin_seconds=bin_seconds,
        n_cols=5,
        title=f"{name}: real vs generated (independent columns)",
    )
    labels = list(_METHODS)
    save_bar(
        plots / "coverage.png",
        labels,
        [float(methods[k]["coverage_anomaly"]) for k in labels],
        title=f"{name} Coverage@τ (this dataset)",
        ylabel="coverage",
    )
    save_bar(
        plots / "arp.png",
        labels,
        [float(methods[k]["arp_anomaly"]) for k in labels],
        title=f"{name} ARP (anomaly)",
        ylabel="ARP",
    )
    (plots / "INDEX.txt").write_text(
        "\n".join(
            [
                f"xfer plots for {name}",
                "same_parent.png — one nominal parent per column",
                "real_vs_generated.png — independent examples",
                "coverage.png / arp.png — this dataset only (not locked S4)",
                "",
            ]
        )
    )


def _write_methods_csv(path: Path, methods: dict[str, Any], tau: float) -> None:
    rows = []
    for name in _METHODS:
        if name not in methods:
            continue
        m = methods[name]
        rows.append(
            {
                "method": name,
                "tau": tau,
                "coverage_anomaly": m.get("coverage_anomaly"),
                "coverage_rare": m.get("coverage_rare"),
                "gap": m.get("gap"),
                "arp_anomaly": m.get("arp_anomaly"),
                "arp_rare": m.get("arp_rare"),
                "diversity": m.get("diversity"),
                "edi": m.get("edi"),
                "occupancy": m.get("occupancy"),
                "shot": m.get("shot"),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


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
