"""Score steered recipes from x_T ~ N(0, I) on the locked S4 protocol.

Same τ, φ, 1536 S3 channel mix, 3-fold OOF as encscore / kindmixscorehybrid.
start_from_noise=True (donor waveform ignored; channel index still conditions ε).
Isolated results/shell_noise_score/. Does not overwrite shell_s4, encscore,
or the ν=0.2 hybrid freeze.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from anogen.config import REPO_ROOT
from anogen.phases.encscore import _abs, _brief, _combined_oof, _jsonable, _load_encoder
from anogen.phases.kindmix import _PARENT50
from anogen.phases.kindmixscore import SCORE_RECIPES, _edi_recipe_union, _stratified_oof
from anogen.shell.diffusion import DiffusionSchedule, denoiser_from_ckpt, torch_available
from anogen.shell.events import types_from_cfg
from anogen.shell.morphology import kinds_for_windows
from anogen.shell.scaler import resolve_scaler

NOISE50 = {
    **_PARENT50,
    "nu": 1.0,
    "start_from_noise": True,
}

NOISE_RECIPES = ("hybrid", "hybrid_needles", "combined")


def run_noisescore(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s1 = _abs(cfg.get("s1_dir", root / "results/shell_s1"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    s4 = _abs(cfg.get("s4_dir", root / "results/shell_s4"), root)
    enc_dir = _abs(cfg.get("enc_dir", root / "results/shell_enc"), root)
    enc_score = _abs(cfg.get("enc_score_dir", root / "results/shell_enc_score"), root)
    out = _abs(cfg.get("noise_score_dir", root / "results/shell_noise_score"), root)
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
    tcfg = dict((cfg.get("shell") or {}).get("tune") or {})
    bsz = int(scfg.get("n_sample", 128))
    n_ref = int(tcfg.get("n_ref_label", 256))
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
            **NOISE50,
        )
        for recipe in ("hybrid", "hybrid_needles"):
            print(
                f"noisescore {variant} {recipe} from-noise OOF n={len(x_cond)}×3",
                flush=True,
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
                variant=variant,
                recipe=recipe,
                proto_kinds=SCORE_RECIPES[recipe],
                seed=seed,
            )
            scored["encoder"] = variant
            scored["recipe"] = recipe
            scored["shot"] = "FS"
            scored["start_from_noise"] = True
            scored["nu"] = 1.0
            methods[f"{variant}_{recipe}"] = scored

        print(
            f"noisescore {variant} combined from-noise OOF n={len(x_cond)}×3",
            flush=True,
        )
        combined = _combined_oof(
            model,
            enc,
            x_cond,
            cond_ch,
            schedule,
            common={k: v for k, v in common.items() if k not in NOISE50},
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
            sample_kw={"nu": 1.0, "start_from_noise": True},
            tag="combined",
        )
        combined["encoder"] = variant
        combined["recipe"] = "combined"
        combined["shot"] = "FS"
        combined["start_from_noise"] = True
        combined["nu"] = 1.0
        methods[f"{variant}_combined"] = combined

    locked = dict(s4_sum.get("methods") or {})
    for name in ("shell", "genias", "posthoc", "unguided"):
        if name in locked:
            methods[name] = dict(locked[name])

    edi_table = _edi_recipe_union(
        s4=s4,
        enc_out=enc_score,
        recipe_out=out,
        gal=gal,
        recipes=NOISE_RECIPES,
    )
    for name, val in edi_table.items():
        if name in methods:
            methods[name]["edi_table"] = val
            if any(str(name).endswith(suf) for suf in ("_hybrid", "_hybrid_needles", "_combined")):
                methods[name]["edi"] = val

    report = {
        "ok": True,
        "skipped": False,
        "tau": tau,
        "tau_source": "s4_frozen",
        "n_donors": int(len(x_cond)),
        "recipes": list(NOISE_RECIPES),
        "start_from_noise": True,
        "nu": 1.0,
        "methods": {k: _brief(v) for k, v in methods.items()},
        "edi_table_union": {
            "k": 16,
            "n_per_gallery": 1536,
            "values": edi_table,
            "note": (
                "GenIAS App. E.2 EDI on S4 + encscore + from-noise fold-0 galleries. "
                "New partition — do not mix with ν=0.2 hybrid *, stratified ‡, "
                "or encscore 9-method EDI."
            ),
        },
        "folds": {k: v.get("folds") for k, v in methods.items() if v.get("folds") is not None},
        "kind_scheme": "esa",
        "note": (
            "From-noise (start_from_noise=True, ν=1, x_T ~ N(0,I)) scored with "
            "frozen S4 τ and feature_pack_v1. Same 1536 channel mix and 3-fold OOF "
            "as kindmixscorehybrid / encscore. parent50 weights otherwise: "
            "λ=0.3, λ_anom=1, 50 DDIM, unit ∇. Combined also uses λ_rare=1. "
            "Fold-0 encoder only. Does not overwrite results/shell_s4, "
            "results/shell_enc_score, or results/shell_kindmix_score_hybrid/."
        ),
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return report
