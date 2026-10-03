"""GenIAS trained with its published objective, as an honest baseline (P6 / L7).

Isolated ``results/shell_genias_fair/``. Does not touch ``results/shell_s3``
(the locked reconstruction-only GenIAS stays the S4 row).

Differences from S3's GenIAS, all from Darban et al. 2025 (arXiv:2502.08262):
learned ψ ≥ 1, triplet-margin perturbation loss (δ_min 0.1, δ_max 0.2, β 0.1),
compact KL (σ_prior 0.5, ζ 0.1), and Algorithm 2 patching per dimension
(whole window for univariate data). Because Alg. 2 compares units² with units,
τ is reported on a grid in both raw and per-window whitened units instead of
pretending one value transfers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anogen.config import REPO_ROOT, panel_path


def run_geniasfair(cfg: dict[str, Any]) -> dict[str, Any]:
    from anogen.phases.encscore import _abs, _jsonable

    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    s4 = _abs(cfg.get("s4_dir", root / "results/shell_s4"), root)
    gcfg = dict((cfg.get("shell") or {}).get("genias_fair") or {})
    out = _abs(gcfg.get("out", root / "results/shell_genias_fair"), root)
    if out.resolve() in {s0.resolve(), s3.resolve(), s4.resolve()}:
        raise RuntimeError("geniasfair must not write into a locked result directory")
    out.mkdir(parents=True, exist_ok=True)

    from anogen.shell.diffusion import torch_available

    if not torch_available():
        report = {"ok": False, "skipped": True, "reason": "torch is not installed"}
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    for label, path in {"S3": s3 / "galleries.npz", "S4": s4 / "summary.json", "S0": s0 / "labeled_arrays.npz"}.items():
        if not path.is_file():
            report = {"ok": False, "skipped": True, "reason": f"{label} missing: {path}"}
            (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            return report

    import torch

    from anogen.phases.fixsweep import _score_fold
    from anogen.phases.tune import _score_gallery
    from anogen.shell.data import load_panel
    from anogen.shell.evaluation import arp_coverage_ci
    from anogen.shell.events import types_from_cfg
    from anogen.shell.genias import genias_patch_alg2, genias_sample, train_genias
    from anogen.shell.morphology import KIND_ORDER, KIND_SLUG, kinds_for_windows
    from anogen.shell.scaler import load_minmax
    from anogen.shell.windows import load_train_index, materialize

    proto = json.loads((s0 / "protocol.json").read_text())
    width = int(proto["W"])
    splits = cfg.get("splits") or {}
    panel = load_panel(
        panel_path(cfg),
        channels=list(cfg.get("channels") or []),
        official_train_end=splits.get("official_train_end", "2007-01-01"),
        allow_test_telemetry=False,
        bin_seconds=int(cfg.get("bin_seconds", 30)),
    )
    x_all = materialize(panel, load_train_index(s0), width)
    gal = np.load(s3 / "galleries.npz")
    x_cond = np.asarray(gal["cond"])
    cond_ch = np.asarray(gal["channel_idx"], dtype=np.int64)
    base = dict((cfg.get("shell") or {}).get("genias") or {})
    ckpt = out / "genias_fair.pt"
    if ckpt.is_file() and not bool(gcfg.get("force", False)):
        from anogen.shell.genias import TCNAE

        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = TCNAE(hidden=int(blob["hidden"]), latent=int(blob["latent"]))
        model.load_state_dict(blob["state_dict"])
        model.eval()
        psi = float(blob["psi_learned"])
        train_info = blob.get("train_info", {})
    else:
        trained = train_genias(
            x_all,
            hidden=int(base.get("hidden", 48)),
            latent=int(gcfg.get("latent", base.get("latent", 16))),
            steps=int(gcfg.get("steps", base.get("steps", 15000))),
            batch_size=int(base.get("batch_size", 128)),
            lr=float(gcfg.get("lr", 1e-3)),
            seed=int(cfg.get("seed", 0)),
            faithful=True,
            beta=float(gcfg.get("beta", 0.1)),
            zeta=float(gcfg.get("zeta", 0.1)),
            sigma_prior=float(gcfg.get("sigma_prior", 0.5)),
            delta_min=float(gcfg.get("delta_min", 0.1)),
            delta_max=float(gcfg.get("delta_max", 0.2)),
            psi_init=float(gcfg.get("psi_init", 2.0)),
            kl_form=str(gcfg.get("kl_form", "paper")),
        )
        model = trained.pop("model")
        psi = float(trained["psi_learned"])
        train_info = {k: v for k, v in trained.items() if not isinstance(v, np.ndarray)}
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hidden": int(base.get("hidden", 48)),
                "latent": int(gcfg.get("latent", base.get("latent", 16))),
                "psi_learned": psi,
                "train_info": train_info,
            },
            ckpt,
        )
    torch.manual_seed(int(cfg.get("seed", 0)))
    raw = genias_sample(model, x_cond, psi=psi, patch_tau=0.0)
    galleries: dict[str, np.ndarray] = {"genias_fair": raw}
    patch_info: dict[str, Any] = {}
    for units in ("raw", "whitened"):
        for tau_p in gcfg.get(f"tau_{units}", [0.01, 0.1, 1.0] if units == "raw" else [1.0, 10.0, 50.0]):
            patched, replaced = genias_patch_alg2(x_cond, raw, float(tau_p), units=units)
            name = f"genias_fair_alg2_{units}_tau{tau_p:g}"
            galleries[name] = patched
            patch_info[name] = {"replaced_frac": float(replaced.mean())}
    np.savez_compressed(out / "galleries.npz", channel_idx=cond_ch, cond=x_cond, **galleries)

    tau = float(json.loads((s4 / "summary.json").read_text())["tau"])
    lab = np.load(s0 / "labeled_arrays.npz")
    meta = pd.read_csv(s0 / "labeled_windows.csv")
    anom = meta[meta["kind"] == "anomaly"].reset_index(drop=True)
    x_a = np.asarray(lab["anomaly"])
    kind_lab = kinds_for_windows(x_a, anom, types_from_cfg(cfg))
    score_kw = dict(
        x_a=x_a,
        x_r=np.asarray(lab["rare"]),
        fold_a=np.asarray(lab["anomaly_fold"]),
        fold_r=np.asarray(lab["rare_fold"]),
        ch_a=np.asarray(lab["anomaly_channel"]),
        ch_r=np.asarray(lab["rare_channel"]),
        fold_tab=pd.read_csv(s0 / "channel_fold_counts.csv"),
        tau=tau,
    )
    scaler = load_minmax(s0 / "minmax_scaler.npz")
    folds = sorted({int(f) for f in score_kw["fold_a"] if int(f) >= 0})
    scores: dict[str, Any] = {}
    for name, g in galleries.items():
        rows = [
            _score_fold(
                g,
                np.full(len(g), "all"),
                cond_ch,
                fold_id=f,
                score_kw=score_kw,
                event_a=anom["event_id"].astype(str).to_numpy(),
                kind_lab=kind_lab,
                scaler=scaler,
                n_boot=int(gcfg.get("n_boot", 1000)),
                seed=int(cfg.get("seed", 0)) + f,
                h=None,
                shell=None,
                cfg=cfg,
                arp_coverage_ci=arp_coverage_ci,
                score_gallery=_score_gallery,
                kind_order=KIND_ORDER,
                kind_slug=KIND_SLUG,
            )
            for f in folds
        ]
        scores[name] = {
            "arp_anomaly": float(np.mean([r["arp_anomaly"] for r in rows])),
            "coverage_anomaly": float(np.mean([r["coverage_anomaly"] for r in rows])),
            "rmse_to_parent": float(np.sqrt(np.mean((g - x_cond) ** 2))),
            "folds": rows,
            **patch_info.get(name, {}),
        }
    report = {
        "ok": True,
        "skipped": False,
        "psi_learned": psi,
        "train": train_info,
        "scores": scores,
        "note": (
            "Published GenIAS objective (learned psi, perturbation + compact KL) and "
            "Alg. 2 whole-window patching. Locked S3 GenIAS unchanged."
        ),
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return {"ok": True, "skipped": False, "dir": str(out), "psi_learned": psi}
