"""Hash-reduction knobs on locked hybrid_needles DDIM.

Three sampler changes, one recipe: (1) do not stamp ∇f onto the last x̂_0,
(2) 20 DDIM steps so ν=0.2 has no repeated times, (3) clip-in-box grads
instead of unit-∇. Isolated results/shell_hashfix/. Does not overwrite
S3/S4 or the hybrid_needles freeze.
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
from anogen.phases.kindmix import _PARENT50
from anogen.phases.plot_tune import _real_kind_tables
from anogen.phases.timeleash import _edi_union, _oof, _write_plots, assert_isolated
from anogen.shell.diffusion import DiffusionSchedule, denoiser_from_ckpt, torch_available
from anogen.shell.events import types_from_cfg
from anogen.shell.morphology import kinds_for_windows
from anogen.shell.scaler import resolve_scaler

HASHFIX = {
    **_PARENT50,
    "ddim_steps": 20,
    "normalize_grad": False,
    "apply_final_grad": False,
}


def run_hashfix(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s1 = _abs(cfg.get("s1_dir", root / "results/shell_s1"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    s4 = _abs(cfg.get("s4_dir", root / "results/shell_s4"), root)
    enc_dir = _abs(cfg.get("enc_dir", root / "results/shell_enc"), root)
    hybrid = _abs(
        cfg.get("kindmix_score_hybrid_dir", root / "results/shell_kindmix_score_hybrid"),
        root,
    )
    out = _abs(cfg.get("hashfix_dir", root / "results/shell_hashfix"), root)
    plots = out / "plots"
    docs_plots = root / "docs" / "hashfix"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    docs_plots.mkdir(parents=True, exist_ok=True)
    assert_isolated(out, s3, s4, hybrid)

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
        "folds": s0 / "channel_fold_counts.csv",
        "time_both": enc_dir / "time_both_fold0.pt",
    }
    for label, path in needed.items():
        if not path.is_file():
            report = {"ok": False, "skipped": True, "reason": f"{label} missing: {path}"}
            (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            return report

    hcfg = dict((cfg.get("shell") or {}).get("hashfix") or {})
    encoders = tuple(hcfg.get("encoders") or ("time_both",))
    force = bool(hcfg.get("force", False))
    seed = int(cfg.get("seed", 0))
    bin_seconds = int(cfg.get("bin_seconds", 30))
    spec = {
        "ddim_steps": int(hcfg.get("ddim_steps", HASHFIX["ddim_steps"])),
        "normalize_grad": bool(hcfg.get("normalize_grad", HASHFIX["normalize_grad"])),
        "apply_final_grad": bool(hcfg.get("apply_final_grad", HASHFIX["apply_final_grad"])),
    }

    s4_sum = json.loads((s4 / "summary.json").read_text())
    tau = float(s4_sum["tau"])
    lab = np.load(s0 / "labeled_arrays.npz")
    gal = np.load(s3 / "galleries.npz")
    fold_tab = pd.read_csv(s0 / "channel_fold_counts.csv")
    meta = pd.read_csv(s0 / "labeled_windows.csv")
    x_a = np.asarray(lab["anomaly"])
    x_r = np.asarray(lab["rare"])
    fold_a = np.asarray(lab["anomaly_fold"])
    fold_r = np.asarray(lab["rare_fold"])
    ch_a = np.asarray(lab["anomaly_channel"])
    ch_r = np.asarray(lab["rare_channel"])
    x_cond = np.asarray(gal["cond"])
    cond_ch = np.asarray(gal["channel_idx"])
    n = min(int(hcfg.get("n", len(x_cond))), len(x_cond))
    x0, ch0 = x_cond[:n], cond_ch[:n]
    anom = meta[meta["kind"] == "anomaly"].reset_index(drop=True)
    types = types_from_cfg(cfg)
    kind_lab = kinds_for_windows(x_a, anom, types)
    examples = _real_kind_tables(x_a, meta, types)
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    den = torch.load(s1 / "denoiser.pt", map_location=device, weights_only=False)
    model = denoiser_from_ckpt(den, device=device_t)
    scaler = resolve_scaler(den, s0)
    schedule = DiffusionSchedule.linear(int(den["n_times"])).to(device_t)
    scfg = dict((cfg.get("shell") or {}).get("steer") or {})
    bsz = int(scfg.get("n_sample", 128))

    methods: dict[str, Any] = {}
    fold0: dict[str, dict[str, np.ndarray]] = {}
    recipes = {"hashfix": spec}
    for enc_name in encoders:
        ckpt = enc_dir / f"{enc_name}_fold0.pt"
        if not ckpt.is_file():
            continue
        enc, shell = _load_encoder(ckpt, x_cond, scfg, device, seed)
        common = dict(
            ref=shell["ref"],
            Q_q=float(shell["Q_q"]),
            tau=float(shell["tau"]),
            c_max=float(scfg.get("c_max", 1.0)),
            device=device,
            scaler=scaler,
            bsz=bsz,
            **{**HASHFIX, **spec},
        )
        key = f"{enc_name}_hashfix"
        print(f"hashfix {key} n={n} steps={spec['ddim_steps']}", flush=True)
        scored, f0 = _oof(
            model,
            enc,
            x0,
            ch0,
            schedule,
            common=common,
            shell=shell,
            device=device,
            kind_lab=kind_lab,
            x_a=x_a,
            fold_a=fold_a,
            score_kw=score_kw,
            out=out,
            variant=key,
            seed=seed,
            force=force,
        )
        scored["encoder"] = enc_name
        scored["hashfix"] = spec
        methods[key] = scored
        fold0[key] = f0

    locked = dict(s4_sum.get("methods") or {})
    for name in ("shell", "unguided", "genias", "posthoc"):
        if name in locked:
            methods[name] = dict(locked[name])

    edi_table = _edi_union(s4=s4, gal=gal, hybrid=hybrid, out=out, keys=list(fold0))
    for name, val in edi_table.items():
        row = methods.setdefault(name, {})
        row["edi_table"] = val
        if name in fold0 or name.endswith("_hybrid_needles"):
            row["edi"] = val

    written = _write_plots(
        plots,
        parent=x0,
        fold0=fold0,
        hybrid=hybrid,
        examples=examples,
        bin_seconds=bin_seconds,
        seed=seed,
        recipes=recipes,
    )
    for png in written:
        shutil.copy2(plots / png, docs_plots / png)

    note = (
        "hybrid_needles parent50 with three hash knobs: apply_final_grad=False "
        "(return last Tweedie x̂_0), ddim_steps=20 (no repeated times at ν=0.2), "
        "normalize_grad=False (clip-in-box). ARP: frozen S4 τ / φ. EDI h is a "
        "new union. Does not overwrite shell_s3, shell_s4, or the hybrid freeze."
    )
    index = _index_text(written, methods, spec, n, note)
    (docs_plots / "INDEX.txt").write_text(index)
    (out / "INDEX.txt").write_text(index)
    report = {
        "ok": True,
        "skipped": False,
        "tau": tau,
        "tau_source": "s4_frozen",
        "n_donors": int(n),
        "encoders": list(encoders),
        "hashfix": spec,
        "methods": {k: _brief(v) for k, v in methods.items()},
        "edi_table_union": {
            "k": 16,
            "n_per_gallery": int(n),
            "marker": "h",
            "values": edi_table,
            "note": (
                "GenIAS App. E.2 EDI on S4 + hybrid_needles fold-0 + hashfix fold-0. "
                "New partition — do not mix with locked S4, hybrid *, editor e, "
                "timeleash t, stratified ‡, from-noise ¶, or genbase †."
            ),
        },
        "plots": written,
        "note": note,
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return report


def _index_text(
    written: list[str],
    methods: dict[str, Any],
    spec: dict[str, Any],
    n: int,
    note: str,
) -> str:
    lines = [
        "hybrid_needles + hash knobs (no final ∇ stamp, 20 DDIM, clip-in-box).",
        f"n={n}  spec={spec}",
        note,
        "",
        "ARP / EDI h (new union; do not mix with * † ‡ ¶ e t):",
        "",
    ]
    for name in (
        "time_both_hashfix",
        "time_recon_hashfix",
        "time_both_hybrid_needles",
        "unguided",
        "shell",
        "genias",
        "posthoc",
    ):
        row = methods.get(name) or {}
        if not row:
            continue
        arp = row.get("arp_anomaly")
        edi = row.get("edi_table", row.get("edi"))
        cov = row.get("coverage_anomaly")
        div = row.get("diversity")
        lines.append(f"  {name}: ARP={arp}  EDI_h={edi}  Cov={cov}  Div={div}")
    lines.append("")
    lines.extend(written)
    lines.append("")
    return "\n".join(lines)
