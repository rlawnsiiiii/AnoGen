"""Parent time-leash: μ(x̂_0 − x_par) during hybrid_needles DDIM.

Isolated results/shell_timeleash/. Does not overwrite S3/S4 or the
hybrid_needles freeze. ARP: frozen S4 τ. EDI is a new union (footnote t).
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
from anogen.phases.kindmix import HYBRID_NEEDLES_PROTO_KINDS, _PARENT50, _kind_alloc_labels, _stratified
from anogen.phases.plot_tune import _real_kind_tables
from anogen.phases.tune import _score_gallery
from anogen.shell.coverage import edi_by_method, mean_pairwise_distance
from anogen.shell.diffusion import DiffusionSchedule, denoiser_from_ckpt, torch_available
from anogen.shell.encoders import embed_shell
from anogen.shell.events import types_from_cfg
from anogen.shell.features import embed_windows
from anogen.shell.morphology import KIND_ORDER, KIND_SLUG, kind_ref_indices, kinds_for_windows
from anogen.shell.plots import save_compare_rows, save_same_parent, save_strip
from anogen.shell.scaler import resolve_scaler
from anogen.shell.steer import band_report

SHORT = {
    "real ESA Point / Global": "Point/Global",
    "real ESA local subsequence": "local subseq",
    "real level shift": "level shift",
    "real ESA global subsequence": "global subseq",
}
COLORS = {
    "real ESA Point / Global": "#2166ac",
    "real ESA local subsequence": "#1b9e77",
    "real level shift": "#d95f02",
    "real ESA global subsequence": "#7570b3",
}
DEFAULT_RECIPES: dict[str, dict[str, Any]] = {
    "l2": {"lam_parent": 0.03, "parent_leash": "l2", "parent_delta": 0.03},
    "gm": {"lam_parent": 0.15, "parent_leash": "gm", "parent_delta": 0.03},
}


def assert_isolated(out: Path, *locked: Path) -> None:
    if out.resolve() in {p.resolve() for p in locked}:
        raise RuntimeError("timeleash must not write into shell_s3, shell_s4, or the hybrid freeze")


def run_timeleash(cfg: dict[str, Any]) -> dict[str, Any]:
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
    out = _abs(cfg.get("timeleash_dir", root / "results/shell_timeleash"), root)
    plots = out / "plots"
    docs_plots = root / "docs" / "timeleash"
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

    tcfg = dict((cfg.get("shell") or {}).get("timeleash") or {})
    recipe_names = tuple(tcfg.get("recipes") or ("l2", "gm"))
    recipes = {k: dict(DEFAULT_RECIPES[k]) for k in recipe_names if k in DEFAULT_RECIPES}
    for name, spec in recipes.items():
        override = dict(tcfg.get(name) or {})
        spec.update(override)
    encoders = tuple(tcfg.get("encoders") or ("time_both",))
    force = bool(tcfg.get("force", False))
    seed = int(cfg.get("seed", 0))
    bin_seconds = int(cfg.get("bin_seconds", 30))

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
    n = min(int(tcfg.get("n", len(x_cond))), len(x_cond))
    x0, ch0 = x_cond[:n], cond_ch[:n]
    anom = meta[meta["kind"] == "anomaly"].reset_index(drop=True)
    types = types_from_cfg(cfg)
    kind_lab = kinds_for_windows(x_a, anom, types)
    examples = _real_kind_tables(x_a, meta, types)
    folds = sorted({int(f) for f in fold_a.tolist() if int(f) >= 0})
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
    schedule = DiffusionSchedule.from_ckpt(den).to(device_t)
    scfg = dict((cfg.get("shell") or {}).get("steer") or {})
    bsz = int(scfg.get("n_sample", 128))

    methods: dict[str, Any] = {}
    fold0: dict[str, dict[str, np.ndarray]] = {}
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
            **_PARENT50,
        )
        for tag, spec in recipes.items():
            key = f"{enc_name}_{tag}"
            print(f"timeleash {key} n={n}×{len(folds)}", flush=True)
            scored, f0 = _oof(
                model,
                enc,
                x0,
                ch0,
                schedule,
                common={**common, **spec},
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
            scored["parent_leash"] = spec
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
        "hybrid_needles parent50 + time-domain donor leash (not unit-∇). "
        "l2: μ(x̂_0−x_par). gm: saturating pull so shelves/needles can stay. "
        "ARP: frozen S4 τ / φ. EDI t is a new union. Does not overwrite "
        "shell_s3, shell_s4, or shell_kindmix_score_hybrid."
    )
    index = _index_text(written, methods, recipes, n, note)
    (docs_plots / "INDEX.txt").write_text(index)
    (out / "INDEX.txt").write_text(index)
    report = {
        "ok": True,
        "skipped": False,
        "tau": tau,
        "tau_source": "s4_frozen",
        "n_donors": int(n),
        "encoders": list(encoders),
        "recipes": recipes,
        "methods": {k: _brief(v) for k, v in methods.items()},
        "edi_table_union": {
            "k": 16,
            "n_per_gallery": int(n),
            "marker": "t",
            "values": edi_table,
            "note": (
                "GenIAS App. E.2 EDI on S4 + hybrid_needles fold-0 + timeleash fold-0. "
                "New partition — do not mix with locked S4, hybrid *, editor e, "
                "stratified ‡, from-noise ¶, or genbase †."
            ),
        },
        "plots": written,
        "note": note,
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return report


def _oof(
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
    force: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    fold_rows = []
    chunks = []
    q_q, delta = float(shell["Q_q"]), float(shell["delta"])
    fold0: dict[str, np.ndarray] = {}
    for fold_id in sorted({int(f) for f in fold_a.tolist() if int(f) >= 0}):
        kind_z: dict[str, torch.Tensor] = {}
        for kind in KIND_ORDER:
            idx, _leaked = kind_ref_indices(kind_lab, fold_a, kind, query_fold=fold_id)
            if len(idx) == 0:
                continue
            kind_z[kind] = torch.from_numpy(embed_shell(enc, x_a[idx], device))
        active = [k for k in KIND_ORDER if k in kind_z]
        cache = out / f"{variant}_fold{fold_id}.npz"
        if cache.is_file() and not force:
            blob = np.load(cache, allow_pickle=True)
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
                proto_kinds=HYBRID_NEEDLES_PROTO_KINDS,
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
        fold_rows.append(scored)
        if fold_id == 0:
            fold0 = {"x": samples, "alloc": np.asarray(alloc), "h": h}
    return (
        {
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
        },
        fold0,
    )


def _edi_union(
    *,
    s4: Path,
    gal: Any,
    hybrid: Path,
    out: Path,
    keys: list[str],
) -> dict[str, float]:
    s4g = np.load(s4 / "shell_gallery.npz")
    embs: dict[str, np.ndarray] = {
        "shell": embed_windows(np.asarray(s4g["x"])),
        "unguided": embed_windows(np.asarray(s4g["unguided"])),
        "genias": embed_windows(np.asarray(gal["genias"])),
        "posthoc": embed_windows(np.asarray(gal["posthoc"])),
    }
    raw = hybrid / "time_both_hybrid_needles_fold0.npz"
    if raw.is_file():
        embs["time_both_hybrid_needles"] = embed_windows(np.asarray(np.load(raw)["x"]))
    for key in keys:
        path = out / f"{key}_fold0.npz"
        if path.is_file():
            embs[key] = embed_windows(np.asarray(np.load(path)["x"]))
    return edi_by_method(embs)


def _write_plots(
    out: Path,
    *,
    parent: np.ndarray,
    fold0: dict[str, dict[str, np.ndarray]],
    hybrid: Path,
    examples: dict[str, dict[str, np.ndarray]],
    bin_seconds: int,
    seed: int,
    recipes: dict[str, dict[str, Any]],
) -> list[str]:
    rng = np.random.default_rng(seed)
    written: list[str] = []
    raw_path = hybrid / "time_both_hybrid_needles_fold0.npz"
    raw = np.asarray(np.load(raw_path)["x"]) if raw_path.is_file() else None
    for key, blob in fold0.items():
        edited, alloc = blob["x"], blob["alloc"]
        if len(edited) != len(parent):
            continue
        overview: dict[str, np.ndarray] = {}
        for kind in KIND_ORDER:
            sl = np.flatnonzero(alloc == kind)
            if len(sl) == 0:
                continue
            slug = KIND_SLUG.get(kind, "kind")
            take = min(16, len(sl))
            pick = rng.choice(sl, size=take, replace=False)
            png = f"{key}_random16_{slug}.png"
            save_strip(
                out / png,
                edited,
                title=f"{key} {SHORT.get(kind, kind)} random {take}/{len(sl)}",
                color=COLORS.get(kind, "#1b9e77"),
                bin_seconds=bin_seconds,
                idx=pick,
            )
            written.append(png)
            overview[SHORT.get(kind, kind)] = edited[
                rng.choice(sl, size=min(8, len(sl)), replace=False)
            ]
            same = rng.choice(sl, size=min(4, len(sl)), replace=False)
            kids: dict[str, np.ndarray] = {}
            if raw is not None and len(raw) == len(parent):
                kids["hybrid_needles"] = raw[same]
            kids[key] = edited[same]
            save_same_parent(
                out / f"{key}_same_parent_{slug}.png",
                parent[same],
                kids,
                bin_seconds=bin_seconds,
            )
            written.append(f"{key}_same_parent_{slug}.png")
            rows: dict[str, np.ndarray] = {}
            if kind in examples:
                rows[kind] = examples[kind]["x"][:4]
            rows["donor"] = parent[pick[:4]]
            if raw is not None and len(raw) == len(parent):
                rows["hybrid_needles"] = raw[pick[:4]]
            rows[key] = edited[pick[:4]]
            save_compare_rows(
                out / f"{key}_vs_{slug}.png",
                rows,
                bin_seconds=bin_seconds,
                n_cols=4,
                colors={kind: "#b2182b", "donor": "#4d4d4d", "hybrid_needles": "#7570b3", key: "#1b9e77"},
                title=f"{kind}: real / donor / hybrid_needles / {key}",
            )
            written.append(f"{key}_vs_{slug}.png")
        if overview:
            spec = recipes.get(key.split("_", 2)[-1], {})
            ov = f"{key}_overview8.png"
            save_compare_rows(
                out / ov,
                overview,
                bin_seconds=bin_seconds,
                n_cols=8,
                colors={SHORT.get(k, k): COLORS.get(k, "#1b9e77") for k in KIND_ORDER},
                title=f"{key} random 8/kind · {spec}",
            )
            written.append(ov)
    return written


def _index_text(
    written: list[str],
    methods: dict[str, Any],
    recipes: dict[str, Any],
    n: int,
    note: str,
) -> str:
    lines = [
        "hybrid_needles + time-domain donor leash (not unit-∇).",
        f"n={n}  recipes={recipes}",
        note,
        "",
        "ARP / EDI t (new union; do not mix with * † ‡ ¶ e):",
        "",
    ]
    for name in (
        "time_both_gm",
        "time_both_l2",
        "time_recon_gm",
        "time_recon_l2",
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
        lines.append(f"  {name}: ARP={arp}  EDI_t={edi}  Cov={cov}  Div={div}")
    lines.append("")
    lines.extend(written)
    lines.append("")
    return "\n".join(lines)
