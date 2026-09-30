"""Post-hoc editor on locked hybrid_needles galleries.

Does not resample DDIM. Does not overwrite shell_s3 / shell_s4 / the
hybrid_needles freeze. ARP uses frozen S4 τ. EDI is a new union (footnote e).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anogen.config import REPO_ROOT
from anogen.phases.encscore import _abs, _brief, _jsonable
from anogen.phases.kindmix import HYBRID_NEEDLES_PROTO_KINDS, _kind_alloc_labels
from anogen.phases.plot_tune import _real_kind_tables
from anogen.phases.tune import _score_gallery
from anogen.shell.coverage import edi_by_method, mean_pairwise_distance
from anogen.shell.editor import DEFAULT_N_KEEP, DEFAULT_PATCH_TAU, apply_editor
from anogen.shell.events import types_from_cfg
from anogen.shell.features import embed_windows
from anogen.shell.morphology import KIND_ORDER, KIND_SLUG, kinds_for_windows
from anogen.shell.plots import save_compare_rows, save_same_parent, save_strip

ENCODERS = ("time_both", "time_recon")
RECIPE = "hybrid_needles"
SHORT = {
    "real ESA Point / Global": "Point/Global",
    "real ESA local subsequence": "local subseq",
    "real level shift": "level shift",
    "real ESA global subsequence": "global subseq",
}
COLORS = {
    "parent": "#4d4d4d",
    "hybrid_needles": "#7570b3",
    "editor": "#1b9e77",
    "real ESA Point / Global": "#2166ac",
    "real ESA local subsequence": "#1b9e77",
    "real level shift": "#d95f02",
    "real ESA global subsequence": "#7570b3",
}


def assert_isolated(out: Path, *locked: Path) -> None:
    if out.resolve() in {p.resolve() for p in locked}:
        raise RuntimeError("editor must not write into shell_s3, shell_s4, or the hybrid freeze")


def run_editor(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    s4 = _abs(cfg.get("s4_dir", root / "results/shell_s4"), root)
    src = _abs(
        cfg.get("kindmix_score_hybrid_dir", root / "results/shell_kindmix_score_hybrid"),
        root,
    )
    out = _abs(cfg.get("editor_dir", root / "results/shell_editor"), root)
    plots = out / "plots"
    docs_plots = root / "docs" / "editor"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    docs_plots.mkdir(parents=True, exist_ok=True)
    assert_isolated(out, s3, s4, src)

    needed = {
        "S0": s0 / "labeled_arrays.npz",
        "S3": s3 / "galleries.npz",
        "S4": s4 / "summary.json",
        "meta": s0 / "labeled_windows.csv",
        "folds": s0 / "channel_fold_counts.csv",
    }
    for label, path in needed.items():
        if not path.is_file():
            report = {"ok": False, "skipped": True, "reason": f"{label} missing: {path}"}
            (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            return report

    ecfg = dict((cfg.get("shell") or {}).get("editor") or {})
    n_keep = int(ecfg.get("n_keep", DEFAULT_N_KEEP))
    patch_tau = float(ecfg.get("patch_tau", DEFAULT_PATCH_TAU))
    force = bool(ecfg.get("force", False))
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

    methods: dict[str, Any] = {}
    fold0: dict[str, dict[str, np.ndarray]] = {}
    missing: list[str] = []
    for enc_name in ENCODERS:
        raw_rows = []
        edit_rows = []
        raw_chunks = []
        edit_chunks = []
        for fold_id in folds:
            cache = src / f"{enc_name}_{RECIPE}_fold{fold_id}.npz"
            if not cache.is_file():
                missing.append(str(cache))
                continue
            blob = np.load(cache, allow_pickle=True)
            raw = np.asarray(blob["x"])
            if len(raw) != len(x_cond):
                report = {
                    "ok": False,
                    "skipped": True,
                    "reason": f"{cache} n={len(raw)} != donors {len(x_cond)}",
                }
                (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
                return report
            alloc = (
                np.asarray(blob["kind_alloc"], dtype=object)
                if "kind_alloc" in blob.files
                else _kind_alloc_labels(len(raw), [k for k in KIND_ORDER if k in set(kind_lab.tolist())])
            )
            edited_path = out / f"{enc_name}_{RECIPE}_editor_fold{fold_id}.npz"
            if edited_path.is_file() and not force:
                edited = np.asarray(np.load(edited_path)["x"])
            else:
                print(f"editor {enc_name} fold{fold_id} n={len(raw)}", flush=True)
                edited = apply_editor(
                    x_cond, raw, alloc, n_keep=n_keep, patch_tau=patch_tau
                )
                np.savez_compressed(
                    edited_path,
                    x=edited,
                    channel_idx=np.asarray(blob["channel_idx"]) if "channel_idx" in blob.files else cond_ch,
                    kind_alloc=np.asarray(alloc).astype(str),
                    n_keep=n_keep,
                    patch_tau=patch_tau,
                )
            raw_chunks.append(raw)
            edit_chunks.append(edited)
            raw_scored = _score_gallery(raw, only_fold=fold_id, **score_kw)
            raw_scored["fold"] = fold_id
            raw_rows.append(raw_scored)
            edit_scored = _score_gallery(edited, only_fold=fold_id, **score_kw)
            edit_scored["fold"] = fold_id
            edit_rows.append(edit_scored)
            if fold_id == 0:
                fold0[enc_name] = {"raw": raw, "edited": edited, "alloc": np.asarray(alloc)}
        if not edit_rows:
            continue
        methods[f"{enc_name}_{RECIPE}"] = _mean_fold_rows(raw_rows, raw_chunks[0])
        methods[f"{enc_name}_{RECIPE}_editor"] = _mean_fold_rows(edit_rows, edit_chunks[0])
        np.savez_compressed(
            out / f"{enc_name}_{RECIPE}_editor.npz", x=np.concatenate(edit_chunks, axis=0)
        )

    if missing and not methods:
        report = {"ok": False, "skipped": True, "reason": f"missing hybrid galleries: {missing[:4]}"}
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    locked = dict(s4_sum.get("methods") or {})
    for name in ("shell", "unguided", "genias", "posthoc"):
        if name in locked:
            methods[name] = dict(locked[name])

    edi_table = _edi_editor_union(s4=s4, gal=gal, src=src, out=out)
    for name, val in edi_table.items():
        row = methods.setdefault(name, {})
        row["edi_table"] = val
        if str(name).endswith("_editor") or name in (
            f"{ENCODERS[0]}_{RECIPE}",
            f"{ENCODERS[1]}_{RECIPE}",
        ):
            row["edi"] = val

    written = _write_plots(
        plots,
        parent=x_cond,
        fold0=fold0,
        examples=examples,
        bin_seconds=bin_seconds,
        seed=seed,
        n_keep=n_keep,
        patch_tau=patch_tau,
    )
    for png in written:
        shutil.copy2(plots / png, docs_plots / png)

    note = (
        "Editor on locked hybrid_needles (no new DDIM). Quiet + level-shift: "
        f"residual copy n_keep={n_keep}. Point/Global: deviation_patch τ={patch_tau}. "
        "ARP: frozen S4 τ / φ / 1536 donors / 3-fold. EDI e is a new union. "
        "Does not overwrite shell_s3, shell_s4, or shell_kindmix_score_hybrid."
    )
    index = _index_text(written, methods, n_keep, patch_tau, note)
    (docs_plots / "INDEX.txt").write_text(index)
    (out / "INDEX.txt").write_text(index)

    report = {
        "ok": True,
        "skipped": False,
        "tau": tau,
        "tau_source": "s4_frozen",
        "n_donors": int(len(x_cond)),
        "n_keep": n_keep,
        "patch_tau": patch_tau,
        "needle_kinds": sorted(HYBRID_NEEDLES_PROTO_KINDS & {"real ESA Point / Global"}),
        "methods": {k: _brief(v) for k, v in methods.items()},
        "edi_table_union": {
            "k": 16,
            "n_per_gallery": int(len(x_cond)),
            "marker": "e",
            "values": edi_table,
            "note": (
                "GenIAS App. E.2 EDI on S4 + hybrid_needles fold-0 + editor fold-0. "
                "New partition — do not mix with locked S4, encscore, hybrid *, "
                "stratified ‡, from-noise ¶, or genbase †."
            ),
        },
        "plots": written,
        "missing": missing,
        "note": note,
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return report


def _mean_fold_rows(rows: list[dict[str, Any]], fold0: np.ndarray) -> dict[str, Any]:
    keys = (
        "coverage_anomaly",
        "coverage_rare",
        "gap",
        "arp_anomaly",
        "arp_rare",
        "mean_min_d_anomaly",
        "mean_min_d_rare",
    )
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["diversity"] = mean_pairwise_distance(embed_windows(fold0))
    out["n"] = int(len(fold0))
    out["folds"] = rows
    return out


def _edi_editor_union(*, s4: Path, gal: Any, src: Path, out: Path) -> dict[str, float]:
    s4g = np.load(s4 / "shell_gallery.npz")
    embs: dict[str, np.ndarray] = {
        "shell": embed_windows(np.asarray(s4g["x"])),
        "unguided": embed_windows(np.asarray(s4g["unguided"])),
        "genias": embed_windows(np.asarray(gal["genias"])),
        "posthoc": embed_windows(np.asarray(gal["posthoc"])),
    }
    for enc_name in ENCODERS:
        raw_p = src / f"{enc_name}_{RECIPE}_fold0.npz"
        ed_p = out / f"{enc_name}_{RECIPE}_editor_fold0.npz"
        if raw_p.is_file():
            embs[f"{enc_name}_{RECIPE}"] = embed_windows(np.asarray(np.load(raw_p)["x"]))
        if ed_p.is_file():
            embs[f"{enc_name}_{RECIPE}_editor"] = embed_windows(np.asarray(np.load(ed_p)["x"]))
    return edi_by_method(embs)


def _write_plots(
    out: Path,
    *,
    parent: np.ndarray,
    fold0: dict[str, dict[str, np.ndarray]],
    examples: dict[str, dict[str, np.ndarray]],
    bin_seconds: int,
    seed: int,
    n_keep: int,
    patch_tau: float,
) -> list[str]:
    rng = np.random.default_rng(seed)
    written: list[str] = []
    for enc_name, blob in fold0.items():
        raw, edited, alloc = blob["raw"], blob["edited"], blob["alloc"]
        overview: dict[str, np.ndarray] = {}
        for kind in KIND_ORDER:
            sl = np.flatnonzero(alloc == kind)
            if len(sl) == 0:
                continue
            slug = KIND_SLUG.get(kind, "kind")
            take = min(16, len(sl))
            pick = rng.choice(sl, size=take, replace=False)
            png = f"{enc_name}_editor_random16_{slug}.png"
            save_strip(
                out / png,
                edited,
                title=f"{enc_name} editor {SHORT.get(kind, kind)} random {take}/{len(sl)}",
                color=COLORS.get(kind, "#1b9e77"),
                bin_seconds=bin_seconds,
                idx=pick,
            )
            written.append(png)
            overview[SHORT.get(kind, kind)] = edited[rng.choice(sl, size=min(8, len(sl)), replace=False)]
            n_same = min(4, len(sl))
            same = rng.choice(sl, size=n_same, replace=False)
            save_same_parent(
                out / f"{enc_name}_same_parent_{slug}.png",
                parent[same],
                {
                    "hybrid_needles": raw[same],
                    "editor": edited[same],
                },
                bin_seconds=bin_seconds,
            )
            written.append(f"{enc_name}_same_parent_{slug}.png")
            rows = {
                "donor": parent[pick[:4]] if take >= 4 else parent[pick],
                "hybrid_needles": raw[pick[:4]] if take >= 4 else raw[pick],
                "editor": edited[pick[:4]] if take >= 4 else edited[pick],
            }
            if kind in examples:
                rows = {kind: examples[kind]["x"][:4], **rows}
            save_compare_rows(
                out / f"{enc_name}_vs_{slug}.png",
                rows,
                bin_seconds=bin_seconds,
                n_cols=4,
                colors={
                    kind: "#b2182b",
                    "donor": COLORS["parent"],
                    "hybrid_needles": COLORS["hybrid_needles"],
                    "editor": COLORS["editor"],
                },
                title=f"{kind}: real / donor / hybrid_needles / editor ({enc_name})",
            )
            written.append(f"{enc_name}_vs_{slug}.png")
        if overview:
            ov = f"{enc_name}_editor_overview8.png"
            save_compare_rows(
                out / ov,
                overview,
                bin_seconds=bin_seconds,
                n_cols=8,
                colors={SHORT.get(k, k): COLORS.get(k, "#1b9e77") for k in KIND_ORDER},
                title=(
                    f"{enc_name} editor · random 8/kind · residual copy n_keep={n_keep}, "
                    f"needles patch τ={patch_tau}"
                ),
            )
            written.append(ov)
    return written


def _index_text(
    written: list[str],
    methods: dict[str, Any],
    n_keep: int,
    patch_tau: float,
    note: str,
) -> str:
    lines = [
        "Editor on hybrid_needles. Quiet + shift: residual copy. Point/Global: deviation-patch.",
        f"n_keep={n_keep}  patch_tau={patch_tau}",
        note,
        "",
        "ARP / EDI e (new union; do not mix with * † ‡ ¶):",
        "",
    ]
    for name in (
        "time_both_hybrid_needles_editor",
        "time_recon_hybrid_needles_editor",
        "time_both_hybrid_needles",
        "time_recon_hybrid_needles",
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
        lines.append(
            f"  {name}: ARP={arp}  EDI_e={edi}  Cov={cov}  Div={div}"
        )
    lines.append("")
    lines.extend(written)
    lines.append("")
    return "\n".join(lines)
