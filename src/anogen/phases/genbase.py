"""Score CutAddPaste and Lai taxonomy on the locked S4 protocol.

Label-free operators, 1536 S3 donors, frozen τ / φ. Isolated
results/shell_genbase/. Does not overwrite shell_s3 or shell_s4.

EDI is a new union (footnote †). ARP for these two can look strong because
both paste or edit real telemetry, same as posthoc_inject.
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
from anogen.phases.plots import _head, _one_per_channel, _pick_mixed
from anogen.phases.s4 import _score_methods
from anogen.shell.baselines import (
    TAXONOMY_FAMILIES,
    cutaddpaste_inject,
    taxonomy_inject,
)
from anogen.shell.coverage import edi_by_method, mean_pairwise_distance
from anogen.shell.features import embed_windows
from anogen.shell.plots import save_compare_rows, save_overlay, save_same_parent, save_strip


def assert_isolated(out: Path, *locked: Path) -> None:
    if out.resolve() in {p.resolve() for p in locked}:
        raise RuntimeError("genbase must not write into shell_s3 or shell_s4")


def run_genbase(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    s4 = _abs(cfg.get("s4_dir", root / "results/shell_s4"), root)
    out = _abs(cfg.get("genbase_dir", root / "results/shell_genbase"), root)
    plots = out / "plots"
    docs_plots = root / "docs" / "genbase"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    docs_plots.mkdir(parents=True, exist_ok=True)
    assert_isolated(out, s3, s4)

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

    s4_sum = json.loads((s4 / "summary.json").read_text())
    tau = float(s4_sum["tau"])
    lab = np.load(s0 / "labeled_arrays.npz")
    gal = np.load(s3 / "galleries.npz")
    fold_tab = pd.read_csv(s0 / "channel_fold_counts.csv")
    x_a, x_r = np.asarray(lab["anomaly"]), np.asarray(lab["rare"])
    fold_a, fold_r = np.asarray(lab["anomaly_fold"]), np.asarray(lab["rare_fold"])
    ch_a, ch_r = np.asarray(lab["anomaly_channel"]), np.asarray(lab["rare_channel"])
    x_cond = np.asarray(gal["cond"])
    cond_ch = np.asarray(gal["channel_idx"])
    post = np.asarray(gal["posthoc"]) if "posthoc" in gal.files else None
    gia = np.asarray(gal["genias"]) if "genias" in gal.files else None

    gcfg = dict((cfg.get("shell") or {}).get("genbase") or {})
    cap_kw = dict(gcfg.get("cutaddpaste") or {})
    tax_kw = dict(gcfg.get("taxonomy") or {})
    seed = int(cfg.get("seed", 0))
    bin_seconds = int(cfg.get("bin_seconds", 30))

    cap_path = out / "cutaddpaste.npz"
    tax_path = out / "taxonomy.npz"
    if cap_path.is_file():
        cap = np.asarray(np.load(cap_path)["x"])
    else:
        print(f"genbase CutAddPaste n={len(x_cond)}", flush=True)
        cap = cutaddpaste_inject(
            x_cond,
            rng=np.random.default_rng(seed + 11),
            width_frac=tuple(cap_kw.get("width_frac") or (0.08, 0.35)),
            trend_scale=float(cap_kw.get("trend_scale", 0.5)),
            jitter_sigma=float(cap_kw.get("jitter_sigma", 0.05)),
            n_paste=int(cap_kw.get("n_paste", 1)),
            channel_idx=cond_ch,
        )
        np.savez_compressed(cap_path, x=cap, channel_idx=cond_ch)
    if tax_path.is_file():
        tax_blob = np.load(tax_path)
        tax = np.asarray(tax_blob["x"])
        tax_lab = np.asarray(tax_blob["family"]) if "family" in tax_blob.files else None
    else:
        print(f"genbase taxonomy n={len(x_cond)}", flush=True)
        fams = tuple(tax_kw.get("families") or TAXONOMY_FAMILIES)
        tax, tax_lab = taxonomy_inject(
            x_cond,
            rng=np.random.default_rng(seed + 17),
            families=fams,
            severity=tuple(tax_kw.get("severity") or (0.5, 2.0)),
            return_labels=True,
            channel_idx=cond_ch,
        )
        np.savez_compressed(tax_path, x=tax, family=tax_lab.astype(str), channel_idx=cond_ch)

    galleries: dict[str, np.ndarray] = {"cutaddpaste": cap, "taxonomy": tax}
    if post is not None:
        galleries["posthoc"] = post
    if gia is not None:
        galleries["genias"] = gia

    print("genbase scoring frozen τ", flush=True)
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
    locked = dict(s4_sum.get("methods") or {})
    for name in ("shell", "unguided"):
        if name in locked:
            methods[name] = dict(locked[name])

    edi_table = _edi_genbase_union(s4=s4, gal=gal, cap=cap, tax=tax)
    for name, val in edi_table.items():
        row = methods.setdefault(name, {})
        row["edi_table"] = val
        if name in ("cutaddpaste", "taxonomy"):
            row["edi"] = val
        elif name in ("posthoc", "genias"):
            # Keep ARP from this run (same galleries as S4). EDI is †.
            row["edi"] = val

    for name, gx in (("cutaddpaste", cap), ("taxonomy", tax)):
        methods[name]["diversity"] = mean_pairwise_distance(embed_windows(gx))
        methods[name]["n"] = int(len(gx))
        methods[name]["label_free"] = True

    written = _write_plots(
        plots,
        parent=x_cond,
        cond_ch=cond_ch,
        x_a=x_a,
        cap=cap,
        tax=tax,
        tax_lab=tax_lab,
        post=post,
        gia=gia,
        bin_seconds=bin_seconds,
        seed=seed,
    )
    for png in written:
        shutil.copy2(plots / png, docs_plots / png)

    index = _index_text(written, methods)
    (docs_plots / "INDEX.txt").write_text(index)
    (out / "INDEX.txt").write_text(index)

    report = {
        "ok": True,
        "skipped": False,
        "tau": tau,
        "tau_source": "s4_frozen",
        "n_donors": int(len(x_cond)),
        "methods": {k: _brief(v) for k, v in methods.items()},
        "edi_table_union": {
            "k": 16,
            "n_per_gallery": int(len(x_cond)),
            "marker": "†",
            "values": edi_table,
            "note": (
                "GenIAS App. E.2 EDI on S4 + CutAddPaste + taxonomy galleries. "
                "New partition — do not mix with locked S4 5-method, encscore "
                "unmarked, hybrid *, stratified ‡, or from-noise ¶ EDI."
            ),
        },
        "folds": {k: v.get("folds") for k, v in methods.items() if v.get("folds") is not None},
        "cutaddpaste": {
            "width_frac": list(cap_kw.get("width_frac") or (0.08, 0.35)),
            "trend_scale": float(cap_kw.get("trend_scale", 0.5)),
            "jitter_sigma": float(cap_kw.get("jitter_sigma", 0.05)),
            "n_paste": int(cap_kw.get("n_paste", 1)),
        },
        "taxonomy": {
            "families": list(tax_kw.get("families") or TAXONOMY_FAMILIES),
            "severity": list(tax_kw.get("severity") or (0.5, 2.0)),
            "counts": _family_counts(tax_lab),
        },
        "plots": written,
        "note": (
            "CutAddPaste (KDD 2024) and Lai 2021 taxonomy scored with frozen "
            "S4 τ and feature_pack_v1. 1536 S3 donors, 3-fold OOF queries, "
            "one gallery each (label-free). Both edit real nominal windows; "
            "ARP is structurally easy. Does not overwrite results/shell_s3 "
            "or results/shell_s4. posthoc_inject is unchanged."
        ),
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return report


def _family_counts(labels: np.ndarray | None) -> dict[str, int]:
    if labels is None or len(labels) == 0:
        return {}
    labs = np.asarray(labels).astype(str)
    return {k: int((labs == k).sum()) for k in TAXONOMY_FAMILIES}


def _edi_genbase_union(
    *,
    s4: Path,
    gal: Any,
    cap: np.ndarray,
    tax: np.ndarray,
) -> dict[str, float]:
    s4g = np.load(s4 / "shell_gallery.npz")
    embs: dict[str, np.ndarray] = {
        "shell": embed_windows(np.asarray(s4g["x"])),
        "unguided": embed_windows(np.asarray(s4g["unguided"])),
        "genias": embed_windows(np.asarray(gal["genias"])),
        "posthoc": embed_windows(np.asarray(gal["posthoc"])),
        "cutaddpaste": embed_windows(cap),
        "taxonomy": embed_windows(tax),
    }
    return edi_by_method(embs)


def _write_plots(
    out: Path,
    *,
    parent: np.ndarray,
    cond_ch: np.ndarray,
    x_a: np.ndarray,
    cap: np.ndarray,
    tax: np.ndarray,
    tax_lab: np.ndarray | None,
    post: np.ndarray | None,
    gia: np.ndarray | None,
    bin_seconds: int,
    seed: int,
) -> list[str]:
    rng = np.random.default_rng(seed)
    written: list[str] = []
    save_strip(
        out / "gen_cutaddpaste.png",
        cap,
        title="CutAddPaste (generated)",
        color="#1b9e77",
        bin_seconds=bin_seconds,
        idx=_pick_mixed(cap, 8, rng, extreme=True),
    )
    written.append("gen_cutaddpaste.png")
    save_strip(
        out / "gen_taxonomy.png",
        tax,
        title="Lai taxonomy (generated)",
        color="#d95f02",
        bin_seconds=bin_seconds,
        idx=_pick_mixed(tax, 8, rng, extreme=True),
    )
    written.append("gen_taxonomy.png")

    if tax_lab is not None:
        rows: dict[str, np.ndarray] = {}
        colors = {
            "point": "#e7298a",
            "contextual": "#7570b3",
            "shapelet": "#1b9e77",
            "seasonal": "#2166ac",
            "trend": "#d95f02",
        }
        for fam in TAXONOMY_FAMILIES:
            sl = tax[np.asarray(tax_lab).astype(str) == fam]
            if len(sl) == 0:
                continue
            idx = _pick_mixed(sl, 4, rng, extreme=True)
            rows[fam] = sl[idx]
        if rows:
            save_compare_rows(
                out / "gen_taxonomy_families.png",
                rows,
                bin_seconds=bin_seconds,
                n_cols=4,
                colors=colors,
                title="Lai taxonomy by family (largest-range mix)",
            )
            written.append("gen_taxonomy_families.png")

    compare = {
        "real anomaly": _head(x_a, _pick_mixed(x_a, 5, rng, extreme=True)),
        "nominal parent": _head(parent, _pick_mixed(parent, 5, rng, extreme=False)),
        "CutAddPaste": _head(cap, _pick_mixed(cap, 5, rng, extreme=True)),
        "Lai taxonomy": _head(tax, _pick_mixed(tax, 5, rng, extreme=True)),
    }
    colors = {
        "real anomaly": "#b2182b",
        "nominal parent": "#4d4d4d",
        "CutAddPaste": "#1b9e77",
        "Lai taxonomy": "#d95f02",
        "post-hoc": "#7570b3",
        "GenIAS ψ=2": "#e7298a",
    }
    if post is not None:
        compare["post-hoc"] = _head(post, _pick_mixed(post, 5, rng, extreme=True))
    if gia is not None:
        compare["GenIAS ψ=2"] = _head(gia, _pick_mixed(gia, 5, rng, extreme=True))
    save_compare_rows(
        out / "genbase_vs_real.png",
        compare,
        bin_seconds=bin_seconds,
        n_cols=5,
        colors=colors,
        title="Real vs generator baselines (independent examples)",
    )
    written.append("genbase_vs_real.png")

    cols = _one_per_channel(cond_ch, n_max=6) or list(range(min(4, len(parent))))
    kids = {"CutAddPaste": cap[cols], "Lai taxonomy": tax[cols]}
    if post is not None and len(post) > max(cols):
        kids = {"post-hoc": post[cols], **kids}
    if gia is not None and len(gia) > max(cols):
        kids["GenIAS ψ=2"] = gia[cols]
    save_same_parent(out / "genbase_same_parent.png", parent[cols], kids, bin_seconds=bin_seconds)
    written.append("genbase_same_parent.png")
    n_ov = min(4, len(cols))
    kids4 = {k: v[:n_ov] for k, v in kids.items()}
    save_overlay(out / "genbase_overlay.png", parent[cols[:n_ov]], kids4, bin_seconds=bin_seconds)
    written.append("genbase_overlay.png")
    return written


def _index_text(written: list[str], methods: dict[str, Any]) -> str:
    lines = [
        "genbase: CutAddPaste (KDD 2024) and Lai 2021 taxonomy.",
        "Same 1536 S3 donors and frozen S4 τ as every other protocol row.",
        "ARP is easy: both edit real nominal windows.",
        "",
    ]
    for name in ("cutaddpaste", "taxonomy", "posthoc", "genias"):
        m = methods.get(name) or {}
        if not m:
            continue
        lines.append(
            f"  {name}: ARP={m.get('arp_anomaly')} "
            f"EDI={m.get('edi')} Cov={m.get('coverage_anomaly')} "
            f"Div={m.get('diversity')}"
        )
    lines.append("")
    lines.extend(f"  {name}" for name in written)
    lines.append("")
    return "\n".join(lines)
