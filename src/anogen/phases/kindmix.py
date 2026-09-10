"""Mixed-gallery kinds: stratified proto, kind-balanced soft, two-recipe mix.

One gallery that can contain shelves *and* quiet crops. Isolated
results/shell_kindmix/. Does not overwrite shell_s4 / shell_hanom / shell_kindproto.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from anogen.config import REPO_ROOT
from anogen.phases.hanom import _abs, _brief, _jsonable, _kind_hits, _load
from anogen.phases.plot_tune import _plot_kinds_vs_generated, _real_kind_tables
from anogen.phases.tune import _score_gallery
from anogen.shell.coverage import mean_pairwise_distance
from anogen.shell.diffusion import DiffusionSchedule, denoiser_from_ckpt, torch_available
from anogen.shell.encoders import embed_shell
from anogen.shell.features import embed_windows
from anogen.shell.events import types_from_cfg
from anogen.shell.morphology import (
    KIND_ORDER,
    KIND_SLUG,
    kind_ref_indices,
    kinds_for_windows,
    rank_for_kind,
)
from anogen.shell.plots import save_compare_rows, save_named_grid
from anogen.shell.scaler import resolve_scaler
from anogen.shell.steer import band_report, chunked_guided_ddim

_PARENT50 = {
    "nu": 0.2,
    "lam": 0.3,
    "ddim_steps": 50,
    "normalize_grad": True,
    "n_correct": 1,
    "lam_anom": 1.0,
    "lam_rare": 0.0,
    "start_from_noise": False,
}

VARIANTS = ("stratified", "kind_soft", "twomix")
EXTRA_VARIANTS = ("proto", "band", "hybrid", "hybrid_needles")
ALL_VARIANTS = VARIANTS + EXTRA_VARIANTS
_PLOT_VARIANT_ORDER = (
    "band",
    "proto",
    "stratified",
    "hybrid",
    "hybrid_needles",
    "kind_soft",
    "twomix",
)
_SLICE_VARIANTS = ("stratified", "hybrid", "hybrid_needles")

# Proto only on kinds that need a structured departure from the parent.
HYBRID_PROTO_KINDS = frozenset({"real level shift"})
HYBRID_NEEDLES_PROTO_KINDS = frozenset({"real level shift", "real ESA Point / Global"})

_COLORS = {
    "stratified": "#d95f02",
    "kind_soft": "#7570b3",
    "twomix": "#1b9e77",
    "proto": "#2166ac",
    "band": "#4d4d4d",
    "hybrid": "#1b9e77",
    "hybrid_needles": "#7570b3",
}


def run_kindmix(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s1 = _abs(cfg.get("s1_dir", root / "results/shell_s1"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    s4 = _abs(cfg.get("s4_dir", root / "results/shell_s4"), root)
    enc_dir = _abs(cfg.get("enc_dir", root / "results/shell_enc"), root)
    out = _abs(cfg.get("kindmix_dir", root / "results/shell_kindmix"), root)
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    if not torch_available():
        report = {"ok": False, "skipped": True, "reason": "torch is not installed"}
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    needed = {
        "S0": s0 / "labeled_arrays.npz",
        "S1": s1 / "denoiser.pt",
        "S3": s3 / "galleries.npz",
        "S4": s4 / "summary.json",
        "time_recon": enc_dir / "time_recon_fold0.pt",
        "time_both": enc_dir / "time_both_fold0.pt",
        "meta": s0 / "labeled_windows.csv",
    }
    for label, path in needed.items():
        if not path.is_file():
            report = {"ok": False, "skipped": True, "reason": f"{label} missing: {path}"}
            (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            return report

    tau_cov = float(json.loads((s4 / "summary.json").read_text())["tau"])
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
    parent = np.asarray(gal["cond"])
    cond_ch = np.asarray(gal["channel_idx"])
    anom = meta[meta["kind"] == "anomaly"].reset_index(drop=True)
    types = types_from_cfg(cfg)
    kind_lab = kinds_for_windows(x_a, anom, types)
    examples = _real_kind_tables(x_a, meta, types)
    mcfg = dict((cfg.get("shell") or {}).get("kindmix") or {})
    n = min(int(mcfg.get("n", 256)), len(parent))
    x0 = parent[:n]
    ch = cond_ch[:n]
    bin_seconds = int(cfg.get("bin_seconds", 30))
    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    den = torch.load(s1 / "denoiser.pt", map_location=device, weights_only=False)
    model = denoiser_from_ckpt(den, device=device_t)
    scaler = resolve_scaler(den, s0)
    schedule = DiffusionSchedule.linear(int(den["n_times"])).to(device_t)
    scfg = dict((cfg.get("shell") or {}).get("steer") or {})
    bsz = int(scfg.get("n_sample", 64))

    score_kw = dict(
        x_a=x_a,
        x_r=x_r,
        fold_a=fold_a,
        fold_r=fold_r,
        ch_a=ch_a,
        ch_r=ch_r,
        fold_tab=fold_tab,
        tau=tau_cov,
    )

    ref_info: dict[str, Any] = {}
    kind_idx: dict[str, np.ndarray] = {}
    for kind in KIND_ORDER:
        idx, leaked = kind_ref_indices(kind_lab, fold_a, kind, query_fold=0)
        kind_idx[kind] = idx
        ref_info[kind] = {
            "n_all": int((kind_lab == kind).sum()),
            "n_ref": int(len(idx)),
            "n_oof": int(((kind_lab == kind) & (fold_a != 0)).sum()),
            "leaked": leaked,
        }

    methods: dict[str, Any] = {}
    galleries: dict[str, dict[str, np.ndarray]] = {}
    written: list[str] = []
    wanted = tuple(mcfg.get("variants") or VARIANTS)
    for vname in wanted:
        if vname not in ALL_VARIANTS:
            raise ValueError(f"unknown kindmix variant {vname}")
    active_kinds = [k for k in KIND_ORDER if len(kind_idx.get(k, ())) > 0]

    for enc_name in ("time_recon", "time_both"):
        enc, shell, ra_all, _rr = _load(
            enc_dir / f"{enc_name}_fold0.pt",
            x0,
            x_a,
            x_r,
            fold_a,
            fold_r,
            scfg,
            device,
            seed,
        )
        kind_z: dict[str, torch.Tensor] = {}
        for kind, idx in kind_idx.items():
            if len(idx) == 0:
                continue
            kind_z[kind] = torch.from_numpy(embed_shell(enc, x_a[idx], device))
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
        vgals: dict[str, np.ndarray] = {}
        for vname in wanted:
            cache = out / f"{enc_name}_{vname}.npz"
            if cache.is_file() and not bool(mcfg.get("force", False)):
                blob = np.load(cache)
                x = np.asarray(blob["x"])
                h = np.asarray(blob["h"])
                alloc = (
                    np.asarray(blob["kind_alloc"], dtype=object)
                    if "kind_alloc" in blob.files
                    else _kind_alloc_labels(len(x), active_kinds)
                )
            else:
                print(f"kindmix {enc_name} {vname} n={n}", flush=True)
                x, h = _generate(
                    vname,
                    model,
                    enc,
                    x0,
                    ch,
                    schedule,
                    common,
                    kind_z,
                    ra_all,
                    seed,
                )
                alloc = (
                    _kind_alloc_labels(len(x), active_kinds)
                    if vname in _SLICE_VARIANTS
                    else np.array([], dtype=object)
                )
                np.savez_compressed(
                    cache,
                    x=x,
                    h=h,
                    channel_idx=ch,
                    Q_q=float(shell["Q_q"]),
                    delta=float(shell["delta"]),
                    variant=np.asarray(vname),
                    kind_alloc=alloc.astype(str) if len(alloc) else np.array([], dtype=str),
                )
            vgals[vname] = x
            scored = _score_gallery(x, only_fold=0, **score_kw)
            scored["occupancy"] = float(np.mean(np.abs(h - shell["Q_q"]) <= shell["delta"]))
            scored["energy"] = band_report(h, float(shell["Q_q"]), float(shell["delta"]))
            scored["diversity"] = float(mean_pairwise_distance(embed_windows(x)))
            scored["kind_hits"] = _kind_hits(x)
            scored["coverage_kind"] = _coverage_by_kind(
                x, x_a, fold_a, ch_a, kind_lab, fold_tab, tau_cov, fold_id=0
            )
            if vname in _SLICE_VARIANTS and len(alloc):
                scored["slice_kind"] = _coverage_slices(
                    x, alloc, x_a, fold_a, ch_a, kind_lab, fold_tab, tau_cov, fold_id=0
                )
                written += _plot_kind_slices(
                    plots, enc_name, vname, x, alloc, examples, bin_seconds
                )
            scored["encoder"] = enc_name
            scored["variant"] = vname
            scored["n"] = int(len(x))
            scored["shot"] = "FS-mix"
            scored["kind_scheme"] = "esa"
            scored["note"] = (
                "Mixed-gallery kind experiment, fold-0 queries, frozen S4 τ. "
                "Not a new encscore freeze. λ_rare=0, parent50."
            )
            methods[f"{enc_name}_{vname}"] = scored
            written += _plot_kinds_vs_generated(
                plots,
                f"{enc_name}_{vname}",
                {"combined": x},
                examples,
                rng,
                bin_seconds,
                png_name=f"real_kinds_vs_{enc_name}_{vname}.png",
            )
        for extra in ALL_VARIANTS:
            if extra in vgals:
                continue
            sib = out / f"{enc_name}_{extra}.npz"
            if sib.is_file():
                vgals[extra] = np.asarray(np.load(sib)["x"])
        galleries[enc_name] = vgals
        written += _plot_variant_grid(plots, enc_name, vgals, examples, bin_seconds)
        written += _plot_subseq_compare(plots, enc_name, vgals, examples, bin_seconds)
        written += _plot_recipe_compare(plots, enc_name, vgals, examples, bin_seconds)
    written += _plot_both_encoders_compare(plots, galleries, examples, bin_seconds)

    (plots / "INDEX.txt").write_text(
        "Mixed galleries on Mission 1 41–46. Docs: docs/KIND_MIX.md.\n"
        "stratified = uniform kind, then proto from R_kind.\n"
        "hybrid = same slices; proto only on level shift, band on the rest.\n"
        "hybrid_needles = proto on level shift + Point/Global, band on subsequences.\n"
        "proto = pooled proto over all train-fold anomalies (not kind-conditional).\n"
        "band = λ_anom=0 (no proto / no h_anom).\n"
        "kind_soft = kind-balanced KDE (π_k = 1/K, mean within kind).\n"
        "twomix = 60% soft-all + 20% shift proto + 20% ESA Point/Global proto.\n"
        "Kinds: ESA Length × Locality + level-shift overlay (docs/ESA_LABELS.md).\n"
        "Plots: nearest in φ; aimed-slice vs real for stratified; "
        "allkinds_* = band vs proto vs stratified on every ESA kind.\n\n"
        + "\n".join(f"  {name}" for name in written)
        + "\n"
    )
    report = {
        "ok": True,
        "skipped": False,
        "tau": tau_cov,
        "tau_source": "s4_frozen",
        "n": n,
        "refs": ref_info,
        "methods": {k: _brief(v) for k, v in methods.items()},
        "plots": written,
        "dir": str(out),
        "note": (
            "ESA-kind mixed galleries (Length × Locality + level-shift overlay). "
            "256 S3 parents, fold-0 Coverage@τ. Does not overwrite shell_s4."
        ),
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return report


def _generate(
    vname: str,
    model: Any,
    enc: Any,
    x0: np.ndarray,
    ch: np.ndarray,
    schedule: Any,
    common: dict[str, Any],
    kind_z: dict[str, torch.Tensor],
    ra_all: torch.Tensor,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(x0)
    if vname == "stratified":
        return _stratified(model, enc, x0, ch, schedule, common, kind_z, seed)
    if vname == "hybrid":
        return _stratified(
            model, enc, x0, ch, schedule, common, kind_z, seed, proto_kinds=HYBRID_PROTO_KINDS
        )
    if vname == "hybrid_needles":
        return _stratified(
            model,
            enc,
            x0,
            ch,
            schedule,
            common,
            kind_z,
            seed,
            proto_kinds=HYBRID_NEEDLES_PROTO_KINDS,
        )
    if vname == "kind_soft":
        ref, ids = _stack_kind_refs(kind_z)
        return chunked_guided_ddim(
            model,
            enc,
            x0,
            ch,
            schedule,
            ref_anom=ref,
            anom_energy_kind="kind_soft",
            anom_kind_ids=ids,
            **common,
        )
    if vname == "twomix":
        return _twomix(model, enc, x0, ch, schedule, common, kind_z, ra_all, seed)
    if vname == "proto":
        return chunked_guided_ddim(
            model,
            enc,
            x0,
            ch,
            schedule,
            ref_anom=ra_all,
            anom_energy_kind="proto",
            anom_proto_seed=seed,
            **common,
        )
    if vname == "band":
        band = {**common, "lam_anom": 0.0, "lam_rare": 0.0}
        return chunked_guided_ddim(model, enc, x0, ch, schedule, **band)
    raise ValueError(f"unknown kindmix variant {vname}")


def _stratified(
    model: Any,
    enc: Any,
    x0: np.ndarray,
    ch: np.ndarray,
    schedule: Any,
    common: dict[str, Any],
    kind_z: dict[str, torch.Tensor],
    seed: int,
    kind_order: tuple[str, ...] | None = None,
    kind_slug: dict[str, str] | None = None,
    proto_kinds: frozenset[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform kind slices. ``proto_kinds=None`` uses proto on every slice.

    Pass a frozenset to proto only those kinds (hybrid: level shift; needles:
    shift + Point/Global). Other slices are band (λ_anom=0).
    """
    order = list(kind_order or KIND_ORDER)
    slugs = kind_slug or KIND_SLUG
    kinds = [k for k in order if k in kind_z]
    n = len(x0)
    counts = _split_counts(n, len(kinds))
    xs, hs = [], []
    start = 0
    for kind, n_k in zip(kinds, counts, strict=True):
        if n_k == 0:
            continue
        sl = slice(start, start + n_k)
        use_proto = proto_kinds is None or kind in proto_kinds
        mode = "proto" if use_proto else "band"
        print(
            f"  stratified {slugs.get(kind, kind)} n={n_k} {mode} refs={kind_z[kind].size(0)}",
            flush=True,
        )
        if use_proto:
            x, h = chunked_guided_ddim(
                model,
                enc,
                x0[sl],
                ch[sl],
                schedule,
                ref_anom=kind_z[kind],
                anom_energy_kind="proto",
                anom_proto_seed=seed + 17 * (1 + order.index(kind)),
                **common,
            )
        else:
            band = {**common, "lam_anom": 0.0, "lam_rare": 0.0}
            x, h = chunked_guided_ddim(model, enc, x0[sl], ch[sl], schedule, **band)
        xs.append(x)
        hs.append(h)
        start += n_k
    return np.concatenate(xs, axis=0), np.concatenate(hs, axis=0)


def _twomix(
    model: Any,
    enc: Any,
    x0: np.ndarray,
    ch: np.ndarray,
    schedule: Any,
    common: dict[str, Any],
    kind_z: dict[str, torch.Tensor],
    ra_all: torch.Tensor,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(x0)
    n_soft = int(round(0.60 * n))
    n_shift = int(round(0.20 * n))
    n_spike = n - n_soft - n_shift
    parts = [
        ("soft_all", n_soft, ra_all, "soft", None),
        ("shift", n_shift, kind_z.get("real level shift"), "proto", seed + 101),
        ("spike", n_spike, kind_z.get("real ESA Point / Global"), "proto", seed + 202),
    ]
    xs, hs = [], []
    start = 0
    for name, n_k, ref, energy, proto_seed in parts:
        if n_k <= 0:
            continue
        if ref is None or (hasattr(ref, "size") and int(ref.size(0)) == 0):
            print(f"  twomix skip {name}: no refs", flush=True)
            continue
        sl = slice(start, start + n_k)
        print(f"  twomix {name} n={n_k} energy={energy}", flush=True)
        extra: dict[str, Any] = {
            "ref_anom": ref,
            "anom_energy_kind": energy,
        }
        if proto_seed is not None:
            extra["anom_proto_seed"] = proto_seed
        x, h = chunked_guided_ddim(model, enc, x0[sl], ch[sl], schedule, **common, **extra)
        xs.append(x)
        hs.append(h)
        start += n_k
    return np.concatenate(xs, axis=0), np.concatenate(hs, axis=0)


def _stack_kind_refs(kind_z: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    parts: list[torch.Tensor] = []
    ids: list[torch.Tensor] = []
    for k, kind in enumerate(KIND_ORDER):
        if kind not in kind_z:
            continue
        z = kind_z[kind]
        parts.append(z)
        ids.append(torch.full((z.size(0),), k, dtype=torch.long))
    return torch.cat(parts, dim=0), torch.cat(ids, dim=0)


def _split_counts(n: int, k: int) -> list[int]:
    if k <= 0:
        return []
    base, rem = divmod(int(n), int(k))
    return [base + (1 if i < rem else 0) for i in range(k)]


def _kind_alloc_labels(n: int, kinds: list[str]) -> np.ndarray:
    """Contiguous slice labels matching ``_stratified`` allocation."""
    labels = np.empty(int(n), dtype=object)
    if not kinds:
        return labels
    start = 0
    for kind, n_k in zip(kinds, _split_counts(int(n), len(kinds)), strict=True):
        labels[start : start + n_k] = kind
        start += n_k
    return labels


def _coverage_by_kind(
    gallery: np.ndarray,
    x_a: np.ndarray,
    fold_a: np.ndarray,
    ch_a: np.ndarray,
    kind_lab: np.ndarray,
    fold_tab: pd.DataFrame,
    tau: float,
    *,
    fold_id: int,
) -> dict[str, dict[str, float]]:
    from anogen.phases.kindproto import _coverage_kind

    out: dict[str, dict[str, float]] = {}
    for kind in KIND_ORDER:
        if not np.any(kind_lab == kind):
            continue
        out[kind] = _coverage_kind(
            gallery, x_a, fold_a, ch_a, kind_lab, kind, fold_tab, tau, fold_id=fold_id
        )
    return out


def _coverage_slices(
    gallery: np.ndarray,
    alloc: np.ndarray,
    x_a: np.ndarray,
    fold_a: np.ndarray,
    ch_a: np.ndarray,
    kind_lab: np.ndarray,
    fold_tab: pd.DataFrame,
    tau: float,
    *,
    fold_id: int,
) -> dict[str, dict[str, float]]:
    from anogen.phases.kindproto import _coverage_kind

    out: dict[str, dict[str, float]] = {}
    alloc = np.asarray(alloc)
    for kind in KIND_ORDER:
        sl = gallery[alloc == kind]
        if len(sl) == 0:
            continue
        out[kind] = _coverage_kind(
            sl, x_a, fold_a, ch_a, kind_lab, kind, fold_tab, tau, fold_id=fold_id
        )
    return out


def _plot_kind_slices(
    out: Path,
    enc_name: str,
    vname: str,
    x: np.ndarray,
    alloc: np.ndarray,
    examples: dict[str, dict[str, np.ndarray]],
    bin_seconds: int,
) -> list[str]:
    from anogen.shell.plots import save_compare_rows, save_named_grid

    if not examples or len(alloc) != len(x):
        return []
    written: list[str] = []
    alloc = np.asarray(alloc)
    col_names = [k for k in KIND_ORDER if k in examples and np.any(alloc == k)]
    cells: dict[tuple[str, str], np.ndarray] = {}
    for kind in col_names:
        gen = x[alloc == kind]
        real = examples[kind]["x"]
        rows = {
            kind: real[:4],
            "aimed slice (best-stat)": gen[rank_for_kind(gen, kind, 4)],
        }
        slug = KIND_SLUG.get(kind, "kind")
        png = f"slice_{enc_name}_{vname}_{slug}.png"
        save_compare_rows(
            out / png,
            rows,
            bin_seconds=bin_seconds,
            n_cols=4,
            colors={kind: "#b2182b", "aimed slice (best-stat)": "#d95f02"},
            title=f"{enc_name} {vname}: real {kind} vs aimed slice",
        )
        written.append(png)
        cells[("real ESA-ADB", kind)] = real[0]
        if len(gen):
            cells[("aimed slice", kind)] = gen[int(rank_for_kind(gen, kind, 1)[0])]
    if col_names:
        grid = f"slice_{enc_name}_{vname}_grid.png"
        save_named_grid(
            out / grid,
            cells,
            row_names=["real ESA-ADB", "aimed slice"],
            col_names=col_names,
            bin_seconds=bin_seconds,
            colors={"real ESA-ADB": "#b2182b", "aimed slice": "#d95f02"},
            title=f"ESA kinds vs aimed stratified slices ({enc_name})",
            share_col_ylim=True,
        )
        written.append(grid)
    return written


def _plot_variant_grid(
    out: Path,
    enc_name: str,
    vgals: dict[str, np.ndarray],
    examples: dict[str, dict[str, np.ndarray]],
    bin_seconds: int,
) -> list[str]:
    from anogen.shell.morphology import nearest_generated

    if not examples:
        return []
    col_names = [k for k in KIND_ORDER if k in examples]
    row_names = ["real ESA-ADB"] + [v for v in _PLOT_VARIANT_ORDER if v in vgals]
    cells: dict[tuple[str, str], np.ndarray] = {}
    for col in col_names:
        cells[("real ESA-ADB", col)] = examples[col]["x"][0]
        for vname in _PLOT_VARIANT_ORDER:
            if vname not in vgals:
                continue
            match = nearest_generated(examples[col]["x"][:1], vgals[vname])
            if len(match):
                cells[(vname, col)] = match[0]
    colors = {"real ESA-ADB": "#b2182b", **_COLORS}
    png = f"real_kinds_vs_{enc_name}_mix.png"
    save_named_grid(
        out / png,
        cells,
        row_names=row_names,
        col_names=col_names,
        bin_seconds=bin_seconds,
        colors=colors,
        title=f"ESA-ADB kinds vs mixed galleries ({enc_name}, nearest in φ)",
        share_col_ylim=True,
    )
    return [png]


def _plot_subseq_compare(
    out: Path,
    enc_name: str,
    vgals: dict[str, np.ndarray],
    examples: dict[str, dict[str, np.ndarray]],
    bin_seconds: int,
) -> list[str]:
    """Real local / global subsequence vs band, pooled proto, and stratified."""
    return _plot_recipe_compare(
        out,
        enc_name,
        vgals,
        examples,
        bin_seconds,
        kinds=(
            "real ESA local subsequence",
            "real ESA global subsequence",
        ),
        prefix="subseq",
        title="Local / global subsequence: band vs pooled proto vs stratified",
    )


def _plot_recipe_compare(
    out: Path,
    enc_name: str,
    vgals: dict[str, np.ndarray],
    examples: dict[str, dict[str, np.ndarray]],
    bin_seconds: int,
    *,
    kinds: tuple[str, ...] | None = None,
    recipes: tuple[str, ...] = ("band", "proto", "stratified"),
    prefix: str = "allkinds",
    title: str = "All ESA kinds: band vs pooled proto vs stratified",
) -> list[str]:
    """Nearest-φ grid and 4-example rows for selected kinds × recipes."""
    from anogen.shell.morphology import nearest_generated

    col_names = [k for k in (kinds or KIND_ORDER) if k in examples]
    recs = [v for v in recipes if v in vgals]
    if not col_names or len(recs) < 2:
        return []
    written: list[str] = []
    row_names = ["real ESA-ADB"] + recs
    cells: dict[tuple[str, str], np.ndarray] = {}
    for col in col_names:
        cells[("real ESA-ADB", col)] = examples[col]["x"][0]
        for rec in recs:
            match = nearest_generated(examples[col]["x"][:1], vgals[rec])
            if len(match):
                cells[(rec, col)] = match[0]
    png = f"{prefix}_{enc_name}_band_proto_strat.png"
    save_named_grid(
        out / png,
        cells,
        row_names=row_names,
        col_names=col_names,
        bin_seconds=bin_seconds,
        colors={"real ESA-ADB": "#b2182b", **_COLORS},
        title=f"{title} ({enc_name})",
        share_col_ylim=True,
    )
    written.append(png)
    for kind in col_names:
        rows: dict[str, np.ndarray] = {kind: examples[kind]["x"][:4]}
        for rec in recs:
            rows[rec] = nearest_generated(examples[kind]["x"][:4], vgals[rec])
        slug = KIND_SLUG.get(kind, "kind")
        cmp = f"{prefix}_{enc_name}_{slug}_band_proto_strat.png"
        save_compare_rows(
            out / cmp,
            rows,
            bin_seconds=bin_seconds,
            n_cols=4,
            colors={kind: "#b2182b", **_COLORS},
            title=f"{kind}: real vs nearest band / proto / stratified ({enc_name})",
        )
        written.append(cmp)
    return written


def _plot_both_encoders_compare(
    out: Path,
    galleries: dict[str, dict[str, np.ndarray]],
    examples: dict[str, dict[str, np.ndarray]],
    bin_seconds: int,
) -> list[str]:
    """One grid: real + band/proto/stratified for both encoders, all ESA kinds."""
    from anogen.shell.morphology import nearest_generated

    col_names = [k for k in KIND_ORDER if k in examples]
    if not col_names:
        return []
    row_names = ["real ESA-ADB"]
    colors = {"real ESA-ADB": "#b2182b"}
    cells: dict[tuple[str, str], np.ndarray] = {}
    for col in col_names:
        cells[("real ESA-ADB", col)] = examples[col]["x"][0]
    for enc_name, enc_color in (("time_recon", "#2166ac"), ("time_both", "#1b9e77")):
        vgals = galleries.get(enc_name) or {}
        for rec, shade in (("band", 0), ("proto", 1), ("stratified", 2)):
            if rec not in vgals:
                continue
            row = f"{enc_name} {rec}"
            row_names.append(row)
            colors[row] = {
                ("time_recon", "band"): "#4d4d4d",
                ("time_recon", "proto"): "#2166ac",
                ("time_recon", "stratified"): "#d95f02",
                ("time_both", "band"): "#737373",
                ("time_both", "proto"): "#1b9e77",
                ("time_both", "stratified"): "#e6ab02",
            }[(enc_name, rec)]
            del shade
            for col in col_names:
                match = nearest_generated(examples[col]["x"][:1], vgals[rec])
                if len(match):
                    cells[(row, col)] = match[0]
    if len(row_names) < 3:
        return []
    png = "allkinds_both_encoders_band_proto_strat.png"
    save_named_grid(
        out / png,
        cells,
        row_names=row_names,
        col_names=col_names,
        bin_seconds=bin_seconds,
        colors=colors,
        title="All ESA kinds: band vs pooled proto vs stratified (nearest in φ)",
        share_col_ylim=True,
    )
    return [png]
