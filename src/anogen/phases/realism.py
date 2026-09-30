"""Raw-window realism audit on existing galleries.

This phase does not generate samples and does not alter feature_pack_v1, S3,
S4, or their frozen threshold.  Published fidelity measures and project-
specific diagnostics are written to separate tables.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anogen.config import REPO_ROOT
from anogen.phases.encscore import _abs, _jsonable
from anogen.shell.events import types_from_cfg
from anogen.shell.morphology import KIND_ORDER, KIND_SLUG, kinds_for_windows
from anogen.shell.realism import (
    INTERPRETABLE_FEATURES,
    channel_span_normalize,
    classifier_two_sample_test,
    diag_cusum,
    diag_envelope,
    diag_excursion_count,
    diag_first_difference,
    diag_persistence,
    diag_signed_peak,
    feature_distribution_distance,
    fft_average_wasserstein,
    marginal_distribution_difference,
    marginal_w1,
    mean_psd,
    pooled_acf,
    autocorrelation_difference,
)
from anogen.shell.scaler import ChannelMinMax, load_minmax


@dataclass
class Gallery:
    name: str
    x: np.ndarray
    channel_idx: np.ndarray
    fold: int | None = None
    kind_alloc: np.ndarray | None = None
    donor_group: np.ndarray | None = None
    source: str = ""


def assert_isolated(out: Path, *locked: Path) -> None:
    if out.resolve() in {p.resolve() for p in locked}:
        raise RuntimeError("realism audit must not write into a locked result directory")


def run_realism(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    s4 = _abs(cfg.get("s4_dir", root / "results/shell_s4"), root)
    hybrid = _abs(
        cfg.get("kindmix_score_hybrid_dir", root / "results/shell_kindmix_score_hybrid"),
        root,
    )
    out = _abs(cfg.get("realism_dir", root / "results/shell_realism"), root)
    plots = out / "plots"
    docs_plots = root / "docs" / "realism"
    assert_isolated(out, s0, s3, s4, hybrid)
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    docs_plots.mkdir(parents=True, exist_ok=True)

    needed = {
        "S0 arrays": s0 / "labeled_arrays.npz",
        "S0 metadata": s0 / "labeled_windows.csv",
        "S0 scaler": s0 / "minmax_scaler.npz",
        "S3 galleries": s3 / "galleries.npz",
        "S4 gallery": s4 / "shell_gallery.npz",
    }
    missing_required = [f"{name}: {path}" for name, path in needed.items() if not path.is_file()]
    if missing_required:
        report = {"ok": False, "skipped": True, "reason": "; ".join(missing_required)}
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    rcfg = _resolved_spec(cfg)
    # This is written before any score so the exact audit choices are recorded.
    (out / "spec.json").write_text(json.dumps(rcfg, indent=2, sort_keys=True) + "\n")

    lab = np.load(s0 / "labeled_arrays.npz")
    meta = pd.read_csv(s0 / "labeled_windows.csv")
    anomaly_meta = meta[meta["kind"] == "anomaly"].reset_index(drop=True)
    x_real = np.asarray(lab["anomaly"], dtype=np.float64)
    real_ch = np.asarray(lab["anomaly_channel"], dtype=np.int64)
    real_fold = np.asarray(lab["anomaly_fold"], dtype=np.int64)
    if len(anomaly_meta) != len(x_real):
        raise ValueError(
            f"anomaly metadata has {len(anomaly_meta)} rows but arrays have {len(x_real)}"
        )
    event_id = anomaly_meta["event_id"].astype(str).to_numpy()
    real_kind = kinds_for_windows(x_real, anomaly_meta, types_from_cfg(cfg)).astype(str)
    scaler = load_minmax(s0 / "minmax_scaler.npz")

    galleries, missing_optional = _load_galleries(root, cfg, s3, s4, hybrid)
    if not galleries:
        report = {"ok": False, "skipped": True, "reason": "no generated galleries found"}
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    c2st_rows: list[dict[str, Any]] = []
    per_window: dict[str, np.ndarray] = {}
    folds = sorted({int(f) for f in real_fold if int(f) >= 0})
    min_n = int(rcfg["n_min"])

    for gallery in galleries:
        target_folds = [gallery.fold] if gallery.fold is not None else folds
        for fold in target_folds:
            rmask = real_fold == int(fold)
            if not np.any(rmask):
                continue
            _audit_pair(
                gallery,
                x_real[rmask],
                real_ch[rmask],
                event_id[rmask],
                scaler,
                fold=int(fold),
                scope="whole_gallery",
                query_kind="all",
                spec=rcfg,
                metric_rows=metric_rows,
                diagnostic_rows=diagnostic_rows,
                c2st_rows=c2st_rows,
                per_window=per_window,
                min_n=min_n,
            )
            for kind in KIND_ORDER:
                kmask = rmask & (real_kind == kind)
                if not np.any(kmask):
                    continue
                _audit_pair(
                    gallery,
                    x_real[kmask],
                    real_ch[kmask],
                    event_id[kmask],
                    scaler,
                    fold=int(fold),
                    scope="whole_gallery",
                    query_kind=kind,
                    spec=rcfg,
                    metric_rows=metric_rows,
                    diagnostic_rows=diagnostic_rows,
                    c2st_rows=None,
                    per_window=per_window,
                    min_n=min_n,
                )
                if gallery.kind_alloc is not None:
                    gmask = np.asarray(gallery.kind_alloc).astype(str) == str(kind)
                    if np.any(gmask):
                        allocated = Gallery(
                            name=gallery.name,
                            x=gallery.x[gmask],
                            channel_idx=gallery.channel_idx[gmask],
                            fold=gallery.fold,
                            kind_alloc=gallery.kind_alloc[gmask],
                            donor_group=(
                                gallery.donor_group[gmask]
                                if gallery.donor_group is not None
                                else np.flatnonzero(gmask)
                            ),
                            source=gallery.source,
                        )
                        _audit_pair(
                            allocated,
                            x_real[kmask],
                            real_ch[kmask],
                            event_id[kmask],
                            scaler,
                            fold=int(fold),
                            scope="allocated_slice",
                            query_kind=kind,
                            spec=rcfg,
                            metric_rows=metric_rows,
                            diagnostic_rows=diagnostic_rows,
                            c2st_rows=None,
                            per_window=per_window,
                            min_n=min_n,
                        )

    metrics = pd.DataFrame(metric_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    c2st = pd.DataFrame(c2st_rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    diagnostics.to_csv(out / "diagnostics.csv", index=False)
    c2st.to_csv(out / "c2st.csv", index=False)
    np.savez_compressed(out / "per_window.npz", **per_window)

    written = _write_plots(
        plots,
        x_real=x_real,
        real_ch=real_ch,
        real_kind=real_kind,
        galleries=galleries,
        scaler=scaler,
        spec=rcfg,
    )
    for name in written:
        shutil.copy2(plots / name, docs_plots / name)
    index = _index_text(written, missing_optional, len(metric_rows), len(diagnostic_rows))
    (out / "INDEX.txt").write_text(index)
    (docs_plots / "INDEX.txt").write_text(index)

    report = {
        "ok": True,
        "skipped": False,
        "phase": "realism",
        "feature_pack_v1": "unchanged; existing ARP/Coverage/EDI are not recomputed",
        "n_real": int(len(x_real)),
        "n_events": int(len(set(event_id.tolist()))),
        "methods": sorted({g.name for g in galleries}),
        "n_metric_rows": len(metric_rows),
        "n_diagnostic_rows": len(diagnostic_rows),
        "n_c2st_rows": len(c2st_rows),
        "missing_optional": missing_optional,
        "plots": written,
        "note": (
            "Raw-window audit only. Published fidelity measures and project-specific "
            "diagnostics are separate. No generator, feature_pack_v1, S3, or S4 artifact "
            "was modified."
        ),
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return report


def _resolved_spec(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = dict((cfg.get("shell") or {}).get("realism") or {})
    return {
        "version": "raw-realism-v1",
        "normalization": "channel_train_span_no_clip",
        "bins": int(raw.get("bins", 50)),
        "max_lag": int(raw.get("max_lag", 64)),
        "median_width": int(raw.get("median_width", 9)),
        "mad_scale": float(raw.get("mad_scale", 4.0)),
        "cusum_min_segment": int(raw.get("cusum_min_segment", 8)),
        "n_min": int(raw.get("n_min", 10)),
        "bootstrap": int(raw.get("bootstrap", 1000)),
        "seeds": [int(v) for v in (raw.get("seeds") or [0, 1, 2, 3, 4])],
        "c2st_test_size": float(raw.get("c2st_test_size", 0.3)),
        "c2st_permutations": int(raw.get("c2st_permutations", 200)),
        "methods": list(
            raw.get("methods")
            or [
                "donor",
                "shell",
                "unguided",
                "genias",
                "genias_patched",
                "posthoc",
                "c1",
                "c2",
                "cutaddpaste",
                "taxonomy",
                "editor",
                "timeleash_l2",
                "timeleash_gm",
                "hashfix",
            ]
        ),
        "references": {
            "marginal_w1": "Stenger et al. 2024 (WD on marginals)",
            "mdd": (
                "Ang et al. 2023 TSGBench M4; formula taken from "
                "src/feature_based_measures.py HistoLoss/calculate_mdd "
                "(density histograms, 50 bins, first time step dropped, "
                "generated mass outside the real range discarded)"
            ),
            "acd": (
                "Ang et al. 2023 TSGBench M5; formula taken from "
                "src/feature_based_measures.py acf_torch/acf_diff/calculate_acd "
                "(mean and variance pooled over windows and time, Euclidean "
                "norm over lags 0..min(max_lag, W)-1)"
            ),
            "fft_awd": (
                "Seyfi et al. 2022 COSCI-GAN defines the channel-averaged "
                "Wasserstein distance between amplitude distributions; the FFT "
                "extraction follows Stenger et al. 2024 because the original "
                "paper knew its generating amplitudes and does not specify it"
            ),
            "feature_distance": "Stenger et al. 2024; Ang et al. 2023",
            "c2st": "Lopez-Paz and Oquab 2017",
            "cusum_diagnostic": "Truong et al. 2020",
        },
        "deviations": {
            "normalization": (
                "TSGBench min-max scales the whole dataset to [0, 1]; this audit "
                "uses the frozen per-channel nominal training span instead"
            ),
            "channel_aggregation": (
                "TSGBench averages over one tensor whose channels share a sample "
                "count; here every window belongs to one channel, so MDD and ACD "
                "are macro-averaged over channels"
            ),
            "fft_dominant_frequency_w1": "project diagnostic, not part of AWD",
            "mdd_out_of_real_range_fraction": (
                "project diagnostic reporting the generated mass the MDD "
                "definition discards"
            ),
        },
    }


def _audit_pair(
    gallery: Gallery,
    real: np.ndarray,
    real_ch: np.ndarray,
    real_event: np.ndarray,
    scaler: ChannelMinMax,
    *,
    fold: int,
    scope: str,
    query_kind: str,
    spec: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    c2st_rows: list[dict[str, Any]] | None,
    per_window: dict[str, np.ndarray],
    min_n: int,
) -> None:
    n_real, n_synth = len(real), len(gallery.x)
    base = {
        "method": gallery.name,
        "fold": fold,
        "scope": scope,
        "kind": query_kind,
        "n_real": n_real,
        "n_events": len(set(np.asarray(real_event).astype(str).tolist())),
        "n_synth": n_synth,
        "source": gallery.source,
        "insufficient_evidence": n_real < min_n,
    }
    if n_real >= min_n and n_synth:
        published: list[tuple[str, dict[str, Any], str]] = [
            (
                "marginal_w1",
                marginal_w1(real, gallery.x, real_ch, gallery.channel_idx, scaler),
                "Stenger2024",
            ),
            (
                "mdd",
                marginal_distribution_difference(
                    real,
                    gallery.x,
                    real_ch,
                    gallery.channel_idx,
                    scaler,
                    bins=int(spec["bins"]),
                ),
                "TSGBench",
            ),
            (
                "acd",
                autocorrelation_difference(
                    real,
                    gallery.x,
                    real_ch,
                    gallery.channel_idx,
                    max_lag=int(spec["max_lag"]),
                ),
                "TSGBench",
            ),
            (
                "fft_awd_amplitude",
                fft_average_wasserstein(real, gallery.x, real_ch, gallery.channel_idx, scaler),
                "COSCI-GAN",
            ),
            (
                "feature_distance_macro",
                feature_distribution_distance(
                    real, gallery.x, real_ch, gallery.channel_idx, scaler
                ),
                "Stenger2024/TSGBench",
            ),
        ]
        for metric, result, reference in published:
            metric_rows.append(
                {
                    **base,
                    "tier": "published",
                    "metric": metric,
                    "value": result["value"],
                    "reference": reference,
                    "details": json.dumps(_jsonable(result), sort_keys=True),
                }
            )
            if metric == "fft_awd_amplitude":
                metric_rows.append(
                    {
                        **base,
                        "tier": "diagnostic",
                        "metric": "fft_dominant_frequency_w1",
                        "value": result["frequency_w1"]["value"],
                        "reference": "project diagnostic (not part of AWD)",
                        "details": json.dumps(_jsonable(result["frequency_w1"]), sort_keys=True),
                    }
                )
            if metric == "mdd":
                metric_rows.append(
                    {
                        **base,
                        "tier": "diagnostic",
                        "metric": "mdd_out_of_real_range_fraction",
                        "value": result["out_of_real_range_fraction"]["value"],
                        "reference": "project diagnostic (mass MDD discards)",
                        "details": json.dumps(
                            _jsonable(result["out_of_real_range_fraction"]), sort_keys=True
                        ),
                    }
                )
            if metric == "feature_distance_macro":
                for feature in INTERPRETABLE_FEATURES:
                    metric_rows.append(
                        {
                            **base,
                            "tier": "published",
                            "metric": f"feature_w1_{feature}",
                            "value": result["features"][feature],
                            "reference": "Stenger2024/TSGBench",
                            "details": "",
                        }
                    )

    _append_diagnostics(
        gallery,
        real,
        real_ch,
        scaler,
        base,
        spec,
        diagnostic_rows,
        per_window,
    )
    if c2st_rows is not None and n_real >= min_n and n_synth:
        groups = (
            gallery.donor_group
            if gallery.donor_group is not None
            else np.arange(n_synth, dtype=np.int64)
        )
        try:
            rows = classifier_two_sample_test(
                real,
                gallery.x,
                real_ch,
                gallery.channel_idx,
                real_event,
                groups,
                scaler,
                seeds=spec["seeds"],
                test_size=float(spec["c2st_test_size"]),
                n_perm=int(spec["c2st_permutations"]),
            )
        except ValueError as exc:
            c2st_rows.append({**base, "seed": np.nan, "error": str(exc)})
        else:
            c2st_rows.extend({**base, **row, "error": ""} for row in rows)


def _append_diagnostics(
    gallery: Gallery,
    real: np.ndarray,
    real_ch: np.ndarray,
    scaler: ChannelMinMax,
    base: dict[str, Any],
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    per_window: dict[str, np.ndarray],
) -> None:
    for source, x, ch in (
        ("real", real, real_ch),
        ("synthetic", gallery.x, gallery.channel_idx),
    ):
        u = channel_span_normalize(x, ch, scaler)
        envelope = diag_envelope(x, ch, scaler)
        peak = diag_signed_peak(u, width=int(spec["median_width"]))
        cusum = diag_cusum(u, min_segment=int(spec["cusum_min_segment"]))
        persistence = diag_persistence(u)
        count = diag_excursion_count(
            u,
            width=int(spec["median_width"]),
            mad_scale=float(spec["mad_scale"]),
        )
        diff = diag_first_difference(u)
        values = {
            "envelope_sample_fraction": np.asarray([envelope["sample_fraction"]]),
            "envelope_window_fraction": np.asarray([envelope["window_fraction"]]),
            "envelope_max_excess": np.asarray([envelope["max_excess"]]),
            "signed_peak_severity": peak["severity"],
            "signed_peak_position": peak["position"],
            "cusum_contrast": cusum["contrast"],
            "cusum_position": cusum["position"],
            "quarter_persistence": persistence,
            "excursion_count": count,
            "diff_std": diff["std"],
            "diff_p99": diff["p99"],
            "diff_p999": diff["p999"],
        }
        for metric, value in values.items():
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            rows.append(
                {
                    **base,
                    "tier": "diagnostic",
                    "sample": source,
                    "metric": metric,
                    "n": len(arr),
                    "mean": float(np.mean(arr)) if len(arr) else np.nan,
                    "median": float(np.median(arr)) if len(arr) else np.nan,
                    "p05": float(np.quantile(arr, 0.05)) if len(arr) else np.nan,
                    "p95": float(np.quantile(arr, 0.95)) if len(arr) else np.nan,
                }
            )
            key = _npz_key(
                gallery.name,
                str(base["fold"]),
                str(base["scope"]),
                str(base["kind"]),
                source,
                metric,
            )
            per_window[key] = arr


def _load_galleries(
    root: Path,
    cfg: dict[str, Any],
    s3: Path,
    s4: Path,
    hybrid: Path,
) -> tuple[list[Gallery], list[str]]:
    requested = set(_resolved_spec(cfg)["methods"])
    galleries: list[Gallery] = []
    missing: list[str] = []
    s3g = np.load(s3 / "galleries.npz")
    cond = np.asarray(s3g["cond"])
    cond_ch = np.asarray(s3g["channel_idx"], dtype=np.int64)
    donor = np.arange(len(cond), dtype=np.int64)

    def add_static(name: str, x: np.ndarray, ch: np.ndarray, source: Path) -> None:
        if name in requested:
            galleries.append(
                Gallery(name, np.asarray(x), np.asarray(ch, dtype=np.int64), donor_group=donor, source=str(source))
            )

    add_static("donor", cond, cond_ch, s3 / "galleries.npz")
    for name, key in (
        ("genias", "genias"),
        ("genias_patched", "genias_patched"),
        ("posthoc", "posthoc"),
    ):
        if key in s3g.files:
            add_static(name, s3g[key], cond_ch, s3 / "galleries.npz")
        elif name in requested:
            missing.append(name)

    s4g = np.load(s4 / "shell_gallery.npz")
    for name, key in (("shell", "x"), ("unguided", "unguided")):
        if key in s4g.files:
            add_static(name, s4g[key], s4g["channel_idx"], s4 / "shell_gallery.npz")

    fold_specs = {
        "c1": (hybrid, "time_both_hybrid_needles_fold{fold}.npz"),
        "c2": (hybrid, "time_recon_hybrid_needles_fold{fold}.npz"),
        "editor": (_abs(cfg.get("editor_dir", root / "results/shell_editor"), root), "time_both_hybrid_needles_editor_fold{fold}.npz"),
        "timeleash_l2": (_abs(cfg.get("timeleash_dir", root / "results/shell_timeleash"), root), "time_both_l2_fold{fold}.npz"),
        "timeleash_gm": (_abs(cfg.get("timeleash_dir", root / "results/shell_timeleash"), root), "time_both_gm_fold{fold}.npz"),
        "hashfix": (_abs(cfg.get("hashfix_dir", root / "results/shell_hashfix"), root), "time_both_hashfix_fold{fold}.npz"),
    }
    for name, (directory, pattern) in fold_specs.items():
        if name not in requested:
            continue
        found = False
        for fold in range(int((cfg.get("shell") or {}).get("n_folds", 3))):
            path = directory / pattern.format(fold=fold)
            if not path.is_file():
                continue
            found = True
            blob = np.load(path, allow_pickle=True)
            x = np.asarray(blob["x"])
            ch = np.asarray(blob["channel_idx"], dtype=np.int64)
            if len(x) != len(ch):
                raise ValueError(f"{path}: x/channel_idx length mismatch")
            alloc = np.asarray(blob["kind_alloc"]).astype(str) if "kind_alloc" in blob.files else None
            galleries.append(
                Gallery(
                    name,
                    x,
                    ch,
                    fold=fold,
                    kind_alloc=alloc,
                    donor_group=np.arange(len(x), dtype=np.int64),
                    source=str(path),
                )
            )
        if not found:
            missing.append(name)

    genbase = _abs(cfg.get("genbase_dir", root / "results/shell_genbase"), root)
    for name in ("cutaddpaste", "taxonomy"):
        if name not in requested:
            continue
        path = genbase / f"{name}.npz"
        if path.is_file():
            blob = np.load(path, allow_pickle=True)
            add_static(name, blob["x"], blob["channel_idx"], path)
        else:
            missing.append(name)
    return galleries, sorted(set(missing))


def _write_plots(
    out: Path,
    *,
    x_real: np.ndarray,
    real_ch: np.ndarray,
    real_kind: np.ndarray,
    galleries: list[Gallery],
    scaler: ChannelMinMax,
    spec: dict[str, Any],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    written: list[str] = []
    # One representative gallery per core method; fold 0 for fold-specific methods.
    core_names = ("c1", "c2", "genias", "posthoc", "cutaddpaste", "taxonomy")
    core: dict[str, Gallery] = {}
    for name in core_names:
        choices = [g for g in galleries if g.name == name and g.fold in (None, 0)]
        if choices:
            core[name] = choices[0]
    colors = {
        "real": "#b2182b",
        "c1": "#2166ac",
        "c2": "#7570b3",
        "genias": "#e7298a",
        "posthoc": "#66a61e",
        "cutaddpaste": "#1b9e77",
        "taxonomy": "#d95f02",
    }
    for kind in KIND_ORDER:
        rm = real_kind == kind
        if not np.any(rm):
            continue
        slug = KIND_SLUG.get(kind, "kind")
        series: dict[str, np.ndarray] = {
            "real": channel_span_normalize(x_real[rm], real_ch[rm], scaler)
        }
        for name, gallery in core.items():
            gx, gch = gallery.x, gallery.channel_idx
            if gallery.kind_alloc is not None:
                gm = np.asarray(gallery.kind_alloc).astype(str) == str(kind)
                if np.any(gm):
                    gx, gch = gx[gm], gch[gm]
            series[name] = channel_span_normalize(gx, gch, scaler)

        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        for name, x in series.items():
            ax.hist(
                x.ravel(),
                bins=int(spec["bins"]),
                density=True,
                histtype="step",
                linewidth=1.2,
                label=name,
                color=colors.get(name),
            )
        ax.set_title(f"{kind}: channel-span marginal")
        ax.set_xlabel("(x - train lo) / (train hi - train lo)")
        ax.set_ylabel("density")
        ax.legend(fontsize=7)
        fig.tight_layout()
        name = f"marginal_{slug}.png"
        fig.savefig(out / name, dpi=140)
        plt.close(fig)
        written.append(name)

        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        for name, x in series.items():
            ax.plot(pooled_acf(x, int(spec["max_lag"])), label=name, color=colors.get(name))
        ax.set_title(f"{kind}: pooled ACF")
        ax.set_xlabel("lag (30 s bins)")
        ax.set_ylabel("ACF")
        ax.legend(fontsize=7)
        fig.tight_layout()
        name = f"acf_{slug}.png"
        fig.savefig(out / name, dpi=140)
        plt.close(fig)
        written.append(name)

        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        for name, x in series.items():
            psd = mean_psd(x)
            ax.semilogy(np.arange(len(psd)), psd + 1e-12, label=name, color=colors.get(name))
        ax.set_title(f"{kind}: mean periodogram")
        ax.set_xlabel("rFFT bin")
        ax.set_ylabel("power")
        ax.legend(fontsize=7)
        fig.tight_layout()
        name = f"psd_{slug}.png"
        fig.savefig(out / name, dpi=140)
        plt.close(fig)
        written.append(name)
    return written


def _index_text(written: list[str], missing: list[str], n_metrics: int, n_diag: int) -> str:
    lines = [
        "Raw-window realism audit. feature_pack_v1 and frozen S4 are unchanged.",
        f"published metric rows={n_metrics}; diagnostic rows={n_diag}",
        f"missing optional methods={missing}",
        "",
        "Published: marginal W1/MDD, ACD, FFT-AWD, feature distances, C2ST.",
        "Diagnostics: training envelope, signed peak, CUSUM, quarter persistence,",
        "contiguous excursions, first-difference texture.",
        "",
        "Plots:",
    ]
    lines.extend(f"  {name}" for name in written)
    lines.append("")
    return "\n".join(lines)


def _npz_key(*parts: str) -> str:
    text = "__".join(parts)
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text)
