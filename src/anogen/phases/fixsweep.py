"""Fix sweep: C1 (hybrid_needles, chanmix) under the position-bias / guidance fixes.

Isolated ``results/shell_fixsweep/``. Never writes into shell_s1..s5, the
hybrid freeze, or shell_chanmix. Every variant is the C1 recipe (parent50,
λ=0.3, λ_anom=1, unit ∇, 50 DDIM, ν=0.2, channel-stratified kinds) with one or
more of the switches documented in docs/POSITION_BIAS.md:

  flip              ramp | avg   zero-retrain FlipEnsemble around the causal ε
  denoiser          path         alternative S1 checkpoint (e.g. bidirectional)
  guidance_space    x | x0       x0 = Jacobian-free guidance
  final_grad_scale  float        last edit scale (1 locked, 0 hashfix)
  burnin            bool         real preceding telemetry as a B-bin prefix
  proto_shift_max   int          per-sample proto time shift (encoder steps)
  lam_repel         float        particle-guidance repulsion
  start_from_noise  bool         x_T ~ N(0, I) (burn-in prefix is then noise)
  guidance_t_window [lo, hi]     guidance only on that part of the trajectory

All variants share the torch seed per fold (common random numbers), so paired
differences against ``c1_repro`` isolate the switch, not the noise draw.

Scores per fold: protocol ARP / Coverage@τ (frozen τ, feature_pack_v1) with
event-cluster two-way bootstrap CIs and paired differences vs c1_repro; raw
realism diagnostics (envelope frequency and magnitude, CUSUM, edge sharpness,
peak position incl. the end of the window) for the whole gallery and per
allocated kind slice; C2ST; calibration rows (oracle train-fold anomalies,
untouched donors, white noise). EDI uses one union of all fold-0 galleries in
this run (partition mark **f**; do not mix with other EDI tables).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anogen.config import REPO_ROOT

DEFAULT_VARIANTS: dict[str, dict[str, Any]] = {
    "c1_repro": {},
    "final025": {"final_grad_scale": 0.25},
    "x0": {"guidance_space": "x0"},
    "x0_final025": {"guidance_space": "x0", "final_grad_scale": 0.25},
    "flip": {"flip": "ramp"},
    "flip_x0": {"flip": "ramp", "guidance_space": "x0"},
    "flip_x0_final025": {"flip": "ramp", "guidance_space": "x0", "final_grad_scale": 0.25},
    "burnin_x0": {"burnin": True, "guidance_space": "x0"},
    "flip_x0_shift": {"flip": "ramp", "guidance_space": "x0", "proto_shift_max": 16},
    "flip_x0_shift_repel": {
        "flip": "ramp",
        "guidance_space": "x0",
        "proto_shift_max": 16,
        "lam_repel": 0.3,
    },
    "flip_x0_from_noise": {"flip": "ramp", "guidance_space": "x0", "start_from_noise": True},
    "flip_x0_contrast": {
        "flip": "ramp",
        "guidance_space": "x0",
        "final_grad_scale": 0.25,
        "contrast": {"real level shift": "step", "real ESA Point / Global": "spike"},
    },
    "flip_x0_contrast_only": {
        "flip": "ramp",
        "guidance_space": "x0",
        "final_grad_scale": 0.25,
        "contrast": {"real level shift": "step", "real ESA Point / Global": "spike"},
        "contrast_only": True,
    },
    "flip_x0_from_matched_noise": {
        "flip": "ramp",
        "guidance_space": "x0",
        "start_from_noise": True,
        "matched_noise": True,
    },
}

# Keys the phase interprets itself; everything else goes to guided_ddim.
_PHASE_KEYS = {
    "flip",
    "denoiser",
    "burnin",
    "proto_shift_max",
    "start_from_noise",
    "encoder",
    "recipe",
    "matched_noise",
    "contrast",
    "contrast_only",
    "lam_contrast",
    "context",
}


def run_fixsweep(cfg: dict[str, Any]) -> dict[str, Any]:
    from anogen.phases.encscore import _abs, _jsonable

    root = Path(cfg.get("_repo_root", REPO_ROOT))
    fcfg = dict((cfg.get("shell") or {}).get("fixsweep") or {})
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s1 = _abs(cfg.get("s1_dir", root / "results/shell_s1"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    s4 = _abs(cfg.get("s4_dir", root / "results/shell_s4"), root)
    enc_dir = _abs(cfg.get("enc_dir", root / "results/shell_enc"), root)
    out = _abs(fcfg.get("out", root / "results/shell_fixsweep"), root)
    locked = {p.resolve() for p in (s0, s1, s3, s4, enc_dir)}
    if out.resolve() in locked:
        raise RuntimeError("fixsweep must not write into a locked result directory")
    out.mkdir(parents=True, exist_ok=True)

    from anogen.shell.diffusion import torch_available

    if not torch_available():
        report = {"ok": False, "skipped": True, "reason": "torch is not installed"}
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    needed = {
        "S0 arrays": s0 / "labeled_arrays.npz",
        "S0 meta": s0 / "labeled_windows.csv",
        "S1": s1 / "denoiser.pt",
        "S3": s3 / "galleries.npz",
        "S4": s4 / "summary.json",
    }
    for label, path in needed.items():
        if not path.is_file():
            report = {"ok": False, "skipped": True, "reason": f"{label} missing: {path}"}
            (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            return report

    import torch

    from anogen.phases.encscore import _load_encoder
    from anogen.phases.kindmix import (
        HYBRID_NEEDLES_PROTO_KINDS,
        HYBRID_PROTO_KINDS,
        _PARENT50,
        channel_stratified_alloc,
        slice_lam_anom,
    )
    from anogen.phases.tune import _score_gallery
    from anogen.shell.coverage import edi_by_method, mean_pairwise_distance
    from anogen.shell.diffusion import DiffusionSchedule, denoiser_from_ckpt, wrap_denoiser
    from anogen.shell.encoders import embed_shell
    from anogen.shell.events import types_from_cfg
    from anogen.shell.features import embed_windows
    from anogen.shell.morphology import KIND_ORDER, KIND_SLUG, kind_ref_indices, kinds_for_windows
    from anogen.shell.scaler import resolve_scaler
    from anogen.shell.steer import chunked_guided_ddim

    recipes = {
        "hybrid_needles": HYBRID_NEEDLES_PROTO_KINDS,
        "hybrid": HYBRID_PROTO_KINDS,
        "stratified": None,
    }
    seed0 = int(cfg.get("seed", 0))
    tau = float(json.loads((s4 / "summary.json").read_text())["tau"])
    lab = np.load(s0 / "labeled_arrays.npz")
    gal = np.load(s3 / "galleries.npz")
    meta = pd.read_csv(s0 / "labeled_windows.csv")
    fold_tab = pd.read_csv(s0 / "channel_fold_counts.csv")
    x_a, x_r = np.asarray(lab["anomaly"]), np.asarray(lab["rare"])
    fold_a, fold_r = np.asarray(lab["anomaly_fold"]), np.asarray(lab["rare_fold"])
    ch_a, ch_r = np.asarray(lab["anomaly_channel"]), np.asarray(lab["rare_channel"])
    x_cond = np.asarray(gal["cond"])
    cond_ch = np.asarray(gal["channel_idx"], dtype=np.int64)
    anom = meta[meta["kind"] == "anomaly"].reset_index(drop=True)
    event_a = anom["event_id"].astype(str).to_numpy()
    kind_lab = kinds_for_windows(x_a, anom, types_from_cfg(cfg))
    score_kw = dict(
        x_a=x_a, x_r=x_r, fold_a=fold_a, fold_r=fold_r, ch_a=ch_a, ch_r=ch_r, fold_tab=fold_tab, tau=tau
    )

    variants = dict(fcfg.get("variants") or DEFAULT_VARIANTS)
    only = fcfg.get("only")
    if only:
        variants = {k: v for k, v in variants.items() if k in set(only)}
    folds = [int(f) for f in (fcfg.get("folds") or sorted({int(f) for f in fold_a if int(f) >= 0}))]
    burn_bins = int(fcfg.get("burnin_bins", 256))
    n_boot = int(fcfg.get("n_boot", 1000))
    force = bool(fcfg.get("force", False))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    scfg = dict((cfg.get("shell") or {}).get("steer") or {})
    bsz = int(fcfg.get("bsz", scfg.get("n_sample", 128)))

    context = None
    prefix_info: dict[str, Any] = {"requested": False}
    if any(bool(v.get("burnin")) and not bool(v.get("start_from_noise")) for v in variants.values()):
        context, prefix_info = donor_context(cfg, x_cond, cond_ch, burn_bins)

    dens: dict[str, tuple[Any, Any, Any]] = {}

    def denoiser_for(path: str | None, flip: str | None) -> tuple[Any, Any, Any]:
        key = f"{path}|{flip}"
        if key not in dens:
            ckpt_path = _abs(path, root) if path else s1 / "denoiser.pt"
            den = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = wrap_denoiser(denoiser_from_ckpt(den, device=device_t), flip)
            dens[key] = (
                model,
                resolve_scaler(den, s0),
                DiffusionSchedule.from_ckpt(den, allow_nonlinear=True).to(device_t),
            )
        return dens[key]

    encs: dict[str, tuple[Any, dict[str, Any]]] = {}

    def encoder_for(name: str) -> tuple[Any, dict[str, Any]]:
        if name not in encs:
            encs[name] = _load_encoder(enc_dir / f"{name}_fold0.pt", x_cond, scfg, device, seed0)
        return encs[name]

    from anogen.shell.evaluation import (
        arp_coverage_ci,
        calibration_galleries,
        paired_arp_difference,
    )

    results: dict[str, Any] = {}
    fold0: dict[str, np.ndarray] = {}
    galleries: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for vname, spec in variants.items():
        spec = dict(spec)
        enc_name = str(spec.get("encoder", fcfg.get("encoder", "time_both")))
        recipe = str(spec.get("recipe", fcfg.get("recipe", "hybrid_needles")))
        if recipe not in recipes:
            raise ValueError(f"unknown recipe {recipe}")
        model, scaler, schedule = denoiser_for(spec.get("denoiser"), spec.get("flip"))
        enc, shell = encoder_for(enc_name)
        guided_kw = {k: v for k, v in spec.items() if k not in _PHASE_KEYS}
        if "guidance_t_window" in guided_kw:
            guided_kw["guidance_t_window"] = tuple(guided_kw["guidance_t_window"])
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
        common.update(guided_kw)
        common["nu"] = edit_nu(schedule, float(common["nu"]))
        if bool(spec.get("start_from_noise")):
            common["start_from_noise"] = True
            if bool(spec.get("burnin")):
                common["burnin_bins"] = burn_bins
                if str(spec.get("context", "before")) == "both":
                    common["burnin_bins_end"] = burn_bins
            if bool(spec.get("matched_noise")):
                common["noise_init_mean"], common["noise_init_std"] = channel_moments(
                    x_cond, cond_ch, scaler
                )
        rows = []
        for fold_id in folds:
            cache = out / f"{vname}_fold{fold_id}.npz"
            fseed = seed0 + 1000 * fold_id
            kind_z: dict[str, Any] = {}
            leaked: dict[str, bool] = {}
            for kind in KIND_ORDER:
                idx, lk = kind_ref_indices(kind_lab, fold_a, kind, query_fold=fold_id)
                if len(idx):
                    kind_z[kind] = torch.from_numpy(embed_shell(enc, x_a[idx], device))
                    leaked[kind] = bool(lk)
            active = [k for k in KIND_ORDER if k in kind_z]
            alloc = channel_stratified_alloc(cond_ch, active, seed=fseed)
            if cache.is_file() and not force:
                blob = np.load(cache, allow_pickle=True)
                x, h = np.asarray(blob["x"]), np.asarray(blob["h"])
            else:
                print(f"fixsweep {vname} fold={fold_id}", flush=True)
                torch.manual_seed(fseed)
                x, h = _generate_slices(
                    chunked_guided_ddim,
                    model,
                    enc,
                    x_cond,
                    cond_ch,
                    schedule,
                    common=common,
                    kind_z=kind_z,
                    alloc=alloc,
                    proto_kinds=recipes[recipe],
                    seed=fseed,
                    kind_order=KIND_ORDER,
                    slice_lam_anom=slice_lam_anom,
                    prefix=context[0] if (context is not None and bool(spec.get("burnin")) and not bool(spec.get("start_from_noise"))) else None,
                    suffix=context[1] if (context is not None and bool(spec.get("burnin")) and str(spec.get("context", "before")) == "both" and not bool(spec.get("start_from_noise"))) else None,
                    proto_shift_max=int(spec.get("proto_shift_max", 0) or 0),
                    contrast=_contrast_plan(
                        spec, kind_lab, fold_a, fold_id, x_a, ch_a, scaler, KIND_ORDER
                    ),
                )
                np.savez_compressed(
                    cache, x=x, h=h, channel_idx=cond_ch, kind_alloc=alloc.astype(str), spec=json.dumps(spec)
                )
            galleries[(vname, fold_id)] = (x, alloc.astype(str))
            if fold_id == folds[0]:
                fold0[vname] = x
            row = _score_fold(
                x,
                alloc.astype(str),
                cond_ch,
                fold_id=fold_id,
                score_kw=score_kw,
                event_a=event_a,
                kind_lab=kind_lab,
                scaler=scaler,
                n_boot=n_boot,
                seed=fseed,
                h=h,
                shell=shell,
                cfg=cfg,
                arp_coverage_ci=arp_coverage_ci,
                score_gallery=_score_gallery,
                kind_order=KIND_ORDER,
                kind_slug=KIND_SLUG,
            )
            row["leaked_kinds"] = [k for k, v in leaked.items() if v]
            rows.append(row)
        results[vname] = {
            "spec": spec,
            "encoder": enc_name,
            "recipe": recipe,
            "folds": rows,
            "mean": _mean_rows(rows),
            "diversity": float(mean_pairwise_distance(embed_windows(fold0[vname]))),
        }
        _write(out, results, cfg, _jsonable)

    # Paired differences vs the reproduction row (same queries, same torch seed).
    base = str(fcfg.get("baseline", "c1_repro"))
    if base in variants:
        for vname in variants:
            if vname == base:
                continue
            diffs = []
            for fold_id in folds:
                qm = _query_mask(ch_a, fold_a, fold_tab, fold_id)
                diffs.append(
                    paired_arp_difference(
                        embed_windows(x_a[qm]),
                        embed_windows(galleries[(vname, fold_id)][0]),
                        embed_windows(galleries[(base, fold_id)][0]),
                        event_a[qm],
                        tau=tau,
                        n_boot=n_boot,
                        seed=seed0 + fold_id,
                        aligned=True,
                    )
                )
            results[vname]["paired_vs_" + base] = diffs

    # Calibration rows: what "known faults", "no anomaly" and "noise" score.
    rng = np.random.default_rng(seed0)
    calib: dict[str, Any] = {}
    for fold_id in folds:
        qm = _query_mask(ch_a, fold_a, fold_tab, fold_id)
        q_emb = embed_windows(x_a[qm])
        for name, g in calibration_galleries(x_a, fold_a, x_cond, fold_id=fold_id, rng=rng).items():
            calib.setdefault(name, []).append(
                arp_coverage_ci(
                    q_emb, embed_windows(g), event_a[qm], tau=tau, n_boot=n_boot, seed=fold_id, resample_gallery=True
                )
            )
    results["_calibration"] = {
        k: {
            "arp_mean": float(np.mean([r["arp"] for r in v])),
            "coverage_mean": float(np.mean([r["coverage"] for r in v])),
            "folds": v,
        }
        for k, v in calib.items()
    }

    # One EDI union for this run (partition f).
    embs = {k: embed_windows(v) for k, v in fold0.items()}
    embs["donor"] = embed_windows(x_cond)
    for key in ("genias", "posthoc"):
        if key in gal.files:
            embs[key] = embed_windows(np.asarray(gal[key]))
    results["_edi_f"] = edi_by_method(embs)
    results["_meta"] = {
        "tau": tau,
        "folds": folds,
        "n_donors": int(len(x_cond)),
        "burnin": prefix_info,
        "n_boot": n_boot,
        "edi_note": "EDI partition f: union of every fold-0 gallery in this run + donor/genias/posthoc.",
    }
    _write(out, results, cfg, _jsonable)
    return {"ok": True, "skipped": False, "dir": str(out), "variants": list(variants)}


def _query_mask(ch_a: np.ndarray, fold_a: np.ndarray, fold_tab: pd.DataFrame, fold_id: int) -> np.ndarray:
    from anogen.phases.tune import _keep_mask

    return _keep_mask(ch_a, fold_a, fold_tab, fold_id=fold_id)


def _generate_slices(
    chunked: Any,
    model: Any,
    enc: Any,
    x0: np.ndarray,
    ch: np.ndarray,
    schedule: Any,
    *,
    common: dict[str, Any],
    kind_z: dict[str, Any],
    alloc: np.ndarray,
    proto_kinds: Any,
    seed: int,
    kind_order: tuple[str, ...],
    slice_lam_anom: Any,
    prefix: np.ndarray | None,
    proto_shift_max: int,
    suffix: np.ndarray | None = None,
    contrast: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``kindmix._stratified`` with per-donor arrays (prefix) sliced per kind.

    Same allocation and proto seeds as ``_stratified`` (proto seed = seed +
    17·(1 + kind index)). The frozen chanmix galleries were drawn without a
    torch seed, so ``c1_repro`` is a fresh draw of the same recipe, not a
    bit-for-bit copy: compare variants against ``c1_repro`` (shared torch seed),
    not against the frozen table.
    """
    labels = np.asarray(alloc).astype(str)
    default_lam = float(common.get("lam_anom", 1.0))
    out_x = np.zeros((len(x0), x0.shape[1]), dtype=np.float32)
    out_h = np.zeros(len(x0), dtype=np.float64)
    for kind in [k for k in kind_order if k in kind_z]:
        idx = np.flatnonzero(labels == kind)
        if len(idx) == 0:
            continue
        use_proto, lam_k = slice_lam_anom(kind, proto_kinds, default_lam)
        kw = dict(common)
        if prefix is not None:
            kw["burnin_prefix"] = prefix[idx]
        if suffix is not None:
            kw["burnin_suffix"] = suffix[idx]
        if use_proto:
            kw.update(
                lam_anom=lam_k,
                ref_anom=kind_z[kind],
                anom_energy_kind="proto",
                anom_proto_seed=seed + 17 * (1 + kind_order.index(kind)),
                anom_proto_shift_max=int(proto_shift_max),
                anom_proto_shift_seed=seed + 31 * (1 + kind_order.index(kind)),
            )
        else:
            kw.update(lam_anom=0.0, lam_rare=0.0)
        plan = (contrast or {}).get(kind)
        if plan is not None:
            from anogen.shell.contrast import sample_targets

            _, delta, wts = sample_targets(
                plan["kind"],
                len(idx),
                x0.shape[1],
                plan["amplitudes"],
                rng=np.random.default_rng(seed + 53 * (1 + kind_order.index(kind))),
            )
            kw.update(contrast_weights=wts, contrast_target=delta, lam_contrast=plan["lam"])
            if plan["only"]:
                for key in ("ref_anom", "anom_energy_kind", "anom_proto_seed", "anom_proto_shift_max", "anom_proto_shift_seed"):
                    kw.pop(key, None)
                kw["lam_anom"] = 0.0
        x, h = chunked(model, enc, x0[idx], ch[idx], schedule, **kw)
        out_x[idx] = x
        out_h[idx] = h
    return out_x, out_h


def _score_fold(
    x: np.ndarray,
    alloc: np.ndarray,
    cond_ch: np.ndarray,
    *,
    fold_id: int,
    score_kw: dict[str, Any],
    event_a: np.ndarray,
    kind_lab: np.ndarray,
    scaler: Any,
    n_boot: int,
    seed: int,
    h: np.ndarray | None,
    shell: dict[str, Any] | None,
    cfg: dict[str, Any],
    arp_coverage_ci: Any,
    score_gallery: Any,
    kind_order: tuple[str, ...],
    kind_slug: dict[str, str],
) -> dict[str, Any]:
    from anogen.shell.evaluation import envelope_frequency_report, position_report
    from anogen.shell.features import embed_windows
    from anogen.shell.realism import (
        channel_span_normalize,
        classifier_two_sample_test,
        diag_cusum,
        diag_first_difference,
        diag_persistence,
    )

    x_a, fold_a, ch_a = score_kw["x_a"], score_kw["fold_a"], score_kw["ch_a"]
    row: dict[str, Any] = {"fold": int(fold_id)}
    row.update(score_gallery(x, only_fold=fold_id, **score_kw))
    qm = _query_mask(ch_a, fold_a, score_kw["fold_tab"], fold_id)
    ci = arp_coverage_ci(
        embed_windows(x_a[qm]),
        embed_windows(x),
        event_a[qm],
        tau=float(score_kw["tau"]),
        n_boot=n_boot,
        seed=seed,
        resample_gallery=True,
    )
    row["arp_ci"], row["coverage_ci"] = ci["arp_ci"], ci["coverage_ci"]
    if h is not None and shell is not None:
        row["occupancy"] = float(np.mean(np.abs(h - float(shell["Q_q"])) <= float(shell["delta"])))

    real_m = fold_a == int(fold_id)
    u_real = channel_span_normalize(x_a[real_m], ch_a[real_m], scaler)
    real_pos = _positions(u_real)

    def diag(xs: np.ndarray, chs: np.ndarray, real_u: np.ndarray) -> dict[str, float]:
        u = channel_span_normalize(xs, chs, scaler)
        d = {
            **envelope_frequency_report(u, real_u),
            **position_report(u, real_positions=_positions(real_u)),
            "cusum_mean": float(np.mean(diag_cusum(u, min_segment=8)["contrast"])),
            "diff_p999": float(np.mean(diag_first_difference(u)["p999"])),
            "persistence_abs": float(np.mean(np.abs(diag_persistence(u)))),
        }
        return d

    row["diag"] = diag(x, cond_ch, u_real)
    row["diag_real"] = {
        **envelope_frequency_report(u_real),
        **position_report(u_real),
        "cusum_mean": float(np.mean(diag_cusum(u_real, min_segment=8)["contrast"])),
        "diff_p999": float(np.mean(diag_first_difference(u_real)["p999"])),
        "persistence_abs": float(np.mean(np.abs(diag_persistence(u_real)))),
    }
    row["diag_kind"] = {}
    for kind in kind_order:
        gm = alloc == kind
        if not np.any(gm):
            continue
        rm = real_m & (kind_lab == kind)
        real_u_k = channel_span_normalize(x_a[rm], ch_a[rm], scaler) if np.any(rm) else None
        row["diag_kind"][kind_slug.get(kind, kind)] = {
            **diag(x[gm], cond_ch[gm], real_u_k if real_u_k is not None else u_real),
            "n_real": int(rm.sum()),
        }
    rcfg = dict((cfg.get("shell") or {}).get("realism") or {})
    try:
        c2 = classifier_two_sample_test(
            x_a[real_m],
            x,
            ch_a[real_m],
            cond_ch,
            event_a[real_m],
            np.arange(len(x)),
            scaler,
            seeds=[int(s) for s in (rcfg.get("seeds") or [0, 1, 2, 3, 4])],
            test_size=float(rcfg.get("c2st_test_size", 0.3)),
            n_perm=int(rcfg.get("c2st_permutations", 200)),
        )
        accs = [r["balanced_accuracy"] for r in c2]
        row["c2st_median"] = float(np.median(accs))
        row["c2st_reject_rate"] = float(np.mean([r["permutation_p"] < 0.05 for r in c2]))
    except ValueError as exc:
        row["c2st_error"] = str(exc)
    return row


def _positions(u: np.ndarray) -> np.ndarray:
    from anogen.shell.evaluation import peak_positions

    return peak_positions(u)


def _mean_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("arp_anomaly", "coverage_anomaly", "gap", "occupancy", "c2st_median", "c2st_reject_rate")
    out: dict[str, Any] = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            out[k] = float(np.mean(vals))
    for k in rows[0].get("diag", {}) if rows else ():
        vals = [r["diag"][k] for r in rows if k in r.get("diag", {})]
        out["diag_" + k] = float(np.mean(vals))
    return out


def _write(out: Path, results: dict[str, Any], cfg: dict[str, Any], jsonable: Any) -> None:
    (out / "summary.json").write_text(json.dumps(jsonable(results), indent=2) + "\n")
    (out / "TABLE.md").write_text(fixsweep_table(results))


def fixsweep_table(results: dict[str, Any]) -> str:
    """Markdown table from a fixsweep summary (also used by docs generation)."""
    real = None
    lines = [
        "# Fix sweep (generated by `anogen ... fixsweep`)",
        "",
        "Mean over folds. ARP CI: event-cluster two-way bootstrap, mean of fold bounds.",
        "Real-anomaly targets in the first row; `start`/`end` = share of dominant",
        "peaks in the first/last 10 % of the window (uniform 0.10).",
        "",
        "| variant | ARP | ARP 95% CI | Cov@τ | C2ST | env exit | env p95 | CUSUM | diff p99.9 | start | end | pos H |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, res in results.items():
        if name.startswith("_"):
            continue
        rows = res.get("folds") or []
        if not rows:
            continue
        if real is None and rows[0].get("diag_real"):
            dr = rows[0]["diag_real"]
            real = dr
            lines.append(
                f"| real anomalies (fold {rows[0]['fold']}) | | | | 0.5 | {dr['exit_frac']:.3f} | "
                f"{dr['excess_p95_given_exit']:.2f} | {dr['cusum_mean']:+.3f} | {dr['diff_p999']:.3f} | "
                f"{dr['peak_at_start']:.3f} | {dr['peak_at_end']:.3f} | {dr['position_entropy']:.3f} |"
            )
        m = res.get("mean") or {}
        lo = np.mean([r["arp_ci"][0] for r in rows])
        hi = np.mean([r["arp_ci"][1] for r in rows])
        lines.append(
            f"| {name} | {m.get('arp_anomaly', float('nan')):.3f} | [{lo:.3f}, {hi:.3f}] | "
            f"{m.get('coverage_anomaly', float('nan')):.3f} | {m.get('c2st_median', float('nan')):.3f} | "
            f"{m.get('diag_exit_frac', float('nan')):.3f} | {m.get('diag_excess_p95_given_exit', float('nan')):.2f} | "
            f"{m.get('diag_cusum_mean', float('nan')):+.3f} | {m.get('diag_diff_p999', float('nan')):.3f} | "
            f"{m.get('diag_peak_at_start', float('nan')):.3f} | {m.get('diag_peak_at_end', float('nan')):.3f} | "
            f"{m.get('diag_position_entropy', float('nan')):.3f} |"
        )
    cal = results.get("_calibration") or {}
    if cal:
        lines += ["", "Calibration rows (same queries, same CI method):", ""]
        for k, v in cal.items():
            lines.append(f"- {k}: ARP {v['arp_mean']:.3f}, Cov@τ {v['coverage_mean']:.3f}")
    edi = results.get("_edi_f") or {}
    if edi:
        lines += ["", "EDI (partition **f**): " + ", ".join(f"{k} {v:.2f}" for k, v in edi.items())]
    return "\n".join(lines) + "\n"


def _contrast_plan(
    spec: dict[str, Any],
    kind_lab: np.ndarray,
    fold_a: np.ndarray,
    fold_id: int,
    x_a: np.ndarray,
    ch_a: np.ndarray,
    scaler: Any,
    kind_order: tuple[str, ...],
) -> dict[str, Any] | None:
    """Per-kind contrast targets with amplitudes read off train-fold examples.

    Uses the same reference rule as the proto term (``kind_ref_indices``:
    event-OOF, falling back to in-fold with ``leaked``), measured in the
    denoiser's scaled units.
    """
    mapping = spec.get("contrast")
    if not mapping:
        return None
    from anogen.shell.contrast import measure_contrast
    from anogen.shell.morphology import kind_ref_indices

    plan: dict[str, Any] = {}
    for kind, ckind in dict(mapping).items():
        if kind not in kind_order:
            raise ValueError(f"contrast kind {kind!r} is not an ESA kind")
        idx, leaked = kind_ref_indices(kind_lab, fold_a, kind, query_fold=fold_id)
        if len(idx) == 0:
            continue
        xs = scaler.transform(x_a[idx], ch_a[idx]) if scaler is not None else x_a[idx]
        amps, _ = measure_contrast(str(ckind), xs)
        plan[kind] = {
            "kind": str(ckind),
            "amplitudes": amps,
            "lam": float(spec.get("lam_contrast", 1.0)),
            "only": bool(spec.get("contrast_only", False)),
            "leaked": bool(leaked),
        }
    return plan


def edit_nu(schedule: Any, nu: float) -> float:
    """ν giving the frozen edit's *noise level* under ``schedule``.

    Identity for the linear schedule (every frozen checkpoint). For another
    schedule (s1v2 is geometric), ν = 0.2 would be a much lighter edit, so the
    ν whose ᾱ matches the linear schedule's ᾱ at ν is returned instead.
    """
    from anogen.shell.diffusion import DiffusionSchedule, nu_for_noise_level

    n = int(schedule.betas.numel())
    lin = DiffusionSchedule.linear(n).alpha_bar
    if np.allclose(schedule.alpha_bar.detach().cpu().numpy(), lin.numpy()):
        return float(nu)
    target = float(lin[int(round(float(nu) * (n - 1)))])
    return nu_for_noise_level(schedule.alpha_bar, target)


def channel_moments(x: np.ndarray, ch: np.ndarray, scaler: Any) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std of windows in the denoiser's scaled units."""
    ch = np.asarray(ch, dtype=np.int64)
    xs = scaler.transform(x, ch) if scaler is not None else np.asarray(x, dtype=np.float32)
    n = int(ch.max()) + 1
    mean = np.array([xs[ch == c].mean() if np.any(ch == c) else 0.0 for c in range(n)], dtype=np.float32)
    std = np.array([xs[ch == c].std() if np.any(ch == c) else 1.0 for c in range(n)], dtype=np.float32)
    return mean, std


def donor_context(
    cfg: dict[str, Any],
    x_cond: np.ndarray,
    cond_ch: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Real telemetry immediately before and after each S3 donor, for burn-in.

    S3 drew the donors with ``default_rng(seed).choice`` per channel from
    ``nominal_index.csv``; the same draw is replayed here and checked against
    the stored ``cond`` windows (exact match required). Prefixes that would
    leave the panel, contain NaN / empty bins, or overlap an anomaly span (+
    guard) fall back to the time-reversed first ``bins`` of the donor, and
    are counted in the returned info.
    """
    from anogen.config import data_root, panel_path
    from anogen.shell.data import load_metadata, load_panel
    from anogen.shell.events import assign_splits, build_event_table
    from anogen.shell.windows import occupancy_mask, window_finite

    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = Path(cfg.get("s0_dir", root / "results/shell_s0"))
    s0 = s0 if s0.is_absolute() else root / s0
    splits = cfg.get("splits") or {}
    proto = json.loads((s0 / "protocol.json").read_text())
    width, n_gen = int(proto["W"]), int(proto["N_generate"])
    panel = load_panel(
        panel_path(cfg),
        channels=list(cfg.get("channels") or []),
        official_train_end=splits.get("official_train_end", "2007-01-01"),
        allow_test_telemetry=False,
        bin_seconds=int(cfg.get("bin_seconds", 30)),
    )
    nominal = pd.read_csv(s0 / "nominal_index.csv")
    ch_nom = nominal["channel_idx"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    picks = []
    for cidx in range(panel.k):
        idx = np.where(ch_nom == cidx)[0]
        picks.append(rng.choice(idx, size=n_gen, replace=False))
    cond_idx = np.concatenate(picks)
    starts = nominal["start"].to_numpy(dtype=np.int64)[cond_idx]
    chans = ch_nom[cond_idx]
    replay = np.stack([panel.Y[s : s + width, c] for s, c in zip(starts, chans, strict=True)])
    if replay.shape != np.asarray(x_cond).shape or not np.allclose(replay, x_cond, equal_nan=False):
        raise RuntimeError("could not replay the S3 donor draw; burn-in prefixes would be misaligned")
    if not np.array_equal(chans, np.asarray(cond_ch)):
        raise RuntimeError("replayed donor channels differ from S3 channel_idx")
    meta = load_metadata(data_root(cfg))
    events = assign_splits(
        build_event_table(meta["labels"], meta["anomaly_types"]),
        path_train_end=splits.get("path_train_end", "2006-10-01"),
        official_train_end=splits.get("official_train_end", "2007-01-01"),
    )
    occ = occupancy_mask(panel, events, guard_bins=int(proto["guard_bins"]), occupy_rares=False)
    b = int(bins)
    prefix = np.empty((len(starts), b), dtype=np.float32)
    suffix = np.empty((len(starts), b), dtype=np.float32)
    fb_pre = np.zeros(len(starts), dtype=bool)
    fb_post = np.zeros(len(starts), dtype=bool)
    for i, (s, c) in enumerate(zip(starts, chans, strict=True)):
        lo = int(s) - b
        ok = lo >= 0 and window_finite(panel.Y, panel.counts, lo, int(c), b) and not occ[lo : int(s)].any()
        if ok:
            prefix[i] = panel.Y[lo : int(s), int(c)]
        else:
            prefix[i] = np.asarray(x_cond[i, 1 : b + 1])[::-1]
            fb_pre[i] = True
        hi0 = int(s) + width
        ok = (
            hi0 + b <= panel.T
            and window_finite(panel.Y, panel.counts, hi0, int(c), b)
            and not occ[hi0 : hi0 + b].any()
        )
        if ok:
            suffix[i] = panel.Y[hi0 : hi0 + b, int(c)]
        else:
            suffix[i] = np.asarray(x_cond[i, width - b - 1 : width - 1])[::-1]
            fb_post[i] = True
    return (prefix, suffix), {
        "requested": True,
        "bins": b,
        "n": int(len(prefix)),
        "fallback_reflect_prefix": int(fb_pre.sum()),
        "fallback_reflect_suffix": int(fb_post.sum()),
        "note": "real neighbouring telemetry; reflect fallback where unavailable, anomalous or past the cut",
    }
