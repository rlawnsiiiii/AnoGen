"""Dev-only supervised augmentation: do generated anomalies help a detector?

Event-OOF, sealed test stays closed. Isolated results/shell_augdetect/.
Does not overwrite shell_s3 or shell_s4. Quiet-kind recipe was not promoted.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anogen.config import REPO_ROOT
from anogen.phases.encscore import _abs, _jsonable
from anogen.shell.detector import eval_detector, fit_detector, needs_torch, paired_event_bootstrap
from anogen.shell.events import types_from_cfg
from anogen.shell.morphology import kinds_for_windows
from anogen.shell.scaler import load_minmax

C1_GALLERY = "time_both_hybrid_needles"
C3_GALLERY = "time_recon_combined"

DEFAULT_ARMS = (
    "real_only",
    "posthoc",
    "taxonomy",
    "cutaddpaste",
    "genias",
    "unguided",
    "genfsdiff_c1",
    "genfsdiff_c3",
    "c1_plus_real",
    "c3_plus_real",
)

_SHIFT = "real level shift"


def assert_isolated(out: Path, *locked: Path) -> None:
    if out.resolve() in {p.resolve() for p in locked}:
        raise RuntimeError("augdetect must not write into shell_s3 or shell_s4")


def run_augdetect(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    s4 = _abs(cfg.get("s4_dir", root / "results/shell_s4"), root)
    hybrid = _abs(
        cfg.get("kindmix_score_hybrid_dir", root / "results/shell_kindmix_score_hybrid"),
        root,
    )
    enc_score = _abs(cfg.get("enc_score_dir", root / "results/shell_enc_score"), root)
    genbase = _abs(cfg.get("genbase_dir", root / "results/shell_genbase"), root)
    out = _abs(cfg.get("augdetect_dir", root / "results/shell_augdetect"), root)
    plots = out / "plots"
    docs_plots = root / "docs" / "augdetect"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    docs_plots.mkdir(parents=True, exist_ok=True)
    assert_isolated(out, s3, s4)

    if not needs_torch():
        report = {"ok": False, "skipped": True, "reason": "torch is not installed"}
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    scaler_path = s0 / "minmax_scaler.npz"
    needed = {
        "S0": s0 / "labeled_arrays.npz",
        "S3": s3 / "galleries.npz",
        "S4": s4 / "shell_gallery.npz",
        "scaler": scaler_path,
        "meta": s0 / "labeled_windows.csv",
    }
    for label, path in needed.items():
        if not path.is_file():
            report = {"ok": False, "skipped": True, "reason": f"{label} missing: {path}"}
            (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            return report

    acfg = dict((cfg.get("shell") or {}).get("augdetect") or {})
    arms = tuple(acfg.get("arms") or DEFAULT_ARMS)
    seeds = [int(s) for s in (acfg.get("seeds") or [0, 1, 2, 3, 4])]
    n_synth = int(acfg.get("n_synth", 256))
    steps = int(acfg.get("steps", 400))
    hidden = int(acfg.get("hidden", 32))
    batch_size = int(acfg.get("batch_size", 32))
    lr = float(acfg.get("lr", 1e-3))
    far = float(acfg.get("target_far", 0.01))
    n_boot = int(acfg.get("n_boot", 1000))
    n_nom_test = int(acfg.get("n_nominal_test", 512))
    device = str(acfg.get("device") or "")
    seed0 = int(cfg.get("seed", 0))
    # Synthetic positives for test fold k only from donors whose S3 index is not
    # a fold-k test nominal (index % 3 == k). Before 2026-10-03 every arm drew
    # from all 1536 donors, so a test negative's own edited copy could be a
    # training positive. Set false only to reproduce docs/AUGDETECT.md.
    donor_disjoint = bool(acfg.get("donor_disjoint", True))
    shift_aug = int(acfg.get("shift_aug", 0))
    flip_test = bool(acfg.get("flip_test", True))
    # Optional label-free score from `anogen diffdetect`, rank-fused with the CNN.
    diff_scores = None
    if acfg.get("fuse_diffscore"):
        dpath = _abs(acfg["fuse_diffscore"], root)
        if not dpath.is_file():
            raise FileNotFoundError(f"fuse_diffscore configured but missing: {dpath}")
        blob = np.load(dpath)
        fresh = np.load(dpath.parent / "fresh_nominal.npz")
        diff_scores = {k: np.asarray(blob[k]) for k in ("anomaly", "rare", "fresh")}
        # Per-noise-level scores (diffdetect pool/fuse variants): re-fused per fold
        # below so the test negatives never enter the standardization.
        if "fresh_t" in blob.files and "fuse" in blob.files and str(blob["fuse"]) != "mean":
            diff_scores.update({k + "_t": np.asarray(blob[k + "_t"]) for k in ("anomaly", "rare", "fresh")})
            diff_scores["fuse"] = str(blob["fuse"])
        diff_scores["fresh_x"] = np.asarray(fresh["x"])
        diff_scores["fresh_ch"] = np.asarray(fresh["channel_idx"], dtype=np.int64)
        _lab = np.load(s0 / "labeled_arrays.npz")
        if len(diff_scores["anomaly"]) != len(_lab["anomaly"]) or len(diff_scores["rare"]) != len(_lab["rare"]):
            raise ValueError("diffdetect scores are not aligned with S0 labeled_arrays")
        if len(diff_scores["fresh"]) != len(diff_scores["fresh_x"]):
            raise ValueError("diffdetect fresh scores and fresh windows differ in length")

    lab = np.load(s0 / "labeled_arrays.npz")
    gal = np.load(s3 / "galleries.npz")
    s4g = np.load(s4 / "shell_gallery.npz")
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
    if len(anom) != len(x_a):
        report = {
            "ok": False,
            "skipped": True,
            "reason": f"labeled anomaly rows {len(anom)} != arrays {len(x_a)}",
        }
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    event_a = anom["event_id"].astype(str).to_numpy()
    types = types_from_cfg(cfg)
    kind_a = kinds_for_windows(x_a, anom, types)
    scaler = load_minmax(scaler_path)

    galleries = {
        "posthoc": (np.asarray(gal["posthoc"]), cond_ch),
        "genias": (np.asarray(gal["genias"]), cond_ch),
        "unguided": (np.asarray(s4g["unguided"]), cond_ch),
    }
    cap_p = genbase / "cutaddpaste.npz"
    tax_p = genbase / "taxonomy.npz"
    if cap_p.is_file():
        galleries["cutaddpaste"] = (np.asarray(np.load(cap_p)["x"]), cond_ch)
    if tax_p.is_file():
        galleries["taxonomy"] = (np.asarray(np.load(tax_p)["x"]), cond_ch)

    folds = sorted({int(f) for f in fold_a.tolist() if int(f) >= 0})
    rows: list[dict[str, Any]] = []
    for arm in arms:
        for fold_id in folds:
            leaked = _arm_leaks_fold0(arm) and fold_id == 0
            for seed in seeds:
                print(f"augdetect {arm} fold={fold_id} seed={seed}", flush=True)
                try:
                    row = _run_one(
                        arm=arm,
                        fold_id=fold_id,
                        seed=seed,
                        x_a=x_a,
                        x_r=x_r,
                        fold_a=fold_a,
                        fold_r=fold_r,
                        ch_a=ch_a,
                        ch_r=ch_r,
                        event_a=event_a,
                        kind_a=kind_a,
                        x_cond=x_cond,
                        cond_ch=cond_ch,
                        galleries=galleries,
                        hybrid=hybrid,
                        enc_score=enc_score,
                        scaler=scaler,
                        n_synth=n_synth,
                        n_nom_test=n_nom_test,
                        steps=steps,
                        hidden=hidden,
                        batch_size=batch_size,
                        lr=lr,
                        far=far,
                        n_boot=n_boot,
                        device=device or None,
                        leaked=leaked,
                        donor_disjoint=donor_disjoint,
                        shift_aug=shift_aug,
                        flip_test=flip_test,
                        diff_scores=diff_scores,
                    )
                except FileNotFoundError as exc:
                    row = {
                        "arm": arm,
                        "fold": fold_id,
                        "seed": seed,
                        "ok": False,
                        "reason": str(exc),
                    }
                rows.append(row)

    _assert_no_nominal_leak(rows)
    summary = _aggregate(rows, arms=arms, seeds=seeds, folds=folds)
    csv_path = out / "metrics.csv"
    _write_csv(csv_path, rows)
    written = _plot_summary(plots, summary)
    for png in written:
        shutil.copy2(plots / png, docs_plots / png)
    index = _index_text(summary, written)
    (docs_plots / "INDEX.txt").write_text(index)
    (out / "INDEX.txt").write_text(index)

    report = {
        "ok": True,
        "skipped": False,
        "primary": "event_recall@1%FAR",
        "n_synth": n_synth,
        "steps": steps,
        "seeds": seeds,
        "arms": list(arms),
        "target_far": far,
        "summary": summary,
        "n_rows": len(rows),
        "plots": written,
        "note": (
            "Dev-only. Sealed test stays closed. Train positives never include "
            "test-fold events. Test nominals are disjoint from train nominals "
            "(S3 cond split by index % 3). genfsdiff_* loads only fold-k "
            "gallery. C1 fold 0 is the level-shift proto leak — reported "
            "separately, not averaged silently. Quiet recipe was not promoted. "
            f"seed0={seed0}."
        ),
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return report


def _arm_leaks_fold0(arm: str) -> bool:
    return arm in {"genfsdiff_c1", "c1_plus_real", "synth_only"}


def _run_one(
    *,
    arm: str,
    fold_id: int,
    seed: int,
    x_a: np.ndarray,
    x_r: np.ndarray,
    fold_a: np.ndarray,
    fold_r: np.ndarray,
    ch_a: np.ndarray,
    ch_r: np.ndarray,
    event_a: np.ndarray,
    kind_a: np.ndarray,
    x_cond: np.ndarray,
    cond_ch: np.ndarray,
    galleries: dict[str, tuple[np.ndarray, np.ndarray]],
    hybrid: Path,
    enc_score: Path,
    scaler: Any,
    n_synth: int,
    n_nom_test: int,
    steps: int,
    hidden: int,
    batch_size: int,
    lr: float,
    far: float,
    n_boot: int,
    device: str | None,
    leaked: bool,
    donor_disjoint: bool = False,
    shift_aug: int = 0,
    flip_test: bool = False,
    diff_scores: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 1000 * fold_id)
    train_a = fold_a != fold_id
    test_a = fold_a == fold_id
    train_r = fold_r != fold_id
    test_r = fold_r == fold_id
    train_ev = set(event_a[train_a].tolist())
    test_ev = set(event_a[test_a].tolist())
    if train_ev & test_ev:
        raise RuntimeError(f"event leak fold {fold_id}: {train_ev & test_ev}")

    nom_test_m = np.arange(len(x_cond)) % 3 == fold_id
    nom_train_m = ~nom_test_m
    x_nom_tr, ch_nom_tr = x_cond[nom_train_m], cond_ch[nom_train_m]
    x_nom_te, ch_nom_te = x_cond[nom_test_m], cond_ch[nom_test_m]
    nom_te_idx = np.flatnonzero(nom_test_m)
    if len(x_nom_te) > n_nom_test:
        pick = rng.choice(len(x_nom_te), size=n_nom_test, replace=False)
        x_nom_te, ch_nom_te = x_nom_te[pick], ch_nom_te[pick]
        nom_te_idx = nom_te_idx[pick]
    if not len(x_nom_tr) or not len(x_nom_te):
        raise RuntimeError("nominal split is empty")

    plus_real = arm.endswith("_plus_real") or arm == "real_only"
    use_real = plus_real
    x_pos_real = x_a[train_a] if use_real else np.zeros((0, x_a.shape[1]), dtype=x_a.dtype)
    ch_pos_real = ch_a[train_a] if use_real else np.zeros(0, dtype=ch_a.dtype)

    x_syn, ch_syn = _synth_for_arm(
        arm,
        fold_id=fold_id,
        rng=rng,
        n_synth=0 if arm == "real_only" else n_synth,
        galleries=galleries,
        hybrid=hybrid,
        enc_score=enc_score,
        cond_ch=cond_ch,
        exclude_donor_fold=fold_id if donor_disjoint else None,
    )
    if arm == "real_only":
        x_pos, ch_pos = x_pos_real, ch_pos_real
    elif plus_real:
        x_pos = np.concatenate([x_pos_real, x_syn], axis=0)
        ch_pos = np.concatenate([ch_pos_real, ch_syn], axis=0)
    else:
        x_pos, ch_pos = x_syn, ch_syn
    if len(x_pos) == 0:
        raise RuntimeError(f"{arm} produced no train positives")

    # 20 % of the training nominals are always kept out of fitting (so the CNN
    # rows do not depend on whether fusion is on); they serve as the CNN's
    # reference distribution for rank fusion (training negatives would sit
    # below every test score and saturate the empirical CDF).
    ref_nom = np.zeros(len(x_nom_tr), dtype=bool)
    ref_nom[rng.choice(len(x_nom_tr), size=max(1, len(x_nom_tr) // 5), replace=False)] = True
    x_fit_nom, ch_fit_nom = x_nom_tr[~ref_nom], ch_nom_tr[~ref_nom]
    x_neg = np.concatenate([x_fit_nom, x_r[train_r]], axis=0) if int(train_r.sum()) else x_fit_nom
    ch_neg = np.concatenate([ch_fit_nom, ch_r[train_r]], axis=0) if int(train_r.sum()) else ch_fit_nom

    x_pos_s = scaler.transform(x_pos, ch_pos)
    x_neg_s = scaler.transform(x_neg, ch_neg)
    x_te_a = scaler.transform(x_a[test_a], ch_a[test_a])
    x_te_n = scaler.transform(x_nom_te, ch_nom_te)
    x_te_r = (
        scaler.transform(x_r[test_r], ch_r[test_r])
        if int(test_r.sum())
        else np.zeros((0, x_a.shape[1]), dtype=np.float32)
    )

    clf = fit_detector(
        x_pos_s,
        x_neg_s,
        seed=seed,
        steps=steps,
        hidden=hidden,
        batch_size=batch_size,
        lr=lr,
        device=device,
        shift_aug=shift_aug,
    )
    ev = eval_detector(
        clf,
        x_anom=x_te_a,
        x_nom=x_te_n,
        x_rare=x_te_r,
        event_id=event_a[test_a],
        kind=kind_a[test_a],
        far=far,
        device=device,
        n_boot=n_boot,
        seed=seed,
    )
    kind_hits = ev.pop("event_hits")
    fused: dict[str, Any] = {}
    if diff_scores is not None:
        from anogen.shell.detector import eval_scores, predict_logits, rank_fuse

        from anogen.phases.diffdetect import _channel_stats

        # Negatives for the fused evaluation are diffdetect's fresh windows (held
        # out from S1 and never seen by the CNN): index % 3 == k is the test part,
        # the rest standardizes the diffusion score per channel.
        fx, fch, fs = diff_scores["fresh_x"], diff_scores["fresh_ch"], diff_scores["fresh"]
        f_test = np.arange(len(fx)) % 3 == fold_id
        d_anom, d_rare = diff_scores["anomaly"], diff_scores["rare"]
        if "fresh_t" in diff_scores:
            from anogen.phases.diffdetect import fuse_scores

            ref_t, ref_c, how = diff_scores["fresh_t"][~f_test], fch[~f_test], diff_scores["fuse"]
            fs = fuse_scores(diff_scores["fresh_t"], fch, ref_t, ref_c, how)
            d_anom = fuse_scores(diff_scores["anomaly_t"], ch_a, ref_t, ref_c, how)
            d_rare = fuse_scores(diff_scores["rare_t"], ch_r, ref_t, ref_c, how)
        mu, sdv = _channel_stats(fs[~f_test], fch[~f_test])

        def zd(s: np.ndarray, c: np.ndarray) -> np.ndarray:
            return (s - mu[c]) / sdv[c]

        ref_cnn = predict_logits(clf, scaler.transform(x_nom_tr[ref_nom], ch_nom_tr[ref_nom]), device=device)
        ref_diff = zd(fs[~f_test], fch[~f_test])

        def fuse(xs: np.ndarray, ds: np.ndarray) -> np.ndarray:
            return rank_fuse(predict_logits(clf, xs, device=device), ds, reference=[ref_cnn, ref_diff])

        evf = eval_scores(
            fuse(x_te_a, zd(d_anom[test_a], ch_a[test_a])),
            fuse(scaler.transform(fx[f_test], fch[f_test]), zd(fs[f_test], fch[f_test])),
            fuse(x_te_r, zd(d_rare[test_r], ch_r[test_r])) if int(test_r.sum()) else np.zeros(0),
            event_id=event_a[test_a],
            kind=kind_a[test_a],
            far=far,
        )
        # CNN alone on the same fresh negatives, so fused-vs-CNN is like for like.
        evc = eval_scores(
            predict_logits(clf, x_te_a, device=device),
            predict_logits(clf, scaler.transform(fx[f_test], fch[f_test]), device=device),
            predict_logits(clf, x_te_r, device=device) if int(test_r.sum()) else np.zeros(0),
            event_id=event_a[test_a],
            kind=kind_a[test_a],
            far=far,
        )
        fused = {
            "event_recall_fused": evf["event_recall"],
            "ap_fused": evf["ap"],
            "rare_far_fused": evf["rare_far"],
            "event_hits_fused": evf["event_hits"],
            "event_recall_cnn_freshneg": evc["event_recall"],
            "ap_cnn_freshneg": evc["ap"],
            "event_hits_cnn_freshneg": evc["event_hits"],
        }
    flipped: dict[str, Any] = {}
    if flip_test:
        # Position-sensitivity check: the same test set read backwards. A
        # detector that learned "anomaly = window start" loses recall here.
        evf = eval_detector(
            clf,
            x_anom=x_te_a[:, ::-1].copy(),
            x_nom=x_te_n[:, ::-1].copy(),
            x_rare=x_te_r[:, ::-1].copy(),
            event_id=event_a[test_a],
            kind=kind_a[test_a],
            far=far,
            device=device,
            n_boot=0,
            seed=seed,
        )
        half = x_te_a.shape[1] // 2
        evr = eval_detector(
            clf,
            x_anom=np.roll(x_te_a, half, axis=1),
            x_nom=np.roll(x_te_n, half, axis=1),
            x_rare=np.roll(x_te_r, half, axis=1),
            event_id=event_a[test_a],
            kind=kind_a[test_a],
            far=far,
            device=device,
            n_boot=0,
            seed=seed,
        )
        # flip moves the event *and* reverses its shape (an up-step becomes a
        # down-step); roll by W/2 moves it with the shape kept, at the price of
        # a seam. A drop in both is position reliance.
        flipped = {
            "event_recall_flipped": evf["event_recall"],
            "window_recall_flipped": evf["window_recall"],
            "ap_flipped": evf["ap"],
            "event_recall_rolled": evr["event_recall"],
            "ap_rolled": evr["ap"],
        }
    row = {
        "arm": arm,
        "fold": int(fold_id),
        "seed": int(seed),
        "ok": True,
        "leaked": bool(leaked),
        "n_pos": int(len(x_pos)),
        "n_pos_real": int(len(x_pos_real)),
        "n_pos_synth": int(len(x_syn) if arm != "real_only" else 0),
        "n_neg": int(len(x_neg)),
        "n_nom_train": int(len(x_nom_tr)),
        "n_fit_nominals": int((~ref_nom).sum()),
        "n_nom_test": int(len(x_nom_te)),
        "n_nom_train_idx": nom_train_m.astype(np.int8).tobytes().hex()[:16],
        "n_nom_test_idx": nom_test_m.astype(np.int8).tobytes().hex()[:16],
        "shift_frac_test": float(np.mean(kind_a[test_a] == _SHIFT)) if int(test_a.sum()) else 0.0,
        **{k: v for k, v in ev.items() if k != "bootstrap"},
        "event_recall_ci_lo": ev["bootstrap"]["lo"],
        "event_recall_ci_hi": ev["bootstrap"]["hi"],
        "event_recall_boot_median": ev["bootstrap"]["median"],
        "n_events_hit": int(sum(1 for v in kind_hits.values() if v)),
        "per_kind_recall": ev["per_kind_recall"],
        "train_events": sorted(train_ev),
        "test_events": sorted(test_ev),
        "event_hits": kind_hits,
        "donor_disjoint": bool(donor_disjoint),
        "shift_aug": int(shift_aug),
        **flipped,
        **fused,
    }
    return row


def _synth_for_arm(
    arm: str,
    *,
    fold_id: int,
    rng: np.random.Generator,
    n_synth: int,
    galleries: dict[str, tuple[np.ndarray, np.ndarray]],
    hybrid: Path,
    enc_score: Path,
    cond_ch: np.ndarray,
    exclude_donor_fold: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if n_synth <= 0:
        return np.zeros((0, 1), dtype=np.float32), np.zeros(0, dtype=np.int64)

    def _disjoint(x: np.ndarray, ch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Galleries are stored in S3 donor order (row i is an edit of donor i).
        if exclude_donor_fold is None or len(x) != len(cond_ch):
            return x, ch
        keep = np.arange(len(x)) % 3 != int(exclude_donor_fold)
        return x[keep], np.asarray(ch)[keep]

    key = {
        "synth_only": "genfsdiff_c1",
        "c1_plus_real": "genfsdiff_c1",
        "c3_plus_real": "genfsdiff_c3",
    }.get(arm, arm)
    if key in galleries:
        x, ch = galleries[key]
        return _subsample(*_disjoint(x, ch), n_synth, rng)
    if key == "genfsdiff_c1":
        path = hybrid / f"{C1_GALLERY}_fold{fold_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        blob = np.load(path)
        ch = np.asarray(blob["channel_idx"]) if "channel_idx" in blob.files else cond_ch
        return _subsample(*_disjoint(np.asarray(blob["x"]), ch), n_synth, rng)
    if key == "genfsdiff_c3":
        path = enc_score / f"{C3_GALLERY}_fold{fold_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        blob = np.load(path)
        ch = np.asarray(blob["channel_idx"]) if "channel_idx" in blob.files else cond_ch
        return _subsample(*_disjoint(np.asarray(blob["x"]), ch), n_synth, rng)
    raise FileNotFoundError(f"unknown augdetect arm {arm}")


def _subsample(
    x: np.ndarray,
    ch: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    ch = np.asarray(ch)
    if len(ch) != len(x):
        ch = ch[: len(x)] if len(ch) >= len(x) else np.resize(ch, len(x))
    if n >= len(x):
        return x, ch
    idx = rng.choice(len(x), size=int(n), replace=False)
    return x[idx], ch[idx]


def _assert_no_nominal_leak(rows: list[dict[str, Any]]) -> None:
    """Train/test nominal masks are complementary (index % 3)."""
    for row in rows:
        if not row.get("ok"):
            continue
        tr, te = set(row.get("train_events") or []), set(row.get("test_events") or [])
        if tr & te:
            raise RuntimeError(f"event leak in {row['arm']} fold {row['fold']}")


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    arms: tuple[str, ...],
    seeds: list[int],
    folds: list[int],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ok = [r for r in rows if r.get("ok")]
    for arm in arms:
        pool = [r for r in ok if r["arm"] == arm]
        if not pool:
            continue
        by_seed: dict[int, dict[str, bool]] = {}
        by_seed_noleak: dict[int, dict[str, bool]] = {}
        for r in pool:
            hits = r.get("event_hits") or {}
            by_seed.setdefault(int(r["seed"]), {}).update(hits)
            if not r.get("leaked"):
                by_seed_noleak.setdefault(int(r["seed"]), {}).update(hits)
        seed_rec = [event_recall_safe(h) for h in by_seed.values()]
        seed_rec_nl = [event_recall_safe(h) for h in by_seed_noleak.values()]
        kinds = sorted({k for r in pool for k in (r.get("per_kind_recall") or {})})
        per_kind = {
            k: float(np.nanmean([r["per_kind_recall"].get(k, float("nan")) for r in pool]))
            for k in kinds
        }
        out[arm] = {
            "event_recall_median": float(np.nanmedian(seed_rec)) if seed_rec else float("nan"),
            "event_recall_mean": float(np.nanmean(seed_rec)) if seed_rec else float("nan"),
            "event_recall_min": float(np.nanmin(seed_rec)) if seed_rec else float("nan"),
            "event_recall_max": float(np.nanmax(seed_rec)) if seed_rec else float("nan"),
            "event_recall_noleak_median": (
                float(np.nanmedian(seed_rec_nl)) if seed_rec_nl else float("nan")
            ),
            "window_recall_mean": float(np.nanmean([r["window_recall"] for r in pool])),
            "ap_mean": float(np.nanmean([r["ap"] for r in pool])),
            "auroc_mean": float(np.nanmean([r["auroc"] for r in pool])),
            "rare_far_mean": float(np.nanmean([r["rare_far"] for r in pool])),
            "nominal_far_mean": float(np.nanmean([r["nominal_far"] for r in pool])),
            "per_kind_recall_mean": per_kind,
            "n_runs": len(pool),
            "leaked_runs": int(sum(1 for r in pool if r.get("leaked"))),
            "seed_event_recall": seed_rec,
            "event_recall_flipped_mean": float(
                np.nanmean([r.get("event_recall_flipped", float("nan")) for r in pool])
            ),
        }
        if any("event_hits_fused" in r for r in pool):
            fused_by_seed: dict[int, dict[str, bool]] = {}
            cnn_fresh_by_seed: dict[int, dict[str, bool]] = {}
            for r in pool:
                fused_by_seed.setdefault(int(r["seed"]), {}).update(r.get("event_hits_fused") or {})
                cnn_fresh_by_seed.setdefault(int(r["seed"]), {}).update(r.get("event_hits_cnn_freshneg") or {})
            out[arm]["event_recall_fused_median"] = float(
                np.nanmedian([event_recall_safe(h) for h in fused_by_seed.values()])
            )
            out[arm]["ap_fused_mean"] = float(np.nanmean([r.get("ap_fused", float("nan")) for r in pool]))
            out[arm]["fused_vs_cnn"] = paired_event_bootstrap(fused_by_seed, cnn_fresh_by_seed)
        out[arm]["_hits_by_seed"] = by_seed
    ref = out.get("real_only", {}).get("_hits_by_seed")
    for arm in list(out):
        hits = out[arm].pop("_hits_by_seed")
        if ref is not None and arm != "real_only":
            out[arm]["vs_real_only"] = paired_event_bootstrap(hits, ref)
    return out


def event_recall_safe(hits: dict[str, bool]) -> float:
    if not hits:
        return float("nan")
    return float(np.mean(list(hits.values())))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = []
    for r in rows:
        d = {
            k: v
            for k, v in r.items()
            if k
            not in {
                "event_hits",
                "event_hits_fused",
                "event_hits_cnn_freshneg",
                "train_events",
                "test_events",
                "per_kind_recall",
            }
        }
        for k, v in (r.get("per_kind_recall") or {}).items():
            d[f"kind::{k}"] = v
        flat.append(d)
    if not flat:
        return
    keys = sorted({k for row in flat for k in row})
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(flat)


def _plot_summary(out: Path, summary: dict[str, Any]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    if not summary:
        return []
    arms = list(summary)
    ev = [summary[a]["event_recall_median"] for a in arms]
    lo = [summary[a]["event_recall_min"] for a in arms]
    hi = [summary[a]["event_recall_max"] for a in arms]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.7 * len(arms)), 4.2))
    xs = np.arange(len(arms))
    ax.bar(xs, ev, color="#2166ac", alpha=0.85)
    ax.errorbar(
        xs,
        ev,
        yerr=[np.clip(np.asarray(ev) - np.asarray(lo), 0, None), np.clip(np.asarray(hi) - np.asarray(ev), 0, None)],
        fmt="none",
        ecolor="#4d4d4d",
        capsize=3,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(arms, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("event recall @ 1% FAR")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Augmentation detector (median over seeds; whiskers min–max)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "event_recall.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(7.0, 0.7 * len(arms)), 4.2))
    rare = [summary[a]["rare_far_mean"] for a in arms]
    ax.bar(xs, rare, color="#b2182b", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(arms, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Rare-Event FAR at the same threshold")
    ax.set_title("Rare-Event false alarms (mean over runs)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "rare_far.png", dpi=140)
    plt.close(fig)

    kinds = sorted({k for a in arms for k in summary[a].get("per_kind_recall_mean") or {}})
    if kinds:
        mat = np.array(
            [[summary[a]["per_kind_recall_mean"].get(k, float("nan")) for k in kinds] for a in arms]
        )
        fig, ax = plt.subplots(figsize=(1.6 * len(kinds) + 2.5, 0.45 * len(arms) + 2.0))
        im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="YlGnBu")
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels([k.replace("real ESA ", "").replace("real ", "") for k in kinds], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels(arms, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.03, label="window recall @ 1% FAR")
        ax.set_title("Per-kind window recall (mean over runs)")
        fig.tight_layout()
        fig.savefig(out / "per_kind_recall.png", dpi=140)
        plt.close(fig)
        return ["event_recall.png", "rare_far.png", "per_kind_recall.png"]
    return ["event_recall.png", "rare_far.png"]


def _index_text(summary: dict[str, Any], written: list[str]) -> str:
    lines = [
        "augdetect: event recall @ 1% nominal FAR. Dev only, sealed test closed.",
        "C1 fold 0 is the level-shift proto leak; noleak median drops that fold.",
        "",
    ]
    for arm, m in summary.items():
        lines.append(
            f"  {arm}: event_rec={m.get('event_recall_median'):.3f} "
            f"noleak={m.get('event_recall_noleak_median'):.3f} "
            f"AP={m.get('ap_mean'):.3f} rareFAR={m.get('rare_far_mean'):.3f}"
        )
    lines.append("")
    lines.extend(f"  {name}" for name in written)
    lines.append("")
    return "\n".join(lines)
