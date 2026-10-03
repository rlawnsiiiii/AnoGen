"""Label-free detection with the nominal diffusion model itself (dev only).

The S1 score was trained on everyday nominals + Rare Events and never saw an
Anomaly. Its denoising error is therefore an anomaly score that needs no
labels (DDPM-based TSAD in the spirit of ImDiffusion / DiffusionAD):

    s(x) = mean over t ∈ T_eval and K noise draws of
           mean_i ( ε_θ(√ᾱ_t x + √(1−ᾱ_t) ε, t, c)[i] − ε[i] )²

computed in the scaled units the score was trained in, then standardized per
channel against *training-fold* donors (index % 3 ≠ k), so channels with
different noise floors share one threshold. With the frozen causal backbone,
the first ``skip_start`` bins are excluded (they are the worst-estimated, see
docs/POSITION_BIAS.md), or the score is computed through ``FlipEnsemble``.

Evaluated exactly like augdetect: event recall at 1 % nominal FAR on the
fold's test donors (index % 3 == k), AP / AUROC, Rare-Event FAR, per kind.
Also writes per-window scores aligned with S0 / S3 arrays so augdetect can
rank-fuse them with the supervised CNN (``shell.augdetect.fuse_diffscore``).

Isolated ``results/shell_diffdetect/``. Sealed test stays closed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anogen.config import REPO_ROOT


def run_diffdetect(cfg: dict[str, Any]) -> dict[str, Any]:
    from anogen.phases.encscore import _abs, _jsonable

    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = _abs(cfg.get("s0_dir", root / "results/shell_s0"), root)
    s1 = _abs(cfg.get("s1_dir", root / "results/shell_s1"), root)
    s3 = _abs(cfg.get("s3_dir", root / "results/shell_s3"), root)
    dcfg = dict((cfg.get("shell") or {}).get("diffdetect") or {})
    out = _abs(dcfg.get("out", root / "results/shell_diffdetect"), root)
    if out.resolve() in {s0.resolve(), s1.resolve(), s3.resolve()}:
        raise RuntimeError("diffdetect must not write into a locked result directory")
    out.mkdir(parents=True, exist_ok=True)

    from anogen.shell.diffusion import torch_available

    if not torch_available():
        report = {"ok": False, "skipped": True, "reason": "torch is not installed"}
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    for label, path in {
        "S0": s0 / "labeled_arrays.npz",
        "S1": s1 / "denoiser.pt",
        "S3": s3 / "galleries.npz",
    }.items():
        if not path.is_file():
            report = {"ok": False, "skipped": True, "reason": f"{label} missing: {path}"}
            (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            return report

    import torch

    from anogen.shell.detector import eval_scores, paired_event_bootstrap
    from anogen.shell.diffusion import DiffusionSchedule, denoiser_from_ckpt, wrap_denoiser
    from anogen.shell.events import types_from_cfg
    from anogen.shell.morphology import kinds_for_windows
    from anogen.shell.scaler import resolve_scaler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lab = np.load(s0 / "labeled_arrays.npz")
    meta = pd.read_csv(s0 / "labeled_windows.csv")
    anom = meta[meta["kind"] == "anomaly"].reset_index(drop=True)
    x_a, x_r = np.asarray(lab["anomaly"]), np.asarray(lab["rare"])
    fold_a, fold_r = np.asarray(lab["anomaly_fold"]), np.asarray(lab["rare_fold"])
    ch_a, ch_r = np.asarray(lab["anomaly_channel"]), np.asarray(lab["rare_channel"])
    event_a = anom["event_id"].astype(str).to_numpy()
    kind_a = kinds_for_windows(x_a, anom, types_from_cfg(cfg))
    gal = np.load(s3 / "galleries.npz")
    x_d, ch_d = np.asarray(gal["cond"]), np.asarray(gal["channel_idx"], dtype=np.int64)

    variants = dict(
        dcfg.get("variants")
        or {
            "causal_skip32": {"skip_start": 32},
            "causal_full": {"skip_start": 0},
            "flip_ramp": {"flip": "ramp", "skip_start": 0},
        }
    )
    t_eval = [int(t) for t in (dcfg.get("t_eval") or [10, 20, 40])]
    n_draws = int(dcfg.get("n_draws", 4))
    bsz = int(dcfg.get("bsz", 256))
    far = float(dcfg.get("target_far", 0.01))
    seed = int(cfg.get("seed", 0))
    folds = sorted({int(f) for f in fold_a if int(f) >= 0})

    results: dict[str, Any] = {}
    for vname, spec in variants.items():
        den_path = _abs(spec["denoiser"], root) if spec.get("denoiser") else s1 / "denoiser.pt"
        den = torch.load(den_path, map_location=device, weights_only=False)
        model = wrap_denoiser(denoiser_from_ckpt(den, device=torch.device(device)), spec.get("flip"))
        scaler = resolve_scaler(den, s0)
        sched = DiffusionSchedule.linear(int(den["n_times"])).to(torch.device(device))
        kw = dict(t_eval=t_eval, n_draws=n_draws, bsz=bsz, skip=int(spec.get("skip_start", 0)), seed=seed)
        cache = out / f"{vname}_scores.npz"
        if cache.is_file() and not bool(dcfg.get("force", False)):
            blob = np.load(cache)
            sa, sr, sd = blob["anomaly"], blob["rare"], blob["donor"]
        else:
            print(f"diffdetect {vname}", flush=True)
            sa = denoising_scores(model, sched, scaler, x_a, ch_a, device=device, **kw)
            sr = denoising_scores(model, sched, scaler, x_r, ch_r, device=device, **kw)
            sd = denoising_scores(model, sched, scaler, x_d, ch_d, device=device, **kw)
            np.savez_compressed(cache, anomaly=sa, rare=sr, donor=sd, t_eval=np.asarray(t_eval))
        rows = []
        for k in folds:
            train_d = np.arange(len(x_d)) % 3 != k
            test_d = ~train_d
            mu, sdv = _channel_stats(sd[train_d], ch_d[train_d])
            z = lambda s, c: (s - mu[c]) / sdv[c]  # noqa: E731
            ta, tr = fold_a == k, fold_r == k
            ev = eval_scores(
                z(sa[ta], ch_a[ta]),
                z(sd[test_d], ch_d[test_d]),
                z(sr[tr], ch_r[tr]),
                event_id=event_a[ta],
                kind=kind_a[ta],
                far=far,
            )
            ev["fold"] = k
            rows.append(ev)
        hits = {0: {e: h for r in rows for e, h in r["event_hits"].items()}}
        results[vname] = {
            "spec": spec,
            "event_recall": float(np.mean(list(hits[0].values()))),
            "ap_mean": float(np.mean([r["ap"] for r in rows])),
            "auroc_mean": float(np.mean([r["auroc"] for r in rows])),
            "rare_far_mean": float(np.mean([r["rare_far"] for r in rows])),
            "window_recall_mean": float(np.mean([r["window_recall"] for r in rows])),
            "folds": rows,
            "_hits": hits,
        }
    base = next(iter(results), None)
    for vname, res in results.items():
        if vname != base:
            res["vs_" + str(base)] = paired_event_bootstrap(res["_hits"], results[base]["_hits"])
    for res in results.values():
        res.pop("_hits")
    report = {
        "ok": True,
        "skipped": False,
        "primary": "event_recall@1%FAR (label-free)",
        "t_eval": t_eval,
        "n_draws": n_draws,
        "results": results,
        "note": "Scores standardized per channel on training-fold donors; test donors index % 3 == k.",
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return {"ok": True, "skipped": False, "dir": str(out), "variants": list(results)}


def _channel_stats(scores: np.ndarray, ch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = int(max(ch.max(), 0)) + 1 if len(ch) else 1
    mu = np.zeros(max(n, 64))
    sd = np.ones(max(n, 64))
    for c in range(n):
        m = ch == c
        if m.sum() >= 2:
            mu[c] = float(np.mean(scores[m]))
            sd[c] = float(max(np.std(scores[m]), 1e-12))
    return mu, sd


def denoising_scores(
    model: Any,
    schedule: Any,
    scaler: Any,
    x: np.ndarray,
    ch: np.ndarray,
    *,
    t_eval: list[int],
    n_draws: int,
    bsz: int,
    skip: int,
    seed: int,
    device: str,
) -> np.ndarray:
    """Mean ε-prediction error per window (higher = less nominal)."""
    import torch

    from anogen.shell.diffusion import q_sample

    if len(x) == 0:
        return np.zeros(0, dtype=np.float64)
    xs = scaler.transform(x, ch) if scaler is not None else np.asarray(x, dtype=np.float32)
    out = np.zeros(len(xs), dtype=np.float64)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    model.eval()
    with torch.no_grad():
        for i in range(0, len(xs), int(bsz)):
            xb = torch.from_numpy(np.asarray(xs[i : i + bsz], dtype=np.float32)).unsqueeze(1).to(device)
            cb = torch.from_numpy(np.asarray(ch[i : i + bsz], dtype=np.int64)).to(device)
            acc = torch.zeros(xb.size(0), device=device)
            for t in t_eval:
                tb = torch.full((xb.size(0),), int(t), dtype=torch.long, device=device)
                for _ in range(int(n_draws)):
                    noise = torch.randn(xb.shape, generator=gen).to(device)
                    xt, _ = q_sample(xb, tb, schedule, noise=noise)
                    err = (model(xt, tb, cb) - noise) ** 2
                    acc += err[..., int(skip) :].mean(dim=(1, 2))
            out[i : i + len(xb)] = (acc / (len(t_eval) * int(n_draws))).cpu().numpy()
    return out
