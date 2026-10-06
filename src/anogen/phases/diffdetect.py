"""Label-free detection with the nominal diffusion model itself (dev only).

The S1 score was trained on everyday nominals + Rare Events and never saw an
Anomaly. Its denoising error is therefore an anomaly score that needs no
labels (DDPM-based TSAD in the spirit of ImDiffusion / DiffusionAD):

    s(x) = mean over t ∈ T_eval and K noise draws of
           mean_i ( ε_θ(√ᾱ_t x + √(1−ᾱ_t) ε, t, c)[i] − ε[i] )²

computed in the scaled units the score was trained in. Variants can pool the
per-bin error over its largest bins (``pool: top16``) and fuse noise levels by
the max of per-channel z-scores (``fuse: zmax``), the multiscale idea of
Mahmood et al. (ICLR 2021); testbed E15/E15b.

Held-out negatives. S1 trained on every window of ``train_index.csv``: all
S3 donors and all labelled Rare Events. Scoring those would be in-sample
and would deflate the false-alarm threshold. Nominal negatives are therefore
*fresh* windows drawn from the panel: same channels, before the sealed cut,
no overlap with any anomaly or rare span (+ guard), and no overlap with any
training window of the same channel. Fresh windows are split by index % 3:
the other two thirds standardize the score per channel (so channels with
different noise floors share one threshold), the matching third is the
fold's test negatives. Every labelled Rare Event was in S1's training set,
so ``rare_far`` is reported but flagged as in-sample. With the frozen causal backbone,
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
    n_fresh = int(dcfg.get("n_fresh_per_channel", 512))
    fresh_path = out / "fresh_nominal.npz"
    if fresh_path.is_file() and not bool(dcfg.get("force", False)):
        fb = np.load(fresh_path)
        x_d, ch_d = np.asarray(fb["x"]), np.asarray(fb["channel_idx"], dtype=np.int64)
    else:
        x_d, ch_d, fresh_info = heldout_nominal_windows(cfg, s0, n_fresh, seed=int(cfg.get("seed", 0)))
        np.savez_compressed(fresh_path, x=x_d, channel_idx=ch_d, **{k: np.asarray(v) for k, v in fresh_info.items()})

    variants = dict(
        dcfg.get("variants")
        or {
            "causal_skip32": {"skip_start": 32},
            "causal_full": {"skip_start": 0},
            "flip_ramp": {"flip": "ramp", "skip_start": 0},
            # Localized, multiscale score (testbed E15/E15b): the mean over 512
            # bins dilutes a 3-bin spike; small t sees spikes, large t sees
            # drifts. Top-16 bins per t, z-scored per channel on held-out
            # nominals, max over t.
            "flip_ramp_top16_multiscale": {
                "flip": "ramp",
                "skip_start": 0,
                "pool": "top16",
                "fuse": "zmax",
                "t_eval": [10, 20, 40, 60, 100, 150],
            },
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
        sched = DiffusionSchedule.from_ckpt(den, allow_nonlinear=True).to(torch.device(device))
        v_t_eval = [int(t) for t in (spec.get("t_eval") or t_eval)]
        pool = str(spec.get("pool", "mean"))
        fuse = str(spec.get("fuse", "mean"))
        from anogen.shell.scaler import load_minmax, unit_scale

        mm_path = s0 / "minmax_scaler.npz"
        u_scale = unit_scale(scaler, load_minmax(mm_path) if mm_path.is_file() else None, ch_d)
        kw = dict(t_eval=_map_t_eval(sched, v_t_eval, u_scale), n_draws=n_draws, bsz=bsz, skip=int(spec.get("skip_start", 0)), seed=seed)
        if pool != "mean":
            kw["pool"] = pool
        cache = out / f"{vname}_scores.npz"
        key = json.dumps({**kw, "denoiser": str(den_path), "flip": spec.get("flip"), "n_fresh": int(len(x_d))}, sort_keys=True)
        blob = np.load(cache) if cache.is_file() else None
        if (
            blob is not None
            and "key" in blob.files
            and "fresh_t" in blob.files
            and str(blob["key"]) == key
            and not bool(dcfg.get("force", False))
        ):
            sa_t, sr_t, sd_t = blob["anomaly_t"], blob["rare_t"], blob["fresh_t"]
        else:
            print(f"diffdetect {vname}", flush=True)
            sa_t = denoising_scores(model, sched, scaler, x_a, ch_a, device=device, **kw)
            sr_t = denoising_scores(model, sched, scaler, x_r, ch_r, device=device, **kw)
            sd_t = denoising_scores(model, sched, scaler, x_d, ch_d, device=device, **kw)
        # 1-D scores aligned with S0 arrays for augdetect's rank fusion. For
        # fuse=zmax these use per-channel stats of *all* fresh windows (score
        # distribution only, no labels); the fold evaluation below re-fits them
        # on the fold's training negatives.
        sa, sr, sd = (fuse_scores(a, c, sd_t, ch_d, fuse) for a, c in ((sa_t, ch_a), (sr_t, ch_r), (sd_t, ch_d)))
        np.savez_compressed(
            cache,
            anomaly=sa,
            rare=sr,
            fresh=sd,
            anomaly_t=sa_t,
            rare_t=sr_t,
            fresh_t=sd_t,
            fresh_channel=ch_d,
            fuse=np.asarray(fuse),
            key=np.asarray(key),
        )
        rows = []
        for k in folds:
            train_d = np.arange(len(x_d)) % 3 != k
            test_d = ~train_d
            fa = fuse_scores(sa_t, ch_a, sd_t[train_d], ch_d[train_d], fuse)
            fr = fuse_scores(sr_t, ch_r, sd_t[train_d], ch_d[train_d], fuse)
            fd = fuse_scores(sd_t, ch_d, sd_t[train_d], ch_d[train_d], fuse)
            mu, sdv = _channel_stats(fd[train_d], ch_d[train_d])
            z = lambda s, c: (s - mu[c]) / sdv[c]  # noqa: E731
            ta, tr = fold_a == k, fold_r == k
            ev = eval_scores(
                z(fa[ta], ch_a[ta]),
                z(fd[test_d], ch_d[test_d]),
                z(fr[tr], ch_r[tr]),
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
            "rare_far_mean_in_sample": float(np.mean([r["rare_far"] for r in rows])),
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
        "note": (
            "Negatives are fresh nominal windows held out from S1 (no overlap with any training "
            "window of the channel or any labelled span). Per-channel standardization on fresh "
            "index % 3 != k; test negatives fresh index % 3 == k. Rare FAR is in-sample (S1 trained "
            "on every labelled rare)."
        ),
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    return {"ok": True, "skipped": False, "dir": str(out), "variants": list(results)}


def heldout_nominal_windows(
    cfg: dict[str, Any],
    s0: Path,
    n_per_channel: int,
    *,
    seed: int = 0,
    max_tries_factor: int = 200,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Nominal windows S1 never trained on (see module docstring)."""
    from anogen.config import data_root, panel_path
    from anogen.shell.data import load_metadata, load_panel
    from anogen.shell.events import assign_splits, build_event_table
    from anogen.shell.windows import load_train_index, occupancy_mask, window_finite

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
    meta = load_metadata(data_root(cfg))
    events = assign_splits(
        build_event_table(meta["labels"], meta["anomaly_types"]),
        path_train_end=splits.get("path_train_end", "2006-10-01"),
        official_train_end=splits.get("official_train_end", "2007-01-01"),
    )
    occ = occupancy_mask(panel, events, guard_bins=int(proto["guard_bins"]), occupy_rares=True)
    train = load_train_index(s0)
    rng = np.random.default_rng(int(seed) + 7919)
    xs, chs = [], []
    found = {}
    for c in range(panel.k):
        used = np.zeros(panel.T, dtype=bool)
        for st in train.loc[train["channel_idx"] == c, "start"].to_numpy(dtype=np.int64):
            used[st : st + width] = True
        blocked = occ | used
        taken = np.zeros(panel.T, dtype=bool)
        got = 0
        for _ in range(int(n_per_channel) * int(max_tries_factor)):
            if got >= int(n_per_channel):
                break
            st = int(rng.integers(0, panel.T - width + 1))
            if blocked[st : st + width].any() or taken[st : st + width].any():
                continue
            if not window_finite(panel.Y, panel.counts, st, c, width):
                continue
            taken[st : st + width] = True
            xs.append(panel.Y[st : st + width, c].astype(np.float32))
            chs.append(c)
            got += 1
        found[panel.channels[c]] = got
    if not xs:
        raise RuntimeError("no held-out nominal windows found; lower n_fresh_per_channel")
    return np.stack(xs), np.asarray(chs, dtype=np.int64), {"found_per_channel": json.dumps(found)}


def _map_t_eval(schedule: Any, t_eval: list[int], unit_scale: float = 1.0) -> list[int]:
    """t_eval is configured as linear-schedule indices in the frozen units; keep
    their (physical) noise levels. ``unit_scale``: see ``scaler.unit_scale``."""
    from anogen.shell.diffusion import DiffusionSchedule

    n = int(schedule.betas.numel())
    lin = DiffusionSchedule.linear(n).alpha_bar.numpy()
    ab = schedule.alpha_bar.detach().cpu().numpy()
    if np.allclose(ab, lin) and abs(float(unit_scale) - 1.0) < 1e-9:
        return [int(t) for t in t_eval]
    out: list[int] = []
    s_hi = np.sqrt((1.0 - ab.min()) / ab.min())
    for t in t_eval:
        sigma = np.sqrt((1.0 - lin[int(t)]) / lin[int(t)]) * float(unit_scale)
        if sigma > s_hi:
            import warnings

            warnings.warn(f"t_eval {t}: σ={sigma:.3g} exceeds the schedule's σ_max {s_hi:.3g}; clamped", stacklevel=2)
        m = int(np.argmin(np.abs(ab - 1.0 / (1.0 + sigma**2))))
        if m not in out:  # clamped levels collapse onto one index: keep it once
            out.append(m)
    return out


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


def pool_bins_np(err: np.ndarray, pool: str = "mean", k: int = 16) -> np.ndarray:
    """Numpy twin of the per-window pooling in ``denoising_scores`` (rows = windows).

    mean   mean over bins (the original score; dilutes short anomalies ~W/len)
    topK   mean of the K largest per-bin errors (``top16``)
    maxK   max over bins of the K-bin moving average (``max16``)
    """
    e = np.asarray(err, dtype=np.float64)
    if pool == "mean":
        return e.mean(axis=1)
    if pool.startswith("top"):
        kk = int(pool[3:] or k)
        return np.sort(e, axis=1)[:, -kk:].mean(axis=1)
    if pool.startswith("max"):
        kk = int(pool[3:] or k)
        c = np.cumsum(np.pad(e, ((0, 0), (1, 0))), axis=1)
        return ((c[:, kk:] - c[:, :-kk]) / kk).max(axis=1)
    raise ValueError(f"unknown pool {pool!r} (mean | topK | maxK)")


def fuse_scores(
    s_t: np.ndarray, ch: np.ndarray, ref_t: np.ndarray, ref_ch: np.ndarray, fuse: str = "mean"
) -> np.ndarray:
    """(N, T) per-noise-level scores -> (N,) window score.

    mean  mean over t of the raw scores (the original diffdetect score)
    zmax  z-score each t per channel on the reference (held-out nominal) rows,
          then the max over t: small t catches spikes, large t catches drifts
    """
    s = np.asarray(s_t, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, None]
    if fuse == "mean":
        return s.mean(axis=1)
    if fuse != "zmax":
        raise ValueError(f"unknown fuse {fuse!r} (mean | zmax)")
    ref = np.asarray(ref_t, dtype=np.float64).reshape(len(ref_ch), -1)
    z = np.zeros_like(s)
    for j in range(s.shape[1]):
        mu, sd = _channel_stats(ref[:, j], np.asarray(ref_ch))
        z[:, j] = (s[:, j] - mu[ch]) / sd[ch]
    return z.max(axis=1)


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
    pool: str = "mean",
) -> np.ndarray:
    """(N, len(t_eval)) ε-prediction error per window and noise level (higher = less nominal).

    The per-bin squared error is averaged over the K draws at each t, then
    pooled over bins (``pool_bins_np``: mean | topK | maxK).
    """
    import torch
    import torch.nn.functional as F

    from anogen.shell.diffusion import q_sample

    if len(x) == 0:
        return np.zeros((0, len(t_eval)), dtype=np.float64)
    xs = scaler.transform(x, ch) if scaler is not None else np.asarray(x, dtype=np.float32)
    out = np.zeros((len(xs), len(t_eval)), dtype=np.float64)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    model.eval()
    with torch.no_grad():
        for i in range(0, len(xs), int(bsz)):
            xb = torch.from_numpy(np.asarray(xs[i : i + bsz], dtype=np.float32)).unsqueeze(1).to(device)
            cb = torch.from_numpy(np.asarray(ch[i : i + bsz], dtype=np.int64)).to(device)
            for j, t in enumerate(t_eval):
                tb = torch.full((xb.size(0),), int(t), dtype=torch.long, device=device)
                acc = torch.zeros_like(xb[..., int(skip) :])
                for _ in range(int(n_draws)):
                    noise = torch.randn(xb.shape, generator=gen).to(device)
                    xt, _ = q_sample(xb, tb, schedule, noise=noise)
                    acc += ((model(xt, tb, cb) - noise) ** 2)[..., int(skip) :]
                err = (acc / int(n_draws)).squeeze(1)  # (B, L - skip)
                if pool == "mean":
                    pooled = err.mean(dim=-1)
                elif pool.startswith("top"):
                    pooled = err.topk(int(pool[3:] or 16), dim=-1).values.mean(dim=-1)
                elif pool.startswith("max"):
                    kk = int(pool[3:] or 16)
                    pooled = F.avg_pool1d(err.unsqueeze(1), kk, stride=1).squeeze(1).amax(dim=-1)
                else:
                    raise ValueError(f"unknown pool {pool!r}")
                out[i : i + len(xb), j] = pooled.cpu().numpy()
    return out
