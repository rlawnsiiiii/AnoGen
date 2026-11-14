#!/usr/bin/env python3
"""Detector runs for the paper: full budget, budget sweep, kind-stratified sweep.

Same protocol as ``anogen augdetect`` (``_run_one``), with every row's event
hits, real training rows, generated donors and test-fold scores kept, so the
appendix can show which events each detector finds at which budget. Sealed
test stays closed; writes results/paper_detect only.

  run:        paper_detector.py --shard i --nshards n     (one process per shard)
  aggregate:  paper_detector.py --aggregate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results/paper_detect"
FIXSWEEP = ROOT / "results/shell_fixsweep_noise"
GFS = "flip_x0_contrast_head_sharp_nomask"
ARMS = (
    "real_only",
    f"fixsweep:{GFS}+real",
    f"fixsweep:{GFS}_noise+real",
    f"fixsweep:{GFS}_mnoise+real",
    "fixsweep:nosteer_flip_x0_recur2+real",
    "fixsweep:nosteer_flip_x0_recur2_noise+real",
    "genias+real",
    "geniasfair:genias_fair+real",
    "cutaddpaste+real",
    "posthoc+real",
)
KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
SEEDS = (0, 1, 2, 3, 4)
FOLDS = (0, 1, 2)
N_SYNTH_FULL = 256


def jobs() -> list[tuple[str, int, str, int, int]]:
    out = [("full", -1, a, f, s) for a in ARMS for f in FOLDS for s in SEEDS]
    for mode in ("uniform", "strat"):
        out += [(mode, k, a, f, s) for k in KS for a in ARMS for f in FOLDS for s in SEEDS]
    return out


def load() -> dict:
    from anogen.config import load_config
    from anogen.phases.augdetect import _arm_leaks_fold0  # noqa: F401
    from anogen.shell.events import types_from_cfg
    from anogen.shell.morphology import kinds_for_windows
    from anogen.shell.scaler import load_minmax

    cfg = load_config(ROOT / "configs/shell_mission1.yaml")
    acfg = cfg["shell"]["augdetect"]
    s0, s3, s4 = ROOT / "results/shell_s0", ROOT / "results/shell_s3", ROOT / "results/shell_s4"
    lab, gal, s4g = np.load(s0 / "labeled_arrays.npz"), np.load(s3 / "galleries.npz"), np.load(s4 / "shell_gallery.npz")
    meta = pd.read_csv(s0 / "labeled_windows.csv")
    anom = meta[meta["kind"] == "anomaly"].reset_index(drop=True)
    x_a = np.asarray(lab["anomaly"])
    kind_a = kinds_for_windows(x_a, anom, types_from_cfg(cfg))
    cond_ch = np.asarray(gal["channel_idx"])
    galleries = {
        "posthoc": (np.asarray(gal["posthoc"]), cond_ch),
        "genias": (np.asarray(gal["genias"]), cond_ch),
        "unguided": (np.asarray(s4g["unguided"]), cond_ch),
        "cutaddpaste": (np.asarray(np.load(ROOT / "results/shell_genbase/cutaddpaste.npz")["x"]), cond_ch),
    }
    one_kw = dict(
        x_a=x_a, x_r=np.asarray(lab["rare"]), fold_a=np.asarray(lab["anomaly_fold"]),
        fold_r=np.asarray(lab["rare_fold"]), ch_a=np.asarray(lab["anomaly_channel"]),
        ch_r=np.asarray(lab["rare_channel"]), event_a=anom["event_id"].astype(str).to_numpy(),
        kind_a=kind_a, x_cond=np.asarray(gal["cond"]), cond_ch=cond_ch, galleries=galleries,
        hybrid=ROOT / "results/shell_kindmix_score_hybrid", enc_score=ROOT / "results/shell_enc_score",
        scaler=load_minmax(s0 / "minmax_scaler.npz"), n_nom_test=int(acfg.get("n_nominal_test", 512)),
        steps=int(acfg.get("steps", 400)), hidden=int(acfg.get("hidden", 32)),
        batch_size=int(acfg.get("batch_size", 32)), lr=float(acfg.get("lr", 1e-3)),
        far=float(acfg.get("target_far", 0.01)), n_boot=int(acfg.get("n_boot", 1000)), device=None,
        donor_disjoint=bool(acfg.get("donor_disjoint", True)), shift_aug=int(acfg.get("shift_aug", 0)),
        flip_test=False, diff_scores=None, fixsweep_dir=FIXSWEEP,
        geniasfair_dir=ROOT / "results/shell_genias_fair", real_frac=None, synth_filter=None,
        novelty_cache={},
    )
    return one_kw


def run_shard(shard: int, nshards: int) -> None:
    from anogen.phases.augdetect import _arm_leaks_fold0, _run_one

    one_kw = load()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"rows_shard{shard}.jsonl"
    done = set()
    if path.is_file():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["mode"], r["k"], r["arm"], r["fold"], r["seed"]))
    todo = [j for i, j in enumerate(jobs()) if i % nshards == shard and j not in done]
    # the matched-marginal galleries are generated last; their rows wait for them
    todo.sort(key=lambda j: "mnoise" in j[2])
    with path.open("a") as fh:
        for mode, k, arm, fold, seed in todo:
            kw = dict(one_kw)
            kw["n_synth"] = N_SYNTH_FULL if mode == "full" else k
            row = _run_one(
                arm=arm, fold_id=fold, seed=seed, leaked=_arm_leaks_fold0(arm) and fold == 0,
                real_cap=None if mode == "full" else k,
                real_strata=one_kw["kind_a"] if mode == "strat" else None,
                keep_detail=True, **kw,
            )
            row.update(mode=mode, k=int(k))
            fh.write(json.dumps(_plain(row)) + "\n")
            fh.flush()
            print(f"{mode} k={k} {arm} fold={fold} seed={seed} rec={row.get('event_recall')}", flush=True)


def _plain(obj):
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def aggregate() -> None:
    from anogen.shell.detector import paired_event_bootstrap

    keyed = {}
    for p in sorted(OUT.glob("rows_shard*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            keyed[(r["mode"], r["k"], r["arm"], r["fold"], r["seed"])] = r
    rows = list(keyed.values())
    one_kw = load()
    ev_kind: dict[str, list[str]] = {}
    for e, k in zip(one_kw["event_a"], one_kw["kind_a"]):
        ev_kind.setdefault(str(e), [])
        if str(k) not in ev_kind[str(e)]:
            ev_kind[str(e)].append(str(k))

    def hits(mode: str, k: int, arm: str) -> dict[int, dict[str, bool]]:
        by: dict[int, dict[str, bool]] = {}
        for r in rows:
            if r["mode"] == mode and r["k"] == k and r["arm"] == arm and r.get("ok"):
                by.setdefault(int(r["seed"]), {}).update(r["event_hits"])
        return by

    def summary(mode: str, k: int, arm: str) -> dict:
        h = hits(mode, k, arm)
        pool = [r for r in rows if r["mode"] == mode and r["k"] == k and r["arm"] == arm and r.get("ok")]
        rec = [float(np.mean(list(v.values()))) for v in h.values()]
        out = {
            "n_runs": len(pool),
            "event_recall_median": float(np.median(rec)),
            "event_recall_mean": float(np.mean(rec)),
            "event_recall_min": float(np.min(rec)),
            "event_recall_max": float(np.max(rec)),
            "seed_event_recall": rec,
            "rare_far_mean": float(np.nanmean([r["rare_far"] for r in pool])),
            "ap_mean": float(np.nanmean([r["ap"] for r in pool])),
            "auroc_mean": float(np.nanmean([r["auroc"] for r in pool])),
            "window_recall_mean": float(np.nanmean([r["window_recall"] for r in pool])),
            "event_hit_rate": {e: float(np.mean([v[e] for v in h.values() if e in v])) for e in sorted(ev_kind)},
        }
        kinds = sorted({k2 for ks in ev_kind.values() for k2 in ks})
        out["kind_event_recall"] = {
            kd: float(np.mean([out["event_hit_rate"][e] for e, ks in ev_kind.items() if kd in ks])) for kd in kinds
        }
        if arm != "real_only":
            out["vs_real_only"] = paired_event_bootstrap(h, hits(mode, k, "real_only"), n_boot=2000, seed=0)
        return out

    report: dict = {"arms": list(ARMS), "k": list(KS), "seeds": list(SEEDS), "event_kinds": ev_kind}
    report["full"] = {a: summary("full", -1, a) for a in ARMS}
    for mode in ("uniform", "strat"):
        report[mode] = {a: {str(k): summary(mode, k, a) for k in KS} for a in ARMS}
    gfs, gfn = f"fixsweep:{GFS}+real", f"fixsweep:{GFS}_noise+real"
    pairs = [(gfs, b) for b in ARMS if b not in ("real_only", gfs)] + [
        (gfn, "fixsweep:nosteer_flip_x0_recur2_noise+real"), (gfn, "posthoc+real"), (gfn, "genias+real")
    ]
    report["pairs"] = {
        f"{a} - {b}": {
            "full": paired_event_bootstrap(hits("full", -1, a), hits("full", -1, b), n_boot=2000, seed=0),
            **{mode: {str(k): paired_event_bootstrap(hits(mode, k, a), hits(mode, k, b), n_boot=2000, seed=0)
                      for k in KS} for mode in ("uniform", "strat")},
        }
        for a, b in pairs
    }
    report["n_rows"] = len(rows)
    (OUT / "summary.json").write_text(json.dumps(_plain(report), indent=1) + "\n")
    print(f"wrote {OUT / 'summary.json'} from {len(rows)} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
    else:
        run_shard(args.shard, args.nshards)


if __name__ == "__main__":
    main()
