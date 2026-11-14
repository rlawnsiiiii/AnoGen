#!/usr/bin/env python3
"""Record the chain of the fold-0 galleries, from a donor and from noise.

Regenerates fold 0 of GenFSDiff (donor start) and GenFSDiff from N(0, I) into
the scratch directory results/paper_trace with ``steer.TRACE`` on, and keeps the
edited clean prediction after every step for a few rows of each kind. Same
spec, seeds and allocation as results/shell_fixsweep_noise; scoring is skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import anogen.phases.fixsweep as fx  # noqa: E402
import anogen.shell.steer as steer  # noqa: E402
from anogen.config import load_config  # noqa: E402
from anogen.shell.scaler import load_minmax  # noqa: E402
from paper_from_noise import BASE, variants  # noqa: E402

OUT = ROOT / "results/paper_trace"
KEEP = 24  # rows kept per kind


class _Stop(Exception):
    pass


def _stop(*_a, **_k):
    raise _Stop


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = variants()
    sc = load_minmax(ROOT / "results/shell_s0/minmax_scaler.npz")
    for name in (BASE, f"{BASE}_noise"):
        cfg = load_config(ROOT / "configs/shell_mission1.yaml")
        cfg["shell"]["fixsweep"].update(
            out=str(OUT / "scratch"), variants={name: spec[name]}, only=None, folds=[0], force=True
        )
        steer.TRACE = []
        keep = fx._score_fold
        fx._score_fold = _stop
        try:
            fx.run_fixsweep(cfg)
        except _Stop:
            pass
        finally:
            fx._score_fold = keep
        calls, steer.TRACE = steer.TRACE, None
        blob = np.load(OUT / "scratch" / f"{name}_fold0.npz", allow_pickle=True)
        x = np.asarray(blob["x"], dtype=np.float64)
        ka = np.asarray(blob["kind_alloc"]).astype(str)
        xs = sc.transform(x, np.asarray(blob["channel_idx"]))
        ref = np.load(ROOT / "results/shell_fixsweep_noise" / f"{name}_fold0.npz", allow_pickle=True)
        rows, ts, traj = [], None, []
        for c in calls:
            for j, fin in enumerate(c["final"]):
                i = int(np.argmin(((xs - fin) ** 2).sum(axis=1)))
                rows.append(i)
                traj.append(c["x0"][j])
                ts = c["t"]
        rows = np.asarray(rows)
        traj = np.stack(traj)
        pick = np.concatenate([np.flatnonzero(ka[rows] == k)[:KEEP] for k in sorted(set(ka.tolist()))])
        np.savez_compressed(
            OUT / f"{name}_fold0_trace.npz",
            t=ts, x0=traj[pick], row=rows[pick], kind=ka[rows[pick]],
            channel_idx=np.asarray(blob["channel_idx"])[rows[pick]],
            max_abs_vs_saved=np.abs(x - np.asarray(ref["x"])).max(),
        )
        print(name, "rows", len(rows), "max |x - saved|", float(np.abs(x - np.asarray(ref["x"])).max()))


if __name__ == "__main__":
    main()
