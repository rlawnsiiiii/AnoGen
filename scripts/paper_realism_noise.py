#!/usr/bin/env python3
"""EDI of Table realism, without and with the from-noise galleries.

Reproduces the partition of the paper's realism table (per fold: the fold's
real anomalies plus every listed gallery, k-means with 16 clusters, seed 0,
in the same gallery order) and then refits it with the from-noise galleries
added. Writes results/paper_realism/edi.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from anogen.shell.baselines import cutaddpaste_inject  # noqa: E402
from anogen.shell.coverage import edi_by_method  # noqa: E402
from anogen.shell.features import embed_windows  # noqa: E402

GFS = "flip_x0_contrast_head_sharp_nomask"
PERFOLD = {
    "ours": GFS,
    "ours, masked": "flip_x0_contrast_head_sharp",
    "shell only": "flip_x0_final025_recur2_decay1",
    "no-steer twin": "nosteer_flip_x0_recur2",
    "C1 (prototype)": "c1_repro",
}
NOISE = {
    "ours, noise": f"{GFS}_noise",
    "ours, matched noise": f"{GFS}_mnoise",
    "twin, noise": "nosteer_flip_x0_recur2_noise",
    "twin, matched noise": "nosteer_flip_x0_recur2_mnoise",
}


def main() -> None:
    gal = np.load(ROOT / "results/shell_s3/galleries.npz", allow_pickle=True)
    gf = np.load(ROOT / "results/shell_genias_fair/galleries.npz", allow_pickle=True)
    lab = np.load(ROOT / "results/shell_s0/labeled_arrays.npz")
    afold, anom = np.asarray(lab["anomaly_fold"]), np.asarray(lab["anomaly"])
    cap = cutaddpaste_inject(
        np.asarray(gal["cond"]), rng=np.random.default_rng(0), channel_idx=np.asarray(gal["channel_idx"])
    )
    static = {
        "GenIAS (locked)": gal["genias"], "GenIAS (faithful)": gf["genias_fair"], "CutAddPaste": cap,
        "handcrafted": gal["posthoc"], "donor": gal["cond"],
    }
    static = {k: embed_windows(np.asarray(v)) for k, v in static.items()}

    def fold_gal(d: dict[str, str], f: int, src: str) -> dict[str, np.ndarray]:
        return {n: embed_windows(np.load(ROOT / src / f"{v}_fold{f}.npz", allow_pickle=True)["x"]) for n, v in d.items()}

    out: dict[str, dict[str, list[float]]] = {"table": {}, "with_noise": {}}
    for f in range(3):
        q = embed_windows(anom[afold == f])
        g = fold_gal(PERFOLD, f, "results/shell_fixsweep")
        g.update(static)
        g2 = dict(g)
        g2["real anomalies"] = q
        for n, v in edi_by_method(g2, n_regions=16, seed=0).items():
            out["table"].setdefault(n, []).append(v)
        g3 = fold_gal(PERFOLD, f, "results/shell_fixsweep")
        g3.update(fold_gal(NOISE, f, "results/shell_fixsweep_noise"))
        g3.update(static)
        g3["real anomalies"] = q
        for n, v in edi_by_method(g3, n_regions=16, seed=0).items():
            out["with_noise"].setdefault(n, []).append(v)
    report = {part: {n: {"folds": v, "mean": float(np.mean(v))} for n, v in d.items()} for part, d in out.items()}
    dst = ROOT / "results/paper_realism"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "edi.json").write_text(json.dumps(report, indent=1) + "\n")
    for part, d in report.items():
        print(part)
        for n, v in d.items():
            print(f"  {n:22s} {v['mean']:.3f}  ({' '.join(f'{x:.2f}' for x in v['folds'])})")


if __name__ == "__main__":
    main()
