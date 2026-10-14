#!/usr/bin/env python3
"""Per-kind morphology diagnostics for fixsweep variants, in channel-span units.

These are the numbers the frozen S4 protocol does not see: whether the window
contains the anomaly it was asked for, how wide and how tall it is, and where
it sits. Project-specific diagnostics, not literature metrics -- they belong
next to the S4 table, never inside it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anogen.config import load_config  # noqa: E402
from anogen.shell.events import types_from_cfg  # noqa: E402
from anogen.shell.morphology import kinds_for_windows  # noqa: E402
from anogen.shell.realism import channel_span_normalize  # noqa: E402
from anogen.shell.scaler import load_minmax  # noqa: E402

SPIKE = "real ESA Point / Global"
STEP = "real level shift"
GUARD = 12  # bins around the peak excluded from the baseline texture


def _peak(w: np.ndarray) -> tuple[int, float]:
    d = np.abs(w - np.median(w))
    p = int(d.argmax())
    return p, float(d[p])


def _baseline(w: np.ndarray) -> float:
    p, _ = _peak(w)
    keep = np.ones(len(w), bool)
    keep[max(0, p - GUARD) : p + GUARD + 1] = False
    return float(np.abs(np.diff(w[keep])).mean())


def spike_stats(u: np.ndarray, real_base: float | None) -> dict[str, float]:
    h, wid, pos, base = [], [], [], []
    for w in u:
        p, a = _peak(w)
        d = np.abs(w - np.median(w))
        h.append(a)
        wid.append(int((d[max(0, p - 8) : p + 9] > 0.5 * a).sum()))
        pos.append(p / len(w))
        base.append(_baseline(w))
    wid_a, base_a = np.asarray(wid), np.asarray(base)
    out = {
        "height": float(np.median(h)),
        "width": float(np.median(wid_a)),
        "share_1bin": float((wid_a == 1).mean()),
        "baseline": float(np.median(base_a)),
        "pos_p10": float(np.quantile(pos, 0.1)),
        "pos_p90": float(np.quantile(pos, 0.9)),
    }
    if real_base is not None:
        out["baseline_ratio"] = out["baseline"] / real_base
        # a draw whose quiet stretch is 3x the real texture is a noise burst,
        # not a spike on a nominal background (ESA_RUN_2026-10-06 §8)
        out["noise_burst"] = float((base_a > 3 * real_base).mean())
    return out


def step_stats(u: np.ndarray, real_rough: float | None) -> dict[str, float]:
    k = int(0.18 * u.shape[1])
    body = np.median(u[:, k:-k], axis=1)
    head, tail = u[:, :k].mean(1) - body, u[:, -k:].mean(1) - body
    is_head = np.abs(head) >= np.abs(tail)
    amp = np.where(is_head, head, tail)
    rough = float(np.median(np.abs(np.diff(u, axis=1)).mean(1)))
    out = {
        "rough": rough,
        "share_head": float(is_head.mean()),
        "amp": float(np.median(np.abs(amp))),
        "share_amp_gt_0.2": float((np.abs(amp) > 0.2).mean()),
    }
    if real_rough is not None:
        out["rough_ratio"] = rough / real_rough
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["_repo_root"])
    s0 = root / cfg["s0_dir"]
    scaler = load_minmax(s0 / "minmax_scaler.npz")
    lab = np.load(s0 / "labeled_arrays.npz")
    meta = pd.read_csv(s0 / "labeled_windows.csv")
    anom = meta[meta["kind"] == "anomaly"].reset_index(drop=True)
    x_a = np.asarray(lab["anomaly"], dtype=np.float64)
    kl = kinds_for_windows(x_a, anom, types_from_cfg(cfg))
    real = channel_span_normalize(x_a, np.asarray(lab["anomaly_channel"]), scaler)

    rs = spike_stats(real[kl == SPIKE], None)
    rt = step_stats(real[kl == STEP], None)
    rows = [("real", rs | {"baseline_ratio": 1.0, "noise_burst": 0.0}, rt | {"rough_ratio": 1.0})]
    for v in args.variants:
        got: dict[str, list[np.ndarray]] = {SPIKE: [], STEP: []}
        for f in range(3):
            p = root / f"results/shell_fixsweep/{v}_fold{f}.npz"
            if not p.is_file():
                continue
            b = np.load(p, allow_pickle=True)
            ka = np.asarray(b["kind_alloc"]).astype(str)
            u = channel_span_normalize(np.asarray(b["x"]), np.asarray(b["channel_idx"]), scaler)
            for k in got:
                got[k].append(u[ka == k])
        if not got[SPIKE]:
            print(f"skip {v}: no folds")
            continue
        rows.append((v, spike_stats(np.concatenate(got[SPIKE]), rs["baseline"]),
                     step_stats(np.concatenate(got[STEP]), rt["rough"])))

    print(f"{'':36s} {'height':>7s} {'width':>6s} {'1-bin':>6s} {'base':>6s} {'burst':>6s} "
          f"{'pos p10-p90':>12s} | {'rough':>6s} {'head':>6s} {'amp':>6s} {'|a|>.2':>7s}")
    for v, s, t in rows:
        print(f"{v:36s} {s['height']:7.3f} {s['width']:6.0f} {s['share_1bin']:6.0%} "
              f"{s['baseline_ratio']:5.2f}x {s['noise_burst']:6.1%} "
              f"{s['pos_p10']:5.2f}-{s['pos_p90']:<6.2f} | {t['rough_ratio']:5.2f}x "
              f"{t['share_head']:6.1%} {t['amp']:6.3f} {t['share_amp_gt_0.2']:7.1%}")
    print("\nreal Point/Global height 0.831, width 1 bin; real level shift is one event, "
          "so share_head 100% is six windows, not a population.")


if __name__ == "__main__":
    main()
