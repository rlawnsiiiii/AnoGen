#!/usr/bin/env python3
"""Many generations of one kind per variant, next to every real window of that kind.

``plot_fixsweep_eyes.py`` shows the single nearest generation per cell, which hides
how often a recipe misses. This shows a row of n draws per variant so consistency is
visible, on the channel-span axis so the [0, 1] training envelope is a fixed rail.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anogen.config import load_config  # noqa: E402
from anogen.shell.events import types_from_cfg  # noqa: E402
from anogen.shell.morphology import kinds_for_windows  # noqa: E402
from anogen.shell.realism import channel_span_normalize  # noqa: E402
from anogen.shell.scaler import load_minmax  # noqa: E402

HOURS = 512 * 30.0 / 3600.0


def _row(axes, u: np.ndarray, label: str, colour: str) -> None:
    for j, ax in enumerate(axes):
        if j < len(u):
            ax.plot(np.linspace(0, HOURS, u.shape[1]), u[j], lw=0.8, color=colour)
            ax.axhline(0.0, lw=0.5, color="0.6", ls=":")
            ax.axhline(1.0, lw=0.5, color="0.6", ls=":")
            ax.set_ylim(-0.35, 1.35)
        else:
            ax.axis("off")
        ax.set_xticks([0, 2, 4])
        ax.set_yticks([0, 1] if j == 0 else [])
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center", labelpad=38)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--kind", default="real level shift")
    ap.add_argument("--out", default="docs/kind_grid")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--variants", nargs="*", required=True)
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
    real = channel_span_normalize(x_a, np.asarray(lab["anomaly_channel"]), scaler)[kl == args.kind]

    rows: list[tuple[str, np.ndarray]] = [("real\n(all)", real[: args.n])]
    fs = root / "results/shell_fixsweep"
    rng = np.random.default_rng(0)
    for v in args.variants:
        got = []
        for f in range(3):
            p = fs / f"{v}_fold{f}.npz"
            if not p.is_file():
                continue
            b = np.load(p, allow_pickle=True)
            m = np.asarray(b["kind_alloc"]).astype(str) == args.kind
            if m.sum():
                u = channel_span_normalize(np.asarray(b["x"]), np.asarray(b["channel_idx"]), scaler)
                got.append(u[m])
        if not got:
            print(f"skip {v}: no {args.kind}")
            continue
        g = np.concatenate(got)
        rows.append((v.replace("flip_x0_contrast_masked_room", "room").replace("flip_x0_", ""),
                     g[rng.choice(len(g), min(args.n, len(g)), replace=False)]))

    n = args.n
    fig, ax = plt.subplots(len(rows), n, figsize=(1.55 * n, 1.25 * len(rows)), squeeze=False)
    colours = ["#9b1c1c"] + list(matplotlib.colormaps["tab10"].colors)
    for i, (label, u) in enumerate(rows):
        _row(ax[i], u, label, colours[i % len(colours)])
    fig.suptitle(f"{args.kind}: {n} draws per recipe, channel-span units (dotted = training envelope)",
                 fontsize=10)
    fig.supxlabel("hours", fontsize=8)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.97))
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{args.kind.replace('/', '-').replace(' ', '_')}.png"
    fig.savefig(dest, dpi=140)
    print(dest)


if __name__ == "__main__":
    main()
