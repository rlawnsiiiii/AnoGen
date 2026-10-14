#!/usr/bin/env python3
"""One figure per recipe: every target kind, real windows above generated ones.

``plot_kind_grid.py`` compares recipes within one kind. This is the other cut --
one recipe across all four kinds -- which is what you want when the question is
"what does this model produce", not "which recipe is better here".
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
KINDS = (
    "real ESA Point / Global",
    "real level shift",
    "real ESA global subsequence",
    "real ESA local subsequence",
)


def _shifted_end(w: np.ndarray) -> str:
    """Which end of the window sits away from the body, for the level-shift rows."""
    k = int(0.18 * len(w))
    body = float(np.median(w[k:-k]))
    head, tail = float(w[:k].mean()) - body, float(w[-k:].mean()) - body
    return "head" if abs(head) >= abs(tail) else "tail"


def _draw(ax, u: np.ndarray, colour: str, mark_end: bool) -> None:
    ax.plot(np.linspace(0, HOURS, len(u)), u, lw=0.8, color=colour)
    for y in (0.0, 1.0):
        ax.axhline(y, lw=0.5, color="0.6", ls=":")
    ax.set_ylim(-0.35, 1.35)
    ax.set_xticks([0, 2, 4])
    ax.tick_params(labelsize=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if mark_end:
        ax.set_title(_shifted_end(u), fontsize=6, color=colour, pad=1.5)


def _focus(args, root: Path, real: np.ndarray, g: np.ndarray) -> None:
    """One kind, six panels wide, every panel large enough to judge by eye."""
    mark = args.only_kind == "real level shift"
    ends = np.array([_shifted_end(w) for w in g]) if mark else None
    rng = np.random.default_rng(abs(hash(args.only_kind)) % (2**32))
    sel = rng.choice(len(g), min(args.n, len(g)), replace=False)
    blocks = [("real (%d of %d)" % (min(6, len(real)), len(real)), real[:6], "#9b1c1c")]
    for r in range(0, len(sel), 6):
        blocks.append(("generated", g[sel[r : r + 6]], "#1f6fb4"))
    fig, ax = plt.subplots(len(blocks), 6, figsize=(2.6 * 6, 1.9 * len(blocks)), squeeze=False)
    for i, (tag, u, colour) in enumerate(blocks):
        for j in range(6):
            a = ax[i, j]
            if j < len(u):
                _draw(a, u[j], colour, mark)
                a.tick_params(labelsize=8)
                if mark:
                    a.set_title(_shifted_end(u[j]), fontsize=10, color=colour, pad=3)
            else:
                a.axis("off")
            a.set_yticks([0, 1] if j == 0 else [])
        ax[i, 0].set_ylabel(tag, fontsize=9, rotation=0, ha="right", va="center", labelpad=8)
    extra = ""
    if ends is not None:
        extra = (f"  —  whole slice: {100 * (ends == 'head').mean():.0f}% shifted at the head, "
                 f"{100 * (ends == 'tail').mean():.0f}% at the tail, over {len(g)} windows")
    fig.suptitle(f"{args.variant} — {args.only_kind}{extra}", fontsize=12)
    fig.supxlabel("hours", fontsize=9)
    fig.tight_layout(rect=(0.04, 0.01, 1, 0.96))
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{args.variant}__{args.only_kind.replace('/', '-').replace(' ', '_')}.png"
    fig.savefig(dest, dpi=140)
    print(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="docs/model_card")
    ap.add_argument("--only-kind", default=None,
                    help="render just this kind, six wide, so the panels are readable")
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
    real_all = channel_span_normalize(x_a, np.asarray(lab["anomaly_channel"]), scaler)

    gen: dict[str, np.ndarray] = {}
    for f in range(3):
        p = root / f"results/shell_fixsweep/{args.variant}_fold{f}.npz"
        if not p.is_file():
            continue
        b = np.load(p, allow_pickle=True)
        ka = np.asarray(b["kind_alloc"]).astype(str)
        u = channel_span_normalize(np.asarray(b["x"]), np.asarray(b["channel_idx"]), scaler)
        for k in KINDS:
            if (ka == k).any():
                gen[k] = np.concatenate([gen[k], u[ka == k]]) if k in gen else u[ka == k]
    if not gen:
        raise SystemExit(f"no folds found for {args.variant}")

    n = args.n
    if args.only_kind:
        _focus(args, root, real_all[kl == args.only_kind], gen[args.only_kind])
        return
    fig, ax = plt.subplots(2 * len(KINDS), n, figsize=(1.35 * n, 1.15 * 2 * len(KINDS)),
                           squeeze=False)
    for i, kind in enumerate(KINDS):
        r = real_all[kl == kind]
        g = gen.get(kind, np.zeros((0, real_all.shape[1])))
        mark = kind == "real level shift"
        tag_g = "generated"
        if mark and len(g):
            # a draw of n is noisy enough to read as biased when it is not, so the
            # row says what the whole slice does (ESA_RUN_2026-10-06 §10.1)
            ends = np.array([_shifted_end(w) for w in g])
            tag_g = f"generated\n{100 * (ends == 'head').mean():.0f}% head / " \
                    f"{100 * (ends == 'tail').mean():.0f}% tail\nover all {len(g)}"
        # one stream per kind, so one kind's sample does not shift another's
        rng = np.random.default_rng(abs(hash(kind)) % (2**32))
        g = g[rng.choice(len(g), min(n, len(g)), replace=False)] if len(g) else g
        for row, (u, colour, tag) in enumerate(
            ((r, "#9b1c1c", f"REAL\n{kind}\n({len(r)} windows)"), (g, "#1f6fb4", tag_g))
        ):
            for j in range(n):
                a = ax[2 * i + row, j]
                if j < len(u):
                    _draw(a, u[j], colour, mark)
                else:
                    a.axis("off")
                a.set_yticks([0, 1] if j == 0 else [])
            ax[2 * i + row, 0].set_ylabel(tag, fontsize=7, rotation=0, ha="right",
                                          va="center", labelpad=6)
    fig.suptitle(f"{args.variant}: every target kind, real above generated "
                 f"(channel-span units, dotted = training envelope)", fontsize=11)
    fig.supxlabel("hours", fontsize=8)
    fig.tight_layout(rect=(0.08, 0.01, 1, 0.975))
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{args.variant}.png"
    fig.savefig(dest, dpi=140)
    print(dest)


if __name__ == "__main__":
    main()
