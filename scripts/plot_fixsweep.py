"""Figures for the fix sweep (numpy + matplotlib; reads saved galleries only).

    .venv/bin/python scripts/plot_fixsweep.py -c configs/shell_mission1.yaml

Writes docs/fixsweep/*.png:

* positions.png      where the dominant excursion sits, per variant, real in grey
* ratios_<m>.png     one metric per chart (env exit, diff p99.9, peak@start),
                     as generated / real; 1 = matches the real anomalies
* kind_<slug>.png    random windows of each allocated kind slice for
                     c1_repro and the variants named with --show (same donors)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anogen.config import load_config
from anogen.shell.evaluation import peak_positions
from anogen.shell.morphology import KIND_SLUG
from anogen.shell.realism import channel_span_normalize
from anogen.shell.scaler import load_minmax

# Reference categorical palette (light), fixed order; real anomalies in neutral ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
REAL = "#52514e"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title: str) -> None:
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--dir", default=None, help="fixsweep results dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", nargs="*", default=["flip_x0_final025", "flip_x0_contrast"])
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config(args.config)
    root = Path(cfg["_repo_root"])
    s0 = root / cfg["s0_dir"]
    fs = Path(args.dir) if args.dir else root / "results/shell_fixsweep"
    out = Path(args.out) if args.out else root / "docs/fixsweep"
    out.mkdir(parents=True, exist_ok=True)
    scaler = load_minmax(s0 / "minmax_scaler.npz")
    lab = np.load(s0 / "labeled_arrays.npz")
    rm = np.asarray(lab["anomaly_fold"]) == args.fold
    real_u = channel_span_normalize(np.asarray(lab["anomaly"])[rm], np.asarray(lab["anomaly_channel"])[rm], scaler)
    real_pos = peak_positions(real_u)

    gals = {}
    for p in sorted(fs.glob(f"*_fold{args.fold}.npz")):
        b = np.load(p, allow_pickle=True)
        gals[p.name[: -len(f"_fold{args.fold}.npz")]] = (
            channel_span_normalize(b["x"], b["channel_idx"], scaler),
            np.asarray(b["kind_alloc"]).astype(str) if "kind_alloc" in b.files else None,
        )
    if not gals:
        raise SystemExit(f"no galleries in {fs}")
    names = list(gals)

    # 1. positions: small multiples, one per variant, real as a grey step outline
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 1.9 * rows), squeeze=False, sharey=True)
    bins = np.linspace(0, 1, 21)
    real_h, _ = np.histogram(real_pos, bins=bins, density=True)
    for i, ax in enumerate(axes.flat):
        if i >= len(names):
            ax.axis("off")
            continue
        pos = peak_positions(gals[names[i]][0])
        h, _ = np.histogram(pos, bins=bins, density=True)
        ax.bar(bins[:-1], h, width=np.diff(bins) * 0.9, align="edge", color=SERIES[0], linewidth=0)
        ax.step(bins, np.append(real_h, real_h[-1]), where="post", color=REAL, linewidth=1.4, label="real anomalies")
        ax.axhline(1.0, color=MUTED, linewidth=0.6, linestyle=":")
        _style(ax, f"{names[i]}  (start {np.mean(pos < 0.1):.2f})")
        if i == 0:
            ax.legend(fontsize=6, frameon=False)
    fig.suptitle(f"Position of the dominant excursion (fold {args.fold}); uniform = dotted line", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "positions.png", dpi=150)
    plt.close(fig)

    # 2. one metric per chart, generated / real
    def metrics(u: np.ndarray) -> dict[str, float]:
        exc = np.maximum(-u, np.maximum(u - 1.0, 0.0)).max(axis=1)
        d = np.abs(np.diff(u, axis=1))
        return {
            "envelope exit fraction": float(np.mean(exc > 0)),
            "first-difference p99.9": float(np.mean(np.quantile(d, 0.999, axis=1))),
            "peaks in first 10 %": float(np.mean(peak_positions(u) < 0.1)),
        }

    real_m = metrics(real_u)
    gm = {n: metrics(gals[n][0]) for n in names}
    for metric in real_m:
        fig, ax = plt.subplots(figsize=(7.0, 0.32 * len(names) + 1.0))
        vals = [gm[n][metric] / max(real_m[metric], 1e-12) for n in names]
        ys = np.arange(len(names))
        ax.barh(ys, vals, height=0.6, color=SERIES[0], linewidth=0)
        ax.axvline(1.0, color=REAL, linewidth=1.2)
        for y, v in zip(ys, vals, strict=True):
            ax.text(v, y, f" {v:.2f}×", va="center", fontsize=7, color=INK)
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("generated / real anomalies (1 = match)", fontsize=8, color=MUTED)
        _style(ax, f"{metric}: real = {real_m[metric]:.3f}")
        ax.grid(True, axis="x", color=GRID, linewidth=0.6)
        ax.grid(False, axis="y")
        fig.tight_layout()
        slug = metric.split()[0].replace("-", "_").lower()
        fig.savefig(out / f"ratios_{slug}.png", dpi=150)
        plt.close(fig)

    # 3. random windows per kind slice
    show = [n for n in ["c1_repro", *args.show] if n in gals]
    alloc = gals[show[0]][1] if show else None
    rng = np.random.default_rng(0)
    written = ["positions.png"] + [f"ratios_{m.split()[0].replace('-', '_').lower()}.png" for m in real_m]
    if alloc is not None:
        for kind in sorted(set(alloc.tolist())):
            idx = np.flatnonzero(alloc == kind)
            pick = rng.choice(idx, size=min(4, len(idx)), replace=False)
            fig, axes = plt.subplots(len(show), len(pick), figsize=(2.4 * len(pick), 1.5 * len(show)), squeeze=False, sharey="row")
            for r, n in enumerate(show):
                for c, j in enumerate(pick):
                    ax = axes[r, c]
                    ax.plot(gals[n][0][j], color=SERIES[r % len(SERIES)], linewidth=1.0)
                    ax.axhspan(0, 1, color=GRID, alpha=0.35, linewidth=0)
                    _style(ax, n if c == 0 else "")
            fig.suptitle(f"{kind}: same donors across variants (grey band = training envelope)", fontsize=9, color=INK, x=0.01, ha="left")
            fig.tight_layout()
            name = f"kind_{KIND_SLUG.get(kind, kind.replace(' ', '_'))}.png"
            fig.savefig(out / name, dpi=150)
            plt.close(fig)
            written.append(name)
    (out / "INDEX.txt").write_text("Fix sweep figures (scripts/plot_fixsweep.py).\n" + "\n".join(written) + "\n")
    print(json.dumps({"written": written, "real": real_m}, indent=1))


if __name__ == "__main__":
    main()
