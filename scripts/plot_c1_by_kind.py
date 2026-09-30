"""What a hybrid_needles gallery actually generates, per targeted kind.

One figure per allocated kind: 16 generations with the nominal donor each was
steered from, plus a real-vs-generated overview. Channel-span units, so the
shaded band is the S0 nominal training envelope.

Defaults to c1 into docs/c1_by_kind/. Reads only; writes no result artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from anogen.config import REPO_ROOT, load_config
from anogen.shell.events import types_from_cfg
from anogen.shell.morphology import KIND_ORDER, KIND_SLUG, kinds_for_windows
from anogen.shell.realism import channel_span_normalize
from anogen.shell.scaler import load_minmax

N_GRID = 16
N_OVERVIEW = 5
SEED = 0
BIN_SECONDS = 30
GEN = "#2166ac"
DONOR = "#9e9e9e"
REAL = "#b2182b"
SHORT = {
    "real ESA Point / Global": "Point/Global",
    "real ESA local subsequence": "local subsequence",
    "real level shift": "level shift",
    "real ESA global subsequence": "global subsequence",
    "real ESA Point / Local": "Point/Local",
}


def _hours(w: int) -> np.ndarray:
    return np.arange(w) * BIN_SECONDS / 3600.0


def _panel(
    ax: plt.Axes,
    t: np.ndarray,
    gen: np.ndarray,
    donor: np.ndarray | None,
    color: str = GEN,
) -> None:
    stack = [gen] if donor is None else [gen, donor]
    lo = min(float(s.min()) for s in stack)
    hi = max(float(s.max()) for s in stack)
    pad = 0.08 * max(hi - lo, 1e-3)
    if donor is not None:
        ax.plot(t, donor, color=DONOR, lw=0.7, alpha=0.85)
    ax.plot(t, gen, color=color, lw=0.8)
    ax.axhspan(0.0, 1.0, color="#4daf4a", alpha=0.12, zorder=0, lw=0)
    for y in (0.0, 1.0):
        ax.axhline(y, color="#4daf4a", lw=0.7, zorder=1)
    ax.set_ylim(lo - pad, hi + pad)
    ax.tick_params(labelsize=7)


def _grid(
    out: Path, kind: str, gen: np.ndarray, donor: np.ndarray, ch: np.ndarray, label: str
) -> str:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(gen), size=min(N_GRID, len(gen)), replace=False)
    cols = 4
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.0 * rows), sharex=True)
    axes = np.atleast_2d(axes)
    t = _hours(gen.shape[1])
    for i, j in enumerate(idx):
        ax = axes[i // cols, i % cols]
        _panel(ax, t, gen[j], donor[j])
        ax.set_title(f"ch{int(ch[j]) + 41}  #{int(j)}", fontsize=8)
        if i // cols == rows - 1:
            ax.set_xlabel("hours", fontsize=8)
    for k in range(len(idx), rows * cols):
        axes[k // cols, k % cols].axis("off")
    fig.suptitle(
        f"{label} targeting {SHORT.get(kind, kind)} — {len(idx)} random generations "
        f"(grey = nominal donor, band = training envelope)",
        fontsize=12,
    )
    fig.tight_layout()
    name = f"{label}_{KIND_SLUG.get(kind, 'kind')}.png"
    fig.savefig(out / name, dpi=145)
    plt.close(fig)
    return name


def _overview(
    out: Path,
    kinds: list[str],
    gen_by_kind: dict[str, np.ndarray],
    real_by_kind: dict[str, np.ndarray],
    label: str,
) -> str:
    rows = [k for k in kinds if len(gen_by_kind.get(k, ()))]
    fig, axes = plt.subplots(
        2 * len(rows), N_OVERVIEW, figsize=(2.9 * N_OVERVIEW, 1.7 * 2 * len(rows)), sharex=True
    )
    rng = np.random.default_rng(SEED)
    for r, kind in enumerate(rows):
        for which, source, color in ((0, real_by_kind.get(kind), REAL), (1, gen_by_kind[kind], GEN)):
            ar = 2 * r + which
            n = 0 if source is None else len(source)
            row_label = ("real" if which == 0 else label) + f"\n{SHORT.get(kind, kind)}"
            pick = rng.choice(n, size=min(N_OVERVIEW, n), replace=False) if n else np.zeros(0, int)
            for c in range(N_OVERVIEW):
                ax = axes[ar, c]
                if c >= len(pick):
                    ax.axis("off")
                    continue
                _panel(ax, _hours(source.shape[1]), source[pick[c]], None, color=color)
                if c == 0:
                    ax.set_ylabel(row_label, fontsize=8)
                if ar == 2 * len(rows) - 1:
                    ax.set_xlabel("hours", fontsize=8)
            if n == 0:
                axes[ar, 0].axis("on")
                axes[ar, 0].set_xticks([])
                axes[ar, 0].set_yticks([])
                axes[ar, 0].set_ylabel(f"{row_label}\n(none in set)", fontsize=8)
    fig.suptitle(f"Real anomalies (red) vs {label} generations (blue), by kind", fontsize=12)
    fig.tight_layout()
    name = f"{label}_vs_real_by_kind.png"
    fig.savefig(out / name, dpi=145)
    plt.close(fig)
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gallery",
        default="results/shell_kindmix_score_hybrid/time_both_hybrid_needles_fold0.npz",
    )
    ap.add_argument("--out", default="docs/c1_by_kind")
    ap.add_argument("--label", default="c1")
    args = ap.parse_args()

    cfg = load_config(Path(REPO_ROOT) / "configs/shell_mission1.yaml")
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = root / "results/shell_s0"
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    scaler = load_minmax(s0 / "minmax_scaler.npz")

    lab = np.load(s0 / "labeled_arrays.npz")
    meta = pd.read_csv(s0 / "labeled_windows.csv")
    anomaly_meta = meta[meta["kind"] == "anomaly"].reset_index(drop=True)
    x_real = np.asarray(lab["anomaly"], dtype=np.float64)
    real_ch = np.asarray(lab["anomaly_channel"], dtype=np.int64)
    real_kind = kinds_for_windows(x_real, anomaly_meta, types_from_cfg(cfg)).astype(str)
    u_real = channel_span_normalize(x_real, real_ch, scaler)

    s3g = np.load(root / "results/shell_s3/galleries.npz")
    cond = np.asarray(s3g["cond"], dtype=np.float64)
    cond_ch = np.asarray(s3g["channel_idx"], dtype=np.int64)
    u_donor = channel_span_normalize(cond, cond_ch, scaler)

    f = np.load(root / args.gallery, allow_pickle=True)
    x = np.asarray(f["x"], dtype=np.float64)
    ch = np.asarray(f["channel_idx"], dtype=np.int64)
    alloc = np.asarray(f["kind_alloc"]).astype(str)
    u_gen = channel_span_normalize(x, ch, scaler)

    kinds = [k for k in KIND_ORDER if np.any(alloc == k)]
    written, gen_by_kind, real_by_kind, counts = [], {}, {}, []
    for kind in kinds:
        m = alloc == kind
        gen_by_kind[kind] = u_gen[m]
        rm = real_kind == kind
        real_by_kind[kind] = u_real[rm] if np.any(rm) else None
        chans = sorted((np.unique(ch[m]) + 41).tolist())
        counts.append(
            f"  {kind}: n={int(m.sum())} on channels {chans}, real n={int(rm.sum())}"
        )
        written.append(_grid(out, kind, u_gen[m], u_donor[: len(u_gen)][m], ch[m], args.label))
    written.append(_overview(out, kinds, gen_by_kind, real_by_kind, args.label))

    (out / "INDEX.txt").write_text(
        f"{args.label} = {args.gallery}, targeted by allocated kind.\n"
        "Channel-span units; green band = S0 nominal training envelope.\n"
        "Grey in the per-kind grids is the nominal donor the window was steered from.\n\n"
        + "\n".join(counts)
        + "\n\nPlots:\n  "
        + "\n  ".join(written)
        + "\n"
    )
    print("\n".join(str(out / n) for n in written))


if __name__ == "__main__":
    main()
