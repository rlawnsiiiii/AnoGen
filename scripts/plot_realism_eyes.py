"""Envelope-centric eyeball plots for the raw-window realism audit.

Channel-span units: 0 and 1 are the S0 nominal training min/max, so the shaded
band is the envelope the diagnostics in docs/REALISM.md 17.4 count against.

Writes docs/realism_eyes/. Reads only; touches no S3/S4/audit artifact.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from anogen.config import REPO_ROOT, load_config
from anogen.shell.events import types_from_cfg
from anogen.shell.morphology import kinds_for_windows
from anogen.shell.realism import channel_span_normalize, diag_envelope
from anogen.shell.scaler import load_minmax

N_OVERLAY = 40
N_EXAMPLE = 4
SEED = 0
BIN_SECONDS = 30
COLORS = {"real": "#b2182b", "c1": "#2166ac", "hashfix": "#1b9e77", "donor": "#777777"}


def _hours(w: int) -> np.ndarray:
    return np.arange(w) * BIN_SECONDS / 3600.0


def _band(ax: plt.Axes) -> None:
    ax.axhspan(0.0, 1.0, color="#4daf4a", alpha=0.13, zorder=0, lw=0)
    for y in (0.0, 1.0):
        ax.axhline(y, color="#4daf4a", lw=0.8, zorder=1)


def _overlay(out: Path, series: dict[str, np.ndarray], rng: np.random.Generator) -> str:
    fig, axes = plt.subplots(1, len(series), figsize=(4.1 * len(series), 3.5), sharey=True)
    for ax, (name, u) in zip(np.atleast_1d(axes), series.items()):
        idx = rng.choice(len(u), size=min(N_OVERLAY, len(u)), replace=False)
        t = _hours(u.shape[1])
        for i in idx:
            ax.plot(t, u[i], color=COLORS[name], lw=0.5, alpha=0.45)
        frac = float(((u < 0.0) | (u > 1.0)).any(axis=1).mean())
        _band(ax)
        ax.set_title(f"{name}  ({100 * frac:.1f}% of windows leave the band)", fontsize=10)
        ax.set_xlabel("hours")
    np.atleast_1d(axes)[0].set_ylabel("(x - train lo) / (train hi - train lo)")
    axes_l = np.atleast_1d(axes)[0]
    axes_l.set_ylim(-1.6, 2.2)
    fig.suptitle(f"{N_OVERLAY} random windows per gallery, nominal training envelope shaded")
    fig.tight_layout()
    name = "envelope_overlay.png"
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    return name


def _examples(
    out: Path,
    real: np.ndarray,
    gens: dict[str, np.ndarray],
    donor: np.ndarray,
    rng: np.random.Generator,
) -> str:
    rows = [("real anomaly", real, "real")] + [(k, v, k) for k, v in gens.items()]
    fig, axes = plt.subplots(
        len(rows), N_EXAMPLE, figsize=(3.4 * N_EXAMPLE, 2.3 * len(rows)), sharex=True
    )
    pick = rng.choice(len(donor), size=N_EXAMPLE, replace=False)
    for r, (label, u, color) in enumerate(rows):
        sel = rng.choice(len(u), size=N_EXAMPLE, replace=False) if label == "real anomaly" else pick
        for c in range(N_EXAMPLE):
            ax = axes[r, c]
            t = _hours(u.shape[1])
            if label != "real anomaly":
                ax.plot(t, donor[sel[c]], color=COLORS["donor"], lw=0.7, alpha=0.7, label="donor")
            ax.plot(t, u[sel[c]], color=COLORS[color], lw=0.9)
            _band(ax)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
            if r == len(rows) - 1:
                ax.set_xlabel("hours")
    fig.suptitle(
        "Same donor down each column (grey). Real anomalies are independent samples.", fontsize=11
    )
    fig.tight_layout()
    name = "example_windows.png"
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    return name


def _excess_ecdf(out: Path, excess: dict[str, np.ndarray]) -> str:
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for name, e in excess.items():
        s = np.sort(e)
        ax.step(np.maximum(s, 1e-4), np.arange(1, len(s) + 1) / len(s), where="post",
                color=COLORS[name], lw=1.6, label=f"{name} (n={len(s)})")
    ax.set_xscale("log")
    ax.set_xlabel("per-window max excess beyond the envelope (channel spans, log)")
    ax.set_ylabel("fraction of windows at or below")
    ax.set_title("Real anomalies rarely leave the envelope; when they do, they go far")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    name = "excess_ecdf.png"
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    return name


def main() -> None:
    cfg = load_config(Path(REPO_ROOT) / "configs/shell_mission1.yaml")
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    s0 = root / "results/shell_s0"

    lab = np.load(s0 / "labeled_arrays.npz")
    meta = pd.read_csv(s0 / "labeled_windows.csv")
    anomaly_meta = meta[meta["kind"] == "anomaly"].reset_index(drop=True)
    scaler = load_minmax(s0 / "minmax_scaler.npz")
    x_real = np.asarray(lab["anomaly"], dtype=np.float64)
    real_ch = np.asarray(lab["anomaly_channel"], dtype=np.int64)
    real_kind = kinds_for_windows(x_real, anomaly_meta, types_from_cfg(cfg)).astype(str)

    s3g = np.load(root / "results/shell_s3/galleries.npz")
    cond, cond_ch = np.asarray(s3g["cond"]), np.asarray(s3g["channel_idx"], dtype=np.int64)
    paths = {
        "c1": root / "results/shell_kindmix_score_hybrid/time_both_hybrid_needles_fold0.npz",
        "hashfix": root / "results/shell_hashfix/time_both_hashfix_fold0.npz",
    }

    out = root / "docs/realism_eyes"
    out.mkdir(parents=True, exist_ok=True)

    u_real = channel_span_normalize(x_real, real_ch, scaler)
    u_donor = channel_span_normalize(cond, cond_ch, scaler)
    gens, excess = {}, {"real": diag_envelope(x_real, real_ch, scaler)["per_window_max_excess"]}
    for name, path in paths.items():
        f = np.load(path, allow_pickle=True)
        x, ch = np.asarray(f["x"], dtype=np.float64), np.asarray(f["channel_idx"], dtype=np.int64)
        gens[name] = channel_span_normalize(x, ch, scaler)
        excess[name] = diag_envelope(x, ch, scaler)["per_window_max_excess"]

    written = [
        _overlay(out, {"real": u_real, **gens}, np.random.default_rng(SEED)),
        _examples(out, u_real, gens, u_donor, np.random.default_rng(SEED)),
        _excess_ecdf(out, excess),
    ]
    kinds = pd.Series(real_kind).value_counts()
    (out / "INDEX.txt").write_text(
        "Envelope eyeballs for the realism audit (docs/REALISM.md 17).\n"
        f"channel-span units; band = S0 nominal train [min, max].\n"
        f"real anomaly windows={len(x_real)}; kinds:\n"
        + kinds.to_string()
        + "\n\nPlots:\n  "
        + "\n  ".join(written)
        + "\n"
    )
    print("\n".join(str(out / n) for n in written))


if __name__ == "__main__":
    main()
