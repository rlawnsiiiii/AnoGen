"""Eyeball the fix-sweep variants against the real anomalies, kind by kind.

    .venv/bin/python scripts/plot_fixsweep_eyes.py -c configs/shell_mission1.yaml

Same layout as paper/figures/c1_*/09_htune_nearest.png: one column per ESA
kind, the real reference on top, one row per variant. Writes docs/fixsweep_eyes/:

* nearest.png     per cell the generated window closest to the real reference
                  in frozen φ — the best case the gallery can show
* aimed.png       the same cells drawn at random instead (seeded), i.e. what a
                  typical draw looks like
* envelope.png    nearest.png in channel-span units with the S0 training
                  envelope drawn, so excursions past it are visible

The first two use raw window units like the paper figures (a shared y-axis per
column), the third channel-span units, where 0 and 1 are the envelope rails.
Reads saved artifacts only; writes no result directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from anogen.config import load_config
from anogen.shell.events import types_from_cfg
from anogen.shell.features import embed_windows
from anogen.shell.morphology import KIND_ORDER, kinds_for_windows
from anogen.shell.realism import channel_span_normalize
from anogen.shell.scaler import load_minmax

BIN_SECONDS = 30
REAL = "#b2182b"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#008300", "#e34948"]
ENVELOPE = "#4daf4a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SHORT = {
    "real ESA Point / Global": "real ESA Point / Global",
    "real ESA local subsequence": "real ESA local subsequence",
    "real level shift": "real level shift",
    "real ESA global subsequence": "real ESA global subsequence",
}
DEFAULT_VARIANTS = [
    "c1_repro",
    "flip_x0_contrast_only",
    "flip_x0_contrast_masked",
    "flip_x0_final025_recur2_decay1",
    "nosteer_flip_x0_masked",
]
# the full names do not fit beside a 1.45 in row
ROW_LABEL = {
    "c1_repro": "C1\n(c1_repro)",
    "flip_x0_contrast_masked_room": "contrast\nmasked room",
    "flip_x0_contrast_head_kindpolish": "head +\nkind polish",
    "flip_x0_contrast_head": "head",
    "flip_x0_contrast_masked_room_polish": "room +\npolish .1",
    "flip_x0_contrast_masked_room_polish2": "room +\npolish .2",
    "flip_x0_contrast_masked_room_decay2": "room +\ndecay2",
    "flip_x0_contrast_only": "contrast\nonly",
    "flip_x0_contrast_masked": "contrast\nmasked",
    "flip_x0_contrast_masked_div": "contrast\nmasked div",
    "flip_x0_final025_recur2_decay1": "recur2\ndecay1",
    "flip_x0_final025_recur2": "recur2",
    "nosteer_flip_x0_masked": "no-steer\ntwin (masked)",
    "nosteer_flip_x0_recur2": "no-steer\ntwin (recur2)",
    "nosteer_c1": "no-steer\ntwin (C1)",
}


def _style(ax: plt.Axes, title: str = "") -> None:
    if title:
        ax.set_title(title, fontsize=8, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7)


def _figure(
    out: Path,
    name: str,
    title: str,
    kinds: list[str],
    rows: list[tuple[str, str, dict[str, np.ndarray]]],
    *,
    envelope: bool,
) -> str:
    """rows = [(label, colour, {kind: (W,) series})]; column 0 of each row is one kind."""
    fig, axes = plt.subplots(
        len(rows), len(kinds), figsize=(2.6 * len(kinds) + 0.6, 1.45 * len(rows)), squeeze=False
    )
    for c, kind in enumerate(kinds):
        series = [r[2][kind] for r in rows if kind in r[2]]
        if envelope:
            lo = min(0.0, min(float(s.min()) for s in series))
            hi = max(1.0, max(float(s.max()) for s in series))
        else:
            lo = min(float(s.min()) for s in series)
            hi = max(float(s.max()) for s in series)
        pad = 0.06 * max(hi - lo, 1e-3)
        for r, (label, colour, by_kind) in enumerate(rows):
            ax = axes[r, c]
            _style(ax, SHORT.get(kind, kind) if r == 0 else "")
            if kind not in by_kind:
                ax.axis("off")
                continue
            y = by_kind[kind]
            t = np.arange(len(y)) * BIN_SECONDS / 3600.0
            if envelope:
                ax.axhspan(0.0, 1.0, color=ENVELOPE, alpha=0.12, zorder=0, lw=0)
                for rail in (0.0, 1.0):
                    ax.axhline(rail, color=ENVELOPE, lw=0.7, zorder=1)
            ax.plot(t, y, color=colour, lw=0.8)
            ax.set_ylim(lo - pad, hi + pad)
            if c == 0:
                ax.set_ylabel(label, fontsize=8, color=INK, rotation=0, ha="right", va="center", labelpad=28)
            if r == len(rows) - 1:
                ax.set_xlabel("hours", fontsize=8, color=MUTED)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / name, dpi=145)
    plt.close(fig)
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--dir", default="results/shell_fixsweep")
    ap.add_argument("--out", default="docs/fixsweep_eyes")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["_repo_root"])
    s0 = root / cfg["s0_dir"]
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    scaler = load_minmax(s0 / "minmax_scaler.npz")

    lab = np.load(s0 / "labeled_arrays.npz")
    meta = pd.read_csv(s0 / "labeled_windows.csv")
    anom = meta[meta["kind"] == "anomaly"].reset_index(drop=True)
    x_real = np.asarray(lab["anomaly"], dtype=np.float64)
    ch_real = np.asarray(lab["anomaly_channel"], dtype=np.int64)
    fold_real = np.asarray(lab["anomaly_fold"], dtype=np.int64)
    kind_real = kinds_for_windows(x_real, anom, types_from_cfg(cfg)).astype(str)

    gals: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in args.variants:
        p = root / args.dir / f"{name}_fold{args.fold}.npz"
        if not p.is_file():
            print(f"skip {name}: no {p}")
            continue
        b = np.load(p, allow_pickle=True)
        gals[name] = (
            np.asarray(b["x"], dtype=np.float64),
            np.asarray(b["channel_idx"], dtype=np.int64),
            np.asarray(b["kind_alloc"]).astype(str),
        )
    if not gals:
        raise SystemExit(f"no galleries under {root / args.dir}")

    first = next(iter(gals.values()))
    kinds = [k for k in KIND_ORDER if np.any(first[2] == k) and np.any(kind_real == k)]

    # One real reference per kind: the held-out fold's window whose φ is most
    # central among that kind's, so the row is typical rather than extreme.
    ref: dict[str, int] = {}
    for kind in kinds:
        idx = np.flatnonzero((kind_real == kind) & (fold_real == args.fold))
        if len(idx) == 0:
            idx = np.flatnonzero(kind_real == kind)
        phi = embed_windows(x_real[idx])
        ref[kind] = int(idx[np.argmin(np.linalg.norm(phi - phi.mean(axis=0), axis=1))])

    rng = np.random.default_rng(0)
    near_raw, near_span, aimed_raw = {}, {}, {}
    chosen: dict[str, dict[str, dict]] = {}
    for name, (x, ch, alloc) in gals.items():
        u = channel_span_normalize(x, ch, scaler)
        near_raw[name], near_span[name], aimed_raw[name] = {}, {}, {}
        chosen[name] = {}
        for kind in kinds:
            cand = np.flatnonzero(alloc == kind)
            if len(cand) == 0:
                continue
            # nearest in the frozen φ of the kind's real reference
            d = np.linalg.norm(
                embed_windows(x[cand]) - embed_windows(x_real[ref[kind] : ref[kind] + 1]), axis=1
            )
            j_near = int(cand[int(np.argmin(d))])
            j_aim = int(rng.choice(cand))
            near_raw[name][kind] = x[j_near]
            near_span[name][kind] = u[j_near]
            aimed_raw[name][kind] = x[j_aim]
            chosen[name][kind] = {
                "nearest_row": j_near,
                "nearest_channel": int(ch[j_near]) + 41,
                "phi_distance": float(d.min()),
                "aimed_row": j_aim,
                "aimed_channel": int(ch[j_aim]) + 41,
            }

    real_raw = {k: x_real[ref[k]] for k in kinds}
    real_span = {k: channel_span_normalize(x_real[ref[k]][None, :], ch_real[ref[k] : ref[k] + 1], scaler)[0] for k in kinds}

    def rows(real: dict, per: dict) -> list[tuple[str, str, dict]]:
        out_rows = [("real\nanomaly", REAL, real)]
        for i, name in enumerate(gals):
            out_rows.append((ROW_LABEL.get(name, name), SERIES[i % len(SERIES)], per[name]))
        return out_rows

    written = [
        _figure(
            out, "nearest.png",
            f"Fix sweep, fold {args.fold}: nearest generation to the real reference in frozen φ",
            kinds, rows(real_raw, near_raw), envelope=False,
        ),
        _figure(
            out, "aimed.png",
            f"Fix sweep, fold {args.fold}: one random generation per targeted kind",
            kinds, rows(real_raw, aimed_raw), envelope=False,
        ),
        _figure(
            out, "envelope.png",
            f"Fix sweep, fold {args.fold}: nearest generation in channel-span units "
            "(band = S0 training envelope)",
            kinds, rows(real_span, near_span), envelope=True,
        ),
    ]
    (out / "INDEX.txt").write_text(
        f"scripts/plot_fixsweep_eyes.py on {args.dir}, fold {args.fold}.\n"
        "Columns are the targeted kinds, the top row one real anomaly of that kind\n"
        "(the φ-central window of the held-out fold). nearest.png picks, per cell,\n"
        "the generated window closest to that reference in frozen φ; aimed.png picks\n"
        "at random (seed 0); envelope.png repeats nearest.png in channel-span units.\n\n"
        "Plots:\n  " + "\n  ".join(written) + "\n\n"
        + json.dumps(chosen, indent=1) + "\n"
    )
    print(json.dumps({"written": written, "kinds": kinds, "variants": list(gals)}, indent=1))


if __name__ == "__main__":
    main()
