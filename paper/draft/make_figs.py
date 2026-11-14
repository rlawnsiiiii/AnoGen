#!/usr/bin/env python3
"""Figures of the paper that the slide script does not draw.

From-noise generation (results/shell_fixsweep_noise, results/paper_trace), the
appendix comparison with the published generators, and the per-event detector
figures (results/paper_detect). Everything is read from saved runs; nothing is
sampled here. Windows are in channel-span units unless a caption says otherwise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIGS = Path(__file__).resolve().parent / "figs"
sys.path.insert(0, str(ROOT / "src"))

INK = "#16304a"
ACCENT = "#1f6fb4"
NOISE = "#7a3e9d"
REAL = "#9b1c1c"
MUTE = "#5c6b7a"
BASE = "#4d4d4d"
GFS = "flip_x0_contrast_head_sharp_nomask"
TWIN = "nosteer_flip_x0_recur2"
NOISE_DIR = ROOT / "results/shell_fixsweep_noise"
DET = ROOT / "results/paper_detect"
KINDS = (
    ("real ESA Point / Global", "Spike"),
    ("real level shift", "Level shift"),
    ("real ESA global subsequence", "Global subseq."),
    ("real ESA local subsequence", "Local subseq."),
)
KIND_COL = {
    "real ESA Point / Global": "#b5533c",
    "real level shift": "#2b6c99",
    "real ESA global subsequence": "#3f7f4f",
    "real ESA local subsequence": "#a07a1c",
}
TEXTW = 6.34  # inches, 16.1 cm

plt.rcParams.update({
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
})


# ------------------------------------------------------------------ data
class Data:
    def __init__(self) -> None:
        from anogen.config import load_config
        from anogen.shell.events import types_from_cfg
        from anogen.shell.morphology import kinds_for_windows
        from anogen.shell.scaler import load_minmax

        cfg = load_config(ROOT / "configs/shell_mission1.yaml")
        self.sc = load_minmax(ROOT / "results/shell_s0/minmax_scaler.npz")
        lab = np.load(ROOT / "results/shell_s0/labeled_arrays.npz")
        meta = pd.read_csv(ROOT / "results/shell_s0/labeled_windows.csv")
        self.anom = meta[meta["kind"] == "anomaly"].reset_index(drop=True)
        x = np.asarray(lab["anomaly"], dtype=np.float64)
        self.ch_a = np.asarray(lab["anomaly_channel"])
        self.fold_a = np.asarray(lab["anomaly_fold"])
        self.kind_a = kinds_for_windows(x, self.anom, types_from_cfg(cfg)).astype(str)
        self.event_a = self.anom["event_id"].astype(str).to_numpy()
        self.real = self.span(x, self.ch_a)
        gal = np.load(ROOT / "results/shell_s3/galleries.npz", allow_pickle=True)
        self.cond_ch = np.asarray(gal["channel_idx"])
        self.static = {
            "Donor": self.span(gal["cond"], self.cond_ch),
            "GenIAS, locked": self.span(gal["genias"], self.cond_ch),
            "GenIAS, reimpl.": self.span(
                np.load(ROOT / "results/shell_genias_fair/galleries.npz")["genias_fair"], self.cond_ch
            ),
            "CutAddPaste": self.span(np.load(ROOT / "results/shell_genbase/cutaddpaste.npz")["x"], self.cond_ch),
            "Hand-built": self.span(gal["posthoc"], self.cond_ch),
        }

    def span(self, x, ch) -> np.ndarray:
        from anogen.shell.realism import channel_span_normalize

        return channel_span_normalize(np.asarray(x, dtype=np.float64), np.asarray(ch), self.sc)

    def fixsweep(self, name: str, fold: int) -> tuple[np.ndarray, np.ndarray]:
        b = np.load(NOISE_DIR / f"{name}_fold{fold}.npz", allow_pickle=True)
        return self.span(b["x"], b["channel_idx"]), np.asarray(b["kind_alloc"]).astype(str)


HOURS = np.linspace(0, 512 * 30 / 3600, 512)


def _panel(a, y, col, *, lw=0.35, ylim=(-0.35, 1.35), guides=True) -> None:
    a.plot(HOURS, y, lw=lw, color=col)
    if guides:
        for yv in (0.0, 1.0):
            a.axhline(yv, lw=0.3, color="0.78", ls=":")
    if ylim is not None:
        a.set_ylim(*ylim)
    a.set_xlim(HOURS[0], HOURS[-1])
    a.set_xticks([])
    a.set_yticks([])
    for s in a.spines.values():
        s.set_color("#c8d2dc")
        s.set_linewidth(0.4)


def _ylim(*ys, pad=0.12) -> tuple[float, float]:
    lo = min(float(np.min(y)) for y in ys)
    hi = max(float(np.max(y)) for y in ys)
    lo, hi = min(lo, -0.05), max(hi, 1.05)
    return lo - pad * (hi - lo), hi + pad * (hi - lo)


def _fixed_or_fit(y, lim=(-0.6, 1.6)) -> tuple[float, float]:
    """A shared range, widened only for a window that would leave it."""
    return lim if (np.min(y) >= lim[0] and np.max(y) <= lim[1]) else _ylim(y)


# ------------------------------------------------------------------ from noise
def fig_noise_gallery(d: Data, path: Path) -> None:
    """Real windows, then GenFSDiff from a donor, then GenFSDiff from N(0, I)."""
    gen = {"donor": {}, "noise": {}}
    for tag, name in (("donor", GFS), ("noise", f"{GFS}_noise")):
        for f in range(3):
            u, ka = d.fixsweep(name, f)
            for k, _ in KINDS:
                gen[tag].setdefault(k, []).append(u[ka == k])
    n_r, n_g = 2, 4
    widths = [1] * n_r + [0.18] + [1] * n_g + [0.18] + [1] * n_g
    fig, ax = plt.subplots(len(KINDS), len(widths), figsize=(TEXTW, 2.45), squeeze=False,
                           gridspec_kw={"width_ratios": widths})
    rng = np.random.default_rng(11)
    for i, (key, label) in enumerate(KINDS):
        r = d.real[d.kind_a == key]
        r = r[rng.choice(len(r), n_r, replace=False)]
        cols = [(r[j], REAL) for j in range(n_r)] + [None]
        for tag, col in (("donor", ACCENT), ("noise", NOISE)):
            g = np.concatenate(gen[tag][key])
            g = g[rng.choice(len(g), n_g, replace=False)]
            cols += [(g[j], col) for j in range(n_g)] + [None]
        for j, c in enumerate(cols[: len(widths)]):
            if c is None:
                ax[i, j].axis("off")
                continue
            _panel(ax[i, j], c[0], c[1])
        ax[i, 0].set_ylabel(label, fontsize=6, rotation=0, ha="right", va="center", labelpad=4)
    ax[0, 0].set_title("REAL", fontsize=6.5, color=REAL, weight="bold", pad=3, loc="left")
    ax[0, n_r + 1].set_title("GenFSDiff FROM A DONOR", fontsize=6.5, color=ACCENT, weight="bold", pad=3,
                             loc="left")
    ax[0, n_r + n_g + 2].set_title("GenFSDiff FROM N(0, I)", fontsize=6.5, color=NOISE, weight="bold",
                                   pad=3, loc="left")
    fig.subplots_adjust(left=0.105, right=0.995, top=0.91, bottom=0.015, wspace=0.08, hspace=0.12)
    fig.savefig(path)
    plt.close(fig)


def fig_noise_chain(d: Data, path: Path) -> None:
    """The edited clean prediction along the chain, same request from both starts."""
    tr = {tag: np.load(ROOT / f"results/paper_trace/{name}_fold0_trace.npz", allow_pickle=True)
          for tag, name in (("donor", GFS), ("noise", f"{GFS}_noise"))}
    final = {"donor": d.fixsweep(GFS, 0)[0], "noise": d.fixsweep(f"{GFS}_noise", 0)[0]}
    shown = (("real ESA Point / Global", "Spike"), ("real level shift", "Level shift"),
             ("real ESA global subsequence", "Global subseq."))
    want = {"noise": (199, 150, 100, 60, 40, 15), "donor": (40, 32, 24, 16, 8, 2)}
    ncol = 7
    fig, ax = plt.subplots(2 * len(shown), ncol, figsize=(TEXTW, 4.1), squeeze=False)
    common = set(tr["noise"]["row"].tolist()) & set(tr["donor"]["row"].tolist())
    for i, (key, label) in enumerate(shown):
        rows = [int(r) for r, k in zip(tr["noise"]["row"], tr["noise"]["kind"]) if k == key and int(r) in common]
        row = rows[0]
        for b, tag in enumerate(("noise", "donor")):
            t = tr[tag]
            j0 = int(np.flatnonzero(t["row"] == row)[0])
            traj, ts = t["x0"][j0], np.asarray(t["t"])
            snaps = [int(np.argmin(np.abs(ts - w))) for w in want[tag]]
            ys = [traj[s] for s in snaps] + [final[tag][row]]
            lim = _ylim(*ys[-3:])
            r_ax = 2 * i + b
            col = NOISE if tag == "noise" else ACCENT
            for j, (y, s) in enumerate(zip(ys, snaps + [None])):
                a = ax[r_ax, j]
                shade = 0.35 + 0.65 * j / (ncol - 1)
                _panel(a, y, col, lw=0.35, ylim=lim)
                a.lines[0].set_alpha(shade)
                a.text(0.03, 0.95, "final" if s is None else f"t = {ts[s]}", fontsize=5.5, color=MUTE,
                       ha="left", va="top", transform=a.transAxes)
            ax[r_ax, 0].set_ylabel(f"{label}\n{'noise' if tag == 'noise' else 'donor'}", fontsize=6,
                                   rotation=0, ha="right", va="center", labelpad=4, color=col)
    fig.subplots_adjust(left=0.115, right=0.995, top=0.995, bottom=0.01, wspace=0.06, hspace=0.12)
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ appendix: generators
def fig_same_donor(d: Data, path: Path) -> None:
    """One donor per row and what each editor makes of it (fold-0 galleries)."""
    gfs, ka = d.fixsweep(GFS, 0)
    gfn, _ = d.fixsweep(f"{GFS}_noise", 0)
    want = (("real ESA Point / Global", 2), ("real level shift", 2), ("real ESA global subsequence", 2),
            ("real ESA local subsequence", 2))
    rng = np.random.default_rng(5)
    rows = []
    for key, n in want:
        idx = np.flatnonzero(ka == key)
        rows += [(int(i), key) for i in rng.choice(idx, n, replace=False)]
    cols = [("Donor", d.static["Donor"], MUTE), ("GenFSDiff", gfs, ACCENT),
            ("GenIAS, locked", d.static["GenIAS, locked"], BASE),
            ("GenIAS, reimpl.", d.static["GenIAS, reimpl."], BASE),
            ("CutAddPaste", d.static["CutAddPaste"], BASE), ("Hand-built", d.static["Hand-built"], BASE),
            ("GenFSDiff, noise", gfn, NOISE)]
    fig, ax = plt.subplots(len(rows), len(cols), figsize=(TEXTW, 5.6), squeeze=False)
    for i, (row, key) in enumerate(rows):
        lim = _ylim(*[c[1][row] for c in cols])
        for j, (name, g, col) in enumerate(cols):
            _panel(ax[i, j], g[row], col, ylim=lim)
            if i == 0:
                ax[i, j].set_title(name, fontsize=6.3, color=col if col != BASE else INK, pad=3)
        lab = dict(KINDS)[key]
        ax[i, 0].set_ylabel(f"{lab}\nch. {41 + int(d.cond_ch[row])}", fontsize=5.8, rotation=0,
                            ha="right", va="center", labelpad=4)
    fig.subplots_adjust(left=0.105, right=0.995, top=0.965, bottom=0.01, wspace=0.06, hspace=0.1)
    fig.savefig(path)
    plt.close(fig)


def fig_nearest(d: Data, path: Path) -> None:
    """Real anomalies and their nearest neighbour in phi from each gallery, z-scored."""
    from anogen.shell.features import embed_windows

    lab_x = np.load(ROOT / "results/shell_s0/labeled_arrays.npz")["anomaly"]
    picks = []
    rng = np.random.default_rng(3)
    for key, _ in KINDS:
        evs = sorted(set(d.event_a[d.kind_a == key].tolist()))
        for ev in rng.permutation(evs)[:2]:
            idx = np.flatnonzero((d.event_a == ev) & (d.kind_a == key))
            picks.append(int(rng.choice(idx)))
    names = [("GenFSDiff", GFS, ACCENT), ("GenFSDiff, noise", f"{GFS}_noise", NOISE), ("Twin", TWIN, MUTE)]
    raw_static = {
        "GenIAS, locked": np.load(ROOT / "results/shell_s3/galleries.npz")["genias"],
        "GenIAS, reimpl.": np.load(ROOT / "results/shell_genias_fair/galleries.npz")["genias_fair"],
        "CutAddPaste": np.load(ROOT / "results/shell_genbase/cutaddpaste.npz")["x"],
        "Hand-built": np.load(ROOT / "results/shell_s3/galleries.npz")["posthoc"],
    }
    emb_static = {k: embed_windows(np.asarray(v)) for k, v in raw_static.items()}
    raw_fold = {}
    for _, n, _ in names:
        for f in range(3):
            raw_fold[(n, f)] = np.asarray(np.load(NOISE_DIR / f"{n}_fold{f}.npz", allow_pickle=True)["x"])
    emb_fold = {k: embed_windows(v) for k, v in raw_fold.items()}

    def z(v):
        v = np.asarray(v, dtype=np.float64)
        return (v - v.mean()) / max(v.std(), 1e-9)

    cols = ["Real"] + [n for n, _, _ in names] + list(raw_static)
    fig, ax = plt.subplots(len(picks), len(cols), figsize=(TEXTW, 5.6), squeeze=False)
    for i, w in enumerate(picks):
        f = int(d.fold_a[w])
        q = embed_windows(lab_x[w : w + 1])[0]
        series = [(z(lab_x[w]), REAL, None)]
        for name, n, col in names:
            dist = np.linalg.norm(emb_fold[(n, f)] - q, axis=1)
            j = int(dist.argmin())
            series.append((z(raw_fold[(n, f)][j]), col, dist[j]))
        for k, v in raw_static.items():
            dist = np.linalg.norm(emb_static[k] - q, axis=1)
            j = int(dist.argmin())
            series.append((z(v[j]), BASE, dist[j]))
        lim = _ylim(*[s[0] for s in series], pad=0.05)
        for j, (y, col, dist) in enumerate(series):
            a = ax[i, j]
            _panel(a, y, col, ylim=lim, guides=False)
            if dist is not None:
                a.text(0.97, 0.05, f"{dist:.2f}", fontsize=5.5, color=INK, ha="right", va="bottom",
                       transform=a.transAxes,
                       bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))
            if i == 0:
                a.set_title(cols[j], fontsize=6.3, color=col if col != BASE else INK, pad=3)
        ax[i, 0].set_ylabel(f"{dict(KINDS)[d.kind_a[w]]}\n{d.event_a[w]}, fold {f}", fontsize=5.8,
                            rotation=0, ha="right", va="center", labelpad=4)
    fig.subplots_adjust(left=0.115, right=0.995, top=0.965, bottom=0.01, wspace=0.06, hspace=0.1)
    fig.savefig(path)
    plt.close(fig)


def fig_random(d: Data, path: Path) -> None:
    """Eight random windows of every gallery, fold 0, channel-span units."""
    rng = np.random.default_rng(23)
    rows = [("Real anomalies", d.real, REAL)]
    for name, n, col in (("GenFSDiff", GFS, ACCENT), ("GenFSDiff, noise", f"{GFS}_noise", NOISE),
                         ("GenFSDiff,\nmatched noise", f"{GFS}_mnoise", NOISE), ("Twin", TWIN, MUTE),
                         ("Twin, noise", f"{TWIN}_noise", MUTE)):
        rows.append((name, d.fixsweep(n, 0)[0], col))
    for name in ("GenIAS, locked", "GenIAS, reimpl.", "CutAddPaste", "Hand-built", "Donor"):
        rows.append((name, d.static[name], BASE if name != "Donor" else MUTE))
    n = 8
    pick = rng.choice(len(d.static["Donor"]), n, replace=False)
    real_pick = np.concatenate([rng.choice(np.flatnonzero(d.kind_a == k), 2, replace=False) for k, _ in KINDS])
    fig, ax = plt.subplots(len(rows), n, figsize=(TEXTW, 6.2), squeeze=False)
    for i, (name, g, col) in enumerate(rows):
        sel = real_pick if name == "Real anomalies" else pick
        for j, r in enumerate(sel):
            _panel(ax[i, j], g[r], col, ylim=_fixed_or_fit(g[r]))
        ax[i, 0].set_ylabel(name, fontsize=6, rotation=0, ha="right", va="center", labelpad=4,
                            color=col if col != BASE else INK)
    fig.subplots_adjust(left=0.13, right=0.995, top=0.995, bottom=0.005, wspace=0.06, hspace=0.12)
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ appendix: detector
def _rows() -> list[dict]:
    keyed = {}
    for p in sorted(DET.glob("rows_shard*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            keyed[(r["mode"], r["k"], r["arm"], r["fold"], r["seed"])] = r
    return list(keyed.values())


ARM_GFS = f"fixsweep:{GFS}+real"
ARM_NOISE = f"fixsweep:{GFS}_noise+real"


def _event_order(d: Data) -> list[str]:
    order = {k: i for i, (k, _) in enumerate(KINDS)}
    evs = sorted(set(d.event_a.tolist()))

    def key(e):
        m = d.event_a == e
        kinds = sorted({order[k] for k in d.kind_a[m]})
        return (int(d.fold_a[m][0]), kinds[0], e)

    return sorted(evs, key=key)


def _event_label(d: Data, e: str) -> str:
    m = d.event_a == e
    ks = [lab for k, lab in KINDS if k in set(d.kind_a[m].tolist())]
    return f"{e} ({', '.join(ks)})"


def fig_events(d: Data, rows: list[dict], path: Path) -> None:
    """Each development event: the window that scores highest under real data alone."""
    summ = json.loads((DET / "summary.json").read_text())
    full = [r for r in rows if r["mode"] == "full" and r["arm"] == "real_only"]
    margin: dict[int, list[float]] = {}
    for r in full:
        for i, s in zip(r["test_idx"], r["test_scores"]):
            margin.setdefault(int(i), []).append(float(s) - float(r["threshold"]))
    evs = _event_order(d)
    ncol = 6
    nrow = int(np.ceil(len(evs) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(TEXTW, 1.05 * nrow + 0.2), squeeze=False)
    hr = {a: summ["full"][a]["event_hit_rate"] for a in ("real_only", ARM_GFS, ARM_NOISE)}
    for n, e in enumerate(evs):
        a = ax[n // ncol, n % ncol]
        idx = np.flatnonzero(d.event_a == e)
        best = max(idx, key=lambda i: np.mean(margin.get(int(i), [-np.inf])))
        k = d.kind_a[best]
        y = d.real[best]
        lo, hi = float(y.min()), float(y.max())
        pad = 0.25 * max(hi - lo, 0.05)
        _panel(a, y, KIND_COL[k], ylim=(lo - pad, hi + 1.6 * pad), guides=False)
        for yv in (0.0, 1.0):
            if lo - pad < yv < hi + 1.6 * pad:
                a.axhline(yv, lw=0.3, color="0.78", ls=":")
        a.set_title(f"{e} · fold {int(d.fold_a[best])} · ch. {41 + int(d.ch_a[best])}", fontsize=5.6,
                    color=INK, pad=2)
        if hi - lo < 1e-6:
            a.text(0.5, 0.3, f"constant at {lo:.2f}", fontsize=5.4, color=KIND_COL[k], ha="center",
                   va="center", transform=a.transAxes)
        txt = f"{hr['real_only'][e] * 5:.0f}/5 · {hr[ARM_GFS][e] * 5:.0f}/5 · {hr[ARM_NOISE][e] * 5:.0f}/5"
        a.text(0.5, -0.05, txt, fontsize=5.4, color=INK, ha="center", va="top", transform=a.transAxes)
        a.text(0.03, 0.95, f"{dict(KINDS)[k]}, {len(idx)} win.", fontsize=5.0, color=KIND_COL[k],
               ha="left", va="top", transform=a.transAxes,
               bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.8))
    for n in range(len(evs), nrow * ncol):
        ax[n // ncol, n % ncol].axis("off")
    fig.subplots_adjust(left=0.01, right=0.995, top=0.96, bottom=0.04, wspace=0.07, hspace=0.42)
    fig.savefig(path)
    plt.close(fig)


def fig_hitmap(d: Data, mode: str, path: Path) -> None:
    """Hit rate over five seeds per event and budget, three arms side by side."""
    summ = json.loads((DET / "summary.json").read_text())
    ks = [str(k) for k in summ["k"]]
    evs = _event_order(d)
    arms = (("real_only", "Real anomalies alone"), (ARM_GFS, "+ GenFSDiff"), (ARM_NOISE, "+ GenFSDiff, noise"))
    fig, ax = plt.subplots(1, 3, figsize=(TEXTW, 3.55), sharey=True)
    for p, (arm, title) in enumerate(arms):
        mat = np.array([[summ[mode][arm][k]["event_hit_rate"][e] for k in ks]
                        + [summ["full"][arm]["event_hit_rate"][e]] for e in evs])
        a = ax[p]
        im = a.imshow(mat, aspect="auto", cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
        a.set_xticks(range(len(ks) + 1))
        a.set_xticklabels(ks + ["all"], fontsize=5.5)
        a.set_xlabel("k", fontsize=6)
        a.set_title(title, fontsize=6.5, color=INK, pad=3)
        a.axvline(len(ks) - 0.5, color="white", lw=1.2)
        for y in range(1, len(evs)):
            if int(d.fold_a[d.event_a == evs[y]][0]) != int(d.fold_a[d.event_a == evs[y - 1]][0]):
                a.axhline(y - 0.5, color=INK, lw=0.6)
        for s in a.spines.values():
            s.set_linewidth(0.4)
        a.tick_params(length=1.5, width=0.4)
    fig.subplots_adjust(left=0.255, right=0.925, top=0.95, bottom=0.12, wspace=0.05)
    cb = fig.colorbar(im, cax=fig.add_axes([0.94, 0.12, 0.012, 0.83]))
    cb.set_label("share of seeds in which the event is found", fontsize=5.5)
    cb.ax.tick_params(labelsize=5, width=0.4, length=1.5)
    cb.outline.set_linewidth(0.4)
    ax[0].set_yticks(range(len(evs)))
    ax[0].set_yticklabels([_event_label(d, e) for e in evs], fontsize=5.2)
    for t, e in zip(ax[0].get_yticklabels(), evs):
        k = [kk for kk, _ in KINDS if kk in set(d.kind_a[d.event_a == e].tolist())][0]
        t.set_color(KIND_COL[k])
    fig.savefig(path)
    plt.close(fig)


def fig_kwindows(d: Data, rows: list[dict], path: Path, k: int = 8, fold: int = 0, seed: int = 0) -> None:
    """The k real and k generated windows one detector of the sweep was given."""

    def row(mode, arm):
        return next(r for r in rows if r["mode"] == mode and r["k"] == k and r["arm"] == arm
                    and r["fold"] == fold and r["seed"] == seed)

    gfs, _ = d.fixsweep(GFS, fold)
    gfn, _ = d.fixsweep(f"{GFS}_noise", fold)
    lines = [
        ("k real windows\n(uniform draw)", [d.real[i] for i in row("uniform", "real_only")["real_idx"]],
         [d.kind_a[i] for i in row("uniform", "real_only")["real_idx"]], REAL),
        ("k generated,\nGenFSDiff", [gfs[i] for i in row("uniform", ARM_GFS)["synth_donor"]], None, ACCENT),
        ("k generated,\nGenFSDiff, noise", [gfn[i] for i in row("uniform", ARM_NOISE)["synth_donor"]], None, NOISE),
        ("k real windows\n(stratified draw)", [d.real[i] for i in row("strat", "real_only")["real_idx"]],
         [d.kind_a[i] for i in row("strat", "real_only")["real_idx"]], REAL),
    ]
    fig, ax = plt.subplots(len(lines), k, figsize=(TEXTW, 2.55), squeeze=False)
    for i, (name, ys, kinds, col) in enumerate(lines):
        for j in range(k):
            a = ax[i, j]
            if j >= len(ys):
                a.axis("off")
                continue
            c = KIND_COL[kinds[j]] if kinds is not None else col
            _panel(a, ys[j], c, ylim=_fixed_or_fit(ys[j]))
        ax[i, 0].set_ylabel(name, fontsize=5.8, rotation=0, ha="right", va="center", labelpad=4)
    fig.subplots_adjust(left=0.14, right=0.995, top=0.995, bottom=0.01, wspace=0.06, hspace=0.12)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    d = Data()
    which = set(sys.argv[1:]) or {"noise", "generators", "detector"}
    if "noise" in which:
        fig_noise_gallery(d, FIGS / "noise_gallery.pdf")
        if (ROOT / "results/paper_trace").is_dir():
            fig_noise_chain(d, FIGS / "noise_chain.pdf")
    if "generators" in which:
        fig_same_donor(d, FIGS / "cmp_same_donor.pdf")
        fig_nearest(d, FIGS / "cmp_nearest.pdf")
        fig_random(d, FIGS / "cmp_random.pdf")
    if "detector" in which:
        rows = _rows()
        fig_events(d, rows, FIGS / "det_events.pdf")
        fig_hitmap(d, "uniform", FIGS / "det_hitmap.pdf")
        fig_hitmap(d, "strat", FIGS / "det_hitmap_strat.pdf")
        fig_kwindows(d, rows, FIGS / "det_kwindows.pdf")
    print("wrote", sorted(p.name for p in FIGS.glob("*.pdf")))


if __name__ == "__main__":
    main()
