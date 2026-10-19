#!/usr/bin/env python3
"""Draw the five data figures the Beamer deck includes.

Everything the deck quotes is reproducible from the run: windows from
`results/shell_s0`, generated windows from `results/shell_fixsweep`, the budget
sweep from `results/shell_augdetect/sweep.json`. Text, equations and tables
live in `anogen_briefing.tex`, not here.

Figures are sized in inches to the space they occupy on a 16x9cm Beamer page
(textwidth = 14cm = 5.51in), so they are included at scale 1 and their fonts
land at the size written here.
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
REAL = "#9b1c1c"
GOOD = "#1a7a4c"
MUTE = "#5c6b7a"
MODEL = "flip_x0_contrast_head_sharp_nomask"
KINDS = (
    ("real ESA Point / Global", "Point / Global"),
    ("real level shift", "level shift"),
    ("real ESA global subsequence", "global subseq."),
    ("real ESA local subsequence", "local subseq."),
)

plt.rcParams.update({
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 7,
})


def _load_windows():
    from anogen.config import load_config
    from anogen.shell.events import types_from_cfg
    from anogen.shell.morphology import kinds_for_windows
    from anogen.shell.realism import channel_span_normalize
    from anogen.shell.scaler import load_minmax

    cfg = load_config(ROOT / "configs/shell_mission1.yaml")
    sc = load_minmax(ROOT / "results/shell_s0/minmax_scaler.npz")
    lab = np.load(ROOT / "results/shell_s0/labeled_arrays.npz")
    meta = pd.read_csv(ROOT / "results/shell_s0/labeled_windows.csv")
    anom = meta[meta["kind"] == "anomaly"].reset_index(drop=True)
    x = np.asarray(lab["anomaly"], dtype=np.float64)
    kl = kinds_for_windows(x, anom, types_from_cfg(cfg))
    real = channel_span_normalize(x, np.asarray(lab["anomaly_channel"]), sc)
    gen: dict[str, np.ndarray] = {}
    for f in range(3):
        b = np.load(ROOT / f"results/shell_fixsweep/{MODEL}_fold{f}.npz", allow_pickle=True)
        ka = np.asarray(b["kind_alloc"]).astype(str)
        u = channel_span_normalize(np.asarray(b["x"]), np.asarray(b["channel_idx"]), sc)
        for k, _ in KINDS:
            if (ka == k).any():
                gen[k] = np.concatenate([gen[k], u[ka == k]]) if k in gen else u[ka == k]
    return real, kl, gen


def fig_diffusion_strip(path: Path) -> None:
    """Five panels of a synthetic 'photo' noised from x0 to xT.

    Drawn procedurally because no stock image ships with the repo. The arrows
    and labels around it are TikZ, in the .tex.
    """
    n = 72
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1)
    img = np.full((n, n), 0.06)
    stars = rng.random((n, n)) > 0.985
    img[stars] = rng.uniform(0.5, 1.0, int(stars.sum()))
    disc = np.hypot(xx - 0.52, yy - 0.60) < 0.38
    shade = np.clip(1.15 - 1.5 * np.hypot(xx - 0.36, yy - 0.42), 0.12, 1.0)
    img[disc] = (0.25 + 0.75 * shade)[disc]
    for cx, cy, cr in ((0.45, 0.52, 0.07), (0.66, 0.70, 0.05),
                       (0.56, 0.84, 0.045), (0.40, 0.75, 0.035)):
        img[(np.hypot(xx - cx, yy - cy) < cr) & disc] *= 0.72
    img = np.clip(img, 0, 1)

    noise = np.random.default_rng(0).standard_normal(img.shape)
    fig, ax = plt.subplots(1, 5, figsize=(5.2, 1.08))
    for a, lv in zip(ax, (0.0, 0.25, 0.55, 0.85, 1.0)):
        a.imshow(np.clip(np.sqrt(1 - lv**2) * img + lv * noise * 0.5 + lv * 0.25, 0, 1),
                 cmap="gray", vmin=0, vmax=1)
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_color("#b9c6d2")
            s.set_linewidth(0.5)
    fig.subplots_adjust(left=0.002, right=0.998, top=0.995, bottom=0.005, wspace=0.08)
    fig.savefig(path)
    plt.close(fig)



def fig_kinds(path: Path) -> None:
    """One real ESA-ADB window per kind, against the channel's nominal band.

    All four are channel 41, so the grey band -- the 1st-99th percentile of
    nominal windows in that channel -- is one reference for the whole row. The
    picks are the clearest window of each kind; the local subsequence is drawn
    with a nominal window behind it because the two are meant to be
    indistinguishable at this scale.
    """
    real, kl, _ = _load_windows()
    anom = pd.read_csv(ROOT / "results/shell_s0/labeled_windows.csv")
    anom = anom[anom["kind"] == "anomaly"].reset_index(drop=True)

    from anogen.shell.realism import channel_span_normalize
    from anogen.shell.scaler import load_minmax

    sc = load_minmax(ROOT / "results/shell_s0/minmax_scaler.npz")
    gal = np.load(ROOT / "results/shell_s3/galleries.npz", allow_pickle=True)
    nom = channel_span_normalize(np.asarray(gal["cond"]), np.asarray(gal["channel_idx"]), sc)
    nom = nom[np.asarray(gal["channel_idx"]) == 0]
    band = np.percentile(nom, [1, 99])

    # (row index into `real`, label, what to look at). Checked against kl below.
    picks = [
        (654, "Point / Global", "one bin, far out"),
        (196, "level shift", "steps, then stays"),
        (42, "global subsequence", "the whole window is out"),
        (90, "local subsequence", "never leaves the band"),
    ]
    for (i, _, _), (key, _) in zip(picks, KINDS, strict=True):
        assert kl[i] == key, f"window {i} is {kl[i]}, not {key}"

    hours = np.linspace(0, 512 * 30 / 3600, real.shape[1])
    fig, ax = plt.subplots(1, 4, figsize=(5.51, 1.38), squeeze=False)
    for j, (i, label, hint) in enumerate(picks):
        a = ax[0, j]
        r = anom.loc[i]
        lo = max(0, int(r["label_lo"] - r["start"]))
        hi = min(len(hours), int(r["label_hi"] - r["start"]))
        a.axhspan(*band, color="#c3cdd7", alpha=0.55, lw=0, zorder=0)
        # Only when the label is a strict subset: elsewhere it would wash the
        # panel, the ESA span running days past the window on either side.
        if hi - lo < len(hours):
            a.axvspan(hours[lo], hours[hi - 1], color=REAL, alpha=0.09, lw=0, zorder=1)
        if label.startswith("local"):
            a.plot(hours, nom[7], lw=0.35, color=MUTE, alpha=0.8, zorder=2)
        a.plot(hours, real[i], lw=0.45, color=REAL, zorder=3)
        a.set_title(label, fontsize=6.5, color=INK, pad=2.5)
        a.text(0.5, -0.045, hint, fontsize=6, color=MUTE, ha="center", va="top",
               transform=a.transAxes)
        a.set_ylim(-0.12, 1.15)
        a.set_xlim(hours[0], hours[-1])
        a.set_xticks([])
        a.set_yticks([] if j else [0.1, 1.0])
        for s in a.spines.values():
            s.set_color("#c8d2dc")
            s.set_linewidth(0.4)
    ax[0, 0].set_yticklabels(["normal", "channel\nspan"], fontsize=5.5, color=MUTE)
    ax[0, 0].tick_params(length=0, pad=1)
    ax[0, 3].text(0.97, 0.90, "a normal window", fontsize=5.5, color=MUTE, ha="right",
                  va="top", transform=ax[0, 3].transAxes)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.865, bottom=0.185, wspace=0.08)
    fig.savefig(path)
    plt.close(fig)


def fig_shell(path: Path) -> None:
    """The shell region, drawn as a ring on a density surface.

    Schematic, not measured: the real encoder space is 32-dimensional and h is a
    log-sum-exp over reference windows, not a density. But the construction is
    the real one -- h = -log p, the 99th percentile Q of h over nominal data,
    and the band |h - Q| <= delta with delta = 0.25 IQR(h) -- so the green ring
    sits where the surface really has dropped to the 1% level, at its real width.
    """
    # 0.365 of a 14cm body = 5.11cm = 2.01in, so the figure is included at
    # scale 1 and the labels below keep the pt size they are set at.
    # Two components, slightly off-centre, so the hill is not a textbook bell.
    mix = [(0.70, (0.00, 0.00), (0.90, 0.75)),
           (0.30, (0.90, 0.50), (0.60, 0.50))]

    def dens(X, Y):
        out = np.zeros(np.shape(X), dtype=float)
        for w, (mx, my), (sx, sy) in mix:
            out += w * np.exp(-0.5 * (((X - mx) / sx) ** 2 + ((Y - my) / sy) ** 2)) / (
                2 * np.pi * sx * sy
            )
        return out

    rng = np.random.default_rng(5)
    k = rng.choice(len(mix), 400_000, p=[w for w, _, _ in mix])
    mu = np.array([m for _, m, _ in mix])
    sd = np.array([t for _, _, t in mix])
    sample = mu[k] + rng.standard_normal((400_000, 2)) * sd[k]
    h_at = -np.log(dens(sample[:, 0], sample[:, 1]))
    Q = float(np.quantile(h_at, 0.99))
    delta = 0.25 * float(np.subtract(*np.percentile(h_at, [75, 25])))
    lvl = float(np.exp(-Q))

    # The band is a closed ring, so it is cheaper and crisper to solve for the
    # radius where h hits each level along every ray than to trace a contour.
    th = np.linspace(0, 2 * np.pi, 193)
    rr = np.linspace(0.5, 8.0, 700)
    hray = -np.log(dens(rr[:, None] * np.cos(th), rr[:, None] * np.sin(th)))
    radius = lambda L: np.array([np.interp(L, hray[:, j], rr) for j in range(len(th))])
    r_in, r_mid, r_out = radius(Q - delta), radius(Q), radius(Q + delta)
    ring = lambda r: (r * np.cos(th), r * np.sin(th))

    lim = 3.6
    g = np.linspace(-lim, lim, 108)
    X, Y = np.meshgrid(g, g)
    Z = dens(X, Y)
    top = float(Z.max())

    fig = plt.figure(figsize=(2.01, 1.40))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, color="0.78", alpha=0.55, rstride=3, cstride=3,
                    linewidth=0.15, edgecolors="0.55", antialiased=True, shade=False)

    # The band, as a quad strip between the two level radii, laid on the plain.
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    xi, yi = ring(r_in)
    xo, yo = ring(r_out)
    quads = [
        [(xi[i], yi[i], lvl), (xo[i], yo[i], lvl),
         (xo[i + 1], yo[i + 1], lvl), (xi[i + 1], yi[i + 1], lvl)]
        for i in range(len(th) - 1)
    ]
    ax.add_collection3d(Poly3DCollection(quads, facecolor=GOOD, alpha=0.55, lw=0))
    for xe, ye in (ring(r_in), ring(r_out)):
        ax.plot(xe, ye, np.full_like(xe, lvl), color=GOOD, lw=0.45, zorder=5)
    xm, ym = ring(r_mid)
    ax.plot(xm, ym, np.full_like(xm, lvl), color=GOOD, lw=1.0, zorder=5)

    # The donor sits where normal telemetry is dense; the edit walks it down the
    # hill and stops in the band, which is the whole point of a band.
    j = int(np.argmin(np.abs(th - np.radians(340.0))))
    xg, yg = xm[j], ym[j]
    xd, yd = 0.30, 0.20
    t = np.linspace(0, 1, 120)
    px, py = xd + (xg - xd) * t, yd + (yg - yd) * t
    pz = np.maximum(dens(px, py), lvl)
    ax.plot(px, py, pz, color=INK, lw=0.9, ls="--", zorder=6)
    ax.scatter([xd], [yd], [dens(np.array(xd), np.array(yd))], s=22, color=ACCENT,
               depthshade=False, zorder=7, edgecolors="white", linewidths=0.5)
    ax.scatter([xg], [yg], [lvl], s=100, color=GOOD, marker="*", depthshade=False,
               zorder=8, edgecolors="white", linewidths=0.5)

    # A bare vertical axis at the screen-left corner, because the height is the
    # whole message. For elev/azim below, screen-x runs as 0.88x + 0.47y, which
    # puts (-lim, -lim) furthest left and (lim, lim) furthest right.
    ax.plot([-lim, -lim], [-lim, -lim], [0, top * 1.04], color=INK, lw=0.6)
    ax.text(-lim, -lim, top * 1.10, "how common", fontsize=6, color=INK,
            ha="left", va="bottom")
    ax.text(xd, yd, float(dens(np.array(xd), np.array(yd))) * 1.16, "donor",
            fontsize=6, color=ACCENT, ha="center", va="bottom", weight="bold")
    halo = dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85)
    ax.text(xg + 0.70, yg - 1.10, lvl, "generated", fontsize=6, color=GOOD,
            ha="left", va="top", weight="bold", bbox=halo)
    ax.text(-1.30, -5.05, lvl, "the shell", fontsize=6, color=GOOD,
            ha="center", va="center", weight="bold", bbox=halo)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, top * 1.05)
    ax.set_box_aspect((1, 1, 0.62), zoom=1.06)
    ax.view_init(elev=26, azim=-62)
    ax.set_axis_off()
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    fig.savefig(path)
    plt.close(fig)


def _fig_shell_3d_unused(path: Path) -> None:
    elev, azim, rad = 16.0, 72.0, 2.5
    rng = np.random.default_rng(1)
    cov = np.array([[1.00, 0.52, 0.10], [0.52, 0.38, 0.03], [0.10, 0.03, 0.13]])
    L = np.linalg.cholesky(cov)
    pts = rng.standard_normal((1100, 3)) @ L.T
    r = np.sqrt((np.linalg.solve(L, pts.T) ** 2).sum(0))

    fig = plt.figure(figsize=(3.05, 2.55))
    ax = fig.add_subplot(111, projection="3d")

    # Only the half of the shell facing away from the camera is drawn, so the
    # cloud inside stays visible and the shell reads as a skin around it. The
    # outward normal of {x = rad L s, |s| = 1} is parallel to L^-T s.
    e_, a_ = np.radians(elev), np.radians(azim)
    cam = np.array([np.cos(e_) * np.cos(a_), np.cos(e_) * np.sin(a_), np.sin(e_)])
    u, v = np.mgrid[0 : 2 * np.pi : 90j, 0 : np.pi : 45j]
    s = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])
    nrm = np.einsum("ij,jkl->ikl", np.linalg.inv(L).T, s)
    far = np.einsum("ikl,i->kl", nrm, cam) < 0
    e = rad * np.einsum("ij,jkl->ikl", L, s)
    ax.plot_surface(*np.where(far, e, np.nan), color=GOOD, alpha=0.10,
                    rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
    ax.plot_wireframe(*np.where(far, e, np.nan), rstride=6, cstride=5,
                      color=GOOD, lw=0.35, alpha=0.55)

    # the silhouette: normal . cam = 0, a great circle in s orthogonal to L^-1 cam
    m = np.linalg.solve(L, cam)
    m /= np.linalg.norm(m)
    e1 = np.cross(m, [0, 0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(m, e1)
    th = np.linspace(0, 2 * np.pi, 400)
    sil_xyz = rad * L @ (np.outer(e1, np.cos(th)) + np.outer(e2, np.sin(th)))
    ax.plot(*sil_xyz, color=GOOD, lw=1.5, zorder=4)

    ax.scatter(*pts[r < 1.45].T, s=3.0, color="0.45", alpha=0.5, lw=0,
               depthshade=False, zorder=6)
    donor = np.array([0.16, -0.20, 0.02])
    # On the silhouette, so it reads unambiguously as sitting on the shell, and
    # at the point of it that projects highest and furthest right on screen.
    rhat = np.cross([0, 0, 1.0], cam)
    rhat /= np.linalg.norm(rhat)
    uhat = np.cross(cam, rhat)
    gen = sil_xyz[:, np.argmax(uhat @ sil_xyz + 0.5 * (rhat @ sil_xyz))]
    ax.plot(*np.stack([donor, gen]).T, color=INK, lw=1.2, ls="--", zorder=9)
    ax.scatter(*donor, s=32, color=ACCENT, depthshade=False, zorder=10)
    ax.scatter(*gen, s=85, color=GOOD, marker="*", depthshade=False, zorder=10,
               edgecolors="white", linewidths=0.5)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-1.8, 1.8)
    ax.set_box_aspect((1, 1, 0.80), zoom=1.72)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.subplots_adjust(left=-0.02, right=1.02, top=1.02, bottom=-0.02)
    fig.savefig(path)
    plt.close(fig)


def fig_gallery(path: Path) -> None:
    real, kl, gen = _load_windows()
    hours = np.linspace(0, 512 * 30 / 3600, real.shape[1])
    n_r, n_g = 3, 5
    fig, ax = plt.subplots(len(KINDS), n_r + n_g + 1, figsize=(5.45, 2.35), squeeze=False,
                           gridspec_kw={"width_ratios": [1] * n_r + [0.22] + [1] * n_g})
    rng = np.random.default_rng(7)
    for i, (key, label) in enumerate(KINDS):
        r = real[kl == key]
        g = gen[key]
        g = g[rng.choice(len(g), n_g, replace=False)]
        ax[i, n_r].axis("off")
        for j in range(n_r + n_g + 1):
            if j == n_r:
                continue
            a = ax[i, j]
            src, col = (r, REAL) if j < n_r else (g, ACCENT)
            idx = j if j < n_r else j - n_r - 1
            if idx < len(src):
                a.plot(hours, src[idx], lw=0.35, color=col)
                for yv in (0.0, 1.0):
                    a.axhline(yv, lw=0.3, color="0.78", ls=":")
                a.set_ylim(-0.35, 1.35)
            else:
                a.axis("off")
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_color("#c8d2dc")
                s.set_linewidth(0.4)
        ax[i, 0].set_ylabel(label, fontsize=6, rotation=0, ha="right", va="center", labelpad=4)
    ax[0, 1].set_title("REAL", fontsize=7, color=REAL, weight="bold", pad=3)
    ax[0, n_r + 3].set_title("GENERATED", fontsize=7, color=ACCENT, weight="bold", pad=3)
    fig.tight_layout(pad=0.15, w_pad=0.25, h_pad=0.25)
    fig.subplots_adjust(left=0.142, right=0.995, top=0.90, bottom=0.015)
    fig.savefig(path)
    plt.close(fig)


def fig_sweep(path: Path) -> None:
    d = json.loads((ROOT / "results/shell_augdetect/sweep.json").read_text())
    c = d["curves"]
    ours = f"fixsweep:{MODEL}+real"
    ks = d["k"]
    fig, ax = plt.subplots(figsize=(2.866, 1.78))
    for name, lab, col, ls, lw in (
        ("real_only", "real anomalies only", "0.25", "--", 1.0),
        (ours, "+ generated", GOOD, "-", 1.4),
    ):
        if name not in c:
            continue
        ax.plot(ks, [p["event_recall_median"] for p in c[name]], marker="o", ms=2.6,
                color=col, ls=ls, lw=lw, label=lab, zorder=3)
        if name == ours:
            ax.fill_between(ks, [p["event_recall_min"] for p in c[name]],
                            [p["event_recall_max"] for p in c[name]],
                            color=col, alpha=0.13, lw=0, zorder=1)
    ax.annotate("", xy=(32, 0.636), xytext=(32, 0.364),
                arrowprops=dict(arrowstyle="<->", color=GOOD, lw=1.1))
    ax.text(32, 0.175, "+0.200", fontsize=7, color=GOOD, weight="bold", ha="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k  =  real anomalies available to the detector")
    ax.set_ylabel("event recall")
    ax.set_ylim(0.15, 0.80)
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(loc="upper left", framealpha=0.95, handlelength=1.6, borderpad=0.35,
              labelspacing=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2.5)
    fig.tight_layout(pad=0.2)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    fig_diffusion_strip(FIGS / "diffusion_strip.pdf")
    fig_kinds(FIGS / "kinds.pdf")
    fig_shell(FIGS / "shell.pdf")
    fig_gallery(FIGS / "gallery.pdf")
    fig_sweep(FIGS / "sweep.pdf")
    print(f"wrote {FIGS}/{{diffusion_strip,kinds,shell,gallery,sweep}}.pdf")


if __name__ == "__main__":
    main()
