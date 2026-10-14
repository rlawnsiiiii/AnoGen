#!/usr/bin/env python3
"""What the contrast objective is: the weight vector w, the scalar d, the edit.

Left column: w[t] for each contrast kind, at one sampled position.
Right column: a flat donor before and after one exact projection onto d = delta,
with the masked bins shaded -- the bins the edit is allowed to touch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from anogen.shell.contrast import contrast_mask, contrast_weights  # noqa: E402

W = 512
HOURS = W * 30.0 / 3600.0
CASES = [
    ("spike", 256, 0.83, "Point / Global: centre minus flanks"),
    ("step_head", 40, 0.80, "level shift: head mean minus tail mean"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/contrast_explainer.png")
    args = ap.parse_args()

    t = np.linspace(0, HOURS, W)
    fig, ax = plt.subplots(len(CASES) + 1, 2, figsize=(11, 2.5 * (len(CASES) + 1)), squeeze=False)
    for i, (kind, pos, delta, title) in enumerate(CASES):
        w = contrast_weights(kind, np.array([pos]), W)[0]
        m = contrast_mask(kind, np.array([pos]), W, dilate=8)[0]
        donor = np.full(W, 0.1) + 0.004 * np.sin(np.linspace(0, 40, W))

        a = ax[i][0]
        a.plot(t, w, lw=0.9, color="#1b4f9c")
        a.axhline(0, lw=0.5, color="0.6")
        a.set_title(f"w[t] for {kind!r}, position {pos}/{W}", fontsize=9)
        a.set_ylabel("weight", fontsize=8)
        a.text(0.98, 0.92, f"$\\sum_t w[t] = {w.sum():.0f}$", transform=a.transAxes,
               ha="right", va="top", fontsize=8)

        wd = w * m  # the projection may only move bins inside the mask
        x = donor - (donor @ w - delta) * wd / (wd * wd).sum()
        b = ax[i][1]
        b.fill_between(t, -0.3, 1.3, where=m, color="#ffe9c7", lw=0, label="generated (mask)")
        b.plot(t, donor, lw=0.9, color="0.45", label=f"donor, d = {donor @ w:+.3f}")
        b.plot(t, x, lw=0.9, color="#c0392b", label=f"after projection, d = {x @ w:+.3f}")
        b.set_ylim(-0.3, 1.3)
        b.set_title(f"{title}  (target $\\delta$ = {delta})", fontsize=9)
        b.legend(fontsize=7, loc="upper right", framealpha=0.9)
        for a_ in (a, b):
            a_.set_xlim(0, HOURS)
            a_.tick_params(labelsize=8)
            for s in ("top", "right"):
                a_.spines[s].set_visible(False)
    # the spike's 25-bin support is invisible at window scale, so zoom it
    p = 256
    w = contrast_weights("spike", np.array([p]), W)[0]
    m = contrast_mask("spike", np.array([p]), W, dilate=8)[0]
    lo, hi = p - 16, p + 17
    z = np.arange(lo, hi)
    gs = ax[0][0].get_gridspec()
    for a_ in ax[-1]:
        a_.remove()
    big = fig.add_subplot(gs[-1, :])
    big.bar(z - p, w[lo:hi], width=0.85,
            color=["#c0392b" if v > 0 else "#1b4f9c" if v < 0 else "0.85" for v in w[lo:hi]])
    big.fill_between([-16, 16], -0.1, 0.4, where=None, color="none")
    for z_ in z[m[lo:hi]]:
        big.axvspan(z_ - p - 0.5, z_ - p + 0.5, color="#ffe9c7", zorder=0)
    big.axhline(0, lw=0.6, color="0.4")
    big.set_xlim(-16.5, 16.5)
    big.set_xlabel("bin, relative to the sampled position (1 bin = 30 s)", fontsize=8)
    big.set_ylabel("weight", fontsize=8)
    big.tick_params(labelsize=8)
    big.set_title("'spike' bin by bin: 8 flank  |  3 gap  |  3 centre  |  3 gap  |  8 flank; "
                  "shaded = generated, flanks stay donor", fontsize=9)
    for s in ("top", "right"):
        big.spines[s].set_visible(False)

    fig.suptitle("The contrast objective: $d(x)=\\sum_t w[t]\\,x[t]$ driven to $\\delta$ "
                 "by the edit $\\lambda(d-\\delta)\\,w/\\|w\\|^2$", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    dest = Path(__file__).resolve().parents[1] / args.out
    fig.savefig(dest, dpi=150)
    print(dest)


if __name__ == "__main__":
    main()
