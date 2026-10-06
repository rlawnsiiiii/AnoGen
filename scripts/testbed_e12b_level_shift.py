"""E12b: does ARP in φ reward smoothing even when every query is a level shift?

E12 showed that galleries without anomaly content score well against mixed
queries. Here every query is a level shift (size 0.25 or 0.5, uniform onset),
and the gallery is the real nominal donors, unchanged or low-pass filtered
(Gaussian, sd 1 or 2 bins). If ARP in frozen φ rises when the donors are
smoothed, φ rewards texture loss, not level shifts. The same ARP without the
three high-frequency bands is the control.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from anogen.shell.coverage import arp, min_distances
from anogen.shell.features import embed_windows
from anogen.testbed.gauss import GPChannel, level_shift

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed")
OUT.mkdir(parents=True, exist_ok=True)
W = 512
ch = GPChannel()
rng = np.random.default_rng(0)


def level_shifts(n: int, r: np.random.Generator, size: float) -> np.ndarray:
    base = ch.sample(n, W, r)
    return level_shift(base, r.integers(40, 470, n), r.choice([-1, 1], n) * size)


donors = ch.sample(512, W, rng)
phi_donors = embed_windows(donors)
out = {}
for size in (0.25, 0.5):
    q = level_shifts(256, np.random.default_rng(1), size)
    pq = embed_windows(q)
    row = {"query_hf_minus_donor_hf": float(pq[:, 9:].mean() - phi_donors[:, 9:].mean())}
    for s in (0, 1, 2):
        g = gaussian_filter1d(donors, s, axis=1) if s else donors
        pg = embed_windows(g)
        row[f"arp_lowpass_{s}"] = float(arp(min_distances(pq, pg)))
        row[f"arp_noHF_lowpass_{s}"] = float(arp(min_distances(pq[:, :9], pg[:, :9])))
    out[f"level shifts only, size {size}"] = row
    print(size, {k: round(v, 3) for k, v in row.items()}, flush=True)
(OUT / "e12b_level_shift_queries.json").write_text(json.dumps(out, indent=1))
