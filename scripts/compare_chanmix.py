"""Was the per-kind behaviour of c1 a kind effect or a channel effect?

Compares the frozen contiguous-allocation gallery against the channel-
stratified one on the realism diagnostics, per targeted kind. Reads only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from anogen.config import REPO_ROOT, load_config
from anogen.shell.morphology import KIND_ORDER
from anogen.shell.realism import (
    channel_span_normalize,
    diag_cusum,
    diag_envelope,
    diag_first_difference,
    diag_signed_peak,
)
from anogen.shell.scaler import load_minmax

GALLERIES = {
    "c1 (contiguous)": "results/shell_kindmix_score_hybrid/time_both_hybrid_needles_fold0.npz",
    "chanmix": "results/shell_chanmix/time_both_hybrid_needles_fold0.npz",
}
SHORT = {
    "real ESA Point / Global": "Point/Global",
    "real ESA local subsequence": "local subseq",
    "real level shift": "level shift",
    "real ESA global subsequence": "global subseq",
}


def _row(u: np.ndarray, x: np.ndarray, ch: np.ndarray, scaler) -> dict[str, float]:
    env = diag_envelope(x, ch, scaler)
    peak = diag_signed_peak(u, width=9)
    cusum = diag_cusum(u, min_segment=8)
    diff = diag_first_difference(u)
    return {
        "env frac": env["window_fraction"],
        "env max": env["max_excess"],
        "peak sev": float(np.mean(peak["severity"])),
        "peak@start": float(np.mean(np.asarray(peak["position"]) < 0.10)),
        "cusum": float(np.mean(cusum["contrast"])),
        "diff p99.9": float(np.mean(diff["p999"])),
    }


def main() -> None:
    cfg = load_config(Path(REPO_ROOT) / "configs/shell_mission1.yaml")
    root = Path(cfg.get("_repo_root", REPO_ROOT))
    scaler = load_minmax(root / "results/shell_s0/minmax_scaler.npz")

    rows = []
    for label, rel in GALLERIES.items():
        f = np.load(root / rel, allow_pickle=True)
        x = np.asarray(f["x"], dtype=np.float64)
        ch = np.asarray(f["channel_idx"], dtype=np.int64)
        alloc = np.asarray(f["kind_alloc"]).astype(str)
        u = channel_span_normalize(x, ch, scaler)
        for kind in KIND_ORDER:
            m = alloc == kind
            if not np.any(m):
                continue
            rows.append(
                {
                    "kind": SHORT.get(kind, kind),
                    "gallery": label,
                    "n_ch": int(len(np.unique(ch[m]))),
                    **_row(u[m], x[m], ch[m], scaler),
                }
            )
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("=== per targeted kind, fold 0 ===")
    print(df.set_index(["kind", "gallery"]).round(4).to_string())
    print()
    print("=== whole gallery ===")
    whole = []
    for label, rel in GALLERIES.items():
        f = np.load(root / rel, allow_pickle=True)
        x = np.asarray(f["x"], dtype=np.float64)
        ch = np.asarray(f["channel_idx"], dtype=np.int64)
        whole.append(
            {"gallery": label, **_row(channel_span_normalize(x, ch, scaler), x, ch, scaler)}
        )
    print(pd.DataFrame(whole).set_index("gallery").round(4).to_string())


if __name__ == "__main__":
    main()
