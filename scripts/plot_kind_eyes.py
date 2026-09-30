"""Random 16 windows per ESA kind (no nearest, no amplitude ranking).

Writes docs/kind_eyes/. Does not touch paper/figures or shell_s3/s4.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from anogen.config import REPO_ROOT, load_config
from anogen.shell.events import types_from_cfg
from anogen.shell.morphology import KIND_ORDER, KIND_SLUG, kinds_for_windows
from anogen.shell.plots import save_compare_rows, save_strip

N = 16
N_OVERVIEW = 8
SEED = 0
COLORS = {
    "real ESA Point / Global": "#2166ac",
    "real ESA local subsequence": "#1b9e77",
    "real level shift": "#d95f02",
    "real ESA global subsequence": "#7570b3",
    "real ESA Point / Local": "#e7298a",
}
SHORT = {
    "real ESA Point / Global": "Point/Global",
    "real ESA local subsequence": "local subseq",
    "real level shift": "level shift",
    "real ESA global subsequence": "global subseq",
    "real ESA Point / Local": "Point/Local",
}


def _sample(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) == 0:
        return x
    k = min(n, len(x))
    idx = rng.choice(len(x), size=k, replace=False)
    return x[idx]


def _dump_family(
    out: Path,
    prefix: str,
    title: str,
    gallery: np.ndarray,
    alloc: np.ndarray,
    bin_seconds: int,
    rng: np.random.Generator,
) -> list[str]:
    written: list[str] = []
    overview: dict[str, np.ndarray] = {}
    for kind in KIND_ORDER:
        sl = gallery[alloc == kind]
        if len(sl) == 0:
            continue
        samp = _sample(sl, N, rng)
        slug = KIND_SLUG.get(kind, "kind")
        png = f"{prefix}_random16_{slug}.png"
        save_strip(
            out / png,
            samp,
            title=f"{prefix}: {SHORT.get(kind, kind)} random {len(samp)}/{len(sl)}",
            color=COLORS.get(kind, "#4d4d4d"),
            bin_seconds=bin_seconds,
            idx=np.arange(len(samp)),
        )
        written.append(png)
        overview[SHORT.get(kind, kind)] = _sample(sl, N_OVERVIEW, rng)
    if overview:
        ov = f"{prefix}_overview8.png"
        save_compare_rows(
            out / ov,
            overview,
            bin_seconds=bin_seconds,
            n_cols=N_OVERVIEW,
            colors={SHORT.get(k, k): COLORS.get(k, "#4d4d4d") for k in KIND_ORDER},
            title=f"{title} · random {N_OVERVIEW} per kind (no nearest, no amp rank)",
        )
        written.append(ov)
    return written


def main() -> None:
    cfg = load_config(REPO_ROOT / "configs/shell_mission1.yaml")
    root = Path(cfg["_repo_root"])
    out = root / "docs" / "kind_eyes"
    out.mkdir(parents=True, exist_ok=True)
    bin_seconds = int(cfg.get("bin_seconds", 30))
    rng = np.random.default_rng(SEED)

    s0 = root / "results" / "shell_s0"
    lab = np.load(s0 / "labeled_arrays.npz")
    meta = __import__("pandas").read_csv(s0 / "labeled_windows.csv")
    x_a = np.asarray(lab["anomaly"])
    anom = meta[meta["kind"] == "anomaly"].reset_index(drop=True)
    kinds = kinds_for_windows(x_a, anom, types_from_cfg(cfg))
    written = _dump_family(out, "real", "real ESA-ADB", x_a, kinds, bin_seconds, rng)

    jobs = [
        (
            "c1_fold0",
            "C1 time_both hybrid_needles fold0 (1536)",
            root / "results/shell_kindmix_score_hybrid/time_both_hybrid_needles_fold0.npz",
        ),
        (
            "c2_fold0",
            "C2 time_recon hybrid_needles fold0 (1536)",
            root / "results/shell_kindmix_score_hybrid/time_recon_hybrid_needles_fold0.npz",
        ),
        (
            "q00_time_both",
            "q00 time_both λ_anom=0 quiet (128)",
            root / "results/shell_quiettune/time_both_q00.npz",
        ),
        (
            "q00_time_recon",
            "q00 time_recon λ_anom=0 quiet (128)",
            root / "results/shell_quiettune/time_recon_q00.npz",
        ),
    ]
    for prefix, title, path in jobs:
        z = np.load(path, allow_pickle=True)
        written += _dump_family(
            out, prefix, title, np.asarray(z["x"]), np.asarray(z["kind_alloc"]), bin_seconds, rng
        )

    (out / "INDEX.txt").write_text(
        "Random windows per kind. Not nearest-in-φ, not amplitude-ranked.\n"
        "C1/C2 = locked hybrid_needles 1536 fold-0. q00 = 128-donor band on quiet kinds.\n"
        "seed=0, 16 per kind (4×4) plus 8-col overview.\n\n"
        + "\n".join(written)
        + "\n"
    )
    print("\n".join(written))


if __name__ == "__main__":
    main()
