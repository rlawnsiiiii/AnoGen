"""Choose σ_min before retraining S1: how much texture each channel can keep (numpy only).

    .venv/bin/python scripts/schedule_check.py -c configs/shell_mission1.yaml

Reads S0's train index and scaler, materializes up to --n windows per channel
in the *scaled* units the denoiser is trained in, and reports per channel:

* window mean and within-window sd (σ_data): the diffusion prior is N(0, I),
  so a mean far from 0 or a σ_data far below 1 changes what every noise level
  means (and the from-noise start, see REVIEW §2.8);
* the share of first-difference variance (texture) and of total variance a
  final Tweedie step at σ_min keeps, for the frozen σ_min = 0.01 and smaller
  values (shell/texture.py; exact for Gaussian windows, an upper bound for
  any sampler);
* the largest σ_min that keeps 95 % of the texture.

Writes results/shell_diagnose/schedule_check.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anogen.config import load_config, panel_path
from anogen.shell.data import load_panel
from anogen.shell.scaler import load_minmax
from anogen.shell.texture import sigma_min_for, texture_report, window_spectrum
from anogen.shell.windows import load_train_index, materialize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--n", type=int, default=1024, help="windows per channel")
    ap.add_argument("--sigmas", type=float, nargs="*", default=[0.01, 0.003, 0.001, 0.0003])
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["_repo_root"])
    s0 = root / cfg["s0_dir"]
    out = root / "results" / "shell_diagnose"
    out.mkdir(parents=True, exist_ok=True)
    splits = cfg.get("splits") or {}
    panel = load_panel(
        panel_path(cfg),
        channels=list(cfg.get("channels") or []),
        official_train_end=splits.get("official_train_end", "2007-01-01"),
        bin_seconds=int(cfg.get("bin_seconds", 30)),
    )
    tab = load_train_index(s0)
    scaler = load_minmax(s0 / "minmax_scaler.npz")
    w = int((cfg.get("shell") or {}).get("W", 512))
    rng = np.random.default_rng(0)
    report: dict[str, dict] = {}
    for c in sorted(tab["channel_idx"].unique()):
        sub = tab[tab["channel_idx"] == c]
        pick = rng.choice(len(sub), size=min(args.n, len(sub)), replace=False)
        sub = sub.iloc[pick].reset_index(drop=True)
        x = materialize(panel, sub, w)
        ch = sub["channel_idx"].to_numpy(dtype=np.int64)
        xs = scaler.transform(x, ch).astype(np.float64)
        r = texture_report(xs, list(args.sigmas))
        r["sigma_min_for_95pct_texture"] = sigma_min_for(window_spectrum(xs), width=w, keep=0.95)
        report[f"channel_idx_{int(c)}"] = r
        print(
            f"ch {int(c)}: mean {r['window_mean']:.3f}  σ_data {r['sigma_data_within_window']:.4f}  "
            f"texture sd/bin {r['texture_sd_per_bin']:.5f}  kept texture @σ_min=0.01: "
            f"{r['kept_texture_sigma_0.01']:.2f}  σ_min for 95 %: {r['sigma_min_for_95pct_texture']:.2g}"
        )
    (out / "schedule_check.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
