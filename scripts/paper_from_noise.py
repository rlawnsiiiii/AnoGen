#!/usr/bin/env python3
"""GenFSDiff from noise: the frozen recipe started at t = T-1 instead of from a donor.

Two starts: pure N(0, I) (primary) and the per-channel matched marginal
N(sqrt(abar_T) m_c, (1 - abar_T + abar_T s_c^2) I) (``matched_noise``). Each has a
no-steer twin with the same start and seeds. Everything else is the frozen
``flip_x0_contrast_head_sharp_nomask`` spec.

Writes results/shell_fixsweep_noise. The donor-start galleries are copied in
(never linked, so a rewrite cannot reach the frozen files) so that one summary
scores both starts on the same EDI union and pairs every start with its twin.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from anogen.config import load_config  # noqa: E402
from anogen.phases.fixsweep import DEFAULT_VARIANTS, run_fixsweep  # noqa: E402

BASE = "flip_x0_contrast_head_sharp_nomask"
TWIN = "nosteer_flip_x0_recur2"
OUT = "results/shell_fixsweep_noise"
STARTS = {
    "noise": {"start_from_noise": True},
    "mnoise": {"start_from_noise": True, "matched_noise": True},
}


def variants() -> dict[str, dict]:
    base, twin = dict(DEFAULT_VARIANTS[BASE]), dict(DEFAULT_VARIANTS[TWIN])
    out = {BASE: base, TWIN: twin}
    for tag, extra in STARTS.items():
        out[f"{BASE}_{tag}"] = {**base, **extra, "twin": f"{TWIN}_{tag}"}
        out[f"{TWIN}_{tag}"] = {**twin, **extra}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default=str(ROOT / "configs/shell_mission1.yaml"))
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = ROOT / OUT
    out.mkdir(parents=True, exist_ok=True)
    for name in (BASE, TWIN):
        for f in range(3):
            src = ROOT / "results/shell_fixsweep" / f"{name}_fold{f}.npz"
            dst = out / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
    fx = cfg.setdefault("shell", {}).setdefault("fixsweep", {})
    fx.update(out=OUT, variants=variants(), only=None, n_boot=args.n_boot, baseline=BASE)
    run_fixsweep(cfg)


if __name__ == "__main__":
    main()
