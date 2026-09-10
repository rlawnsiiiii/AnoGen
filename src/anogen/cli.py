"""AnoGen CLI: steered diffusion and GenIAS."""

from __future__ import annotations

import argparse
import json
import sys

from anogen.config import load_config

_PHASES = (
    "s0",
    "s1",
    "s1recon",
    "s2",
    "s3",
    "s4",
    "tune",
    "enc",
    "encscore",
    "s5",
    "s6",
    "plots",
    "xfer",
    "hanom",
    "kindproto",
    "kindmix",
    "kindmixscore",
    "kindmixhtune",
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="anogen",
        description="Shell-steering diffusion and GenIAS anomaly generation",
    )
    parser.add_argument("-c", "--config", required=True, help="YAML config")
    parser.add_argument(
        "phase",
        choices=_PHASES,
        help="s0–s6, tune, enc, encscore, plots, xfer, hanom, kindproto, kindmix, kindmixscore, kindmixhtune",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    cfg["phase"] = args.phase

    report = _dispatch(args.phase, cfg)
    print(json.dumps(report, indent=2))
    if not report.get("ok", False) and not report.get("skipped"):
        sys.exit(1)


def _dispatch(phase: str, cfg: dict) -> dict:
    if phase == "s0":
        from anogen.phases.s0 import run_s0

        return run_s0(cfg)
    if phase == "s1":
        from anogen.phases.s1 import run_s1

        return run_s1(cfg)
    if phase == "s1recon":
        from anogen.phases.s1_recon import run_s1_recon

        return run_s1_recon(cfg)
    if phase == "s2":
        from anogen.phases.s2 import run_s2

        return run_s2(cfg)
    if phase == "s3":
        from anogen.phases.s3 import run_s3

        return run_s3(cfg)
    if phase == "s4":
        from anogen.phases.s4 import run_s4

        return run_s4(cfg)
    if phase == "tune":
        from anogen.phases.tune import run_tune

        return run_tune(cfg)
    if phase == "enc":
        from anogen.phases.enc import run_enc

        return run_enc(cfg)
    if phase == "encscore":
        from anogen.phases.encscore import run_encscore

        return run_encscore(cfg)
    if phase == "s5":
        from anogen.phases.s5 import run_s5

        return run_s5(cfg)
    if phase == "s6":
        return {
            "ok": False,
            "skipped": True,
            "reason": "S6 is the one-shot sealed test. Do not run until S4/S5 freeze.",
        }
    if phase == "plots":
        from anogen.phases.plots import run_plots

        return run_plots(cfg)
    if phase == "xfer":
        from anogen.phases.xfer import run_xfer

        return run_xfer(cfg)
    if phase == "hanom":
        from anogen.phases.hanom import run_hanom

        return run_hanom(cfg)
    if phase == "kindproto":
        from anogen.phases.kindproto import run_kindproto

        return run_kindproto(cfg)
    if phase == "kindmix":
        from anogen.phases.kindmix import run_kindmix

        return run_kindmix(cfg)
    if phase == "kindmixscore":
        from anogen.phases.kindmixscore import run_kindmixscore

        return run_kindmixscore(cfg)
    if phase == "kindmixhtune":
        from anogen.phases.kindmixhtune import run_kindmix_htune

        return run_kindmix_htune(cfg)
    print(f"{phase} is not implemented yet.", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
