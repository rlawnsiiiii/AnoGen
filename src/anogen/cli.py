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
    "kindmixscorehybrid",
    "kindmixhtune",
    "noisescore",
    "ablatelambda",
    "quiettune",
    "genbase",
    "augdetect",
    "editor",
    "timeleash",
    "hashfix",
    "realism",
    "fixsweep",
    "geniasfair",
    "s1bidir",
    "diffdetect",
    "s1v2",
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
        help="s0–s6, tune, enc, encscore, plots, xfer, hanom, kindproto, kindmix, kindmixscore, kindmixscorehybrid, kindmixhtune, noisescore, ablatelambda, quiettune, genbase, augdetect, editor, timeleash, hashfix, realism, fixsweep, geniasfair, s1bidir, s1v2, diffdetect",
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
    if phase == "kindmixscorehybrid":
        from anogen.phases.kindmixscore import run_kindmixscore_hybrid

        return run_kindmixscore_hybrid(cfg)
    if phase == "kindmixhtune":
        from anogen.phases.kindmixhtune import run_kindmix_htune

        return run_kindmix_htune(cfg)
    if phase == "noisescore":
        from anogen.phases.noisescore import run_noisescore

        return run_noisescore(cfg)
    if phase == "ablatelambda":
        from anogen.phases.ablate_lambda import run_ablate_lambda

        return run_ablate_lambda(cfg)
    if phase == "quiettune":
        from anogen.phases.quiettune import run_quiettune

        return run_quiettune(cfg)
    if phase == "genbase":
        from anogen.phases.genbase import run_genbase

        return run_genbase(cfg)
    if phase == "augdetect":
        from anogen.phases.augdetect import run_augdetect

        return run_augdetect(cfg)
    if phase == "editor":
        from anogen.phases.editor import run_editor

        return run_editor(cfg)
    if phase == "timeleash":
        from anogen.phases.timeleash import run_timeleash

        return run_timeleash(cfg)
    if phase == "hashfix":
        from anogen.phases.hashfix import run_hashfix

        return run_hashfix(cfg)
    if phase == "realism":
        from anogen.phases.realism import run_realism

        return run_realism(cfg)
    if phase == "fixsweep":
        from anogen.phases.fixsweep import run_fixsweep

        return run_fixsweep(cfg)
    if phase == "geniasfair":
        from anogen.phases.geniasfair import run_geniasfair

        return run_geniasfair(cfg)
    if phase == "diffdetect":
        from anogen.phases.diffdetect import run_diffdetect

        return run_diffdetect(cfg)
    if phase == "s1v2":
        from anogen.phases.s1 import run_s1

        # Bidirectional S4-D *and* the log-spaced schedule (docs/TESTBED.md E10),
        # into its own directory. Frozen S1 untouched.
        cfg = dict(cfg)
        shell = dict(cfg.get("shell") or {})
        diff = dict(shell.get("diffusion") or {})
        diff.update(bidirectional=True, schedule="geometric")
        diff.update(dict(shell.get("diffusion_v2") or {}))
        shell["diffusion"] = diff
        cfg["shell"] = shell
        cfg["s1_dir"] = cfg.get("s1_v2_dir", "results/shell_s1_v2")
        cfg["_keep_s0_scaler"] = True
        return run_s1(cfg)
    if phase == "s1bidir":
        from anogen.phases.s1 import run_s1

        # Same S1 recipe with a bidirectional S4-D, into its own directory so the
        # frozen causal checkpoint is never overwritten.
        cfg = dict(cfg)
        shell = dict(cfg.get("shell") or {})
        diff = dict(shell.get("diffusion") or {})
        diff["bidirectional"] = True
        shell["diffusion"] = diff
        cfg["shell"] = shell
        cfg["s1_dir"] = cfg.get("s1_bidir_dir", "results/shell_s1_bidir")
        cfg["_keep_s0_scaler"] = True  # never rewrite the frozen S0 scaler file
        return run_s1(cfg)
    print(f"{phase} is not implemented yet.", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
