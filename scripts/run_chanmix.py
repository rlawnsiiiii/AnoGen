"""Regenerate hybrid_needles with channel-stratified kind allocation.

Same recipe and hyper-parameters as c1; the only change is that each targeted
kind draws donors from all six channels instead of a contiguous block. Writes
results/shell_chanmix/ and leaves the frozen shell_kindmix_score_hybrid,
shell_s3 and shell_s4 untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

from anogen.config import REPO_ROOT, load_config
from anogen.phases.kindmixscore import run_kindmixscore

OUT = "results/shell_chanmix"


def main() -> None:
    root = Path(REPO_ROOT)
    cfg = load_config(root / "configs/shell_mission1.yaml")
    frozen = root / "results/shell_kindmix_score_hybrid"

    cfg["kindmixscore_recipes"] = ("hybrid_needles",)
    cfg["kindmix_score_hybrid_dir"] = str(root / OUT)
    shell = dict(cfg.get("shell") or {})
    shell["kindmixscore"] = {**dict(shell.get("kindmixscore") or {}), "channel_stratified": True}
    cfg["shell"] = shell

    out = root / OUT
    if out.resolve() == frozen.resolve():
        raise RuntimeError("refusing to overwrite the frozen hybrid_needles gallery")
    report = run_kindmixscore(cfg)
    print(json.dumps({k: v for k, v in report.items() if k != "methods"}, indent=2)[:1200])


if __name__ == "__main__":
    main()
