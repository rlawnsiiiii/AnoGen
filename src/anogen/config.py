"""YAML config load."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path is not None else REPO_ROOT / "configs" / "shell_mission1.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {cfg_path} is not a mapping")
    cfg["_config_path"] = str(cfg_path)
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def output_dir(cfg: dict[str, Any]) -> Path:
    out = Path(cfg.get("output_dir", "results"))
    if not out.is_absolute():
        out = Path(cfg["_repo_root"]) / out
    out.mkdir(parents=True, exist_ok=True)
    return out


def causaldiscovery_root(cfg: dict[str, Any]) -> Path:
    root = Path(cfg.get("causaldiscovery_root", "/mnt/extras/SSD/AI/CausalDiscovery"))
    if not root.is_absolute():
        root = Path(cfg["_repo_root"]) / root
    return root


def data_root(cfg: dict[str, Any]) -> Path:
    """ESA mission folder (labels.csv). Relative paths are under CausalDiscovery."""
    root = Path(cfg.get("data_root", "data/ESA-Mission1"))
    if not root.is_absolute():
        root = causaldiscovery_root(cfg) / root
    if (root / "labels.csv").exists():
        return root
    for name in ("ESA-Mission1", "ESA-Mission2"):
        nested = root / name
        if (nested / "labels.csv").exists():
            return nested
    return root


def panel_path(cfg: dict[str, Any]) -> Path:
    path = Path(cfg.get("panel_npz", "results/mission1/panel_light.npz"))
    if not path.is_absolute():
        path = causaldiscovery_root(cfg) / path
    return path
