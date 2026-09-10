"""Frozen S0 protocol. Later phases read this file; they do not edit it."""

from __future__ import annotations

from typing import Any

import numpy as np

PROTOCOL_VERSION = "shell-s0-v1"

# Encoder training objective for S2. Locked here so S2 cannot quietly switch.
SHELL_ENCODER = "reconstruction"


def default_protocol(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    shell = dict((cfg or {}).get("shell") or {})
    win = dict(shell.get("win") or {})
    tau = dict(shell.get("tau") or {})
    return {
        "version": PROTOCOL_VERSION,
        "W": int(shell.get("W", 512)),
        "N_generate": int(shell.get("N_generate", 256)),
        "n_folds": int(shell.get("n_folds", 3)),
        "n_nominal_per_channel": int(shell.get("n_nominal_per_channel", 4096)),
        "max_crops_per_event": int(shell.get("max_crops_per_event", 8)),
        "guard_bins": int(shell.get("guard_bins", 16)),
        "min_anomaly_windows": int(shell.get("min_anomaly_windows", 5)),
        "min_fs_windows": int(shell.get("min_fs_windows", 8)),
        "train_includes_rares": bool(shell.get("train_includes_rares", True)),
        "rare_upsample": int(shell.get("rare_upsample", 1)),
        "eval_embedding": "feature_pack_v1",
        "shell_encoder": SHELL_ENCODER,
        "labels": {
            "nominal": (
                "Everyday windows: train pool, Q_q sample, and edit donors."
            ),
            "anomaly": (
                "Never train the ZS denoiser or encoder. Never set Q_q from them. "
                "Query set for Coverage@tau. Fold-0 anomalies choose tau. "
                "May select {q, delta, lambda, nu} via OOF, then freeze. "
                "S5 train-fold only: adapter positives and classifier label 1."
            ),
            "rare": (
                "Valid ops. Included in the S1/S2/S3 train pool (no upsample yet). "
                "Anomaly spans stay out of everyday sampling. "
                "Collision set for Coverage@tau. S5 train-fold: classifier hard negatives. "
                "Held-out rares stay the collision set when folds are used."
            ),
        },
        "tau": {
            "rule": "unguided_coverage_target",
            "unguided_target": float(tau.get("unguided_target", 0.10)),
            "note": (
                "On a calibration fold, set tau to the unguided_target quantile "
                "of min embedding distances from real anomaly windows to the "
                "unguided-diffusion gallery. Freeze that tau for every method."
            ),
        },
        "win": {
            "occupancy_min": float(win.get("occupancy_min", 0.8)),
            "diversity_ratio_min": float(win.get("diversity_ratio_min", 0.5)),
            "must_beat": list(win.get("must_beat") or ["genias", "posthoc"]),
            "note": (
                "ZS shell wins only if its gap beats both GenIAS and post-hoc, "
                "occupancy >= occupancy_min, and diversity >= diversity_ratio_min "
                "times the unguided diversity. Tie or loss is reported as-is."
            ),
        },
    }


def choose_tau(unguided_min_dists: np.ndarray, target: float = 0.10) -> float:
    """Tau such that unguided coverage equals ``target`` on the calibration set."""
    d = np.asarray(unguided_min_dists, dtype=np.float64)
    if d.size == 0:
        raise ValueError("cannot choose tau from empty distances")
    return float(np.quantile(d, target))


def win_rule(
    *,
    shell_gap: float,
    baseline_gaps: dict[str, float],
    occupancy: float,
    diversity: float,
    unguided_diversity: float,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Apply the pre-committed win rule. Does not retune anything."""
    win = protocol["win"]
    must_beat = list(win["must_beat"])
    beats = {name: float(shell_gap) > float(baseline_gaps[name]) for name in must_beat}
    occ_ok = float(occupancy) >= float(win["occupancy_min"])
    if unguided_diversity <= 0:
        div_ok = False
        div_ratio = 0.0
    else:
        div_ratio = float(diversity) / float(unguided_diversity)
        div_ok = div_ratio >= float(win["diversity_ratio_min"])
    won = occ_ok and div_ok and all(beats.values())
    return {
        "won": won,
        "beats": beats,
        "occupancy_ok": occ_ok,
        "diversity_ok": div_ok,
        "diversity_ratio": div_ratio,
        "shell_gap": float(shell_gap),
    }
