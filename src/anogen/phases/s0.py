"""S0: window tables, event OOF, sealed guard, frozen protocol."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from anogen.config import data_root, output_dir, panel_path
from anogen.shell.data import SealedTelemetryAccessError, load_metadata, load_panel
from anogen.shell.events import assign_splits, build_event_table, panel_pairs
from anogen.shell.features import FEATURE_DIM, FEATURE_NAME
from anogen.shell.folds import assign_event_folds, channel_fold_counts
from anogen.shell.protocol import default_protocol
from anogen.shell.scaler import fit_channel_minmax
from anogen.shell.windows import (
    compose_train_index,
    extract_labeled_windows,
    materialize,
    occupancy_mask,
    sample_nominal,
)


def run_s0(cfg: dict[str, Any]) -> dict[str, Any]:
    proto = default_protocol(cfg)
    out = output_dir(cfg)
    (out / "protocol.json").write_text(json.dumps(proto, indent=2) + "\n")

    allow = bool(cfg.get("allow_test_telemetry", False))
    if allow and str(cfg.get("phase", "s0")) != "s6":
        raise SealedTelemetryAccessError(
            "allow_test_telemetry is only legal for the one-shot S6 sealed run"
        )

    channels = list(
        cfg.get("channels")
        or [
            "channel_41",
            "channel_42",
            "channel_43",
            "channel_44",
            "channel_45",
            "channel_46",
        ]
    )
    splits = cfg.get("splits") or {}
    path_end = splits.get("path_train_end", "2006-10-01")
    official_end = splits.get("official_train_end", "2007-01-01")
    width = int(proto["W"])
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    labels_dir = data_root(cfg)
    npz = panel_path(cfg)
    if not (labels_dir / "labels.csv").exists():
        report = {
            "ok": False,
            "skipped": True,
            "reason": f"labels.csv not found under {labels_dir}",
        }
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    if not npz.is_file():
        report = {
            "ok": False,
            "skipped": True,
            "reason": f"panel not found: {npz}",
        }
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    meta = load_metadata(labels_dir)
    events = assign_splits(
        build_event_table(meta["labels"], meta["anomaly_types"]),
        path_train_end=path_end,
        official_train_end=official_end,
    )
    panel = load_panel(
        npz,
        channels=channels,
        official_train_end=official_end,
        allow_test_telemetry=allow,
        bin_seconds=int(cfg.get("bin_seconds", 30)),
    )
    pairs = panel_pairs(events, channels)
    labeled = extract_labeled_windows(
        panel,
        pairs,
        width=width,
        max_crops_per_event=int(proto["max_crops_per_event"]),
        allow_test_telemetry=allow,
    )
    sealed_skipped = int(labeled.attrs.get("sealed_skipped", 0))
    labeled = assign_event_folds(
        labeled,
        n_folds=int(proto["n_folds"]),
        seed=int(cfg.get("seed", 0)),
    )
    include_rares = bool(proto.get("train_includes_rares", True))
    occ = occupancy_mask(
        panel,
        events,
        guard_bins=int(proto["guard_bins"]),
        occupy_rares=not include_rares,
    )
    nominal = sample_nominal(
        panel,
        occ,
        width=width,
        n_per_channel=int(proto["n_nominal_per_channel"]),
        rng=rng,
    )

    anomaly = labeled[labeled["is_anomaly"]].reset_index(drop=True) if len(labeled) else labeled
    rare = labeled[labeled["is_rare"]].reset_index(drop=True) if len(labeled) else labeled
    fold_tab = channel_fold_counts(
        labeled, min_anomaly_windows=int(proto["min_anomaly_windows"])
    )
    channel_tab = _channel_totals(labeled, nominal, channels)

    x_a = materialize(panel, anomaly, width)
    x_r = materialize(panel, rare, width)
    _write_table(out / "events.csv", _events_for_csv(events))
    _write_table(out / "labeled_windows.csv", labeled)
    train = compose_train_index(
        nominal,
        rare,
        upsample=int(proto.get("rare_upsample", 1)) if include_rares else 1,
    )
    if not include_rares:
        train = nominal
    _write_table(out / "nominal_index.csv", nominal)
    _write_table(out / "train_index.csv", train)
    _write_table(out / "channel_fold_counts.csv", fold_tab)
    scfg = dict((cfg.get("shell") or {}).get("scaler") or {})
    kind = str(scfg.get("kind", "minmax")).lower()
    scaler_summary: dict[str, Any] = {"kind": "none"}
    if kind == "minmax":
        x_train = materialize(panel, train, width)
        ch_train = (
            train["channel_idx"].to_numpy(dtype=np.int64)
            if len(train)
            else np.zeros(0, dtype=np.int64)
        )
        fr = tuple(scfg.get("feature_range") or (0.0, 1.0))
        scaler = fit_channel_minmax(
            x_train, ch_train, panel.k, feature_range=(float(fr[0]), float(fr[1]))
        )
        scaler.save(out / "minmax_scaler.npz")
        scaler_summary = scaler.summary()
    np.savez_compressed(
        out / "labeled_arrays.npz",
        anomaly=x_a,
        rare=x_r,
        anomaly_start=_col(anomaly, "start"),
        anomaly_channel=_col(anomaly, "channel_idx"),
        anomaly_fold=_col(anomaly, "fold"),
        rare_start=_col(rare, "start"),
        rare_channel=_col(rare, "channel_idx"),
        rare_fold=_col(rare, "fold"),
    )
    if len(nominal):
        np.savez_compressed(
            out / "nominal_index.npz",
            start=nominal["start"].to_numpy(dtype=np.int64),
            channel_idx=nominal["channel_idx"].to_numpy(dtype=np.int64),
        )

    n_anom_win = int(len(anomaly))
    n_anom_ev = int(anomaly["event_id"].nunique()) if n_anom_win else 0
    viable = (
        n_anom_ev >= int(proto["n_folds"])
        and n_anom_win >= int(proto["min_anomaly_windows"]) * int(proto["n_folds"])
    )
    report = {
        "ok": True,
        "skipped": False,
        "viable": viable,
        "protocol_version": proto["version"],
        "W": width,
        "eval_embedding": FEATURE_NAME,
        "eval_dim": FEATURE_DIM,
        "shell_encoder": proto["shell_encoder"],
        "panel_T": panel.T,
        "panel_k": panel.k,
        "n_events": int(len(events)),
        "n_dev_events": int(events["in_pretest_pool"].sum()),
        "n_test_events": int(events["in_test_pool"].sum()),
        "n_panel_pairs": int(len(pairs)),
        "n_anomaly_windows": n_anom_win,
        "n_anomaly_events_windowed": n_anom_ev,
        "n_rare_windows": int(len(rare)),
        "n_nominal_windows": int(len(nominal)),
        "n_train_windows": int(len(train)),
        "scaler": scaler_summary,
        "train_includes_rares": include_rares,
        "rare_upsample": int(proto.get("rare_upsample", 1)),
        "sealed_skipped": sealed_skipped,
        "n_fold_channel_below_min": int(fold_tab["below_min"].sum()) if len(fold_tab) else 0,
        "channel_counts": channel_tab,
        "win_rule": proto["win"],
        "tau_rule": proto["tau"],
        "note": (
            "Train pool is everyday nominals plus Rare Events (valid ops). "
            "Anomalies stay out of S1/S2/S3. True anomalies select "
            "{q, delta, lambda, nu} via OOF coverage. Do not retune after S4."
        ),
    }
    if not viable:
        report["hint"] = (
            f"Too few development anomaly windows ({n_anom_win} from "
            f"{n_anom_ev} events) for W={width}. Shorten W (try 256) before "
            "adding channels or a multivariate model."
        )
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    if df is None or len(df) == 0 or name not in df.columns:
        return np.zeros(0, dtype=np.int64)
    return df[name].to_numpy(dtype=np.int64)


def _events_for_csv(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["channels"] = out["channels"].map(lambda xs: "|".join(str(x) for x in xs))
    return out


def _write_table(path, df: pd.DataFrame) -> None:
    if df is None or len(df) == 0:
        pd.DataFrame().to_csv(path, index=False)
        return
    df.to_csv(path, index=False)


def _channel_totals(
    labeled: pd.DataFrame, nominal: pd.DataFrame, channels: list[str]
) -> list[dict[str, Any]]:
    rows = []
    for ch in channels:
        lab = labeled[labeled["channel"] == ch] if len(labeled) else labeled
        nom = nominal[nominal["channel"] == ch] if len(nominal) else nominal
        n_a = int(lab["is_anomaly"].sum()) if len(lab) else 0
        n_r = int(lab["is_rare"].sum()) if len(lab) else 0
        rows.append(
            {
                "channel": ch,
                "n_anomaly_windows": n_a,
                "n_anomaly_events": int(lab.loc[lab["is_anomaly"], "event_id"].nunique())
                if n_a
                else 0,
                "n_rare_windows": n_r,
                "n_rare_events": int(lab.loc[lab["is_rare"], "event_id"].nunique()) if n_r else 0,
                "n_nominal_sampled": int(len(nom)),
                "below_min_total": n_a < 5,
            }
        )
    return rows
