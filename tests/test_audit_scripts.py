"""Smoke-test the numpy-only audit scripts on fake artifacts in the frozen formats."""

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from anogen.shell.scaler import ChannelMinMax

ROOT = Path(__file__).resolve().parents[1]


def _fake(tmp: Path) -> Path:
    rng = np.random.default_rng(0)
    w, n_a, n_r, n_d = 64, 60, 20, 90
    s0, s3, s4 = (tmp / d for d in ("s0", "s3", "s4"))
    for d in (s0, s3, s4):
        d.mkdir()
    ev = np.repeat([f"id_{i}" for i in range(12)], 5)
    fold = np.repeat(np.arange(12) % 3, 5)
    ch_a = rng.integers(0, 6, n_a)
    x_a = 0.5 + 0.1 * rng.standard_normal((n_a, w))
    np.savez(
        s0 / "labeled_arrays.npz",
        anomaly=x_a, rare=0.5 + 0.1 * rng.standard_normal((n_r, w)),
        anomaly_fold=fold, rare_fold=rng.integers(0, 3, n_r),
        anomaly_channel=ch_a, rare_channel=rng.integers(0, 6, n_r),
    )
    pd.DataFrame({"kind": ["anomaly"] * n_a, "event_id": ev}).to_csv(s0 / "labeled_windows.csv", index=False)
    pd.DataFrame(
        [{"channel": f"channel_{41 + c}", "fold": f, "below_min": False} for c in range(6) for f in range(3)]
    ).to_csv(s0 / "channel_fold_counts.csv", index=False)
    ChannelMinMax(lo=np.zeros(6), hi=np.ones(6)).save(s0 / "minmax_scaler.npz")
    ch = np.repeat(np.arange(6), n_d // 6)
    cond = 0.5 + 0.1 * rng.standard_normal((n_d, w))
    np.savez(s3 / "galleries.npz", cond=cond, genias=cond + 1e-3, posthoc=cond + 0.05, channel_idx=ch)
    np.savez(s4 / "shell_gallery.npz", x=cond * 0.9 + 0.05, unguided=cond[::-1], channel_idx=ch)
    (s4 / "summary.json").write_text(json.dumps({"tau": 1.0}))
    cfg = tmp / "cfg.yaml"
    cfg.write_text(f"s0_dir: {s0}\ns3_dir: {s3}\ns4_dir: {s4}\ns1_dir: {tmp / 's1'}\n")
    return cfg


def test_audit_metrics_runs(tmp_path, monkeypatch, capsys):
    cfg = _fake(tmp_path)
    monkeypatch.setattr(sys, "argv", ["audit_metrics.py", "-c", str(cfg), "--n-boot", "20", "--out", str(tmp_path / "audit")])
    runpy.run_path(str(ROOT / "scripts/audit_metrics.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "donor (real nominal)" in out and "unguided nu=1" in out
    blob = json.loads((tmp_path / "audit/audit.json").read_text())
    row = blob["rows"]["post-hoc"][0]
    assert 0.0 < row["arp"] <= 1.0 and row["arp_ci"][0] <= row["arp"] <= row["arp_ci"][1]
    assert "prdc_precision" in row and "coverage" in row


def test_diagnose_causality_numpy_part(tmp_path, monkeypatch, capsys):
    cfg = _fake(tmp_path)
    monkeypatch.setattr(sys, "argv", ["diagnose_causality.py", "-c", str(cfg)])
    runpy.run_path(str(ROOT / "scripts/diagnose_causality.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "real anomalies" in out and "unguided nu=1" in out
