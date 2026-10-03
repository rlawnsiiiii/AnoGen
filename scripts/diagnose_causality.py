"""Five-minute check of the position-bias diagnosis on the real artifacts.

    .venv/bin/python scripts/diagnose_causality.py -c configs/shell_mission1.yaml

Writes results/shell_diagnose/causality.json and prints a short report.

1. Causality probe (torch): perturb x_t at bin s and see whether ε_θ changes at
   any bin < s. A causal backbone gives exactly zero there.
2. ε-MSE by position (torch): validation windows, several t. A causal backbone
   is worst at the start of the window (least context), best at the end.
3. Peak position on saved galleries (numpy only): share of dominant peaks in
   the first and last 10 % of the window. The causal diagnosis predicts
   start >> 0.10 and end << 0.10 for every DDIM gallery, including unguided;
   real anomalies and donors should sit near 0.10 on both sides.

Prediction from docs/TESTBED.md (exact Gaussian denoisers): causal unguided
start ≈ 0.4–0.6 and end ≈ 0.0–0.02; bidirectional ≈ 0.10 / 0.10.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anogen.config import load_config
from anogen.shell.evaluation import position_report
from anogen.shell.realism import channel_span_normalize
from anogen.shell.scaler import load_minmax


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--n-val", type=int, default=512)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["_repo_root"])
    s0, s1, s3, s4 = (root / cfg.get(k, f"results/shell_{k[:2]}") for k in ("s0_dir", "s1_dir", "s3_dir", "s4_dir"))
    out = root / "results" / "shell_diagnose"
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    try:
        import torch

        from anogen.shell.diffusion import DiffusionSchedule, denoiser_from_ckpt, q_sample
        from anogen.shell.scaler import resolve_scaler

        den = torch.load(s1 / "denoiser.pt", map_location="cpu", weights_only=False)
        model = denoiser_from_ckpt(den)
        model.eval()
        w = int(den.get("W", 512))
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            x = torch.randn(4, 1, w, generator=g)
            t = torch.full((4,), 40, dtype=torch.long)
            c = torch.zeros(4, dtype=torch.long)
            base = model(x, t, c)
            probes = {}
            for s in (w // 4, w // 2, 3 * w // 4):
                x2 = x.clone()
                x2[..., s] += 1.0
                d = (model(x2, t, c) - base).abs().squeeze(1)
                probes[str(s)] = {
                    "max_change_before": float(d[:, :s].max()),
                    "max_change_after": float(d[:, s:].max()),
                }
        report["causality_probe"] = probes
        report["is_causal"] = all(v["max_change_before"] < 1e-6 for v in probes.values())

        # ε-MSE by position on train-pool windows (the val split is not stored).
        lab_csv = s0 / "train_index.csv"
        if lab_csv.is_file():
            from anogen.config import panel_path
            from anogen.shell.data import load_panel
            from anogen.shell.windows import load_train_index, materialize

            splits = cfg.get("splits") or {}
            panel = load_panel(
                panel_path(cfg),
                channels=list(cfg.get("channels") or []),
                official_train_end=splits.get("official_train_end", "2007-01-01"),
                bin_seconds=int(cfg.get("bin_seconds", 30)),
            )
            tab = load_train_index(s0)
            rng = np.random.default_rng(0)
            pick = rng.choice(len(tab), size=min(args.n_val, len(tab)), replace=False)
            sub = tab.iloc[pick].reset_index(drop=True)
            xs = materialize(panel, sub, w)
            ch = sub["channel_idx"].to_numpy(dtype=np.int64)
            scaler = resolve_scaler(den, s0)
            if scaler is not None:
                xs = scaler.transform(xs, ch)
            sched = DiffusionSchedule.linear(int(den["n_times"]))
            x0 = torch.from_numpy(xs).unsqueeze(1)
            cb = torch.from_numpy(ch)
            by_t = {}
            with torch.no_grad():
                for tt in (5, 20, 40, 100, 199):
                    tb = torch.full((len(x0),), tt, dtype=torch.long)
                    torch.manual_seed(tt)
                    xt, noise = q_sample(x0, tb, sched)
                    err = ((model(xt, tb, cb) - noise) ** 2).squeeze(1).mean(dim=0).numpy()
                    by_t[str(tt)] = {
                        "first16": float(err[:16].mean()),
                        "middle": float(err[w // 2 - 8 : w // 2 + 8].mean()),
                        "last16": float(err[-16:].mean()),
                        "start_over_end": float(err[:16].mean() / max(err[-16:].mean(), 1e-12)),
                    }
            report["eps_mse_by_position"] = by_t
    except ImportError:
        report["torch"] = "not installed: probes 1-2 skipped"

    # 3. Peak position on saved galleries (numpy only).
    scaler = load_minmax(s0 / "minmax_scaler.npz")
    lab = np.load(s0 / "labeled_arrays.npz")
    sets = {"real anomalies": (lab["anomaly"], lab["anomaly_channel"])}
    if (s3 / "galleries.npz").is_file():
        g3 = np.load(s3 / "galleries.npz")
        sets["donors (real nominal)"] = (g3["cond"], g3["channel_idx"])
        sets["genias"] = (g3["genias"], g3["channel_idx"])
    if (s4 / "shell_gallery.npz").is_file():
        g4 = np.load(s4 / "shell_gallery.npz")
        sets["unguided nu=1"] = (g4["unguided"], g4["channel_idx"])
        sets["shell ZS"] = (g4["x"], g4["channel_idx"])
    for name, rel in (
        ("C1 chanmix fold0", "results/shell_chanmix/time_both_hybrid_needles_fold0.npz"),
        ("C1 contiguous fold0", "results/shell_kindmix_score_hybrid/time_both_hybrid_needles_fold0.npz"),
        ("hashfix fold0", "results/shell_hashfix/time_both_hashfix_fold0.npz"),
        ("from noise C3 fold0", "results/shell_noise_score/time_recon_combined_fold0.npz"),
    ):
        p = root / rel
        if p.is_file():
            b = np.load(p, allow_pickle=True)
            sets[name] = (b["x"], b["channel_idx"])
    pos = {}
    for name, (x, ch) in sets.items():
        u = channel_span_normalize(np.asarray(x), np.asarray(ch), scaler)
        pos[name] = position_report(u)
    report["peak_position"] = pos
    (out / "causality.json").write_text(json.dumps(report, indent=2) + "\n")

    if "is_causal" in report:
        print(f"backbone causal: {report['is_causal']}")
    for tt, r in (report.get("eps_mse_by_position") or {}).items():
        print(f"t={tt:>3}: eps-MSE first16/middle/last16 = {r['first16']:.4g}/{r['middle']:.4g}/{r['last16']:.4g}")
    print(f"{'gallery':28s} start  end   entropy")
    for name, r in pos.items():
        print(f"{name:28s} {r['peak_at_start']:.3f}  {r['peak_at_end']:.3f}  {r['position_entropy']:.3f}")


if __name__ == "__main__":
    main()
