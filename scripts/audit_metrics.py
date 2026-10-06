"""Re-score every saved gallery with intervals, controls and fidelity/diversity split.

    .venv/bin/python scripts/audit_metrics.py -c configs/shell_mission1.yaml

Numpy only, reads saved artifacts, writes results/shell_metric_audit/ and
prints a markdown table. Does not change φ, τ or any frozen number: the point
estimates of ARP and Coverage@τ are the protocol values (fold mean).

What it adds per gallery (fold-wise, then averaged):

* ARP and Coverage@τ with event-cluster *two-way* bootstrap 95 % CIs
  (queries resampled by event, gallery resampled by row)
* paired ARP difference vs the untouched donors and vs unguided DDIM
  (same queries) — is the gain anomaly-specific or a sampler property?
* precision / recall / density / coverage in φ against the fold's real
  anomalies (fidelity and spread kept apart)
* which φ feature carries the query→nearest distance
* position of the dominant excursion (start / end / entropy) and envelope
  exit frequency vs magnitude, in channel-span units
* a channel-balanced, event-grouped ROCKET C2ST AUC (0.5 = indistinguishable
  from the fold's real anomalies) and ARP in the ROCKET PCA space
  (shell/rocket.py; testbed E18: the linear raw-window C2ST has no power,
  ROCKET detects texture, amplitude and position defects). ``--rocket-kernels 0``
  switches it off.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from anogen.config import load_config
from anogen.shell.evaluation import (
    PHI_NAMES,
    arp_coverage_ci,
    envelope_frequency_report,
    feature_attribution,
    paired_arp_difference,
    peak_positions,
    position_report,
    prdc,
    phi_standardizer,
    query_keep_mask,
    standardize_phi,
)
from anogen.shell.features import embed_windows
from anogen.shell.realism import channel_span_normalize
from anogen.shell.scaler import load_minmax


def _galleries(root: Path, s3: Path, s4: Path) -> dict[str, dict]:
    """name -> {"fold": None | {k: (x, ch)}, "all": (x, ch)}"""
    out: dict[str, dict] = {}
    g3 = np.load(s3 / "galleries.npz")
    ch = np.asarray(g3["channel_idx"])
    out["donor (real nominal)"] = {"all": (g3["cond"], ch)}
    out["GenIAS (locked S3)"] = {"all": (g3["genias"], ch)}
    out["post-hoc"] = {"all": (g3["posthoc"], ch)}
    if (s4 / "shell_gallery.npz").is_file():
        g4 = np.load(s4 / "shell_gallery.npz")
        out["unguided nu=1"] = {"all": (g4["unguided"], g4["channel_idx"])}
        out["shell ZS"] = {"all": (g4["x"], g4["channel_idx"])}
    gb = root / "results/shell_genbase"
    for name in ("cutaddpaste", "taxonomy"):
        p = gb / f"{name}.npz"
        if p.is_file():
            b = np.load(p, allow_pickle=True)
            out[name] = {"all": (b["x"], b["channel_idx"])}
    fold_sets = {
        "C1 contiguous": "results/shell_kindmix_score_hybrid/time_both_hybrid_needles_fold{k}.npz",
        "C1 chanmix": "results/shell_chanmix/time_both_hybrid_needles_fold{k}.npz",
        "C2 chanmix": "results/shell_chanmix/time_recon_hybrid_needles_fold{k}.npz",
        "C3 combined": "results/shell_enc_score/time_recon_combined_fold{k}.npz",
        "C1 from noise": "results/shell_noise_score/time_both_hybrid_needles_fold{k}.npz",
        "C3 from noise": "results/shell_noise_score/time_recon_combined_fold{k}.npz",
        "hashfix": "results/shell_hashfix/time_both_hashfix_fold{k}.npz",
        "editor": "results/shell_editor/time_both_hybrid_needles_editor_fold{k}.npz",
        "GenIAS faithful": None,
    }
    for name, pattern in fold_sets.items():
        if pattern is None:
            continue
        folds = {}
        for k in range(3):
            p = root / pattern.format(k=k)
            if p.is_file():
                b = np.load(p, allow_pickle=True)
                folds[k] = (b["x"], b["channel_idx"] if "channel_idx" in b.files else ch)
        if folds:
            out[name] = {"fold": folds}
    gf = root / "results/shell_genias_fair/galleries.npz"
    if gf.is_file():
        b = np.load(gf)
        out["GenIAS faithful (Alg.2 off)"] = {"all": (b["genias_fair"], b["channel_idx"])}
    fs = root / "results/shell_fixsweep"
    if fs.is_dir():
        for p in sorted(fs.glob("*_fold0.npz")):
            name = p.name[: -len("_fold0.npz")]
            folds = {}
            for k in range(3):
                q = fs / f"{name}_fold{k}.npz"
                if q.is_file():
                    b = np.load(q, allow_pickle=True)
                    folds[k] = (b["x"], b["channel_idx"])
            out[f"fixsweep:{name}"] = {"fold": folds}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=None, help="output dir (default results/shell_metric_audit)")
    ap.add_argument("--rocket-kernels", type=int, default=300, help="0 = skip the ROCKET C2ST / ARP")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["_repo_root"])
    s0, s3, s4 = (root / cfg[k] for k in ("s0_dir", "s3_dir", "s4_dir"))
    out = Path(args.out) if args.out else root / "results/shell_metric_audit"
    out.mkdir(parents=True, exist_ok=True)
    tau = float(json.loads((s4 / "summary.json").read_text())["tau"])
    lab = np.load(s0 / "labeled_arrays.npz")
    meta = pd.read_csv(s0 / "labeled_windows.csv")
    anom = meta[meta["kind"] == "anomaly"].reset_index(drop=True)
    x_a, fold_a, ch_a = np.asarray(lab["anomaly"]), np.asarray(lab["anomaly_fold"]), np.asarray(lab["anomaly_channel"])
    ev = anom["event_id"].astype(str).to_numpy()
    fold_tab = pd.read_csv(s0 / "channel_fold_counts.csv")
    scaler = load_minmax(s0 / "minmax_scaler.npz")
    gals = _galleries(root, s3, s4)
    folds = sorted({int(f) for f in fold_a if int(f) >= 0})

    def fold_gallery(spec: dict, k: int):
        if "all" in spec:
            return spec["all"]
        return spec["fold"].get(k)

    ref_names = ("donor (real nominal)", "unguided nu=1")
    rocket_space = None
    if args.rocket_kernels > 0:
        from anogen.shell.coverage import arp as _arp
        from anogen.shell.coverage import min_distances
        from anogen.shell.rocket import RocketSpace, rocket_c2st

        d_x, d_ch = gals["donor (real nominal)"]["all"]
        rocket_space = RocketSpace(n_kernels=args.rocket_kernels, seed=0).fit(
            channel_span_normalize(np.asarray(d_x), np.asarray(d_ch), scaler)
        )
    phi_stats = phi_standardizer(embed_windows(np.asarray(gals["donor (real nominal)"]["all"][0])))
    # Oracle: real anomaly windows of the *other* folds as the gallery (the
    # honest ceiling for "reproduces known faults").
    gals["oracle (real anomalies, other folds)"] = {
        "fold": {k: (x_a[fold_a != k], ch_a[fold_a != k]) for k in sorted({int(f) for f in fold_a if int(f) >= 0})},
        # crops of one event are dependent: group them for the C2ST split
        "groups": {k: ev[fold_a != k] for k in sorted({int(f) for f in fold_a if int(f) >= 0})},
    }
    rows = {}
    for name, spec in gals.items():
        per = []
        for k in folds:
            gk = fold_gallery(spec, k)
            if gk is None:
                continue
            x, ch = np.asarray(gk[0]), np.asarray(gk[1])
            qm = query_keep_mask(ch_a, fold_a, fold_tab, fold_id=k)
            q_emb, g_emb = embed_windows(x_a[qm]), embed_windows(x)
            r = arp_coverage_ci(q_emb, g_emb, ev[qm], tau=tau, n_boot=args.n_boot, seed=k, resample_gallery=True)
            # Sensitivity: ARP in a per-feature standardized φ (constants dropped),
            # standardized on the real nominal donors. Not the frozen protocol.
            r["arp_phi_std"] = float(
                1.0 / (1.0 + np.mean(np.min(np.linalg.norm(
                    standardize_phi(q_emb, phi_stats)[:, None, :] - standardize_phi(g_emb, phi_stats)[None, :, :], axis=2
                ), axis=1)))
            )
            # Sensitivity: φ without the three upper log-FFT bands (texture);
            # testbed E12 shows frozen-φ ARP is dominated by texture match.
            r["arp_phi_no_hf"] = float(
                1.0 / (1.0 + np.mean(np.min(np.linalg.norm(
                    q_emb[:, None, :9] - g_emb[None, :, :9], axis=2
                ), axis=1)))
            )
            r.update({f"prdc_{k}": v for k, v in prdc(q_emb, g_emb).items()})
            if rocket_space is not None:
                rm_ = fold_a == k
                u_r = channel_span_normalize(x_a[rm_], ch_a[rm_], scaler)
                f_g = rocket_space.features(channel_span_normalize(x, ch, scaler))
                g_groups = spec.get("groups", {}).get(k, np.arange(len(x)))
                try:
                    r["c2st_rocket_auc"] = rocket_c2st(
                        rocket_space, u_r, None, ch_a[rm_], ch, ev[rm_], g_groups, feat_synth=f_g
                    )["auc"]
                except ValueError:
                    pass  # left out of the mean rather than turning it into NaN
                u_q = channel_span_normalize(x_a[qm], ch_a[qm], scaler)
                r["arp_rocket"] = _arp(min_distances(rocket_space.embed(u_q), rocket_space.embed_features(f_g)))
            r["phi_attribution"] = feature_attribution(q_emb, g_emb, PHI_NAMES)
            u = channel_span_normalize(x, ch, scaler)
            rm = fold_a == k
            u_real = channel_span_normalize(x_a[rm], ch_a[rm], scaler)
            r.update(position_report(u, real_positions=peak_positions(u_real)))
            r.update(envelope_frequency_report(u, u_real))
            for ref in ref_names:
                rg = fold_gallery(gals.get(ref, {}), k) if ref in gals else None
                if rg is None or ref == name:
                    continue
                d = paired_arp_difference(q_emb, g_emb, embed_windows(rg[0]), ev[qm], tau=tau, n_boot=args.n_boot, seed=k)
                r[f"arp_minus_{ref.split()[0]}"] = d["arp_diff"]
                r[f"arp_minus_{ref.split()[0]}_ci"] = d["arp_diff_ci"]
            per.append(r)
        if per:
            rows[name] = per
    # real anomalies themselves, for the position / envelope targets
    real_u = channel_span_normalize(x_a, ch_a, scaler)
    target = {**position_report(real_u), **envelope_frequency_report(real_u)}
    (out / "audit.json").write_text(json.dumps({"tau": tau, "target": target, "rows": rows}, indent=1, default=float))

    def m(per, key):
        vals = [p[key] for p in per if key in p]
        return float(np.mean(vals)) if vals else float("nan")

    def ci(per, key):
        vals = [p[key] for p in per if key in p]
        return f"[{np.mean([v[0] for v in vals]):.3f}, {np.mean([v[1] for v in vals]):.3f}]" if vals else ""

    lines = [
        "| gallery | ARP | ARP 95% CI | ARP − donor | ARP − unguided | ARP (std φ) | ARP (φ no HF) | C2ST ROCKET AUC | Cov@τ | precision | density | coverage(PRDC) | start | end | env exit |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| real anomalies (target) | | | | | | | 0.5 | | | | | {target['peak_at_start']:.3f} | {target['peak_at_end']:.3f} | {target['exit_frac']:.3f} |",
    ]
    for name, per in rows.items():
        lines.append(
            f"| {name} | {m(per, 'arp'):.3f} | {ci(per, 'arp_ci')} | {m(per, 'arp_minus_donor'):+.3f} | "
            f"{m(per, 'arp_minus_unguided'):+.3f} | {m(per, 'arp_phi_std'):.3f} | {m(per, 'arp_phi_no_hf'):.3f} | {m(per, 'c2st_rocket_auc'):.3f} | {m(per, 'coverage'):.3f} | {m(per, 'prdc_precision'):.3f} | "
            f"{m(per, 'prdc_density'):.3f} | {m(per, 'prdc_coverage'):.3f} | "
            f"{m(per, 'peak_at_start'):.3f} | {m(per, 'peak_at_end'):.3f} | {m(per, 'exit_frac'):.3f} |"
        )
    table = "\n".join(lines)
    (out / "TABLE.md").write_text(table + "\n")
    print(table)
    print("\nφ attribution of the query→nearest distance (fold mean, top 3 features):")
    for name, per in rows.items():
        att = {f: np.mean([p["phi_attribution"][f] for p in per]) for f in PHI_NAMES}
        top = sorted(att.items(), key=lambda kv: -kv[1])[:3]
        print(f"  {name:30s} " + ", ".join(f"{k} {v:.2f}" for k, v in top))


if __name__ == "__main__":
    main()
