"""Consolidated GenFSDiff-vs-baselines table: frozen S4 protocol + realism audit.

ARP comes from each method's own phase run; all of them carry the same
s4_frozen tau, so ARP is comparable across rows. EDI unions differ per run and
are reported for context only.

Writes docs/COMPARISON.md and docs/comparison.csv. Reads only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from anogen.config import REPO_ROOT

# realism-audit name -> (phase run, key inside summary["methods"])
S4_SOURCE = {
    "c1": ("shell_kindmix_score_hybrid", "time_both_hybrid_needles"),
    "c2": ("shell_kindmix_score_hybrid", "time_recon_hybrid_needles"),
    "shell": ("shell_kindmix_score_hybrid", "shell"),
    "unguided": ("shell_kindmix_score_hybrid", "unguided"),
    "genias": ("shell_kindmix_score_hybrid", "genias"),
    "posthoc": ("shell_kindmix_score_hybrid", "posthoc"),
    "hashfix": ("shell_hashfix", "time_both_hashfix"),
    "timeleash_l2": ("shell_timeleash", "time_both_l2"),
    "timeleash_gm": ("shell_timeleash", "time_both_gm"),
    "editor": ("shell_editor", "time_both_hybrid_needles_editor"),
    "cutaddpaste": ("shell_genbase", "cutaddpaste"),
    "taxonomy": ("shell_genbase", "taxonomy"),
}
LABEL = {
    "c1": "GenFSDiff (c1, time_both hybrid_needles)",
    "c2": "GenFSDiff (c2, time_recon hybrid_needles)",
    "timeleash_gm": "GenFSDiff + parent leash (gm)",
    "timeleash_l2": "GenFSDiff + parent leash (l2)",
    "hashfix": "GenFSDiff, apply_final_grad=False",
    "editor": "GenFSDiff + editor patch",
    "shell": "shell-only steering (no anomaly term)",
    "unguided": "unguided diffusion",
    "genias": "GenIAS",
    "genias_patched": "GenIAS (patched -- inert, == donor)",
    "posthoc": "post-hoc injection",
    "cutaddpaste": "CutAddPaste",
    "taxonomy": "taxonomy injection",
    "donor": "donor (real nominal, no edit)",
}
ORDER = [
    "c1", "c2", "timeleash_gm", "timeleash_l2", "editor", "hashfix",
    "shell", "unguided", "genias", "genias_patched", "posthoc",
    "cutaddpaste", "taxonomy", "donor",
]


def _s4(root: Path) -> pd.DataFrame:
    rows = []
    for name, (run, key) in S4_SOURCE.items():
        path = root / "results" / run / "summary.json"
        if not path.is_file():
            continue
        d = json.load(open(path))
        m = (d.get("methods") or {}).get(key)
        if not m:
            continue
        rows.append(
            {
                "method": name,
                "arp": m.get("arp_anomaly"),
                "edi": m.get("edi"),
                "s4_run": run,
            }
        )
    return pd.DataFrame(rows).set_index("method")


def main() -> None:
    root = Path(REPO_ROOT)
    audit = root / "results/shell_realism"

    met = pd.read_csv(audit / "metrics.csv")
    met = met[(met["scope"] == "whole_gallery") & (met["kind"] == "all")]
    pub = met[met["tier"] == "published"].pivot_table(
        index="method", columns="metric", values="value", aggfunc="mean"
    )
    diagm = met[met["tier"] == "diagnostic"].pivot_table(
        index="method", columns="metric", values="value", aggfunc="mean"
    )

    dia = pd.read_csv(audit / "diagnostics.csv")
    dia = dia[(dia["scope"] == "whole_gallery") & (dia["kind"] == "all")]
    real = dia[dia["sample"] == "real"].groupby("metric")["mean"].mean()
    syn = dia[dia["sample"] == "synthetic"].pivot_table(
        index="method", columns="metric", values="mean", aggfunc="mean"
    )

    c2 = pd.read_csv(audit / "c2st.csv").groupby("method").agg(
        c2st_acc=("balanced_accuracy", "median"),
        c2st_reject=("permutation_p", lambda s: float((s < 0.05).mean())),
    )

    pos = {}
    pw = np.load(audit / "per_window.npz", allow_pickle=True)
    for k in pw.files:
        p = k.split("__")
        if len(p) < 6 or p[2] != "whole_gallery" or p[3] != "all":
            continue
        if p[-1] != "signed_peak_position":
            continue
        v = np.asarray(pw[k], dtype=float)
        if v.size:
            pos.setdefault((p[0], p[-2]), []).append(float((v < 0.10).mean()))
    peak_start = pd.Series(
        {m: float(np.mean(v)) for (m, s), v in pos.items() if s == "synthetic"}
    )
    real_peak_start = float(np.mean([x for (m, s), v in pos.items() if s == "real" for x in v]))

    t = pd.DataFrame(index=[m for m in ORDER if m in pub.index])
    t["method"] = [LABEL[m] for m in t.index]
    t["ARP"] = _s4(root)["arp"]
    t["EDI"] = _s4(root)["edi"]
    t["C2ST acc"] = c2["c2st_acc"]
    t["C2ST rej"] = c2["c2st_reject"]
    t["MDD"] = pub["mdd"]
    t["ACD"] = pub["acd"]
    t["W1"] = pub["marginal_w1"]
    t["AWD"] = pub["fft_awd_amplitude"]
    t["featW1"] = pub["feature_distance_macro"]
    t["MDD disc."] = diagm["mdd_out_of_real_range_fraction"]
    t["env frac"] = syn["envelope_window_fraction"]
    t["env max"] = syn["envelope_max_excess"]
    t["CUSUM"] = syn["cusum_contrast"]
    t["diff p99.9"] = syn["diff_p999"]
    t["peak@start"] = peak_start

    ref = pd.Series(
        {
            "method": "real ESA anomalies (target)",
            "C2ST acc": 0.5,
            "MDD": 0.0,
            "ACD": 0.0,
            "W1": 0.0,
            "AWD": 0.0,
            "featW1": 0.0,
            "MDD disc.": 0.0,
            "env frac": real["envelope_window_fraction"],
            "env max": real["envelope_max_excess"],
            "CUSUM": real["cusum_contrast"],
            "diff p99.9": real["diff_p999"],
            "peak@start": real_peak_start,
        },
        name="__real__",
    )
    t = pd.concat([pd.DataFrame([ref]), t])
    t.to_csv(root / "docs/comparison.csv")

    body = t.set_index("method").round(4)
    md = [
        "# GenFSDiff vs. baselines",
        "",
        "Generated by `scripts/make_comparison_table.py`. Frozen S4 protocol",
        "(ARP, EDI) plus the raw-window realism audit (`docs/REALISM.md`).",
        "",
        "All rows carry the same `s4_frozen` tau, so ARP is comparable. EDI unions",
        "are per-run and are context only. The first row is the real-anomaly target:",
        "distances want 0, C2ST accuracy wants 0.5, diagnostics want the real value.",
        "",
        "| " + " | ".join(["method"] + list(body.columns)) + " |",
        "|" + "|".join(["---"] * (len(body.columns) + 1)) + "|",
    ]
    for name, row in body.iterrows():
        cells = ["" if pd.isna(v) else f"{v:g}" for v in row]
        md.append("| " + " | ".join([str(name)] + cells) + " |")
    md += [
        "",
        "Columns: **ARP** anomalous representation proximity (frozen S4, higher is",
        "better). **EDI** entropy diversity index. **C2ST acc** median balanced",
        "accuracy of a linear real-vs-synthetic classifier over 3 folds x 5 seeds",
        "(0.5 = indistinguishable); **C2ST rej** fraction of the 15 runs rejecting at",
        "0.05. **MDD/ACD/W1/AWD/featW1** published distances, lower is better.",
        "**MDD disc.** fraction of generated mass TSGBench's MDD silently discards.",
        "**env frac** fraction of windows leaving the nominal training envelope;",
        "**env max** largest excess in channel spans; **CUSUM** mean step contrast;",
        "**diff p99.9** first-difference tail (edge sharpness); **peak@start**",
        "fraction of windows whose dominant peak sits in the first 10% of the span",
        "(uniform would be 0.10).",
        "",
    ]
    md += _within_run(root)
    (root / "docs/COMPARISON.md").write_text("\n".join(md))
    print("\n".join(md[9:]))


def _within_run(root: Path) -> list[str]:
    """EDI unions are per-run, so the only safe EDI read is inside one run."""
    out = [
        "## Within-run ARP / EDI",
        "",
        "EDI is defined against a per-run union, so across-run EDI in the table",
        "above is indicative only. These blocks compare inside a single run.",
        "",
    ]
    for run in ("shell_editor", "shell_timeleash", "shell_hashfix"):
        path = root / "results" / run / "summary.json"
        if not path.is_file():
            continue
        m = json.load(open(path))["methods"]
        rows = {
            k: {
                "ARP": v.get("arp_anomaly"),
                "EDI": v.get("edi"),
                "Coverage@tau": v.get("coverage_anomaly"),
                "Div": v.get("diversity"),
            }
            for k, v in m.items()
        }
        t = pd.DataFrame(rows).T.round(4)
        out += [f"`{run}`", "", "| method | " + " | ".join(t.columns) + " |",
                "|" + "|".join(["---"] * (len(t.columns) + 1)) + "|"]
        for name, row in t.iterrows():
            cells = ["" if pd.isna(v) else f"{v:g}" for v in row]
            out.append("| " + " | ".join([str(name)] + cells) + " |")
        out.append("")
    return out


if __name__ == "__main__":
    main()
