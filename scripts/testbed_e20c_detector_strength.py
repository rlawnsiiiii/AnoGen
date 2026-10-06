"""E20c: does the E20 gain survive a better-tuned detector?

E20's logistic detector uses a fixed C = 0.1 and is weaker on real anomalies
alone (0.49) than the ridge detector with its cross-validated penalty (0.59).
A gain from generated positives that only shows on a weak detector would not
be worth much. This sweeps C for the logistic detector and reports real_only,
+recipe/rel+div and +oracle at each C, on E20's first repetitions (same seeds).

Run: PYTHONPATH=src python scripts/testbed_e20c_detector_strength.py [out] [reps]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import testbed_e20_detect_utility as e20  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed")
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
CS = (0.01, 0.1, 1.0, 10.0)
W, NK = e20.W, e20.N_PER_KIND


def recall_at_far(c, f_pos, f_neg, f_tn, f_ta, kind_test):
    x = np.concatenate([f_pos, f_neg])
    y = np.r_[np.ones(len(f_pos)), np.zeros(len(f_neg))]
    clf = LogisticRegression(C=float(c), class_weight="balanced", max_iter=5000).fit(x, y)
    s_nom, s_an = clf.decision_function(f_tn), clf.decision_function(f_ta)
    hits = s_an > np.quantile(s_nom, 0.99)
    out = {"recall@1%FAR": float(hits.mean()),
           "auroc": float(roc_auc_score(np.r_[np.zeros(len(s_nom)), np.ones(len(s_an))], np.r_[s_nom, s_an]))}
    out.update({f"recall {k}": float(hits[kind_test == k].mean()) for k in e20.KINDS})
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict]] = {}
    for rep in range(REPS):
        rng = np.random.default_rng(1000 + rep)  # same draws as E20's repetition rep
        real = e20.few_shot_events(rng)
        x_real = np.concatenate([real[k] for k in e20.KINDS])
        neg = e20.ch.sample(e20.N_NEG, W, rng)
        shell = e20.Shell.from_nominal(e20.enc, e20.ch.sample(512, W, rng), e20.ch.sample(512, W, rng))
        donors = neg[rng.permutation(e20.N_NEG)[: NK * len(e20.KINDS)]]
        test_nom = e20.ch.sample(e20.N_TEST_NOM, W, rng)
        kind_test = np.repeat(np.array(e20.KINDS), e20.N_TEST_PER_KIND)
        test_an = np.concatenate([e20.inject(k, e20.ch.sample(e20.N_TEST_PER_KIND, W, rng),
                                             e20.positions(k, e20.N_TEST_PER_KIND, rng),
                                             e20.amplitudes(k, e20.N_TEST_PER_KIND, rng)) for k in e20.KINDS])
        seed = 7 + 100 * rep
        div, _ = e20.recipe_slices(e20.DEN["flip_ramp"], donors, real, seed=seed, shell=shell, relative=True,
                                   jitter=(0.5, 1.5), shift_max=16)
        space = e20.RocketSpace(n_kernels=500, n_components=64, seed=rep).fit(neg[:1000])
        f_neg, f_tn, f_ta, f_real = (space.features(a) for a in (neg, test_nom, test_an, x_real))
        pos = {"real_only": f_real, "+recipe/rel+div": np.concatenate([f_real, space.features(div)]),
               "+oracle": np.concatenate([f_real, space.features(e20.oracle(donors, np.random.default_rng(seed)))])}
        for c in CS:
            for arm, fp in pos.items():
                rows.setdefault(f"C={c:g} | {arm}", []).append(recall_at_far(c, fp, f_neg, f_tn, f_ta, kind_test))
        print(rep, {k: round(v[-1]["recall@1%FAR"], 3) for k, v in rows.items()}, flush=True)
    summary = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in rows.items()}
    (OUT / "e20c_detector_strength.json").write_text(json.dumps({"reps": REPS, "summary": summary, "per_rep": rows},
                                                                indent=1))
    for k, s in summary.items():
        print(k, {m: round(v, 3) for m, v in s.items()})


if __name__ == "__main__":
    main()
