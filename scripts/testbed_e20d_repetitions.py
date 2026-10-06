"""E20d: the E20 headline with honest intervals (20 repetitions).

E20's paired bootstrap resamples the test anomalies only. The training side
(which two events are seen, the generator's seeds) varies between repetitions
and is the larger source of uncertainty, so a claim needs intervals across
repetitions. This reruns the key arms of E20 for 20 repetitions (the first five
are E20's) with three detectors (logistic C = 0.1, logistic C = 10, ridge) and
reports, per paired difference, the mean over repetitions with a t-interval and
a two-level bootstrap (repetitions, then test anomalies within each).

Run: PYTHONPATH=src python scripts/testbed_e20d_repetitions.py [out] [reps]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import testbed_e20_detect_utility as e20  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/testbed")
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
W, NK = e20.W, e20.N_PER_KIND
ARMS = ("real_only", "+twin", "+C1-like", "+posthoc", "+recipe", "+recipe/rel+div", "+oracle")
DETECTORS = {"logistic C=0.1": ("logistic", 0.1), "logistic C=10": ("logistic", 10.0), "ridge": ("ridge", None)}
PAIRS = (("+recipe/rel+div", "real_only"), ("+recipe", "real_only"), ("+recipe/rel+div", "+recipe"),
         ("+recipe/rel+div", "+C1-like"), ("+recipe/rel+div", "+twin"), ("+recipe/rel+div", "+posthoc"),
         ("+C1-like", "real_only"), ("+twin", "real_only"), ("+posthoc", "real_only"), ("+oracle", "real_only"))


def fit_hits(kind, c, f_pos, f_neg, f_tn, f_ta):
    x = np.concatenate([f_pos, f_neg])
    y = np.r_[np.ones(len(f_pos)), np.zeros(len(f_neg))]
    if kind == "ridge":
        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 13), class_weight="balanced").fit(x, y)
    else:
        clf = LogisticRegression(C=float(c), class_weight="balanced", max_iter=5000).fit(x, y)
    s_nom, s_an = clf.decision_function(f_tn), clf.decision_function(f_ta)
    auc = float(roc_auc_score(np.r_[np.zeros(len(s_nom)), np.ones(len(s_an))], np.r_[s_nom, s_an]))
    return s_an > np.quantile(s_nom, 0.99), auc


def across(diffs_by_rep: list[np.ndarray], rng: np.random.Generator, n_boot: int = 2000) -> dict[str, float]:
    means = np.array([d.mean() for d in diffs_by_rep])
    n = len(means)
    half = float(stats.t.ppf(0.975, n - 1) * means.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    boot = np.empty(n_boot)
    for b in range(n_boot):
        reps = rng.integers(0, n, n)
        boot[b] = np.mean([diffs_by_rep[r][rng.integers(0, len(diffs_by_rep[r]), len(diffs_by_rep[r]))].mean()
                           for r in reps])
    return {"mean": float(means.mean()), "t_lo": float(means.mean() - half), "t_hi": float(means.mean() + half),
            "boot_lo": float(np.quantile(boot, 0.025)), "boot_hi": float(np.quantile(boot, 0.975)),
            "share_of_reps_positive": float(np.mean(means > 0)), "per_rep": [float(m) for m in means]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    hits: dict[tuple[str, str], list[np.ndarray]] = {(d, a): [] for d in DETECTORS for a in ARMS}
    aucs: dict[tuple[str, str], list[float]] = {(d, a): [] for d in DETECTORS for a in ARMS}
    kinds: list[np.ndarray] = []
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
        den = e20.DEN["flip_ramp"]
        synth = {
            "+twin": e20.recipe_slices(den, donors, real, seed=seed, shell=shell, steer=False)[0],
            "+C1-like": e20.c1_like(donors, real, seed=seed, shell=shell),
            "+posthoc": e20.posthoc_inject(donors, rng=np.random.default_rng(seed)).astype(np.float64),
            "+recipe": e20.recipe_slices(den, donors, real, seed=seed, shell=shell)[0],
            "+recipe/rel+div": e20.recipe_slices(den, donors, real, seed=seed, shell=shell, relative=True,
                                                 jitter=(0.5, 1.5), shift_max=16)[0],
            "+oracle": e20.oracle(donors, np.random.default_rng(seed)),
        }
        space = e20.RocketSpace(n_kernels=500, n_components=8, seed=rep).fit(neg[:1000])
        f_neg, f_tn, f_ta, f_real = (space.features(a) for a in (neg, test_nom, test_an, x_real))
        f_pos = {"real_only": f_real, **{a: np.concatenate([f_real, space.features(x)]) for a, x in synth.items()}}
        for dname, (kind, c) in DETECTORS.items():
            for arm in ARMS:
                h, auc = fit_hits(kind, c, f_pos[arm], f_neg, f_tn, f_ta)
                hits[(dname, arm)].append(h)
                aucs[(dname, arm)].append(auc)
        kinds.append(kind_test)
        print(f"rep {rep} ({time.time() - t0:.0f}s): "
              + " | ".join(f"{d} " + " ".join(f"{a}={hits[(d, a)][-1].mean():.3f}" for a in ARMS) for d in DETECTORS),
              flush=True)
    rng = np.random.default_rng(0)
    res: dict = {"reps": REPS, "arms": {}, "pairs": {}}
    for d in DETECTORS:
        res["arms"][d] = {a: {"recall_mean": float(np.mean([h.mean() for h in hits[(d, a)]])),
                              "recall_sd_over_reps": float(np.std([h.mean() for h in hits[(d, a)]], ddof=1)),
                              "auroc_mean": float(np.mean(aucs[(d, a)])),
                              **{f"recall {k}": float(np.mean([h[kd == k].mean() for h, kd in zip(hits[(d, a)], kinds,
                                                                                                    strict=True)]))
                                 for k in e20.KINDS}}
                          for a in ARMS}
        res["pairs"][d] = {}
        for a, b in PAIRS:
            diffs = [ha.astype(float) - hb.astype(float) for ha, hb in zip(hits[(d, a)], hits[(d, b)], strict=True)]
            res["pairs"][d][f"{a} - {b}"] = across(diffs, rng)
    (OUT / "e20d_repetitions.json").write_text(json.dumps(res, indent=1))
    for d in DETECTORS:
        print(f"\n{d}")
        for a in ARMS:
            r = res["arms"][d][a]
            print(f"  {a:18s} recall {r['recall_mean']:.3f} (sd over reps {r['recall_sd_over_reps']:.3f})")
        for k, p in res["pairs"][d].items():
            print(f"  {k:32s} {p['mean']:+.3f}  t [{p['t_lo']:+.3f}, {p['t_hi']:+.3f}]  "
                  f"2-level boot [{p['boot_lo']:+.3f}, {p['boot_hi']:+.3f}]  positive in {p['share_of_reps_positive']:.0%}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
