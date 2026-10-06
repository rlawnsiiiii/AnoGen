"""E20: do generated anomalies help a supervised detector? (detector utility)

The question the paper should lead with, asked where the answer is known.
Few-shot setting on the simulated channel: three anomaly kinds (spike, level
shift, drift), two training events per kind, three overlapping crops per
event (18 real training windows, the ESA situation in miniature). Each arm
adds up to 255 synthetic positives (85 per kind) to the real ones and trains
the same detector against 2000 nominal windows.

Generators:
  posthoc         shell/baselines.posthoc_inject (step, pulse, ramp, scale)
  unguided        causal denoiser, nu = 1 (S4's unguided row)
  C1-like         frozen sampler: causal denoiser, x-space guidance, unit kicks,
                  last kick undenoised, shell band + prototypes of the real
                  windows for every kind
  twin            the recipe's sampler, masks and seeds with every objective off
  recipe          flip-ramp denoiser, x0-space guidance, final 0.25, two
                  recurrences, guidance decay 1; spikes and steps as masked
                  contrast edits with absolute targets re-sampled from the real
                  windows, drift by prototypes (fixsweep flip_x0_contrast_masked)
  recipe/rel      the same, contrast targets relative to the donor: delta =
                  w . donor + Delta, so the planted anomaly has size Delta
                  whatever the donor's own contrast at that position
                  (fixsweep contrast_relative: true)
  recipe/rel+jitter  contrast sizes Delta x U(0.5, 1.5): a wider amplitude
                  range than the two training events show
                  (fixsweep contrast_amp_jitter: [0.5, 1.5])
  recipe/rel+shift  drift prototypes shifted in time by up to +-16 encoder
                  steps (128 bins) per sample (fixsweep proto_shift_max: 16)
  recipe/rel+div  both
  recipe/rel/bidir  recipe/rel on a bidirectional denoiser (after a retrain)
  oracle          the true anomaly process injected into donors (ceiling)

Training variants:
  plain           every generated window is a positive
  kNN filter      keep a generated window only if its distance to the nominal
                  training windows (5-NN, ROCKET PCA space) exceeds the 99th
                  percentile of the same distance for held-out nominal windows
                  (augdetect synth_filter: rocket_knn)
  weighted        all positives, synthetic ones weighted so that they carry
                  the same total weight as the real ones (the positive class
                  keeps its total weight; augdetect real_frac: 0.5)

Detectors: ROCKET (500 kernels, ppv + max) with a ridge classifier (squared
loss, ROCKET's default) and with an L2 logistic regression (the CNN's BCE
loss), stand-ins for augdetect's CNN (all pool over time). Test: 300 new anomalies per kind
with amplitudes over the whole range and uniform positions, 3000 nominal
windows. Primary: recall at 1 % nominal false alarms; also AUROC and AP.
Repeated over 5 draws of the few-shot events and of all generation seeds;
paired differences per test anomaly, pooled over repetitions, bootstrap CI.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from anogen.shell.baselines import posthoc_inject
from anogen.shell.contrast import contrast_mask, contrast_weights, measure_contrast
from anogen.shell.rocket import RocketSpace
from anogen.shell.steer import proto_shift_draws, shift_time_embedding_np
from anogen.testbed.gauss import (
    BlockEncoder,
    GPChannel,
    GuideTerm,
    LinearDenoiser,
    Schedule,
    Shell,
    guided_ddim,
    level_shift,
    proto_energy,
    spike,
)

OUT = Path(sys.argv[1] if (__name__ == "__main__" and len(sys.argv) > 1) else "results/testbed")
REPS = int(sys.argv[2]) if (__name__ == "__main__" and len(sys.argv) > 2) else 5
W, CROP = 512, 64
N_PER_KIND, N_NEG, N_TEST_NOM, N_TEST_PER_KIND = 85, 2000, 3000, 300
KINDS = ("spike", "level shift", "drift")
AMP = {"spike": (0.04, 0.16), "level shift": (0.04, 0.16), "drift": (0.06, 0.20)}
CONTRAST = {"spike": "spike", "level shift": "step"}  # drift: prototypes
ch, sch = GPChannel(), Schedule()
enc = BlockEncoder(W, 8)
DEN = {k: LinearDenoiser(ch, W, sch, k) for k in ("causal", "flip_ramp", "bidir")}


def drift(x: np.ndarray, pos: np.ndarray, size: np.ndarray, length: int = 128) -> np.ndarray:
    out = np.array(x, dtype=np.float64, copy=True)
    t = np.arange(out.shape[1])
    for i in range(len(out)):
        out[i] += size[i] * np.clip((t - pos[i]) / float(length), 0.0, 1.0)
    return out


def inject(kind: str, base: np.ndarray, pos: np.ndarray, size: np.ndarray) -> np.ndarray:
    if kind == "spike":
        return spike(base, pos, size, width=3)
    if kind == "level shift":
        return level_shift(base, pos, size, ramp=6)
    return drift(base, pos, size)


def positions(kind: str, n: int, rng: np.random.Generator, width: int = W) -> np.ndarray:
    lo, hi = {"spike": (16, width - 20), "level shift": (40, width - 40), "drift": (0, width - 200)}[kind]
    return rng.integers(lo, hi, n)


def amplitudes(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    a, b = AMP[kind]
    return rng.choice([-1.0, 1.0], n) * rng.uniform(a, b, n)


def few_shot_events(rng: np.random.Generator, n_events: int = 2) -> dict[str, np.ndarray]:
    """Per kind: n_events events, each seen through 3 crops offset by CROP bins."""
    out = {}
    for kind in KINDS:
        wins = []
        for _ in range(n_events):
            long = ch.sample(1, W + 2 * CROP, rng)
            lo, hi = {"spike": (2 * CROP + 16, W - 20), "level shift": (2 * CROP + 40, W - 40),
                      "drift": (2 * CROP + 12, W - 200)}[kind]
            p = int(rng.integers(lo, hi))
            x = inject(kind, long, np.array([p]), amplitudes(kind, 1, rng))[0]
            wins += [x[o : o + W] for o in (0, CROP, 2 * CROP)]
        out[kind] = np.asarray(wins)
    return out


def contrast_fn(w: np.ndarray, wd: np.ndarray, delta: np.ndarray):
    nn = np.maximum((wd * wd).sum(axis=1), 1e-12)

    def fn(x: np.ndarray):
        d = (x * w).sum(axis=1) - delta
        return d**2, (d / nn)[:, None] * wd

    return fn


def recipe_slices(den, donors, real, *, seed, shell, relative=False, steer=True, jitter=None, shift_max=0):
    """flip_x0_contrast_masked on the testbed: masked contrasts for spike/step, prototypes for drift.

    Returns the windows and, for the contrast slices, the realized size of the
    planted anomaly (contrast of the output minus that of the donor at the
    requested position) next to the requested one.
    """
    out, realized = [], {}
    for k, kind in enumerate(KINDS):
        d = donors[k * N_PER_KIND : (k + 1) * N_PER_KIND]
        r = np.random.default_rng(seed + 53 * (1 + k))
        common = dict(rng=np.random.default_rng(seed + k), nu=0.2, space="x0", final_scale=0.25,
                      n_recur=2, guidance_decay=1.0)
        if kind in CONTRAST:
            ck = CONTRAST[kind]
            guard = max(8, W // 16) if ck == "step" else 12
            pos = r.integers(max(guard, int(0.1 * W)), min(W - guard, int(0.9 * W)), len(d))
            real_amp, _ = measure_contrast(ck, real[kind])
            w = contrast_weights(ck, pos, W)
            base = (d * w).sum(axis=1)
            size = r.choice(real_amp, len(d))
            if jitter is not None:
                size = size * np.random.default_rng(seed + 71 * (1 + k)).uniform(jitter[0], jitter[1], len(d))
            delta = base + size if relative else size
            m = contrast_mask(ck, pos, W, dilate=8)
            terms = [GuideTerm(0.3, shell.band), GuideTerm(1.0, contrast_fn(w, w * m, delta), project=True)] if steer else []
            x = guided_ddim(den, d, terms, gen_mask=m, **common)
            realized[kind] = {"requested_size_abs_median": float(np.median(np.abs(delta - base))),
                              "realized_size_abs_median": float(np.median(np.abs((x * w).sum(axis=1) - base))),
                              "realized_below_0.02": float(np.mean(np.abs((x * w).sum(axis=1) - base) < 0.02))}
            out.append(x)
        else:
            pz = enc.encode(real[kind])[r.integers(0, len(real[kind]), len(d))]
            if shift_max:
                pz = shift_time_embedding_np(pz, proto_shift_draws(len(d), shift_max, seed=seed + 31 * (1 + k)),
                                             enc.n_blocks)
            terms = [GuideTerm(0.3, shell.band), GuideTerm(1.0, lambda x, pz=pz: proto_energy(enc, x, pz))] if steer else []
            out.append(guided_ddim(den, d, terms, **common))
    return np.concatenate(out), realized


def c1_like(donors, real, *, seed, shell):
    out = []
    for k, kind in enumerate(KINDS):
        d = donors[k * N_PER_KIND : (k + 1) * N_PER_KIND]
        r = np.random.default_rng(seed + 17 * (1 + k))
        pz = enc.encode(real[kind])[r.integers(0, len(real[kind]), len(d))]
        terms = [GuideTerm(0.3, shell.band), GuideTerm(1.0, lambda x, pz=pz: proto_energy(enc, x, pz))]
        out.append(guided_ddim(DEN["causal"], d, terms, rng=np.random.default_rng(seed + k), nu=0.2,
                               space="x", final_scale=1.0))
    return np.concatenate(out)


def oracle(donors, rng):
    return np.concatenate([inject(kind, donors[k * N_PER_KIND : (k + 1) * N_PER_KIND],
                                  positions(kind, N_PER_KIND, rng), amplitudes(kind, N_PER_KIND, rng))
                           for k, kind in enumerate(KINDS)])


def evaluate(f_pos, f_neg, f_test_nom, f_test_an, kind_test, weight=None, detector="ridge"):
    x = np.concatenate([f_pos, f_neg])
    y = np.r_[np.ones(len(f_pos)), np.zeros(len(f_neg))]
    sw = None if weight is None else np.r_[weight, np.ones(len(f_neg))]
    if detector == "ridge":
        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 13), class_weight="balanced")
    else:
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)
    clf.fit(x, y, sample_weight=sw)
    s_nom, s_an = clf.decision_function(f_test_nom), clf.decision_function(f_test_an)
    thr = float(np.quantile(s_nom, 0.99))
    hits = s_an > thr
    yy = np.r_[np.zeros(len(s_nom)), np.ones(len(s_an))]
    ss = np.r_[s_nom, s_an]
    return hits, {
        "recall@1%FAR": float(hits.mean()),
        **{f"recall {k}": float(hits[kind_test == k].mean()) for k in KINDS},
        "auroc": float(roc_auc_score(yy, ss)),
        "ap": float(average_precision_score(yy, ss)),
    }


def paired(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> dict[str, float]:
    d = a.astype(float) - b.astype(float)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {"diff": float(d.mean()), "lo": float(np.quantile(boot, 0.025)), "hi": float(np.quantile(boot, 0.975))}


GENS = ("posthoc", "unguided", "C1-like", "twin", "recipe", "recipe/rel", "recipe/rel+jitter", "recipe/rel+shift",
        "recipe/rel+div", "recipe/rel/bidir", "oracle")
VARIANTS = ("plain", "kNN filter", "weighted")
DETECTORS = ("ridge", "logistic")
ARMS = ("real_only",) + tuple(f"+{g}" + ("" if v == "plain" else f" | {v}") for g in GENS for v in VARIANTS)
PAIRS = (
    ("+recipe", "real_only"), ("+recipe/rel", "real_only"), ("+recipe/rel+div", "real_only"),
    ("+recipe/rel", "+recipe"), ("+recipe/rel+jitter", "+recipe/rel"), ("+recipe/rel+shift", "+recipe/rel"),
    ("+recipe/rel+div", "+recipe/rel"), ("+recipe/rel+div", "+twin"), ("+recipe/rel+div", "+C1-like"),
    ("+recipe/rel+div", "+posthoc"), ("+recipe/rel/bidir", "+recipe/rel"), ("+recipe", "+twin"), ("+recipe", "+C1-like"),
    ("+C1-like", "real_only"), ("+posthoc", "real_only"), ("+unguided", "real_only"), ("+twin", "real_only"),
    ("+oracle", "real_only"), ("+oracle", "+recipe/rel+div"),
    ("+twin | kNN filter", "+twin"), ("+unguided | kNN filter", "+unguided"),
    ("+recipe/rel+div | kNN filter", "+recipe/rel+div"), ("+oracle | kNN filter", "+oracle"),
    ("+recipe/rel+div | weighted", "+recipe/rel+div"), ("+oracle | weighted", "+oracle"),
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    per_rep: list[dict] = []
    hits_all: dict[tuple[str, str], list[np.ndarray]] = {(dt, a): [] for dt in DETECTORS for a in ARMS}
    kinds_all: list[np.ndarray] = []
    for rep in range(REPS):
        rng = np.random.default_rng(1000 + rep)
        real = few_shot_events(rng)
        x_real = np.concatenate([real[k] for k in KINDS])
        neg = ch.sample(N_NEG, W, rng)
        shell = Shell.from_nominal(enc, ch.sample(512, W, rng), ch.sample(512, W, rng))
        donors = neg[rng.permutation(N_NEG)[: N_PER_KIND * len(KINDS)]]  # training nominals, as in augdetect
        test_nom = ch.sample(N_TEST_NOM, W, rng)
        kind_test = np.repeat(np.array(KINDS), N_TEST_PER_KIND)
        test_an = np.concatenate([inject(k, ch.sample(N_TEST_PER_KIND, W, rng), positions(k, N_TEST_PER_KIND, rng),
                                         amplitudes(k, N_TEST_PER_KIND, rng)) for k in KINDS])
        seed = 7 + 100 * rep
        realized = {}
        synth = {"posthoc": posthoc_inject(donors, rng=np.random.default_rng(seed)).astype(np.float64),
                 "unguided": guided_ddim(DEN["causal"], donors, [], rng=np.random.default_rng(seed), nu=1.0),
                 "C1-like": c1_like(donors, real, seed=seed, shell=shell)}
        synth["twin"], _ = recipe_slices(DEN["flip_ramp"], donors, real, seed=seed, shell=shell, steer=False)
        synth["recipe"], realized["recipe"] = recipe_slices(DEN["flip_ramp"], donors, real, seed=seed, shell=shell)
        synth["recipe/rel"], realized["recipe/rel"] = recipe_slices(DEN["flip_ramp"], donors, real, seed=seed,
                                                                    shell=shell, relative=True)
        for name, kw in (("recipe/rel+jitter", {"jitter": (0.5, 1.5)}), ("recipe/rel+shift", {"shift_max": 16}),
                         ("recipe/rel+div", {"jitter": (0.5, 1.5), "shift_max": 16})):
            synth[name], realized[name] = recipe_slices(DEN["flip_ramp"], donors, real, seed=seed, shell=shell,
                                                        relative=True, **kw)
        synth["recipe/rel/bidir"], realized["recipe/rel/bidir"] = recipe_slices(DEN["bidir"], donors, real, seed=seed,
                                                                                shell=shell, relative=True)
        synth["oracle"] = oracle(donors, np.random.default_rng(seed))
        print(f"rep {rep}: generated ({time.time() - t0:.0f}s)", flush=True)

        space = RocketSpace(n_kernels=500, n_components=64, seed=rep).fit(neg[:1000])
        f_neg, f_tn, f_ta, f_real = (space.features(a) for a in (neg, test_nom, test_an, x_real))
        f_syn = {g: space.features(synth[g]) for g in GENS}
        # novelty filters, both calibrated on nominal training windows only
        nn = NearestNeighbors(n_neighbors=5).fit(space.embed_features(f_neg[:1500]))
        knn_thr = float(np.quantile(nn.kneighbors(space.embed_features(f_neg[1500:]))[0].mean(axis=1), 0.99))
        keep = {g: {"kNN filter": nn.kneighbors(space.embed_features(f_syn[g]))[0].mean(axis=1) > knn_thr} for g in GENS}
        print(f"rep {rep}: features and filters ({time.time() - t0:.0f}s)", flush=True)

        row = {"realized": realized, "pass_rate": {}, "n_real": int(len(x_real))}
        kind_syn = np.repeat(np.array(KINDS), N_PER_KIND)
        for g in GENS:
            row["pass_rate"][g] = {k: float(keep[g]["kNN filter"][kind_syn == k].mean()) for k in KINDS}
        for dt in DETECTORS:
            row[dt] = {}
            hits, row[dt]["real_only"] = evaluate(f_real, f_neg, f_tn, f_ta, kind_test, detector=dt)
            hits_all[(dt, "real_only")].append(hits)
            for g in GENS:
                for v in VARIANTS:
                    arm = f"+{g}" + ("" if v == "plain" else f" | {v}")
                    sel = keep[g]["kNN filter"] if v == "kNN filter" else np.ones(len(f_syn[g]), dtype=bool)
                    fs, wgt = f_syn[g][sel], None
                    if v.startswith("weighted") and len(fs):
                        # synthetic total weight = real total weight; the positive class keeps its total weight
                        n_pos = len(f_real) + len(fs)
                        c = n_pos / (2.0 * len(f_real))
                        wgt = np.r_[np.full(len(f_real), c), np.full(len(fs), c * len(f_real) / len(fs))]
                    hits, m = evaluate(np.concatenate([f_real, fs]), f_neg, f_tn, f_ta, kind_test, weight=wgt, detector=dt)
                    m["n_synth"] = int(len(fs))
                    row[dt][arm] = m
                    hits_all[(dt, arm)].append(hits)
            print(f"rep {rep} {dt} ({time.time() - t0:.0f}s): real_only {row[dt]['real_only']['recall@1%FAR']:.3f} | "
                  + " | ".join(f"{a} {row[dt][a]['recall@1%FAR']:.3f}" for a in ARMS[1:]
                               if " | " not in a or a.endswith("| weighted")), flush=True)
        kinds_all.append(kind_test)
        per_rep.append(row)

    rng = np.random.default_rng(0)
    kinds_cat = np.concatenate(kinds_all)
    summary = {dt: {} for dt in DETECTORS}
    for dt in DETECTORS:
        for arm in ARMS:
            keys = [k for k in per_rep[0][dt][arm] if isinstance(per_rep[0][dt][arm][k], (int, float))]
            summary[dt][arm] = {k: float(np.mean([r[dt][arm][k] for r in per_rep])) for k in keys}
            summary[dt][arm]["recall_sd_over_reps"] = float(np.std([r[dt][arm]["recall@1%FAR"] for r in per_rep]))
    pass_rate = {g: {k: float(np.mean([r["pass_rate"][g][k] for r in per_rep])) for k in KINDS} for g in GENS}
    realized = {g: {k: {m: float(np.mean([r["realized"][g][k][m] for r in per_rep])) for m in per_rep[0]["realized"][g][k]}
                    for k in per_rep[0]["realized"][g]} for g in per_rep[0]["realized"]}
    pairs = {dt: {} for dt in DETECTORS}
    for dt in DETECTORS:
        for a, b in PAIRS:
            ha, hb = np.concatenate(hits_all[(dt, a)]), np.concatenate(hits_all[(dt, b)])
            pairs[dt][f"{a} - {b}"] = {"all": paired(ha, hb, rng),
                                       **{k: paired(ha[kinds_cat == k], hb[kinds_cat == k], rng) for k in KINDS},
                                       "per_rep": [float((x.astype(float) - y.astype(float)).mean())
                                                   for x, y in zip(hits_all[(dt, a)], hits_all[(dt, b)], strict=True)]}
    res = {"setup": {"W": W, "reps": REPS, "n_synth_per_kind": N_PER_KIND, "n_real_train": 6 * len(KINDS),
                     "n_neg": N_NEG, "n_test_nominal": N_TEST_NOM, "n_test_per_kind": N_TEST_PER_KIND, "amp": AMP,
                     "detectors": "ROCKET 500 kernels (ppv, max) + RidgeClassifierCV (balanced) / "
                                  "LogisticRegression C=0.1 (balanced)",
                     "filter": "5-NN distance in ROCKET PCA-64 space to 1500 nominal training windows; "
                               "threshold = its 99th percentile on the other 500"},
           "summary": summary, "pairs": pairs, "pass_rate": pass_rate, "realized": realized, "per_rep": per_rep}
    (OUT / "e20_detect_utility.json").write_text(json.dumps(res, indent=1))
    for dt in DETECTORS:
        print(f"\n## {dt}\n\n| arm | n synth | recall @1% FAR | spike | level shift | drift | AUROC | AP |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for arm in ARMS:
            s_ = summary[dt][arm]
            print(f"| {arm} | {s_.get('n_synth', 0):.0f} | {s_['recall@1%FAR']:.3f} ± {s_['recall_sd_over_reps']:.3f} | "
                  f"{s_['recall spike']:.3f} | {s_['recall level shift']:.3f} | {s_['recall drift']:.3f} | {s_['auroc']:.3f} | {s_['ap']:.3f} |")
        print("\n| pair | Δ recall (95% CI) | spike | level shift | drift | per rep |")
        print("|---|---|---:|---:|---:|---|")
        for name, p in pairs[dt].items():
            a = p["all"]
            print(f"| {name} | {a['diff']:+.3f} [{a['lo']:+.3f}, {a['hi']:+.3f}] | {p['spike']['diff']:+.3f} | "
                  f"{p['level shift']['diff']:+.3f} | {p['drift']['diff']:+.3f} | {' '.join(f'{v:+.2f}' for v in p['per_rep'])} |")
    print("\npass rates (kNN filter):", json.dumps(pass_rate, indent=None)[:2000])
    print("realized sizes:", json.dumps(realized)[:2000])
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
