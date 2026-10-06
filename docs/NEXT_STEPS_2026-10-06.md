# Next steps on the workstation (PyTorch + ESA data), 06.10.2026

Everything on branch `claude-fixes` (bundle `fixes/position-bias-and-guidance`)
was written and tested in a sandbox **without PyTorch and without ESA data**.
The numpy parts are tested; the torch parts were desk-checked only. Every
change is an opt-in switch with the frozen behaviour as the default, and every
phase below writes into its own `results/` directory, so no frozen artifact is
touched. `git checkout main` returns to the frozen state.

Background: [POSITION_BIAS.md](POSITION_BIAS.md) (root causes and switches),
[TESTBED.md](TESTBED.md) (the evidence, E1–E20d), the GenFSDiff project doc
"GenFSDiff: what is wrong and how to fix it" (theory and decision rule).

The order matters: steps 1–6 need no retraining and decide what the paper can
claim; step 7 is the one retrain. Run times are not known yet; the ones given
are guesses from the earlier docs.

---

## 0. Set up

```bash
git checkout claude-fixes
uv sync --extra neural            # torch; the numpy parts work without it
```

The ESA panel path and splits come from `configs/shell_mission1.yaml`, as before.

## 1. Tests, with torch (minutes)

```bash
.venv/bin/pytest -q
```

Must pass, and only run with torch:

| Test | Guards |
|---|---|
| `tests/test_steer_equivalence.py::test_defaults_reproduce_legacy` | with every switch off the sampler is bit-for-bit the frozen one |
| the rest of `tests/test_steer_equivalence.py` | masks, recurrence, chunking, flip ensemble, bidirectional backbone, geometric schedule |
| `tests/test_train_denoiser_keys.py` | the S1 retrain bug fixed earlier (schedule key), v-prediction and EMA checkpoints |
| `tests/test_shell_augdetect.py` | `real_frac` batch composition, the detector |

If `test_defaults_reproduce_legacy` fails, stop: the frozen numbers would no
longer be reproducible. Send the output.

## 2. Confirm the diagnosis (≈ 5 min, no generation)

```bash
.venv/bin/python scripts/diagnose_causality.py -c configs/shell_mission1.yaml
```

Writes `results/shell_diagnose/causality.json`. Expect:

- `backbone causal: True`;
- ε-MSE in the first 16 bins well above the last 16;
- for unguided and C1: start share ≫ 0.10 and end share ≪ 0.10 (real ≈ 0.11 / 0.10).

If the end share is not low, the causal explanation is wrong and steps 4–7
need rethinking before anything else runs.

## 3. Re-score the saved galleries (minutes, numpy)

```bash
.venv/bin/python scripts/audit_metrics.py -c configs/shell_mission1.yaml
```

Writes `results/shell_metric_audit/TABLE.md`: donor, unguided and oracle rows,
density and coverage, ARP in frozen φ, standardized φ and φ without the
high-frequency bands, and the ROCKET C2ST AUC (event-grouped). Questions it
answers:

- Does C1 still beat unguided and the donors once ARP is computed without the
  texture bands? (Testbed E12: ARP in frozen φ is mostly a texture score.)
- Is C1 separable from real anomalies in the ROCKET C2ST (AUC ≫ 0.5), and
  by how much more than the oracle row?

## 4. Zero-retrain sweep (the longest step)

```bash
.venv/bin/anogen -c configs/shell_mission1.yaml fixsweep
```

23 variants × 3 folds. To get the important ones first, add to
`shell.fixsweep` in the config:

```yaml
only: [c1_repro, nosteer_c1, flip_x0_final025_recur2_decay1, nosteer_flip_x0_recur2,
       flip_x0_contrast_masked, flip_x0_contrast_masked_div, nosteer_flip_x0_masked]
```

then remove `only` and rerun for the rest (cached variants are reused).
Read `results/shell_fixsweep/TABLE.md` against the decision rule in
POSITION_BIAS.md §7:

- start share ≤ 0.20 and end share ≥ 0.05;
- envelope exit fraction ≤ 3× real, with the 95th percentile of exit size
  ≥ 50 % of C1's;
- diff p99.9 ≤ 1.5× real (C1: 2.8×);
- paired ARP vs `c1_repro`: 95 % CI lower bound > −0.02;
- "What steering adds": paired ARP against the no-steer twin > 0 in frozen
  and standardized φ (otherwise the paper claims type targeting, not realism).

The galleries feed step 6.

## 5. Label-free detection (diffdetect)

```bash
.venv/bin/anogen -c configs/shell_mission1.yaml diffdetect
```

Default variants: `causal_skip32`, `causal_full`, `flip_ramp`,
`flip_ramp_top16_multiscale` (top-16 bins, six noise levels, z-max). Writes
`results/shell_diffdetect/{summary.json, <variant>_scores.npz, fresh_nominal.npz}`.
Expect the multiscale variant to beat `causal_full` on spikes and drifts
(testbed E15/E15b); level shifts stay hard for any denoising score.

## 6. Supervised detection, the headline (augdetect)

In `shell.augdetect` of the config, uncomment `fuse_diffscore`, `arms` and
`compare` (all are prepared there), e.g.:

```yaml
fuse_diffscore: results/shell_diffdetect/flip_ramp_top16_multiscale_scores.npz
arms: [real_only, "genias+real", "posthoc+real", "unguided+real", c1_plus_real,
       "fixsweep:flip_x0_contrast_masked+real", "fixsweep:flip_x0_contrast_masked_div+real",
       "fixsweep:nosteer_flip_x0_masked+real"]
compare:
  - ["fixsweep:flip_x0_contrast_masked_div+real", "genias+real"]
  - ["fixsweep:flip_x0_contrast_masked_div+real", c1_plus_real]
  - ["fixsweep:flip_x0_contrast_masked_div+real", "fixsweep:flip_x0_contrast_masked+real"]
```

```bash
.venv/bin/anogen -c configs/shell_mission1.yaml augdetect
```

`results/shell_augdetect/INDEX.txt` lists event recall at 1 % FAR per arm and
the paired differences A − B (bootstrap over the 22 events, seeds averaged):
on all folds, on the no-leak folds, and on the score fused with step 5. Every
fixsweep arm is paired with its no-steer twin automatically.

What the paper may claim (declared before the run):

| Claim | Needs |
|---|---|
| generated anomalies improve detection | recipe `+real` − `real_only`: 95 % interval above 0 |
| steering improves detection | recipe − its no-steer twin: above 0 |
| better than GenIAS | recipe − GenIAS `+real`: above 0 |
| no position shortcut | flipped and rolled recall close to plain recall |

Check each on the no-leak folds too: fold 0 holds the only level-shift event,
and the fixsweep galleries leak it there. With 22 events one event is 4.5
points of recall, so read the intervals, not the medians.

Optional runs on the same arms:

- **Fair GenIAS** (published objective): `anogen geniasfair` first, then add
  `"geniasfair:genias_fair+real"` to `arms` and compare against it. `genias+real`
  is the S3 GenIAS gallery.
- **Novelty filter**: `synth_filter: {kind: rocket_knn, q: 0.99}`. Report
  `synth_pass_rate_mean` per arm (the share of generated windows outside the
  nominal set). In the testbed it rescued near-nominal galleries and was
  neutral for the recipe; keep the unfiltered run as the main one.
- **Real share per batch**: `real_frac: 0.5`. No effect with BCE in the
  testbed; worth one run.

Testbed expectation (E20d, 20 repetitions, BCE detector): the recipe adds
+4.5 points [2.7, 6.4] over real anomalies, C1-like positives add nothing,
the no-steer twin costs 17 points, perfect positives would add 23. The ESA
magnitudes will differ; the signs are what to check.

## 7. One retrain, only if step 4 leaves texture or edge problems

```bash
.venv/bin/python scripts/schedule_check.py -c configs/shell_mission1.yaml --standardize
```

Set `diffusion_v2.sigma_min` to the largest value that keeps 95 % of the
texture on every channel and check that `sigma_max` covers the printed bound.
Then uncomment `diffusion_v2` and `scaler_v2` in the config:

```yaml
shell:
  diffusion_v2: {sigma_min: <from schedule_check>, sigma_max: 10.0, ema_decay: 0.999}
  scaler_v2: {kind: standard}     # per-channel z-score, S1 only
```

```bash
.venv/bin/anogen -c configs/shell_mission1.yaml s1v2     # bidirectional + log-spaced, results/shell_s1_v2
```

Check `results/shell_s1_v2/summary.json` and `loss_by_t.png` against the frozen
S1 (mid-noise bins must not be worse). Then add under `shell.fixsweep.variants`

```yaml
v2_x0: {denoiser: results/shell_s1_v2/denoiser.pt, guidance_space: x0, final_grad_scale: 0.25,
        n_recur: 2, guidance_decay: 1.0, burnin: true, context: both}
```

and rerun fixsweep (step 4). For detection, add the checkpoint as a diffdetect
variant (`denoiser: results/shell_s1_v2/denoiser.pt`, `pool: top16`,
`fuse: zmax`, `t_eval: [10, 20, 40, 60, 100, 150]`, no flip) and rerun steps
5–6. `parameterization: v` is available but brought no gain in the testbed
(E16); use it only if from-noise samples still show a level bias.

## 8. After the runs: where the work is

The testbed puts most of the detection headroom in the anomaly kinds that are
still generated from prototypes (E20: perfect positives beat the recipe by
21 points on level shifts and 40 on drifts, 0 on spikes). On ESA those are the
global and local subsequences, 18 of 22 events. If step 6 confirms that,
the next generator work is learning those kinds (the doc's Tier 2: masked
kind tokens à la AnomalyDiffusion, or a few-shot adapter à la FaultDiffusion,
evaluated on held-out events only), not more steering terms.

## Results

Steps 1–6 ran on 06.10.2026; step 7 was not run (`schedule_check` output and
the reason are in §7 there). Numbers, verdicts against the decision rule and
the five follow-ups: [ESA_RUN_2026-10-06.md](ESA_RUN_2026-10-06.md).

## What to send back

- the pytest summary (and full output of any failure);
- `results/shell_diagnose/causality.json`;
- `results/shell_metric_audit/TABLE.md`;
- `results/shell_fixsweep/TABLE.md` and `summary.json`;
- `results/shell_diffdetect/summary.json`;
- `results/shell_augdetect/INDEX.txt` and `summary.json`;
- for step 7: `results/shell_s1_v2/summary.json`.
