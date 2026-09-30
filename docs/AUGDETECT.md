# Downstream supervised augmentation (dev only)

Isolated `results/shell_augdetect/`. Does **not** overwrite
`results/shell_s3` or `results/shell_s4`. Sealed test / S6 stays closed.

```bash
.venv/bin/anogen -c configs/shell_mission1.yaml augdetect
```

Claim under test: adding generated anomalies to detector training
improves detection and reduces false alarms on real Rare Events.

---

## 0. Declared before running

- **Primary:** event-level recall at 1% nominal false-alarm rate.
- **Secondary:** AP, Rare-Event FAR at that same threshold, per-ESA-kind
  window recall.
- 22 anomaly events, ~7 per test fold. Paired bootstrap over **events**,
  not windows. Median over 5 seeds. Wide intervals and “no significant
  difference” are expected and will be reported as such.
- 636 of 672 dev anomaly windows are global-subsequence. Quiettune did
  **not** promote a texture recipe, so do not expect that kind to move
  the aggregate. Pre-registered per-kind breakdown is what keeps a
  Point/Global or shelf-only gain from looking like cherry-picking.

---

## 1. Setup

Detector: `AnomalyClassifier` 1-D CNN from
[`src/anogen/shell/adapters.py`](../src/anogen/shell/adapters.py), trained
by [`src/anogen/shell/detector.py`](../src/anogen/shell/detector.py)
(does **not** edit S5 `train_classifier`). Min-max scaled with
`results/shell_s0/minmax_scaler.npz`. No per-window z-score (that would
erase level-shift DC).

Train fold \(k\) held out:

| Arm | Train positives |
|---|---|
| `real_only` | all train-fold real anomaly windows |
| `posthoc` / `taxonomy` / `cutaddpaste` / `genias` / `unguided` | \(N=256\) synthetic, \(k=0\) real |
| `genfsdiff_c1` | \(N=256\) from `time_both_hybrid_needles_fold{k}` |
| `genfsdiff_c3` | \(N=256\) from `time_recon_combined_fold{k}` |
| `c1_plus_real` / `c3_plus_real` | train-fold reals **plus** \(N=256\) |

Train negatives: S3 donors with `index % 3 != k`, plus train-fold Rare
Events. Test nominals: `index % 3 == k` (disjoint). Test positives:
fold-\(k\) real anomaly windows. 3 folds × 5 seeds.

Leakage:

- `genfsdiff_*` on test fold \(k\) loads only the `fold{k}` gallery.
- Level-shift proto **leaks fold 0**. C1 fold-0 runs are tagged
  `leaked` and the no-leak median drops them. They are not averaged in
  silently.
- Train and test event IDs are asserted disjoint.

---

## 2. Results

150 runs (10 arms × 3 folds × 5 seeds). Primary number is **median
event recall over seeds**, with each seed pooling the 22 OOF events.
Whiskers in the plot are min–max over seeds, not a CI. 0.591 = 13/22
events; 0.636 = 14/22.

| Arm | Event rec. | no-leak | min–max | AP | Rare FAR | Window rec. |
|---|---:|---:|---:|---:|---:|---:|
| `real_only` | 0.591 | 0.591 | 0.45–0.64 | 0.470 | 0.034 | 0.151 |
| `posthoc` | 0.591 | 0.591 | 0.45–0.64 | 0.475 | 0.037 | 0.153 |
| `taxonomy` | 0.364 | 0.364 | 0.36–0.41 | 0.363 | 0.020 | 0.064 |
| `cutaddpaste` | 0.409 | 0.409 | 0.36–0.41 | 0.364 | 0.020 | 0.066 |
| `genias` | 0.409 | 0.409 | 0.27–0.41 | 0.363 | 0.019 | 0.065 |
| `unguided` | 0.136 | 0.136 | 0.00–0.18 | 0.306 | **0.006** | 0.029 |
| `genfsdiff_c1` | 0.591 | **0.643** | 0.55–0.64 | 0.449 | 0.049 | 0.100 |
| `genfsdiff_c3` | 0.591 | 0.591 | 0.55–0.64 | 0.431 | 0.042 | 0.081 |
| **`c1_plus_real`** | **0.636** | **0.714** | 0.59–0.68 | **0.486** | 0.039 | 0.155 |
| **`c3_plus_real`** | **0.636** | 0.636 | 0.64–0.64 | 0.483 | 0.030 | 0.152 |

C1 fold 0 is the level-shift proto leak (5 of 15 C1 runs). The no-leak
column drops those runs.

Per-kind **window** recall (mean over runs):

| Arm | Point/Global | global subseq | local subseq | level shift |
|---|---:|---:|---:|---:|
| `real_only` | 0.10 | 0.15 | 0 | 0.60 |
| `posthoc` | 0.14 | 0.15 | 0 | **0.80** |
| `genfsdiff_c1` | **0.20** | 0.10 | 0 | 0.17 |
| `c1_plus_real` | 0.17 | 0.16 | 0 | 0.57 |
| `c3_plus_real` | 0.18 | 0.15 | 0 | 0.20 |

Local subsequence is 6 windows total — 0 for every arm.

### How to read it

- **No significant augmentation win.** `c1_plus_real` / `c3_plus_real`
  are +1 event vs `real_only` (14/22 vs 13/22). Seed ranges overlap.
  With 22 events that is the expected “wide interval, no significant
  difference.”
- **C1/C3 as the only positives match `real_only`** on event recall
  (0.591). CutAddPaste / taxonomy / GenIAS as the only positives drop
  to 0.36–0.41. Unguided is 0.14. The diffusion galleries are a
  better *substitute* for real positives than the cheap operators;
  they are not a proven *add-on*.
- **Rare-Event FAR.** Unguided is the most conservative (0.006) because
  it barely fires. `c3_plus_real` is 0.030 vs `real_only` 0.034 —
  not a real reduction. C1 synth-only is a bit worse (0.049).
- **Per-kind.** Post-hoc’s step family is the shelf detector (level-shift
  window recall 0.80). C1 synth-only is the needle detector
  (Point/Global 0.20 vs 0.10). C3 mixed into real positives *dilutes*
  shelves (0.20 vs 0.60) because combined \(f\) does not emit them.
  Global subsequence, 636/672 windows, does not move.

Plots: [`event_recall.png`](augdetect/event_recall.png),
[`rare_far.png`](augdetect/rare_far.png),
[`per_kind_recall.png`](augdetect/per_kind_recall.png).
CSV: `results/shell_augdetect/metrics.csv`.
