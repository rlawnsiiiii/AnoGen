# Next steps — 30.09.2026

Snapshot after the raw-window realism audit ([REALISM.md](REALISM.md)) and the
channel-stratified re-run ([CHANMIX.md](CHANMIX.md)). Numbers here are all
measured, not estimated; each row says where it came from.

---

## 0. Where we are

**Best model: C1 = `time_both` `hybrid_needles`, parent50** (λ=0.3, λ_anom=1,
λ_rare=0, 50 DDIM, unit ∇). Unchanged by anything in the audit — nothing
tested beat it, and the two things that scored better on realism did so by
generating less.

Report it under the **chanmix** allocation, not the frozen contiguous one. Same
model; the contiguous split gave each targeted kind its most favourable
channels, so its numbers are inflated.

| | ARP | EDI | Cov@τ | Div |
|---|---:|---:|---:|---:|
| C1 contiguous (frozen) | 0.584 | 2.60 \* | 0.039 | 7.52 |
| **C1 chanmix** | **0.565** | **2.52 c** | **0.022** | **7.33** |
| C2 chanmix (`time_recon`) | 0.557 | 2.50 c | 0.010 | 7.88 |

`time_both` still beats `time_recon` under the unconfounded allocation, so that
choice was not an artifact.

Why not the alternatives: **editor** wins ARP (0.630) but is a blender — it
keeps only the first 3 rFFT bins of the generation and takes all fast content
from the real donor, destroys level shifts outright, and drops EDI 2.38 → 1.83.
**hashfix** has the best C2ST (0.525) only because it barely deviates from the
donor: max envelope excess 0.23 against real anomalies' 4.02, ARP down to 0.459.
**timeleash_gm** is the most separable thing in the table (C2ST 0.838).

---

## 1. Confirmed limitations

Diagnostics in channel-span units, whole gallery, against real anomalies.

| | real | C1 chanmix | ratio |
|---|---:|---:|---|
| envelope window fraction | 0.039 | 0.575 | **15× too often** |
| envelope max excess | 4.02 | 4.14 | correct |
| CUSUM contrast | −0.005 | 1.13 | step that real lacks |
| first-difference p99.9 | 0.169 | 0.475 | **3× too sharp** |
| peak in first 10% of window | 0.114 | 0.470 | **4× over-concentrated** |

**L1 — Level-shift prototype is the worst slice and it is the steering, not the
data.** CUSUM 4.50, peak@start 0.917, envelope fraction 0.987, sharpness 0.869.
Every number is unchanged when the slice moves off channels 44–45 onto all six,
which is what chanmix proved. It is a quarter of the gallery. Visually it is one
stereotyped shape across 16 draws: a transient in the first 0.2–0.3 h that
overshoots the envelope, rings, then settles — not a shelf.

**L2 — Start-of-window position bias is in the sampler, not the steering.**
`unguided` shows it at 0.425 with no steering at all, and `hashfix` still shows
0.392 with the final gradient off. Neither λ nor `apply_final_grad` touches it.
This is the most dangerous defect for downstream use: a detector trained on C1
can learn "anomaly = window start", which will inflate augdetect and not
transfer.

**L3 — `apply_final_grad` is a binary between too violent and too tame.**
C1 (True): envelope 0.68, sharpness 0.52. hashfix (False): envelope 0.007,
sharpness 0.045. Real sits between both. The magnitude of C1's excursions is
already right; only their frequency and sharpness are wrong.

**L4 — C1 is trivially separable.** C2ST median balanced accuracy 0.795,
rejecting in 14 of 15 runs, against 0.49–0.51 for the injection baselines.

**L5 — Local vs global subsequence is not being targeted.** Both slices produce
downward needles. There are only 12 real local-subsequence windows in the whole
label set, so the distinction cannot be validated either way.

**L6 — Level-shift references leak on fold 0.** All six Mission-1 level shifts
sit in fold 0, so `kind_ref_indices` falls back to in-fold references and sets
`leaked=True`. The fold-0 level-shift prototype is built from the windows it is
scored against. Known and flagged in code, but it means fold-0 level-shift
numbers are not event-OOF.

**L7 — GenIAS baseline is not honest yet.** `genias_patched` is bit-identical to
the donor: at τ=0.2 the patch never fires against a maximum deviation of 3.1e-2.
Separately, our `deviation_patch` is **per timestep**, whereas GenIAS Algorithm 2
replaces the **entire trajectory** of a dimension when `‖X−X̃‖² > τ·amplitude`.
The criterion is dimensionally inconsistent in the paper too (units² vs units),
so τ does not transfer across preprocessing — at our scaling even τ=0.05 fires
on 0.07% of windows.

**L8 — No confidence intervals anywhere.** `bootstrap: 1000` has been in the
config since the audit spec was frozen and is still not wired into any metric.
We cannot currently say whether ARP 0.565 vs 0.584 is meaningful.

---

## 2. Next steps, in priority order

### P1 — Run the realism audit on chanmix *(minutes, no GPU)*

Point the audit's `c1` entry at `results/shell_chanmix/`. Until this is done,
`docs/COMPARISON.md`, `results/shell_realism/` and REALISM.md §17.1–17.7 all
describe the contiguous gallery, so the scorecard and the gallery we would
report are not the same thing. Cheapest item on the list by a wide margin.

### P2 — Fix the level-shift prototype *(L1, L6)*

Biggest single lever: 25% of the gallery, worst on four of five diagnostics,
and we know the cause is the steering. Hypotheses worth testing, cheapest first:

1. **Ramp λ_anom over t.** The overshoot-then-ring signature is what you expect
   when a large guidance gradient lands at high noise and later steps pull back.
   Try applying the level-shift proto only in the low-noise tail of the DDIM
   trajectory.
2. **The proto has 6 references and they leak on fold 0.** Six windows is
   extremely few-shot for a prototype. Check whether the proto energy is
   degenerate at that count before blaming the schedule.
3. **A step is broadband; the encoder may not require one.** The editor result
   showed that 3 rFFT bins cannot hold a shelf. If the encoder's level-shift
   prototype is satisfiable by an onset transient, the steering has no reason to
   produce a persistent step. Worth inspecting what the proto actually matches.

### P3 — Make the final gradient kick continuous *(L3, L4)*

Generalize `apply_final_grad` from `bool` to a float scale in [0, 1] in
`guided_ddim`, then sweep. C1 is 1.0 and hashfix is 0.0; real anomalies sit
between. One parameter, existing code path, directly targets envelope frequency
and edge sharpness without touching excursion magnitude, which is already right.

### P4 — Diagnose the sampler position bias *(L2)*

`unguided` has it too, so this is the backbone or the noising, not the steering.
A cheap discriminating test: measure peak@start on the from-noise galleries in
`results/shell_noise_score/` (ν=1, no donor) against the ν=0.2 galleries. If
from-noise also shows it, the bias is in the TSDiff backbone; if only ν=0.2
does, it is the donor/noising interaction and likely fixable in sampling.

### P5 — Wire bootstrap confidence intervals *(L8)*

The config already carries `bootstrap: 1000`. Without CIs, none of the small
differences in the comparison table can be called.

### P6 — Make the GenIAS baseline honest *(L7)*

Implement Algorithm 2 as published (per-dimension, whole-trajectory, squared L2
norm) as a separate function so our editor's per-timestep variant is unaffected,
then choose τ on our scaling and document that the paper's τ range does not
transfer. Note that for univariate windows the rule degenerates to "replace this
window or don't", which is worth stating rather than hiding.

### P7 — Re-check augdetect with the position bias in mind *(L2)*

Any augmentation gain from C1 needs a control for whether the detector learned
window position rather than anomaly morphology. Shuffling or cropping the
generated windows is a cheap ablation.

---

## 3. Things more compute will not fix

**Per-kind evidence barely exists.** Of 672 real anomaly windows: 636 global
subsequence, 18 Point/Global, 12 local subsequence, 6 level shift. Per fold
every non-global kind is at or below the `n_min = 10` floor, so the audit
produced published metric rows for `all` and global subsequence only. No
per-kind realism claim is supportable without more labelled events.

**The published distributional metrics do not rank generators.** The untouched
donor wins marginal W1, MDD and FFT-AWD against real anomaly windows, because a
512-bin window with a short fault span is mostly nominal. They are necessary,
not sufficient — good for catching artifacts, useless as a quality ranking. Only
C2ST and the diagnostics discriminate. Do not put MDD/ACD/W1/AWD in a table
without that caveat.

**MDD hides 39% of C1's mass.** TSGBench's definition discards generated values
outside the real per-timestep range. The `mdd_out_of_real_range_fraction`
diagnostic exists precisely so this is visible; quote it alongside MDD.

---

## 4. Claims the paper can and cannot support today

| Claim | Status |
|---|---|
| Comparable or superior realism vs GenIAS / post-hoc | Needs care — true on ARP, false on C2ST, and the GenIAS arm needs P6 |
| Nearly doubles diversity | Supported (EDI 2.52 vs 1.60 GenIAS, 1.95 post-hoc), but EDI unions must match |
| Generates from pure noise without degrading ARP/EDI | Supported by `shell_noise_score`, unaffected by this audit |
| **Type-targeted generation** | Only under chanmix. The frozen gallery gave each type its own channels, and the level-shift type does not produce a shelf |
| Realistic outputs | Qualified: C2ST 0.795 separable, envelope exits 15× too frequent |

---

Prepared 30.09.2026. Evidence: [REALISM.md](REALISM.md) §17,
[CHANMIX.md](CHANMIX.md), [COMPARISON.md](COMPARISON.md),
`results/shell_realism/`, `results/shell_chanmix/`.
