# `flip_x0_contrast_head_sharp`: what the model is, and how it differs from C1

The recipe recommended in [ESA_RUN_2026-10-06.md §10.3](ESA_RUN_2026-10-06.md).
It supersedes `flip_x0_contrast_head_kindpolish`, which is the same recipe
without the two switches of §10: `contrast_mirror`, which randomises which
end of the window the level shift sits on, and `contrast_geom`, which
narrows the spike's centre from 3 bins to the 1 bin the real excursions
actually occupy. Everything below applies to both unless noted.
It is not a new network: the denoiser, the encoder, the schedule and the
donors are all C1's. It is a different way of *steering* the same sampler.
This document is the spec-level description; §9 of the run document has the
evidence for why each piece is there.

## The three names in this file

| name | what it is |
|---|---|
| `c1_repro` | the earlier baseline recipe, reproduced here for comparison |
| `flip_x0_contrast_head_kindpolish` | the recipe this document describes in full, from "What both models share" to "Reproducing" |
| `flip_x0_contrast_head_sharp` | `..._kindpolish` plus the two switches in the last section |
| `flip_x0_contrast_head_sharp_nomask` | `..._sharp` with the RePaint mask dropped; **the current recommendation** (ESA_RUN §11.5) |

`..._sharp` is not a different model: it is the same chain with
`contrast_mirror: 0.5` and `contrast_geom: {spike_half: 0}` set. Everything
between here and "The two §10 switches" describes both.

## What both models share

Everything up to the guidance term is the frozen stack:

| | |
|---|---|
| denoiser | the S1 causal S4-D, trained on **nominal windows only** (`results/shell_s1/denoiser.pt`) |
| encoder | `time_both`, frozen at fold 0 |
| sampler | DDIM, `ddim_steps` 50, `nu` 0.2 (so t runs 39 → 0 of 200), `n_correct` 1 |
| donors | the 1536 S3 nominal windows, 256 per channel, W = 512 bins × 30 s = 4.27 h |
| allocation | `channel_stratified_alloc`, see below |
| recipe | `hybrid_needles`: the prototype term is enabled only for `real level shift` and `real ESA Point / Global`; the other kinds run the band term |
| scoring | the frozen S4 protocol (ARP, Coverage@τ, EDI, Div, τ), untouched |

Both are run by `anogen fixsweep` and both are scored against a **no-steer
twin** — the identical chain with every objective switched off — so the
reported ΔARP isolates what the steering did rather than what the donor was.

### `channel_stratified_alloc`: which donor is asked for which kind

Every gallery is 1536 windows and there are four target kinds, so each donor
has to be assigned one kind to aim at. `channel_stratified_alloc`
(`phases/kindmix.py`) does that assignment **inside each channel**: it shuffles
the channel's 256 donors, cuts them into four equal parts and deals one kind to
each part, rotating which kind gets the remainder as it moves to the next
channel. The result for a real gallery:

| channel | Point / Global | global subseq. | local subseq. | level shift |
|---|---|---|---|---|
| 41 … 46, each | 64 | 64 | 64 | 64 |
| **total** | **384** | **384** | **384** | **384** |

The alternative — the contiguous split this replaced — is what makes it worth
a name. S3 writes its donors out in per-channel blocks, so slicing that array
into four contiguous parts hands each kind a *disjoint set of channels*. Any
difference measured between kinds would then be partly a difference between
channels, and "the level-shift slice is rough" could just mean "channel 43 is
rough". Splitting within the channel removes that confound, and with equal
per-channel counts the totals come out identical to the contiguous split, so
nothing else about the protocol changes. [CHANMIX.md](CHANMIX.md) has the
measurement that motivated it.

It matters twice over for this model: the per-kind numbers in §9 of the run
document are only interpretable because every kind sees all six channels, and
`contrast_donor_room` is defined as a permutation *within* a channel precisely
so it cannot disturb this balance.

## C1 (`c1_repro`)

C1 is the config defaults with no overrides at all; its spec is literally
`{"twin": "nosteer_c1"}`. The steering is:

- **objective**: band term (`lam` 0.3) pushing the encoder energy to the
  quantile `Q_q`, plus, on the two needle kinds, a **prototype** term
  (`lam_anom` 1.0) pulling the window towards the embedding of a real anomaly
  of that kind, drawn event-out-of-fold;
- **gradient**: taken in **x-space** (through the denoiser Jacobian),
  unit-normalized, clipped at `c_max`, applied at full strength on every step;
- **extent**: the whole window is generated; there is no mask, so every bin
  can move;
- **final step**: the full gradient is subtracted (`final_grad_scale` 1.0).

## `flip_x0_contrast_head_kindpolish`

```python
{
    "flip": "ramp",
    "guidance_space": "x0",
    "final_grad_scale": 0.25,
    "contrast": {"real level shift": "step_head", "real ESA Point / Global": "spike"},
    "contrast_pos": {"real level shift": (0.0625, 0.18)},
    "contrast_polish": {"real level shift": 0.35},
    "contrast_only": True,
    "contrast_headroom": True,
    "mask": True,
    "mask_dilate": 8,
    "n_recur": 2,
    "guidance_decay": 1.0,
    "twin": "nosteer_flip_x0_masked",
}
```

### The seven differences, and what each one is for

**1. `contrast_only` — the objective is a linear constraint, not a prototype.**
The single biggest change. Instead of "move towards the embedding of a real
anomaly", the target is a scalar linear functional of the window,
d(x) = Σ w[t]·x[t], driven to a sampled amplitude δ. Each step adds the exact
projection λ(d − δ)·w/‖w‖², which lands d on δ at λ = 1. The band and
prototype terms are off entirely (`lam_anom` 0).

This is what removes the position collapse. A prototype carries the real
window's *position* along with its shape, and every real anomaly window in
ESA-ADB is cut centred on its event, so C1 reproduces that centring. The
contrast carries only a shape and an amplitude; the position is a free
parameter the recipe samples.

#### What d and w actually are

`x` is one window: 512 bins of one channel, in the denoiser's scaled units.
`w` is a fixed vector of 512 numbers built by
`shell/contrast.py::contrast_weights(kind, position, width)`, so
d(x) = Σ_t w[t]·x[t] is a single number — a weighted average of the window.
Nothing about w is learned or optimised; it is written down from the
morphology, and it always sums to zero, which makes d a **difference of
averages** and therefore blind to the window's overall level.

| kind | w[t] | d(x) reads |
|---|---|---|
| `spike` | +1/3 on the 3 bins at the position, −1/16 on the 8 bins each side (with a 3-bin gap), 0 elsewhere | how far the excursion stands above its own local baseline |
| `step` | +1/n on every bin after the split, −1/n on every bin before it | post-split mean minus pre-split mean |
| `step_head` | the mirror: + on the head, − on the tail | head mean minus tail mean |

![the contrast objective](contrast_explainer.png)

Two things are sampled per window, and they are the only freedom: the
**position** (uniform over the allowed range, which `contrast_pos` can narrow)
fixes *where* w's support sits, and the **amplitude** δ is resampled from the
values `measure_contrast` reads off the real anomalies of that kind, drawn
event-out-of-fold. So "a level shift of −0.81 starting at bin 37" is a
complete specification, and w is just how that sentence is written as a vector.

The edit follows from one line of algebra. Adding c·w to x changes d by
c·(w·w) = c‖w‖², so c = −(d − δ)/‖w‖² lands exactly on d = δ. That is the
projection λ(d − δ)·w/‖w‖² with λ = 1, applied to the denoiser's current x̂₀ at
every step. Under a mask the same algebra runs on w ⊙ mask, which keeps λ = 1
exact while confining the edit to the generated bins.

Because the edit is proportional to w, it is **piecewise constant**: for a step
it adds one constant to the head and another to the tail. It cannot by itself
make a segment rough — which is how §9 of the run document established that the
roughness came from the denoiser arguing with the constraint, not from the
constraint.

Note what d does *not* constrain. A step contrast fixes the shifted segment's
**mean** and says nothing about its variance, so a window with the right mean
and twenty times the real texture satisfies it perfectly. That is the whole
subject of §4b and §9.

**2. `contrast: {...}` — a different functional per kind.**
Each kind gets its own `w`, chosen so that d(x) *is* the quantity that defines
that kind of anomaly. The bottom panel of the figure above shows the spike's
support bin by bin.

`spike`, for Point / Global — 25 bins of support, 12.5 minutes:

| | bins (relative to position p) | weight each |
|---|---|---|
| left flank | p−12 … p−5 (8 bins) | −1/16 |
| gap | p−4 … p−2 (3 bins) | 0 |
| centre | p−1, p, p+1 (3 bins) | +1/3 |
| gap | p+2 … p+4 (3 bins) | 0 |
| right flank | p+5 … p+12 (8 bins) | −1/16 |

So d = (mean of 3 centre bins) − (mean of 16 flank bins): how far a 90-second
excursion stands above the baseline *immediately around it*. Not its absolute
value — a point anomaly is defined relative to its neighbourhood, and these
channels drift. The 3-bin gaps keep the spike's own rise and fall out of the
baseline estimate. The mask is `[p−4, p+4]`, the centre plus the gaps, so the
**flanks are never generated**: they stay donor, which is what stops the
constraint being satisfied by pushing the baseline down instead of the spike up.

`step_head`, for the level shift — support is the whole window:

| | bins | weight each |
|---|---|---|
| head | 0 … p−1 | +1/p |
| tail | p … 511 | −1/(512−p) |

So d = (mean of the head) − (mean of the tail): the difference between two
operating points, which is what a level shift *is*. `step` is the same with the
signs swapped, i.e. the tail as the shifted side; §9.3 of the run document is
about why the head is the right one here.

Both sum to zero, which is what makes them differences rather than levels, and
is why the same `w` works unchanged on a donor sitting at 0.1 of the channel
span and on one at 0.9.

C1 has nothing corresponding to this. Its per-kind difference is only *which*
prototype embedding it pulls towards (`kind_ref_indices` picks event-OOF
references of that kind) and, under `hybrid_needles`, whether the prototype
term is on at all. The loss itself — a distance in encoder space — is identical
for all four kinds. Nowhere does C1 state what a spike or a step *is* in the
time domain; the shape is whatever the encoder's embedding of the reference
happens to imply, and the position arrives as part of the same package.

**3. `mask` + `mask_dilate` 8 — a RePaint edit, not a whole-window rewrite.**
Only the bins the anomaly occupies are generated; the rest follow the noised
donor at every step and are restored exactly at the end. For `step_head` the
mask is `[0, p + dilate]`, i.e. the lifted head plus the transition, so the
whole tail is the donor verbatim. This is why the tail of a generated level
shift is as clean as the donor reconstruction while C1's whole window is
rewritten and comes out at 2.07× the real baseline texture.

The mask also becomes the **bin-level label** of the generated anomaly, which
C1 cannot provide — it has no notion of where the anomaly is.

**4. `guidance_space: "x0"` + `final_grad_scale` 0.25 + `flip: "ramp"` +
`n_recur` 2 + `guidance_decay` 1.0 — the position-bias fixes from
[POSITION_BIAS.md](POSITION_BIAS.md).**
The x0-space gradient skips the backward pass through the causal denoiser,
whose one-sided convolution is what concentrates gradient mass at the start of
the window; `flip: "ramp"` runs the donor through a time-reversed denoiser so
the unconstrained end is not always the same end; `n_recur` 2 is RePaint time
travel; `guidance_decay` 1.0 weights the *gradient* terms by σ_t (not the
contrast projection, which is weighted by `contrast_decay`, left at 0 here).

**5. `contrast_headroom` — the sampled amplitude is checked against the
parent.** The six real level-shift amplitudes are resampled onto donors that
may have no room for them; this flips the sign into the side that does. With
`step_head` it is nearly inert (the head lift already points into the roomy
direction), and it is kept as a guard for the one +3.987 outlier amplitude.

**6. `contrast_pos` — where the split may fall.** Restricts the level-shift
split to 0.0625–0.18 W. **This is the one place the model is deliberately less
free than C1, and it is an open question** — see the caveat below.

**7. `contrast_polish` 0.35 on the level shift only — an SDEdit pass.**
After the guided chain, re-noise to t = 0.35·(T−1) and run 10 **unguided**
DDIM steps back to 0 under the same mask, then re-apply the projection once so
the amplitude the polish removed comes back. The guided chain leaves the right
mean on the lifted stretch plus broadband residue from arguing with a score
trained only on nominal data; the free chain re-renders that residue as
nominal texture. It is per-kind because running it on the spike halves a
3-bin excursion (0.830 → 0.303); that measurement predates `spike_half: 0`,
but the reason it is per-kind is unchanged.

## Measured differences

Numbers are for `flip_x0_contrast_head_sharp`, 3 event-OOF folds, with
`..._kindpolish` alongside so the cost of the two §10 switches is visible.

### The frozen S4 protocol

Unchanged since S4; CIs are the two-way event-cluster bootstrap.

| | `..._head_sharp` | `..._head_kindpolish` | `c1_repro` | no-steer twin |
|---|---|---|---|---|
| ARP (anomaly) | 0.499 [0.258, 0.766] | 0.496 [0.256, 0.764] | **0.567** [0.373, 0.764] | 0.322 [0.146, 0.748] |
| ΔARP vs twin, frozen φ | **+0.177** [+0.018, +0.330] | +0.174 [+0.016, +0.332] | +0.136 [−0.018, +0.292] | — |
| ΔARP vs twin, standardized φ | +0.077 [−0.049, +0.195] | +0.073 [−0.050, +0.193] | — | — |
| Coverage@τ | **0.033** [0.002, 0.044] | 0.020 | 0.015 | 0.213 |
| ARP (rare) | 0.791 | 0.791 | 0.772 | 0.852 |
| ARP (ROCKET φ) | 0.024 | 0.021 | 0.031 | 0.015 |
| ROCKET C2ST AUC (0.5 = indistinguishable) | 0.881 | **0.876** | 0.966 | 0.763 |
| C2ST median / reject rate | 0.634 / 0.867 | 0.632 / 0.867 | 0.662 / 0.867 | 0.523 / 0.133 |
| Div | **9.68** | 8.66 | 7.30 | 1.69 |
| gap / occupancy | +0.008 / 0.662 | −0.004 / 0.668 | −0.008 / 0.432 | +0.030 / 0.038 |
| mean min distance (anomaly) | 1.044 | 1.053 | 0.860 | 2.426 |

### Morphology diagnostics

Project-specific, **not** part of the protocol — they answer "did the window
get the anomaly it was asked for", which ARP and C2ST do not. Reproduce with
`scripts/measure_morphology.py`.

| | real | `..._head_sharp` | `..._head_kindpolish` | `c1_repro` | `final025_recur2_decay1` |
|---|---|---|---|---|---|
| spike height | 0.831 | **0.988** | 1.040 | 0.447 | 0.496 |
| spike width at half max (bins) | 1 | **1** | 3 | 1 | 1 |
| share exactly 1 bin | 100% | 53% | 27% | 53% | **85%** |
| baseline texture vs real | 1.00× | **0.97×** | 0.97× | 2.00× | 1.08× |
| draws that are a noise burst | 0.0% | **0.0%** | 0.0% | 22.8% | 21.8% |
| spike position p10–p90 | 0.50–0.50 | 0.18–0.82 | 0.18–0.83 | 0.04–0.50 | 0.50–0.50 |
| level-shift roughness vs real | 1.00× | 1.53× | 1.55× | 1.70× | 1.58× |
| level shift at the head | 100%¹ | **50.1%** | 100% | 100% | 100% |
| level-shift amplitude | 0.328 | 0.569 | 0.568 | 0.297 | 0.303 |
| envelope exit fraction | 0.039 | 0.555 | 0.557 | 0.563 | 0.532 |
| bin-level anomaly label | — | yes (the mask) | yes | no | no |

¹ six windows from a single event, so this is that event, not a population.

### Detector utility (`augdetect`)

Event recall at 1% nominal FAR, event-OOF, 5 seeds, dev split only; the sealed
test stays closed. "noleak" drops C1's fold-0 level-shift prototype leak.
Paired differences bootstrap over events.

| arm | event recall | noleak | AP | rare FAR |
|---|---|---|---|---|
| `real_only` | 0.591 | 0.591 | 0.472 | 0.026 |
| `c1_plus_real` | 0.636 | 0.714 | 0.479 | 0.051 |
| `flip_x0_contrast_only+real` | 0.636 | 0.714 | **0.483** | 0.059 |
| `flip_x0_final025_recur2_decay1+real` | **0.727** | 0.714 | 0.472 | 0.088 |
| **`flip_x0_contrast_head_sharp+real`** | 0.545 | 0.643 | 0.470 | 0.036 |
| `nosteer_flip_x0_masked+real` (its twin) | 0.545 | 0.571 | 0.468 | 0.024 |

| paired difference | all events | noleak | fused with the diffusion score |
|---|---|---|---|
| vs its no-steer twin | +0.036 [−0.100, +0.173] p=0.32 | +0.071 [−0.014, +0.200] p=0.13 | +0.036 [+0.000, +0.082] p=0.04 |
| vs `real_only` | +0.018 [−0.100, +0.136] p=0.41 | +0.043 [−0.057, +0.186] p=0.32 | +0.036 [−0.009, +0.100] p=0.06 |
| vs `c1_plus_real` | **−0.064 [−0.136, −0.009]** | −0.071 [−0.171, +0.000] | **−0.064 [−0.127, −0.018]** |
| vs `flip_x0_contrast_only+real` | −0.073 [−0.164, +0.009] | −0.043 [−0.157, +0.057] | **−0.145 [−0.300, −0.027]** |

**This recipe does not improve detection** — but that was measured at a
single operating point where every arm already has all ~450 real training
anomalies. ESA_RUN §11 adds the curve over the positive budget k and finds
the gain is real at k ≤ 64; at k = 32 `..._head_sharp` is +0.109 [+0.009,
+0.227] and `..._sharp_nomask` is +0.200 [+0.072, +0.336]. Read this section
as the fixed-budget result, not the whole story.

At the frozen operating point: Against `real_only` the gain is
+0.018 with a CI straddling zero, and against C1 it is *significantly worse*,
−0.064 [−0.136, −0.009]. Only the comparison against its own no-steer twin is
positive, and only in the fused view. **No detection claim may be attached to
this recipe.**

The detection winner is `flip_x0_final025_recur2_decay1` at 0.727 — though
even that is only marginal against real data alone: +0.145 [−0.027, +0.327],
p = 0.053, and significant only when fused with the diffusion score. It is
also the recipe with 21.8% noise-burst draws and a fully collapsed spike
position, so on this data the two objectives disagree.

#### Why the two leaders differ: the edit-footprint hypothesis

The steering's edit can be read off exactly, because every variant has a
no-steer twin run with the same chain, seeds and masks: `x − x_twin` *is* the
edit. Share of bins moved by more than 0.05 in channel-span units:

| | whole gallery | Point/Global | level shift |
|---|---|---|---|
| `flip_x0_contrast_head_sharp` | 20.6% | **0.9%** | 14.0% |
| `flip_x0_final025_recur2_decay1` | 49.3% | **69.2%** | 68.9% |

A generated Point/Global from the recommended recipe differs from a plain
nominal donor in about **five bins out of 512**. That is the whole point of
the masked contrast design — it is why the morphology is right and why the
window comes with a bin-level label — but to a window-level detector such a
window is nearly a nominal one.

**Hypothesis: detector utility tracks how much of the window the steering
changes, not how faithfully the result matches the real morphology.** The
supporting observation is that `flip_x0_contrast_head_sharp+real` scores
0.545 event recall, *exactly* its no-steer twin's 0.545 — and the twin
contains no anomaly at all. `flip_x0_final025_recur2_decay1`, which rewrites
two thirds of every window, beats its own twin by +0.191 [−0.018, +0.400].

This is a hypothesis, not a result. Four things would test it, and the first
two can be run on the existing stack:

1. Widen the contrast support at fixed morphology (`step_span`, a longer
   masked segment, functionals for the two subsequence kinds). If the
   hypothesis holds, detection rises with footprint while the morphology
   diagnostics stay put.
2. Drop `contrast_only`, so the prototype term deforms the rest of the window
   as well. Detection should rise and C2ST should get worse.
3. Train on the union of the two galleries; if the mechanisms are
   complementary rather than competing, the union beats both.
4. Score **bin-level** localization rather than event recall. The contrast
   recipes are the only ones that emit a bin-level label, and if the
   hypothesis is right they should *win* that endpoint while losing this one.
   This is the measurement most likely to change the recommendation, and it
   is not implemented.

Numbers from `results/shell_augdetect/INDEX.txt`; the pre-§10 run is kept at
`results/shell_augdetect_pre_head/`.

## The level shift's position is pinned on purpose, and it is free when unpinned

`contrast_pos` puts the level-shift split at 0.0625–0.18 W because all six real
level shifts split at 0.072 W. By eye that makes every generated level shift
put its bump on the left, which looks exactly like the causal position bias.
It is not. Measured split positions:

| variant | position requested | achieved p05–p95 | median |
|---|---|---|---|
| real level shift | — | 0.072–0.072 | 0.072 |
| `flip_x0_contrast_head_kindpolish` | 0.0625–0.18 | 0.063–0.145 | 0.065 |
| `flip_x0_contrast_head_sharp` | 0.0625–0.18, half mirrored | 0.063–0.145 from *its own* end | 0.065 |
| **`flip_x0_contrast_head_wide`** | **0.0625–0.80** | **0.063–0.759** | **0.406** |
| `flip_x0_contrast_masked_room` (tail step) | 0.10–0.90 | 0.145–0.932 | 0.492 |
| `c1_repro` | **none** | 0.063–0.068 | 0.065 |
| `flip_x0_final025_recur2_decay1` | **none** | 0.063–0.070 | 0.067 |

C1 lands at the window start *without being asked to*, in a band 0.005 W wide
— that is the bias. `flip_x0_contrast_head_wide` is the recommended recipe
with nothing changed but the requested range, and it tracks the request across
the whole window. So the position is a free, controllable parameter and the
left-hand bump is the pin, not the bias.
[kind_grid_wide/real_level_shift.png](kind_grid_wide/real_level_shift.png)
shows it.

Widening it is not free, because a longer lifted stretch holds more bins
outside the score's conditional support:

| | level-shift roughness vs real | split position p05–p95 | spike |
|---|---|---|---|
| `flip_x0_contrast_head_sharp` | **1.53×** | 0.063–0.145, either end | 0.988 |
| `flip_x0_contrast_head_kindpolish` | 1.55× | 0.063–0.145 | 1.040 |
| `flip_x0_contrast_head_wide` | 2.0× | 0.063–0.759 | 1.040 |
| `c1_repro` | 1.70× | 0.063–0.068 | 0.447 |

Use the pinned recipe to match the real distribution for ARP / C2ST, and
`flip_x0_contrast_head_wide` for augmentation sets, where a detector must not
learn "level shifts happen at the start of the window". Point/Global is
identical either way, since `contrast_pos` only touches the level shift.

## Reproducing

```bash
anogen -c configs/shell_mission1.yaml fixsweep        # add `only:` to run a subset

# one recipe, all four kinds, real above generated
python scripts/plot_model_card.py -c configs/shell_mission1.yaml \
    --variant flip_x0_contrast_head_sharp

# one recipe, one kind, six panels wide so it can be judged by eye
python scripts/plot_model_card.py -c configs/shell_mission1.yaml \
    --variant flip_x0_contrast_head_sharp --only-kind "real level shift"

# several recipes compared within one kind
python scripts/plot_kind_grid.py -c configs/shell_mission1.yaml \
    --kind "real level shift" --out docs/kind_grid_best \
    --variants flip_x0_contrast_head_sharp c1_repro nosteer_flip_x0_masked

# the morphology diagnostics table above
python scripts/measure_morphology.py -c configs/shell_mission1.yaml \
    --variants flip_x0_contrast_head_sharp c1_repro

anogen -c configs/shell_mission1.yaml augdetect       # the detector-utility table
```

`plot_model_card.py` prints the whole-slice head/tail share in the level-shift
title, because a sample of a dozen draws from a 50/50 slice reads as biased
about a twentieth of the time.

Galleries land in `results/shell_fixsweep/<variant>_fold<f>.npz` with `x`,
`channel_idx`, `kind_alloc` and, for masked variants, the bin-level `mask`.

## The sampler, in equations

Both models run the same chain. Let ε_θ be the frozen S1 causal denoiser
(trained on **nominal** windows only), c the channel index, ᾱ_t the schedule,
and T = 200. Sampling starts at t₀ = ν(T−1) = 39 with ν = 0.2 — the donor is
noised only one fifth of the way, so what comes out is an *edit* of the donor,
not a fresh draw — and runs 50 DDIM steps to t = 0.

One step, with `guidance_space: "x0"`:

    ε    = ε_θ(x_t, t, c)                          (FlipEnsemble, below)
    x̂₀   = (x_t − √(1−ᾱ_t)·ε) / √ᾱ_t              DDIM's current guess of the clean window
    E    = the edit, see below                      in x̂₀ units
    x_{t'} = √ᾱ_{t'}·(x̂₀ − E) + √(1−ᾱ_{t'})·ε

`flip: "ramp"` replaces the first line with a FlipEnsemble, which is what
removes the causal denoiser's start-of-window bias without retraining:

    ε[i] = r[i]·ε_θ(x_t)[i] + (1−r[i])·flip(ε_θ(flip x_t))[i],   r[i] = i/(L−1)

`guidance_space: "x0"` means E is computed as a function of x̂₀ and subtracted
there, so no gradient is backpropagated through ε_θ. The last step applies
`final_grad_scale`·E instead of E.

### C1's edit: a gradient in encoder space

Let e(·) be the frozen `time_both` encoder and A the event-OOF embeddings of
the real anomalies of the window's assigned kind. C1 descends

    h_soft(z, A) = −τ · log Σ_{a∈A} exp(−‖z − e(a)‖² / τ),      z = e(x̂₀)

a soft minimum of the distance to the reference set, plus a band term that
pulls the window's energy toward the quantile Q_q. The edit is the normalized
gradient, λ = 0.3, clipped:

    E = λ · ∇_{x̂₀} h / ‖∇_{x̂₀} h‖,   clipped to c_max

Nothing in this says what a spike or a step *is*. The shape is whatever the
encoder's embedding of the reference implies — **and so is the position**,
because every real anomaly window in ESA-ADB is cut centred on its event.

### This model's edit: an exact projection onto a linear constraint

For the two kinds with a functional, λ_anom = 0: the encoder term is off
entirely. The objective is a scalar linear constraint

    d(x) = ⟨w, x⟩ = Σ_t w[t]·x[t]  →  δ,      Σ_t w[t] = 0

with w built from the morphology (`contrast_weights`) and δ resampled from
the real anomalies of that kind. The edit is the exact projection

    E = (d(x̂₀) − δ) · w / ‖w‖²

which lands on the target in one step, since

    d(x̂₀ − E) = ⟨w, x̂₀⟩ − (d − δ)·⟨w, w⟩/‖w‖² = d − (d − δ) = δ

A gradient step on f = (d − δ)² would give ∇f = 2(d − δ)w, i.e. the same
direction with step size η; the projection is that step at η = 1/(2‖w‖²), the
Newton step for a quadratic. Using it rather than a fixed η matters because
‖w‖² depends on where the contrast sits: for `step_head` at split p,

    ‖w‖² = 1/p + 1/(W − p)

which runs 0.0333 at p = 32 to 0.0133 at p = 92 over the allowed range alone.
A fixed step size would plant amplitudes differing by that factor; the
projection is scale-free.

Three things are sampled per window and they are the model's entire freedom:

    p ~ U(pos_range · W)            where the contrast sits
    δ ~ empirical{measure_contrast(kind, real anomalies)}   how big it is
    mirror ~ Bernoulli(0.5)         which end carries it (level shift only)

The remaining two kinds have no functional, so they keep C1's h_soft edit
unchanged.

### What `nomask` drops

`..._head_sharp` additionally runs RePaint: with a binary mask m,

    x_t ← m ⊙ x_t + (1 − m) ⊙ q(x₀^donor, t)      at every step
    x   ← m ⊙ x + (1 − m) ⊙ x₀^donor              at the end

so only the anomaly's own bins are ever generated and m is exportable as a
bin-level label. `..._sharp_nomask` removes both lines. The chain then evolves
the whole window, which is why the edit footprint goes from 0.9% of a
Point/Global window to 28.7% (ESA_RUN §11.3), and — unexpectedly — why the
morphology improves: the mask had been forcing a discontinuity at the seam
between generated and donor bins that the chain had to resolve inside the
masked region, and that was the source of the texture defect (§11.4).

## The two §10 switches

`contrast_mirror: {kind: frac}` time-reverses the contrast and its mask for a
random `frac` of the windows. A reversed `step_head` at split *p* is a `step`
at *W − p* with the same amplitude sign, so the shifted side moves to the tail
without touching the amplitudes or the headroom fit. At `frac = 0.5` the
splits land 50.1% head / 49.9% tail and every other measured quantity is
unchanged to three decimals.

`contrast_geom: {kind: {...}}` passes the contrast's shape parameters
(`spike_half`, `flank_gap`, `flank_len`, `step_span`) to `measure_contrast`,
`contrast_weights` and `contrast_mask` *together*, which is the point: the
reference amplitudes and the planted contrast have to be read in the same
units. `spike_half: 0` gives the spike a 1-bin centre. Both default to the
frozen geometry, so `test_defaults_reproduce_legacy` is unaffected.

Why 1 bin: all 18 real Point/Global excursions are exactly 1 bin wide at half
maximum, and three independently learned templates (matched filter, whitened
matched filter, logistic regression) all concentrate on one bin — see
[template_candidates_v2.png](template_candidates_v2.png) and ESA_RUN §10.2.
A 3-bin centre reads the real amplitude as 0.274 instead of 0.808.
