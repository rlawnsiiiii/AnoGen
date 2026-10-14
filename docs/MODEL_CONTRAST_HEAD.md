# `flip_x0_contrast_head_kindpolish`: what the model is, and how it differs from C1

The recipe recommended in [ESA_RUN_2026-10-06.md §9.4](ESA_RUN_2026-10-06.md).
It is not a new network: the denoiser, the encoder, the schedule and the
donors are all C1's. It is a different way of *steering* the same sampler.
This document is the spec-level description; §9 of the run document has the
evidence for why each piece is there.

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
3-bin excursion (0.830 → 0.303).

## Measured differences

| | this model | `c1_repro` |
|---|---|---|
| ARP | 0.496 [0.256, 0.764] | 0.567 [0.373, 0.764] |
| ΔARP vs no-steer twin, frozen φ | +0.174 [+0.016, +0.332] | +0.136 [−0.018, +0.292] |
| ROCKET C2ST AUC (0.5 = indistinguishable) | 0.876 | 0.966 |
| Div | 8.66 | 7.30 |
| draws that are a noise burst, not an anomaly | **0.0%** | 23.2% |
| Point/Global spike height (real 0.830) | 1.040 | 0.446 |
| baseline texture vs real | 1.17× | 2.07× |
| level-shift roughness vs real | 1.5× | 1.7× |
| spike position p10–p90 | 0.18–0.83 | 0.04–0.50 |
| bin-level anomaly label | yes (the mask) | no |

C1 is ahead on ARP by 0.071, inside a CI half-width of 0.25, and behind on
everything that measures whether the window looks like the thing it was asked
for.

## The level shift's position is pinned on purpose, and it is free when unpinned

`contrast_pos` puts the level-shift split at 0.0625–0.18 W because all six real
level shifts split at 0.072 W. By eye that makes every generated level shift
put its bump on the left, which looks exactly like the causal position bias.
It is not. Measured split positions:

| variant | position requested | achieved p05–p95 | median |
|---|---|---|---|
| real level shift | — | 0.072–0.072 | 0.072 |
| `flip_x0_contrast_head_kindpolish` | 0.0625–0.18 | 0.063–0.145 | 0.065 |
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
| `flip_x0_contrast_head_kindpolish` | **1.5×** | 0.063–0.145 | 1.040 |
| `flip_x0_contrast_head_wide` | 2.0× | 0.063–0.759 | 1.040 |
| `c1_repro` | 1.7× | 0.063–0.068 | 0.446 |

Use the pinned recipe to match the real distribution for ARP / C2ST, and
`flip_x0_contrast_head_wide` for augmentation sets, where a detector must not
learn "level shifts happen at the start of the window". Point/Global is
identical either way, since `contrast_pos` only touches the level shift.

## Reproducing

```bash
anogen -c configs/shell_mission1.yaml fixsweep        # add `only:` to run a subset
python scripts/plot_kind_grid.py -c configs/shell_mission1.yaml \
    --kind "real level shift" --out docs/kind_grid_best \
    --variants flip_x0_contrast_head_kindpolish c1_repro nosteer_flip_x0_masked
```

Galleries land in `results/shell_fixsweep/<variant>_fold<f>.npz` with `x`,
`channel_idx`, `kind_alloc` and, for masked variants, the bin-level `mask`.
