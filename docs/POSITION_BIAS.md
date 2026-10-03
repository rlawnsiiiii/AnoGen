# Position bias, sharp edges and envelope exits: root causes and fixes

Written 03.10.2026. Addresses L1–L4 and P2–P4 of
[NEXT_STEPS_2026-09-30.md](NEXT_STEPS_2026-09-30.md). Mechanism evidence is
from the torch-free testbed ([TESTBED.md](TESTBED.md)); the ESA numbers still
have to come from `fixsweep` (§5). Every switch below defaults to the frozen
behaviour, and `tests/test_steer_equivalence.py` checks bit-for-bit
equality of the refactored sampler with the old one.

---

## 1. The backbone is causal

`shell/tsdiff.py` builds each S4-D layer as a one-sided convolution:

```
y[t] = Σ_{s ≤ t} K[t − s] u[s] + D u[t]          (rfft/irfft, nfft = 2L, keep [:L])
```

Every other op in `TSDiffBackbone` is pointwise in time (1×1 convs,
LayerNorm over channels, the time-embedding FiLM). So

```
ε_θ(x_t)[i]  depends only on  x_t[0 .. i].
```

A denoiser has the whole noisy window; nothing about the task is causal. The
restriction has three consequences, all of which match symptoms already in
the repo.

**(a) The first bins are the worst-estimated.** At bin 0 the network sees one
noisy sample. In the testbed the best possible causal estimator has 5.7× the
x̂₀ error of a bidirectional one in the first 16 bins at the edit time t = 40,
and the same error in the last 16 bins.

**(b) Unguided sampling piles artefacts at the window start and none at the
end.** In the testbed, with exact Gaussian denoisers and no steering, the
dominant peak lands in the first 10 % of the window in **0.61** of samples at
ν = 1 and **0.41** at ν = 0.2. In the last 10 % it lands in 0.006 and 0.014.
The bidirectional estimator gives 0.10 / 0.07. The repo measured 0.425 for
unguided and 0.392 for hashfix (the final gradient is off there, so this is
the sampler alone). L2 is therefore the backbone, not the steering, as
NEXT_STEPS suspected.

**(c) Every x-space guidance step is smeared toward the start.** Guidance
takes `∇_{x_t} f(x̂₀(x_t)) = J^T ∇_{x̂₀} f`, and `J = ∂x̂₀/∂x_t` is lower
triangular for a causal network. A gradient that asks for a change at bin
256 is written onto bins ≤ 256. In the testbed it lands with its centre of
mass at **25 %** of the window instead of 50 %. A gradient that is flat in
time puts 34 % of its energy into the first 10 % of the window, against 13 %
with a bidirectional estimator. Unit normalization then hands those early
bins the whole step budget. This is the most likely reason the level-shift
prototype produces an early transient (peak@start 0.917, CUSUM 4.5) rather
than a shelf.

## 2. The kick is constant and the last one is not denoised

`guided_ddim` subtracts `λ · unit(∇)` (clipped to ±c_max) at **every** step,
whatever the noise level. The last step writes `x̂₀ − λ·unit(∇)` with no
further denoising. With unit normalization the per-step magnitude never
shrinks as the constraint is met. The last raw kick is a 0.3-norm vector
that can sit on a handful of bins, which matches C1's edges being 3× too
sharp (diff p99.9 0.475 vs 0.169) and the envelope being exited 15× too
often.

hashfix showed the binary version of this (L3): turning the last kick off
drops envelope exits from 0.68 to 0.007. The testbed reproduces the
sharpness ratio. On the level-shift slice, diff p99.9 is 0.163 under the
repo sampler against 0.050 for the target, and 0.064 with the last edit
scaled to 0.25.

## 3. Few prototypes, one position

`chunked_guided_ddim` draws one prototype per sample; the level-shift slice
has 6 references for 384 samples, so ~64 samples chase the same embedding,
and with a temporal encoder (`time_both`, 8 × 64) that embedding fixes *where*
the event is. NEXT_STEPS P2.1 is right that this is over-determined.

## 4. From noise starts off the training distribution

`DiffusionSchedule.linear(200)` with β ∈ [1e-4, 2e-2] ends at
ᾱ_T = 0.132, so q(x_T | x₀) = N(0.364·x₀, 0.868·I). The denoiser never sees
a zero-mean input at t = T−1, but `start_from_noise=True` feeds it
N(0, I) (the terminal-SNR flaw of Lin et al., WACV 2024). The level offset
this creates is invisible to φ, which z-scores every window, so the "from
noise without degrading ARP/EDI" result cannot see it. In the testbed the
generated level is biased by −0.2 to −0.3 nominal standard deviations. The
same fact means "unguided ν = 1" keeps 36 % of the donor amplitude, so it
is still a donor edit and not a draw from the prior.

## 4b. The schedule cannot reproduce fine texture

The linear schedule's smallest noise level is σ = 0.01 in scaled units, and
it crosses the low-noise range in a few steps. With *exact* denoisers,
deterministic DDIM then reproduces almost all of the total variance of a
Gaussian telemetry model but only 36–68 % of its first-difference variance,
at any step count (testbed E10, closed form). ESA shows the same signature:
unguided first-difference p99.9 is 0.71× that of the donors, and shell ZS
and hashfix are 0.75–0.77×. Two consequences:

- every DDIM gallery is too smooth before any steering, and
- that texture is what dominates φ, so ARP is sensitive to it (E7, REVIEW §2.1).

A log-spaced schedule (σ 1e-3 → 10) reproduces 89–97 % of the texture and
puts the terminal SNR near zero (fixing §4 too). It needs an S1 retrain.

---

## 5. Fixes (all opt-in; defaults reproduce the frozen runs)

| Switch | Where | What it does | Retrain? | Targets |
|---|---|---|---|---|
| `diffusion.bidirectional: true` / `anogen s1bidir` | `tsdiff.S4D` | second S4-D kernel on the time-reversed input, summed (standard bidirectional S4) | **yes** (S1, into `results/shell_s1_bidir`) | 1a–c at the root |
| `anogen s1v2` (`diffusion.schedule: geometric`, `sigma_min/max`, + bidirectional) | `diffusion.DiffusionSchedule.geometric`, `from_ckpt` | log-spaced noise levels; checkpoints record their schedule and every phase rebuilds it from the checkpoint; fixsweep keeps the frozen edit's *noise level* (ν is a fraction of the schedule) | **yes** (S1, into `results/shell_s1_v2`) | 4, 4b, and 1 at the root |
| `flip: ramp` (fixsweep) | `diffusion.FlipEnsemble` | ε = w⊙ε_θ(x) + (1−w)⊙flip(ε_θ(flip x)), w = i/(L−1) | no | 1a–c, assumes nominal ≈ time-reversible |
| `guidance_space: x0` | `steer.guided_ddim` | ∇ w.r.t. x̂₀, edit x̂₀, re-noise with the same ε (MPGD-style); no backward pass through the denoiser | no | 1c, and halves guidance cost |
| `final_grad_scale: 0.25` | `steer.guided_ddim` | continuous version of `apply_final_grad` (P3) | no | 2 |
| `burnin: true` (+`burnin_bins`, 256) | `steer.guided_ddim`, `fixsweep.donor_prefix` | the chain runs on the real preceding 256 bins + the window; objectives see only the window | no | 1a–b for a causal backbone |
| `proto_shift_max: 16` | `steer.shift_time_embedding` | per-sample time shift of the prototype embedding (edge-replicate, ±16 encoder steps = ±128 bins) | no | 3 |
| `lam_repel` (+`repel_project`) | `steer._repulsion` | particle-guidance RBF repulsion inside a batch, projected off the band gradient (P2a) | no | 3 |
| `contrast: {kind: step\|spike}` (+`contrast_only`) | `shell/contrast.py`, `steer.guided_ddim` | projection onto x̂₀·w = δ (step: whole-window mean difference; spike: centre vs flanks) at a uniformly sampled position, δ resampled from the train-fold examples; no prototype, no position collapse | no | 3, L1, L5 |
| `noise_init_mean/std` (`matched_noise: true`) | `steer.guided_ddim` | x_T ~ N(√ᾱ_T m_c, (1−ᾱ_T+ᾱ_T s_c²)I) | no | 4 |
| `guidance_t_window` | `steer.guided_ddim` | guidance only on part of the trajectory (P2.2) | no | 2 |

Not adopted: noise-space (classifier-guidance) steering
`ε' = ε + √(1−ᾱ)·∇`. In a DDIM chain whose consecutive times are close, the
x̂₀ shift and the ε shift cancel to first order. With ν = 0.2, many repeated
times and unit gradients the steering is inert: 0 % shelves at 1× and 3×
strength in the testbed. It is left in the testbed only to document that.

What the testbed says to expect, in order of cost:

1. **Zero-retrain:** `flip: ramp` + `guidance_space: x0` +
   `final_grad_scale: 0.25` fixes the start bias of the sampler (0.41 → 0.15
   at ν = 0.2, unguided) and of guidance (level-shift slice 0.27 → 0.04). It
   brings edge sharpness to within 1.2× of the target (0.163 → 0.058 against
   0.050). The testbed's envelope numbers are not informative, because its
   step sizes are synthetic, so judge envelope frequency on ESA only.
2. **Retrain S1 bidirectional** (`anogen s1bidir`, same 20k steps): the
   clean version of the same fix. In the testbed it is the only option with
   start ≈ end ≈ 0.10 in every slice and no time-reversibility assumption.
3. `burnin` (256 bins) is an alternative to `flip` for parent starts. 128
   bins is not enough (0.35 / 0.18 start share).
4. The step contrast (testbed E6) is the stronger answer to the six
   shared prototypes. It reproduces the real step-position distribution
   (W1 0.021 vs 0.076 for the prototypes) and amplitude at the right edge
   sharpness, with somewhat fewer clean shelves (0.75 vs 0.90 real). Run it with and without
   the prototype (`flip_x0_contrast`, `flip_x0_contrast_only`).
5. `proto_shift_max` widens the step-position spread toward the real one
   (sd 0.20 → 0.23, real 0.24), at the cost of a lower shelf rate. On ESA,
   judge it on `cusum_position` / `signed_peak_position`, never on EDI, which
   is position-blind.

---

## 6. How to run (workstation, in this order)

```bash
# 0. 5 min, no generation: confirm the diagnosis on the real checkpoint and galleries
.venv/bin/python scripts/diagnose_causality.py -c configs/shell_mission1.yaml
#    expects: backbone causal = True; eps-MSE first16 >> last16;
#    unguided / hashfix / C1: start >> 0.10 and end << 0.10; donors and real ≈ 0.10 / 0.10

# 1. minutes, numpy only: re-score every saved gallery with CIs, controls, PRDC
.venv/bin/python scripts/audit_metrics.py -c configs/shell_mission1.yaml

# 2. zero-retrain sweep (14 variants × 3 folds × 1536 windows; x0-space variants
#    skip the backward pass through the denoiser and are faster than C1)
.venv/bin/anogen -c configs/shell_mission1.yaml fixsweep

# 3. retrain S1 (same steps/backbone size): bidirectional only, and bidirectional +
#    log-spaced schedule. Then add under shell.fixsweep.variants
#      bidir_x0: {denoiser: results/shell_s1_bidir/denoiser.pt, guidance_space: x0, final_grad_scale: 0.25}
#      v2_x0:    {denoiser: results/shell_s1_v2/denoiser.pt,    guidance_space: x0, final_grad_scale: 0.25}
#    and rerun fixsweep (cached variants are reused)
.venv/bin/anogen -c configs/shell_mission1.yaml s1bidir
.venv/bin/anogen -c configs/shell_mission1.yaml s1v2

# 4. honest GenIAS baseline (P6), then the downstream check with the leak fixed
.venv/bin/anogen -c configs/shell_mission1.yaml geniasfair
.venv/bin/anogen -c configs/shell_mission1.yaml augdetect
```

`fixsweep` writes `results/shell_fixsweep/{summary.json,TABLE.md}`. All
variants share the torch seed per fold, so the paired ARP differences against
`c1_repro` isolate the switch. `c1_repro` is a fresh draw of the frozen C1
recipe, because the frozen galleries were made without a torch seed.

## 7. Decision rule, declared before the ESA run

A variant replaces C1 as the paper default only if, averaged over the three
folds:

- the start share is ≤ 0.20 and the end share ≥ 0.05 (real 0.114 at the start; the
  diagnosis predicts end ≪ 0.10 for C1);
- the envelope exit fraction is ≤ 3× the real one (C1: 15×) **and** the 95th
  percentile of the exit magnitude stays ≥ 50 % of C1's, so it is not won by
  shrinking every excursion (the hashfix failure);
- diff p99.9 ≤ 1.5× the real one (C1: 2.8×);
- the paired ARP difference vs `c1_repro` has a 95 % CI whose lower bound is
  > −0.02, i.e. it gives up no more ARP than the noise in the comparison.

C2ST is reported but not used to select: the linear raw-window C2ST cannot tell real anomalies
from real nominal donors (0.509), so it tests "looks like telemetry", not "looks like an anomaly".
