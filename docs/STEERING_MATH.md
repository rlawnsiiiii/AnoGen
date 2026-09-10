# Shell-steering diffusion

Readable in a plain editor. No LaTeX required. Code: `src/anogen/shell/diffusion.py` (denoiser) and `src/anogen/shell/steer.py` (shell + guided DDIM). Plan: `shell-steering_ts_anomalies_f6233086.plan.md`.

The method is a **composition** of known pieces (DDPM, DDIM, Guided-DiffTime, Sehwag / Um). The experiment is: does a **quantile-band shell** on a **univariate telemetry** score generate windows that cover held-out ESA **Anomalies** without colliding with ESA **Rare Events**?

---

## 0. How to use Anomalies and Rare Events

ESA-ADB labels two different things. Do not mix them.

| ESA label | Meaning | Headline (ZS, S1–S4) | Few-shot (S5) |
|---|---|---|---|
| Everyday nominal | Typical ops | **Train** denoiser, encoder, reference set R. **Set Q_q** from held-out nominals. Donor windows for the edit. | Same frozen score and same Q_q (leash). |
| Rare Event | Unusual but valid | **In the train pool** (S1/S2/S3) with everyday nominals. No upsample yet (`rare_upsample: 1`). Anomaly spans stay occupied. Still the **collision set** for Coverage@τ. S4 reports their energy vs Q_q. | Train-fold rares are **hard negatives** for the classifier (label 0). Held-out rares stay the collision set. |
| Anomaly | Confirmed fault | **Do not train** the denoiser or encoder. **Do not** set Q_q from them. **Query set** for Coverage@τ. Fold-0 anomalies **choose τ** (so unguided coverage is 10%). May **select** `{q, δ, λ, ν*}` via OOF, then freeze. S4 **reports** their energy vs Q_q. | Train-fold anomalies fit the residual adapter and the classifier (label 1). Held-out events stay queries. |

Rares go into `train_index.csv` (everyday + rare windows). Everyday-only `nominal_index.csv` is still the edit-donor gallery. Upsampling rares is a later switch (`rare_upsample` > 1).

**Do not** set Q_q from true anomalies in the ZS run. That aims the shell at labeled faults and the zero-shot claim is gone. Anomaly-calibrated Q* is a separate FS table, not this protocol.

**Do not** retune λ / q / δ after reading the S4 energy diagnostic. The diagnostic explains a win or a collision; it is not a second training loop.

### What each label is for, in one line

- **Nominals** define “typical” (the score and the band).
- **Anomalies** tell you whether generated windows hit **faults**.
- **Rares** tell you whether you hit the **wrong unusual thing**.

S4 gap = Coverage@τ(held-out anomalies) − Coverage@τ(held-out rares).

`Category` still decides train / query / collision roles above. Plot-row kinds
are official `Length` × `Locality` plus one invented level-shift overlay on
global subsequences. [ESA_LABELS.md](ESA_LABELS.md).

---

## 1. What is generated

A sample is a univariate window x of length W (default W = 512 bins × 30 s ≈ 4.3 h) on Mission 1 channels 41–46.

- Denoiser ε_θ(x_t, t, c) is trained on **nominal** windows only. c is a channel index, not a fault class.
- Shell = frozen reconstruction encoder e(·) + a quantile of a kernel energy on held-out **nominal** embeddings.
- Sampling is an **edit**: noise a real nominal to a moderate t*, then DDIM back with an extra force. Not a draw from x_T ~ N(0, I).

```
nominal window  --q_sample(ν)-->  x_{t*}  --guided DDIM-->  edited window
                                      ^
                          shell error (and optional S5 classifier)
```

---

## 2. Papers this logic is taken from

Nothing here is a single-paper clone.

| Layer | Paper | What this repo takes |
|---|---|---|
| Forward process + ε-loss | Ho, Jain, Abbeel, DDPM, NeurIPS 2020 ([arXiv:2006.11239](https://arxiv.org/abs/2006.11239)) | Linear β_t. x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε. Train ‖ε_θ − ε‖². |
| Deterministic reverse | Song, Meng, Ermon, DDIM, ICLR 2021 ([arXiv:2010.02502](https://arxiv.org/abs/2010.02502)) | η = 0 DDIM: x_{t'} = √ᾱ_{t'} x̂_0 + √(1−ᾱ_{t'}) ε_θ. |
| Add ∇ of a side objective | Dhariwal & Nichol, NeurIPS 2021 ([arXiv:2105.05233](https://arxiv.org/abs/2105.05233)) | Classifier-guidance template. |
| Closest time-series recipe | Coletta et al., Guided-DiffTime ([arXiv:2307.01717](https://arxiv.org/abs/2307.01717)) | Freeze the TS denoiser. Evaluate a differentiable constraint on Tweedie x̂_0. Backprop ∇_{x_t} f_c(x̂_0). No retraining. |
| Low-density + manifold leash + unit grads | Sehwag et al., CVPR 2022 ([arXiv:2203.17260](https://arxiv.org/abs/2203.17260)) | Push toward atypical regions; a second force stays on-manifold; ∇* = ∇ / ‖∇‖. |
| Pin rarity to a level | Um, Lee, Ye, minority guidance ([arXiv:2301.12334](https://arxiv.org/abs/2301.12334)) | Q_q is the “desired likelihood level,” not argmax OOD. |
| Feature-space kNN energy (idea only) | GOOD 2025 ([arXiv:2510.17131](https://arxiv.org/abs/2510.17131)); BOOD 2025 ([arXiv:2508.00350](https://arxiv.org/abs/2508.00350)) | Steer in embedding space toward a sparse / boundary band. |
| TS backbone | TSDiff ([arXiv:2307.11494](https://arxiv.org/abs/2307.11494)) | Residual S4-D score (`backbone: tsdiff`). U-Net remains as `backbone: unet`. |
| Reference generator, not steering | GenIAS ([arXiv:2502.08262](https://arxiv.org/abs/2502.08262)) | TCN-VAE; inflate latent std; deviation-patch (Alg. 2): keep x̃ only where (x−x̃)² > τ_patch · amp. |
| FS adapters (causal-plan note) | FaultDiffusion ([arXiv:2511.15174](https://arxiv.org/abs/2511.15174)) | Freeze the score, add a small residual. |

**This repo / plan adds:** soft kernel energy h_soft against a nominal reference set; quadratic band (h_soft(x̂_0) − Q_q)² as f_c; start from a lightly noised real nominal (ν* ≈ 0.2); score by anomaly coverage minus rare collision.

An older causal plan wrote a geometric “score-shell walk” with a unit normal and tangent noise. This repo dropped the tangent walk and used the quadratic band.

---

## 3. Unguided diffusion (S1)

### 3.1 Forward process (DDPM)

T = 200 discrete times (`n_times`). Linear schedule:

```
β_t  ∈  [10⁻⁴,  2×10⁻²]
α_t  =  1 − β_t
ᾱ_t  =  α_1 α_2 … α_t
```

Closed-form noising (`q_sample`):

```
x_t  =  √ᾱ_t · x_0  +  √(1 − ᾱ_t) · ε
ε    ~  N(0, I)
```

That is q(x_t | x_0) = N( √ᾱ_t x_0 ,  (1 − ᾱ_t) I ).

### 3.2 Training (ε-prediction)

Ho et al.: the variational bound, after the ε-reparameterization, is (up to weighting)

```
L(θ)  =  E_{x_0, t, ε}  ‖ ε − ε_θ(x_t, t, c) ‖²
```

That is `F.mse_loss(pred, noise)` in `train_denoiser`. 1-D U-Net, sinusoidal time embedding, channel embedding after the first block.

Clean x_0 is min-max scaled per channel on the train pool to [0, 1] (`ChannelMinMax`) before the forward process. Invert after DDIM so plots and coverage stay in raw units. The encoder and classifier stay raw: `guided_ddim` inverses x̂_0 before those heads. Do not re-scale the noised x_t. This is the CSDI / DDPM-style affine, not TSDiff's MeanScaler.

Score and noise predictor are the same object:

```
∇_{x_t} log q(x_t | x_0)  =  − ε / √(1 − ᾱ_t)

s_θ(x_t, t)  ≈  − ε_θ(x_t, t, c) / √(1 − ᾱ_t)
```

Following the **score** walks toward higher **nominal** density. Unguided DDIM therefore looks nominal. That is why λ = 0 is the control: if unguided already covers real anomalies, the shell is unnecessary.

On channels whose raw values live on a small alphabet (stairs, 0/1 dropouts), this Gaussian \(L^2\) denoiser emits the **posterior mean** and fills the gaps between levels. Post-hoc and GenIAS keep the stairs because they copy the parent. Fixes (snap, residual diffusion, categorical / index models): [DISCRETE_CHANNELS.md](DISCRETE_CHANNELS.md).

### 3.3 Tweedie / DDIM x̂_0

Invert the forward equation with the predicted noise:

```
x̂_0(x_t)  =  ( x_t − √(1 − ᾱ_t) · ε_θ )  /  √ᾱ_t
```

This is the posterior mean of x_0 given x_t under a Gaussian forward process (Tweedie’s formula). **Every guidance term is evaluated on x̂_0, not on the noisy x_t.** The constraint is defined on the clean window that will be emitted (Guided-DiffTime / DPS).

### 3.4 Deterministic DDIM (η = 0)

```
x_{t'}  =  √ᾱ_{t'} · x̂_0  +  √(1 − ᾱ_{t'}) · ε_θ
```

Last step (t' < 0): x ← x̂_0. Code: `ddim_sample`. Default: 20 subsampled times from t* down to 0.

### 3.5 Partial noising (SDEdit-style)

```
t*  =  round( ν · (T − 1) )
```

Guided default ν = 0.2. Unguided control ν = 1.0. Small ν keeps the donor window’s coarse shape. The shell arm is an **edit of a real nominal**, not a draw from prior noise.

---

## 4. The shell (S2)

### 4.1 Encoder

`ConvEncoder`: 1-D conv autoencoder, reconstruction MSE on nominal windows. Only e(x) ∈ R³² is used at sample time. Protocol lock for S2/S4: reconstruction, not contrastive.

OOF ablations (temporal map, 3-class SupCon) live in `ShellEncoder` / phase `enc` and do **not** replace this lock. See [ENCODER_ABLATION.md](ENCODER_ABLATION.md).

### 4.2 Soft energy

Reference set R = {r_1, …, r_M} of nominal embeddings (`n_ref = 512`):

```
h_soft(x)  =  −τ log  Σ_{r ∈ R}  exp( −‖ e(x) − e(r) ‖² / τ )
```

Code: `soft_energy`. Two readings:

1. **Softmin of squared distances.** τ → 0: h_soft → min_r ‖e(x)−e(r)‖² (nearest neighbor). τ → ∞: mean-distance energy.
2. **Negative log kernel density.** A Gaussian KDE on embeddings has p_τ(z) ∝ Σ_r exp(−‖z − e(r)‖² / τ), so h_soft = −τ log p_τ(e(x)) + const. Large h = low density in encoder space.

Bandwidth if not passed in:

```
τ  =  median { ‖e(r) − e(r')‖  :  r ≠ r' }
```

clamped at 10⁻³.

Raw diffusion NLL is a bad density proxy (Sehwag §3.2.1). This repo uses a frozen embedding energy instead.

### 4.3 Quantile band — set from nominals, not from anomalies

On held-out **nominal** validation windows only:

```
Q_q  =  Quantile_q( { h_soft(x) : x ∈ X_val_nominal } )
q    =  0.99   (config; plan also allows 0.95, 0.975)
δ    =  max( 0.25 · IQR(h_val) ,  10⁻⁴ )
```

The shell is the band

```
| h_soft(x) − Q_q |  ≤  δ
```

Occupancy = fraction of guided samples in that band. Win rule requires occupancy ≥ 0.8.

Q_0.99 means “as rare as the rarest 1% of **nominals** in encoder space.” That is not “looks like a labeled Anomaly.” Rare Events are excluded from R and from this quantile because low density ≠ fault.

### 4.4 Shell error (the constraint f_c)

```
f_band(x̂_0)  =  ( h_soft(x̂_0) − Q_q )²
```

A 1-D quadratic well in energy, centered on the quantile. Autodiff through e and the Tweedie map gives ∇_{x_t} f_band.

| `mode` | f_c | Meaning |
|---|---|---|
| `"band"` | (h − Q_q)² | leash: push out if too typical, pull back if overshot |
| `"push"` | −h_soft | maximize atypicality, no quantile |
| `"off"` | 0 | unguided DDIM from the same t* |

`"push"` tests whether the **band** matters, versus “just go OOD.”

---

## 5. Guided reverse step

At each subsampled time t, with t' the next time (or −1):

1. ε ← ε_θ(x_t, t, c)
2. x̂_0 ← (x_t − √(1−ᾱ_t) ε) / √ᾱ_t
3. h ← h_soft(x̂_0);  f ← f_c(x̂_0)
4. g ← clip( ∇_{x_t} f , ±c_max )
   — or, if `normalize_grad` (S5): g ← clip( g / ‖g‖ , ±c_max )
5. Optional S5 classifier: g ← λ g + λ_cls ∇_{x_t}(−ℓ(x̂_0))
6. Update:

```
x_{t'}  =  √ᾱ_{t'} · x̂_0  +  √(1 − ᾱ_{t'}) · ε  −  λ g
```

Last step: x ← x̂_0 − λ g.

Defaults: λ = 0.3, c_max = 1.0, 20 DDIM steps, ν = 0.2.

### Why the sign is a push-then-pull

```
∇ (h − Q_q)²  =  2 (h − Q_q) ∇h
```

Subtract λ times that gradient:

- h < Q_q (too typical): walk **along** ∇h → energy **up**
- h > Q_q (overshot): walk **against** ∇h → energy **down**
- h = Q_q: gradient is 0 → stay

A 1-D controller on rarity, autodiffed through x̂_0.

### Relation to the older geometric walk

The causal-fault plan wrote

```
x_{r−1}  =  DDIM  +  η_n [h* − h(x̂_0)] n̂  +  η_t (I − n̂ n̂ᵀ) ξ
n̂       =  ∇h / ‖∇h‖
```

The first two terms have the same **direction** as −∇(h − h*)²:

```
−∇(h − Q)²  =  2 (Q − h) ∇h   ∝   (h* − h) n̂
```

Shipped code: quadratic residual, **no** tangent noise, clip-in-box (unit-normalize only in S5), force subtracted from the predicted sample.

Guided-DiffTime instead perturbs the **noise**:

```
ε̂  ←  ε̂  −  ρ √(1 − ᾱ_t) ∇_{x_t} f_c(x̂_0)
```

then ordinary DDIM with ε̂. Subtracting λ g from x_{t'} is a cousin, not the same algebra. λ here is in **window units per step**.

### Extra corrections (n_correct)

At each DDIM time t, guidance can be applied `n_correct` times before advancing to t'. The extra passes stay at t (re-encode x̂_0, recompute g, write x_t ← DDIM_t − g). This is more GD on the constraint, not a longer diffusion horizon. At ν = 0.2, t* ≈ 40, so `ddim_steps > 41` mostly repeats times; raise `n_correct` or ν if you want a longer walk.

### Contrastive energies (rare vs anomaly)

The shell band cannot separate Rare Events from Anomalies: S4 put rares on Q_q. After the locked S4 table, `tune` adds two OOF terms on the **same** encoder e(·) and the **same** bandwidth τ as §4.2. All three energies are evaluated on Tweedie x̂_0, like f_band.

Write R_nom, R_anom, R_rare for the three reference **embedding** sets (not raw windows):

```
R_nom   =  { e(r) : r ∈ nominal refs used for Q_q }          # same R as §4.2
R_anom  =  { e(a) : a is an Anomaly window, event fold ≠ k }
R_rare  =  { e(u) : u is a Rare Event window, event fold ≠ k }
```

Fold k is the fold being scored. Train-fold only: no window whose `event_id` is in fold k enters R_anom or R_rare. R_nom is not OOF in that sense — it is the frozen nominal cloud from `shell_from_nominal`.

Then h_nom, h_anom, h_rare are the **same** soft-energy formula as §4.2, only the sum changes:

```
h_nom (x)  =  −τ log  Σ_{z ∈ R_nom}   exp( −‖ e(x) − z ‖² / τ )
h_anom(x)  =  −τ log  Σ_{z ∈ R_anom}  exp( −‖ e(x) − z ‖² / τ )
h_rare(x)  =  −τ log  Σ_{z ∈ R_rare}  exp( −‖ e(x) − z ‖² / τ )
```

Code: `soft_energy(z, ref, tau)` three times — `ref_t` / `ref_anom` / `ref_rare` in `guided_ddim`. τ is **not** re-estimated on anomalies or rares; it stays the nominal median pairwise from §4.2. If R_anom or R_rare is empty, that term is dropped (`ref_*=None`).

Readings (same as §4.2):

- τ → 0: h_anom → min_{a} ‖e(x)−e(a)‖², h_rare → min_{u} ‖e(x)−e(u)‖².
- Small h_anom = embedding sits on the train-fold **fault** cloud.
- Small h_rare = embedding sits on the train-fold **rare** cloud.
- Large h_* = far from that cloud (low kernel density vs that set).

The sampling objective is the sum of the three terms:

```
f(x̂_0)  =  λ (h_nom(x̂_0) − Q_q)²  +  λ_anom h_anom(x̂_0)  +  λ_rare ( − h_rare(x̂_0) )
```

`guided_ddim` subtracts a clipped (or unit-normalized) ∇_{x_t} f. Term by term:

```
∇ (h_nom − Q_q)²  =  2 (h_nom − Q_q) ∇ h_nom     # band, §5
∇ h_anom          =  pull toward R_anom          # minimize energy vs faults
∇ (−h_rare)       =  push away from R_rare       # maximize energy vs rares
```

- λ_anom: walk **toward** known faults (make e(x̂_0) look like train-fold Anomalies).
- λ_rare: walk **away** from Rare Events (make e(x̂_0) *not* look like train-fold rares).

Band-only is λ_anom = λ_rare = 0 (locked S4). Combined (tune id 14) is λ = 0.3, λ_anom = λ_rare = 1. Contrast-only (tune c15) is λ = 0, λ_anom = λ_rare = 3.

The sum in \(h_\mathrm{anom}\) is a KDE. Far from every \(a_i\) the gradient aims at the **mean of \(R_\mathrm{anom}\)**. Alternatives that only need *one* nearby fault (min, local \(k\)NN, one proto per sample): [H_ANOM.md](H_ANOM.md). Default `anom_energy_kind="soft"` is this formula; locked tables stay on soft.

That is sampling-time contrast, not a new score. S5 still goes further: residual adapter + classifier with rares as hard negatives.

`tune` is an OOF search. It does **not** overwrite `results/shell_s4`. The **coverage** τ stays the S4 freeze (a different τ from the kernel bandwidth above). Sealed test stays closed.

### Sehwag α–β and Um

Sehwag: α increases hardness (leave the mode); β stays on the real manifold; gradients are unit-normalized.

- ZS: the quadratic band is both α and β; DDIM’s nominal score is extra leash.
- S5: `lam_cls` maximizes an anomaly logit (α); `lam_shell` is the leash (β); `normalize_grad=True` is ∇*.

Sehwag trains the diffusion model and the hardness head on the **full long-tailed set, including tail images**. That is why S5 may see train-fold anomalies, and why S2/S4 (nominal-only) is a stricter copy of the image papers.

Um: sample at a **chosen** minority level, not argmax rarity. Q_q is that level. `mode="push"` is the unbounded ascent Um warns against.

---

## 6. Few-shot (S5)

Frozen backbone plus a tiny residual:

```
ε_FS(x_t, t, c)  =  ε_θ(x_t, t, c)  +  a_ψ(x_t, t, c)
```

a_ψ is trained with the same ε-MSE on a mix of **train-fold anomaly** windows and matched nominals (`anom_frac = 0.5`). Backbone weights stay frozen.

Classifier ℓ(x̂_0): 1-D conv + linear logit, BCE.

- Positives: train-fold **Anomalies**
- Negatives: matched **nominals** plus train-fold **Rare Events** (hard negatives)

The rare hard-negatives exist so the head cannot get away with “any excursion = fault.” If it cannot reject rares, it is not used for guidance.

```
f_FS  =  λ_shell (h − Q_q)²  +  λ_cls (−ℓ(x̂_0))
```

Defaults: λ_shell = 0.15, λ_cls = 0.8, still ν = 0.2, still the **S2** Q_q. The shell is not refit on anomalies. Held-out-fold rares stay the collision metric (never in this loss).

---

## 7. Evaluation (S0 / S4) — not a detector

Locked scoring embedding is **not** e(·). It is `feature_pack_v1` (`embed_windows`): 12 handcrafted numbers (z-moments, first-difference moments, peak |z|, excursion length, four rFFT log-energy bands). Frozen in S0.

For a generated gallery G and a query set Q:

```
d(q, G)         =  min_{g ∈ G}  ‖ φ(q) − φ(g) ‖
Coverage@τ(Q,G) =  fraction of q in Q with d(q, G) ≤ τ
```

τ is chosen **once** on fold-0 anomalies vs the **unguided** gallery so unguided coverage equals 0.10:

```
τ  =  Quantile_0.10( { d(a, G_unguided) } )
```

Then freeze τ for every method.

```
gap  =  Coverage@τ(held-out Anomalies, G)
     −  Coverage@τ(held-out Rare Events, G)
```

Diversity = mean pairwise ‖φ(g) − φ(g')‖ inside G.

GenIAS-style scores on the **same locked φ** (not their Deep SVDD):

```
ARP(Q,G)  =  1 / (1 + mean_q d(q, G))
EDI(G)    =  Shannon entropy of G over k-means bins of the union of all galleries
```

ARP is reported for Anomalies and for Rare Events. Higher ARP = nearer on average. EDI is one number per gallery. Do not retune from these.

Win rule (frozen in S0): ZS shell beats **both** GenIAS and post-hoc on gap, occupancy ≥ 0.8, diversity at least half of unguided. Tie or loss is reported. Do not retune the shell after S4.

S4 also writes `energy_diagnostic`: mean h_soft and band occupancy of real Anomalies and Rare Events vs the frozen Q_q. That answers “is the nominal 0.99 band already where rares live?” It does not change Q_q.

---

## 8. Code map

| Symbol | Code |
|---|---|
| β_t, ᾱ_t | `DiffusionSchedule.linear` |
| x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε | `q_sample` |
| ε_θ | `UNet1D` |
| x̂_0, DDIM | `ddim_sample`, `guided_ddim` |
| e(·) | `ConvEncoder.encode` |
| h_soft, τ, Q_q, δ | `soft_energy`, `shell_from_nominal` |
| f_band, f_push | `guided_ddim` `mode` |
| n_correct | `guided_ddim` `n_correct` |
| h_nom, h_anom, h_rare | `soft_energy` vs `ref` / `ref_anom` / `ref_rare` |
| clip(∇, ±c_max) | `grad.clamp(-c_max, c_max)` |
| ∇* | `_unit_grad` |
| band occupancy of a labeled set | `band_report` |
| ε_FS | `AdaptedDenoiser` |
| ℓ (rares as extra negatives) | `AnomalyClassifier`, `train_classifier(..., x_rare=)` |
| Coverage / gap / ARP | `score_generator`, `arp` |
| EDI | `edi_by_method` |
| τ freeze | `choose_tau` |
| Label roles | `protocol.json` → `labels` |

Pipeline: s0 windows + protocol → s1 nominal denoiser → s2 encoder + Q_q → s3 GenIAS / post-hoc → s4 OOF coverage + energy diagnostic → tune (optional sampling HPs / contrastive energies) → enc (optional encoder ablations) → encscore (1536) → s5 event-OOF adapters (no GenIAS retrain; see [S5.md](S5.md), [REPO.md](REPO.md)).

---

## 9. What this is not

- Not classifier-free guidance (no jointly trained conditional / unconditional pair).
- Not a new SDE (no probability-flow / EDM / rectified flow).
- Not causal generation (no SCM, no PCMCI+, no do(root)). That is the CausalDiscovery track.
- Not “shell-steering has never been done.” It has, on images. This is the telemetry-window instance.
