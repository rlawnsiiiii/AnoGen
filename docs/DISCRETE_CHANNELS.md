# Discrete / quantized channels — why diffusion smears them, what each fix means

Eyes on extra Mission 1 (`results/xfer/mission1_extra/plots/`): real windows, post-hoc, and GenIAS keep **flat levels + vertical jumps**. Unguided, `time_recon` band, and `time_recon` combined look **continuous** (wavy, hairy). Same-parent strips show the donor already is a small alphabet; diffusion does not stay on it.

This note writes the six ideas from that discussion as math. Nothing here is implemented. Locked S4 (ch 41–46) is untouched. Sampling math of the continuous score: [STEERING_MATH.md](STEERING_MATH.md). Transfer scores: [XFER.md](XFER.md).

Notation: a window is \(x \in \mathbb{R}^W\) (W = 512). Parent / donor is \(x^\mathrm{par}\) (`galleries.npz` key `cond`). Generated window is \(x^\mathrm{gen}\). Channel index \(c\). The shipped score is ε-prediction under a **Gaussian** forward process in min-max units.

---

## 0. Why post-hoc / GenIAS look discrete and the score does not

### 0.1 What “discrete” means here

A channel (or a single window) has a small **alphabet**

```
A  =  { v_1, …, v_K } ⊂ ℝ
```

Typical on extra M1:

- ch 47–49: K small, values in a ~5×10⁻⁴ raw band (stairs / pulses around 0.0017–0.0024).
- ch 73–75: often two levels (rectangles 1 → 0 → 1).

A window is on the alphabet if every bin lands on some \(v_k\):

```
x_i  ∈  A    for all i = 1…W
```

Jumps are instantaneous: \(x_i = v_a\), \(x_{i+1} = v_b\), \(a \neq b\). No values between \(v_a\) and \(v_b\).

A cheap occupancy number (not implemented):

```
snap_frac(x, A)  =  (1/W)  #{ i : min_{v ∈ A} |x_i − v|  ≤  ε }
```

ε is a tiny absolute tolerance (e.g. 10⁻⁶ raw, or half the minimum gap in A). Real stairs and post-hoc (except ramps) have snap_frac ≈ 1. Diffusion today is ≈ 0.

### 0.2 Post-hoc copies the alphabet

```
x^post_i  =  x^par_i  +  r_i
```

\(r\) is a step, pulse, ramp, or global scale ([`baselines.py`](../src/anogen/shell/baselines.py)). For step/pulse, \(r_i\) is 0 or a constant \(a\). Then

```
x^post_i  ∈  A          if r_i = 0
x^post_i  ∈  A + a      if r_i = a
```

The **shape** of the stairs is the parent. Discreteness is inherited, not learned.

### 0.3 GenIAS is a small edit of the same parent

TCN-VAE: encode \(x^\mathrm{par}\) → \((\mu, \log\sigma^2)\), decode \(\tilde x = g_\psi(z)\) with \(z = \mu + \psi\,\sigma\odot\varepsilon\). ψ = 2 inflates the latent. If \(\|\tilde x - x^\mathrm{par}\|\) is small (it was, on 41–46), \(\tilde x\) stays visually on the stairs. Alg. 2 deviation-patch keeps \(\tilde x\) only where the residual exceeds τ·amp; that test kept 0% of bins on locked M1.

Again: **copy + tiny decode error**, not a draw from a continuous prior.

### 0.4 The score assumes a continuous Gaussian

Forward (STEERING_MATH §3.1), after per-channel min-max \(S_c\):

```
u_0   =  S_c(x_0) ∈ [0, 1]^W
u_t   =  √ᾱ_t · u_0  +  √(1 − ᾱ_t) · ε
ε     ~  N(0, I)
```

Support of \(u_t\) is all of \(\mathbb{R}^W\) for t > 0. The alphabet is gone after one noise step.

Training is posterior-mean ε-MSE:

```
L(θ)  =  E  ‖ ε − ε_θ(u_t, t, c) ‖²
```

The optimal \(\hat u_0\) is \(\mathbb{E}[u_0 | u_t]\). If two stairs \(v, v'\) are both plausible,

```
E[u_0 | u_t]  ≈  (v + v') / 2
```

which is **not** in A. That is the wavy / in-between traces.

DDIM (η = 0) then emits that mean:

```
û_0     =  ( u_t − √(1−ᾱ_t) ε_θ ) / √ᾱ_t
u_{t'}  =  √ᾱ_{t'} û_0  +  √(1−ᾱ_{t'}) ε_θ
```

Twenty steps, S4-D smoothness, and (for combined \(f\)) a continuous \(\nabla_{u_t} f\) all push further off A. ν = 1 must rebuild A from N(0, I). ν = 0.2 starts closer to \(u_0\) but still adds Gaussian noise at t* = round(0.2·199) ≈ 40, then continuous guidance.

Min-max does not create a lattice: it only rescales A into [0, 1]. Residual DDIM error of size 10⁻² in scaled units is larger than the raw gap on ch 47–49.

---

## 1. Snap after DDIM (projection onto a codebook)

**Idea.** Keep the shipped Gaussian score. After invert-scale, **replace each bin by the nearest code**. Eyes jump to “looks quantized.” The score is unchanged.

### 1.1 Parent alphabet (per sample)

```
A_par(x^par)  =  unique values of  (x^par_1, …, x^par_W)
```

On a stair parent, |A_par| is typically 2–10.

```
Π_{A}(y)_i  =  argmin_{v ∈ A}  |y_i − v|
x^snap      =  Π_{A_par}( x^gen )
```

Ties: pick the closer of the two neighbors, or the parent’s value at i (keeps dwells).

This is the Euclidean projection of \(\mathbb{R}^W\) onto \(A^W\).

### 1.2 Channel codebook (shared across windows)

Fit once on that channel’s train-pool raw values (or unique values with count ≥ m):

```
A_c  =  { v_1, …, v_{K_c} }
x^snap  =  Π_{A_c}( x^gen )
```

Use this when a generated **anomaly** should use a level the parent never visited (e.g. the “0” of a 1→0 dropout while the donor stayed at 1). Parent snap cannot invent a new letter.

### 1.3 Soft / gated snap

Do not snap a bin that has already moved a lot (so a pulse can sit off-lattice):

```
δ_i = |x^gen_i − x^par_i|
x^out_i  =  Π_A(x^gen)_i     if  δ_i ≤ η · std(x^par)
         =  x^gen_i          otherwise
```

η ~ 1–3. Ramps in post-hoc would also fail a hard snap; gated snap is the analogue.

### 1.4 What this is and is not

- **Is:** a deterministic post-process. `snap_frac` → 1 by construction (hard snap).
- **Is not:** a new score, not guidance, not discrete diffusion.
- **Risk:** a bad continuous \(x^\mathrm{gen}\) becomes “pretty stairs.” Always score **both** raw DDIM and snapped. If Coverage@τ only rises after snap, you projected junk onto A.

Guidance (optional second step): evaluate \(f\) on \(\Pi_A(\hat x_0)\) with a straight-through estimator

```
x̂_0^used  =  x̂_0  +  sg( Π_A(x̂_0) − x̂_0 )
```

so \(\nabla_{u_t} f\) sees discrete levels but backprop goes through \(\hat x_0\). Not required for a first plot.

---

## 2. Diffuse the residual, add back the parent

**Idea.** Never ask the score to rebuild the stairs. Only generate an **edit** \(r\).

```
x_0  =  x^par  +  r_0
```

Train and sample a denoiser on \(r\) (in raw or scaled units):

```
r_t   =  √ᾱ_t · r_0  +  √(1 − ᾱ_t) · ε
L(θ)  =  E ‖ ε − ε_θ(r_t, t, c; x^par) ‖²
```

Conditioning on \(x^\mathrm{par}\) (concat, or FiLM, or add \(x^\mathrm{par}\) as a second channel) tells the score “these stairs stay.”

At sample time:

```
r^gen   =  DDIM( r_{t*}, ε_θ(· | x^par) )     t* = round(ν (T−1))
x^gen   =  x^par  +  r^gen
```

ν = 0 and \(r_{t*}=0\) ⇒ \(x^\mathrm{gen}=x^\mathrm{par}\) (identity edit). Small ν ⇒ a local residual on intact stairs.

### 2.1 Masked residual

If the fault is a short pulse, restrict the support:

```
r_0  =  m ⊙ (x_0 − x^par),     m_i ∈ {0, 1}
x^gen  =  x^par  +  m ⊙ r^gen
```

\(m\) can be known (post-hoc style span) or predicted. Campaigns need a **long** \(m\); spikes a **short** \(m\).

### 2.2 Relation to the shipped sampler

Shipped `unguided_from_nominal` / `guided_ddim` noise **the whole window** \(x_0\), not \(r_0\). Residual diffusion is SDEdit on the **difference**, with \(x^\mathrm{par}\) held fixed. Combined \(f\) still applies to \(x^\mathrm{par}+r\) (raw \(\hat x_0\)) if you want the same shell; the score only has to produce \(r\).

### 2.3 Residual can stay continuous

\(r^\mathrm{gen}\) is still Gaussian. Stairs stay exact; the **edit** can look hairy. Combine with §1 on \(x^\mathrm{par}+r\) or snap \(r\) to \(\{0\}\cup (A_c - A_\mathrm{par})\) if the edit should also be a level change.

---

## 3. Treat discrete channels as discrete

**Idea.** If K is small, do not put a Gaussian on \(\mathbb{R}^W\). Put a model on **indices** in \(\{1,\ldots,K\}^W\).

### 3.1 Codebook

Same \(A_c = \{v_1,\ldots,v_K\}\) as §1.2. Encode a raw window:

```
s_i  =  argmin_k |x_i − v_k|     ∈ {1,…,K}
x_i  =  v_{s_i}
```

### 3.2 Categorical diffusion on s

A discrete DDPM (Austin et al. / multinomial diffusion): each bin is a K-way categorical. Forward corrupts toward uniform (or absorbing) on the simplex. Reverse predicts logits \(\ell_\theta(\cdot, t, c) \in \mathbb{R}^{W\times K}\). Sample \(s^\mathrm{gen}\), decode \(x_i = v_{s_i}\).

Support of every sample is \(A_c^W\) **exactly**. Jumps are just \(s_i \neq s_{i+1}\).

Bit-diffusion is the K = 2^b special case (pack levels into bits; each bit is binary diffusion). Natural for ch 73–75 if they are truly two-state.

### 3.3 Gaussian diffusion **on the index**, then decode

Treat \(s_i\) as a real number in {1,…,K}, run the **shipped** ε-model on \(s\) (or on \(s/K\)), then

```
ŝ_i     =  clip( round(s^gen_i), 1, K )
x^gen_i =  v_{ŝ_i}
```

Cheaper than a new categorical stack. Rounding is a snap in index space. The score can still emit 2.4; you project to 2.

### 3.4 When this is the right model

Use when `snap_frac` of **real data** vs \(A_c\) is already ≈ 1 (the channel *is* an alphabet). Do not force this on a smooth analog channel: rounding invents fake stairs.

Estimate K from the train pool: unique values, or the smallest k-means k with SSE / Var < 10⁻⁴. If K > ~32 or the residual after quantization is a large fraction of Var(x), stay Gaussian (§2 + optional §1).

---

## 4. Do not share one score across “digital” and “analog”

**Idea.** Extra M1 mixes two physics in one 6-way TSDiff.

```
ε_θ(u_t, t, c)     # one network, c ∈ {0,…,5}
```

Loss is the same L2 on ch 47 (tiny stairs) and ch 75 (0/1 rectangles). Channel embedding is a vector add; it does not change the Gaussian assumption.

Fixes, in increasing cost:

1. **Two denoisers.** Group A = {47,48,49} (or “K small”), group B = {73,74,75} (or “binary”). Same `guided_ddim`, different `denoiser.pt`.
2. **Per-channel head.** Shared trunk, last layer (or a small FiLM) per \(c\). Still Gaussian, but the residual noise scale can differ.
3. **Per-channel model class.** Group A: §3 (index / categorical). Group B: binary diffusion or residual on {0,1}. Analog leftovers: shipped score.

The compact xfer budget (4k steps, one backbone) made sharing worse. Locked S1 also shares one score, but 41–46 are closer to analog, so the smear was less visible.

---

## 5. Training / sampling tweaks (help a little, do not create a lattice)

These stay inside STEERING_MATH §3. They reduce smear; they do not put mass on A.

```
t*          =  round( ν · (T − 1) )     # smaller ν ⇒ less of the donor is destroyed
n_DDIM      =  20  →  50–100            # finer η=0 path; still a posterior mean
n_train     =  4k  →  closer to S1 20k  # better ε_θ; still L2
normalize_grad = False                  # combined’s unit ∇ is extra jitter on stairs
```

Why they cannot suffice: the **optimal** ε under L2 is still \(\mathbb{E}[\varepsilon|u_t]\), whose implied \(\hat u_0\) is off-alphabet (§0.4). More steps / more train move toward that optimum, not toward \(\Pi_A\).

Use these **with** §1 or §2, not instead.

---

## 6. Do not “fix” it by becoming GenIAS

GenIAS and post-hoc look discrete **because they copy \(x^\mathrm{par}\)**. If the method becomes “VAE or inject, then optionally snap,” that is a baseline, not shell-steering.

A fair comparison after a discreteness fix:

| Recipe | What is generated | Alphabet |
|---|---|---|
| Post-hoc | \(x^\mathrm{par}+r_\mathrm{hand}\) | inherited |
| GenIAS ψ=2 | decode(encode(\(x^\mathrm{par}\))) | almost inherited |
| Unguided / band / combined (now) | Gaussian DDIM on \(x\) | destroyed |
| §1 snap | Gaussian DDIM, then \(\Pi_A\) | forced |
| §2 residual | DDIM on \(r\), \(x^\mathrm{par}\) frozen | inherited + continuous \(r\) |
| §3 categorical | sample \(s\in\{1..K\}^W\) | exact |

Say which row you are claiming as the method.

---

## 7. How to tell if a fix worked

Compute on the same 768 (or 1536) donors, raw units, **before** retuning λ.

```
A_par^(n)     = unique(x^par(n))
snap_frac_par = mean_n snap_frac( x^gen(n), A_par^(n) )
snap_frac_c   = mean_n snap_frac( x^gen(n), A_{c(n)} )

edit_l2       = mean_n  ‖ x^gen(n) − x^par(n) ‖ / ( √W · std(x^par(n)) )
parent_corr   = mean_n  corr( x^gen(n), x^par(n) )
```

| Outcome | snap_frac | edit_l2 | Eyes |
|---|---|---|---|
| Post-hoc / GenIAS today | ~1 | small | stairs |
| Diffusion today | ~0 | medium | waves |
| §1 only, no real edit | ~1 | ~0 | parent |
| §1 + useful edit | ~1 | > 0 | stairs + a pulse/shift |
| §2 done right | ~1 on parent bins | = ‖r‖ | stairs + residual |

Keep Coverage@τ / ARP on locked \(\varphi\) for cite-vs-GenIAS. Add kind-conditional coverage ([morphology.py](../src/anogen/shell/morphology.py)): snapping must not be the only reason coverage moved.

---

## 8. Suggested order if we implement later

1. **§1 parent snap** on existing `results/xfer/mission1_extra/score/*.npz` (no retrain). Plot vs `same_parent.png`. Report snap_frac and Coverage@τ before/after.
2. If generated anomalies need new levels: **channel codebook** snap.
3. If band/combined still smear before the snap: **§2 residual** retrain on extra M1 only (`results/xfer/`, not `shell_s*`).
4. If ch 73–75 stay mushy: **§3 binary / index** on that group only (**§4** split).
5. §5 only as a knob on top of 1–4.

Do not overwrite `results/shell_s4`. Do not run S6.
