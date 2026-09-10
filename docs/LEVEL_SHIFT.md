# How to generate a **real level shift**

Plain-language note. Idea A (kind-conditional proto) is implemented: [KIND_PROTO.md](KIND_PROTO.md), phase `kindproto`. Ideas B/C are not. Does not overwrite locked S4.

What we have already tried (soft / nearest / kNN / proto, with and without \(h_\mathrm{rare}\)): [H_ANOM.md](H_ANOM.md). Live kind rule: [`assign_anomaly_kinds`](../src/anogen/shell/morphology.py) — official ESA `Length` × `Locality`, plus this overlay. “Level shift” is **our** crop statistic, not an ESA type: [ESA_LABELS.md](ESA_LABELS.md). Post-hoc step: [`posthoc_inject`](../src/anogen/shell/baselines.py). Eyes: `results/shell_hanom/norare/plots/real_kinds_vs_time_both_hanom.png` (morphology-kind rows; generated before the ESA map).

---

## 0. Words (including “DC” and \(\mu_L\))

A generated window is a vector \(x = (x_1,\ldots,x_W)\) with \(W=512\) bins (~4.3 h at 30 s). On channels 41–46 the **carrier** is the repeating analog oscillation (~6 cycles in that window).

**Mean of a piece of the window**

Split the window in half (256 bins each, ~2.15 h):

```
μ_L  =  mean of the Left  half   =  (x_1 + … + x_{256}) / 256
μ_R  =  mean of the Right half   =  (x_{257} + … + x_{512}) / 256
```

“Left” = earlier time, “Right” = later time. That is all \(\mu_L\) / \(\mu_R\) means. The live kind label **real level shift** is

```
Length = Subsequence  and  Locality = Global  and  |μ_L − μ_R| ≥ 0.01
```

(`shift_cut = 0.01`, raw units). Official Point / local-subsequence windows are
never relabeled, even if the half-means differ. See [ESA_LABELS.md](ESA_LABELS.md)
§4. The **legacy** morphology rule was “not short (span ≥ 100) and the same
`dmu` test”; `assign_morphology_kinds` still implements that for locked tables.

**DC** is electrical slang for **direct current**: the **standing level** (offset / baseline), as opposed to the wiggle on top of it. In this repo:

| Phrase | Meaning on 41–46 |
|---|---|
| DC / baseline / offset | The average height of the oscillation (e.g. 0.81 vs 0.95) |
| AC / carrier | The ~hourly oscillation around that height |
| A **DC change** / **step** / **shelf** | The baseline **jumps once** and **stays** at the new height |

A **spike** is a brief kick; the baseline after it is the same as before. A **level shift** is a new baseline for the rest of the window.

Sketch (not to scale):

```
spike:          ───/\───           same height left and right

level shift:    ════╗
                    ╚══════        high μ_L, low μ_R (or the reverse)
```

The red “real level shift” panels start near **0.95** for ~0.5 h, then drop by ~**0.15** to ~0.78 and **keep** oscillating there. That is one DC change, then a new DC.

**Whole-window mean vs half-window means**

```
μ_all  =  mean of all 512 bins
```

Raising \(\mu_\mathrm{all}\) (the whole series sits higher) is **not** a level shift. A level shift needs \(\mu_L\) and \(\mu_R\) **different**. Soft combined usually keeps \(\mu_\mathrm{all}\) at the parent’s ~0.81. Proto can move \(\mu_\mathrm{all}\) (the long-campaign column sits near 0.94). That is why proto can *look* better. It still has \(\mu_L \approx \mu_R\).

---

## 1. Why current \(f\) does not make a shelf

Locked combined (and the H_ANOM variants) steer in **encoder space** \(e(x)\):

```
f  =  λ (h_nom − Q_q)²  +  λ_anom h_anom  [+ λ_rare (−h_rare)]
```

\(h_\mathrm{anom}\) only says “look like some train-fold fault embedding.” It never says “make \(\mu_L\) and \(\mu_R\) differ.”

Also:

- Sampling starts from a **nominal parent** (\(\nu=0.2\)). That parent already has one baseline (~0.81). The score likes to keep the carrier around that baseline.
- Encoder gradients are **local** (short ticks), not “add a constant to 256 bins.”
- \(\varphi\) (`feature_pack_v1`) **z-scores each window** (subtracts \(\mu_\mathrm{all}\)). Absolute height is thrown away, so “nearest in \(\varphi\)” can pair a real shelf with a quiet wave.
- There are only **6** real windows labeled level shift out of 672. Random proto almost never draws one.

Dropping \(h_\mathrm{rare}\) made more spikes. It did not create \(|\mu_L-\mu_R|\).

---

## 2. Idea A — kind-conditional proto

**Proto today:** for each generated window \(b\), pick **one** random train-fold anomaly embedding \(a_{i(b)}\) from the whole set \(R_\mathrm{anom}\) (up to 256 mixed faults) and walk toward it:

```
h_anom^{(b)}  =  ‖ e(x̂_0^{(b)}) − a_{i(b)} ‖²
```

Most of those \(a_i\) are long campaigns or spikes. So proto usually means “look like a campaign/spike,” not “look like a shelf.”

**Kind-conditional proto:** build a **smaller** set that only contains windows we already call level shift:

```
R_shift  =  { e(a) : a is a labeled Anomaly, official global subsequence,
                     and |μ_L(a) − μ_R(a)| ≥ 0.01 }
```

That is the same \(\mu_L,\mu_R\) as §0, computed on the **real** ref window \(a\), not on the generated \(x\). Then draw \(i(b)\) only from \(R_\mathrm{shift}\) (6-ish events — very few).

Optionally start from noise or a larger \(\nu\), so the parent baseline is not glued at 0.81.

**What this can do:** proto already moves the *whole* baseline when the ref is a high window. A shift-only ref is a window that is high then low. If `time_both`’s time map stores “early high, late low,” the walk might smear a step. **What it may not do:** with 6 refs, and an encoder that was never asked to keep offset, you may only get a new global \(\mu_\mathrm{all}\), still \(\mu_L\approx\mu_R\).

Implemented: same `anom_energy_kind="proto"`, different ref list, plus a 200-step from-noise recipe. Isolated `results/shell_kindproto/`. Not S4.

On 41–46 every real level-shift window is fold 0, so those refs **leak** the query fold. Treat that gallery as a capability test, not a few-shot cite.

Mixed gallery (keep shelves *and* quiet kinds): [KIND_MIX.md](KIND_MIX.md).

**Result:** aiming at those six embeddings **does** emit a shelf (\(\mathrm{P}(\lvert\mu_L-\mu_R\rvert\ge 0.01)\approx 0.96\), median gap 0.022, eyes 0.95→0.78). Median raw L2 to a real shelf is 0.29 — near-copies of the leaked refs. Coverage@τ stays 0 (\(\varphi\) z-scores DC). 200 from-noise steps keep the step and add jitter. Details: [KIND_PROTO.md](KIND_PROTO.md).

---

## 3. Idea B — ask for \(\mu_L \neq \mu_R\) in \(f\) (the real fix)

Do not hope the encoder invents a shelf. Put the kind rule into the constraint, on the **predicted clean window** \(\hat x_0\) (same place as the band):

```
μ_L(x̂_0)  =  mean(x̂_0[1 : W/2])
μ_R(x̂_0)  =  mean(x̂_0[W/2+1 : W])

f_shift   =  λ_s  ( |μ_L(x̂_0) − μ_R(x̂_0)| − m* )²
```

\(m^\star\) is the **target gap** in raw units (try 0.05–0.15; the red examples are ~0.15). \(\lambda_s\) is a new weight. Then

```
f  =  λ (h_nom − Q_q)²  +  λ_anom h_anom  +  f_shift
```

(or drop \(h_\mathrm{anom}\) for a shift-only recipe). Guided DDIM subtracts \(\nabla f\) as now.

**Signed** variant if you want “drop” not “either way”:

```
f_shift  =  λ_s  ( (μ_L − μ_R) − s* )²     # s* = +0.15 means left higher than right
```

**Change-point** \(t_0\) (the red drop is near 0.5 h, not exactly mid-window):

```
μ_before  =  mean(x̂_0[1 : t_0])
μ_after   =  mean(x̂_0[t_0+1 : W])
f_shift   =  λ_s  ( |μ_before − μ_after| − m* )²
```

Pick \(t_0\) at random, or try a few (e.g. 0.2 W, 0.5 W).

This is the same statistic as the kind label. Soft / nearest / proto never write this term, so they never optimize it.

Judge success by \(\mathrm{P}(|\mu_L-\mu_R|\ge 0.01)\) and **eyes in raw units**, not only nearest-in-\(\varphi\).

---

## 4. Idea C — hybrid: diffusion carrier + hand step

Post-hoc already implements a step on a real parent:

```
x_t  ←  x_t  +  a     for all t ≥ t_0
```

(`fam == "step"` in `posthoc_inject`). Today \(a\) is \(0.5\)–\(2\times\) the window’s own standard deviation — a **small** wiggle, not a 0.15 raw shelf.

**Hybrid:** keep the diffusion oscillation, then add a step with \(a\) in **raw units** (e.g. \(\pm 0.1\)). You will get a plot that looks like the red row. It is not “steering learned a shift.” Use it as an exhibit, not as a cite-vs-GenIAS method.

A different “hybrid” — band on quiet ESA kinds, proto on shift / needles —
is [HYBRID.md](HYBRID.md). That one does not add a hand step.

---

## 5. What “proto looks better than soft” actually is

On `real_kinds_vs_time_both_hanom.png` (norare), level-shift column:

| Row | What you see | \(\mu_L\) vs \(\mu_R\) |
|---|---|---|
| real | high ~0.95, then drop to ~0.78 and stay | **different** |
| soft | smooth carrier at ~0.81 the whole time | same |
| proto | noisier carrier, still one baseline | same (or only a tiny gap) |

Proto is better as **texture** and as **permission to leave the parent’s height**. It is not a better *shift*. The long-campaign proto panel sitting at ~0.94 is a whole-window DC move, the same mechanism.

---

## 6. What I would run (if we implement)

1. Kind-conditional proto (idea A) + from-noise / 200 DDIM steps — **done**, [KIND_PROTO.md](KIND_PROTO.md).
2. \(f_\mathrm{shift}\) (idea B) if A does not make a visible shelf — this is the methods claim for that kind.
3. Hybrid (idea C) only if we need a figure tomorrow.

Do not retune \(\lambda_\mathrm{anom}\) / \(k\) / \(\lambda_\mathrm{rare}\) to chase this kind. Do not overwrite `results/shell_s4`.
