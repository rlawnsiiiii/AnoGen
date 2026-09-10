# time_recon + combined \(f\) vs GenIAS / post-hoc (Mission 1, continuous)

Focus is locked Mission 1 channels **41–46** (smooth analog oscillation). Discrete extra-M1 stairs are out of scope here. Sampling math: [STEERING_MATH.md](STEERING_MATH.md). Scores vs GenIAS: [PAPER_CANDIDATES.md](PAPER_CANDIDATES.md). \(h_\mathrm{anom}\) nearest / kNN / proto (same 41–46, not a new S4): [H_ANOM.md](H_ANOM.md). Level-shift ideas: [LEVEL_SHIFT.md](LEVEL_SHIFT.md). From-noise figure: `results/shell_plots/shell_tune/real_vs_time_recon_combined_from_noise.png`.

This note is **advantages and a publishability read**, not a new S4 freeze. Coverage@τ / ARP numbers stay in PAPER_CANDIDATES §2.

---

## 1. Can this be a paper?

**Not as “we beat GenIAS at generating ESA Anomalies.”** On the locked protocol (same \(\varphi\), τ = 0.105, 1536 donors):

| | Coverage@τ anom | ARP anom | Div | Occupancy |
|---|---:|---:|---:|---:|
| Post-hoc | **0.267** | 0.32 | 1.21 | — |
| GenIAS ψ=2 | 0.195 | 0.32 | 1.09 | — |
| Unguided ν=1 | 0.109 | **0.57** | 2.34 | — |
| **time_recon + combined \(f\)** | **0.023** | 0.54 | **6.61** | **0.78** |

Combined **loses Coverage@τ by an order of magnitude**. That is why “better anomaly generator than GenIAS” is **not** supported on the locked win rule.

**If you follow GenIAS’s own primary metric (ARP anom)** — their “is the gallery close enough to real anomalies, on average?” — combined is **not behind**. ARP 0.54 vs GenIAS 0.32 / post-hoc 0.32 (and 0.57 unguided). Combined also **wins diversity** (Div 6.61 vs ~1.1; EDI 2.54 vs ~1.1–1.4) and can **start from noise**. Combined \(f\) is **few-shot**: more labeled faults can enter \(R_\mathrm{anom}\) at sample time without retraining ε or \(e\). Those are extra advantages vs parent-copying refs. They do not cancel the Coverage@τ loss.

**What can be published (honest framing):**

1. **Analysis / negative result** on ESA-ADB: a nominal quantile shell, temporal encoder, and few-shot contrast energies do not cover labeled Anomalies in the small τ-ball. Rares sit on \(Q_q\). That is a real finding.
2. **On GenIAS’s metric stack (ARP + EDI):** combined is competitive-or-better on ARP and clearly better on diversity, while being a *different product* (from-noise + leave-the-donor + growing few-shot refs). See §3.5.
3. **A generation mechanism that is not parent-copying.** Combined \(f\) + optional `start_from_noise` can emit a window **without a donor waveform**, and can **leave** a donor when one is given. The from-noise plot is the exhibit.
4. **Controllable steering** on a frozen continuous score: band / toward train-fold faults / away from rares, same ε-network.

A later methods paper still needs either a coverage-relevant \(f\) (long campaigns) or a metric declared *before* Coverage and ARP disagreed.

`time_recon` itself never trains on anomalies (recon on everyday + rares). Combined \(f\) is **few-shot**: train-fold anomaly/rare embeddings enter \(h_\mathrm{anom}\) and \(h_\mathrm{rare}\) at sample time.

---

## 2. How `real_vs_time_recon_combined_from_noise.png` is made

Script: [`src/anogen/phases/plot_from_noise.py`](../src/anogen/phases/plot_from_noise.py) (`run_plot_from_noise`). Drawing: `save_compare_rows` in [`src/anogen/shell/plots.py`](../src/anogen/shell/plots.py). Gallery on disk: `results/shell_plots/shell_tune/time_recon_combined_from_noise.npz`.

**Columns are independent examples.** There is no shared parent in this figure (unlike `same_parent_*.png`). Each panel is one window, y-scale **per panel** (not shared across the grid).

### 2.1 Generated row (teal) — not a real anomaly

128 windows, **no donor time series**:

```
dummy x_0 = 0     (ignored)
x_T       ~  N(0, I)     in per-channel min-max units
channel   =  0,1,…,5 cycling   (6-way embedding only)
```

Then 50-step DDIM with **combined** \(f\) (tune id 14) and `start_from_noise=True`:

```
f(x̂_0)  =  λ (h_nom − Q_q)²  +  λ_anom h_anom  +  λ_rare (−h_rare)
λ = 0.3,  λ_anom = λ_rare = 1,  unit-normalized ∇,  n_correct = 1
```

- \(e(\cdot)\): fold-0 `time_recon` (`results/shell_enc/time_recon_fold0.pt`).
- \(R_\mathrm{nom}\), \(Q_q\), kernel \(\tau\): from S3 **nominal** parents (`galleries.npz` `cond`), `shell_from_nominal`.
- \(R_\mathrm{anom}\), \(R_\mathrm{rare}\): up to 256 **train-fold** (fold ≠ 0) real anomaly / rare windows. Event-OOF. Few-shot.
- Score: frozen S1 TSDiff. Invert min-max after DDIM so plots are raw units.

The four teal panels are a **farthest-point subset** of those 128 in \(\varphi\) (`diverse_idx`), not the largest-range four (those are `gen_time_recon_combined_from_noise_extreme.png`).

Occupancy vs this encoder’s \(Q_q\) was ~0.76–0.78 (same family as donor-started combined).

### 2.2 Red rows — real ESA-ADB anomalies, by official type (+ overlay)

From S0 `labeled_arrays.npz` + `labeled_windows.csv`, joined to
`anomaly_types.csv` on `event_id`. Full mapping: [ESA_LABELS.md](ESA_LABELS.md).
[`assign_anomaly_kinds`](../src/anogen/shell/morphology.py) uses official
`Length` × `Locality`. The only invented name is **real level shift**.

| Row label | Rule |
|---|---|
| real ESA Point / Local | official Point × Local |
| real ESA Point / Global | official Point × Global |
| real ESA local subsequence | official Subsequence × Local |
| real level shift | official Subsequence × Global **and** abs(μ_L − μ_R) ≥ 0.01 |
| real ESA global subsequence | leftover official Subsequence × Global |

Older PNGs in `results/shell_plots/` used the legacy morphology names (short
spike / campaign). New plots use the rows above.

`pick_kind_examples` takes up to four windows per kind, **typical amplitude** (near the kind’s median range), preferring distinct `event_id`s — not the max-spike cherry-pick.

Teal = generated from noise. Red = real labeled Anomaly windows. Do not read a column as “this generated sample is that kind.” There is no pairing.

---

## 3. Advantages on *continuous* Mission 1 series

41–46 are a repeating analog oscillation (~6 cycles / 4.3 h) plus excursions. Here the Gaussian score is in its comfort zone (unlike extra-M1 stairs). Compared with the two references:

### 3.1 Can exist without a parent (the actual differentiator)

| Method | Needs a real nominal window? | What it emits |
|---|---|---|
| Post-hoc | **yes** | \(x^\mathrm{par} + r_\mathrm{hand}\) (step / pulse / ramp / scale) |
| GenIAS ψ=2 | **yes** | decode(encode(\(x^\mathrm{par}\))); RMSE ~8×10⁻⁴, stays on the donor |
| Unguided ν=1 | donor only as a start for q_sample; ν=1 destroys it | draw from the **nominal** score |
| **Combined + from noise** | **no waveform** | DDIM from \(x_T\sim\mathcal N(0,I)\), steered by \(f\) |

GenIAS and post-hoc cannot produce the teal row. They are **editors**. Combined-from-noise is **generation** (channel index + frozen \(Q_q\) / OOF refs still come from real data; the *path* does not).

On continuous 41–46, unguided and combined both recover the **carrier oscillation**. That is the score doing its job. Combined then adds short localized kicks (up-spike, dropout) that unguided does not aim for.

### 3.2 Can leave the parent when a donor *is* used

Default sampling is still an edit: ν = 0.2, then −∇f. Combined (50 steps, unit ∇, λ_anom/λ_rare = 1) **moves** the window. GenIAS ψ=2 does not (Alg. 2 kept 0% of bins). Post-hoc moves only along four hand templates.

Diversity (mean pairwise \(\|\varphi(g)-\varphi(g')\|\)): combined **6.61** vs unguided 2.34 vs GenIAS 1.09 vs post-hoc 1.21. Eyes: parent oscillation plus a short spike/dropout, not a photocopy.

This is “leave the donor / sit near \(Q_q\),” **not** “cover every ESA kind.” Long campaigns (amp ~0.025, multi-day) are missing in teal and in donor-started combined.

### 3.3 Steering is explicit; references are not

```
f  =  λ (h_nom − Q_q)²     +     λ_anom h_anom     +     λ_rare (−h_rare)
      stay at a rarity level        look like train-fold faults     not like rares
```

Knobs: λ, λ_anom, λ_rare, ν, start_from_noise. Same frozen ε.

Post-hoc: pick a family and a severity. No rarity level, no rare-repulsion.  
GenIAS: one scalar ψ on latent noise. No Anomaly vs Rare Event split.

On continuous series you can *ask* for “oscillation + fault-like tick, not a rare.” Whether \(\varphi\) agrees is the Coverage@τ failure.

### 3.4 Texture vs cartoon (eyes, 41–46 only)

- **Post-hoc** adds a clean geometric overlay. Easy to see, easy to cover a small τ-ball, looks synthetic on a smooth carrier.
- **GenIAS** keeps the real carrier (good texture) and barely edits (bad as an anomaly injector).
- **Combined** keeps a diffusion-like carrier (same family as unguided) and overlays a **short, irregular** excursion. Closer to “real short spike / dropout” than a ruler-drawn pulse; worse than both refs at *matching a specific labeled window* in \(\varphi\).

### 3.5 Extra advantages vs GenIAS / post-hoc

These hold on continuous 41–46 **without** claiming a Coverage@τ win.

**ARP anom (GenIAS’s “anomalous / realistic enough”).**  
GenIAS App. E treats ARP as primary realism: \(1/(1+\mathrm{mean}\,d)\), how close the gallery is to real anomaly queries **on average**. Combined **0.54** vs GenIAS **0.32** and post-hoc **0.32**. On that metric we are not merely comparable — we **beat** both refs. Unguided is still slightly higher (0.57) because many ESA faults sit near the nominal cloud in z-scored \(\varphi\). So: *if the question is GenIAS’s ARP*, combined is “anomalous enough” in their sense. Do **not** translate that into “more extreme than a labeled dropout.” ARP is average closeness, not a τ-ball hit rate. Coverage@τ stays the strict recall check, and there we lose.

**Diversity.**  
Div 6.61 vs unguided 2.34 vs post-hoc 1.21 vs GenIAS 1.09. EDI 2.54 vs 1.89 / 1.37 / 1.12 (near \(\ln 16\)). Parent-copying refs stay in a tight \(\varphi\) clump; combined leaves the donor. GenIAS says credit EDI only if ARP is credible — here ARP is the *higher* of the two stories, so the diversity win is fair to report next to ARP (still not a substitute for Coverage@τ).

**Generate from noise.**  
`start_from_noise=True`: \(x_T\sim\mathcal N(0,I)\), no donor waveform. GenIAS and post-hoc cannot do this; they are editors of a real window. Channel index + frozen \(Q_q\) / OOF refs still come from data; the path does not. Exhibit: `real_vs_time_recon_combined_from_noise.png` (§2). Protocol scores (1536 / 3-fold): `results/shell_noise_score/summary.json` and [PAPER_CANDIDATES.md](PAPER_CANDIDATES.md) §2 — occupancy holds (~0.78 combined), Div ~14, Coverage stays far below GenIAS.

**Few-shot / online-capable refs.**  
Combined \(f\) is semi-supervised at **sample time**, not a second training loop:

```
h_anom(x)  =  soft energy vs  R_anom     (embeddings of labeled Anomalies)
h_rare(x)  =  soft energy vs  R_rare     (embeddings of labeled Rare Events)
```

\(e(\cdot)\) (`time_recon`) and ε stay frozen. New confirmed faults append to \(R_\mathrm{anom}\); new rares append to \(R_\mathrm{rare}\). \(\nabla f\) changes on the next draw. That is the intended online use: as labels arrive, the attract/repel sets grow without retraining the score. GenIAS has one scalar ψ and never sees anomaly embeddings. Post-hoc has four hand templates.

Honest limits: the scored run is **event-OOF** (held-out events stay queries, up to 256 train-fold refs). We have **not** run an incremental “labels arrive over time” curve. Growing \(R_\mathrm{anom}\) does not automatically raise Coverage@τ — we already used a large OOF set and still sit at 0.023. The *capacity* to ingest more faults is real; a monotone online gain is a claim for a later experiment.

### 3.6 What is *not* an advantage

- **Coverage@τ / gap** — post-hoc then GenIAS win. Combined 0.023.
- **ARP as “more anomalous than unguided”** — unguided 0.57 ≥ combined 0.54. Combined’s ARP win is **vs GenIAS/post-hoc**, not vs the unguided ablation.
- **Zero-shot** — combined \(f\) uses train-fold anoms/rares. Band-only `time_recon` is ZS but coverage ~0.08, same as locked shell.
- **All ESA kinds** — nobody does long campaigns. Combined prefers short spikes.
- **Discrete / quantized channels** — Gaussian posterior mean smears stairs; GenIAS/post-hoc copy them. Not this note’s setting.
- **Demonstrated online improvement** — architecture allows growing \(R_\mathrm{anom}\); no arrival-schedule run yet.

---

## 4. One-line comparison (continuous 41–46)

| If you need… | Use |
|---|---|
| A real-looking **copy** of a given window | GenIAS ψ=1 / ψ=2 |
| A **guaranteed visible** step/pulse on that window | Post-hoc |
| GenIAS-style **ARP** (“close enough on average”) **and** high Div, optional **from noise**, growing **few-shot** refs | **time_recon + combined \(f\)** |
| A window **with no donor**, still on the analog carrier, plus a short kick | **time_recon + combined \(f\), from noise** |
| Cite-vs-GenIAS Coverage@τ | You do not have a win; report the table |

---

## 5. Artifacts

| File | What |
|---|---|
| `real_vs_time_recon_combined_from_noise.png` | This figure (teal generated / red real kinds) |
| `gen_time_recon_combined_from_noise.png` | 8 diverse generated windows only |
| `gen_time_recon_combined_from_noise_extreme.png` | 8 largest-range generated |
| `time_recon_combined_from_noise.npz` | 128 windows, \(h\), channel idx |
| `from_noise_summary.json` | occupancy, recipe |
| Donor-started combined | `results/shell_enc_score/time_recon_combined_fold*.npz`, `shell_tune/gen_time_recon_tune_combined.png` |
