# Semifinal results — four paper candidates

Collection for the GenFSDiff write-up. ESA-ADB Mission 1 **channels 41–46**,
frozen S4 \(\tau \approx 0.105\), \(\varphi=\) `feature_pack_v1`, **1536**
S3 donors, **3-fold OOF**. Plots are copies under
[`figures/`](figures/). Sources stay in `docs/findings/`,
`docs/kindmix_htune/`, and `results/shell_plots/shell_tune/`.

This is **not** a new S4 freeze and does **not** beat GenIAS on Coverage@τ.
Scores here follow GenIAS: **ARP** (realism) and **EDI** (diversity).

---

## 1. The four candidates

Same frozen TSDiff score. Parent start is \(\nu=0.2\), 50 DDIM, unit
\(\nabla\), \(\lambda=0.3\).

| # | Name | Encoder | \(f\) | Why it is a candidate |
|---|---|---|---|---|
| **C1** | `time_both` + **hybrid_needles** | temporal map, recon **+** auxiliary SupCon (few-shot encoder) | proto on **level shift** and **Point/Global**; **band only** on local / global subsequence; \(\lambda_\mathrm{anom}=1\), \(\lambda_\mathrm{rare}=0\) | Best steered **ARP**. One gallery with needles, shelves, and quiet crops. |
| **C2** | `time_recon` + **hybrid_needles** | temporal map, recon MSE only (no anomaly labels in \(e\)) | same hybrid_needles \(f\) | Same mixed-kind recipe with a zero-shot-legal encoder. Highest **Div** among parent-start rows. |
| **C3** | `time_recon` + **combined** | same as C2 | soft \(h_\mathrm{anom}\) over all train-fold faults **and** rare repulsion: \(\lambda_\mathrm{anom}=\lambda_\mathrm{rare}=1\) | Best occupancy (~0.78). Least-bad *encscore* row. Has the from-noise eyes. |
| **C4** | `time_both` + **combined** | same as C1 | same combined \(f\) as C3 | Same \(f\), few-shot encoder. Occupancy holds; ARP a bit below C3. |

**Default if you need one row:** **C1**.
**Default if you need a zero-shot encoder:** **C2**.
**Default if you need the shell / occupancy story (or from-noise):** **C3**.

Hybrid_needles \(f\):

```
f = λ (h_nom − Q_q)²  +  λ_anom ‖ e(x̂_0) − a_{i(b)} ‖²
```

with \(\lambda_\mathrm{anom}=0\) on subsequence slices. Combined \(f\):

```
f = λ (h_nom − Q_q)²  +  λ_anom h_anom^soft  +  λ_rare (−h_rare)
```

\(e(\cdot)\) is the unpooled \(8\times 64=\mathbb{R}^{512}\) map.
Proto \(a_{i(b)}\) is **one frozen anomaly embedding**, drawn once and held
for the DDIM. Combined \(h_\mathrm{anom}\) is a **KDE over all** train-fold
anomaly embeddings (campaign barycenter).

---

## 2. Scores (ARP and EDI)

ARP is comparable across every row:
\(\mathrm{ARP}=1/(1+\mathrm{mean}_q \min_g\|\varphi(q)-\varphi(g)\|)\).

EDI is Shannon entropy of the gallery over **16 k-means bins of a union**.
Different footnotes are **different partitions**. Do not subtract EDI
across `*`, unmarked, and `¶`. Div (mean pairwise \(\|\varphi(g)-\varphi(g')\|\))
does not depend on the union.

| Method | Start | ARP anom | EDI | Div | Occ. | Cov@τ anom |
|---|---|---:|---:|---:|---:|---:|
| **C1** time_both + hybrid_needles | parent \(\nu=0.2\) | **0.584** | **2.60\*** | 7.52 | 0.47 | 0.039 |
| **C2** time_recon + hybrid_needles | parent \(\nu=0.2\) | 0.576 | 2.54\* | **7.94** | 0.45 | 0.023 |
| **C3** time_recon + combined | parent \(\nu=0.2\) | 0.536 | 2.54 | 6.61 | **0.78** | 0.023 |
| **C4** time_both + combined | parent \(\nu=0.2\) | 0.512 | 2.51 | 6.23 | 0.77 | 0.011 |
| C3 from noise | \(x_T\sim\mathcal N(0,I)\) | 0.520 | 2.41¶ | 14.34 | 0.78 | 0.014 |
| C1 from noise | \(x_T\sim\mathcal N(0,I)\) | 0.578 | 2.64¶ | 14.15 | 0.44 | 0.020 |
| Unguided DDIM \(\nu=1\) | q_sample of a parent | 0.568 | 1.89 / 1.58\* | 2.34 | — | 0.109 |
| GenIAS \(\psi=2\) | VAE inflate | 0.317 | 1.12 / 0.85\* | 1.09 | — | 0.195 |
| Post-hoc inject | step/pulse/ramp/scale | 0.319 | 1.37 / 0.93\* | 1.21 | — | **0.267** |

\* Hybrid-table union (`shell_kindmix_score_hybrid`, S4 + encscore + four
hybrid fold-0 galleries). Unmarked combined EDI is the encscore 9-method
union. ¶ From-noise union (`shell_noise_score`). GenIAS/post-hoc ARP is
the same number in every table; their EDI is **not**.

Artifacts: `results/shell_kindmix_score_hybrid/summary.json`,
`results/shell_enc_score/summary.json`,
`results/shell_noise_score/summary.json`.

### How to read this for the paper

- On **ARP**, C1–C4 all beat GenIAS (0.32) and post-hoc (0.32). C1 also
  edges unguided (0.568).
- On **EDI**, C1–C4 sit near \(\ln 16 \approx 2.77\). That is spread in
  \(\varphi\), not proof of ESA-type coverage.
- **Coverage@τ stays ≪ GenIAS.** C1’s 0.039 is still ~5× below 0.195, and
  it is **global subsequence only**. Aimed Point/Global and level shift
  are Coverage **0** (`feature_pack_v1` z-scores DC).
- Occupancy is the band-slice fraction. Combined (C3/C4) sits on \(Q_q\);
  hybrid_needles (C1/C2) is ~half band / half proto.
- Level-shift proto **leaks fold 0** (all six `id_83` shelves). Cite as a
  capability, not few-shot OOF.
- Local vs global subsequence are **not** two generators: same band
  process, different plot labels.

---

## 3. What the eyes show (one line each)

| Kind | C1 / C2 (hybrid_needles) | C3 / C4 (combined) |
|---|---|---|
| Point / Global | **Needles** when proto is on (\(\lambda_\mathrm{anom}=1\)). | Short spikes; no type switch. |
| Level shift | **Shelves** (fold-0 leak). \(\lambda_\mathrm{anom}=0.3\) kills the shelf. | Nearest-\(\varphi\) is still the carrier — **no shelf**. |
| Local subsequence | Quiet carrier (band). Not aimed. | Quiet / small hitch. |
| Global subsequence | Quiet carrier (band). Majority ESA kind; not a distinct in-window campaign. | Same; combined does not invent multi-hour campaigns. |

---

## 4. Candidate C1 — `time_both` + hybrid_needles

**ARP 0.584 · EDI 2.60\*** · Div 7.52 · occ. 0.47 · Cov 0.039.

Encoder: fold-0 `time_both` (recon + auxiliary 3-class SupCon). SupCon did
**not** separate anom/rare/nom (\(h\) AUROC ~chance); treat it as a
few-shot regularizer, not a classifier.

Plots are fold-0, 128 S3 parents unless noted (htune / findings). Protocol
ARP/EDI above are 1536 / 3-fold.

### 4.1 Aimed overview (all four kinds)

Red = real ESA crop. Blue rows = this recipe at \(\lambda_\mathrm{anom}=1.0 / 0.5 / 0.3\).
Use the **\(\lambda_\mathrm{anom}=1.0\)** row (locked weights).

![C1 aimed overview](figures/c1_time_both_hybrid_needles/01_aimed_overview.png)

### 4.2 Point / Global — aimed (4 examples)

Needles \(\lambda_\mathrm{anom}=1\) is the C1 slice. `hybrid` (green) is
band-on-points and does **not** make needles — that is why the recipe is
hybrid_needles.

![C1 Point/Global aimed](figures/c1_time_both_hybrid_needles/02_point_global_aimed.png)

### 4.3 Point / Global — nearest in \(\varphi\)

![C1 Point/Global nearest](figures/c1_time_both_hybrid_needles/03_point_global_nearest.png)

### 4.4 Level shift — aimed (4)

Shelves at \(\lambda_\mathrm{anom}\ge 0.5\). Bottom row (\(\lambda_\mathrm{anom}=0.3\))
is a failed shelf.

![C1 level shift aimed](figures/c1_time_both_hybrid_needles/04_level_shift_aimed.png)

### 4.5 Level shift — nearest in \(\varphi\)

![C1 level shift nearest](figures/c1_time_both_hybrid_needles/05_level_shift_nearest.png)

### 4.6 Local subsequence — aimed (band, not proto)

Quiet oscillation. All shortlist rows are the same band process.

![C1 local subsequence aimed](figures/c1_time_both_hybrid_needles/06_local_subseq_aimed.png)

### 4.7 Global subsequence — aimed (band, not proto)

![C1 global subsequence aimed](figures/c1_time_both_hybrid_needles/07_global_subseq_aimed.png)

### 4.8 λ grid, aimed (htune)

Locked cell is `l03_a10`. Stronger shell (`l10`, `l15`) hashes quiet
kinds and kills shelves.

![C1 htune aimed](figures/c1_time_both_hybrid_needles/08_htune_aimed.png)

### 4.9 λ grid, nearest in \(\varphi\)

![C1 htune nearest](figures/c1_time_both_hybrid_needles/09_htune_nearest.png)

### 4.10 Recommended-cell aimed strip

![C1 recommend aimed](figures/c1_time_both_hybrid_needles/10_recommend_aimed.png)

---

## 5. Candidate C2 — `time_recon` + hybrid_needles

**ARP 0.576 · EDI 2.54\*** · Div **7.94** · occ. 0.45 · Cov 0.023.

Same \(f\) as C1. Encoder never sees anomaly labels (recon on everyday +
Rare Events only). Eyes: slightly noisier shelves (early jitter on the
step), needles still fire.

### 5.1 Aimed overview

![C2 aimed overview](figures/c2_time_recon_hybrid_needles/01_aimed_overview.png)

### 5.2 Point / Global — aimed

![C2 Point/Global aimed](figures/c2_time_recon_hybrid_needles/02_point_global_aimed.png)

### 5.3 Point / Global — nearest in \(\varphi\)

![C2 Point/Global nearest](figures/c2_time_recon_hybrid_needles/03_point_global_nearest.png)

### 5.4 Level shift — aimed

![C2 level shift aimed](figures/c2_time_recon_hybrid_needles/04_level_shift_aimed.png)

### 5.5 Level shift — nearest in \(\varphi\)

![C2 level shift nearest](figures/c2_time_recon_hybrid_needles/05_level_shift_nearest.png)

### 5.6 Local subsequence — aimed (band)

![C2 local subsequence aimed](figures/c2_time_recon_hybrid_needles/06_local_subseq_aimed.png)

### 5.7 Global subsequence — aimed (band)

![C2 global subsequence aimed](figures/c2_time_recon_hybrid_needles/07_global_subseq_aimed.png)

### 5.8 λ grid, aimed

![C2 htune aimed](figures/c2_time_recon_hybrid_needles/08_htune_aimed.png)

### 5.9 λ grid, nearest in \(\varphi\)

![C2 htune nearest](figures/c2_time_recon_hybrid_needles/09_htune_nearest.png)

### 5.10 Recommended-cell aimed strip

![C2 recommend aimed](figures/c2_time_recon_hybrid_needles/10_recommend_aimed.png)

---

## 6. Candidate C3 — `time_recon` + combined \(f\)

**ARP 0.536 · EDI 2.54** · Div 6.61 · occ. **0.78** · Cov 0.023.

Soft \(h_\mathrm{anom}\) over the pooled fault set + rare repulsion.
Emits **spikes** and leaves the parent; does **not** emit a 0.95→0.80
shelf. Morphology names in some of these figures (`short spike`, `long
campaign`, …) are an older overlay; ESA names are in C1/C2.

### 6.1 Eight diverse generated windows (parent start)

![C3 gallery](figures/c3_time_recon_combined/01_gallery_8.png)

### 6.2 Real kinds vs nearest generated in \(\varphi\)

Nearest match to a real **level shift** is a carrier oscillation — combined
does not make the shelf.

![C3 real kinds nearest](figures/c3_time_recon_combined/02_real_kinds_nearest.png)

### 6.3 Same nominal parent, recipe overlay

Red/violet = combined vs occupancy (orange hash) vs contrast (green).
Combined is the tame leave-the-parent edit.

![C3 same parent overlay](figures/c3_time_recon_combined/03_same_parent_overlay.png)

### 6.4 Same parent, grid

![C3 same parent grid](figures/c3_time_recon_combined/04_same_parent_grid.png)

### 6.5 Combined on both encoders vs real kinds

![C3 both encoders](figures/c3_time_recon_combined/05_real_kinds_both_encoders.png)

### 6.6 Real vs tune recipes (band / occupancy / contrast / combined)

![C3 real vs tune](figures/c3_time_recon_combined/06_real_vs_tune_recipes.png)

### 6.7 From noise — eight diverse windows

Same combined \(f\), **no donor waveform** (\(x_T\sim\mathcal N(0,I)\)).
Protocol: ARP **0.520**, EDI **2.41¶**, Div **14.3**, occ. **0.78**.
ARP holds; Div doubles; Coverage does not rise.

![C3 from noise gallery](figures/c3_time_recon_combined/07_from_noise_gallery.png)

### 6.8 From noise — largest-range eight (not the diverse subset)

![C3 from noise extreme](figures/c3_time_recon_combined/08_from_noise_extreme.png)

### 6.9 From noise vs real ESA kinds

Teal = generated (independent columns, no shared parent). Red = real
kinds. Still no shelf in the generated row.

![C3 from noise vs real](figures/c3_time_recon_combined/09_from_noise_vs_real_kinds.png)

C1 from noise (no extra eyes on disk): ARP **0.578**, EDI **2.64¶**,
Div 14.2, occ. 0.44, Cov 0.020.

---

## 7. Candidate C4 — `time_both` + combined \(f\)

**ARP 0.512 · EDI 2.51** · Div 6.23 · occ. 0.77 · Cov 0.011.

Same combined energy as C3. Eyes are the same family (spikes, no shelf).
From-noise protocol (no dedicated PNGs): ARP 0.551, EDI 2.39¶, Div 13.9,
occ. 0.78, Cov 0.027 — the only combined row whose Coverage **rises**
off a parent.

### 7.1 Eight diverse generated windows

![C4 gallery](figures/c4_time_both_combined/01_gallery_8.png)

### 7.2 Real kinds vs nearest generated in \(\varphi\)

![C4 real kinds nearest](figures/c4_time_both_combined/02_real_kinds_nearest.png)

### 7.3 Same nominal parent, overlay

![C4 same parent overlay](figures/c4_time_both_combined/03_same_parent_overlay.png)

### 7.4 Same parent, grid

![C4 same parent grid](figures/c4_time_both_combined/04_same_parent_grid.png)

### 7.5 Real vs tune recipes

![C4 real vs tune](figures/c4_time_both_combined/05_real_vs_tune_recipes.png)

### 7.6 Combined on both encoders vs real kinds

Same file as C3 §6.5.

![C4 both encoders](figures/c4_time_both_combined/06_real_kinds_both_encoders.png)

---

## 8. Baselines (for the figure board, not candidates)

ARP/EDI in the table in §2. These strips are the locked S4 galleries
(1536 protocol), not hybrid_needles.

![Real anomalies](figures/baselines/real_anomalies.png)

![GenIAS](figures/baselines/gen_genias.png)

![Post-hoc](figures/baselines/gen_posthoc.png)

---

## 9. Plot grammar

| Tag | Meaning |
|---|---|
| **Aimed** | Best-stat window **inside the allocated kind slice**. “What we asked for.” |
| **Nearest** | Nearest in locked \(\varphi\) from the **whole** gallery. A quiet real crop often matches a band window from another slice — not proof of aiming. |
| **Gallery 8** | Farthest-point subset in \(\varphi\) (diverse), except `*_extreme.png` = max range. |
| **Same parent** | One nominal donor; recipes overlaid. Combined / hybrid_needles are different runs. |
| Eyes vs scores | C1/C2 PNGs = 128-donor htune/findings. ARP/EDI = 1536 / 3-fold. Do not put htune Coverage (almost always 0) in a protocol table. |

---

## 10. What not to claim

- “We beat GenIAS at generating ESA Anomalies.” Coverage@τ says no.
- “Trained only on nominals.” Denoiser = everyday **+ rares**. C1/C4
  encoder sees train-fold anomalies. Combined / proto refs are few-shot
  at **sample** time.
- “Contrastive encoder separates nom / rare / anom.” It does not.
- “All ESA types.” Point/Global and (leaky) level shift, yes. Local /
  global subsequence, no as controllable types. Point/Local empty on 41–46.
- Mixing EDI `*` / unmarked / `¶`.
- Overwriting `results/shell_s4` or running S6.

Longer score table and kill list: [`docs/PAPER_CANDIDATES.md`](../docs/PAPER_CANDIDATES.md).
Hybrid allocation: [`docs/HYBRID.md`](../docs/HYBRID.md).
Sampling: [`docs/STEERING_MATH.md`](../docs/STEERING_MATH.md).
