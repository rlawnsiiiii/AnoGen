# Mixed galleries: shelves and quiet kinds together

Isolated `results/shell_kindmix/`. Does **not** overwrite `results/shell_s4`, `results/shell_hanom`, or `results/shell_kindproto`.

```bash
.venv/bin/anogen -c configs/shell_mission1.yaml kindmix
```

The problem: **kind-conditional proto** can emit a real level shift, but a gallery aimed at only that kind leaves the quiet blob. **Locked soft** `h_anom` over all of `R_anom` sits on the campaign barycenter, so nearest-φ pairs look like subtle / medium / campaign — and never like a shelf.

This note is the “best of both” without new shape terms (`f_shift`, boxcars). Same `f`, same proto / soft math, different **who is in the sum** and **how samples are allocated**.

**New `kindmix` runs** partition refs with official ESA `Length` × `Locality`
plus the level-shift overlay. Mapping: [ESA_LABELS.md](ESA_LABELS.md).

256-donor ESA eyes: `results/shell_kindmix_esa/` (stratified only).
§2 protocol (1536 / 3-fold): `results/shell_kindmix_score_esa/` — table in
[PAPER_CANDIDATES.md](PAPER_CANDIDATES.md) §2.

**This note’s §6 scores and `results/shell_kindmix/` caches** used the legacy
morphology five-name split (`assign_morphology_kinds`). That morphology 1536
freeze stays in `results/shell_kindmix_score/`. Twomix’s needle slice is
`real ESA Point / Global` (was morphology “short spike”).

---

## 0. What kind-conditional proto did (and did not)

Full run: [KIND_PROTO.md](KIND_PROTO.md). Phase `kindproto`. Idea A in [LEVEL_SHIFT.md](LEVEL_SHIFT.md).

### 0.1 Math

The cached run below used morphology kinds (`assign_morphology_kinds`). On
Mission 1 41–46 that yields five sets

```
R_k  =  { e(a) : a is an Anomaly window we assigned kind k }
```

Kinds `k` in those caches: short spike, short subtle, level shift, medium
event, long campaign. Live `kindmix` uses the ESA names in [ESA_LABELS.md](ESA_LABELS.md) §3.

**Random proto** (H_ANOM) draws one ref from the **pooled** set `R_anom = union_k R_k` (up to 256 mixed faults, ~88% campaigns):

```
i(b)  ~  Uniform({1, …, |R_anom|})
h_anom^(b)  =  || e(x̂_0^(b)) − a_{i(b)} ||²
```

**Kind-conditional proto** keeps the same energy, but the draw is inside one kind. One gallery per `k`:

```
i(b)  ~  Uniform({1, …, |R_k|})
h_anom^(b)  =  || e(x̂_0^(b)) − a_{i(b)} ||²
```

The rest of combined `f` is unchanged (`λ_rare = 0` in that run):

```
f  =  λ (h_nom − Q_q)²  +  λ_anom h_anom
```

Guided DDIM subtracts `∇_x̂ f` as in [STEERING_MATH.md](STEERING_MATH.md). The proto index `i(b)` is drawn once and **held for the whole trajectory**.

Level-shift and medium `R_k` **leak fold 0** on 41–46 (every real shelf and every medium window is fold 0). Those two galleries are a capability test, not a few-shot cite.

### 0.2 Result

| Aimed kind | Worked? |
|---|---|
| **level shift** | **Yes.** P(abs(μ_L − μ_R) ≥ 0.01) ≈ 0.91–1.00. Eyes: 0.95→0.78 shelf. Near-copies of the six leaked refs (median raw L2 0.29). |
| short spike | Partial: a needle, usually **down** at ~2 h (reals are **up**). |
| short subtle | **No.** Carrier + jitter. The kind is “amp < 0.1”; unit ∇ un-subtles it. |
| medium event | **No.** Edge ticks, lost oscillation. 30 leaked refs did not help: e(medium) ≈ e(everyday). |
| **long campaign** | **No.** Whole-window DC bump + edge ticks. A campaign is span ≥ 4000 bins; one W=512 crop cannot be that kind. The real crop *looks* like the everyday carrier. |

Coverage@τ stayed 0 (φ z-scores DC). 200 from-noise DDIM steps added jitter, not missing kinds.

So: **kindproto restores shelves (and needles) when you point at those embeddings. It does not restore long campaign / medium / subtle as distinct in-window types.** Soft “recovered” those quiet kinds only because the gallery *was* the campaign barycenter and nearest-φ matches a quiet real to a quiet generated wave.

Best of both = a **mixed gallery**, not one force that is a shelf and a campaign at once.

---

## 1. Locked soft (what we had before proto)

```
h_soft(z)  =  −τ log Σ_{i in R_anom}  exp( −||z − a_i||² / τ )

∇_z h_soft  =  2 ( z − Σ_i w_i a_i )
w           =  softmax(−d_i² / τ)
```

Start on a nominal parent (ν = 0.2): far from every `a_i`, so `w_i ≈ 1/K` and the walk aims at the **mean of all** train-fold anomaly embeddings. On 41–46 that mean is “average campaign.” Quiet nearest-φ pairings come for free. Shelves do not.

---

## 2. Variant A — stratified proto

Same proto energy as kindproto. **One** gallery. For each window, draw the kind first (uniform, not empirical), then a proto inside that kind:

```
k(b)  ~  Uniform({spike, subtle, shift, medium, campaign})
i(b)  ~  Uniform(R_{k(b)})
h_anom^(b)  =  || e(x̂_0^(b)) − a_{i(b)} ||²
```

Empirical P(k) is ≈ 88% campaign and ≈ 1% shift — that is locked soft again. Uniform P(k) puts ≈ 20% of a 256-gallery on shelves (~51 windows).

Implementation: split the 256 parents into five contiguous slices (52+51+51+51+51) and run proto on `R_k` for each slice. Concatenate.

Expect: shift slice = shelves; campaign / medium / subtle slices = quiet carrier (what soft looked like); spike slice = needles. Nearest-φ can then find **all** real kinds in **one** gallery.

---

## 3. Variant B — kind-balanced soft

One energy, one walk per window, no per-sample kind draw. Keep the KDE, stop letting campaigns own the sum.

```
h_kind(z)  =  −τ log Σ_{k=1..K}  π_k · ( 1/|R_k|  Σ_{i in R_k} exp( −||z − a_i||² / τ ) )

π_k  =  1/K
```

The inner mean (not the raw sum) is required: otherwise |R_campaign| ≫ |R_shift| still dominates. Far away, ∇h aims at the **mean of the five kind-means**. Nearby, softmax falls into one cluster.

Code: `kind_balanced_soft_energy` / `anom_energy_kind="kind_soft"` in [`src/anogen/shell/steer.py`](../src/anogen/shell/steer.py).

Risk: the far-field point is a **compromise** (a bit of shelf smear on a carrier). If every panel looks muddy, prefer A or C — mixture of **samples**, not mixture of **forces**.

Do **not** only upsample the six shelves inside a flat KDE (256 copies of one shelf). Then every sample becomes a shelf again.

---

## 4. Variant C — two-recipe mix

No new energy. Concatenate two (three) recipes you already trust:

| Slice | Fraction | Energy | Refs |
|---|---:|---|---|
| soft-all | 60% | locked `h_soft` | first 256 event-OOF anomalies (campaign-heavy) |
| shift proto | 20% | kind-conditional proto | `R_shift` (fold-0 leak) |
| spike proto | 20% | kind-conditional proto | `R_spike` (event-OOF) |

The spike slice is optional in the idea; this run includes it so needles are not only 20% of stratified. Score Coverage / ARP / Div on the **union**. Eyes: nearest-φ should pick quiet kinds from the soft slice and shelves from the shift slice.

---

## 5. What stayed the same

| Piece | This run |
|---|---|
| Score ε | frozen `results/shell_s1/denoiser.pt` |
| Encoder | fold-0 `time_recon` / `time_both` |
| Band | soft KDE, nominal τ, `Q_q` |
| λ, ν, steps | id 14 band + `h_anom`: 0.3 / 0.2 / 50, unit ∇, `λ_rare=0` |
| Donors | first 256 of S3 `cond` |
| Coverage τ | S4 freeze 0.105, fold-0 queries only |

Not a new encscore / S4 freeze.

---

## 6. Scores (morphology caches)

Compact 256 / fold-0 on the **old morphology kinds**. **Do not mix** with locked S4 or with the ESA 256 run in `results/shell_kindmix_esa/`. Coverage@τ is **0** on every row.

| Encoder | Variant | Cov. | ARP anom | ARP rare | Div | Occ. | frac amp≥0.1 | frac shift |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| time_recon | **stratified** | 0 | **0.34** | 0.43 | **6.83** | 0.00 | 0.71 | **0.24** |
| time_recon | kind_soft | 0 | 0.19 | 0.24 | 2.36 | 0 | 0.00 | 0 |
| time_recon | twomix | 0 | 0.27 | 0.31 | 5.64 | 0 | 0.38 | **0.20** |
| time_both | **stratified** | 0 | **0.37** | 0.45 | **7.02** | 0.00 | 0.71 | **0.22** |
| time_both | kind_soft | 0 | 0.17 | 0.21 | 2.47 | 0 | 0.01 | 0 |
| time_both | twomix | 0 | 0.24 | 0.28 | 5.98 | 0.01 | 0.38 | **0.18** |

`frac shift` is P(abs(μ_L − μ_R) ≥ 0.01). On stratified / twomix that is about the **allocated shift slice** (~20%). Those windows are real shelves. kind_soft’s far-field compromise emitted **no** shelves.

Stratified / twomix put needles and shelves in the **same** gallery as quiet crops, so nearest-φ can pair all five kinds. That is a mixed bag, not a better campaign generator (the campaign slice is still kind-conditional proto). Kind-balanced soft collapsed to one quiet mode.

### 6.1 ESA-kind stratified (256, 2026-09-10)

`results/shell_kindmix_esa/`. Four present kinds (Point/Local empty), 64+64+64+64. Fold-0 only. Coverage@τ still **0**.

| Encoder | Cov. | ARP anom | ARP rare | Div | Occ. | frac amp≥0.1 | frac shift |
|---|---:|---:|---:|---:|---:|---:|---:|
| time_recon | 0 | 0.27 | 0.31 | 8.09 | 0.01 | 0.79 | **0.25** |
| time_both | 0 | **0.39** | 0.47 | 7.21 | 0.01 | 0.75 | **0.25** |

Aimed-slice shift rate is ~1.00 (recon) / 0.97 (both). Point/Global slice amp≥0.1 is 0.92–1.00. Local / global subsequence slices do not match the real crops. Protocol 1536: [PAPER_CANDIDATES.md](PAPER_CANDIDATES.md) §2.

---

## 7. Plots

Same pairing as `results/shell_hanom/plots/real_kinds_vs_time_recon_proto.png`: red = real kind examples, teal = **nearest in φ** from that variant’s **mixed** 256-window gallery.

Morphology caches (`results/shell_kindmix/plots/`):

| File | What |
|---|---|
| `plots/real_kinds_vs_{time_recon,time_both}_{stratified,kind_soft,twomix}.png` | Hanom-style kind rows |
| `plots/real_kinds_vs_{time_recon,time_both}_mix.png` | Real vs three variants, nearest-φ, shared y per column |

ESA stratified (`results/shell_kindmix_esa/plots/`):

| File | What |
|---|---|
| `real_kinds_vs_{time_recon,time_both}_stratified.png` | Nearest-φ to each ESA kind |
| `slice_*_grid.png` | Real vs best-stat **aimed slice** |
| `slice_*_{esa_point_global,esa_local_subseq,level_shift,esa_global_subseq}.png` | Four reals vs four aimed windows |

Look at Point/Global and level shift first. Local / global subsequence should stay a quiet real crop, not jitter.

---

## 8. Do not

- Overwrite `results/shell_s4` or run `anogen s4` / `s6`.
- Cite these 256 / fold-0 numbers as a win-rule table.
- Add `h_soft(R_all) + λ h_proto(R_shift)` on **every** sample (forces fight).
- Expect a W=512 crop to satisfy “span ≥ 4000” as a label. Campaign success here means “looks like the real campaign *crop*.”
- Treat morphology names (spike / campaign) as official ESA-ADB types.
  Live kinds are Length × Locality plus one overlay ([ESA_LABELS.md](ESA_LABELS.md)).
- Mix `results/shell_kindmix/` (morphology) with `results/shell_kindmix_esa/` (ESA).

---

## 9. Hybrid slices + λ / λ_anom sweep

How the per-slice energy is chosen, and what you can actually aim at:
[HYBRID.md](HYBRID.md). Recommended \((\lambda,\lambda_\mathrm{anom})\) and
the grid: [KIND_MIX_HTUNE.md](KIND_MIX_HTUNE.md). Shortlist eyes
(`time_recon` / `time_both` × hybrid / needles × `l03_a*`):
[FINDINGS.md](FINDINGS.md). Full-grid plots:
[`docs/kindmix_htune/`](kindmix_htune/INDEX.txt).

```bash
.venv/bin/anogen -c configs/shell_mission1.yaml kindmixhtune
```

**Recommend `hybrid_needles` at locked parent50:** \(\lambda=0.3\),
\(\lambda_\mathrm{anom}=1.0\). Proto on level shift and Point/Global; band
on local / global subsequence. Your hybrid (band on points too) quiets the
gallery but loses needles. Full table in [KIND_MIX_HTUNE.md](KIND_MIX_HTUNE.md).
