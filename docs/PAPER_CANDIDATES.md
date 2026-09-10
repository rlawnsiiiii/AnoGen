# Paper candidates — models, logic, vs GenIAS

What we have actually run on ESA-ADB Mission 1 channels 41–46, and what is worth a paper.

Locked training log: [RESULTS.md](RESULTS.md). Sampling math: [STEERING_MATH.md](STEERING_MATH.md). Encoder variants: [ENCODER_ABLATION.md](ENCODER_ABLATION.md). Adapters: [S5.md](S5.md). Other channels / Mission 2: [XFER.md](XFER.md) (separate τ and galleries — do not mix into the tables below). Combined \(f\) vs GenIAS/post-hoc (continuous 41–46, from-noise figure): [COMBINED_VS_REFS.md](COMBINED_VS_REFS.md). Replacing the \(h_\mathrm{anom}\) KDE (nearest / kNN / proto): [H_ANOM.md](H_ANOM.md) — isolated, not a new S4 freeze. Code / which script scored which row: [REPO.md](REPO.md). Waveforms: `results/shell_plots/shell_tune/` and `results/shell_plots/shell_s5/`.

**Two different stories — do not mix them.**

1. **Locked S4 (cite vs GenIAS).** Zero-shot shell = pooled recon encoder + **band-only** steering. No `time_both`, no contrastive energies. Win rule **lost**.
2. **Later 1536-window scores** (`encscore`, then `s5`, then `kindmixscore` / `kindmixscorehybrid`, frozen S4 τ). `time_recon` / `time_both` × band, combined \(f\), **stratified proto**, or **hybrid / hybrid_needles**; S5 adapters on the same score. Same φ / τ / donor count as S4. **Does not overwrite** `results/shell_s4`. Combined, stratified, hybrid, and S5 are few-shot (event-OOF). Fold-0 encoder weights only.

S6 stays sealed.

---

## 0. Menu: encoder × steering

Every diffusion sample is the same edit. Only \(e(\cdot)\) and \(f\) change.

```
nominal donor  --q_sample(ν)-->  x_{t*}  --DDIM − ∇f-->  edited window
```

```
f  =  λ (h_nom − Q_q)²     +     λ_anom h_anom     +     λ_rare (−h_rare)
       └── band / occupancy ─┘   └── toward faults ─┘   └── away from rares ─┘
```

\(h\) is always soft energy in encoder space. Contrast refs are **train-fold** only (few-shot). Band-only is zero-shot.

### Encoders we trained

Shared 3-layer stride-2 conv → map \((B,64,64)\). Then pool or keep time.

| Encoder | \(e(x)\) | Trained on | Sees train-fold anomalies? | Shot | Params | Where used |
|---|---|---|---|---|---:|---|
| **pool_recon (S2, locked)** | \(\mathbb{R}^{32}\) after GAP | recon MSE, everyday + rares, 400 steps | no | **ZS** | 181k | Official S4 “Shell ZS” |
| **time_recon** | flatten \(8\times 64=\mathbb{R}^{512}\), no GAP | recon MSE, same pool, 800 steps | no | **ZS** | 44k | Enc ablation + tune plots |
| pool_supcon | \(\mathbb{R}^{32}\) | 3-class SupCon | yes | FS | 181k | Enc only (collapsed) |
| time_supcon | \(\mathbb{R}^{512}\) | 3-class SupCon | yes | FS | 44k | Enc only (collapsed) |
| **time_both** | \(\mathbb{R}^{512}\) | MSE + SupCon, 1200 steps | **yes** | **FS** | 44k | Enc ablation + tune plots |

`time_recon` is the ZS-legal upgrade of S2 (local \(\nabla h\), no fault labels). `time_both` is few-shot: the encoder itself has seen train-fold anomalies.

### Stratified proto (`time_recon` / `time_both`)

The 256-donor eyes in [KIND_MIX.md](KIND_MIX.md) used this recipe. §2 scores it on the locked 1536 / 3-fold protocol (`kindmixscore`). Same frozen TSDiff score and same fold-0 encoder weights as encscore. **Not** a new S4 freeze. 256-donor kindmix numbers stay out of the table below.

Both models share the same \(e\) family and the same \(f\). They differ only in how \(e\) was trained.

```
time_recon   e(x) ∈ R^{512}   =  flatten(8 × 64 time map)     recon MSE only (ZS-legal)
time_both    e(x) ∈ R^{512}   =  same map                     MSE + 3-class SupCon (FS encoder)
```

Sampling is still the locked edit. Combined (id 14) used a **soft** \(h_\mathrm{anom}\) over the pooled train-fold anomaly set and \(\lambda_\mathrm{rare}=1\). Stratified keeps the band, **drops the rare term**, and replaces the anomaly KDE with **one prototype per generated window**, with kinds drawn **uniformly** (not empirically — campaigns are ~88% of \(R_\mathrm{anom}\)):

```
nominal donor  --q_sample(ν=0.2)-->  x_{t*}  --DDIM − ∇f-->  edited window

f  =  λ (h_nom − Q_q)²  +  λ_anom h_anom^(b)
     └── band, λ=0.3 ─┘    └── proto, λ_anom=1 ─┘
λ_rare  =  0
ν = 0.2,  50 DDIM steps,  unit ∇,  n_correct = 1
```

\(h_\mathrm{nom}\) is the locked soft energy vs nominal refs (same \(\tau\), \(Q_q\) as that encoder’s band). \(h_\mathrm{anom}\) is **not** that KDE. Stratified now partitions refs with official ESA `Length` × `Locality` plus the level-shift overlay (`kinds_for_windows`, [ESA_LABELS.md](ESA_LABELS.md)). The 2026-09-10 morphology five-name split is frozen in `results/shell_kindmix_score/` and is **not** this table.

```
R_k  =  { e(a) : a is a labeled Anomaly of ESA kind k,
                event fold ≠ query fold if that set is non-empty }

k ∈ { Point/Local, Point/Global, local subsequence, level shift, global subsequence }
```

On 41–46 pretest, **Point/Local is empty** (official `id_42` is not in the pool). The other four kinds are present. Level-shift \(R_k\) is empty event-OOF when the query fold is 0 (all six `id_83` shelf crops are fold 0) — that slice then uses the fold-0 windows (`leaked=True`). Point/Global, local subsequence, and global subsequence stay event-OOF. Folds 1 and 2 can use the fold-0 shelves as OOF refs.

For generated window \(b\), draw the kind first (uniform over **present** kinds), then one proto inside that kind, and **hold** \(i(b)\) for the whole DDIM trajectory:

```
k(b)        ~  Uniform({Point/Global, local subseq, shift, global subseq})
i(b)        ~  Uniform(R_{k(b)})
h_anom^(b)  =  || e(x̂_0^(b)) − a_{i(b)} ||²
```

Code: `proto_energy` / `anom_energy_kind="proto"`. Implementation does not sample \(k(b)\) independently per row: it **splits the 1536 donors into four contiguous slices** (384+384+384+384) and runs proto on \(R_k\) for each slice, then concatenates. The 256-donor ESA run is 64+64+64+64.

Far from the proto, \(\nabla h_\mathrm{anom}\) is \(2(e(x)-a_{i(b)})\) in encoder space, backpropped through \(e\) onto the Tweedie \(\hat x_0\). Because \(e\) is a time map (no GAP), that gradient can write a short tick (spike slice) or a DC step (shift slice) instead of only the campaign barycenter.

Do **not** mix this with combined \(f\): combined uses soft \(h_\mathrm{anom}\) over all train-fold faults **and** \(\lambda_\mathrm{rare}=1\). Stratified is few-shot on the proto refs, zero-shot on the score and (for `time_recon`) on the encoder.

### Steering recipes we sampled

| Recipe | λ | λ_anom, λ_rare | ν | steps / n_correct | ∇ | Shot | What it is trying to do |
|---|---:|---|---:|---|---|---|---|
| **S4 band** (locked) | 0.3 | 0, 0 | 0.2 | 20 / 1 | clip | ZS | Sit on the nominal quantile \(Q_q\) |
| **Occupancy** (tune z7) | 1.5 | 0, 0 | 0.2 | 50 / 4 | unit | ZS | Same, harder (land in the band) |
| **Contrast** (tune c15) | **0** | **3, 3** | 0.5 | 50 / 1 | unit | FS | Look like train-fold faults, not rares |
| **Combined** (tune id 14 / STEERING_MATH) | 0.3 | 1, 1 | 0.2 | 50 / 1 | unit | FS | Band **and** contrast together |
| **Stratified proto** (kindmix A / parent50) | 0.3 | 1, **0** | 0.2 | 50 / 1 | unit | FS | Band + one proto per window, kinds allocated uniformly |

Locked **Shell ZS = pool_recon × S4 band**.  
Newer plots = `{time_recon, time_both} × {band, occupancy, contrast, combined}`.

### What each combo looks like (eyes)

| | S4 band | Occupancy (strong λ) | Contrast | Combined (band + contrast) |
|---|---|---|---|---|
| **pool_recon (S2)** | Mild edit of parent. Official S4. Occ. 0.20 | Tall up-spikes, jitter. Occ. 0.55, faults vanish | Downward dropouts. Best tune ARP 0.45 | High occ. (0.76 search), **0** Coverage@τ |
| **time_recon** | Stays on parent oscillation | Noisy up-needles | Localized down-spikes | Same family as contrast; still not long campaigns |
| **time_both** | Same, slightly stronger ticks | Same junk spikes | Closest to real **short** spikes | Best qualitative match to dropouts; fails level-shift / multi-day |

Nobody reproduces the majority ESA kind: multi-day low-amplitude campaigns (amp ~0.025). Max-range plots hid that — see `real_anomaly_kinds.png`.

---

## 1. What “comparable to GenIAS” means

Same queries, same φ (`feature_pack_v1`), same τ = 0.105, same 1536-donor gallery, 3-fold OOF.

| Table | Encoder | Steering | vs GenIAS? |
|---|---|---|---|
| **Locked S4** | pool_recon | S4 band only | **yes** (official win rule) |
| **encscore (this table)** | time_recon, time_both | band or combined | **yes** (same τ/φ/1536; not a new S4 freeze) |
| **kindmixscore (this table)** | time_recon, time_both | stratified proto | **yes** (same τ/φ/1536; not a new S4 freeze) |
| **kindmixscorehybrid (this table)** | time_recon, time_both | hybrid / hybrid_needles | **yes** (same τ/φ/1536; isolated dir) |
| Tune regen (c15 / z7 / id 10) | pool_recon | contrast or occupancy | loosely |
| Enc steer probe | all five encoders | S4 band, 192 donors, fold 0 | **no** |
| `shell_tune` PNGs | time_recon, time_both | all four recipes, 256 windows | eyes only |
| **S5 (this table)** | TSDiff + residual \(a_\psi\) | adapter-only / adapter+S2 band / adapter×combined | **yes** (same τ/φ/1536) |

Win rule (frozen): beat **both** GenIAS and post-hoc on gap, occupancy ≥ 0.8, diversity ≥ 0.5 × unguided. Do not retune λ / q / δ after S4.

```
d(q, G)         = min_g ‖ φ(q) − φ(g) ‖
Coverage@τ      = fraction with d ≤ τ
gap             = Coverage(anomalies) − Coverage(rares)
ARP             = 1 / (1 + mean d)
Div             = mean pairwise ‖φ(g) − φ(g')‖ inside G
EDI             = Shannon entropy of G over 16 k-means bins of the union of the galleries in this table
```

GenIAS ([arXiv:2502.08262](https://arxiv.org/abs/2502.08262) App. E) reports **ARP** (realism) and **EDI** (diversity), the latter motivated by AnomalyDiffusion. EDI is \(-\sum_i p_i\log p_i\) after one \(k=16\) k-means on \(\bigcup_m G_m\); max is \(\ln 16 \approx 2.77\). They treat ARP as primary: read EDI only if the gallery actually resembles real anomalies. **Div** is our win-rule spread (not in GenIAS); it does not depend on the other methods.

Coverage and ARP disagree. τ is a small ball. High ARP can mean “near the many near-nominal faults,” not “looks like a dropout.” High Div / EDI with low Coverage@τ is spread-out junk, not useful diversity.

Locked S4-only EDI (5-method union) is in [RESULTS.md](RESULTS.md) and must not be mixed with the 9-method EDI below — the partition changes when combined galleries join the union.

---

## 2. Scores vs GenIAS (locked S4 + encscore)

τ = 0.105, φ = `feature_pack_v1`, 1536 donors, 3-fold OOF. Locked S4 occupancy **0.20**. Official win: **lost**.

The first block is the frozen S4 table. **Shell ZS is pool_recon × band only.** The second block is `encscore` (2026-09-09): same protocol, new encoder and/or combined \(f\). The third block is **stratified proto** (`kindmixscore`, 2026-09-10 ESA kinds): same τ/φ/1536, parent50, official `Length` × `Locality` plus the level-shift overlay. The fourth block is **hybrid / hybrid_needles** (`kindmixscorehybrid`, 2026-09-10): same protocol, proto only on structured kinds, band on the rest. The fifth block is **from noise** (`noisescore`, 2026-09-10): same recipes with \(x_T\sim\mathcal N(0,I)\). The sixth block is **S5** (2026-09-09): residual adapter on the locked TSDiff score. Occupancy for encoder rows is vs **that encoder’s** \(Q_q\), not S2’s. Combined, stratified, hybrid, and S5 use event-OOF refs / train-fold anomalies. `time_both` encoder saw fold-1/2 anomalies at train time. Artifacts: `results/shell_enc_score/summary.json`, `results/shell_kindmix_score_esa/summary.json`, `results/shell_kindmix_score_hybrid/summary.json`, `results/shell_noise_score/summary.json`, `results/shell_s5/summary.json`. Morphology-kind 1536 freeze (do not mix): `results/shell_kindmix_score/summary.json`.

† S5 EDI is from the **S5-extended** k-means union (S4 + encscore + S5 galleries). ‡ Stratified EDI is from a **new** union (S4 + encscore + the two stratified fold-0 galleries). \* Hybrid EDI is from another union (S4 + encscore + the four hybrid / hybrid_needles fold-0 galleries). ¶ From-noise EDI is yet another union (`shell_noise_score`). Do not compare † / ‡ / \* / ¶ / the 9-method EDI to each other. **Div is comparable** across the whole table.

| Method | Shot | Encoder | Steering | Cov. anom | Cov. rare | Gap | ARP anom | Occ. | Div | EDI |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Unguided DDIM ν=1 | ZS | — (g=0) | none | 0.109 | 0.092 | +0.017 | 0.568 | — | 2.34 | 1.89 |
| **Shell ZS (locked S4)** | ZS | **pool_recon** | **band λ=0.3** | 0.081 | 0.071 | +0.010 | 0.393 | 0.20 | 2.01 | 1.83 |
| GenIAS ψ=2 (unpatched) | ZS | TCN-VAE | inflate z | 0.195 | 0.196 | −0.001 | 0.317 | — | 1.09 | 1.12 |
| GenIAS + paper patch | ZS | same | Alg. 2 (0% kept) | 0.354 | 0.324 | +0.030 | 0.320 | — | 1.11 | 1.18 |
| Post-hoc inject | — | — | step/pulse/ramp/scale | **0.267** | 0.220 | **+0.047** | 0.319 | — | 1.21 | 1.37 |
| time_recon + band | ZS | time_recon | original \(f\) | 0.076 | 0.077 | −0.000 | 0.433 | 0.02 | 2.09 | 1.83 |
| time_both + band | FS enc | time_both | original \(f\) | 0.083 | 0.100 | −0.017 | 0.459 | 0.01 | 2.07 | 1.85 |
| **time_recon + combined \(f\)** | FS | time_recon | band + contrast | 0.023 | 0.016 | +0.007 | 0.536 | **0.78** | 6.61 | 2.54 |
| time_both + combined \(f\) | FS | time_both | band + contrast | 0.011 | 0.011 | −0.000 | 0.512 | 0.77 | 6.23 | 2.51 |
| time_recon + stratified proto | FS | time_recon | band + ESA-kind proto | **0.000** | 0.000 | 0.000 | 0.450 | 0.07 | 7.34 | 2.31‡ |
| time_both + stratified proto | FS | time_both | band + ESA-kind proto | **0.000** | 0.001 | −0.001 | 0.487 | 0.07 | 6.75 | 2.31‡ |
| time_recon + hybrid | FS | time_recon | proto on shift only | 0.024 | 0.026 | −0.002 | 0.520 | 0.62 | 5.34 | 2.25* |
| time_recon + hybrid_needles | FS | time_recon | proto on shift + Point/Global | 0.023 | 0.029 | −0.007 | 0.576 | 0.45 | **7.94** | 2.54* |
| time_both + hybrid | FS | time_both | proto on shift only | 0.038 | 0.036 | +0.002 | 0.511 | 0.64 | 5.52 | 2.31* |
| **time_both + hybrid_needles** | FS | time_both | proto on shift + Point/Global | 0.039 | 0.025 | +0.014 | **0.584** | 0.47 | 7.52 | 2.60* |
| S5 adapter-only | FS | — (g=0) | adapted ε, ν=0.2 | 0.063 | 0.076 | −0.013 | 0.455 | — | 2.04 | 1.66† |
| S5 adapter+cls | FS | pool_recon | adapted + S2 band (cls **dropped**) | 0.021 | 0.024 | −0.003 | 0.416 | 0.69 | 2.60 | 2.01† |
| S5 + time_recon combined | FS | time_recon | adapted ε + combined \(f\) | **0.000** | 0.002 | −0.002 | 0.352 | 0.77 | 4.71 | 2.36† |
| S5 + time_both combined | FS | time_both | adapted ε + combined \(f\) | 0.001 | 0.000 | +0.001 | 0.336 | 0.78 | 4.37 | 2.31† |

Energy diagnostic on locked S2 (do not retune): rares sit on \(Q_q\) (mean h ≈ −0.393 vs −0.390). Anomalies: mean −0.20, std 1.25.

**How to read the new rows.** Original band \(f\) on a temporal encoder is the same story as locked Shell ZS: Coverage@τ stays ~0.08, still less than half of GenIAS (0.195) and far below post-hoc (0.267). ARP rises a bit (0.39 → 0.43–0.46) but occupancy collapses (0.20 → 0.01–0.02) because \(Q_q\) lives in a new embedding. `time_both` + band covers **rares more than anomalies** (gap −0.017). Combined \(f\) does the opposite trade: occupancy jumps to 0.77–0.78 (near the 0.8 win bar) and ARP approaches unguided (0.51–0.54 vs 0.57), **beating GenIAS ARP** (0.32). Coverage@τ falls to 0.01–0.02 — an order of magnitude below GenIAS. Gap is ~0. Combined also wins both diversity scores (Div 6.2–6.6 vs unguided 2.3; EDI 2.51–2.54 vs unguided 1.89, near the \(\ln 16\) ceiling). That is “leave the parent,” not “cover labeled faults”: GenIAS says not to credit EDI when realism fails, and Coverage@τ is the realism check that failed. Combined is few-shot and still loses the protocol (coverage and the win rule). Among the four encscore rows, **time_recon + combined** is the least bad: only positive gap, best ARP, occupancy 0.78. It is not a GenIAS-beating generator.

**Stratified proto (2026-09-10, ESA kinds).** Same 1536 / 3-fold protocol as encscore. Math: §0. Kinds are official Point/Global, local subsequence, level shift (overlay), and global subsequence — Point/Local has no pretest windows. Coverage@τ is **0** on every fold and **every kind** (combined was 0.011–0.023). Occupancy collapses to 0.07 — proto overpowers the band. ARP 0.45–0.49 beats GenIAS (0.32) and the morphology-kind stratified freeze (0.43–0.47) but **loses to combined** (0.51–0.54) and sits **closer to rares than to anomalies** (ARP rare 0.55 / 0.62). Div is in the combined family. Fold 0 still leaks the six level-shift refs. Do not drop the 256-donor kindmix table into this one. Neither row is a protocol upgrade over **time_recon + combined**.

**Hybrid / hybrid_needles (2026-09-10, 1536 / 3-fold).** Same τ/φ/donors/parent50 as stratified. Isolated `results/shell_kindmix_score_hybrid/` — does not rewrite the ESA stratified freeze. `hybrid` protos only level shift (3/4 slices are band). `hybrid_needles` protos shift + Point/Global (2/4 band). Occupancy tracks that split (0.62–0.64 vs 0.45–0.47): band slices sit near \(Q_q\); proto slices knock occupancy down. Coverage@τ is **0.023–0.039** — above stratified (0) and combined (0.011–0.023), still **5× below GenIAS** (0.195). That coverage is **global subsequence only** (band-like campaigns). Aimed Point/Global and level shift stay at Coverage **0** (`feature_pack_v1` z-scores DC). Local-subsequence Coverage 0.08–0.17 on `time_both` is \(n=4\) queries — ignore it. `time_both` + hybrid_needles is the strongest hybrid row: gap **+0.014**, ARP **0.584** (above unguided 0.568; fold 2 is a 0.77 spike). ARP rare is still ~0.78–0.80 — galleries remain closer to rares. Div peaks at **7.94** (`time_recon` needles). Still loses the frozen win rule (occupancy ≪ 0.8; Coverage ≪ GenIAS / post-hoc). Eyes stay [FINDINGS.md](FINDINGS.md); this block is the protocol cite.

**From noise (2026-09-10, same 1536 / 3-fold).** `start_from_noise=True`, ν=1, \(x_T\sim\mathcal N(0,I)\). Channel index still conditions ε; no donor waveform. Isolated `results/shell_noise_score/` (`noisescore`). Same \(f\) / encoders / OOF refs as the ν=0.2 rows. Unguided in the main table is **not** this: it is q_sample(ν=1) from a real parent. EDI ¶ is a new union — compare **Div**, not EDI, to the ν=0.2 rows.

| Method | Start | Cov. anom | Gap | ARP anom | Occ. | Div | EDI |
|---|---|---:|---:|---:|---:|---:|---:|
| time_recon + hybrid | parent ν=0.2 | 0.024 | −0.002 | 0.520 | 0.62 | 5.34 | 2.25* |
| time_recon + hybrid | **noise** | 0.012 | +0.010 | 0.549 | 0.59 | 14.74 | 2.58¶ |
| time_recon + hybrid_needles | parent ν=0.2 | 0.023 | −0.007 | 0.576 | 0.45 | 7.94 | 2.54* |
| time_recon + hybrid_needles | **noise** | 0.013 | +0.011 | 0.544 | 0.44 | 14.62 | 2.64¶ |
| time_recon + combined \(f\) | parent ν=0.2 | 0.023 | +0.007 | 0.536 | 0.78 | 6.61 | 2.54 |
| time_recon + combined \(f\) | **noise** | 0.014 | +0.013 | 0.520 | 0.78 | 14.34 | 2.41¶ |
| time_both + hybrid | parent ν=0.2 | 0.038 | +0.002 | 0.511 | 0.64 | 5.52 | 2.31* |
| time_both + hybrid | **noise** | 0.020 | +0.015 | 0.577 | 0.61 | 14.74 | 2.58¶ |
| time_both + hybrid_needles | parent ν=0.2 | 0.039 | +0.014 | 0.584 | 0.47 | 7.52 | 2.60* |
| time_both + hybrid_needles | **noise** | 0.020 | +0.011 | 0.578 | 0.44 | 14.15 | 2.64¶ |
| time_both + combined \(f\) | parent ν=0.2 | 0.011 | −0.000 | 0.512 | 0.77 | 6.23 | 2.51 |
| time_both + combined \(f\) | **noise** | 0.027 | +0.021 | 0.551 | 0.78 | 13.87 | 2.39¶ |
| Unguided (q_sample ν=1) | parent destroyed | 0.109 | +0.017 | 0.568 | — | 2.34 | 1.89 |

Occupancy **holds** (combined 0.78; hybrid ~0.60; needles ~0.44). The shell leash does not need a parent. Div **doubles** (5–8 → 14–15) — that is the real “from scratch” gain; unguided Div is only 2.34 because q_sample(ν=1) still starts from a real window. Coverage **drops** for hybrid (0.024–0.039 → 0.012–0.020): the ν=0.2 Coverage was mostly inherited campaign-like parents. `time_both` + combined is the exception (0.011 → 0.027). Rare Coverage collapses (~0.03 → ~0.002–0.009), so gaps look better. ARP stays in the same band (0.52–0.58). Aimed level shift stays Coverage 0. Point/Global 0.11–0.17 is \(n=6\) — do not cite. Fold 2 ARP is still a 0.66–0.77 spike. Still ≪ GenIAS Coverage (0.195). Best from-noise cite: **`time_both` + combined** (Cov 0.027, gap +0.021, occ. 0.78, ARP 0.55, Div 13.9) or **`time_both` + hybrid_needles** if you want the mixed-kind recipe (Cov 0.020, ARP 0.578, occ. 0.44, Div 14.2).

**Can the model emit each ESA type?** Eyes: `results/shell_kindmix_esa/plots/` (256 donors, fold 0; aimed-slice grids + nearest-φ). Slice `P(|μ_L−μ_R|≥0.01)` and `P(amp≥0.1)` below are fold-0 1536.

| ESA kind | Windows (41–46) | Aimed? | time_recon | time_both |
|---|---:|---|---|---|
| Point / Local | 0 | — | no pretest queries | no pretest queries |
| Point / Global | 18 | yes (OOF) | **Needles.** Slice amp≥0.1 = 0.86. Best-stat often **down**; nearest-φ can be **up**. | **Needles.** Slice amp≥0.1 = 0.74. Same down/up mix. |
| local subsequence | 12 | yes (OOF) | No. Jitter on the carrier; not the quiet crop. | No. Same; slice amp≥0.1 = 0.03 (too quiet / wrong texture). |
| level shift | 6 | yes (fold-0 **leak**) | **Shelves exist** in the mix (nearest-φ 0.95→0.80). Best-stat cherries an edge spike. Slice shift rate **1.00**. | **Yes.** Typical windows are 0.95→0.80 shelves. Slice shift rate **0.97**. |
| global subsequence | 636 | yes (OOF) | No as a distinct in-window type. Noisier carrier; campaign crops look everyday. | No. Same. Slice amp≥0.1 = 0.25. |

So: **Point/Global and level shift, yes** (shift only with leaked refs). **Local and global subsequence, no.** Coverage@τ cannot see the shelves — `feature_pack_v1` z-scores each window and throws DC away.

**S5 (2026-09-09).** Residual \(a_\psi\) is 54,593 params (17% of the 316k score). Classifier **never** passed the rare-hard-neg gate (fold 1 collapsed to “everything is an anomaly”; folds 0/2 ~chance). So `adapter+cls` is adapted ε + S2 band leash only. Adapter-only coverage 0.063 is **below** unguided (0.109) and Shell ZS (0.081); gap is negative (hits rares more). S5 + combined occupies the band (0.77–0.78) but Coverage@τ falls to **0–0.001** — worse than combined without the adapter (0.023). ARP also drops (0.54 → 0.35). The adapter does not fix the protocol; stacking it on combined makes the gallery less like labeled faults in \(\varphi\). Plots: `results/shell_plots/shell_s5/`. Audit: [S5.md](S5.md).

---

## 3. Newer methods (after S4)

Same frozen score (min-max TSDiff, 316k). New \(e\) and/or new \(f\). They do **not** overwrite `results/shell_s4`.

### 3.1 Encoders — identification vs steering interface

Energy used for AUROC and the enc steer probe:

```
h(x)  =  −τ log Σ_r exp( −‖ e(x) − e(r) ‖² / τ )     (r = nominal refs)
```

| Variant | AUROC anom vs rare | AUROC anom vs nom | Locality (top 10% ‖∇h‖²) | Steer occ. (192) | Steer ARP (192, band) |
|---|---:|---:|---:|---:|---:|
| pool_recon | 0.529 | 0.538 | 0.15 | 0.06 | 0.426 |
| **time_recon** | **0.530** | **0.548** | **0.36** | 0.02 | 0.409 |
| pool_supcon | 0.529 | 0.537 | 0.22 | 0.65 | 0.421 |
| time_supcon | 0.529 | 0.545 | 0.29 | 0.76 | 0.410 |
| **time_both** | 0.529 | 0.546 | 0.34 | 0.69 | **0.461** |

AUROC stays chance: **no encoder tells a fault from a rare** in \(h\). Dropping GAP more than doubles gradient locality (0.15 → 0.36). That is the real encoder win.

- **Promote later as ZS S2:** `time_recon` (no anomaly labels, local ∇, 44k).
- **Promote only as few-shot:** `time_both` (best small-probe ARP 0.46).
- **Kill:** SupCon-only. Pooled \(h\) collapsed (nom/rare/anom ≈ −0.112). High occupancy there is a thin δ on a dead energy, not a better shell.

The 192-donor steer probe used **S4 band only** (λ=0.3, ν=0.2). The 1536 `encscore` band rows above replace that probe for vs-GenIAS numbers.

### 3.2 Steering on the locked S2 encoder (`shell_tune`)

Grid of 17 recipes, then regen of three winners. Encoder still **pool_recon**. τ frozen.

| Winner | Recipe | Regen n | Occ. | Cov. anom | Gap | ARP anom |
|---|---|---:|---:|---:|---:|---:|
| z7 occupancy | λ=1.5, n_correct=4, no contrast | 512 | **0.55** | 0.001 | +0.001 | 0.34 |
| **c15 contrast** | λ=0, λ_a=λ_r=3, ν=0.5 | 1536 | 0.21 | 0.027 | ~0 | **0.45** |
| c10 contrast (best search gap) | λ=0, λ_a=λ_r=1, ν=0.2 | 1536 | 0.20 | 0.010 | −0.009 | 0.39 |
| id 14 combined (search only) | λ=0.3 + λ_a=λ_r=1 | 192 | **0.76** | **0** | −0.02 | 0.50 |

c15 ARP (0.45) beats locked shell (0.39) and GenIAS (0.32), still loses to unguided (0.57). Coverage stays ≪ GenIAS (0.027 vs 0.195). Occupancy-max makes waveforms worse.

### 3.3 `time_recon` / `time_both` × recipes (eyes + 1536 scores)

256-parent strips: `results/shell_plots/shell_tune/`. 1536 scores: §2 / `results/shell_enc_score/`. ESA stratified proto 1536: `results/shell_kindmix_score_esa/`. 256-donor ESA eyes: `results/shell_kindmix_esa/plots/`.

| Recipe | time_recon | time_both |
|---|---|---|
| Contrast (c15 HPs) | Sharp localized down-spikes | Same, closest to real short events (id_109/110/…) |
| Occupancy (z7 HPs) | High-frequency up-needles | Same, worse |
| S4 band | Parent + small glitch | Parent + small glitch |
| Combined (id 14 HPs) | Band leash + some spikes | Best “looks like a dropout” of the four; still not a level shift or long campaign |

Nearest-φ match (`real_kinds_vs_combined_encoders.png`): generated spike ↔ real short spike works; generated series next to a real level-shift or long campaign is still a spike or a near-parent wave.

### 3.4 S5 adapters (2026-09-09)

Event-OOF residual on the locked TSDiff score. Full audit and file list: [S5.md](S5.md). Scripts: [REPO.md](REPO.md).

```
ε_FS  =  ε_θ  +  a_ψ     (54,593 params; backbone 315,841 frozen)
```

Classifier ℓ never used: it failed acc_rare ≥ 0.5 on every fold. `adapter+cls` = adapted ε + S2 band λ=0.15 only.

Plots (`results/shell_plots/shell_s5/`): `gen_{recipe}.png`, `*_extreme.png`, `real_vs_{recipe}.png`, and the same three with `_from_noise` (fold-0 adapter, \(x_T\sim\mathcal{N}(0,I)\)). Adapter-only stays near the parent oscillation with ticks; S5+combined is the same short-spike family as combined without the adapter.

---

## 4. Shared pieces and references (short)

**Score (all diffusion).** Min-max TSDiff, 6× S4-D residual blocks, hidden 64, 315,841 params. 20k steps. ν=0.2 parent shape corr 0.49 (not identity). Older U-Net / raw-scale scores are obsolete. S5 now uses this same score.

**Unguided ν=1.** Best locked ARP (0.57) because many faults look nominal in φ. Required ablation, not a method.

**GenIAS ψ=2.** TCN-VAE, spatial z, decode from z only, per-window z-score. ψ=1 corr 0.996; ψ=2 RMSE 8e-4 — **stays on the parent**. Paper patch keeps 0% of timesteps here. Cite **unpatched**.

**Post-hoc.** Handcrafted step/pulse/ramp/scale. Best locked gap and Coverage@τ. The bar if the paper is “we generate anomalies.”

**S5 adapters.** Residual CNN on frozen ε (54,593 params). Event-OOF. Classifier failed the rare-hard-neg gate on all folds. Coverage@τ worse than unguided; S5+combined coverage ≈ 0. [S5.md](S5.md).

---

## 5. Ranking

| Priority | Candidate | Shot | vs GenIAS (1536)? | Verdict |
|---|---|---|---|---|
| Bar | Post-hoc | — | yes | Coverage / gap winner. Not our method |
| Official | pool_recon × S4 band | ZS | yes (S4) | Lost win rule |
| Scored | time_recon × band | ZS | yes | Same coverage as shell (~0.08). Not an upgrade |
| Scored | time_both × band | FS enc | yes | Hits rares more (gap −0.017) |
| Scored | **time_recon × combined \(f\)** | FS | yes | Best new ARP (0.54), occ. 0.78, coverage **0.023** |
| Scored | time_both × combined \(f\) | FS | yes | Same trade, slightly worse coverage |
| Scored | time_recon × stratified proto | FS | yes | Cov **0**, occ. 0.07, ARP 0.45, Div 7.34. Needles + leaked shelves |
| Scored | time_both × stratified proto | FS | yes | Cov **0**, occ. 0.07, ARP 0.49, Div 6.75. Best shelves. Not a protocol win |
| Scored | time_* × hybrid | FS | yes | Cov 0.024–0.038, occ. 0.62–0.64, ARP 0.51–0.52. Band-heavy |
| Scored | **time_both × hybrid_needles** | FS | yes | Cov **0.039**, gap **+0.014**, ARP **0.584**, occ. 0.47, Div 7.52. Still ≪ GenIAS Cov |
| Scored | from-noise (ν=1, no parent) | FS | yes | Occ. holds; Div ~14; hybrid Cov 0.012–0.020. `time_both` combined Cov 0.027 |
| Ablation | Unguided ν=1 | ZS | yes | Best ARP (0.57). No injected faults |
| Reference | GenIAS ψ=2 | ZS | yes | Coverage 0.195; mild editor |
| Ablation | pool_recon × contrast (c15) | FS | loosely | ARP 0.45, coverage 0.027 |
| Scored | S5 adapter-only | FS | yes | Cov 0.063, gap −0.013. Mild edit |
| Scored | S5 + time_* combined | FS | yes | Occ ~0.78, coverage **0**. Worse than combined alone |
| Kill | occupancy-max, SupCon-only | — | — | Band-sitting / collapse |

**No scored method beats GenIAS on Coverage@τ, or post-hoc on gap.** Combined \(f\) wins occupancy; hybrid_needles now leads steered ARP (and edges unguided) but Coverage stays ~0.04. ESA-kind stratified proto and S5 do not change the Coverage ranking.  
Coverage@τ: post-hoc > GenIAS > unguided > band shells (~0.08) > S5 adapter-only (0.063) > hybrid / needles (0.023–0.039) ≥ combined (~0.01–0.02) ≫ S5+combined ≈ stratified (**0**).  
ARP: time_both hybrid_needles (0.584) > time_recon hybrid_needles (0.576) ≳ unguided (0.568) > time_recon combined > … > GenIAS. Fold-2 needles ARP is a 0.77 spike — do not over-read the mean.  
Div: time_recon hybrid_needles (7.94) > time_both hybrid_needles (7.52) > time_recon stratified (7.34) > combined (no adapter) > … > GenIAS.  
Eyes (256 / 128, ESA kinds): `hybrid_needles` emits **Point/Global needles and level-shift shelves** in one gallery. Protocol (1536): those aimed kinds still have Coverage@τ **0**; the 0.02–0.04 Coverage is global subsequence.

---

## 6. What a paper can claim

**A — negative / analysis (honest today).**  
A nominal quantile shell does not generate ESA Anomalies. Temporal encoders, combined \(f\), stratified proto, and a few-shot residual adapter do not fix Coverage@τ: band stays ~0.08, combined drops to ~0.02, stratified and S5+combined to ~0, while GenIAS is 0.20 and post-hoc 0.27. Combined *does* occupy its own band (~0.78) and raise ARP toward unguided — that is “leave the parent / sit on \(Q_q\),” not “cover labeled faults.” Stratified keeps combined-like Div but kills occupancy and coverage. The adapter did not give a working anomaly classifier (rares as hard negatives). Rares already sit on the locked \(Q_q\). AUROC of \(h\) stays chance.

**B — methods.** Not supported by these 1536 numbers. A later paper would need a different \(f\) or a metric declared *before* seeing that ARP and Coverage disagree.

**Not viable.** “time_recon / time_both + combined beats GenIAS.” It beats GenIAS only on ARP (and hybrid_needles now does too), and loses badly on the protocol coverage number. Same for “hybrid_needles is the best steered model”: it can emit needles and leaked shelves, and it is the best *hybrid* protocol row (Cov 0.039, ARP 0.584), but Coverage@τ is still an order of magnitude below GenIAS and the aimed kinds stay at 0.

---

## 7. Next run checklist

1. Do not promote combined \(f\) as a Coverage@τ win. It failed that test at 1536.
2. Do not promote stratified proto from the 256-donor kindmix plots. At 1536 / 3-fold on ESA kinds, Coverage@τ is 0 and occupancy is ~0.07. It *can* emit Point/Global and (leaky) level shifts; it cannot emit the majority global-subsequence campaigns.
2b. Hybrid / hybrid_needles **are scored** at 1536 (`kindmixscorehybrid`). Cite `time_both` + hybrid_needles as the best hybrid row, not as a Coverage@τ win. Occupancy 0.45–0.64 is the band-slice fraction, not a shell that covers aimed kinds.
2c. From-noise (`noisescore`) occupancy holds and Div doubles (~14). Do not cite it as a Coverage win — hybrid Cov falls; only `time_both` + combined rises (0.011 → 0.027). Do not mix unguided (q_sample ν=1 from a parent) with `start_from_noise`.
3. If you keep going: a constraint that targets **long-campaign** morphology, not another shove toward \(Q_q\) or train-fold spikes.
4. S5 is scored. Do not stack the adapter on combined \(f\) expecting Coverage@τ to rise. S6 once after freeze.

---

## 8. Artifacts

| Need | Path |
|---|---|
| Locked S4 vs GenIAS | `results/shell_s4/summary.json` |
| time_recon / time_both vs GenIAS | `results/shell_enc_score/summary.json` |
| time_* × ESA stratified proto (1536) | `results/shell_kindmix_score_esa/summary.json` |
| time_* × hybrid / hybrid_needles (1536) | `results/shell_kindmix_score_hybrid/summary.json` |
| time_* × hybrid / combined from noise (1536) | `results/shell_noise_score/summary.json` |
| Morphology stratified freeze (1536) | `results/shell_kindmix_score/summary.json` |
| ESA stratified eyes (256) | `results/shell_kindmix_esa/` |
| Tune grid / regen (pool_recon) | `results/shell_tune/summary.json`, `sweep.csv` |
| Encoder metrics | `results/shell_enc/metrics.csv` |
| S4 waveforms | `results/shell_plots/gen_shell.png`, `gen_genias.png`, `same_parent_overlay.png` |
| New encoder × recipe strips | `results/shell_plots/shell_tune/gen_time_{recon,both}_tune_{contrast,occupancy,combined}.png` |
| Real kinds vs combined | `real_anomaly_kinds.png`, `real_kinds_vs_combined_encoders.png` |
| S5 scores + adapters | `results/shell_s5/summary.json`, `adapter_fold{k}.pt` |
| S5 waveforms (ν=0.2 and from noise) | `results/shell_plots/shell_s5/` |
| Code / eval map | [REPO.md](REPO.md), [S5.md](S5.md) |
| Other channels / Mission 2 | [XFER.md](XFER.md), `results/xfer/` |
