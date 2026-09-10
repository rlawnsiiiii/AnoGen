# Transfer eval — other channels and Mission 2

Separate from locked Mission 1 channels 41–46. This tree retrains and scores **on new data**. It does **not** overwrite `results/shell_s*`, does **not** reuse τ = 0.105, and does **not** run S6.

Reproduce:

```bash
.venv/bin/anogen -c configs/xfer.yaml xfer
# or one dataset:
.venv/bin/anogen -c configs/xfer_mission2.yaml xfer
.venv/bin/anogen -c configs/xfer_mission1_extra.yaml xfer
# extra unused-M2 panels (after scout):
.venv/bin/anogen -c configs/xfer_explore.yaml xfer
```

Artifacts: `results/xfer/{mission2,mission1_extra,mission2_s1b,mission2_s5}/` and `results/xfer/summary.json`.  
Scout (uniqueness / density / labels, no train): `results/xfer/scout/`.
Code: `src/anogen/phases/xfer.py`, panels in `src/anogen/shell/xfer_panel.py`.  
Scoring keep-mask uses real channel names (`keep_fold_mask` in `src/anogen/shell/folds.py`). Locked S4 still hardcodes `channel_41+c` and is left alone.

---

## Why retrain

The frozen Mission 1 denoiser is a **6-way channel-conditioned** TSDiff. Channel index 0 on Mission 2 is not channel 41. Applying `results/shell_s1/denoiser.pt` zero-shot would be a silent domain error.

Each transfer dataset therefore gets its own compact denoiser, GenIAS, and `time_recon` encoder. Sampling math is unchanged ([STEERING_MATH.md](STEERING_MATH.md)):

```
nominal --q_sample(ν)--> x_t* --DDIM − ∇f--> edited window
f = λ(h_nom − Q_q)² + λ_anom h_anom + λ_rare(−h_rare)
```

Band = first term only (ZS). Combined = all three (few-shot: train-fold anomaly / rare refs).

---

## Datasets

Sealed test stays closed (`allow_test_telemetry: false`). Panels are cropped at `official_train_end`.

| Dataset | Mission | Channels | Bin | Official cut | Panel |
|---|---|---|---:|---|---|
| **mission2** | ESA-ADB M2 | 9, 13, 14, 15, 18, 20 | 60 s | 2001-10-01 | Slice of CausalDiscovery `results/stage_g1/panel.npz` (47-ch) |
| **mission2_s1b** | ESA-ADB M2 | 11, 16, 17, 19, 23, 24 | 60 s | 2001-10-01 | Leftover continuous `subsystem_1` (not the first M2 six) |
| **mission2_s5** | ESA-ADB M2 | 70, 73, 74, 75, 81, 82 | 60 s | 2001-10-01 | `subsystem_5` / unit 4–5 (same family as locked M1 41–46) |
| **mission1_extra** | ESA-ADB M1 | 47, 48, 49, 73, 74, 75 | 30 s | 2007-01-01 | Mean-bin pickles up to the cut (not 41–46) |

M2 channels = most pretest Anomaly rows in the 47-ch G1 panel.  
M1 extra = densest analog extras (pickle size / finite W=512 runs), not the raw label-count leaders. First pick (14, 21, 29, 47, 48, 50) failed: 14/21/29/50 are ~33% finite and produced almost no valid windows (channel_50 had 77 nominals). 47/48/49/73/74/75 are ~dense like 41–46.

Six channels so the embedding stays 6-way, same capacity as S1.

| Dataset | Windowed pretest Anomaly | Rare windows | Nominals | Note |
|---|---:|---:|---:|---|
| mission2 | 72 windows / 17 events | 1491 | 6144 | Rares ≫ anomalies. Gap is rare-dominated. OOF refs thin. |
| mission1_extra | 516 windows / 35 events | 285 | 6144 | Held-out M1 analogs, not a re-score of 41–46. |

**W = 512** on both. At 60 s that is ~8.5 h windows; at 30 s ~4.3 h. Same protocol width, not the same physical duration.

---

## Methods (same on every dataset)

| Method | What | Shot |
|---|---|---|
| Unguided ν=1 | DDIM from the new denoiser, no ∇f | ZS |
| Post-hoc | Handcrafted inject on the same nominal donors | — |
| GenIAS ψ=2 | New TCN-VAE on that dataset’s train pool | — |
| **time_recon + band** | Best ZS steered recipe (locked S4 HPs: λ=0.3, ν=0.2, 20 steps) | ZS |
| **time_recon + combined** | Best few-shot steered recipe (id 14: λ=0.3, λ_a=λ_r=1, 50 steps, unit-norm) | FS |

Not run: S5 adapters, `time_both`, pool_recon “Shell ZS”, S6.

Eval: locked `feature_pack_v1` (12-D). Coverage@τ, gap, ARP, Div, EDI.  
**τ is chosen per dataset** from unguided vs fold-0 anomalies (target 0.10).  
EDI is Shannon entropy on one k=16 k-means of **that dataset’s five galleries** (combined = fold 0). Do not mix with S4 / encscore / S5 EDI.

---

## Compact budget (not S4)

Documented as reduced. Do not cite these numbers as a new S4 freeze.

| Knob | Locked M1 S4 | Transfer |
|---|---:|---:|
| Denoiser steps | 20 000 | 4 000 |
| GenIAS steps | 15 000 | 2 500 |
| `time_recon` steps | 800 | 400 |
| Nominals / channel | 4 096 | 1 024 |
| Donors / channel | 256 (1 536 total) | 128 (768 total) |
| `min_anomaly_windows` | 5 | 3 (M2 is sparse) |

Resume: existing `panel.npz` / `s0/` / `s1/denoiser.pt` / `s3/galleries.npz` / `enc/time_recon.pt` / score npzs are reused unless `xfer.force: true`.

---

## What “win” means here

`win_band_vs_refs` applies the S0 occupancy / diversity / beat-GenIAS-and-post-hoc rule to **time_recon + band** on **this** τ and gallery. It is informal. Locked S4 remains the only official cite-vs-GenIAS table ([PAPER_CANDIDATES.md](PAPER_CANDIDATES.md) §2).

Coverage@τ is **not** comparable across datasets or to τ = 0.105.

---

## Results

Compact retrain, 768 donors, per-dataset τ. **Do not mix with locked S4** (τ = 0.105, 1536 donors, ch 41–46). Informal win rule: band lost on both datasets (occupancy and gap).

### Mission 2 (60 s, ch 9 / 13 / 14 / 15 / 18 / 20)

τ = **0.889** · 72 anomaly / 1491 rare query windows · EDI = this 5-method union.

| Method | Cov. anom | Gap | ARP anom | Div | EDI | Occ. |
|---|---:|---:|---:|---:|---:|---:|
| Unguided ν=1 | 0.129 | +0.073 | 0.340 | **10.54** | **2.41** | — |
| Post-hoc | 0.270 | **+0.131** | **0.419** | 5.71 | 1.70 | — |
| GenIAS ψ=2 | **0.278** | +0.111 | 0.406 | 6.12 | 1.72 | — |
| time_recon + band | 0.115 | +0.078 | 0.327 | 6.88 | 2.21 | 0.012 |
| time_recon + combined | 0.090 | +0.065 | 0.294 | 8.74 | 2.36 | 0.56 |

GenIAS / post-hoc win Coverage@τ and ARP. Unguided wins Div / EDI. Band occupancy collapsed (0.012). Combined is more diverse than band but worse coverage — same pattern as locked M1.

Plots: `results/xfer/mission2/plots/`.

### Mission 1 extra (30 s, ch 47 / 48 / 49 / 73 / 74 / 75)

τ = **1.140** · 516 anomaly / 285 rare query windows · EDI = this 5-method union.

| Method | Cov. anom | Gap | ARP anom | Div | EDI | Occ. |
|---|---:|---:|---:|---:|---:|---:|
| Unguided ν=1 | 0.081 | −0.048 | 0.293 | 9.56 | **2.37** | — |
| Post-hoc | **0.926** | +0.021 | **0.613** | 4.11 | 1.23 | — |
| GenIAS ψ=2 | 0.902 | **+0.048** | 0.602 | 4.28 | 1.23 | — |
| time_recon + band | 0.088 | −0.052 | 0.296 | 6.10 | 2.12 | 0.50 |
| time_recon + combined | 0.074 | −0.051 | 0.274 | **11.40** | 2.08 | 0.66 |

τ is large because unguided sits far from fold-0 anomalies in φ (10th-percentile min distance 1.14 vs 0.105 on locked 41–46). That inflates post-hoc / GenIAS Coverage@τ (~0.9) — they land close (mean min-d ≈ 0.64). Steered recipes stay near unguided (~0.08) and have **negative** gap (more rare hits than anomaly hits). Combined again wins Div, loses Coverage@τ.

Plots: `results/xfer/mission1_extra/plots/`.

Eyes: post-hoc and GenIAS keep the **stairs / binary dropouts**; unguided and `time_recon` look continuous. That is the Gaussian score smearing a small alphabet, not a τ bug. Ideas and formulas: [DISCRETE_CHANNELS.md](DISCRETE_CHANNELS.md).

Unguided-only strips (judge stairs yourself): `results/xfer/mission1_extra/plots/unguided_discrete/` — start with `overview_one_per_channel.png` and `channel_47_parent_vs_unguided.png`.

---

## Script → gallery → score

| Row | Generate | Score | On disk |
|---|---|---|---|
| Unguided | `unguided_from_nominal` | `xfer.py` `_score_methods` | `score/unguided.npz` |
| Post-hoc | `posthoc_inject` | same | `s3/galleries.npz` `posthoc` |
| GenIAS ψ=2 | `genias_sample` | same | `s3/galleries.npz` `genias` |
| time_recon + band | `chunked_guided_ddim` `_BAND` | same | `score/time_recon_band.npz` |
| time_recon + combined | `_COMBINED` + event-OOF refs | fold-wise, then mean | `score/time_recon_combined_fold{k}.npz` |

τ: unguided vs fold-0 anomalies on **this** S0 table.  
Keep-mask: `keep_fold_mask(..., channels=cfg.channels)` — not `channel_41+c`.

---

## Do not

- Overwrite `results/shell_s4` or re-run `anogen s4` / `s6`.
- Quote transfer Coverage@τ next to locked S4 0.267 / 0.195 / 0.109 / 0.081.
- Mix EDI from this 5-method union with the 5-method S4, 9-method encscore, or S5-extended unions.
- Apply Mission 1 `denoiser.pt` to Mission 2.
- Treat post-hoc / GenIAS “looking discrete” as a score win. They copy the parent alphabet; see [DISCRETE_CHANNELS.md](DISCRETE_CHANNELS.md).
