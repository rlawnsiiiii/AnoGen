# Training results (S1 denoiser + S3 GenIAS + ZS S2/S4)

Which model is a paper candidate, exact architectures, and what is comparable to GenIAS: [PAPER_CANDIDATES.md](PAPER_CANDIDATES.md).

Retrained 2026-09-08 on ESA-ADB Mission 1 channels 41–46. Train pool = 24,576 everyday windows + 943 rares (25,519). True anomalies stay out of S1/S2/S3. S2/S4 below use the min-max TSDiff score. Do not retune λ / q / δ from this table.

Look at the PNGs first. Numbers are secondary.

## Where to look

| File | What it answers |
|---|---|
| `results/shell_s1/loss_curve.png` | Did the denoiser’s ε-loss fall and stay down? |
| `results/shell_s1/loss_by_t.png` | Which diffusion times are still hard? |
| `results/shell_s1/unguided_vs_parent.png` | Do ν=1 samples look like the telemetry? |
| `results/shell_s1/recon_vs_parent.png` | λ=0 DDIM vs the same parent at ν=0.05 / 0.2 / 1 |
| `results/shell_s1/recon_sweep.json` | RMSE, shape corr, shuffled-parent null |
| `results/shell_s3/loss_curve.png` | Did GenIAS recon fall in *whitened* units? |
| `results/shell_s3/recon_vs_parent.png` | Does ψ=1 track the parent oscillation? |
| `results/shell_plots/train_s1_*.png`, `train_s3_*.png` | Same figures, copied next to the galleries |
| `results/shell_plots/gen_genias.png` | New GenIAS gallery (not the old spike/plateau) |
| `results/shell_plots/gen_shell.png` | ZS steered edits (extreme range first) |
| `results/shell_plots/same_parent_overlay.png` | One parent, every generator |
| `results/shell_s4/summary.json` | Coverage@τ, gap, occupancy, ARP/EDI, win rule |
| `results/shell_plots/arp_tsne.png` | φ t-SNE: real anomalies/rares vs galleries |
| `results/shell_plots/arp_bars.png` | ARP on anomalies vs rares |

## What was wrong before

GenIAS decoded from a global z **plus a copy of the input**. Training MSE collapsed by fitting the ~0.8 DC level. Samples were a flat plateau with TCN edge spikes. Loss was never logged, so that collapse looked like “low loss.”

The denoiser’s last-batch loss was logged; there was no curve and no val-by-t. Weights also predated rares-in-train.

## Fixes in this retrain

- **GenIAS:** per-window z-score inside the VAE so the loss is the oscillation, not the DC. Spatial latent (~W/8), decode from z only, reflect-pad TCN.
- **Both:** train/val curves, EMA train loss, grad clip. Denoiser also logs ε-MSE by time bin.
- S1 now trains on the 25,519-window pool (rares included).

## S1 — 1-D ε-U-Net (20k steps, batch 128, T=200)

| Check | Value | Verdict |
|---|---|---|
| Train EMA ε-MSE | 0.00161 | Down from O(1), flat after ~2k steps |
| Val ε-MSE | 0.00160 | Tracks train; no overfit |
| loss_dropped | true | Val last < val first |
| Sample mean / data mean | 0.794 / 0.792 | Matched |
| Sample std / data std | 0.0190 / 0.0188 | Matched |
| Feature L2 vs nominal (vs noise) | 0.535 (3.47) | Closer than noise |

Val error is concentrated at **t = 0–24** (~0.012); later bins are ~0.001 or lower. That is the usual “fine detail” bin.

### Unguided parent recon (λ=0, 256 windows)

Noise a real parent to ν·T, DDIM back with no shell. Two baselines: the noised window, and a shuffled parent pairing (identity null). Two random parents have RMSE 0.026 to each other (mostly different DC levels).

| ν | t | DDIM RMSE | Noised RMSE | Shape corr | DDIM vs shuffled parent |
|---|---|---|---|---|---|
| 0.05 | 10 | 0.0067 | 0.081 | 0.27 | 0.028 |
| 0.20 | 40 | 0.0077 | 0.289 | 0.09 | 0.025 |
| 1.00 | 199 | 0.0073 | 1.056 | ~0 | 0.026 |

It **did learn**: DDIM beats the noised input at every ν, and matched-parent RMSE (0.007) is far below shuffled (0.026), so the window’s **level** comes back. It does **not** lock the parent’s oscillation. Shape corr is only 0.27 at the lightest noise and ~0 at the S2/S4 edit ν=0.2. Overlays look like a smooth series in the right band, not a copy of that parent. Same RMSE at ν=0.05 and ν=1 is the giveaway: this is manifold / DC recovery, not identity recon.

## S1 — TSDiff S4-D backbone (retrained, same 20k / loss)

`backbone: tsdiff`: 6 residual S4-D blocks, hidden 64, d_state 64, **315,841** params (smaller than the U-Net). The table below is the **raw-scale** ε-MSE run (no min-max). Config now defaults to this score; `backbone: unet` still builds the old net.

| Check | U-Net | TSDiff S4-D |
|---|---|---|
| Val ε-MSE | 0.00160 | 0.00182 |
| Feature L2 vs nominal | 0.535 | 0.624 |
| ν=0.05 shape corr | 0.27 | **0.41** |
| ν=0.20 shape corr | 0.09 | **0.15** |
| ν=1.0 shape corr | ~0 | ~0 |
| ν=0.20 DDIM RMSE | 0.0077 | 0.0078 |

S4 helped a bit at light noise (global RF). It did **not** fix identity recon at the edit ν=0.2. Loss still plateaus by ~2k steps.

## S1 — TSDiff + per-channel min-max (retrained 2026-09-08)

`shell.scaler.kind: minmax`: fit lo/hi on the 25,519 train windows, map each channel to [0, 1], train ε-MSE there, invert after DDIM. Encoder / classifier stay in raw units. Do not compare val ε-MSE to the raw-scale rows above (different units).

| Check | Raw TSDiff | Min-max TSDiff |
|---|---|---|
| Val ε-MSE | 0.00182 (raw) | 0.0147 (scaled) |
| Feature L2 vs nominal | 0.624 | **0.251** |
| Sample mean / data mean | 0.790 / 0.792 | 0.791 / 0.792 |
| ν=0.05 shape corr | 0.41 | **0.82** |
| ν=0.20 shape corr | 0.15 | **0.49** |
| ν=1.0 shape corr | ~0 | 0.03 |
| ν=0.20 DDIM RMSE | 0.0078 | **0.0054** |
| ν=0.20 vs shuffled parent | 0.026 | 0.025 |

ν=0.05 now tracks the parent oscillation (shape corr 0.82; RMSE 0.0034 vs noised-only 0.016). The S2/S4 edit ν=0.2 is better but not identity (0.49, not ≳ 0.8). ν=1 is still generation. Low-t val error is still the hard bin (t 0–24).

## S3 — GenIAS TCN-VAE (15k steps, batch 128, ψ=2)

| Check | Value | Verdict |
|---|---|---|
| Whitened val recon MSE | 0.0103 | Down from ~1 (mean predictor) |
| Raw recon R² | 0.994 | DC + shape |
| Centered shape R² | 0.993 | Oscillation, not just mean |
| Per-window shape corr (ψ=1) | 0.996 | Tracks parent |
| ψ=1 RMSE to parent | 0.00059 | Tight recon |
| ψ=2 RMSE to parent | 0.00084 | Small edit; posterior is tight |
| loss_dropped | true | |

First retrain *without* z-scoring still had a pretty log-loss curve and a **flat** recon (shape failed). Whitened loss is the one that matches the pictures.

ψ=2 stays close to the parent. That is expected if σ_z is small: inflation barely moves the decode. GenIAS is a working reconstructor / mild editor here, not a large anomaly injector.

**Deviation patch (paper Alg. 2), added after the fact.** Keep x̃ only where (x − x̃)² > τ_patch · (max x − min x); else keep the parent. Paper τ_patch = 0.2 (also tried 0.05 / 0.01). On these windows that test keeps **0% of timesteps**: ψ=2 RMSE is 8e-4, while τ·amp is ~0.004. The patched gallery is the nominal parent. Coverage of that copy is 0.35 / 0.32 (anom/rare) — that is “everyday donors vs labels,” not generated anomalies. Official S4 `genias` stays the **unpatched** decode. `genias_patched` is stored for the diagnostic. The paper also trains a perturbation loss that forces the decode off the parent; we do not.

## S2 / S4 — zero-shot shell (frozen λ=0.3, q=0.99, ν=0.2)

Encoder refit on the min-max score. Samples finite, no explode. S2 occupancy 0.07; S4 occupancy **0.20** (need ≥ 0.8). Most steered windows sit *below* the band (68%), not on Q_q.

τ = 0.105 from unguided vs fold-0 anomalies (10% target). That is a real τ now; the old 0.45 was from a broken score.

| Method | Cov. anomaly | Cov. rare | Gap | Diversity |
|---|---|---|---|---|
| Shell ZS | 0.081 | 0.071 | **+0.010** | 2.01 |
| Unguided | 0.109 | 0.092 | +0.017 | 2.34 |
| GenIAS ψ=2 (unpatched) | 0.195 | 0.196 | −0.001 | 1.08 |
| GenIAS + paper patch | 0.354 | 0.324 | +0.030 | 1.11 |
| Post-hoc | 0.267 | 0.220 | **+0.047** | 1.21 |

Win rule: **lost**. Beats GenIAS on gap, loses to post-hoc, occupancy fail, diversity OK (0.86 × unguided).

### GenIAS-style ARP / EDI (same locked φ, OOF queries)

ARP = 1 / (1 + mean min-distance). Higher = gallery sits closer to that query set on average. EDI = Shannon entropy over 16 k-means bins of the union. Not Deep SVDD. Same freeze: do not retune λ / q / δ.

| Method | ARP anomaly | ARP rare | mean min-d anom | EDI |
|---|---|---|---|---|
| Unguided | **0.568** | 0.840 | **0.83** | **2.38** |
| Shell ZS | 0.393 | 0.838 | 1.65 | 2.28 |
| Post-hoc | 0.319 | **0.862** | 2.47 | 1.92 |
| GenIAS ψ=2 | 0.317 | 0.844 | 2.50 | 1.60 |
| GenIAS + paper patch | 0.320 | 0.853 | 2.12 | 1.62 |

Coverage@τ and ARP disagree on the ranking. τ is the 10% unguided quantile — a small ball. Post-hoc puts more anomalies *inside that ball* (0.27 coverage) but the typical anomaly is still far (mean min-d 2.47). Unguided is nearer the average anomaly (ARP 0.57) because many ESA faults look almost nominal in φ. All four methods hug rares (ARP-rare ≈ 0.84–0.86). Shell is a mild edit of unguided on both scores, not a jump toward faults.

Energy diagnostic (do not retune from this): real rares sit on the nominal Q_q (mean h ≈ −0.393 vs Q_q = −0.390). Real anomalies have mean −0.204 and std 1.25 — the 0.99 nominal band is not where faults live. Steered coverage is *below* unguided. The shell did not invent ESA anomalies from a nominal score.

## Encoder ablation (2026-09-09)

Five OOF encoders: pooled recon (S2-like), temporal recon (no GAP, \(e\in\mathbb{R}^{512}\)), pooled/temporal SupCon (everyday vs train-fold rare vs train-fold anomaly), temporal recon+SupCon. Full write-up: [ENCODER_ABLATION.md](ENCODER_ABLATION.md). Does not overwrite S2/S4.

| Variant | AUROC anom vs rare | Locality | Steer ARP (192, fold 0) |
|---|---:|---:|---:|
| pool_recon | 0.529 | 0.15 | 0.426 |
| **time_recon** | **0.530** | **0.36** | 0.409 |
| pool_supcon | 0.529 | 0.22 | 0.421 |
| time_supcon | 0.529 | 0.29 | 0.410 |
| time_both | 0.529 | 0.34 | **0.461** |

Identification does not move (all AUROC ≈ 0.53). Temporal latent **does** localize \(\nabla h\). **Most promising for a future S2:** `time_recon`. SupCon collapsed. S6 stays sealed.

## S5 adapters (2026-09-09)

Re-run on the locked TSDiff score (not the old U-Net / τ=0.45). Write-up: [S5.md](S5.md). Code map: [REPO.md](REPO.md). Does not overwrite S4.

Classifier failed the rare-hard-neg gate on all three folds. Adapter-only Coverage@τ **0.063** (below unguided 0.109). S5 + combined occupancy ~0.78 but coverage **0–0.001**. Plots: `results/shell_plots/shell_s5/`.

