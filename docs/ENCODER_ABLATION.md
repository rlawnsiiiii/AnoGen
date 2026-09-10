# Encoder ablation (2026-09-09)

Does **not** overwrite the locked S2 `ConvEncoder` or the S4 table. Phase: `anogen -c configs/shell_mission1.yaml enc`. Artifacts: `results/shell_enc/`.

The S2 encoder is a 3-layer stride-2 conv AE with **global average pool** → \(e(x)\in\mathbb{R}^{32}\), trained **400 steps (~1 epoch)** of reconstruction MSE on everyday + rares. Steering uses \(h_{\mathrm{soft}}\) in that space. Two hypotheses for why \(\nabla_x h\) cannot make local faults and why rares sit on \(Q_q\):

1. Pooling smears the gradient over the whole window.
2. Recon-on-nominals never asks \(e\) to separate everyday / rare / anomaly.

This run tests both, event-OOF. Primary rank (declared before fitting): **AUROC of \(h_{\mathrm{soft}}\) (nominal reference set) for held-out anomalies vs held-out rares.** Higher means the shell energy can tell a fault from a Rare Event.

## Architectures

Shared backbone (same as S2, width \(W=512\)):

```
Conv1d 1→32,  k=7, stride 2
Conv1d 32→64, k=5, stride 2
Conv1d 64→64, k=5, stride 2
→ feature map (B, 64, 64)   # W/8
```

| Variant | After the map | \(e(x)\) | Decoder | Params |
|---|---|---|---|---|
| **pool_recon** (S2-like) | `AdaptiveAvgPool1d(1)` + Linear 64→32 | \(\mathbb{R}^{32}\) | from the 32-D code (same as S2) | 181k |
| **time_recon** | 1×1 conv 64→8, **no pool** | flatten \(\mathbb{R}^{8\times 64}=\mathbb{R}^{512}\) | ConvTranspose **from the map**, not from the vector | 44k |
| **pool_supcon** | same pool as S2 | \(\mathbb{R}^{32}\) | unused at train time | 181k |
| **time_supcon** | same as time_recon | \(\mathbb{R}^{512}\) | unused | 44k |
| **time_both** | same as time_recon | \(\mathbb{R}^{512}\) | from the map + SupCon on \(e(x)\) | 44k |

Energy is always

```
h(x)  =  −τ log Σ_r exp( −‖ e(x) − e(r) ‖² / τ )
```

with \(r\) a nominal reference set. For the temporal models that is Frobenius distance on the \(8\times 64\) map, so \(\nabla_x h\) can concentrate on a local interval.

## Losses

**Recon** (`pool_recon`, `time_recon`):

```
L = MSE(decode(x), x)
```

Fit on everyday windows **plus train-fold Rare Events** (S2 train pool). **No anomalies.** 800 Adam steps, batch 64, \(lr=10^{-3}\). About 2 epochs (S2 was 400 steps / ~1 epoch).

**SupCon** (`pool_supcon`, `time_supcon`) — Khosla et al., 3 classes:

```
z = e(x) / ‖e(x)‖
L = − mean_i  (1/|P(i)|) Σ_{p ∈ P(i)} log
        exp(z_i · z_p / τ_c) / Σ_{a ≠ i} exp(z_i · z_a / τ_c)
```

\(\tau_c=0.1\). Classes: 0 = everyday, 1 = train-fold rare, 2 = **train-fold anomaly**. Balanced batch: 32 per class. Held-out-fold labels never enter the loss. 1200 steps.

**Both** (`time_both`): \(L = \mathrm{MSE} + \mathrm{SupCon}\) on the same balanced batch (recon therefore sees train-fold anomalies). 1200 steps.

Eval fold \(k\): train on folds \(\neq k\), score fold \(k\) anomalies/rares. Three folds, then average. Sealed test unused.

Steer probe (fold-0 encoder only, 192 donors, frozen S4 \(\tau=0.105\), \(\lambda=0.3\), \(\nu=0.2\)): not comparable to the S4 1536-donor table.

## Results (mean of 3 OOF folds)

| Variant | AUROC anom vs rare | AUROC anom vs nom | Locality (top 10% \(\|\nabla h\|^2\)) | Steer occ. | Steer ARP anom |
|---|---:|---:|---:|---:|---:|
| pool_recon (S2-like) | 0.529 | 0.538 | 0.15 | 0.06 | 0.426 |
| **time_recon** | **0.530** | **0.548** | **0.36** | 0.02 | 0.409 |
| pool_supcon | 0.529 | 0.537 | 0.22 | 0.65 | 0.421 |
| time_supcon | 0.529 | 0.545 | 0.29 | **0.76** | 0.410 |
| time_both | 0.529 | 0.546 | 0.34 | 0.69 | **0.461** |

Chance AUROC is 0.5. Fold 2 is below chance for every variant (~0.48): those 126 anomaly windows sit inside the everyday blob in every embedding.

SupCon **collapsed** the pooled space: mean \(h\) for nom / rare / anom is \(-0.112 / -0.112 / -0.110\). The 3-class batch did not build a shell that isolates faults.

Locality behaved as designed: dropping GAP **more than doubles** the fraction of \(\nabla_x h\) on the peak 10% of timesteps (0.15 → 0.36). Temporal \(e\) is a better *steering interface*. It is not a better *identifier*.

## Which encoder is most promising?

**For the energy that S4 actually uses (\(h\) vs \(Q_q\)):** none of these is a real upgrade. Anom-vs-rare AUROC is 0.529 ± 0.001. Architecture and SupCon do not move the identification failure.

**For a next steering implementation:** **`time_recon`**.

- Same label policy as S2 (no anomaly training).
- Local \(\nabla h\) (the change that can, in principle, edit a spike).
- Tiny (44k), recon loss logged.
- Primary metric tied with the others; secondary AUROC anom-vs-nom is the highest of the five.

**`time_both`** is the runner-up if you accept train-fold anomalies in the encoder (few-shot): best fold-0 steered ARP (0.46) and still local. Treat as an S5-style encoder, not a ZS shell.

Do **not** promote `pool_supcon` / `time_supcon` as the S2 replacement: collapse + no AUROC gain. High occupancy there is a thin \(\delta\) on a collapsed \(h\), not a better band.

Plots: `results/shell_enc/auroc_anom_vs_rare.png`, `auroc_anom_vs_nom.png`, `locality.png`, `steer_arp_anomaly.png`, `loss_curves.png`. Curves: `*_loss.npz`. Fold-0 weights: `*_fold0.pt`.

S2/S4 stay on `pool_recon` until you explicitly refit S2 with `time_recon` and re-run S4 (new \(Q_q\), new steer). S6 stays sealed.
