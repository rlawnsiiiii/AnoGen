# Kind-conditional proto

Implements [LEVEL_SHIFT.md](LEVEL_SHIFT.md) idea A. Isolated `results/shell_kindproto/`. Does **not** overwrite `results/shell_s4` or `results/shell_hanom`.

```bash
.venv/bin/anogen -c configs/shell_mission1.yaml kindproto
```

## What it is

Random proto draws one train-fold anomaly embedding from the **whole** \(R_\mathrm{anom}\) (mostly long campaigns). Kind-conditional proto builds a **per-kind** set and walks toward that kind only:

```
R_kind   =  { e(a) : a is a labeled anomaly of this kind }
i(b)    ~  Uniform(R_kind)     # one proto per generated window, fixed for the DDIM
h_anom   =  ‖ e(x̂_0) − a_{i(b)} ‖²
```

**New runs** use ESA `Length` × `Locality` plus the level-shift overlay
(`kinds_for_windows`). Mapping and math: [ESA_LABELS.md](ESA_LABELS.md).

**This note’s scores and `results/shell_kindproto/` caches** were generated
with the legacy morphology rule (`assign_morphology_kinds`: short spike /
subtle / level shift / medium / campaign). Do not mix those npz files with
ESA-kind plot rows. Re-run `kindproto` (or set `force: true`) if you want
galleries aimed at the official buckets.

Yes: this is how you **ask for a type on purpose**. One gallery per kind. Whether the walk actually emits that waveform is the experiment.

## Denoising steps

Locked S1 schedule is \(T=\) `n_times` **= 200**. Combined (tune id 14) uses **50 DDIM steps** from \(\nu=0.2\), i.e. \(t_\mathrm{start}\approx 40\). Extra steps in that range only repeat times.

This run:

| Recipe | Start | DDIM steps | Why |
|---|---|---:|---|
| `parent50` | nominal parent, \(\nu=0.2\) | 50 | Same as id 14; only the ref list changes |
| `noise200` | \(x_T\sim\mathcal N(0,I)\) | **200** | One iterate per diffusion time; parent baseline is not glued |

\(\lambda_\mathrm{rare}=0\). \(\lambda=0.3\), \(\lambda_\mathrm{anom}=1\), unit ∇. Frozen S1 + fold-0 `time_recon` / `time_both`. 128 S3 parents. Fold-0 Coverage@τ only (frozen \(\tau=0.105\)).

## Ref leak

On Mission 1 41–46, **all 6 level shifts and all 30 medium events are fold 0**. Event-OOF refs for those kinds are empty. Those two galleries use the fold-0 windows themselves (`leaked=True`). That is a capability test (“point at a shelf embedding”), not a cite-vs-GenIAS few-shot number. Spike / subtle / campaign stay event-OOF.

## Scores

Compact 128 / fold-0. **Do not mix** with locked S4 (1536, 3-fold) or hanom (256). Coverage@τ is still **0** on every row: \(\varphi\) z-scores each window, so a real shelf and a generated shelf can sit far apart after the DC is removed.

| Encoder | Recipe | Aimed kind | Cov. | ARP anom | ARP rare | Div | Occ. | frac amp≥0.1 | frac \|Δμ\|≥0.01 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| time_recon | parent50 | short spike | 0 | 0.24 | 0.27 | 10.66 | 0.01 | **0.84** | 0 |
| time_recon | parent50 | short subtle | 0 | 0.21 | 0.25 | 5.83 | 0.01 | 0.59 | 0 |
| time_recon | parent50 | **level shift**† | 0 | 0.22 | 0.29 | 4.04 | 0 | 1.00 | **1.00** |
| time_recon | parent50 | medium event† | 0 | 0.28 | 0.39 | 3.96 | 0 | 0.42 | 0.17 |
| time_recon | parent50 | long campaign | 0 | 0.33 | 0.46 | 4.87 | 0 | 0.52 | 0.03 |
| time_recon | noise200 | short spike | 0 | 0.22 | 0.25 | 9.67 | 0.03 | **0.98** | 0 |
| time_recon | noise200 | short subtle | 0 | 0.15 | 0.17 | 6.37 | 0.02 | 0.89 | 0 |
| time_recon | noise200 | **level shift**† | 0 | 0.22 | 0.25 | 9.76 | 0 | 1.00 | **1.00** |
| time_recon | noise200 | medium event† | 0 | 0.18 | 0.21 | 6.09 | 0 | 0.95 | 0.14 |
| time_recon | noise200 | long campaign | 0 | 0.22 | 0.27 | 6.89 | 0.02 | 0.87 | 0.03 |
| time_both | parent50 | short spike | 0 | 0.17 | 0.19 | 9.73 | 0.01 | **0.87** | 0 |
| time_both | parent50 | short subtle | 0 | 0.30 | 0.40 | 5.38 | 0.01 | 0.55 | 0 |
| time_both | parent50 | **level shift**† | 0 | 0.24 | 0.29 | 4.75 | 0 | 1.00 | **0.96** |
| time_both | parent50 | medium event† | 0 | 0.32 | 0.46 | 4.17 | 0.01 | 0.43 | 0.02 |
| time_both | parent50 | long campaign | 0 | 0.36 | 0.52 | 4.87 | 0.01 | 0.51 | 0.01 |
| time_both | noise200 | short spike | 0 | 0.17 | 0.19 | 9.27 | 0.02 | **0.90** | 0 |
| time_both | noise200 | short subtle | 0 | 0.18 | 0.21 | 5.49 | 0.02 | 0.68 | 0 |
| time_both | noise200 | **level shift**† | 0 | 0.21 | 0.24 | 7.03 | 0 | 1.00 | **0.91** |
| time_both | noise200 | medium event† | 0 | 0.25 | 0.31 | 6.04 | 0 | 0.92 | 0.02 |
| time_both | noise200 | long campaign | 0 | 0.27 | 0.34 | 5.94 | 0.03 | 0.74 | 0 |

† Refs leak fold-0. `cov that kind` was 0 on every kind that has fold-0 queries (including the six leaked shelves). Short subtle has **no** fold-0 queries.

Eyes + raw L2 (time_both parent50, aimed at level shift): median \(\lvert\Delta\mu\rvert=0.022\) (reals are 0.023 / one 0.112). Median L2 to the nearest of the six real shelves is **0.29**. These are close reconstructions of those six windows, not a new shelf family. `noise200` still makes the step (frac 0.91) but is noisier (median amp 0.52 vs 0.28; median L2 0.75).

| Aimed kind | Did it generate that kind? |
|---|---|
| short spike | **Yes, as a needle** — usually a **down** spike at ~2 h. Reals are **up**. |
| short subtle | No — carrier + jitter. “Best-stat” here is smallest amp, not a distinct kind. |
| **level shift** | **Yes** (leaky refs). High then drop to ~0.78 and stay. Typical windows do this too, not only the max-\(\lvert\Delta\mu\rvert\) cherry-pick. |
| medium event | No — edge ticks, lost oscillation. |
| long campaign | No — whole-window DC bump + edge ticks. Not a multi-hour campaign. |

Aiming is **kind-specific**: spike galleries have frac \(\lvert\Delta\mu\rvert\ge 0.01 = 0\); shift galleries have 0.91–1.00. Random proto never did that.

200 DDIM steps from noise did **not** beat parent50 on cleanliness or ARP. Extra unique times add jitter, not a better shelf.

## Plots

Best-stat is the **kind rule** (largest \(\lvert\mu_L-\mu_R\rvert\) for a shift gallery, largest amplitude for a spike), not nearest-\(\varphi\). Shared y per column on the overview grids.

| File | What |
|---|---|
| `plots/real_vs_aimed_time_both.png` | Real vs best-stat `parent50` / `noise200` |
| `plots/real_vs_aimed_{encoder}_{recipe}.png` | Real / best / typical per kind |
| `plots/real_vs_aimed_{encoder}_{recipe}_{slug}.png` | Four reals vs four best-stat gens |
| `plots/aimed_{encoder}_{recipe}_{slug}.png` | Eight best-stat generated windows |
| `plots/level_shift_{encoder}_{recipe}.png` | Shelf check, raw units |

Look at `level_shift_*` first: a real shelf is high then low. If best \(\lvert\Delta\mu\rvert\) is still a flat carrier, aiming at the embedding was not enough.

To keep shelves **and** quiet kinds in one gallery (stratified proto, kind-balanced soft, two-recipe mix): [KIND_MIX.md](KIND_MIX.md).
