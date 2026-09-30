# Repository map — where code and scores live

Read this before re-running a number. Locked S4 (`results/shell_s4`) is not overwritten. S6 stays sealed.

CLI: `.venv/bin/anogen -c configs/shell_mission1.yaml <phase>`  
(or `uv run anogen …`). Config paths: `configs/shell_mission1.yaml`.

## Layout

```
src/anogen/
  cli.py                 phase dispatch
  config.py              YAML + CausalDiscovery panel path
  phases/
    s0.py                windows, event OOF, protocol.json
    s1.py                TSDiff denoiser → results/shell_s1/denoiser.pt
    s1_recon.py          recon diagnostics (no new weights)
    s2.py                pool_recon encoder + band smoke → shell_s2/encoder.pt
    s3.py                GenIAS + post-hoc galleries → shell_s3/
    s4.py                locked Coverage@τ / ARP / EDI / win rule
    tune.py              OOF sampling HPs (does not overwrite S4)
    enc.py               encoder ablations → shell_enc/*_fold0.pt
    encscore.py          time_recon / time_both × band|combined @ 1536
    s5.py                event-OOF adapters + S5×combined scores + plots
    xfer.py              compact retrain + score on other channels / M2
    hanom.py             nearest / knn / proto h_anom on locked 41–46
    kindproto.py         kind-conditional proto (per-kind \(R_\mathrm{anom}\))
    kindmix.py           stratified proto / kind-balanced soft / two-recipe mix
    kindmixscore.py      time_* × stratified / hybrid / hybrid_needles @ 1536 / 3-fold
    kindmixhtune.py      λ / λ_anom sweep on stratified + hybrid slices
    noisescore.py        hybrid / needles / combined from x_T ~ N(0,I) @ 1536
    ablate_lambda.py     λ_shell=0 proto-only and no-steer @ 1536
    quiettune.py         small λ_anom on hybrid_needles subsequence slices
    genbase.py           CutAddPaste + Lai taxonomy @ 1536 / frozen τ
    editor.py            residual-copy / deviation-patch on hybrid_needles
    timeleash.py         hybrid_needles + time-domain donor leash
    hashfix.py           last Tweedie / 20 DDIM / clip-in-box ∇
    realism.py           raw-window fidelity audit on saved galleries
    augdetect.py         supervised augmentation of a 1-D CNN (dev only)
    plots.py             S4-era gallery PNGs
    plot_tune.py         tune / encoder×recipe strips
    plot_from_noise.py   time_recon+combined from x_T ~ N(0,I)
  shell/
    diffusion.py         schedule, DDIM, unguided_from_nominal, ckpt load
    tsdiff.py            S4-D residual ε-backbone
    steer.py             ConvEncoder, guided_ddim, start_from_noise, anom_energy
    encoders.py          ShellEncoder (pool / time, recon / SupCon)
    adapters.py          ResidualAdapter, AdaptedDenoiser, AnomalyClassifier
    detector.py          fit/eval for augdetect (does not edit S5 train_classifier)
    genias.py            TCN-VAE, ψ inflate, Alg. 2 patch
    coverage.py          Coverage@τ, ARP, Div, EDI
    features.py          locked φ = feature_pack_v1 (12-D)
    morphology.py        duration+waveform kinds + kind-conditional ref index
    plots.py             save_strip / save_compare_rows
    scaler.py            per-channel min-max
    protocol.py          frozen S0 protocol + win rule
    windows.py / data.py / baselines.py  (posthoc, CutAddPaste, Lai taxonomy)
    editor.py            residual copy + deviation-patch on a donor
    realism.py           published fidelity measures + anomaly diagnostics
    xfer_panel.py        slice G1 panel or mean-bin pickles (sealed crop)
    folds.py             event OOF + keep_fold_mask (named channels)
docs/                    PAPER_CANDIDATES, RESULTS, STEERING_MATH, ESA_LABELS, S5, XFER, DISCRETE_CHANNELS, COMBINED_VS_REFS, H_ANOM, LEVEL_SHIFT, KIND_PROTO, KIND_MIX, HYBRID, KIND_MIX_HTUNE, FINDINGS, QUIET_TUNE, GEN_BASELINES, EDITOR, TIME_LEASH, HASHFIX, REALISM, COMPARISON, CHANMIX, NEXT_STEPS_2026-09-30, AUGDETECT, this file
configs/shell_mission1.yaml
configs/xfer*.yaml       Mission 2 + M1 extra (not 41–46)
results/                 gitignored artifacts
tests/test_shell_*.py
```

## Checkpoints

| File | What |
|---|---|
| `results/shell_s1/denoiser.pt` | Frozen TSDiff score (~316k). Every diffusion sample. |
| `results/shell_s2/encoder.pt` | Locked S2 pool_recon + Q_q / τ / δ |
| `results/shell_s3/genias.pt` | TCN-VAE (baseline, not the score) |
| `results/shell_enc/{name}_fold0.pt` | Encoder ablations, fold 0 only |
| `results/shell_s5/adapter_fold{k}.pt` | OOF residual adapters (new S5) |

## Which script scored which table row

All generation-quality numbers in `PAPER_CANDIDATES.md` §2 use **the same** `φ` (`src/anogen/shell/features.py`), **the same** τ from `results/shell_s4/summary.json`, and **the same** 1536 S3 donors (`galleries.npz` → `cond`). Event-OOF queries: `src/anogen/phases/tune.py` `_score_gallery` / `src/anogen/phases/s4.py` `_score_methods`.

| Table row | Generate | Score | Gallery on disk |
|---|---|---|---|
| Unguided ν=1 | `s4.py` → `unguided_from_nominal` (`shell/diffusion.py`) | `s4.py` `_score_methods` | `shell_s4/shell_gallery.npz` key `unguided` |
| Shell ZS | `s4.py` → `guided_ddim` band, S2 encoder | same | `shell_s4/shell_gallery.npz` key `x` |
| GenIAS ψ=2 | `s3.py` → `genias_sample` | `s4.py` | `shell_s3/galleries.npz` key `genias` |
| GenIAS + patch | `s3.py` Alg. 2 | `s4.py` | `shell_s3/galleries.npz` key `genias_patched` |
| Post-hoc | `s3.py` → `posthoc_inject` | `s4.py` | `shell_s3/galleries.npz` key `posthoc` |
| CutAddPaste / taxonomy | `genbase.py` | `genbase.py` → `_score_methods` | `shell_genbase/{cutaddpaste,taxonomy}.npz` |
| Downstream CNN | galleries above | `augdetect.py` event recall @ 1% FAR | `shell_augdetect/metrics.csv` |
| time_* + band | `encscore.py` `_BAND` + `chunked_guided_ddim` | `encscore.py` → `_score_methods` | `shell_enc_score/{variant}_band.npz` |
| time_* + combined | `encscore.py` `_combined_oof` | fold-wise `_score_gallery` | `shell_enc_score/{variant}_combined_fold{k}.npz` |
| S5 adapter-only | `s5.py` `unguided_from_nominal` on `AdaptedDenoiser` | `s5.py` → `_score_gallery` | `shell_s5/fold{k}_galleries.npz` |
| S5 adapter+cls | `s5.py` `chunked_guided_ddim` S2 band + `AnomalyClassifier` | same | same npz keys `adapter_cls` |
| S5 + time_* combined | `s5.py` adapted model + `_COMBINED` + OOF refs | same | keys `adapter_time_*_combined` |
| xfer (M2 / M1 extra) | `xfer.py` new denoiser + GenIAS + time_recon | `xfer.py` (per-dataset τ) | `results/xfer/{name}/` |
| h_anom nearest / knn / proto | `hanom.py` combined \(f\) + `anom_energy_kind` | fold-0 `_score_gallery`, 256 donors | `results/shell_hanom/` |
| kind-conditional proto | `kindproto.py` proto on \(R_\mathrm{kind}\) | fold-0 `_score_gallery`, 128 donors | `results/shell_kindproto/` |
| mixed kinds (morphology caches) | `kindmix.py` | fold-0 `_score_gallery`, 256 donors | `results/shell_kindmix/` |
| mixed kinds (ESA) | `kindmix.py` stratified | fold-0 `_score_gallery`, 256 donors | `results/shell_kindmix_esa/` |
| time_* + ESA stratified proto | `kindmixscore.py` `_stratified` + OOF refs | fold-wise `_score_gallery` | `shell_kindmix_score_esa/{variant}_stratified_fold{k}.npz` |
| time_* + hybrid / hybrid_needles | `kindmixscore.py` `run_kindmixscore_hybrid` | same | `shell_kindmix_score_hybrid/{variant}_{recipe}_fold{k}.npz` |
| editor on hybrid_needles | `editor.py` residual copy + deviation-patch | fold-wise `_score_gallery` | `shell_editor/{variant}_hybrid_needles_editor_fold{k}.npz` |
| time leash on hybrid_needles | `timeleash.py` `lam_parent` in `guided_ddim` | same | `shell_timeleash/time_both_{l2,gm}_fold{k}.npz` |
| hash knobs on hybrid_needles | `hashfix.py` `apply_final_grad=False`, 20 DDIM, clip-in-box | same | `shell_hashfix/time_both_hashfix_fold{k}.npz` |
| raw-window realism audit | `realism.py` reads saved galleries only; no φ/S4 change | published raw fidelity + diagnostics | `shell_realism/{metrics,diagnostics,c2st}.csv` |
| hybrid_needles, channel-stratified kinds | `kindmixscore.py` with `shell.kindmixscore.channel_stratified` | same | `shell_chanmix/{variant}_hybrid_needles_fold{k}.npz` |
| time_* + hybrid / combined from noise | `noisescore.py` `start_from_noise` | same | `shell_noise_score/{variant}_{recipe}_fold{k}.npz` |
| time_* + morphology stratified (freeze) | same script, old kinds | same | `shell_kindmix_score/` |

EDI for the **encscore 9-method** table: `encscore.py` `_edi_table_union` → `shell_enc_score/summary.json` `edi_table_union`.  
EDI for **S5-extended** table: `s5.py` `_edi_s5_union` → `shell_s5/summary.json` `edi_table_union` (new union; do not mix with the 9-method EDI).  
EDI for **stratified** rows: `kindmixscore.py` `_edi_strat_union` → `shell_kindmix_score_esa/summary.json` `edi_table_union` (S4 + encscore + two ESA stratified fold-0 galleries; mark ‡).  
EDI for **hybrid / hybrid_needles** rows: `kindmixscore.py` `_edi_recipe_union` → `shell_kindmix_score_hybrid/summary.json` `edi_table_union` (S4 + encscore + four hybrid fold-0 galleries; mark \*). Do not mix ‡ / \*.  
EDI for **from-noise** rows: `noisescore.py` → `shell_noise_score/summary.json` `edi_table_union` (mark ¶). Div is comparable to the ν=0.2 table; EDI is not.  
EDI for **CutAddPaste / taxonomy**: `genbase.py` → `shell_genbase/summary.json` `edi_table_union` (mark †). Do not mix † with \* / unmarked / ¶ / ‡.  
EDI for **editor**: `editor.py` → `shell_editor/summary.json` `edi_table_union` (mark **e**). Do not mix with \* / † / unmarked / ¶ / ‡.  
EDI for **time leash**: `timeleash.py` → `shell_timeleash/summary.json` `edi_table_union` (mark **t**). Do not mix with **e** / \* / † / unmarked / ¶ / ‡.  
EDI for **hash knobs**: `hashfix.py` → `shell_hashfix/summary.json` `edi_table_union` (mark **h**). Do not mix with **t** / **e** / \* / † / unmarked / ¶ / ‡.  
EDI for **chanmix**: `kindmixscore.py` with `channel_stratified` → `shell_chanmix/summary.json` `edi_table_union` (mark **c**; 11 members, no `hybrid` recipe). Same recipe name as \* but a different partition — do not mix with **h** / **t** / **e** / \* / † / unmarked / ¶ / ‡.  
Div is always `coverage.mean_pairwise_distance` on fold-0 / single 1536 gallery — comparable across tables.

## Plot scripts

| Figure family | Script | Output |
|---|---|---|
| S4 galleries | `phases/plots.py` | `results/shell_plots/gen_*.png` |
| Encoder × recipe strips | `phases/plot_tune.py` | `shell_plots/shell_tune/` |
| time_recon+combined from noise | `phases/plot_from_noise.py` | `shell_tune/gen_time_recon_combined_from_noise.png` |
| S5 ν=0.2 and from-noise | `phases/s5.py` `_write_plots` | `shell_plots/shell_s5/` |
| Kind-conditional proto | `phases/kindproto.py` | `results/shell_kindproto/plots/` |
| Mixed-gallery kinds | `phases/kindmix.py` | `results/shell_kindmix/plots/` |
| Hybrid λ / λ_anom sweep | `phases/kindmixhtune.py` | `docs/kindmix_htune/` (copy; also `results/shell_kindmix_htune/plots/`) |
| Quiet-kind λ_anom | `phases/quiettune.py` | `docs/quiettune/` |
| CutAddPaste / taxonomy | `phases/genbase.py` | `docs/genbase/` |
| Editor on hybrid_needles | `phases/editor.py` | `docs/editor/` |
| Time-domain donor leash | `phases/timeleash.py` | `docs/timeleash/` |
| Hash knobs (last Tweedie / 20 DDIM / clip ∇) | `phases/hashfix.py` | `docs/hashfix/` |
| Raw-window realism audit | `phases/realism.py` | `docs/realism/` |
| Envelope eyeballs (real vs c1 vs hashfix) | `scripts/plot_realism_eyes.py` | `docs/realism_eyes/` |
| Generations per targeted kind | `scripts/plot_c1_by_kind.py` | `docs/c1_by_kind/`, `docs/chanmix_by_kind/` |
| Downstream detector | `phases/augdetect.py` | `docs/augdetect/` |
| λ_shell=0 / no-steer | `phases/ablate_lambda.py` | `docs/ablate_lambda/` (copy; also `results/shell_ablate_lambda/plots/`) |

Shared drawing: `shell/plots.py` `save_strip`, `save_compare_rows`. Kinds: `shell/morphology.py`.

## Sampling entry points (to read first)

| Call | File |
|---|---|
| ε-DDIM, no ∇f | `unguided_from_nominal` in `shell/diffusion.py` |
| Guided DDIM (band / combined / cls / from noise) | `guided_ddim` in `shell/steer.py` |
| Adapted ε | `AdaptedDenoiser.forward` in `shell/adapters.py` |
| GenIAS decode | `genias_sample` in `shell/genias.py` |

## What not to run

- `anogen s4` — would overwrite the locked table.
- `anogen s6` — sealed.
- Mixing `results/xfer/` Coverage@τ / EDI with locked S4. See [XFER.md](XFER.md).
- Old `results/shell_s5/summary.json` from before 2026-09-09 (U-Net, τ=0.45). The new S5 replaces that directory.
