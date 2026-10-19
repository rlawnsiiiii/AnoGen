# AnoGen

Univariate shell-steering diffusion for ESA-ADB Mission 1, channels 41–46. The generator used in the paper draft is `flip_x0_contrast_head_sharp_nomask`. Its specification, the edit it applies, and the numbers quoted for it are in [docs/flip_x0_contrast_head_sharp_nomask.md](docs/flip_x0_contrast_head_sharp_nomask.md). Kind names are in [docs/ESA_LABELS.md](docs/ESA_LABELS.md). The run that selected this recipe is [docs/ESA_RUN_2026-10-06.md](docs/ESA_RUN_2026-10-06.md) §11.

## Paper figures

`paper/draft/main.tex` includes three figures. All three are drawn by `slides/141026/make_figs.py` from the saved run, with `MODEL = flip_x0_contrast_head_sharp_nomask`. Nothing in that script retrains or resamples.

| Figure | What it shows | Saved inputs |
|---|---|---|
| `figs/kinds.pdf` | one real window of each anomaly kind, channel 41 | `results/shell_s0/labeled_arrays.npz`, `labeled_windows.csv`, `minmax_scaler.npz`; nominal band from `results/shell_s3/galleries.npz` key `cond`. The script also opens the three galleries below, because the loader is shared. |
| `figs/gallery.pdf` | three real windows and five generated windows per kind | `results/shell_fixsweep/flip_x0_contrast_head_sharp_nomask_fold{0,1,2}.npz` (`x`, `kind_alloc`, `channel_idx`) |
| `figs/sweep.pdf` | event recall against the number of real training anomalies | `results/shell_augdetect/sweep.json`, arms `real_only` and `fixsweep:flip_x0_contrast_head_sharp_nomask+real` |

The `+0.200` mark on the sweep figure is the paired difference at k = 32 in that `sweep.json` (`vs_real_only`, interval [+0.073, +0.318]). It is not the vertical gap between the two median curves.

From a checkout that already has those result files:

```bash
cd /mnt/extras/SSD/AI/AnoGen
uv sync --extra neural
cd paper/draft
make figs    # redraw the three PDFs into paper/draft/figs/
make         # compile main.pdf with tectonic
```

`make figs` runs `slides/141026/make_figs.py` and copies `kinds.pdf`, `gallery.pdf` and `sweep.pdf`. The script also writes `diffusion_strip.pdf` and `shell.pdf` under `slides/141026/figs/` for the briefing deck; the paper does not include them. Tectonic is `~/.local/bin/tectonic`. The first `make` needs network so it can fetch LaTeX packages. There is no TeX Live on this machine.

To regenerate the galleries the gallery figure reads, rather than redraw from the saved ones, the phases below have to exist first: `s0`, `s1` (`results/shell_s1/denoiser.pt`), `s3` (`galleries.npz`), `s4` (`summary.json`, for the frozen τ), and `enc` (`results/shell_enc/time_both_fold0.pt`). Then run only this recipe, so the other fixsweep variants are left as they are. Copy `configs/shell_mission1.yaml`, and under `shell.fixsweep` set

```yaml
only: [flip_x0_contrast_head_sharp_nomask]
```

The per-fold cache is keyed by the spec. A matching `flip_x0_contrast_head_sharp_nomask_fold{0,1,2}.npz` is reused; delete those three files to force a new sample. The comparison with every edit off is the separate variant `nosteer_flip_x0_recur2`. Add it to `only` if that pair should be regenerated too. Then:

```bash
uv run anogen -c <that copy> fixsweep
```

The sweep figure reads `results/shell_augdetect/sweep.json` as it stands. Re-running `augdetect` into that directory replaces the file the paper figure is drawn from. Leave it unless the detector curve itself is being redone. The sweep block is commented out in `configs/shell_mission1.yaml`; the saved file is the one `make figs` uses.

## Other notes

Sampling math: [docs/STEERING_MATH.md](docs/STEERING_MATH.md). Position-bias switches: [docs/POSITION_BIAS.md](docs/POSITION_BIAS.md). Torch-free mechanism checks: [docs/TESTBED.md](docs/TESTBED.md). Locked S4 vs later scores: [docs/PAPER_CANDIDATES.md](docs/PAPER_CANDIDATES.md). Which script scored which row: [docs/REPO.md](docs/REPO.md). Adapters: [docs/S5.md](docs/S5.md). Other channels / Mission 2: [docs/XFER.md](docs/XFER.md).

CausalDiscovery is the data and sealed-split repo. This project reads `panel_light.npz` and labels from there; it does not copy telemetry. There is no SCM, PCMCI+, or intervention code here.

```bash
cd /mnt/extras/SSD/AI/AnoGen
uv sync --extra neural --extra dev
uv run anogen -c configs/shell_mission1.yaml s0   # windows, OOF, protocol
uv run anogen -c configs/shell_mission1.yaml s1   # nominal denoiser
uv run anogen -c configs/shell_mission1.yaml s2   # shell encoder + guided DDIM
uv run anogen -c configs/shell_mission1.yaml s3   # GenIAS + post-hoc references
uv run anogen -c configs/shell_mission1.yaml s4   # coverage: shell vs GenIAS vs post-hoc vs unguided
uv run anogen -c configs/shell_mission1.yaml tune # OOF sampling HPs + rare/anom energies
uv run anogen -c configs/shell_mission1.yaml enc  # encoder ablations (temporal / SupCon)
uv run anogen -c configs/shell_mission1.yaml encscore  # time_recon / time_both @ 1536
uv run anogen -c configs/shell_mission1.yaml s5   # event-OOF adapters + S5×combined
uv run anogen -c configs/shell_mission1.yaml plots
uv run anogen -c configs/shell_mission1.yaml hanom      # h_anom nearest / knn / proto
uv run anogen -c configs/shell_mission1.yaml kindproto  # aim proto at one anomaly kind
uv run anogen -c configs/shell_mission1.yaml kindmix    # stratified / kind-balanced soft / mix
uv run anogen -c configs/shell_mission1.yaml kindmixscore  # stratified proto @ 1536 / 3-fold
uv run anogen -c configs/shell_mission1.yaml kindmixscorehybrid  # hybrid / needles @ 1536 / 3-fold
uv run anogen -c configs/shell_mission1.yaml noisescore  # same recipes from x_T ~ N(0,I)
uv run anogen -c configs/shell_mission1.yaml ablatelambda  # λ_shell=0 and no-steer
uv run anogen -c configs/shell_mission1.yaml kindmixhtune  # λ / λ_anom + hybrid slices
uv run anogen -c configs/xfer.yaml xfer   # M2 + extra M1 channels (isolated)
uv run python scripts/diagnose_causality.py -c configs/shell_mission1.yaml  # 5 min: causal backbone check
uv run python scripts/audit_metrics.py -c configs/shell_mission1.yaml       # CIs, controls, PRDC on saved galleries
uv run anogen -c configs/shell_mission1.yaml fixsweep    # position-bias / guidance fixes (docs/POSITION_BIAS.md)
uv run anogen -c configs/shell_mission1.yaml s1bidir     # bidirectional S4-D into results/shell_s1_bidir
uv run anogen -c configs/shell_mission1.yaml geniasfair  # GenIAS with its published objective
uv run pytest -q
```

PNGs: `results/shell_plots/` (`INDEX.txt` lists them). Start with `real_vs_generated.png` and `same_parent_grid.png`. Sealed Mission 1 test stays closed until S6.

Train pool is everyday nominals **plus Rare Events** (`train_index.csv`). Anomalies stay out. Bump `shell.rare_upsample` later if rares should be counted more than once. The current S1/S2/S3 checkpoints were fit before that change; re-run those phases to train on the new pool.
