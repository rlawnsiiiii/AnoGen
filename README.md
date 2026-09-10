# AnoGen

Univariate **shell-steering diffusion** for ESA-ADB Mission 1 channels 41–46, with **GenIAS** and a handcrafted **post-hoc** injector as references.

Core sampling math and how Anomalies / Rare Events are used: [docs/STEERING_MATH.md](docs/STEERING_MATH.md) (§0). ESA-ADB `Length` / `Locality` map plus the invented level-shift overlay: [docs/ESA_LABELS.md](docs/ESA_LABELS.md). Paper-candidate overview (architectures, locked S4 vs GenIAS): [docs/PAPER_CANDIDATES.md](docs/PAPER_CANDIDATES.md). Which script scored which row: [docs/REPO.md](docs/REPO.md). Adapters: [docs/S5.md](docs/S5.md). Other channels / Mission 2: [docs/XFER.md](docs/XFER.md). Hybrid slices (band vs proto per kind): [docs/HYBRID.md](docs/HYBRID.md). Recommended \(\lambda,\lambda_\mathrm{anom}\): [docs/KIND_MIX_HTUNE.md](docs/KIND_MIX_HTUNE.md). Shortlist eyes (`time_recon` / `time_both` × hybrid / needles): [docs/FINDINGS.md](docs/FINDINGS.md).

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
uv run anogen -c configs/shell_mission1.yaml kindmixhtune  # λ / λ_anom + hybrid slices
uv run anogen -c configs/xfer.yaml xfer   # M2 + extra M1 channels (isolated)
uv run pytest -q
```

PNGs: `results/shell_plots/` (`INDEX.txt` lists them). Start with `real_vs_generated.png` and `same_parent_grid.png`. Sealed Mission 1 test stays closed until S6.

Train pool is everyday nominals **plus Rare Events** (`train_index.csv`). Anomalies stay out. Bump `shell.rare_upsample` later if rares should be counted more than once. The current S1/S2/S3 checkpoints were fit before that change; re-run those phases to train on the new pool.
