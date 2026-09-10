# Implementation order

Steered diffusion is the method. GenIAS and post-hoc are the references. No SCM or graph.

Label roles (Anomaly vs Rare Event vs nominal): [STEERING_MATH.md](STEERING_MATH.md) §0. Official ESA `Length` / `Locality` vs the one invented level-shift overlay: [ESA_LABELS.md](ESA_LABELS.md). Train pool = everyday + rares. Q_q from that pool’s held-out split. Anomalies stay out of S1/S2/S3. Candidate ranking vs GenIAS: [PAPER_CANDIDATES.md](PAPER_CANDIDATES.md). Combined vs refs (continuous 41–46): [COMBINED_VS_REFS.md](COMBINED_VS_REFS.md). \(h_\mathrm{anom}\) nearest / kNN / proto: [H_ANOM.md](H_ANOM.md). Level-shift ideas (DC, \(\mu_L\)): [LEVEL_SHIFT.md](LEVEL_SHIFT.md). Kind-conditional proto: [KIND_PROTO.md](KIND_PROTO.md). Mixed galleries (stratified / kind-balanced soft / two-recipe): [KIND_MIX.md](KIND_MIX.md). Hybrid slices: [HYBRID.md](HYBRID.md). λ sweep: [KIND_MIX_HTUNE.md](KIND_MIX_HTUNE.md). Shortlist plots: [FINDINGS.md](FINDINGS.md). Other channels / Mission 2 (isolated retrain): [XFER.md](XFER.md).

| Step | What |
|---|---|
| S0 | Windows, event OOF, sealed guard, freeze protocol |
| S1 | Univariate denoiser on nominal windows |
| S2 | Shell encoder, Q_q, DDIM + band guidance |
| S3 | GenIAS and post-hoc on the same windows |
| S4 | OOF coverage: shell vs GenIAS vs post-hoc vs unguided |
| tune | OOF sampling-HP search (steps, λ, ν, n_correct, rare/anom energies). Does not overwrite S4. |
| enc | Encoder ablations: temporal latent vs pool, recon vs 3-class SupCon. Does not overwrite S2. |
| S5 | Event-OOF adapters on locked TSDiff; S5 × time_* combined. Docs: [S5.md](S5.md), [REPO.md](REPO.md) |
| S6 | One sealed-test pass after freeze |
| plots | Time-series PNGs |
| xfer | Compact retrain + score on M2 and extra M1 channels. Isolated `results/xfer/`. Docs: [XFER.md](XFER.md), [DISCRETE_CHANNELS.md](DISCRETE_CHANNELS.md) |
| hanom | Three \(h_\mathrm{anom}\) strategies on locked 41–46. Isolated `results/shell_hanom/`. Docs: [H_ANOM.md](H_ANOM.md) |
| kindproto | Kind-conditional proto + 200-step from-noise. Isolated `results/shell_kindproto/`. Docs: [KIND_PROTO.md](KIND_PROTO.md) |
| kindmix | Stratified proto, kind-balanced soft, two-recipe mix. Isolated `results/shell_kindmix/`. Docs: [KIND_MIX.md](KIND_MIX.md) |
| kindmixscore | ESA-kind stratified proto at 1536 / 3-fold. Isolated `results/shell_kindmix_score_esa/`. Morphology freeze stays in `results/shell_kindmix_score/`. |
| kindmixhtune | λ / λ_anom sweep on stratified + hybrid slices. Isolated `results/shell_kindmix_htune/`. Docs: [HYBRID.md](HYBRID.md), [KIND_MIX_HTUNE.md](KIND_MIX_HTUNE.md) |

CausalDiscovery path (read-only): `/mnt/extras/SSD/AI/CausalDiscovery`.
