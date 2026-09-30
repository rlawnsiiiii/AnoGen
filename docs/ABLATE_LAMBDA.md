# λ_shell = 0 and no-steer ablations

Isolated `results/shell_ablate_lambda/`. Does **not** overwrite
`results/shell_s4` or the ν=0.2 hybrid / combined freezes.

```bash
.venv/bin/anogen -c configs/shell_mission1.yaml ablatelambda
```

Same 1536 / 3-fold protocol as encscore. parent50 otherwise (ν=0.2, 50 DDIM,
unit ∇, λ_rare=0). Fold-0 encoder weights only.

| Tag | λ (shell) | λ_anom | What it is |
|---|---:|---:|---|
| `stratified_protoonly` | 0 | 1 | Proto on every present ESA kind |
| `hybrid_needles_protoonly` | 0 | 1 | Proto on shift + Point/Global; other slices unguided |
| `nosteer` | 0 | 0 | Zero force (ν=0.2 parent edit) |

Plots (aimed / nearest per kind): `results/shell_ablate_lambda/plots/` and
[`docs/ablate_lambda/`](ablate_lambda/INDEX.txt). Start with
`ablate_time_both_compare_aimed.png`. ARP / EDI:
`results/shell_ablate_lambda/summary.json`. EDI is a **new** union — compare
ARP to GenIAS/post-hoc freely; do not mix this EDI with hybrid `*` or the
9-method table.

## Scores (1536 / 3-fold, 2026-09-11)

| Method | λ | λ_anom | ARP anom | EDI |
|---|---:|---:|---:|---:|
| GenIAS ψ=2 | — | — | 0.317 | 0.82 |
| Post-hoc | — | — | 0.319 | 0.90 |
| Unguided ν=1 | — | — | 0.568 | 1.57 |
| time_recon stratified proto-only | 0 | 1 | 0.405 | 2.31 |
| time_both stratified proto-only | 0 | 1 | 0.398 | 2.26 |
| time_recon hybrid_needles proto-only | 0 | 1 | 0.578 | 2.44 |
| time_both hybrid_needles proto-only | 0 | 1 | **0.583** | 2.39 |
| time_recon no-steer | 0 | 0 | 0.470 | 1.54 |
| time_both no-steer | 0 | 0 | 0.451 | 1.52 |

Locked parent50 hybrid_needles (λ=0.3, λ_anom=1) was ARP 0.584. Dropping the
shell barely changes that row. Full stratified proto without the shell is
worse (ARP ~0.40, Coverage 0). No-steer (ν=0.2, zero ∇f) is **not** the
locked unguided ν=1 row.
