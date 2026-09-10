# \(h_\mathrm{anom}\) strategies — nearest / kNN / proto

Locked combined \(f\) reuses the **nominal KDE** for attraction to labeled faults. That can pull a sample toward the **barycenter** of \(R_\mathrm{anom}\), which on Mission 1 41–46 is mostly long campaigns (near-nominal). Coverage@τ / ARP only need **some** labeled fault nearby.

This note lists three replacements for \(h_\mathrm{anom}\) only. \(h_\mathrm{nom}\) stays the soft KDE (the band is a density level). \(h_\mathrm{rare}\) stays full-set soft energy (repel the rare cloud). Combined λ / ν / steps stay tune id 14. Frozen S1 + fold-0 `time_recon` / `time_both`. **Does not overwrite** `results/shell_s4`.

Sampling math of the locked terms: [STEERING_MATH.md](STEERING_MATH.md). Eyes vs GenIAS: [COMBINED_VS_REFS.md](COMBINED_VS_REFS.md). How to get a **level shift** (what DC and \(\mu_L\) mean): [LEVEL_SHIFT.md](LEVEL_SHIFT.md). Kind rows on new plots are ESA `Length` × `Locality` plus the level-shift overlay: [ESA_LABELS.md](ESA_LABELS.md). Cached `results/shell_hanom/` figures used the older morphology names.

Reproduce:

```bash
.venv/bin/anogen -c configs/shell_mission1.yaml hanom
```

Artifacts: `results/shell_hanom/` (npz) and `results/shell_hanom/plots/` (same layout as `real_kinds_vs_time_both_combined.png`).

---

## 0. What is wrong with locked \(h_\mathrm{anom}\)

```
h_anom(x)  =  −τ log  Σ_i exp( −‖z − a_i‖² / τ )
∇_z h_anom =  2 ( z − Σ_i w_i a_i ),   w = softmax(−d_i² / τ)
```

\(\tau\) is the **nominal** median pairwise distance. Default sampling starts on a nominal parent (\(\nu=0.2\)): first steps are far from every \(a_i\), so \(w_i \approx 1/K\) and the walk aims at the **mean of all** train-fold anomaly embeddings.

On 41–46, \(R_\mathrm{anom}\) is 594 / 672 long campaigns, 36 short spikes, 6 level shifts. The mean is “average campaign.” Soft energy is the right object for \(Q_q\). It is the wrong attractor for a sparse, multi-modal, mostly-near-nominal fault set.

Eval is already min-distance in \(\varphi\). These three strategies make \(h_\mathrm{anom}\) match that idea: **close to one (or a few) real faults is enough.**

---

## 1. Three strategies

Code: `anom_energy` in [`src/anogen/shell/steer.py`](../src/anogen/shell/steer.py). `guided_ddim(..., anom_energy_kind=...)`. Default `soft` is unchanged (locked S4 / encscore / tune).

### 1.1 Nearest (`nearest`) — hard min

```
h_anom(x)  =  min_i  ‖ e(x̂_0) − a_i ‖²
∇_z h      =  2 (z − a_{i*})     # i* = nearest ref
```

\(\tau \to 0\) limit of the locked soft energy. The force always points at **one** fault. Between two clusters you go to the closer one, not into the gap.

Use when the claim is “Coverage@τ-like: some labeled anomaly nearby.”

### 1.2 Local kNN (`knn`) — soft energy on the \(k\) nearest only

```
N_k(z)     =  the k refs with smallest ‖z − a_i‖²
h_anom(x)  =  −τ log  Σ_{a ∈ N_k} exp( −‖z − a‖² / τ )
```

Default \(k=8\). Same kernel \(\tau\) as the band, but the sum cannot see the rest of \(R_\mathrm{anom}\). Starting on a nominal, the first \(k\) are the nearest campaigns/spikes — not the global mean of 256 refs. The neighborhood can change during the walk.

Use when you still want a smooth KDE, but only **locally**.

### 1.3 One proto per sample (`proto`)

```
i(b)       ~  Uniform({1…|R_anom|})    # drawn once, kept for the whole DDIM
h_anom^{(b)}  =  ‖ e(x̂_0^{(b)}) − a_{i(b)} ‖²
```

Each generated window is assigned **one** train-fold anomaly embedding and walks toward that point only. Gallery diversity comes from which proto you drew. A level-shift proto can be requested; a soft KDE over all 256 refs cannot.

Use when the claim is kinds / controllable few-shot: “look like **this** fault.”

---

## 2. What stayed the same

```
f  =  λ (h_nom − Q_q)²  +  λ_anom h_anom  +  λ_rare (−h_rare)
```

| Piece | Locked | This run |
|---|---|---|
| Score ε | `results/shell_s1/denoiser.pt` | same |
| Encoder | fold-0 `time_recon` / `time_both` | same |
| Band / rare | soft KDE, nominal \(\tau\) | band same; rare dropped in the `norare` family |
| λ, λ_anom, λ_rare, ν, steps | id 14 (0.3 / 1 / 1 / 0.2 / 50, unit ∇) | same, or λ_rare = 0 (`results/shell_hanom/norare/`) |
| Donors | first 256 of S3 `cond` (same as `shell_tune` plots) | same |
| Contrast refs | event-OOF, fold ≠ 0, up to 256 | same |
| Coverage τ | S4 freeze 0.105 | fold-0 queries only (not a new encscore freeze) |

`soft` baseline plots reuse `results/shell_plots/shell_tune/{time_recon,time_both}_tune_galleries.npz` key `combined`. Not regenerated.

---

## 3. Scores (256 donors, fold-0, frozen τ)

Compact. **Do not mix** with locked S4 (1536, 3-fold) or encscore. Locked `time_recon` + soft combined at 1536 was Cov 0.023 / ARP 0.54 / occ 0.78.

| Encoder | \(h_\mathrm{anom}\) | Cov. anom | Gap | ARP anom | ARP rare | Div | Occ. |
|---|---|---:|---:|---:|---:|---:|---:|
| time_recon | nearest | 0 | 0 | 0.38 | 0.69 | 2.95 | 0.00 |
| time_recon | knn | 0 | 0 | 0.39 | 0.67 | 2.89 | 0.04 |
| time_recon | proto | 0 | 0 | 0.39 | 0.59 | 5.10 | 0.01 |
| time_both | nearest | 0 | 0 | 0.39 | 0.68 | 2.74 | 0.00 |
| time_both | knn | 0 | 0 | 0.38 | 0.71 | 2.11 | 0.08 |
| time_both | **proto** | 0 | 0 | **0.42** | 0.62 | **5.88** | 0.01 |

None of the three beats soft combined on Coverage@τ or occupancy. All six rows have **Coverage@τ = 0** on this 256 / fold-0 cut. Occupancy collapses (0–0.08): the new \(h_\mathrm{anom}\) overpowers the band. ARP anom (0.38–0.42) is **below** soft combined (0.54) and **below ARP rare** — the galleries sit closer to rares than to anomalies.

**Proto** is the only strategy that does what we asked on diversity (Div 5.1–5.9 vs ~2.1–3.0). Assigning one fault per window spreads the gallery; min / kNN still collapse toward the nearest *cluster* (mostly campaigns).

Eyes (`real_kinds_vs_time_both_hanom.png`, `real_kinds_vs_time_both_proto.png`): **level shift is still missing.** Nearest-φ match for that kind is the carrier oscillation for soft, nearest, knn, and proto. Short spikes still appear. kNN looks noisier. Proto can emit a sharper early spike; it does not emit a 0.95→0.80 shelf. Q1-vs-Q4 |jump| ≥ 0.02 is 0% for nearest/knn and 0.4–3.5% for proto (likely a one-sided spike, not a shelf).

Changing the *sum* in \(h_\mathrm{anom}\) does not add a duration / DC term. Six real shift windows in \(R_\mathrm{anom}\) are not enough for proto to learn a shelf if \(e(\cdot)\) does not represent that offset (and \(\varphi\) z-scores it away). A later kind-conditional proto (only shift refs) or an explicit \(\lvert\mu_L-\mu_R\rvert\) constraint is a different \(f\).

Locked time_* + combined (1536) stays in [PAPER_CANDIDATES.md](PAPER_CANDIDATES.md) §2.

### 3.1 Drop \(h_\mathrm{rare}\) (\(\lambda_\mathrm{rare}=0\))

Coverage was already 0, so the rare term was not earning its keep on the protocol. Rerun: same three strategies, same 256 parents,

```
f  =  λ (h_nom − Q_q)²  +  λ_anom h_anom
```

Isolated `results/shell_hanom/norare/`. Does not overwrite the with-rare galleries.

| Encoder | \(h_\mathrm{anom}\) | Cov. | ARP anom | ARP rare | Div | Occ. | frac amp≥0.1 | frac \|Δμ\|≥0.01 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| time_recon | nearest | 0 | 0.37 | 0.56 | 4.24 | 0.05 | 0.07 | 0 |
| time_recon | knn | 0 | 0.35 | 0.52 | 5.49 | 0.05 | 0.17 | 0 |
| time_recon | proto | 0 | 0.33 | 0.42 | 5.50 | 0.02 | **0.49** | 0.04 |
| time_both | nearest | 0 | 0.40 | 0.64 | 3.85 | 0.06 | 0.03 | 0 |
| time_both | knn | 0 | 0.38 | 0.57 | 4.87 | 0.05 | 0.07 | 0 |
| time_both | proto | 0 | 0.40 | 0.56 | 5.43 | 0.01 | **0.48** | 0.01 |

Coverage@τ stays **0**. Dropping rare-repulsion does **not** unlock every kind.

What changed: more **spikes** (proto ~48% of windows have range ≥ 0.1 vs ~44% with rare; nearest/knn spike rate rose from ~0–1% to 3–17%). Div rose for nearest/knn. ARP rare fell (less glued to the rare cloud) but ARP anom did not rise. frac \|Δμ\|≥0.01 is still ~0–4% — and those hits are one-sided spikes, not shelves.

Eyes (`norare/plots/real_kinds_vs_time_both_hanom.png`, `real_kinds_vs_time_both_proto.png`):

| Kind | After dropping \(h_\mathrm{rare}\)? |
|---|---|
| short spike | yes (often an early burst, not a mid-window needle) |
| short subtle | carrier + noise; not a distinct kind |
| **level shift** | **no** — nearest-φ is still a flat-mean oscillation |
| medium event | no — not a sustained mid-window excursion |
| long campaign | no — proto may sit a bit higher; not a multi-hour campaign |

\(h_\mathrm{rare}\) was not the reason we miss shifts / campaigns / medium events. Those need a duration or DC term, or kind-conditional protos that \(e(\cdot)\) can actually represent.

---

## 4. Plots

Same pairing as `real_kinds_vs_time_both_combined.png`: red = real kind examples, teal/other = **nearest in \(\varphi\)** from that strategy’s 256-window gallery. Columns are independent; no shared parent.

| File | What |
|---|---|
| `plots/real_kinds_vs_{time_recon,time_both}_{nearest,knn,proto}.png` | Per encoder × strategy, **with** \(h_\mathrm{rare}\) |
| `plots/real_kinds_vs_{time_recon,time_both}_hanom.png` | Kinds × soft / nearest / knn / proto (with rare) |
| `norare/plots/` | Same filenames, \(\lambda_\mathrm{rare}=0\) |
| `gen_{encoder}_{strategy}.png` | 8 diverse generated windows |

Level-shift column: did any strategy emit a shelf, or is nearest-φ still a carrier wave?

---

## 5. Do not

- Overwrite `results/shell_s4` or re-run `anogen s4` / `s6`.
- Cite these 256 / fold-0 numbers as a new win-rule table.
- Change \(h_\mathrm{nom}\) to min-distance (the band would stop meaning \(Q_q\)).
- Expect proto to invent kinds that are missing from \(R_\mathrm{anom}\) (6 real level-shift windows). Kind-conditional proto: [KIND_PROTO.md](KIND_PROTO.md). Mixed gallery (soft’s quiet kinds + proto shelves): [KIND_MIX.md](KIND_MIX.md). Other shelf ideas: [LEVEL_SHIFT.md](LEVEL_SHIFT.md).
