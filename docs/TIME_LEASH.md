# Time leash: donor in sample space, not only in \(h_\mathrm{nom}\)

Isolated `results/shell_timeleash/`. Does **not** overwrite `results/shell_s3`,
`results/shell_s4`, or the `hybrid_needles` freeze. Plots:
[`docs/timeleash/`](timeleash/INDEX.txt).

```bash
.venv/bin/anogen -c configs/shell_mission1.yaml timeleash
```

**Not promoted.** ARP is comparable to other 1536 / frozen-τ rows. EDI **t**
is a **new** union — do not mix with unmarked / \* / ¶ / † / ‡ / **e**.

Base recipe is locked C1 `hybrid_needles` parent50. Added term, in
**scaled** window units, subtracted from \(\hat x_0\) each DDIM step
(not unit-normalized):

```
l2:   total  +=  μ (x̂_0 − x_par)
gm:   total  +=  μ (x̂_0 − x_par) / (1 + ((x̂_0 − x_par)/δ)²)
```

| Tag | μ | δ | Intent |
|---|---:|---:|---|
| `l2` | 0.03 | — | Literal suggestion 2 |
| `gm` | 0.15 | 0.03 | Saturate large residuals so a shelf/needle can stay |

Code: `lam_parent` / `parent_leash` in
[`src/anogen/shell/steer.py`](../src/anogen/shell/steer.py) `guided_ddim`.

---

## 1. Scores (C1 `time_both`, 1536 / 3-fold)

| Method | ARP anom | EDI **t** | Div | Cov@τ anom |
|---|---:|---:|---:|---:|
| C1 `hybrid_needles` (raw) | **0.584** | **2.55** | 7.52 | 0.039 |
| **gm** time leash | 0.570 | 2.52 | 7.77 | 0.026 |
| **l2** time leash | 0.544 | 2.41 | 6.86 | 0.041 |
| Unguided (S4 ARP) | 0.568 | 1.57 | 2.34 | 0.109 |
| Post-hoc | 0.319 | 0.93 | 1.21 | 0.267 |

Raw C1 ARP is the locked hybrid table (footnote \*). EDI **t** re-clusters
that fold-0 gallery with the two leash galleries; do not quote 2.55 as \*.

Neither leash beat C1 ARP. Coverage stayed ~0.03–0.04.

---

## 2. Eyes

Start with same-parent (grey donor / teal raw C1 / orange leash):

- [`time_both_gm_same_parent_level_shift.png`](timeleash/time_both_gm_same_parent_level_shift.png)
- [`time_both_gm_same_parent_esa_point_global.png`](timeleash/time_both_gm_same_parent_esa_point_global.png)
- [`time_both_gm_same_parent_esa_local_subseq.png`](timeleash/time_both_gm_same_parent_esa_local_subseq.png)

Random 8: [`time_both_gm_overview8.png`](timeleash/time_both_gm_overview8.png),
[`time_both_l2_overview8.png`](timeleash/time_both_l2_overview8.png).

| Kind | vs editor | vs raw C1 |
|---|---|---|
| Level shift | **Keeps the sharp shelf** (editor turned it into a U-bowl) | Same 0.95→0.78 drop as proto |
| Point/Global | Needles still sit on hash; some spikes weaken | Not a fix for collapse |
| Local / global | Still the DDIM carrier, not the donor | Teal ≈ orange |

---

## 3. Read

The editor failed on shelves because a 3-bin FFT residual-copy **throws
away the step**. A time leash **during** DDIM does not: proto’s unit
\(\nabla h_\mathrm{anom}\) is O(1) per step, while \(\mu(x-x_\mathrm{par})\)
on a 0.2 shelf is ~0.006 (`l2`) or saturates to ~0 (`gm`). The shelf wins.

The same scale comparison is why **points and quiet slices barely move**.
Unit-normalized band/proto kicks are O(1). Hash of size 0.02 gives a
parent pull \(\mu r \sim 10^{-3}\) (`l2`) or ~0.002 (`gm`). The carrier is
still rebuilt by ε. To make the donor leash compete you would have to
drop unit-\(\nabla\) on band/proto, or unit-normalize the parent term
too (which undoes `gm` saturation and risks the editor’s glue-to-parent
failure).

Not a C1 replacement. Useful negative: **in-sampling L2 at this μ does
not replace residual-copy**, and **GM is how to keep shelves if you
do leash in time**.
