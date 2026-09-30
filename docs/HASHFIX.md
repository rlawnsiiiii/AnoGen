# Hash knobs: last Tweedie, 20 DDIM, clip-in-box \(\nabla\)

Isolated `results/shell_hashfix/`. Does **not** overwrite `results/shell_s3`,
`results/shell_s4`, or the `hybrid_needles` freeze. Plots:
[`docs/hashfix/`](hashfix/INDEX.txt).

```bash
.venv/bin/anogen -c configs/shell_mission1.yaml hashfix
```

**Not promoted.** ARP is comparable to other 1536 / frozen-τ rows. EDI **h**
is a **new** union — do not mix with unmarked / \* / ¶ / † / ‡ / **e** / **t**.

Base recipe is locked C1 `hybrid_needles` parent50. Three sampler knobs
together (the hash-reduction pack):

| Knob | C1 parent50 | This run |
|---|---|---|
| Last DDIM step | \(x = \hat x_0 - \nabla f\) | \(\hat x_0\) only (`apply_final_grad=False`) |
| DDIM steps | 50 (repeats \(t\) below \(t^\star\approx 40\)) | 20 |
| \(\nabla\) | unit-normalized then clip | clip-in-box (`normalize_grad=False`) |

Code: `apply_final_grad` in
[`src/anogen/shell/steer.py`](../src/anogen/shell/steer.py) `guided_ddim`.
Default is **True** so locked S4 / C1 stay unchanged.

---

## 1. Scores (C1 `time_both`, 1536 / 3-fold)

| Method | ARP anom | EDI **h** | Div | Cov@τ anom | occ. |
|---|---:|---:|---:|---:|---:|
| C1 `hybrid_needles` (raw) | **0.584** | **2.60** | 7.52 | 0.039 | — |
| **hashfix** (1+2+3) | 0.459 | 1.89 | 2.22 | 0.062 | 0.012 |
| Unguided (S4 ARP) | 0.568 | 1.93 | 2.34 | 0.109 | — |
| Post-hoc | 0.319 | 1.42 | 1.21 | 0.267 | — |

Raw C1 ARP is the locked hybrid table (footnote \*). EDI **h** re-clusters
that fold-0 gallery with hashfix; do not quote 2.60 as \*. Occupancy is
\(\lvert h-Q_q\rvert\le\delta\) on the last Tweedie energy.

Hashfix ARP and Div sit next to **unguided**, not C1. Coverage rose a
little because the gallery is closer to the donor.

---

## 2. Eyes

Same-parent (grey donor / teal raw C1 / orange hashfix) is the
judgment:

- [`time_both_hashfix_same_parent_esa_point_global.png`](hashfix/time_both_hashfix_same_parent_esa_point_global.png)
- [`time_both_hashfix_same_parent_level_shift.png`](hashfix/time_both_hashfix_same_parent_level_shift.png)
- [`time_both_hashfix_same_parent_esa_local_subseq.png`](hashfix/time_both_hashfix_same_parent_esa_local_subseq.png)

Random 16 Point/Global:
[`time_both_hashfix_random16_esa_point_global.png`](hashfix/time_both_hashfix_random16_esa_point_global.png).
Overview: [`time_both_hashfix_overview8.png`](hashfix/time_both_hashfix_overview8.png).

| Kind | vs raw C1 |
|---|---|
| Point/Global | Needles **gone**. Orange ≈ parent. No hash, no spike. |
| Level shift | Shelf **gone**. Orange ≈ parent. |
| Local / global | C1 hash on quiet slices is gone; so is any proto leftover. Carrier ≈ donor. |

---

## 3. Read

The three knobs **did** kill the hash. They also **killed the steer**.

Unit \(\nabla\) was O(1) per window. Clip-in-box of the raw
\(\partial h/\partial x_t\) is orders of magnitude smaller, so band and
proto barely move \(\hat x_0\). Dropping the last-step subtract then
removes the one remaining stamp of that already-tiny force. Twenty DDIM
steps also drop the repeated-\(t\) kicks that parent50 used as extra GD.

Result: a slightly denoised copy of the S3 parent. Occupancy 0.012
confirms it is not on the shell.

Not a candidate. Next isolated test, if any, is **knob 1 alone**
(clean last Tweedie, keep unit \(\nabla\) and 50 steps) — that is the
piece that stamps hash onto the **output** without changing the
trajectory. Knobs 2–3 change how hard the constraint is enforced.

---

CLI: `.venv/bin/anogen -c configs/shell_mission1.yaml hashfix`
