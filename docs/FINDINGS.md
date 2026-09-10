# Findings — mixed-kind generation (shortlist)

What we actually keep after the ESA-kind, hybrid, and λ / λ_anom work.
Eyes are fold-0, 128 S3 parents. **Not** a new S4 freeze. Coverage@τ is 0
on these galleries — do not cite them as a win-rule table.

Protocol cite vs GenIAS stays [PAPER_CANDIDATES.md](PAPER_CANDIDATES.md)
§2 (`time_recon` + combined \(f\)). This note is the **mixed-kind**
recipe: one gallery that can hold a shelf, a needle, and quiet crops.

Method: [HYBRID.md](HYBRID.md). Full grid: [KIND_MIX_HTUNE.md](KIND_MIX_HTUNE.md).
Kinds: [ESA_LABELS.md](ESA_LABELS.md). Plots live here (not gitignored):
[`docs/findings/`](findings/INDEX.txt).

---

## 0. Shortlist

**Encoders:** `time_recon` (ZS-legal flatten) and `time_both` (MSE+SupCon).  
**Recipes:** `hybrid` and `hybrid_needles`.  
**Weights:** \(\lambda=0.3\), \(\lambda_\mathrm{anom}\in\{1.0,0.5,0.3\}\)
(`l03_a10`, `l03_a05`, `l03_a03`). Parent50 otherwise
(\(\nu=0.2\), 50 DDIM, unit \(\nabla\), \(\lambda_\mathrm{rare}=0\)).

| Recipe | Proto on | Band on (\(\lambda_\mathrm{anom}=0\)) |
|---|---|---|
| `hybrid` | level shift | Point/Global + local + global subsequence |
| `hybrid_needles` | level shift + Point/Global | local + global subsequence |

You can **aim** only at kinds that use proto. Band slices are the same
shell edit with different parent windows. Local vs global subsequence
are not two generators. [HYBRID.md](HYBRID.md) §2.

**If you want one default:** `hybrid_needles` + `l03_a10`
(\(\lambda=0.3\), \(\lambda_\mathrm{anom}=1.0\)). That is the only cell
that keeps shelves **and** needles on both encoders. `hybrid` at the
same weights is quieter on points (no needle). `l03_a03` loses the
shelf.

---

## 1. Aimed overview (all four kinds)

One best-stat window per kind slice. Red = real ESA-ADB.

| Encoder | All six galleries | `hybrid` only | `hybrid_needles` only |
|---|---|---|---|
| `time_recon` | [shortlist](findings/time_recon_shortlist_aimed.png) | [hybrid λ=0.3](findings/time_recon_hybrid_l03_aimed.png) | [needles λ=0.3](findings/time_recon_hybrid_needles_l03_aimed.png) |
| `time_both` | [shortlist](findings/time_both_shortlist_aimed.png) | [hybrid λ=0.3](findings/time_both_hybrid_l03_aimed.png) | [needles λ=0.3](findings/time_both_hybrid_needles_l03_aimed.png) |

Open those first. Then the per-kind four-example strips below.

---

## 2. Per kind

Point/Local has no pretest windows. Four present kinds.

### Point / Global

Aimed only under `hybrid_needles`. `hybrid` uses band here — needles
collapse.

| Encoder | Aimed (4) | Nearest in φ (4) |
|---|---|---|
| `time_recon` | [aimed](findings/time_recon_esa_point_global_aimed4.png) | [nearest](findings/time_recon_esa_point_global_nearest4.png) |
| `time_both` | [aimed](findings/time_both_esa_point_global_aimed4.png) | [nearest](findings/time_both_esa_point_global_nearest4.png) |

| Gallery | `time_recon` \(P(\mathrm{amp}\ge 0.1)\) | `time_both` \(P(\mathrm{amp}\ge 0.1)\) |
|---|---:|---:|
| hybrid `l03_a10` | 0.09 | 0.03 |
| hybrid `l03_a05` | 0.00 | 0.03 |
| hybrid `l03_a03` | 0.03 | 0.00 |
| needles `l03_a10` | **1.00** | **0.81** |
| needles `l03_a05` | 0.75 | 0.31 |
| needles `l03_a03` | 0.09 | 0.13 |

### Level shift

Aimed on both recipes (proto on \(R_\mathrm{shift}\)). Fold-0 leak
(all six real shelves). \(\lambda_\mathrm{anom}=0.3\) does not make a
shelf.

| Encoder | Aimed (4) | Nearest in φ (4) |
|---|---|---|
| `time_recon` | [aimed](findings/time_recon_level_shift_aimed4.png) | [nearest](findings/time_recon_level_shift_nearest4.png) |
| `time_both` | [aimed](findings/time_both_level_shift_aimed4.png) | [nearest](findings/time_both_level_shift_nearest4.png) |

| Gallery | `time_recon` \(P(\|\Delta\mu\|\ge 0.01)\) | `time_both` \(P(\|\Delta\mu\|\ge 0.01)\) |
|---|---:|---:|
| hybrid `l03_a10` | 1.00 | 0.88 |
| hybrid `l03_a05` | 1.00 | 1.00 |
| hybrid `l03_a03` | **0** | **0** |
| needles `l03_a10` | 1.00 | 0.97 |
| needles `l03_a05` | 1.00 | 1.00 |
| needles `l03_a03` | **0** | **0** |

### Local subsequence

Band on every shortlist gallery. Quiet carrier; not a targeted type.
Stratified proto (not in this shortlist) hashed these crops.

| Encoder | Aimed (4) | Nearest in φ (4) |
|---|---|---|
| `time_recon` | [aimed](findings/time_recon_esa_local_subseq_aimed4.png) | [nearest](findings/time_recon_esa_local_subseq_nearest4.png) |
| `time_both` | [aimed](findings/time_both_esa_local_subseq_aimed4.png) | [nearest](findings/time_both_esa_local_subseq_nearest4.png) |

### Global subsequence

Same band process as local subsequence. Majority ESA kind on 41–46
(~636 / 672 windows). We never emit a distinct “campaign” in a W=512
crop; success here means “looks like the real crop” (quiet parent).

| Encoder | Aimed (4) | Nearest in φ (4) |
|---|---|---|
| `time_recon` | [aimed](findings/time_recon_esa_global_subseq_aimed4.png) | [nearest](findings/time_recon_esa_global_subseq_nearest4.png) |
| `time_both` | [aimed](findings/time_both_esa_global_subseq_aimed4.png) | [nearest](findings/time_both_esa_global_subseq_nearest4.png) |

Median TV on these two band slices is ~0.0010–0.0013 (real crops
~0.0029). Full stratified at the same λ was 0.004–0.010.

---

## 3. Slice metrics (fold 0, 32 windows / kind)

From [`docs/kindmix_htune/metrics.csv`](kindmix_htune/metrics.csv).
Occ. = fraction with \(|h_\mathrm{nom}-Q_q|\le\delta\). ARP = anomaly
ARP on the mixed gallery.

| Encoder | Recipe | tag | \(\lambda_\mathrm{anom}\) | Occ. | ARP | Shift \(P\) | Point \(P(\mathrm{amp})\) | TV loc / glob |
|---|---|---|---:|---:|---:|---:|---:|---:|
| time_recon | hybrid | l03_a10 | 1.0 | 0.00 | 0.37 | 1.00 | 0.09 | 0.0013 / 0.0012 |
| time_recon | hybrid | l03_a05 | 0.5 | 0.01 | 0.32 | 1.00 | 0.00 | 0.0013 / 0.0012 |
| time_recon | hybrid | l03_a03 | 0.3 | 0.01 | 0.36 | 0 | 0.03 | 0.0013 / 0.0013 |
| time_recon | needles | l03_a10 | 1.0 | 0.01 | 0.37 | 1.00 | **1.00** | 0.0011 / 0.0012 |
| time_recon | needles | l03_a05 | 0.5 | 0.00 | 0.36 | 1.00 | 0.75 | 0.0012 / 0.0012 |
| time_recon | needles | l03_a03 | 0.3 | 0.02 | 0.45 | 0 | 0.09 | 0.0012 / 0.0013 |
| time_both | hybrid | l03_a10 | 1.0 | 0.02 | 0.34 | 0.88 | 0.03 | 0.0012 / 0.0012 |
| time_both | hybrid | l03_a05 | 0.5 | 0.02 | 0.39 | 1.00 | 0.03 | 0.0013 / 0.0012 |
| time_both | hybrid | l03_a03 | 0.3 | 0.03 | 0.30 | 0 | 0.00 | 0.0013 / 0.0012 |
| time_both | needles | l03_a10 | 1.0 | 0.02 | 0.35 | 0.97 | **0.81** | 0.0013 / 0.0010 |
| time_both | needles | l03_a05 | 0.5 | 0.00 | 0.33 | 1.00 | 0.31 | 0.0013 / 0.0012 |
| time_both | needles | l03_a03 | 0.3 | 0.03 | 0.41 | 0 | 0.13 | 0.0013 / 0.0012 |

---

## 4. What we learned (whole thread)

1. **Official kinds** are ESA `Length` × `Locality`. “Level shift” is
   our overlay on global subsequences (\(|\mu_L-\mu_R|\ge 0.01\)).
   [ESA_LABELS.md](ESA_LABELS.md).

2. **Band** never invents a shelf or a reliable needle
   (\(P(\mathrm{shift})=0\), Point \(P(\mathrm{amp})\approx 0.02\)).
   It is the right energy for quiet crops.

3. **Kind-conditional proto** invents shelves and needles when you
   point at those \(R_k\). On local / global subsequence it adds
   jitter, not the kind. [KIND_PROTO.md](KIND_PROTO.md),
   [KIND_MIX.md](KIND_MIX.md).

4. **Hybrid slices** apply proto only where it works. That is the
   noise fix. It is also why you cannot aim at subsequence types.
   [HYBRID.md](HYBRID.md).

5. **\(\lambda_\mathrm{anom}\)** at \(\lambda=0.3\): 1.0 keeps
   structured kinds; 0.5 keeps shelves, weakens needles on
   `time_both`; 0.3 drops the shelf. Raising \(\lambda\) to 1.0–1.5
   hashes band slices and kills the shelf. [KIND_MIX_HTUNE.md](KIND_MIX_HTUNE.md).

6. **Occupancy** stays ~0–0.04 on these 128-donor eyes whenever proto
   is on. Coverage@τ stays 0 (\(\varphi\) z-scores DC). Shift proto
   **leaks fold 0**.

7. **Locked protocol (1536 / 3-fold)** is now scored for hybrid too
   (`kindmixscorehybrid`). Best hybrid row: `time_both` +
   `hybrid_needles` (Cov 0.039, gap +0.014, ARP 0.584, occ. 0.47).
   Combined still wins occupancy (0.78). Coverage is still ≪ GenIAS
   (0.195) and comes from global subsequence, not aimed kinds.
   From \(x_T\sim\mathcal N(0,I)\) (`noisescore`): occupancy holds,
   Div ~14, hybrid Cov falls to 0.012–0.020.
   [PAPER_CANDIDATES.md](PAPER_CANDIDATES.md) §2. These 128-donor
   eyes are not that table.

---

## 5. How to read a plot

- **Aimed** = best-stat window(s) **inside the allocated kind slice**
  (32 donors). That is “what we asked for.”
- **Nearest** = nearest in locked \(\varphi\) from the **whole**
  128-gallery. Quiet real crops often match a band window from another
  slice. That pairing is not proof we aimed at the type.

Caches: `results/shell_kindmix_htune/{time_recon,time_both}_{hybrid,hybrid_needles}_l03_a*.npz`.
Reproduce: `anogen -c configs/shell_mission1.yaml kindmixhtune`.
