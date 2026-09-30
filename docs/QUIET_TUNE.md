# Quiet-kind λ_anom: subsequence texture, not Locality

Isolated `results/shell_quiettune/`. Does **not** overwrite
`results/shell_s4`, `results/shell_kindmix_htune/`, or any 1536 freeze.
Plots: [`docs/quiettune/`](quiettune/INDEX.txt).

```bash
.venv/bin/anogen -c configs/shell_mission1.yaml quiettune
```

**Not promoted.** The 1536 / 3-fold recipe stays `hybrid_needles` with
\(\lambda_\mathrm{anom}=0\) on the two subsequence slices.

---

## 0. What this was for

Under `hybrid` and `hybrid_needles` the local- and global-subsequence
slices get \(\lambda_\mathrm{anom}=0\), so they run identical band energy
and cannot be aimed. Those two kinds are 648 of 672 dev anomaly windows.

Band median total variation sits at about 0.0010–0.0013. Full stratified
proto sits at 0.004–0.010. Real crops sit between at about 0.0029. The
existing htune pick in [`KIND_MIX_HTUNE.md`](KIND_MIX_HTUNE.md) scores
roughness **one-sided** (`max(0, tv/target − 1)`), so a band slice at
`tv/target ≈ 0.4` looks perfect. Too-smooth is invisible. This sweep
uses a **two-sided** match, `|tv/target − 1|`, plus a second shape
statistic (high/low rFFT energy from `feature_pack_v1`).

Matching texture is **not** hitting the ESA kind. `Locality` is a
property of the window relative to its surroundings. A window-only
generator scored by a per-window \(\varphi\) cannot express it.
Coverage@τ on these slices staying at 0 is expected even with a perfect
roughness match. The claim on the table was “the quiet slices are no
longer visibly wrong,” not “we can aim at Local versus Global.”

---

## 1. Setup

Base recipe: locked `hybrid_needles` parent50 (\(\lambda=0.3\),
\(\lambda_\mathrm{anom}=1\) on level shift and Point/Global,
\(\lambda_\mathrm{rare}=0\), \(\nu=0.2\), 50 DDIM, unit \(\nabla\)).
Per-kind \(\lambda_\mathrm{anom}\) on the two quiet slices only; the
`frozenset` path for `hybrid_needles` is unchanged.

| tag | \(\lambda_\mathrm{anom}\) local | \(\lambda_\mathrm{anom}\) global |
|---|---:|---:|
| `q00` | 0 | 0 | current recipe (control) |
| `q05` | 0.05 | 0.05 | |
| `q10` | 0.10 | 0.10 | |
| `q15` | 0.15 | 0.15 | |
| `q20` | 0.20 | 0.20 | |

Both encoders, 128 S3 donors, fold 0. Real-crop targets (median):

| Kind | TV | HF/LF spec |
|---|---:|---:|
| local subsequence | 0.00291 | 0.120 |
| global subsequence | 0.00293 | 0.123 |

**Pick rule** (declared before looking at ARP or Coverage): among
settings that keep

- shift-slice \(P(|\mu_L-\mu_R|\ge 0.01)\ge 0.8\), and
- Point/Global-slice \(P(\mathrm{amp}\ge 0.1)\ge 0.5\),

choose the lowest two-sided TV error on the two subsequence slices.
Tie-break: two-sided HF/LF spectrum error, then occupancy, then ARP.

---

## 2. Numbers (fold-0, 128 donors)

Every cell kept the structured-kind gates (shift \(P\ge 0.875\),
Point \(P\ge 0.90\)). Occupancy 0.008–0.07. Coverage@τ ≈ 0.

| Encoder | tag | TV local / global | spec local / global | Occ. | ARP anom |
|---|---|---:|---:|---:|---:|
| time_recon | q00 | 0.00128 / 0.00122 | 0.022 / 0.015 | 0.023 | 0.352 |
| time_recon | q05 | 0.00131 / 0.00127 | 0.054 / 0.041 | 0.023 | 0.485 |
| time_recon | q10 | 0.00133 / 0.00122 | 0.104 / 0.161 | 0.016 | 0.474 |
| time_recon | **q15** (auto) | 0.00140 / 0.00121 | 0.123 / 0.228 | 0.055 | 0.442 |
| time_recon | q20 | 0.00142 / 0.00115 | 0.174 / 0.189 | 0.031 | 0.456 |
| time_both | **q00** (auto) | 0.00116 / 0.00127 | 0.019 / 0.027 | 0.008 | 0.343 |
| time_both | q05 | 0.00125 / 0.00115 | 0.060 / 0.049 | 0.047 | 0.495 |
| time_both | q10 | 0.00120 / 0.00114 | 0.090 / 0.133 | 0.055 | 0.400 |
| time_both | q15 | 0.00120 / 0.00116 | 0.120 / 0.199 | 0.062 | 0.497 |
| time_both | q20 | 0.00125 / 0.00116 | 0.110 / 0.169 | 0.070 | 0.453 |

Automatic pick: `time_recon` **q15** (\(\mathrm{err}_{TV}=1.10\)),
`time_both` **q00** (\(\mathrm{err}_{TV}=1.17\)). All
\(\mathrm{err}_{TV}\) values sit near 1.1–1.2 because generated TV
never left ~0.0012 against a 0.0029 target. The HF/LF ratio *does*
move (q10 is the closest spectrum on time_recon), but the extra energy
is high-frequency hash, not the hitch that makes the real crops rough.

CSV: [`docs/quiettune/metrics.csv`](quiettune/metrics.csv).

---

## 3. Eyes

Use the **aimed** strips (best-stat inside the allocated kind slice).
Nearest-in-\(\varphi\) will match a quiet real crop to a band window
from another slice — not proof of aiming.

- [`quiet_time_recon_aimed.png`](quiettune/quiet_time_recon_aimed.png) /
  [`quiet_time_both_aimed.png`](quiettune/quiet_time_both_aimed.png)
  — all four kinds, one aimed window per \(\lambda\).
- Local subsequence, four examples:
  [`quiet_time_recon_esa_local_subseq_aimed4.png`](quiettune/quiet_time_recon_esa_local_subseq_aimed4.png),
  [`quiet_time_both_esa_local_subseq_aimed4.png`](quiettune/quiet_time_both_esa_local_subseq_aimed4.png).
- Global subsequence, four examples:
  [`quiet_time_recon_esa_global_subseq_aimed4.png`](quiettune/quiet_time_recon_esa_global_subseq_aimed4.png),
  [`quiet_time_both_esa_global_subseq_aimed4.png`](quiettune/quiet_time_both_esa_global_subseq_aimed4.png).
- TV vs \(\lambda\) with the real target as a dashed line:
  [`quiet_time_recon_tv.png`](quiettune/quiet_time_recon_tv.png),
  [`quiet_time_both_tv.png`](quiettune/quiet_time_both_tv.png).
- Auto-pick vs band control:
  [`quiet_time_recon_recommend_aimed.png`](quiettune/quiet_time_recon_recommend_aimed.png),
  [`quiet_time_both_recommend_aimed.png`](quiettune/quiet_time_both_recommend_aimed.png).

What the eyes show: **q00 is the only row that keeps the orbital
period.** \(\lambda_\mathrm{anom}\ge 0.05\) flattens the carrier and
adds hash; on global subsequence it also injects edge spikes. Needles
and shelves stay intact (the structured-kind gates held). The goldilocks
texture between band (0.0012) and stratified proto (0.004–0.010) is
not in \(\{0.05,0.10,0.15,0.20\}\). Small proto on 6 local / 256 global
refs, with unit-\(\nabla\) competing against \(\lambda=0.3\) band, hashes
before it roughens.

---

## 4. Decision

Do **not** add a quiet-\(\lambda\) recipe to `kindmixscore.py`. Do **not**
score 1536 / 3-fold. Keep C1/C2 as `hybrid_needles` with band on the
quiet slices.

The automatic two-sided-TV pick would have promoted `time_recon` q15.
That is a ~0.0001 TV crawl toward 0.0029 and a worse picture. Eyes over
the rule when they disagree; the rule was a texture proxy, not the
claim.

Open limitation: Local vs Global subsequence are still the same band
process. A follow-up that could express Locality is a two-reference-set
term (global nominal cloud plus a donor-neighbourhood cloud), not a
larger \(\lambda_\mathrm{anom}\) on the same proto.

CSV and INDEX: [`docs/quiettune/`](quiettune/).
