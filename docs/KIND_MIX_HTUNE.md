# λ / λ_anom sweep and recommended hyperparameters

Fold-0, 128 S3 parents, ESA kinds. Isolated
`results/shell_kindmix_htune/`. Does **not** overwrite `results/shell_s4`
or `results/shell_kindmix_esa/`. Not a new encscore / S4 freeze.
Coverage@τ is 0 on almost every cell — do not cite this table as a win
rule.

Method: [HYBRID.md](HYBRID.md). Shortlist (`l03_a10` / `a05` / `a03`,
both encoders, both hybrids) with per-kind links: [FINDINGS.md](FINDINGS.md).
Full-grid plots: [`docs/kindmix_htune/`](kindmix_htune/INDEX.txt). CSV:
[`docs/kindmix_htune/metrics.csv`](kindmix_htune/metrics.csv).

```bash
.venv/bin/anogen -c configs/shell_mission1.yaml kindmixhtune
```

---

## 0. Recommendation

**Recipe:** `hybrid_needles`  
**Encoder:** either `time_recon` or `time_both` (same pick)  
**Weights:** locked parent50

| Symbol | Value | Role |
|---|---:|---|
| \(\lambda\) | **0.3** | shell / band leash on \((h_\mathrm{nom}-Q_q)^2\) |
| \(\lambda_\mathrm{anom}\) | **1.0** | proto pull; **only** on level-shift and Point/Global slices |
| \(\lambda_\mathrm{rare}\) | 0 | off |
| \(\nu\) | 0.2 | start from a lightly noised nominal parent |
| DDIM steps | 50 | same as tune id 14 |
| \(\nabla\) | unit | `normalize_grad=True` |

Do **not** raise \(\lambda\) to quiet the series. Do **not** drop
\(\lambda_\mathrm{anom}\) below 0.5 if you still want shelves. Do **not**
put Point/Global on band if you still want needles.

What that recipe actually aims at: [HYBRID.md](HYBRID.md) §2. Short
version: level shift and Point/Global are proto-targeted; local and
global subsequence are the same band process (not two controllable types).

---

## 1. Setup

Three recipes × six \((\lambda,\lambda_\mathrm{anom})\) × two encoders.
Parent50 otherwise. Present kinds (Point/Local empty), 32 windows each.

| tag | \(\lambda\) | \(\lambda_\mathrm{anom}\) |
|---|---:|---:|
| `l03_a10` | 0.3 | 1.0 | locked |
| `l03_a05` | 0.3 | 0.5 | |
| `l03_a03` | 0.3 | 0.3 | |
| `l10_a10` | 1.0 | 1.0 | |
| `l10_a05` | 1.0 | 0.5 | |
| `l15_a05` | 1.5 | 0.5 | |

**Pick rule** (declared before looking at Coverage@τ): among settings with

- shift-slice \(P(|\mu_L-\mu_R|\ge 0.01)\ge 0.8\), and
- Point/Global-slice \(P(\mathrm{amp}\ge 0.1)\ge 0.5\),

choose the lowest excess roughness on local + global subsequence
(median \(|x_{t+1}-x_t|\) vs the real crops; real median TV \(\approx 0.0029\)).
Tie-break: occupancy, then ARP.

---

## 2. What the grid did

### 2.1 Recipe at locked weights (\(\lambda=0.3\), \(\lambda_\mathrm{anom}=1\))

| Recipe | Shift \(P\) | Point \(P(\mathrm{amp})\) | TV local / global | Occ. | ARP anom |
|---|---:|---:|---:|---:|---:|
| stratified | 0.94–1.00 | 0.97 | 0.004–0.010 | 0.01–0.03 | 0.30–0.33 |
| `hybrid` | 0.88–1.00 | **0.03–0.09** | 0.0012 | 0.00–0.02 | 0.34–0.37 |
| **`hybrid_needles`** | 0.97–1.00 | **0.81–1.00** | **0.0010–0.0013** | 0.01–0.02 | 0.35–0.37 |

`hybrid` (band on points) quiets subsequences and keeps shelves, but the
Point/Global slice is no longer a needle. `hybrid_needles` keeps both
structured kinds and the quiet band subsequences. Stratified keeps the
structured kinds and hashes the quiet ones (2–3× real TV).

Eyes: `htune_*_recipes_parent50_aimed.png`,
`htune_*_recommend_aimed.png`.

### 2.2 \(\lambda_\mathrm{anom}\) at \(\lambda=0.3\) (`hybrid_needles`)

| \(\lambda_\mathrm{anom}\) | Shift \(P\) | Point \(P(\mathrm{amp})\) | Note |
|---:|---:|---:|---|
| **1.0** | 0.97–1.00 | 0.81–1.00 | recommended |
| 0.5 | 1.00 | 0.31–0.75 | `time_both` misses the needle gate (0.31) |
| 0.3 | **0** | 0.09–0.13 | proto too weak; no shelf |

\(\lambda_\mathrm{anom}=0.3\) is not a quieter shelf. It is no shelf.

### 2.3 Stronger shell \(\lambda\in\{1.0,1.5\}\)

Does **not** smooth band slices. Median TV on local/global goes up
(0.0012 → 0.003–0.007). Aimed grids look hashed. Shift rate collapses
(\(P\le 0.28\)) — the leash overpowers proto. Occupancy stays ~0–0.04
everywhere. Unit \(\nabla\) times a larger \(\lambda\) over-corrects each
DDIM step.

Eyes: `htune_*_hybrid_needles_aimed.png`,
`htune_*_stratified_aimed.png`, `htune_*_metrics.png`.

---

## 3. Recommended vs runners-up

| Setting | Why not |
|---|---|
| `hybrid` \(\lambda=0.3\), \(\lambda_\mathrm{anom}=1\) | Quiet, but Point \(P(\mathrm{amp})=0.03\)–0.09 |
| `hybrid_needles` \(\lambda=0.3\), \(\lambda_\mathrm{anom}=0.5\) | Fine on `time_recon` (Point \(P=0.75\)); fails the needle gate on `time_both` (0.31) |
| stratified \(\lambda=0.3\), \(\lambda_\mathrm{anom}=0.5\) | Still shelves + some needles (`time_recon` Point \(P=0.63\)); subsequence TV still high |
| anything with \(\lambda\ge 1\) | No reliable shelf; noisier band |
| anything with \(\lambda_\mathrm{anom}=0.3\) | No shelf |

If you only care about quiet subsequences and shelves, and you do not
need needles, `hybrid` at the same weights is the quieter gallery. The
declared pick kept both structured kinds, so it is `hybrid_needles`.

---

## 4. What this sweep does not fix

- **Coverage@τ** stays 0 (\(\varphi\) z-scores DC; a real shelf and a
  generated shelf can sit far apart). One fluke 0.0037 on
  `time_both` stratified `l03_a03` — ignore it.
- **Occupancy** stays ~0–0.04. Proto still knocks samples out of
  \(|h_\mathrm{nom}-Q_q|\le\delta\). Raising \(\lambda\) does not repair
  that here.
- **Shift proto leaks fold 0** (all six real shelves). Capability, not
  a few-shot cite.
- **Local vs global subsequence** are not two generators. See
  [HYBRID.md](HYBRID.md) §2.

This is not a replacement for **time_recon + combined** on the locked
1536 / 3-fold table ([PAPER_CANDIDATES.md](PAPER_CANDIDATES.md) §2).
It is the sampling recipe when you want one mixed gallery that can
contain a shelf, a needle, and quiet crops without hashing the quiet
ones.
