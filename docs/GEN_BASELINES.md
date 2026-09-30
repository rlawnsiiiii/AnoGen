# Generator baselines: CutAddPaste and Lai taxonomy

Isolated `results/shell_genbase/`. Does **not** overwrite `results/shell_s3`
or `results/shell_s4`. Same frozen S4 \(\tau\approx 0.105\), \(\varphi=\)
`feature_pack_v1`, **1536** S3 donors, 3-fold OOF queries as every other
protocol row. EDI is a **new** 6-method union (footnote **†**). Do not mix
with unmarked / \* / ¶ / ‡.

```bash
.venv/bin/anogen -c configs/shell_mission1.yaml genbase
```

Plots: [`docs/genbase/`](genbase/INDEX.txt).

`posthoc_inject` is unchanged (locked S4 row).

---

## 0. Why these two

GenIAS (TCN-VAE, \(\psi=2\)) and the unnamed step/pulse/ramp/scale family
were the only cheap generators. CutAddPaste (Wang et al., KDD 2024) and
the Lai et al. 2021 NeurIPS D&B taxonomy are citable operators with the
same window API.

Both **edit real nominal windows**. That is the same structural advantage
as `posthoc_inject`. ARP (min distance in \(\varphi\) to real anomalies)
and Coverage@τ can look strong because the gallery still contains real
telemetry, not a VAE decode. GenIAS loses partly because its decode is
smooth. State that before a reviewer does.

---

## 1. Operators

Donors: `results/shell_s3/galleries.npz` key `cond` (1536), with
`channel_idx`. Patches are drawn from a **different window of the same
channel**. Cross-channel paste mixes DC levels and is not CutAddPaste as
published.

### 1.1 CutAddPaste (`cutaddpaste_inject`)

Cut a contiguous patch from another same-channel donor (width 8–35% of
\(W=512\)), add a linear trend (endpoint \(\pm 0.5\times\) host std) and
Gaussian jitter (\(\sigma=0.05\times\) std), paste at a random offset,
\(n_\mathrm{paste}=1\).

### 1.2 Lai taxonomy (`taxonomy_inject`)

Uniform over `{point, contextual, shapelet, seasonal, trend}`:

| Family | Edit |
|---|---|
| point | 1–3 timestamps, \(\pm(2\)–\(5)\times\) severity \(\times\) std |
| contextual | one timestamp vs a local neighbourhood, not the window std |
| shapelet | overwrite a segment with a reversed same-channel patch (DC-matched) |
| seasonal | add a sine of random period on a random span |
| trend | linear ramp on a random span |

This **supersedes** the unnamed step/pulse/ramp/scale family for new
comparisons. The locked S4 `posthoc` row still uses `posthoc_inject`.

This run: 289 / 299 / 325 / 326 / 297 windows per family.

---

## 2. Scores (1536 / frozen τ / 3-fold)

ARP is the same formula as every other table. EDI **†** is Shannon
entropy over 16 k-means bins of {shell, unguided, GenIAS, post-hoc,
CutAddPaste, taxonomy}. Div does not depend on the union.

| Method | ARP anom | EDI † | Div | Cov@τ anom |
|---|---:|---:|---:|---:|
| **CutAddPaste** | 0.323 | 1.60 | 1.12 | **0.285** |
| **Lai taxonomy** | **0.383** | **2.36** | **2.78** | 0.152 |
| Post-hoc (locked S4 operator) | 0.319 | 1.86 | 1.21 | 0.267 |
| GenIAS \(\psi=2\) | 0.317 | 1.52 | 1.09 | 0.195 |
| Unguided \(\nu=1\) (locked S4 ARP) | 0.568 | 2.28 | 2.34 | 0.109 |
| Shell ZS (locked S4 ARP) | 0.393 | 2.20 | 2.01 | 0.081 |
| C1 hybrid_needles (other table) | 0.584 | 2.60\* | 7.52 | 0.039 |

C1 ARP/EDI stay on the hybrid \* union. Only ARP/Div/Coverage are
comparable to this table.

---

## 3. How to read it

- **Coverage.** Same-channel CutAddPaste is the Coverage leader among
  cheap ops (0.285), slightly above post-hoc (0.267) and well above
  GenIAS (0.195). C1’s 0.039 is still ~7× below. Pasting real telemetry
  is what Coverage@τ rewards; steered diffusion is not winning that
  column.
- **ARP.** Taxonomy 0.383 edges GenIAS/post-hoc (~0.32) and CutAddPaste
  (0.323). None of them approach C1 (0.584) or unguided (0.568). The
  “ARP for free” warning is a structural caveat, not what happened:
  \(\varphi\) z-scores DC, so a pasted nominal chunk does not land next
  to a real anomaly just because it is real.
- **EDI † / Div.** Taxonomy is the diverse cheap gallery (five families).
  CutAddPaste is a close cousin of post-hoc (Div 1.12 vs 1.21).
- **Eyes.** Extreme-range strips over-represent in-channel DC pastes
  (same sensor, different operating point). The fair picture is
  `genbase_same_parent.png` / `genbase_overlay.png`: CutAddPaste swaps a
  same-channel texture block; taxonomy is a spike, a sine, or a ramp;
  GenIAS hugs the parent.

---

## 4. Plots

- [`gen_cutaddpaste.png`](genbase/gen_cutaddpaste.png) — 8 mixed extreme/random.
- [`gen_taxonomy.png`](genbase/gen_taxonomy.png)
- [`gen_taxonomy_families.png`](genbase/gen_taxonomy_families.png) — one row per family.
- [`genbase_vs_real.png`](genbase/genbase_vs_real.png)
- [`genbase_same_parent.png`](genbase/genbase_same_parent.png) — one parent per channel.
- [`genbase_overlay.png`](genbase/genbase_overlay.png)

Code: [`src/anogen/shell/baselines.py`](../src/anogen/shell/baselines.py),
[`src/anogen/phases/genbase.py`](../src/anogen/phases/genbase.py).
Artifacts: `results/shell_genbase/{cutaddpaste,taxonomy}.npz`.
