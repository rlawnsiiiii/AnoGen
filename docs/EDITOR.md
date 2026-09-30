# Editor: donor carrier + slow hitch from DDIM

Isolated `results/shell_editor/`. Does **not** overwrite `results/shell_s3`,
`results/shell_s4`, or the `hybrid_needles` freeze. No new DDIM: the editor
is a post-process of the locked C1/C2 galleries. Plots:
[`docs/editor/`](editor/INDEX.txt).

```bash
.venv/bin/anogen -c configs/shell_mission1.yaml editor
```

**Not promoted** to the paper pack. ARP is comparable to other 1536 / frozen-τ
rows. EDI **e** is a **new** union — do not mix with unmarked / \* / ¶ / † / ‡.

---

## 0. What this is

Quiet kinds under `hybrid_needles` are band edits of a nominal **donor**.
DDIM rebuilds all 512 bins, so the analog oscillation comes back hashed.
The editor copies the donor’s fast residual and only keeps a slow hitch
from the generated window:

```
x  =  parent  +  slow(gen) − slow(parent)
```

`slow` = first `n_keep=3` rFFT bins (DC + periods ≳ 2.1 h). Needles are
high-frequency, so Point/Global uses GenIAS deviation-patch
(`τ=0.2`) instead: keep `gen` only where `(parent − gen)² > τ · amp`.

Code: [`src/anogen/shell/editor.py`](../src/anogen/shell/editor.py).

---

## 1. Scores (1536 / frozen τ / 3-fold)

ARP is the same formula as every other table. EDI **e** is Shannon entropy
over 16 k-means bins of {shell, unguided, GenIAS, post-hoc, C1/C2
`hybrid_needles`, C1/C2 editor} fold-0 galleries.

| Method | ARP anom | EDI **e** | Div | Cov@τ anom |
|---|---:|---:|---:|---:|
| **C1 editor** (`time_both`) | **0.630** | 1.83 | 6.70 | 0.169 |
| **C2 editor** (`time_recon`) | **0.638** | 1.83 | 8.04 | 0.163 |
| C1 `hybrid_needles` (raw) | 0.584 | **2.38** | 7.52 | 0.039 |
| C2 `hybrid_needles` (raw) | 0.576 | **2.39** | 7.94 | 0.023 |
| Unguided ν=1 (locked S4 ARP) | 0.568 | 1.36 | 2.34 | 0.109 |
| Shell ZS (locked S4 ARP) | 0.393 | 1.28 | 2.01 | 0.081 |
| GenIAS ψ=2 | 0.317 | 0.71 | 1.09 | 0.195 |
| Post-hoc | 0.319 | 0.80 | 1.21 | 0.267 |

C1 gap 0.014 → **0.005** (rares also move closer). Same structural ARP
boost as CutAddPaste / post-hoc: the gallery still contains real
telemetry texture. Coverage@τ rises because φ sees the donor carrier,
not because local vs global subsequence were aimed.

---

## 2. Eyes

Start with same-parent strips (grey donor / teal DDIM / orange editor):

- [`time_both_same_parent_esa_local_subseq.png`](editor/time_both_same_parent_esa_local_subseq.png)
- [`time_both_same_parent_esa_global_subseq.png`](editor/time_both_same_parent_esa_global_subseq.png)
- [`time_both_same_parent_level_shift.png`](editor/time_both_same_parent_level_shift.png)
- [`time_both_same_parent_esa_point_global.png`](editor/time_both_same_parent_esa_point_global.png)

Random 16 per kind (no nearest, no amp rank):
[`time_both_editor_overview8.png`](editor/time_both_editor_overview8.png),
then `time_both_editor_random16_*.png`. C2 mirrors under `time_recon_*`.

| Kind | What the editor did |
|---|---|
| Local / global subsequence | DDIM glitches gone. Series ≈ the donor. Band never made a slow hitch, so residual copy has almost nothing to add. |
| Level shift | Sharp 0.95→0.78 shelf becomes a **smooth U-bowl**. `n_keep=3` cannot hold a step; that energy is not in the first three FFT bins. Carrier is cleaner. |
| Point/Global | Deviation-patch keeps the needle, restores the donor around it. Failed-needle hash windows stay messy if they exceed τ. |

---

## 3. Read

The hash on quiet slices was the rebuilt carrier. Copying it back fixes
eyes and ARP/Coverage. It does **not** invent a subsequence anomaly —
those slices become edited nominals. Level-shift proto still needs a
better slow/fast split (or no FFT brick-wall) if the shelf must stay
sharp. EDI **e** drops vs raw `hybrid_needles` because the gallery
collapses toward the donor cloud.
