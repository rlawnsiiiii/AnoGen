# Chanmix: every targeted kind sees every channel

Isolated `results/shell_chanmix/`. Does **not** overwrite `results/shell_s3`,
`results/shell_s4`, or the `hybrid_needles` freeze. Plots:
[`docs/chanmix_by_kind/`](chanmix_by_kind/INDEX.txt), against the frozen
gallery in [`docs/c1_by_kind/`](c1_by_kind/INDEX.txt).

```bash
.venv/bin/python scripts/run_chanmix.py
```

**Not a new model.** Same `hybrid_needles` recipe, same parent50
hyper-parameters, same encoders. The only change is which donor is assigned
which target kind, so this is a **less confounded measurement of C1**, not a
better generator. ARP is comparable to other 1536 / frozen-τ rows. EDI **c**
is a **new** union — do not mix with unmarked / \* / ¶ / † / ‡ / **e** /
**t** / **h**.

---

## 0. The confound

S3 lays the 1536 donors out in per-channel blocks of 256, in channel order.
`_kind_alloc_labels` then split that array into **contiguous** slices, one per
kind. The two line up exactly, so each targeted kind only ever saw one or two
channels:

| targeted kind | contiguous | chanmix |
|---|---|---|
| Point/Global | 41 (256), 42 (128) | 64 on each of 41–46 |
| local subsequence | 42 (128), 43 (256) | 64 on each of 41–46 |
| level shift | 44 (256), 45 (128) | 64 on each of 41–46 |
| global subsequence | 45 (128), 46 (256) | 64 on each of 41–46 |

Channels 41 and 46 never shared a kind. Every per-kind number in this repo was
therefore a kind effect **and** a channel effect, and the abstract's
type-targeting claim could not be separated from channel identity.

`channel_stratified_alloc` splits within each channel instead. Per-kind totals
are unchanged at 384, so nothing else about the protocol moves. Code:
[`src/anogen/phases/kindmix.py`](../src/anogen/phases/kindmix.py)
`channel_stratified_alloc`; `_stratified` takes the labels through `alloc=` and
scatters results back into donor order, so `x0[i]` stays the parent of `x[i]`
under either allocation. Config: `shell.kindmixscore.channel_stratified`,
default **false** so the freeze is unchanged.

---

## 1. Scores (1536 / frozen τ / 3-fold)

| Method | ARP anom | EDI **c** | Div | Cov@τ anom |
|---|---:|---:|---:|---:|
| C1 `hybrid_needles` contiguous (frozen) | **0.584** | 2.60 \* | 7.52 | **0.039** |
| **C1 chanmix** (`time_both`) | 0.565 | 2.52 | 7.33 | 0.022 |
| **C2 chanmix** (`time_recon`) | 0.557 | 2.50 | 7.88 | 0.010 |

ARP and Coverage both fall. That is **not** the model degrading — the
contiguous split happened to hand each kind its most favourable channels, and
removing that gives the inflation back. These are the honest numbers.

`time_both` still beats `time_recon` on ARP, EDI and Coverage under the
unconfounded allocation, so that ranking was not an artifact of the confound.

---

## 2. Per-kind diagnostics (fold 0)

Channel-span units. `env frac` = fraction of windows leaving the nominal
training envelope, `peak@start` = fraction whose dominant peak is in the first
tenth of the span. Real anomalies: env frac 0.039, CUSUM −0.005, diff p99.9
0.169, peak@start 0.114.

| kind | | env frac | env max | CUSUM | diff p99.9 | peak@start |
|---|---|---:|---:|---:|---:|---:|
| Point/Global | contiguous | 0.734 | 4.09 | 0.05 | 0.774 | 0.193 |
| | chanmix | 0.625 | 4.14 | 0.04 | 0.686 | 0.315 |
| local subseq | contiguous | 0.943 | 2.58 | −0.12 | 0.251 | 0.357 |
| | chanmix | **0.341** | 2.95 | −0.03 | **0.164** | 0.326 |
| level shift | contiguous | 0.987 | 3.54 | 4.44 | 0.889 | 0.870 |
| | chanmix | 0.987 | 4.04 | **4.50** | 0.869 | **0.917** |
| global subseq | contiguous | 0.023 | 0.49 | −0.09 | 0.065 | 0.383 |
| | chanmix | **0.346** | **3.60** | 0.00 | 0.182 | 0.323 |
| whole gallery | contiguous | 0.672 | 4.09 | 1.07 | 0.495 | 0.451 |
| | chanmix | 0.575 | 4.14 | 1.13 | 0.475 | 0.470 |

Reproduce: `scripts/compare_chanmix.py`.

---

## 3. Read

**The level-shift transient is intrinsic to the proto steering.** Every number
in that row is unchanged when the slice moves off channels 44–45 onto all six.
It is by far the worst slice — CUSUM 4.5 against a whole-gallery 1.13 and a
real −0.005, 92% of peaks in the first tenth of the window, 99% of windows
outside the envelope — and it is a quarter of the gallery. Sixteen random
draws across six channels produce one stereotyped shape:
[`chanmix_level_shift.png`](chanmix_by_kind/chanmix_level_shift.png). Whatever
fixes this has to be in the steering, not in the data allocation.

**The clean global-subsequence slice was a channel artifact.** Off channels
45–46 it goes from 2% to 35% envelope exits and from 0.49 to 3.60 max excess.
Local subsequence moves the other way — its violence was channel, not kind.
Those two effects roughly cancel, which is why the whole-gallery row barely
moves.

**Nothing here fixes C1.** The confound was hiding *where* the damage was, not
reducing it. The value of this run is that it makes the per-kind table
interpretable and points at the level-shift prototype as the single
highest-value target.

---

## 4. Status

Partially scored. `results/shell_chanmix/summary.json` has ARP, EDI **c**,
Coverage@τ and Div; the per-kind diagnostics above come from
`scripts/compare_chanmix.py`. The gallery has **not** been through the realism
audit, so `docs/COMPARISON.md`, `docs/REALISM.md` §17.1–17.7 and
`results/shell_realism/` all still describe the contiguous gallery. Pointing
the audit's `c1` entry at `results/shell_chanmix/` is the cheapest way to close
that gap — the realism phase reads saved galleries and needs no GPU.

Any claim about **type-targeted** generation should cite this run, not the
frozen one. Whole-gallery realism and diversity claims are barely affected
either way.

---

CLI: `.venv/bin/python scripts/run_chanmix.py`
