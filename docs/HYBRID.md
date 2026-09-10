# Hybrid slices: band on quiet kinds, proto on structured kinds

How the mixed gallery decides **per window** whether \(h_\mathrm{anom}\) is on.
Results and the λ pick: [KIND_MIX_HTUNE.md](KIND_MIX_HTUNE.md).
Shortlist plots: [FINDINGS.md](FINDINGS.md). Kinds:
[ESA_LABELS.md](ESA_LABELS.md). Proto math: [KIND_PROTO.md](KIND_PROTO.md).
Sampling: [STEERING_MATH.md](STEERING_MATH.md) §7.

Code: `_stratified(..., proto_kinds=...)` in
[`src/anogen/phases/kindmix.py`](../src/anogen/phases/kindmix.py).
This is **not** LEVEL_SHIFT idea C (hand-added step on a diffusion carrier).

---

## 0. Two forces

Parent50 otherwise (\(\nu=0.2\), 50 DDIM, unit \(\nabla\), \(\lambda_\mathrm{rare}=0\)):

```
f  =  λ (h_nom − Q_q)²  +  λ_anom h_anom
```

| Term | What it does |
|---|---|
| \(\lambda (h_\mathrm{nom}-Q_q)^2\) | **Band / shell leash.** Stay near the parent’s rarity level. No kind. |
| \(\lambda_\mathrm{anom} h_\mathrm{anom}\) | **Kind pull.** Proto: one ref from a chosen \(R_k\), held for the whole trajectory. |

Band is the first term only (`lam_anom=0`, no `ref_anom`). Proto is both, with

```
R_k     =  { e(a) : a is a train-fold window of kind k }
i(b)   ~  Uniform(R_k)
h_anom  =  ‖ e(x̂_0) − a_{i(b)} ‖²
```

Proto is the only term that can **invent** a structured departure from the
parent (a shelf, a needle). It is also the term that injects high-frequency
jitter when \(R_k\) is a quiet subsequence crop.

---

## 1. Allocation, then energy

Same split as stratified. Present ESA kinds on 41–46 pretest (Point/Local
empty): Point/Global, local subsequence, level shift, global subsequence.
Uniform counts (`_split_counts`): 128 donors → 32 + 32 + 32 + 32.

```
for each allocated kind k:
    if k in proto_kinds:
        guided DDIM with proto on R_k          # λ and λ_anom both on
    else:
        guided DDIM with band                  # λ_anom forced to 0
```

`proto_kinds=None` is full stratified (proto on every slice).

| Recipe | `proto_kinds` | Band slices |
|---|---|---|
| `stratified` | all present kinds | — |
| `hybrid` | `{level shift}` | Point/Global + local + global subsequence |
| `hybrid_needles` | `{level shift, Point/Global}` | local + global subsequence |

Constants: `HYBRID_PROTO_KINDS`, `HYBRID_NEEDLES_PROTO_KINDS`.

---

## 2. Can you aim at a type?

**Aim** = at sample time, \(h_\mathrm{anom}\) uses that kind’s \(R_k\). The
slice label alone does not aim. Band windows in a “local subsequence” slot
and band windows in a “global subsequence” slot run the **same** \(f\).

| Kind | `hybrid` (band on points) | `hybrid_needles` (recommended) | Stratified proto |
|---|---|---|---|
| Level shift | **aimed** (proto \(R_\mathrm{shift}\)) | **aimed** | aimed |
| Point / Global | not aimed (band) | **aimed** (proto \(R_\mathrm{point}\)) | aimed |
| Local subsequence | not aimed (band) | not aimed (band) | aimed, but fails (jitter) |
| Global subsequence | not aimed (band) | not aimed (band) | aimed, but fails (jitter) |
| Point / Local | no pretest windows | no pretest windows | no pretest windows |

So:

- **Your hybrid** (proto only on shift): the only type you can target is
  **level shift**. Point and both subsequence slots are unlabeled shell
  edits of the parent.
- **`hybrid_needles`**: you can target **level shift and Point/Global**.
  You still cannot target local vs global subsequence. Those two slices
  are the same band process, just different row labels in the plot.
- **Stratified** *tries* to target every kind. Eyes and
  [KIND_MIX_HTUNE.md](KIND_MIX_HTUNE.md): proto on subsequence \(R_k\)
  does not emit that kind. It hashes the carrier. Proto on shift and
  Point/Global does emit shelves and needles.

Kind-conditional generation that **works** on 41–46 is therefore only:

1. Level shift — proto on the six (fold-0 leaked) shift refs.
2. Point / Global — proto on the Point/Global refs (need `hybrid_needles`
   or stratified).

Local / global subsequence are **not** controllable types under this \(f\).
Band is used there because it is what those real crops look like (quiet
parent), not because the sampler was told “be a local subsequence.”

Fold-0 shift refs leak (`id_83`). Shift proto is a capability test, not a
few-shot cite. See [KIND_PROTO.md](KIND_PROTO.md) § Ref leak.

---

## 3. What a band “kind” slice is

Bookkeeping. The gallery still has a contiguous 32-window block labeled
`real ESA local subsequence` so aimed-slice plots and
`P(|\Delta\mu|\ge 0.01)` can be computed **on that block**. The windows
were not steered toward local-subsequence embeddings. Compare two band
blocks from the same run: they differ only by which S3 parents they
started on.

Nearest-in-\(\varphi\) can still pair a real local/global crop to a band
window, because both sit on the everyday carrier. That pairing is not
proof the generator aimed at the type.

---

## 4. Why not proto on every kind

| Aimed with proto | What you get |
|---|---|
| Level shift | Shelf. \(P(\|\mu_L-\mu_R\|\ge 0.01)\approx 0.94\)–1.00. |
| Point / Global | Needle. \(P(\mathrm{amp}\ge 0.1)\approx 0.81\)–1.00. |
| Local / global subsequence | Parent + jitter. Median \(\|\Delta x\|\) 2–3× the real crop. |

Quiet ESA kinds are “the parent, plus a small hitch.” \(h_\mathrm{anom}\)
toward those \(R_k\) (6 local refs; 256 campaign-like global refs) overfits
texture. Band never leaves the parent far enough to invent a shelf or a
reliable needle — that is why points stay on proto in `hybrid_needles`.

---

## 5. How to run

Not a default `kindmix` variant (that would rewrite
`results/shell_kindmix/`). Sweep + plots:

```bash
.venv/bin/anogen -c configs/shell_mission1.yaml kindmixhtune
```

Or pass `shell.kindmix.variants: [hybrid_needles]` and a dedicated
`kindmix_dir`. Do **not** overwrite `results/shell_s4` or the morphology
freeze in `results/shell_kindmix/`.
