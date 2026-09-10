# ESA-ADB labels → repo kinds

How official Mission 1 fields become the five names in `KIND_ORDER`, and which
rule we invented ourselves.

Official source: Kotowski et al., *European Space Agency Dataset and Benchmark
for Real-World Anomaly Detection in Spacecraft Time Series*, DMLR 2026
([arXiv:2406.17826](https://arxiv.org/abs/2406.17826)). Files:
`labels.csv` and `anomaly_types.csv` under the configured ESA mission folder
(`data_root`, usually CausalDiscovery `data/ESA-Mission1/`).

Code: `src/anogen/shell/events.py` (persist / join) and
`src/anogen/shell/morphology.py` (`assign_anomaly_kinds`, `kinds_for_windows`).

---

## 1. What ESA-ADB ships

### `labels.csv` (one row per annotated fragment)

| Column | Meaning |
|---|---|
| `ID` | Event id (`id_1`, …). Several fragments can share one id. |
| `Channel` | Channel name |
| `StartTime`, `EndTime` | Closed time range of that fragment |

No waveform name. No “type” column.

### `anomaly_types.csv` (one row per event `ID`)

| Column | Official values (Mission 1) | What it is |
|---|---|---|
| `ID` | `id_1` … `id_200` | Join key |
| `Class` | `class_1` … `class_22` | Anonymized SOE **cause group** |
| `Subclass` | `subclass_1` … `subclass_6`, `unknown` | Variation inside that group |
| `Category` | `Anomaly`, `Rare Event`, `Communication Gap` | Operational role |
| `Dimensionality` | `Univariate`, `Multivariate` | How many channels the event hits |
| `Locality` | `Local`, `Global` | Contextual vs globally outlying |
| `Length` | `Point`, `Subsequence` | Short peak vs longer run |

The paper’s type taxonomy is those last three attributes (adjusted
Blázquez-García et al.): **uni/multi × local/global × point/subsequence**.
There is **no** official “level shift”, “spike”, or “campaign”.

`Class` / `Subclass` are numbered on purpose. The supplement maps them to SOE
causes (attitude disturbances, resets, power drops, …). Those English names are
**not** in the public CSV. The same `Class` can be `Anomaly` or `Rare Event`.

Shipped `Dimensionality` / `Locality` / `Length` are inferred from **all**
Mission 1 channels. ESA says re-run `infer_anomaly_types.py` for a channel
subset. This repo does **not**. We take the CSV as shipped.

The paper also defines **invalid segments**. Mission 1’s CSV has no such
`Category` (4 communication gaps, 118 anomalies, 78 rares).

---

## 2. `Category` — train / query roles (unchanged)

`build_event_table` still uses `Category` to decide who is a fault vs valid
unusual ops. That is **not** a kind label.

| ESA `Category` | Repo use |
|---|---|
| `Anomaly` | Fault. Out of S1/S2/S3. Query set for Coverage@τ. Few-shot refs for `h_anom`. |
| `Rare Event` | Valid unusual ops. In the train pool. Collision set. `h_rare`. |
| `Communication Gap` | Not used (none in the 41–46 pretest windows). |

Roles in sampling: [STEERING_MATH.md](STEERING_MATH.md) §0.

Window `kind` in `labeled_windows.csv` remains `anomaly` / `rare` / everyday —
that is still `Category`, not the ESA type taxonomy.

---

## 3. Map: official `Length` × `Locality` → `KIND_ORDER`

Live phases (`kindmix`, `kindproto`, `hanom`, `plot_tune`, `plot_from_noise`,
`s5` plots) label **Anomaly windows** with `kinds_for_windows` →
`assign_anomaly_kinds`.

Join: window `event_id` → `anomaly_types.csv` `ID`. Existing S0 CSVs do not
store `Length` / `Locality`; the join does not require re-running S0. New S0
runs also persist `esa_length`, `esa_locality`, `esa_class`, `esa_subclass`,
`esa_dimensionality` on the event / pair / labeled-window tables.

```
official Length      official Locality     repo name                      slug
---------------------------------------------------------------------------
Point                Local                 real ESA Point / Local         esa_point_local
Point                Global                real ESA Point / Global        esa_point_global
Subsequence          Local                 real ESA local subsequence     esa_local_subseq
Subsequence          Global + overlay      real level shift               level_shift
Subsequence          Global, leftover      real ESA global subsequence    esa_global_subseq
anything else                              real ESA unknown               (not in KIND_ORDER)
```

Point events keep the official Local / Global bit. They are **not** re-tested
with amplitude. A quiet official Point stays Point.

`Dimensionality` and `Class` / `Subclass` are stored but **not** used in the
kind name. Do not write `class_3` → spike: on 41–46 that class also owns the
shelf (`id_83`) and a campaign (`id_185`).

---

## 4. Invented rule: `real level shift`

ESA has no trend / shelf / change-point field. Lai et al. 2021 (NeurIPS)
separate **trend** anomalies (mean shift or slope change) from point /
contextual / shapelet / seasonal. That is the motivation for one overlay.

The overlay applies **only** to official **global subsequences**. Point events
and local subsequences are never relabeled, even if the crop’s half-means
differ.

On a window `x` of length `W` (default 512):

```
μ_L   =  mean( x[0 : W/2] )
μ_R   =  mean( x[W/2 : W] )
dmu   =  abs( μ_L − μ_R )

real level shift   iff   Length = Subsequence
                         and Locality = Global
                         and dmu ≥ shift_cut
```

Default `shift_cut = 0.01` (raw min-max units on the crop). Code:
`window_shape_stats` / `assign_anomaly_kinds`.

This is a **crop** statistic. The same ESA event can produce both labels:
some 512-bin crops sit across the step (`dmu` large), later crops sit on one
side of the shelf (`dmu` small) and stay `real ESA global subsequence`.
On 41–46 that happens for `id_83` (`class_3`, 22.5 h official Subsequence):
6 crops are level shift, 24 are global subsequence.

What this is **not**:

- not an ESA `Category` or `Class`
- not official `Length=Subsequence` by itself (almost every long event is that)
- not a change-point annotation in `labels.csv`
- not the legacy morphology rule “any non-short window with `dmu ≥ 0.01`”

---

## 5. Census — Mission 1 channels 41–46 (S0 pretest)

672 Anomaly windows / 22 events. All shipped `Dimensionality=Multivariate`
(full-mission inference, not our six channels). `real ESA Point / Local` is
empty here: official Point Local `id_42` is Univariate and not in this pool.

| Repo kind | Windows | Events | Official type | Events |
|---|---:|---:|---|---|
| real ESA Point / Local | 0 | 0 | — | — |
| real ESA Point / Global | 18 | 3 | Point × Global | `id_109`, `id_110`, `id_114` (`class_3`) |
| real ESA local subsequence | 12 | 2 | Subsequence × Local | `id_85` (`class_21`), `id_193` (`class_22`) |
| real level shift | 6 | 1† | Subsequence × Global + `dmu` | `id_83` (`class_3`) |
| real ESA global subsequence | 636 | 17† | leftover Subsequence × Global | campaigns, plus official short Subsequence spikes `id_90`, `id_93`, `id_107` |

† `id_83` is in both shift and global subsequence.

Official `Length=Point` is a **subset** of what the old morphology called
“short spike.” Half of those morphology spikes were official **Subsequence**
(spans 27–63 bins: `id_90`, `id_93`, `id_107`). They now sit in global
subsequence. That is the intended map: we take ESA’s Length, we do not
re-derive “short” from `label_hi − label_lo`.

---

## 6. Legacy morphology (do not use for new runs)

`assign_morphology_kinds` is the old duration + amplitude five-name split.
It does **not** read ESA type columns.

```
span  =  label_hi − label_lo     # event duration in 30 s bins
amp   =  max(x) − min(x)
dmu   =  abs( mean(first W/2) − mean(last W/2) )
```

| Legacy name | Rule (defaults) |
|---|---|
| real short spike | span < 100 **and** amp ≥ 0.1 |
| real short subtle | span < 100, smaller amp |
| real medium event | leftover with span < 4000 |
| real long campaign | leftover with span ≥ 4000 |
| real level shift | not short, and `dmu ≥ 0.01` |

Locked tables / caches that still cite this rule:

- `results/shell_kindmix_score/` (morphology 1536 freeze; not the live §2 row)
- cached `results/shell_kindmix/` and `results/shell_kindproto/` galleries
  (generated before this map)

Live §2 stratified numbers use ESA kinds: `results/shell_kindmix_score_esa/`.
256-donor ESA eyes: `results/shell_kindmix_esa/`.

A new `kindmix` / `kindproto` / plot run uses §3–4, not this table. Do not
mix a morphology-kind gallery with ESA-kind plot rows.

On the same 672 windows, morphology vs ESA:

```
                         Point/G   global subseq   local subseq   level shift
short spike                   18              18              0             0
short subtle                   0               0              6             0
level shift                    0               0              0             6
medium event                   0              24              6             0
long campaign                  0             594              0             0
```

The six morphology shelves are the same six `id_83` crops. Morphology “medium”
on `id_83` is now leftover global subsequence. Morphology “short subtle”
(`id_85`) is official local subsequence.

---

## 7. Do not

- Call `real level shift` an official ESA-ADB type.
- Cite `Class` as if we conditioned proto on cause groups (we do not).
- Expect a `W=512` crop to *be* a multi-day campaign because ESA `Length` is
  Subsequence. That name only means “not a Point event.”
- Re-infer types with ESA’s script unless we decide to (not done).
- Overwrite `results/shell_s4` or the morphology freeze in
  `results/shell_kindmix_score/`.
