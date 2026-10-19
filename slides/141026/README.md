# Briefing deck — build and view

Ten Beamer slides. `anogen_briefing.tex` is the whole deck; `make_figs.py` draws
only the five data figures it includes.

## Build

```bash
cd slides/141026
make              # figures + PDF
make slides       # PDF only, when the figures have not changed
make figs         # figures only
make preview      # one PNG per slide in preview/
make clean
```

The engine is [tectonic](https://tectonic-typesetting.github.io), already installed at
`~/.local/bin/tectonic`. It downloads the LaTeX packages it needs on first use and caches
them, so the first build after a fresh checkout needs network and takes about a minute;
later builds take ~3 s. There is no TeX Live on this box, so `pdflatex` and `latexmk` are
not available.

To compile by hand, without `make`:

```bash
cd slides/141026
tectonic -X compile --synctex --keep-logs anogen_briefing.tex
```

On a machine that does have TeX Live, the source is plain Beamer and needs no changes:

```bash
latexmk -pdf anogen_briefing.tex     # or: pdflatex anogen_briefing.tex (run twice)
```

## View

This is a remote SSH session with no display, so `evince` and friends will not work —
the PDF has to render inside Cursor.

**LaTeX Workshop** (installed, configured in `.vscode/settings.json`):

- Open `anogen_briefing.tex`, then `Ctrl+Alt+V` to open the PDF beside it.
- Saving the `.tex` rebuilds automatically. `Ctrl+Alt+B` builds on demand.
- `Ctrl+Alt+J` jumps from the cursor to that spot in the PDF; `Ctrl+click` in the PDF
  jumps back to the source line.
- It is set up to call tectonic by absolute path, because the extension does not read
  `~/.bashrc` and so would not find it on `PATH`.

**Without the extension**, `make preview` writes `preview/p-01.png` … `p-10.png`. Cursor
opens PNGs natively, including over SSH, and they refresh when rebuilt.

A plain PDF viewer such as `tomoki1207.pdf` also works, but it is a UI-side extension:
it cannot be installed from the remote, only from the Extensions panel in your local
Cursor window.

## Editing

| What | Where |
|---|---|
| palette, frame-title style, box style | preamble of `anogen_briefing.tex` |
| a slide's text, equations, tables | the `\begin{frame}` block for it |
| the five figures | `make_figs.py` |

Useful macros: `\gd{...}` green, `\bd{...}` red, `\mt{...}` grey.
`\begin{eqbox}[good]{title}` gives the green box, `[bad]` the red one, no argument the
blue one.

Two things that will bite:

- The base font is **9pt**, not 11pt. A 16×9 cm Beamer page is small relative to the type
  and these frames are dense; at 11pt every one of them overflowed. Raise it if you cut
  content, and watch for `Overfull \vbox` in the build output — that means a frame is
  running off the bottom.
- The figures are sized in **inches to the space they occupy on the page** and included at
  scale 1, so their labels match the body text. Resizing one with `width=` in the `.tex`
  scales its fonts too; change `figsize` in `make_figs.py` instead.

## Where the numbers come from

Everything quoted is from the run, not retyped from a draft:

| Slide | Source |
|---|---|
| nominal volume on the title slide (7 years/channel, 24,576 vs 672 windows) | `results/shell_s0/summary.json`: `panel_T` 7364160 bins x 30 s, `n_nominal_windows`, `n_anomaly_windows` |
| ESA-ADB window and event counts | `docs/ESA_LABELS.md` §5, `results/shell_s0/summary.json` |
| the four real example windows | `results/shell_s0/labeled_arrays.npz` rows 654, 196, 42, 90 (`labeled_windows.csv` events id_109, id_83, id_185, id_193, all channel 41); nominal band from `results/shell_s3/galleries.npz` key `cond` |
| EDI column | recomputed, not the stored `_edi_f`: `edi_by_method` (k=16, seed 0) on a partition built per fold from the union of that fold's six galleries, averaged over the three folds |
| realism + diversity table | ARP and Coverage@τ: `results/shell_s4/summary.json` (GenIAS, handcrafted injection) and `results/shell_fixsweep/summary.json` (ours, `mean.arp_anomaly`); diversity: `mean_pairwise_distance(embed_windows(·))`; CutAddPaste scored with `_score_methods` from `phases/s4.py` |
| real-anomaly ceiling | `results/shell_fixsweep/summary.json`, `_calibration.oracle_train_fold_anomalies` |
| morphology figures under it | `scripts/measure_morphology.py` |
| DDIM sampler and step counts | `anogen.shell.diffusion.ddim_sample` (50 steps from `T-1`); the frozen edit's `nu` 0.2 / `ddim_steps` 50 are `_PARENT50` in `phases/kindmix.py`, and `steer.guided_ddim` sets `t_start = round(nu*(T-1))` = 40 |
| detection curve and gain table | `results/shell_augdetect/sweep.json`, key `vs_real_only` |
| detector head-to-head vs GenIAS | `results/shell_augdetect_vsgenias/summary.json` |
| per-kind gallery | `results/shell_s0`, `results/shell_fixsweep/<variant>_fold{0,1,2}.npz` |

ARP and diversity are comparable across the S4 baselines and our fixsweep variants because
both runs use the same frozen `tau` (0.10480065396584726), the same three event-held-out
folds with the same per-fold anomaly counts, and the same locked `feature_pack_v1`.
`embed_windows` z-scores each window, so raw and min-max-scaled units give identical
morphology features. Re-scoring GenIAS and the handcrafted injector with
`_score_methods` reproduces the stored S4 ARP to machine precision — that check is what
licenses putting CutAddPaste in the same column.

The "nothing done to it" row is the `donor` gallery: `results/shell_s3/galleries.npz`
key `cond`, the conditioning windows unmodified. It is in the table on purpose — without
it, GenIAS's and CutAddPaste's scores look like realism rather than like inaction.

The detector head-to-head was run with:

```bash
anogen augdetect -c <copy of shell_mission1.yaml with augdetect.arms/compare replaced
                     and augdetect_dir: results/shell_augdetect_vsgenias>
```

There is also a raw-window distribution audit in `results/shell_realism_deck/`
(MDD, ACD, marginal W1, FFT amplitude W1, C2ST, from a `realism`-phase run with
`realism_dir` redirected). It is deliberately **not** on the slide. On four of its five
metrics the untouched donor window scores best, and GenIAS and CutAddPaste land within
0.003 of it, so those metrics rank plausibility rather than anomaly generation and cannot
order the methods. The per-fold numbers are also unstable: MDD and feature distance are
20–30× larger on fold 2 than on folds 0 and 1 for every method, so a three-fold mean is
essentially fold 2.

The four example windows on the data slide are each the clearest window of their kind,
and for the global subsequence that is not the typical one: only 84 of the 636 global
subsequence windows visibly leave the nominal band, the other 552 looking ordinary over
4.27 h because the label is global over the channel's whole history, not over the window.
`fig_kinds` asserts that each picked row still carries the kind it is labelled with, so
the figure fails loudly rather than silently mislabels if the window index table changes.

EDI is partition-dependent: `edi_by_method` fits k-means on the *union* of the galleries
handed to it, so a method's entropy changes if the set of compared methods changes. The
column is therefore only a ranking within its own partition, and it is not comparable with
the `_edi_f` block stored in `results/shell_fixsweep/summary.json` (which partitions on
fold-0 galleries only and excludes the real anomalies). The slide's values come from a
per-fold partition over real anomalies, ours, GenIAS, CutAddPaste, handcrafted and donor:
1.48 / 2.44 / 1.14 / 1.27 / 1.31 / 1.20, against a ceiling of log 16 = 2.77. Mean pairwise
distance orders the same methods identically (5.05 / 10.53 / 1.09 / 1.12 / 1.21 / 1.11),
which is why swapping the column did not change the claim.

Narrative and method are in `docs/ESA_RUN_2026-10-06.md` §11 and
`docs/MODEL_CONTRAST_HEAD.md`.
