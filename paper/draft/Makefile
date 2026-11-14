PY       ?= ../../.venv/bin/python
TECTONIC ?= $(HOME)/.local/bin/tectonic
SLIDE_FIGS = ../../slides/141026/figs

SLIDE_PDF = figs/kinds.pdf figs/gallery.pdf figs/sweep.pdf
PAPER_PDF = figs/noise_gallery.pdf figs/noise_chain.pdf \
            figs/cmp_same_donor.pdf figs/cmp_nearest.pdf figs/cmp_random.pdf \
            figs/det_events.pdf figs/det_hitmap.pdf figs/det_hitmap_strat.pdf \
            figs/det_kwindows.pdf

.PHONY: all figs clean

all: main.pdf

# kinds, gallery and sweep are drawn by the slide script from the saved run.
# Grouped target so one change of make_figs.py runs the script once.
$(SLIDE_PDF) &: ../../slides/141026/make_figs.py
	$(PY) ../../slides/141026/make_figs.py
	cp $(SLIDE_FIGS)/kinds.pdf $(SLIDE_FIGS)/gallery.pdf $(SLIDE_FIGS)/sweep.pdf figs/

# From-noise, generator-comparison and detector figures, read from
# results/shell_fixsweep_noise, results/paper_trace and results/paper_detect.
$(PAPER_PDF) &: make_figs.py
	$(PY) make_figs.py

figs: $(SLIDE_PDF) $(PAPER_PDF)

main.pdf: main.tex refs.bib $(SLIDE_PDF) $(PAPER_PDF)
	$(TECTONIC) main.tex

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out
