PY       ?= ../../.venv/bin/python
TECTONIC ?= $(HOME)/.local/bin/tectonic
SLIDE_FIGS = ../../slides/141026/figs

.PHONY: all figs clean

all: main.pdf

# kinds, gallery and sweep are drawn by the slide script from the saved run.
# Grouped target so one change of make_figs.py runs the script once.
figs/kinds.pdf figs/gallery.pdf figs/sweep.pdf &: ../../slides/141026/make_figs.py
	$(PY) ../../slides/141026/make_figs.py
	cp $(SLIDE_FIGS)/kinds.pdf $(SLIDE_FIGS)/gallery.pdf $(SLIDE_FIGS)/sweep.pdf figs/

figs: figs/kinds.pdf figs/gallery.pdf figs/sweep.pdf

main.pdf: main.tex refs.bib figs/kinds.pdf figs/gallery.pdf figs/sweep.pdf
	$(TECTONIC) main.tex

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out
