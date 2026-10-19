PY       ?= ../../.venv/bin/python
TECTONIC ?= tectonic -X compile --keep-logs
FIGS      = figs/diffusion_strip.pdf figs/kinds.pdf figs/shell.pdf figs/gallery.pdf figs/sweep.pdf

.PHONY: all slides figs preview clean

all: figs slides

slides: anogen_briefing.pdf

anogen_briefing.pdf: anogen_briefing.tex $(FIGS)
	$(TECTONIC) anogen_briefing.tex

# the figures read results/shell_s0, results/shell_fixsweep and
# results/shell_augdetect/sweep.json, so they only change when a phase is rerun
figs: $(FIGS)

$(FIGS): make_figs.py
	$(PY) make_figs.py

# one PNG per slide, for a quick look without opening a PDF viewer
preview: anogen_briefing.pdf
	mkdir -p preview && rm -f preview/*.png
	pdftoppm -png -r 110 anogen_briefing.pdf preview/p

clean:
	rm -f anogen_briefing.pdf anogen_briefing.log anogen_briefing.synctex.gz preview/*.png
