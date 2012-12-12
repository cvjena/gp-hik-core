-include Makefile.cfg

ifeq "$(COMPRESSION)" "0"
DVIPDF_ARG+=-dAutoFilterColorImages=false -dColorImageFilter=/FlateEncode
endif

ifeq "$(LETTER)" "1"
DVIPS_ARG+=-t letter
endif

#LATEX=latex
LATEX=pdflatex

.PRECIOUS:%.aux %.bbl

%.dvi:%.tex

%.aux:%.tex
	$(LATEX) $<

%.bbl:%.tex %.aux
	bibtex $*
	$(LATEX) $<
	bibtex $*

%.dvi:%.tex %.bbl
	$(LATEX) $<

%.ps:%.dvi
	dvips -j0 -P generic $(DVIPS_ARG) $< -o $@

#this is the old version using dvipdf, which can not handle letter papersize
#%.pdf:%.dvi
#	dvipdf $(DVIPDF_ARG) $<

#this is the new version, going manually via dvips and ps2pdf
#this is exactly the same as the dvipdf script does
%.pdf:%.ps
	ps2pdf14 $(DVIPDF_ARG) $< $@

