#!/bin/bash

xelatex lrec2026-example.tex
bibtex lrec2026-example.aux
bibtex languageresource.aux
xelatex lrec2026-example.tex
xelatex lrec2026-example.tex