#!/bin/bash

latex tables.tex
latex tables.tex
dvips -P pdf -o tables.ps tables
ps2pdf13 tables.ps tables.pdf


