# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc
SOURCEDIR     = source
BUILDDIR      = build
PACKAGEDIR    = ../torchnaut

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: Run sphinx-apidoc before every build command
%: Makefile
	@echo "Generating API documentation..."
	@$(SPHINXAPIDOC) -o "$(SOURCEDIR)" "$(PACKAGEDIR)" --force
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
