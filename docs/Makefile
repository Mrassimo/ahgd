# Makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = source
BUILDDIR     = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets for development
clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

pdf:
	@$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS) $(O)
	@make -C "$(BUILDDIR)/latex" all-pdf
	@echo "Build finished. The PDF file is in $(BUILDDIR)/latex."

epub:
	@$(SPHINXBUILD) -b epub "$(SOURCEDIR)" "$(BUILDDIR)/epub" $(SPHINXOPTS) $(O)
	@echo "Build finished. The epub file is in $(BUILDDIR)/epub."

linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest:
	@$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest" $(SPHINXOPTS) $(O)
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."

coverage:
	@$(SPHINXBUILD) -b coverage "$(SOURCEDIR)" "$(BUILDDIR)/coverage" $(SPHINXOPTS) $(O)
	@echo "Testing of coverage in the sources finished, look at the " \
	      "results in $(BUILDDIR)/coverage/python.txt."

# Development helpers
install-deps:
	uv pip install sphinx sphinx-rtd-theme sphinx-autoapi myst-parser furo sphinx-autobuild

serve: html
	cd $(BUILDDIR)/html && python -m http.server 8000

watch:
	sphinx-autobuild --host 0.0.0.0 --port 8000 "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Quality checks
spellcheck:
	@command -v aspell >/dev/null 2>&1 || { echo >&2 "aspell is required but not installed."; exit 1; }
	@find $(SOURCEDIR) -name "*.rst" -exec aspell check {} \;

lint:
	@command -v doc8 >/dev/null 2>&1 || { echo >&2 "doc8 is required but not installed. Run: pip install doc8"; exit 1; }
	@doc8 $(SOURCEDIR)

# Deployment helpers
deploy-gh-pages: html
	@echo "Deploying to GitHub Pages..."
	@git checkout gh-pages || git checkout -b gh-pages
	@cp -r $(BUILDDIR)/html/* .
	@git add .
	@git commit -m "Update documentation" || true
	@git push origin gh-pages
	@git checkout main

# CI/CD helpers
ci-build: clean html linkcheck
	@echo "CI build complete"

ci-deploy: ci-build deploy-gh-pages
	@echo "CI deployment complete"