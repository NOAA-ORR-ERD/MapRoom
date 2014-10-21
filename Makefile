# Rebuild all generated files

LOADER_DIR := maproom/layers/loaders
LOADER_COG := $(LOADER_DIR)/__init__.py
LOADERS := $(wildcard $(LOADER_DIR)/[^_]*.py)

IDENTIFIER_DIR := maproom/file_type
IDENTIFIER_COG := $(IDENTIFIER_DIR)/__init__.py
IDENTIFIERS := $(wildcard $(IDENTIFIER_DIR)/[^_]*.py)


COGS := $(LOADER_COG) $(IDENTIFIER_COG)

all:: $(COGS) deps

$(LOADER_COG): $(LOADERS)
	cog.py -r $@
	touch $@

$(IDENTIFIER_COG): $(IDENTIFIERS)
	cog.py -r $@
	touch $@

deps:
	cd deps/pytriangle-1.6.1; python setup.py install
	python setup_library.py build_ext --inplace


.PHONY: print-% deps

print-%: ; @ echo $* = $($*)
