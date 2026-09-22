PYTHON ?= python

.PHONY: install build-data run test

install:
	$(PYTHON) -m pip install -e .[dev]

build-data:
	$(PYTHON) -m text_classifier.ingest

run:
	$(PYTHON) -m text_classifier.app

test:
	$(PYTHON) -m pytest
