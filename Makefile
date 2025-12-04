PYTHON ?= python3
WINDOWS_PYTHON ?= python
VENV := .venv
ACTIVATE := $(VENV)/bin/activate
WINDOWS_ACTIVATE := $(VENV)/Scripts/activate

.PHONY: venv windows_venv install windows_install format windows_format test windows_test run windows_run

venv:
	$(PYTHON) -m venv $(VENV)

windows_venv:
	$(WINDOWS_PYTHON) -m venv $(VENV)

install: venv
	. $(ACTIVATE) && pip install .[dev]

windows_install: windows_venv
	. $(WINDOWS_ACTIVATE) && pip install .[dev]

format: venv
	. $(ACTIVATE) && black .
	. $(ACTIVATE) && isort .

windows_format: windows_venv
	. $(WINDOWS_ACTIVATE) && black .
	. $(WINDOWS_ACTIVATE) && isort .

test: venv
	. $(ACTIVATE) && pytest

windows_test: windows_venv
	. $(WINDOWS_ACTIVATE) && pytest

run: venv
	. $(ACTIVATE) && python main.py

windows_run: windows_venv
	. $(WINDOWS_ACTIVATE) && python main.py
