# WildfireRisk-EU — Pipeline Makefile
# Requires: Python env 'geo311' activated; msys64 make available.
# Usage: make <target>

PYTHON := python
PYTEST := pytest
RUFF   := ruff

.DEFAULT_GOAL := help

.PHONY: help install lint test acquire preprocess features score validate outputs \
       acquire-dynamic features-dynamic model all all-v2 clean \
       era5-diagnostic weight-sensitivity

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install:  ## Install package in editable mode
	$(PYTHON) -m pip install -e ".[dev]"

lint:  ## Run ruff linter
	$(RUFF) check src/ tests/

lint-fix:  ## Run ruff with auto-fix
	$(RUFF) check --fix src/ tests/

test:  ## Run test suite
	$(PYTEST) tests/

test-cov:  ## Run tests with coverage report
	$(PYTEST) tests/ --cov=src --cov-report=term-missing

# ── Pipeline stages ────────────────────────────────────────────────────────────

acquire:  ## Stage 1a — Download structural data sources
	$(PYTHON) -m src.acquire.buildings
	$(PYTHON) -m src.acquire.dem
	$(PYTHON) -m src.acquire.corine
	$(PYTHON) -m src.acquire.effis
	$(PYTHON) -m src.acquire.era5_fwi
	$(PYTHON) -m src.acquire.firms

acquire-dynamic:  ## Stage 1b — Download ERA5 hourly event-context data (v2)
	$(PYTHON) -m src.acquire.era5_dynamic

preprocess:  ## Stage 2 — Clean, harmonize, load to DuckDB
	$(PYTHON) -m src.preprocess.buildings
	$(PYTHON) -m src.preprocess.terrain
	$(PYTHON) -m src.preprocess.vegetation
	$(PYTHON) -m src.preprocess.fwi
	$(PYTHON) -m src.preprocess.fire_history

features:  ## Stage 3a — Extract structural per-building features
	$(PYTHON) -m src.features.terrain
	$(PYTHON) -m src.features.vegetation
	$(PYTHON) -m src.features.fire_weather
	$(PYTHON) -m src.features.fire_history

features-dynamic:  ## Stage 3b — Compute event-context dynamic features (v2)
	$(PYTHON) -m src.features.dynamic_event

score:  ## Stage 4 — Compute risk scores
	$(PYTHON) -m src.scoring.engine

validate:  ## Stage 5 — Backtest against 4 LOEO events
	$(PYTHON) -m src.validation.validator

model:  ## Stage 5b — LightGBM event-context model + SHAP (v2)
	$(PYTHON) -m src.model.event_model

outputs:  ## Stage 6 — Generate deliverables
	$(PYTHON) -m src.outputs.result_table
	$(PYTHON) -m src.outputs.risk_map
	$(PYTHON) -m src.outputs.validation_report
	$(PYTHON) -m src.outputs.executive_summary

# ── Optional diagnostics (not part of main pipeline) ──────────────────────────

era5-diagnostic:  ## Diagnostic — ERA5 grid resolution analysis
	$(PYTHON) -m src.validation.era5_resolution_diagnostic

weight-sensitivity:  ## Diagnostic — Monte Carlo weight sensitivity analysis
	$(PYTHON) -m src.scoring.weight_sensitivity

# ── End-to-end pipelines ──────────────────────────────────────────────────────

all: acquire preprocess features score validate outputs  ## Run v1 structural pipeline end-to-end

all-v2: acquire acquire-dynamic preprocess features features-dynamic score validate model outputs  ## Run full v2 pipeline end-to-end

clean:  ## Remove processed data and outputs (keeps raw data)
	@echo "Removing processed data and outputs..."
	@rm -rf data/processed/*
	@rm -rf outputs/tables/* outputs/maps/* outputs/reports/* outputs/summaries/*
	@echo "Done. Raw data in data/raw/ preserved."
