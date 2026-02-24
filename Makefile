UV ?= uv

.PHONY: sync install lock run-baseline run-feature-prepare test lint format typecheck verify clean

sync:
	$(UV) sync --all-groups

install: sync

lock:
	$(UV) lock

run-baseline:
	$(UV) run python src/paddy_yield_ml/pipelines/baseline.py

run-feature-prepare:
	$(UV) run python src/paddy_yield_ml/pipelines/feature_prepare.py

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format .

typecheck:
	$(UV) run ty check src tests

verify: lint typecheck test

clean:
	$(UV) run python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache','.ruff_cache','.mypy_cache','htmlcov']]"
