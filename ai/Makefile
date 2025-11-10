.PHONY: help fmt lint typecheck test up down ingest reindex eval clean

help:
	@echo "Available targets:"
	@echo "  fmt          - Format code with black and ruff"
	@echo "  lint         - Lint code with ruff"
	@echo "  typecheck    - Type check with mypy"
	@echo "  test         - Run tests with pytest"
	@echo "  up           - Start docker-compose services"
	@echo "  down         - Stop docker-compose services"
	@echo "  ingest       - Run ingestion pipeline"
	@echo "  reindex      - Rebuild vector index"
	@echo "  eval         - Run evaluation suite"
	@echo "  clean        - Clean cache and temporary files"

fmt:
	black app/
	ruff check --fix app/

lint:
	ruff check app/

typecheck:
	mypy app/ --ignore-missing-imports

test:
	pytest app/tests/ -v

up:
	docker-compose -f ../docker-compose.yml up -d

down:
	docker-compose -f ../docker-compose.yml down

ingest:
	python -m app.scripts.ingest --seed https://www.irs.gov/ --max-pages 1500 --concurrency 4

reindex:
	python -m app.scripts.rebuild_index

eval:
	python -m app.scripts.eval_suite

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	rm -rf .coverage htmlcov/


