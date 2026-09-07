.PHONY: install up down pipeline run test

install:
	pip install -e ".[dev]"

up:
	docker compose up -d

down:
	docker compose down

pipeline:
	python scripts/build_pipeline.py

run:
	uvicorn causal_kg.webapp.main:app --reload --port 8001 --app-dir src

test:
	pytest tests/ -v
