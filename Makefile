.PHONY: install migrate seed api ui test lint fmt typecheck eval check up down

install:
	uv sync --frozen --extra dev --extra dashboard

migrate:
	alembic upgrade head

seed:
	python scripts/seed.py

api:
	python scripts/serve.py

ui:
	streamlit run dashboard/app.py

test:
	pytest

lint:
	ruff check .

fmt:
	ruff format agent tests evals dashboard scripts

typecheck:
	mypy

eval:
	python scripts/evaluate.py --trials 3

eval-live:
	python scripts/evaluate.py --mode live --trials 3

load:
	python scripts/loadtest.py --sessions 20 --messages 5

check: lint typecheck test eval

up:
	docker compose up --build

down:
	docker compose down
