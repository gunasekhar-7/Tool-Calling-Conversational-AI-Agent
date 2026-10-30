.PHONY: install migrate seed api ui test lint fmt typecheck eval check up down

install:
	uv pip install --python .venv/Scripts/python.exe -e ".[dev]" || pip install -e ".[dev]"

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

check: lint typecheck test eval

up:
	docker compose up --build

down:
	docker compose down
