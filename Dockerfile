# syntax=docker/dockerfile:1.7

# ---------- build ----------
FROM python:3.12-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first so a source-only change does not reinstall them.
COPY pyproject.toml ./
RUN python -m venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-cache \
        "fastapi>=0.115" "uvicorn[standard]>=0.30" "pydantic>=2.8" \
        "pydantic-settings>=2.4" "sqlalchemy[asyncio]>=2.0" \
        "psycopg[binary,pool]>=3.2" "alembic>=1.13" "langgraph>=1.2,<2" \
        "langgraph-checkpoint-postgres>=2.0" "langchain-core>=1.0" \
        "langchain-openai>=1.0" "httpx>=0.27"

# The application itself, without dev extras or the dashboard's Streamlit.
COPY agent ./agent
COPY migrations ./migrations
COPY scripts ./scripts
COPY alembic.ini ./
RUN uv pip install --python /opt/venv/bin/python --no-cache --no-deps .

# ---------- runtime ----------
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY --from=build --chown=10001:10001 /app /app

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"

CMD ["sh", "-c", "alembic upgrade head && exec uvicorn agent.api.main:create_app --factory --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
