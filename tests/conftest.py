from __future__ import annotations

import contextlib
import json
import os
from collections.abc import AsyncIterator, Callable
from uuid import UUID

import httpx
import psycopg
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from agent.api.main import create_app
from agent.config import Settings
from agent.db.models import Base
from agent.db.seed import CUSTOMER_A, CUSTOMER_B, reset_and_seed
from agent.db.session import create_session_factory

HOST = os.environ.get("PGHOST", "127.0.0.1")
PORT = int(os.environ.get("PGPORT", "55432"))
USER = os.environ.get("PGUSER", "postgres")
PASSWORD = os.environ.get("PGPASSWORD", "postgres")
TEST_DB = "pm_test"

ADMIN_DSN = f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/postgres"
TEST_URL = f"postgresql+psycopg://{USER}:{PASSWORD}@{HOST}:{PORT}/{TEST_DB}"

# One token per customer, exactly like production. There is no master key.
CUSTOMER_TOKENS = {str(CUSTOMER_A): "test-key-a", str(CUSTOMER_B): "test-key-b"}
CUSTOMER_KEYS = {token: UUID(customer) for customer, token in CUSTOMER_TOKENS.items()}
AUTH = {"Authorization": f"Bearer {CUSTOMER_TOKENS[str(CUSTOMER_A)]}"}
AUTH_B = {"Authorization": f"Bearer {CUSTOMER_TOKENS[str(CUSTOMER_B)]}"}


def auth_for(customer_id: object) -> dict[str, str]:
    return {"Authorization": f"Bearer {CUSTOMER_TOKENS[str(customer_id)]}"}


def _pg_available() -> bool:
    try:
        with psycopg.connect(ADMIN_DSN, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


PG_AVAILABLE = _pg_available()


def _ensure_database() -> None:
    with psycopg.connect(ADMIN_DSN, autocommit=True) as conn:
        row = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)).fetchone()
        if row is None:
            conn.execute(f'CREATE DATABASE "{TEST_DB}"')


if PG_AVAILABLE:
    _ensure_database()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if PG_AVAILABLE:
        return
    skip = pytest.mark.skip(reason=f"PostgreSQL not reachable at {HOST}:{PORT}")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        database_url=TEST_URL,
        customer_api_keys=CUSTOMER_KEYS,
        environment="test",
        max_message_chars=2000,
        rate_limit_per_min=1000,
        rate_limit_burst=1000,
        approval_ttl_seconds=900,
    )


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(TEST_URL, pool_pre_ping=True)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
async def seeded(engine: AsyncEngine) -> AsyncIterator[None]:
    factory = create_session_factory(engine)
    async with factory() as db:
        await reset_and_seed(db)
        await db.commit()
    yield


@pytest.fixture
async def factory(engine: AsyncEngine, seeded: None) -> async_sessionmaker[AsyncSession]:
    return create_session_factory(engine)


@pytest.fixture
async def db(
    factory: async_sessionmaker[AsyncSession], seeded: None
) -> AsyncIterator[AsyncSession]:
    async with factory() as session:
        yield session


@contextlib.asynccontextmanager
async def running_client(
    app_factory: Callable[[], object],
) -> AsyncIterator[httpx.AsyncClient]:
    app = app_factory()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", timeout=30
        ) as client:
            yield client


@pytest.fixture
def app_client(settings: Settings, seeded: None):
    @contextlib.asynccontextmanager
    async def _open(model=None) -> AsyncIterator[httpx.AsyncClient]:
        app = create_app(settings, model=model, checkpointer=InMemorySaver())
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver", timeout=30
            ) as client:
                yield client

    return _open


async def create_session(client: httpx.AsyncClient, customer_id: str) -> str:
    """Create a session as that customer.

    The request has no body: identity comes only from the bearer token, so passing
    a customer id here selects which token to use, never what the server believes.
    """
    response = await client.post("/v1/sessions", headers=auth_for(customer_id))
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["customer_id"] == str(customer_id), body
    return body["session_id"]


async def collect_events(
    client: httpx.AsyncClient, session_id: str, message: str
) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    name = ""
    async with client.stream(
        "POST",
        f"/v1/sessions/{session_id}/messages",
        json={"message": message},
        headers=AUTH,
    ) as response:
        assert response.status_code == 200, await response.aread()
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                events.append((name, json.loads(line[len("data: ") :])))
    return events


async def truncate_all(engine: AsyncEngine) -> None:
    from agent.db.seed import TRUNCATE_ORDER

    async with engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE {TRUNCATE_ORDER} RESTART IDENTITY CASCADE"))
