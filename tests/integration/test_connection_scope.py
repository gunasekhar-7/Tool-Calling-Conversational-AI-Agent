from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from agent.api.main import create_app
from agent.config import Settings
from agent.db.seed import CUSTOMER_A
from tests.conftest import collect_events, create_session
from tests.fakes import ScriptedChatModel

pytestmark = pytest.mark.integration


class _ProbeModel(ScriptedChatModel):
    """Records how many pool connections were checked out when the model ran."""

    observed: list[int] = Field(default_factory=list)
    engine: Any = None

    async def _astream(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        self.observed.append(self.engine.pool.checkedout())
        for chunk in super()._stream(messages, stop, run_manager, **kwargs):
            yield chunk


@contextlib.asynccontextmanager
async def _client(settings: Settings, model: Any) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings, model=model, checkpointer=InMemorySaver())
    async with app.router.lifespan_context(app):
        if hasattr(model, "engine"):
            model.engine = app.state.engine
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", timeout=30
        ) as client:
            yield client


async def test_a_model_call_never_holds_a_pooled_connection(
    settings: Settings, seeded: None
) -> None:
    # One connection in the pool: if a turn kept it while the model was thinking,
    # the probe would observe 1 (and a second concurrent turn would deadlock).
    narrow = settings.model_copy(update={"db_pool_size": 1, "db_max_overflow": 0})
    model = _ProbeModel(responses=[AIMessage(content="I can help with that.")])

    async with _client(narrow, model) as client:
        session_id = await create_session(client, str(CUSTOMER_A))
        for _ in range(2):
            events = await collect_events(client, session_id, "What can you do?")
            assert "done" in dict(events)

    assert model.observed == [0, 0], "the model ran with a pooled connection checked out"
