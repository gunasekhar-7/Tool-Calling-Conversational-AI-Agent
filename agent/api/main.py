from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from langchain_core.language_models.chat_models import BaseChatModel
from psycopg_pool import AsyncConnectionPool

from agent.agent.guardrails.rate_limit import RateLimiter
from agent.api.errors import install_error_handlers
from agent.api.routes import approvals, chat, health, sessions
from agent.config import Settings, get_settings
from agent.db.session import create_engine, create_session_factory
from agent.logging import configure_logging, request_id_var
from agent.runtime import AgentRuntime

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    model: BaseChatModel | None = None,
    checkpointer: Any | None = None,
) -> FastAPI:
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_engine(resolved.database_url)
        factory = create_session_factory(engine)
        pool: AsyncConnectionPool | None = None
        saver = checkpointer

        if saver is None:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            pool = AsyncConnectionPool(
                conninfo=resolved.checkpoint_url,
                max_size=5,
                open=False,
                kwargs={"autocommit": True},
            )
            await pool.open()
            saver = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
            await saver.setup()

        selected_model = model
        if selected_model is None and resolved.has_model_credentials:
            from agent.agent.llm import build_chat_model

            selected_model = build_chat_model(resolved)

        app.state.settings = resolved
        app.state.engine = engine
        app.state.session_factory = factory
        app.state.checkpointer = saver
        app.state.rate_limiter = RateLimiter(resolved.rate_limit_per_min, resolved.rate_limit_burst)
        app.state.runtime = AgentRuntime(
            settings=resolved,
            model=selected_model,
            checkpointer=saver,
            session_factory=factory,
        )
        logger.info(
            "service_started",
            extra={"extra_fields": {"environment": resolved.environment}},
        )
        try:
            yield
        finally:
            await engine.dispose()
            if pool is not None:
                await pool.close()

    app = FastAPI(
        title="Order support agent",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.settings = resolved

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        return response

    install_error_handlers(app)
    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(chat.router)
    app.include_router(approvals.router)
    return app
