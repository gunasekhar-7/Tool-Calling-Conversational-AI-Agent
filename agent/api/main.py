from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.language_models.chat_models import BaseChatModel
from psycopg_pool import AsyncConnectionPool

from agent.agent.guardrails.rate_limit import RateLimiter
from agent.api.errors import install_error_handlers
from agent.api.routes import approvals, chat, health, sessions
from agent.config import Settings, get_settings
from agent.db.session import create_engine, create_session_factory
from agent.logging import configure_logging, request_id_var
from agent.observability import Metrics
from agent.runtime import AgentRuntime
from agent.tracing import flush_tracing

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
        engine = create_engine(
            resolved.database_url,
            pool_size=resolved.db_pool_size,
            max_overflow=resolved.db_max_overflow,
        )
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
        app.state.runtime = AgentRuntime(
            settings=resolved,
            model=selected_model,
            checkpointer=saver,
            session_factory=factory,
            metrics=app.state.metrics,
        )
        logger.info(
            "service_started",
            extra={"extra_fields": {"environment": resolved.environment}},
        )
        try:
            yield
        finally:
            # Export buffered traces before the process gives up its exporters.
            flush_tracing()
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
    # Built outside the lifespan so a route can report a metric even when the
    # request never reaches a database-backed startup path.
    app.state.metrics = Metrics()
    app.state.rate_limiter = RateLimiter(
        rate_per_min=resolved.rate_limit_per_min, burst=resolved.rate_limit_burst
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_var.set(rid)
        try:
            # Streaming body-size enforcement: read and count actual bytes
            # rather than trusting Content-Length header (which can be spoofed/missing).
            if resolved.enforce_body_size_streaming:
                body_bytes = b""
                async for chunk in request.stream():
                    body_bytes += chunk
                    if len(body_bytes) > resolved.max_request_bytes:
                        request.app.state.metrics.observe_body_size_rejection()
                        response = JSONResponse(
                            status_code=413,
                            content={
                                "code": "request_too_large",
                                "message": (
                                    "Request body exceeds the configured limit of "
                                    f"{resolved.max_request_bytes} bytes."
                                ),
                                "request_id": rid,
                            },
                        )
                        response.headers["X-Request-ID"] = rid
                        return response
                # Re-create request with buffered body for downstream handlers
                request._body = body_bytes

            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        route = request.scope.get("route")
        label = getattr(route, "path", None) or "unmatched"
        request.app.state.metrics.observe_request(label, request.method, response.status_code)
        return response

    if resolved.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        )

    install_error_handlers(app)
    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(chat.router)
    app.include_router(approvals.router)
    return app
