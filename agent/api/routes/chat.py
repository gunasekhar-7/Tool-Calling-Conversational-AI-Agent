from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from agent.agent.guardrails.input_guard import screen_input
from agent.api.deps import Principal, get_runtime, get_settings, require_principal
from agent.api.sse import with_keepalive
from agent.config import Settings
from agent.errors import InvalidInput, ModelUnavailable, PolicyDenied, RateLimited
from agent.logging import request_id_var
from agent.runtime import AgentRuntime
from agent.schemas import MessageRequest

router = APIRouter()


@router.post("/v1/sessions/{session_id}/messages")
async def send_message(
    request: Request,
    session_id: UUID,
    body: MessageRequest,
    principal: Principal = Depends(require_principal),
) -> StreamingResponse:
    settings: Settings = get_settings(request)
    runtime: AgentRuntime = get_runtime(request)

    await runtime.ensure_session(session_id, customer_id=principal.customer_id)
    await runtime.assert_no_pending_approval(session_id)

    if runtime.model is None:
        raise ModelUnavailable("LLM_MODEL and OPENAI_API_KEY must be configured")

    metrics = request.app.state.metrics
    screen = screen_input(body.message, settings.max_message_chars)
    if not screen.allowed:
        metrics.observe_block(screen.reason)
        if screen.reason in {"message_too_long", "empty_message"}:
            raise InvalidInput(f"message rejected: {screen.reason}")
        raise PolicyDenied(f"message rejected: {screen.reason}")

    limiter = request.app.state.rate_limiter
    async with runtime.session_factory() as limit_db:
        allowed, retry_after = await limiter.check(limit_db, principal.customer_id)
        await limit_db.commit()
    if not allowed:
        metrics.observe_block("rate_limited")
        raise RateLimited("too many messages for this account", retry_after=retry_after)

    stream = with_keepalive(
        runtime.stream_turn(
            session_id=session_id,
            message=screen.sanitized,
            request_id=request_id_var.get() or "-",
            customer_id=principal.customer_id,
        ),
        interval=settings.sse_keepalive_seconds,
    )
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
