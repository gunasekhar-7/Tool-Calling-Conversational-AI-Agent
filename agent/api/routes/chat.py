from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from agent.agent.guardrails.input_guard import screen_input
from agent.api.deps import get_runtime, get_settings, require_api_key
from agent.api.sse import with_keepalive
from agent.config import Settings
from agent.errors import InvalidInput, PolicyDenied, RateLimited
from agent.logging import request_id_var
from agent.runtime import AgentRuntime
from agent.schemas import MessageRequest

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/v1/sessions/{session_id}/messages")
async def send_message(
    request: Request,
    session_id: UUID,
    body: MessageRequest,
) -> StreamingResponse:
    settings: Settings = get_settings(request)
    runtime: AgentRuntime = get_runtime(request)

    await runtime.ensure_session(session_id)

    if runtime.model is None:
        from agent.errors import ModelUnavailable

        raise ModelUnavailable("LLM_MODEL and OPENAI_API_KEY must be configured")

    screen = screen_input(body.message, settings.max_message_chars)
    if not screen.allowed:
        if screen.reason in {"message_too_long", "empty_message"}:
            raise InvalidInput(f"message rejected: {screen.reason}")
        raise PolicyDenied(f"message rejected: {screen.reason}")

    limiter = request.app.state.rate_limiter
    allowed, retry_after = limiter.check(str(session_id))
    if not allowed:
        raise RateLimited("too many messages for this session", retry_after=retry_after)

    stream = with_keepalive(
        runtime.stream_turn(
            session_id=session_id,
            message=screen.sanitized,
            request_id=request_id_var.get() or "-",
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
