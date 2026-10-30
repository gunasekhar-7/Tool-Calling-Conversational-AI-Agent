from __future__ import annotations

import hmac
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agent.agent.graph import build_graph, graph_config
from agent.agent.guardrails.grounding import repair_or_template, verify
from agent.config import Settings
from agent.db.repositories import AuditRepo, PendingActionRepo, ReceiptRepo, SessionRepo
from agent.errors import (
    ApprovalExpired,
    ApprovalMismatch,
    ApprovalPending,
    Conflict,
    DomainError,
    ModelUnavailable,
    NotFound,
    WriteFailed,
)
from agent.schemas import ApprovalResponse, TranscriptMessage
from agent.tools import ALL_TOOL_SPECS, to_langchain_tools
from agent.tools.context import ToolContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnEvent:
    event: str
    data: dict[str, Any]


def _unpack(part: Any) -> tuple[Any, Any] | None:
    if isinstance(part, tuple) and len(part) == 2:
        return part[0], part[1]
    if isinstance(part, dict) and part.get("type") == "messages":
        data = part.get("data")
        if isinstance(data, tuple) and len(data) == 2:
            return data[0], data[1]
        return data, {}
    return None


def _text_of(message: Any) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str) and text:
        return text
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _final_text(state: dict[str, Any]) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            return _text_of(message).strip()
    return ""


class AgentRuntime:
    """Shared by the HTTP API and the evaluation harness."""

    def __init__(
        self,
        *,
        settings: Settings,
        model: BaseChatModel | None,
        checkpointer: Any,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self.settings = settings
        self.model = model
        self.checkpointer = checkpointer
        self.session_factory = session_factory

    async def ensure_session(self, session_id: UUID, *, customer_id: UUID | None = None) -> None:
        """Resolve the session and, when a caller is given, enforce ownership.

        A session that belongs to someone else is reported as missing, so the API
        never confirms that another customer's session id exists.
        """
        async with self.session_factory() as db:
            session = await SessionRepo(db).get(session_id)
            if customer_id is not None and session.customer_id != customer_id:
                raise NotFound("session not found")

    async def assert_no_pending_approval(self, session_id: UUID) -> None:
        """Refuse a new message while a write is staged.

        The paused graph holds an interrupt; feeding it a new user message would
        resume that interrupt with the message as the answer. Failing fast keeps the
        approval card the only way forward.
        """
        async with self.session_factory() as db:
            pending = await PendingActionRepo(db).find_open(session_id)
        if pending is not None:
            raise ApprovalPending(
                "a staged action is awaiting your decision; approve or reject it first"
            )

    async def _context(
        self,
        db: AsyncSession,
        session_id: UUID,
        request_id: str,
        customer_id: UUID | None = None,
    ) -> ToolContext:
        session = await SessionRepo(db).get(session_id)
        if customer_id is not None and session.customer_id != customer_id:
            # Same answer as a missing session: the API never confirms that another
            # customer's conversation exists.
            raise NotFound("session not found")
        return ToolContext(
            customer_id=session.customer_id,
            session_id=session.id,
            db=db,
            request_id=request_id,
        )

    def _graph(self, ctx: ToolContext, *, model: BaseChatModel | None = None) -> Any:
        selected = model if model is not None else self.model
        tools = to_langchain_tools(ALL_TOOL_SPECS, ctx) if selected is not None else []
        return build_graph(
            model=selected,
            tools=tools,
            checkpointer=self.checkpointer,
            ctx=ctx,
            max_tool_calls=self.settings.max_tool_calls,
            approval_ttl_seconds=self.settings.approval_ttl_seconds,
        )

    async def stream_turn(
        self,
        *,
        session_id: UUID,
        message: str,
        request_id: str,
        customer_id: UUID | None = None,
    ) -> AsyncIterator[TurnEvent]:
        async with self.session_factory() as db:
            # Ownership is answered before capability: a foreign session is 404 even
            # when the service has no model configured.
            ctx = await self._context(db, session_id, request_id, customer_id)
            if self.model is None:
                raise ModelUnavailable("model is not configured")
            graph = self._graph(ctx)
            config = graph_config(str(session_id), self.settings.max_tool_calls)

            emitted_tools: list[str] = []
            streamed = ""
            try:
                async for part in graph.astream(
                    {
                        "messages": [HumanMessage(content=message)],
                        "customer_id": str(ctx.customer_id),
                    },
                    config=config,
                    stream_mode="messages",
                    version="v2",
                    durability="sync",
                ):
                    unpacked = _unpack(part)
                    if unpacked is None:
                        continue
                    chunk, _meta = unpacked
                    if isinstance(chunk, AIMessageChunk):
                        for call in getattr(chunk, "tool_call_chunks", None) or []:
                            name = call.get("name") if isinstance(call, dict) else None
                            if name and name not in emitted_tools:
                                emitted_tools.append(name)
                                yield TurnEvent("tool_call", {"name": name})
                        text = _text_of(chunk)
                        if text:
                            streamed += text
                            yield TurnEvent("token", {"text": text})

                state = await graph.aget_state(config)
                values = state.values or {}

                for message_obj in values.get("messages", []):
                    for call in getattr(message_obj, "tool_calls", None) or []:
                        name = call.get("name")
                        if name and name not in emitted_tools:
                            emitted_tools.append(name)
                            yield TurnEvent("tool_call", {"name": name})

                if state.interrupts:
                    payload = state.interrupts[0].value
                    await db.commit()
                    yield TurnEvent("confirmation_required", dict(payload))
                    return

                answer = _final_text(values) or streamed
                receipts = await ReceiptRepo(db).list_for_session(session_id)
                report = verify(answer, receipts)
                if not report.ok:
                    unsupported = [claim.text for claim in report.unsupported]
                    logger.warning(
                        "ungrounded_answer",
                        extra={"extra_fields": {"unsupported": unsupported}},
                    )
                    yield TurnEvent("grounding_warning", {"unsupported": unsupported})
                    if self.model is not None:
                        answer = await repair_or_template(
                            answer=answer,
                            report=report,
                            receipts=receipts,
                            model=self.model,
                        )
                    yield TurnEvent("replace", {"text": answer})

                await db.commit()
                yield TurnEvent("done", {"text": answer, "tool_calls": emitted_tools})
            except Exception as exc:
                await db.rollback()
                logger.exception("turn_failed")
                yield TurnEvent(
                    "error",
                    {
                        "code": getattr(exc, "code", "internal_error"),
                        "message": getattr(exc, "message", None) or type(exc).__name__,
                        "request_id": request_id,
                    },
                )

    async def resolve_approval(
        self,
        *,
        session_id: UUID,
        pending_action_id: UUID,
        decision: str,
        idempotency_key: str,
        preview_hash: str,
        request_id: str,
        customer_id: UUID | None = None,
    ) -> ApprovalResponse:
        async with self.session_factory() as db:
            ctx = await self._context(db, session_id, request_id, customer_id)
            if self.model is None:
                raise ModelUnavailable("model is not configured")
            action = await PendingActionRepo(db).get_for_session(pending_action_id, session_id)

            if action.status == "executed":
                if action.consume_key == idempotency_key:
                    return ApprovalResponse(
                        status="executed",
                        ticket_id=(action.result or {}).get("ticket_id"),
                        replayed=True,
                    )
                raise Conflict("this action was already executed")
            if action.status == "rejected":
                raise Conflict("this action was rejected")
            if action.status == "expired":
                raise ApprovalExpired("this approval has expired")
            moment = datetime.now(UTC)
            if action.expires_at <= moment:
                # Bookkeeping only: record the lapse, then exit. No write executes.
                action.status = "expired"
                action.decided_at = moment
                await db.commit()
                raise ApprovalExpired("this approval has expired")
            if not hmac.compare_digest(preview_hash, action.args_hash):
                raise ApprovalMismatch("approval does not match the staged action")

            graph = self._graph(ctx)
            config = graph_config(str(session_id), self.settings.max_tool_calls)
            snapshot = await graph.aget_state(config)
            if not snapshot.interrupts:
                raise Conflict("no approval is pending for this session")

            resume = {
                "decision": decision,
                "idempotency_key": idempotency_key,
                "preview_hash": preview_hash,
            }
            try:
                result_state = await graph.ainvoke(
                    Command(resume=resume),
                    config=config,
                    durability="sync",
                )
                refreshed = await PendingActionRepo(db).get_for_session(
                    pending_action_id, session_id
                )
                ticket_id = (refreshed.result or {}).get("ticket_id")
                await AuditRepo(db).record(
                    session_id=session_id,
                    actor=f"customer:{ctx.customer_id}",
                    action=f"approval.{decision}",
                    target=str(pending_action_id),
                    detail={"ticket_id": ticket_id, "status": refreshed.status},
                )
                # The single commit for this request. Everything the graph flushed --
                # the ticket, the terminal state, the audit row -- lands together or not
                # at all.
                await db.commit()
            except DomainError:
                await db.rollback()
                raise
            except Exception as exc:
                # A failure after the INSERT must not leave a ticket behind, and the
                # caller must be able to tell that nothing was created.
                await db.rollback()
                logger.exception("approval_write_failed")
                raise WriteFailed("the ticket could not be created; nothing was saved") from exc

            reply = _final_text(result_state) if isinstance(result_state, dict) else ""
            return ApprovalResponse(
                status=refreshed.status,
                ticket_id=ticket_id,
                replayed=False,
                reply=reply or None,
            )

    async def get_transcript(
        self, session_id: UUID, request_id: str, *, customer_id: UUID | None = None
    ) -> list[TranscriptMessage]:
        async with self.session_factory() as db:
            ctx = await self._context(db, session_id, request_id, customer_id)
            graph = self._graph(ctx, model=None)
            state = await graph.aget_state(
                graph_config(str(session_id), self.settings.max_tool_calls)
            )
            values = state.values or {}
            out: list[TranscriptMessage] = []
            for message in values.get("messages", []):
                if isinstance(message, HumanMessage):
                    out.append(TranscriptMessage(role="user", content=_text_of(message)))
                elif isinstance(message, AIMessage):
                    calls = [call["name"] for call in getattr(message, "tool_calls", None) or []]
                    text = _text_of(message)
                    if text or calls:
                        out.append(
                            TranscriptMessage(role="assistant", content=text, tool_calls=calls)
                        )
                else:
                    out.append(TranscriptMessage(role="tool", content=_json_preview(message)))
            return out


def _json_preview(message: Any) -> str:
    try:
        return json.dumps(json.loads(_text_of(message)), default=str)[:500]
    except (ValueError, TypeError):
        return _text_of(message)[:500]
