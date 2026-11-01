from __future__ import annotations

import hmac
import json
import logging
import time
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
        metrics: Any | None = None,
    ) -> None:
        self.settings = settings
        self.model = model
        self.checkpointer = checkpointer
        self.session_factory = session_factory
        self.metrics = metrics

    def _record_turn(self, started: float, outcome: str) -> None:
        if self.metrics is None:
            return
        self.metrics.turn_seconds.labels(outcome=outcome).observe(time.perf_counter() - started)

    async def ensure_session(self, session_id: UUID, *, customer_id: UUID | None = None) -> None:
        """Resolve the session and, when a caller is given, enforce ownership.

        A session that belongs to someone else is reported as missing, so the API
        never confirms that another customer's session id exists.
        """
        async with self.session_factory() as db:
            session = await SessionRepo(db).get(session_id)
            if customer_id is not None and session.customer_id != customer_id:
                raise NotFound("session not found")

    async def settle_approval_state(
        self, session_id: UUID, request_id: str, *, customer_id: UUID | None = None
    ) -> int | None:
        """Close a paused approval that can no longer be decided.

        Returns the number of messages already in the checkpoint, so the caller can
        tell old tool calls apart from the ones its own turn produces, or ``None`` when
        no write was ever staged and there is no interrupt to look for.

        A live proposal still raises ``ApprovalPending``. An expired one is resumed as
        a rejection with ``silent`` set: the write node records a tool result and ends,
        so the provider never receives an assistant tool call without an answer. This
        has to run before the SSE response starts, otherwise the 409 would arrive as a
        200 stream.
        """
        owner, sid = await self._resolve(session_id, customer_id)
        async with self.session_factory() as db:
            action = await PendingActionRepo(db).find_latest(sid)
        if action is None:
            # No write was staged in this session, so no interrupt can exist. Read-only
            # conversations must not pay for a second graph build on every turn.
            return None
        ctx = self._context(owner, sid, request_id)
        graph = self._graph(ctx, model=None)
        config = graph_config(str(sid), self.settings.max_tool_calls)
        state = await self._settle_stale_approval(graph, config, sid)
        return len(state.values.get("messages", [])) if state.values else 0

    async def _settle_stale_approval(self, graph: Any, config: dict[str, Any], sid: UUID) -> Any:
        """Resume an interrupt whose approval is no longer decidable.

        The row is the source of truth for what happened: executed approvals replay the
        ticket id, everything else becomes a rejection. Settlement is silent because
        the user is not waiting on this turn; generating a reply would spend a model
        call nobody reads.
        """
        state = await graph.aget_state(config)
        if not state.interrupts:
            return state

        async with self.session_factory() as db:
            action = await PendingActionRepo(db).find_latest(sid)
            if action is not None and action.status == "proposed":
                moment = datetime.now(UTC)
                if action.expires_at > moment:
                    raise ApprovalPending(
                        "a staged action is awaiting your decision; approve or reject it first"
                    )
                action.status = "expired"
                action.decided_at = moment
            status = action.status if action is not None else "expired"
            ticket_id = (action.result or {}).get("ticket_id") if action is not None else None
            await db.commit()

        if status == "executed":
            resume = {
                "decision": "approve",
                "status": "executed",
                "ticket_id": ticket_id,
                "replayed": True,
                "silent": True,
            }
        else:
            resume = {
                "decision": "reject",
                "status": "rejected",
                "ticket_id": None,
                "replayed": False,
                "silent": True,
            }

        try:
            await graph.ainvoke(Command(resume=resume), config=config, durability="sync")
        except Exception as exc:
            # A concurrent settlement can win the race; that is success, not failure.
            state = await graph.aget_state(config)
            if not state.interrupts:
                return state
            logger.exception("stale_approval_close_failed")
            raise Conflict("a previous approval is still open; start a new session") from exc
        return await graph.aget_state(config)

    async def _resolve(
        self, session_id: UUID, customer_id: UUID | None = None
    ) -> tuple[UUID, UUID]:
        """Resolve a session and its owner using a connection only for the lookup."""
        async with self.session_factory() as db:
            session = await SessionRepo(db).get(session_id)
            if customer_id is not None and session.customer_id != customer_id:
                # Same answer as a missing session: the API never confirms that another
                # customer's conversation exists.
                raise NotFound("session not found")
            return session.customer_id, session.id

    def _context(self, customer_id: UUID, session_id: UUID, request_id: str) -> ToolContext:
        return ToolContext(
            customer_id=customer_id,
            session_id=session_id,
            session_factory=self.session_factory,
            request_id=request_id,
            metrics=self.metrics,
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
        prior_count: int | None = None,
    ) -> AsyncIterator[TurnEvent]:
        started = time.perf_counter()
        # Ownership is answered before capability: a foreign session is 404 even
        # when the service has no model configured.
        owner, sid = await self._resolve(session_id, customer_id)
        if self.model is None:
            raise ModelUnavailable("model is not configured")
        ctx = self._context(owner, sid, request_id)
        graph = self._graph(ctx)
        config = graph_config(str(session_id), self.settings.max_tool_calls)

        if prior_count is None:
            # Direct callers (tests, scripts) do not go through the route preflight.
            snapshot = await graph.aget_state(config)
            prior_count = len(snapshot.values.get("messages", [])) if snapshot.values else 0

        emitted_tools: list[str] = []
        streamed = ""
        try:
            async for part in graph.astream(
                {
                    "messages": [HumanMessage(content=message)],
                    "customer_id": str(ctx.customer_id),
                    # The budget is per turn. `tool_calls` is checkpointed state, so
                    # leaving it alone would spend the allowance once across the
                    # whole conversation and silently disable tools on later turns.
                    "tool_calls": 0,
                    "limit_reached": False,
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

            # Only this turn's messages: a state-wide scan would replay every
            # historical tool call on every later turn.
            for message_obj in values.get("messages", [])[prior_count:]:
                for call in getattr(message_obj, "tool_calls", None) or []:
                    name = call.get("name")
                    if name and name not in emitted_tools:
                        emitted_tools.append(name)
                        yield TurnEvent("tool_call", {"name": name})

            if state.interrupts:
                payload = state.interrupts[0].value
                # The write node committed the staged row in its own short
                # transaction, so there is nothing left to save before the card.
                self._record_turn(started, "awaiting_approval")
                yield TurnEvent("confirmation_required", dict(payload))
                return

            answer = _final_text(values) or streamed
            async with self.session_factory() as db:
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

            self._record_turn(started, "done")
            yield TurnEvent("done", {"text": answer, "tool_calls": emitted_tools})
        except Exception as exc:
            # No turn-wide transaction exists to roll back: tools commit their own
            # short transactions, and a staged write is already durable by design.
            self._record_turn(started, "error")
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
        owner, sid = await self._resolve(session_id, customer_id)
        if self.model is None:
            raise ModelUnavailable("model is not configured")
        ctx = self._context(owner, sid, request_id)

        # Replay is answered without touching the graph: after the first approval the
        # interrupt is gone, but a retried request must still return its ticket.
        async with self.session_factory() as db:
            peek = await PendingActionRepo(db).get_for_session(pending_action_id, sid)
            if peek.status == "executed":
                if peek.consume_key == idempotency_key:
                    return ApprovalResponse(
                        status="executed",
                        ticket_id=(peek.result or {}).get("ticket_id"),
                        replayed=True,
                    )
                raise Conflict("this action was already executed")
            if peek.status == "rejected":
                raise Conflict("this action was rejected")
            if peek.status == "expired":
                raise ApprovalExpired("this approval has expired")

        graph = self._graph(ctx)
        config = graph_config(str(sid), self.settings.max_tool_calls)
        snapshot = await graph.aget_state(config)
        if not snapshot.interrupts:
            raise Conflict("no approval is pending for this session")

        async with self.session_factory() as db:
            action = await PendingActionRepo(db).get_for_session(pending_action_id, sid)
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

            # The single write transaction for this request. The ticket, the terminal
            # state and the audit row commit together, before the graph is resumed.
            # No connection is held while the model writes its reply.
            try:
                result = await PendingActionRepo(db).consume(
                    pending_id=pending_action_id,
                    session_id=sid,
                    decision=decision,
                    consume_key=idempotency_key,
                    preview_hash=preview_hash,
                )
                await AuditRepo(db).record(
                    session_id=sid,
                    actor=f"customer:{owner}",
                    action=f"approval.{decision}",
                    target=str(pending_action_id),
                    detail={"ticket_id": result.ticket_id, "status": result.status},
                )
                await db.commit()
                if self.metrics is not None:
                    self.metrics.observe_approval(decision)
            except DomainError:
                await db.rollback()
                raise
            except Exception as exc:
                # A failure after the INSERT must not leave a ticket behind, and the
                # caller must be able to tell that nothing was created.
                await db.rollback()
                logger.exception("approval_write_failed")
                raise WriteFailed("the ticket could not be created; nothing was saved") from exc

        resume = {
            "decision": decision,
            "status": result.status,
            "ticket_id": result.ticket_id,
            "replayed": result.replayed,
        }
        try:
            result_state = await graph.ainvoke(
                Command(resume=resume),
                config=config,
                durability="sync",
            )
        except Exception:
            # The write is already committed; reporting failure would invite the
            # client to retry a decision that succeeded.
            logger.exception("approval_reply_failed")
            return ApprovalResponse(
                status=result.status,
                ticket_id=result.ticket_id,
                replayed=result.replayed,
                reply=None,
            )

        reply = _final_text(result_state) if isinstance(result_state, dict) else ""
        return ApprovalResponse(
            status=result.status,
            ticket_id=result.ticket_id,
            replayed=result.replayed,
            reply=reply or None,
        )

    async def get_transcript(
        self, session_id: UUID, request_id: str, *, customer_id: UUID | None = None
    ) -> list[TranscriptMessage]:
        owner, sid = await self._resolve(session_id, customer_id)
        ctx = self._context(owner, sid, request_id)
        graph = self._graph(ctx, model=None)
        state = await graph.aget_state(graph_config(str(sid), self.settings.max_tool_calls))
        values = state.values or {}
        out: list[TranscriptMessage] = []
        for message in values.get("messages", []):
            if isinstance(message, HumanMessage):
                out.append(TranscriptMessage(role="user", content=_text_of(message)))
            elif isinstance(message, AIMessage):
                calls = [call["name"] for call in getattr(message, "tool_calls", None) or []]
                text = _text_of(message)
                if text or calls:
                    out.append(TranscriptMessage(role="assistant", content=text, tool_calls=calls))
            else:
                out.append(TranscriptMessage(role="tool", content=_json_preview(message)))
        return out


def _json_preview(message: Any) -> str:
    try:
        return json.dumps(json.loads(_text_of(message)), default=str)[:500]
    except (ValueError, TypeError):
        return _text_of(message)[:500]
