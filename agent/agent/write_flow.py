from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Literal

from langchain_core.messages import ToolMessage
from langgraph.graph import END
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict, ValidationError

from agent.agent.guardrails.policy import authorize
from agent.agent.state import AgentState
from agent.errors import NotFound
from agent.observability import record_tool
from agent.tools.context import ToolContext
from agent.tools.tickets_write import CreateTicketArgs, stage_create_ticket

logger = logging.getLogger(__name__)

WRITE_TOOL_NAMES = {"create_ticket"}


class ApprovalResume(BaseModel):
    """The outcome the approval endpoint has already committed.

    The write executes in the endpoint's transaction, before the graph resumes. The
    node's job on resume is to report that outcome to the model, not to decide it.
    """

    model_config = ConfigDict(extra="forbid")
    decision: Literal["approve", "reject"]
    status: Literal["executed", "rejected"]
    ticket_id: str | None = None
    replayed: bool = False
    # Set when the runtime is closing an expired approval: there is nobody to
    # reply to, so the turn ends after recording the tool result.
    silent: bool = False


def _tool_error(tool_call_id: str, error: str, message: str) -> ToolMessage:
    return ToolMessage(
        content=json.dumps({"ok": False, "error": error, "message": message}),
        tool_call_id=tool_call_id,
    )


def make_write_node(
    ctx: ToolContext, *, ttl_seconds: int
) -> Callable[[AgentState], Awaitable[Command]]:
    async def write_node(state: AgentState) -> Command:
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", []) or [])
        write_calls = [call for call in calls if call["name"] in WRITE_TOOL_NAMES]

        if len(write_calls) != 1 or len(calls) != 1:
            # Do not half-execute a mixed batch; let the model retry one action at a time.
            messages = [
                _tool_error(
                    call["id"],
                    "one_write_action_per_turn",
                    "Propose exactly one write action per turn, without other tool calls.",
                )
                for call in calls
            ]
            return Command(goto="agent", update={"messages": messages})

        call = write_calls[0]
        try:
            args = CreateTicketArgs.model_validate(call.get("args") or {})
        except ValidationError as exc:
            return Command(
                goto="agent",
                update={
                    "messages": [
                        _tool_error(call["id"], "invalid_arguments", exc.errors()[0]["msg"])
                    ]
                },
            )

        decision = authorize("create_ticket", args.model_dump())
        if not decision.allowed:
            return Command(
                goto="agent",
                update={"messages": [_tool_error(call["id"], "policy_denied", decision.reason)]},
            )

        # Staging is idempotent on the tool-call id because LangGraph re-runs this
        # node from the top when the interrupt is resumed. It commits its own short
        # transaction, so no connection is held while the human decides.
        stage_key = f"{ctx.session_id}:{call['id']}"
        try:
            staged, preview, already_staged = await stage_create_ticket(
                args,
                ctx,
                stage_key=stage_key,
                ttl_seconds=ttl_seconds,
            )
        except NotFound:
            # Ownership failures are model-visible, never an unhandled crash, and
            # never reveal whether the reference exists for someone else.
            return Command(
                goto="agent",
                update={
                    "messages": [
                        _tool_error(
                            call["id"],
                            "not_found",
                            "That order does not belong to this customer.",
                        )
                    ]
                },
            )
        if not already_staged:
            record_tool(ctx, "create_ticket", "staged")
        logger.info(
            "write_staged",
            extra={
                "extra_fields": {
                    "tool": "create_ticket",
                    "pending_action_id": str(staged.id),
                    "status": staged.status,
                }
            },
        )

        resume = interrupt(preview)
        try:
            approval = ApprovalResume.model_validate(resume)
        except ValidationError:
            return Command(
                goto="agent",
                update={
                    "messages": [
                        _tool_error(call["id"], "invalid_approval", "Approval payload was invalid.")
                    ]
                },
            )

        content = {
            "ok": approval.status == "executed",
            "status": approval.status,
            "ticket_id": approval.ticket_id,
            "replayed": approval.replayed,
        }
        record_tool(ctx, "create_ticket", approval.status)
        logger.info(
            "write_resolved",
            extra={
                "extra_fields": {
                    "pending_action_id": str(staged.id),
                    "status": approval.status,
                    "ticket_id": approval.ticket_id,
                }
            },
        )
        return Command(
            goto=END if approval.silent else "agent",
            update={
                "messages": [ToolMessage(content=json.dumps(content), tool_call_id=call["id"])],
                "pending_action_id": None,
            },
        )

    return write_node
