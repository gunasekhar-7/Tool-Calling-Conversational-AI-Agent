from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from agent.db.repositories import OrdersRepo, PendingActionRepo, hash_args
from agent.tools.context import ToolContext
from agent.tools.registry import ToolSpec


class CreateTicketArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str | None = Field(default=None, pattern=r"^ORD-\d{4}$")
    subject: str = Field(min_length=3, max_length=120)
    body: str = Field(min_length=3, max_length=2000)
    priority: Literal["low", "normal", "high"] = "normal"


@dataclass(frozen=True)
class StagedAction:
    """A detached view of the staged row.

    The staging transaction is committed and closed before the graph pauses, so the
    graph must not carry a live ORM object (or a database connection) into the wait.
    """

    id: UUID
    status: str
    args_hash: str
    expires_at: datetime
    result: dict[str, Any] | None


async def stage_create_ticket(
    args: CreateTicketArgs,
    ctx: ToolContext,
    *,
    stage_key: str,
    ttl_seconds: int,
) -> tuple[StagedAction, dict[str, Any], bool]:
    """Stage the proposal and commit it, ready for the approval card.

    Returns the detached action, the preview shown to the human, and whether this
    stage key already existed (the graph replays the node after ``interrupt``).
    """

    async with ctx.session() as db:
        already_staged = await PendingActionRepo(db).exists_for_stage_key(stage_key)
        if args.order_id:
            # Ownership is checked here, not trusted from the model. A failure raises
            # before anything is written.
            await OrdersRepo(db).get_for_customer(args.order_id, ctx.customer_id)

        payload = args.model_dump()
        action = await PendingActionRepo(db).stage(
            session_id=ctx.session_id,
            customer_id=ctx.customer_id,
            tool_name="create_ticket",
            args=payload,
            ttl_seconds=ttl_seconds,
            stage_key=stage_key,
        )
        staged = StagedAction(
            id=action.id,
            status=action.status,
            args_hash=action.args_hash,
            expires_at=action.expires_at,
            result=dict(action.result) if action.result else None,
        )
        preview = {
            "action": "create_ticket",
            "order_id": args.order_id,
            "subject": args.subject,
            "body": args.body,
            "priority": args.priority,
            "preview_hash": staged.args_hash,
            "pending_action_id": str(staged.id),
            "expires_at": staged.expires_at.isoformat(),
        }
        await db.commit()
    return staged, preview, already_staged


async def create_ticket_stub(args: CreateTicketArgs, ctx: ToolContext) -> dict[str, Any]:
    """Registered handler. Routing sends write calls to the approval node instead."""
    return {
        "ok": False,
        "error": "approval_required",
        "message": "Ticket creation must be confirmed by a human before it is executed.",
    }


WRITE_TICKET_TOOLS = [
    ToolSpec(
        name="create_ticket",
        description=(
            "Propose a support ticket for the current customer. This never creates the "
            "ticket immediately: a human must approve the exact details first."
        ),
        args_model=CreateTicketArgs,
        kind="write",
        handler=create_ticket_stub,
    )
]


def preview_hash(args: dict[str, Any]) -> str:
    return hash_args(args)
