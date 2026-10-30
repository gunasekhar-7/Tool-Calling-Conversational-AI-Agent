from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.db.models import PendingAction
from agent.db.repositories import OrdersRepo, PendingActionRepo, hash_args
from agent.tools.context import ToolContext
from agent.tools.registry import ToolSpec


class CreateTicketArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str | None = Field(default=None, pattern=r"^ORD-\d{4}$")
    subject: str = Field(min_length=3, max_length=120)
    body: str = Field(min_length=3, max_length=2000)
    priority: Literal["low", "normal", "high"] = "normal"


async def stage_create_ticket(
    args: CreateTicketArgs,
    ctx: ToolContext,
    *,
    stage_key: str,
    ttl_seconds: int,
) -> tuple[PendingAction, dict[str, Any]]:
    if args.order_id:
        # Ownership is checked here, not trusted from the model.
        await OrdersRepo(ctx.db).get_for_customer(args.order_id, ctx.customer_id)

    payload = args.model_dump()
    action = await PendingActionRepo(ctx.db).stage(
        session_id=ctx.session_id,
        customer_id=ctx.customer_id,
        tool_name="create_ticket",
        args=payload,
        ttl_seconds=ttl_seconds,
        stage_key=stage_key,
    )
    preview = {
        "action": "create_ticket",
        "order_id": args.order_id,
        "subject": args.subject,
        "body": args.body,
        "priority": args.priority,
        "preview_hash": action.args_hash,
        "pending_action_id": str(action.id),
        "expires_at": action.expires_at.isoformat(),
    }
    return action, preview


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
