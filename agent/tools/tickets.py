from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent.db.repositories import TicketsRepo
from agent.errors import NotFound
from agent.tools.context import ToolContext
from agent.tools.receipts import record_receipt
from agent.tools.registry import ToolSpec
from agent.tools.serialize import serialize_ticket


class TicketIdArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticket_id: str = Field(
        pattern=r"^TCK-[0-9A-F]{8}$", description="Ticket id such as TCK-1A2B3C4D"
    )


async def get_ticket(args: TicketIdArgs, ctx: ToolContext) -> dict[str, object]:
    async with ctx.session() as db:
        try:
            ticket = await TicketsRepo(db).get_for_customer(args.ticket_id, ctx.customer_id)
        except NotFound:
            # Record the observed absence so a grounded reply may name the reference.
            await record_receipt(
                db,
                ctx,
                "get_ticket",
                args.model_dump(),
                {"ticket_id": args.ticket_id, "found": False},
            )
            await db.commit()
            return {
                "ok": False,
                "error": "not_found",
                "message": "No ticket with that reference belongs to this customer.",
            }
        payload = serialize_ticket(ticket)
        await record_receipt(db, ctx, "get_ticket", args.model_dump(), {"ticket": payload})
        await db.commit()
        return {"ok": True, "ticket": payload}


READ_TICKET_TOOLS = [
    ToolSpec(
        name="get_ticket",
        description="Fetch one support ticket that belongs to the current customer.",
        args_model=TicketIdArgs,
        kind="read",
        handler=get_ticket,
    )
]
