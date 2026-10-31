from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field

from agent.db.repositories import OrdersRepo
from agent.errors import NotFound
from agent.tools.context import ToolContext
from agent.tools.receipts import record_receipt
from agent.tools.registry import ToolSpec
from agent.tools.serialize import money, serialize_order

REFUND_WINDOW_DAYS = 30


class OrderIdArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str = Field(pattern=r"^ORD-\d{4}$", description="Order id such as ORD-1001")


class ListOrdersArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: int = Field(default=5, ge=1, le=10)
    status: str | None = Field(
        default=None,
        description="Optional status filter: processing, shipped, delivered, cancelled",
    )


async def get_order_status(args: OrderIdArgs, ctx: ToolContext) -> dict[str, object]:
    async with ctx.session() as db:
        try:
            order = await OrdersRepo(db).get_for_customer(args.order_id, ctx.customer_id)
        except NotFound:
            # A denial is itself a fact: it grounds "I couldn't find ORD-9999" without
            # revealing whether the reference exists for anyone else.
            await record_receipt(
                db,
                ctx,
                "get_order_status",
                args.model_dump(),
                {"order_id": args.order_id, "found": False},
            )
            await db.commit()
            return {
                "ok": False,
                "error": "not_found",
                "message": "No order with that reference belongs to this customer.",
            }
        payload = serialize_order(order)
        await record_receipt(db, ctx, "get_order_status", args.model_dump(), {"order": payload})
        await db.commit()
        return {"ok": True, "order": payload}


async def list_customer_orders(args: ListOrdersArgs, ctx: ToolContext) -> dict[str, object]:
    async with ctx.session() as db:
        orders = await OrdersRepo(db).list_for_customer(
            ctx.customer_id, limit=args.limit, status=args.status
        )
        payload = [serialize_order(order) for order in orders]
        await record_receipt(
            db, ctx, "list_customer_orders", args.model_dump(), {"orders": payload}
        )
        await db.commit()
        return {"ok": True, "orders": payload, "count": len(payload)}


async def check_refund_eligibility(args: OrderIdArgs, ctx: ToolContext) -> dict[str, object]:
    async with ctx.session() as db:
        try:
            order = await OrdersRepo(db).get_for_customer(args.order_id, ctx.customer_id)
        except NotFound:
            await record_receipt(
                db,
                ctx,
                "check_refund_eligibility",
                args.model_dump(),
                {"order_id": args.order_id, "found": False},
            )
            await db.commit()
            return {
                "ok": False,
                "error": "not_found",
                "message": "No order with that reference belongs to this customer.",
            }

        now = datetime.now(UTC)
        reasons: list[str] = []
        if order.status != "delivered" or order.delivered_at is None:
            reasons.append("order_not_delivered")
        window_end = (order.delivered_at or order.placed_at) + timedelta(days=REFUND_WINDOW_DAYS)
        if order.delivered_at is not None and now > window_end:
            reasons.append("outside_refund_window")

        payload = {
            "order_id": order.id,
            "eligible": not reasons,
            "reasons": reasons,
            "window_ends_at": window_end.isoformat(),
            "refund_amount_cents": order.total_cents,
            "refund_amount_display": money(order.total_cents, order.currency),
            "policy": f"Refunds accepted within {REFUND_WINDOW_DAYS} days of delivery.",
        }
        await record_receipt(
            db, ctx, "check_refund_eligibility", args.model_dump(), {"eligibility": payload}
        )
        await db.commit()
        return {"ok": True, **payload}


READ_ORDER_TOOLS = [
    ToolSpec(
        name="get_order_status",
        description=(
            "Look up the status, items and totals of one order that belongs to the current "
            "customer. Use this before describing any order."
        ),
        args_model=OrderIdArgs,
        kind="read",
        handler=get_order_status,
    ),
    ToolSpec(
        name="list_customer_orders",
        description=(
            "List the current customer's most recent orders. Use when the customer has not "
            "given an order reference."
        ),
        args_model=ListOrdersArgs,
        kind="read",
        handler=list_customer_orders,
    ),
    ToolSpec(
        name="check_refund_eligibility",
        description=(
            "Check whether an order is inside the 30-day refund window. Read-only; it does "
            "not start a refund."
        ),
        args_model=OrderIdArgs,
        kind="read",
        handler=check_refund_eligibility,
    ),
]
