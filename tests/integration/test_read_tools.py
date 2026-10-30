from __future__ import annotations

import json

import pytest

from agent.db.repositories import ReceiptRepo, SessionRepo
from agent.db.seed import CUSTOMER_A, CUSTOMER_B
from agent.tools.context import ToolContext
from agent.tools.orders import (
    ListOrdersArgs,
    OrderIdArgs,
    check_refund_eligibility,
    get_order_status,
    list_customer_orders,
)
from agent.tools.tickets import TicketIdArgs, get_ticket

pytestmark = pytest.mark.integration


async def _ctx(db, customer_id=CUSTOMER_A) -> ToolContext:
    session = await SessionRepo(db).create(customer_id)
    await db.flush()
    return ToolContext(
        customer_id=session.customer_id,
        session_id=session.id,
        db=db,
        request_id="test-request",
    )


async def test_own_order_returns_serialized_payload(db) -> None:
    ctx = await _ctx(db)
    result = await get_order_status(OrderIdArgs(order_id="ORD-1001"), ctx)
    assert result["ok"] is True
    order = result["order"]
    assert order["id"] == "ORD-1001"
    assert order["total_display"] == "$129.99"
    assert len(order["items"]) == 2


async def test_foreign_order_returns_not_found_without_data(db) -> None:
    ctx = await _ctx(db, CUSTOMER_A)
    result = await get_order_status(OrderIdArgs(order_id="ORD-2001"), ctx)
    assert result == {
        "ok": False,
        "error": "not_found",
        "message": "No order with that reference belongs to this customer.",
    }
    assert "order" not in result


async def test_receipt_is_written_for_successful_read(db) -> None:
    ctx = await _ctx(db)
    await get_order_status(OrderIdArgs(order_id="ORD-1001"), ctx)
    receipts = await ReceiptRepo(db).list_for_session(ctx.session_id)
    assert len(receipts) == 1
    assert receipts[0].tool_name == "get_order_status"
    assert receipts[0].facts["order"]["id"] == "ORD-1001"


async def test_denied_read_receipts_only_the_denial(db) -> None:
    """The denial is what grounds "I couldn't find ORD-2001"; nothing else may leak."""
    ctx = await _ctx(db)
    await get_order_status(OrderIdArgs(order_id="ORD-2001"), ctx)
    receipts = await ReceiptRepo(db).list_for_session(ctx.session_id)
    assert len(receipts) == 1
    assert receipts[0].facts == {"order_id": "ORD-2001", "found": False}
    assert "2001" in json.dumps(receipts[0].facts)
    assert "total_cents" not in json.dumps(receipts[0].facts)


async def test_list_orders_is_scoped(db) -> None:
    ctx = await _ctx(db, CUSTOMER_B)
    result = await list_customer_orders(ListOrdersArgs(limit=10), ctx)
    ids = {order["id"] for order in result["orders"]}
    assert ids == {"ORD-2001", "ORD-2002"}


async def test_refund_eligibility_inside_window(db) -> None:
    ctx = await _ctx(db)
    result = await check_refund_eligibility(OrderIdArgs(order_id="ORD-1001"), ctx)
    assert result["eligible"] is True
    assert result["refund_amount_cents"] == 12999


async def test_refund_eligibility_outside_window(db) -> None:
    ctx = await _ctx(db)
    result = await check_refund_eligibility(OrderIdArgs(order_id="ORD-1003"), ctx)
    assert result["eligible"] is False
    assert "outside_refund_window" in result["reasons"]


async def test_ticket_lookup_is_scoped(db) -> None:
    ctx = await _ctx(db, CUSTOMER_A)
    own = await get_ticket(TicketIdArgs(ticket_id="TCK-1111AAAA"), ctx)
    assert own["ok"] is True
    assert own["ticket"]["id"] == "TCK-1111AAAA"

    other = await get_ticket(TicketIdArgs(ticket_id="TCK-2222BBBB"), ctx)
    assert other["ok"] is False
    assert other["error"] == "not_found"
