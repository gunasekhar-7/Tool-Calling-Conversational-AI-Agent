from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.db.models import (
    AuditLog,
    Customer,
    Order,
    OrderItem,
    PendingAction,
    Session,
    Ticket,
    TicketComment,
    ToolReceipt,
)

CUSTOMER_A = UUID("11111111-1111-4111-8111-111111111111")
CUSTOMER_B = UUID("22222222-2222-4222-8222-222222222222")

TICKET_A = "TCK-1111AAAA"
TICKET_B = "TCK-2222BBBB"

TRUNCATE_ORDER = (
    "ticket_comments, tickets, tool_receipts, pending_actions, audit_log, "
    "sessions, order_items, orders, customers"
)


def _now() -> datetime:
    return datetime.now(UTC)


def build_customers() -> list[Customer]:
    return [
        Customer(id=CUSTOMER_A, name="Ada Lovelace", email="ada@example.com"),
        Customer(id=CUSTOMER_B, name="Grace Hopper", email="grace@example.com"),
    ]


def build_orders() -> list[Order]:
    now = _now()
    return [
        Order(
            id="ORD-1001",
            customer_id=CUSTOMER_A,
            status="delivered",
            placed_at=now - timedelta(days=14),
            delivered_at=now - timedelta(days=10),
            total_cents=12999,
            currency="USD",
        ),
        Order(
            id="ORD-1002",
            customer_id=CUSTOMER_A,
            status="shipped",
            placed_at=now - timedelta(days=2),
            total_cents=5400,
            currency="USD",
        ),
        Order(
            id="ORD-1003",
            customer_id=CUSTOMER_A,
            status="delivered",
            placed_at=now - timedelta(days=50),
            delivered_at=now - timedelta(days=45),
            total_cents=21050,
            currency="USD",
        ),
        Order(
            id="ORD-1004",
            customer_id=CUSTOMER_A,
            status="processing",
            placed_at=now - timedelta(days=1),
            total_cents=1999,
            currency="USD",
        ),
        Order(
            id="ORD-2001",
            customer_id=CUSTOMER_B,
            status="delivered",
            placed_at=now - timedelta(days=9),
            delivered_at=now - timedelta(days=5),
            total_cents=8800,
            currency="USD",
        ),
        Order(
            id="ORD-2002",
            customer_id=CUSTOMER_B,
            status="cancelled",
            placed_at=now - timedelta(days=7),
            total_cents=3500,
            currency="USD",
        ),
    ]


def build_items() -> list[OrderItem]:
    return [
        OrderItem(order_id="ORD-1001", sku="MECH-KB-01", qty=1, unit_price_cents=9999),
        OrderItem(order_id="ORD-1001", sku="CABLE-USB-C", qty=2, unit_price_cents=1500),
        OrderItem(order_id="ORD-1002", sku="MOUSE-ERG-02", qty=1, unit_price_cents=5400),
        OrderItem(order_id="ORD-1003", sku="MONITOR-27-4K", qty=1, unit_price_cents=21050),
        OrderItem(order_id="ORD-1004", sku="DESK-MAT-01", qty=1, unit_price_cents=1999),
        OrderItem(order_id="ORD-2001", sku="HEADSET-ANC", qty=1, unit_price_cents=8800),
        OrderItem(order_id="ORD-2002", sku="STAND-ALU", qty=1, unit_price_cents=3500),
    ]


def build_tickets() -> list[Ticket]:
    now = _now()
    return [
        Ticket(
            id=TICKET_A,
            customer_id=CUSTOMER_A,
            order_id="ORD-1002",
            subject="Where is my mouse?",
            body="The tracking page has not moved in two days.",
            priority="high",
            status="open",
            created_at=now - timedelta(days=1),
        ),
        Ticket(
            id=TICKET_B,
            customer_id=CUSTOMER_B,
            order_id=None,
            subject="Invoice copy",
            body="Please send a copy of the invoice.",
            priority="low",
            status="resolved",
            created_at=now - timedelta(days=12),
        ),
    ]


def build_comments() -> list[TicketComment]:
    return [
        TicketComment(
            ticket_id=TICKET_A,
            author="agent:demo",
            body="Carrier shows the parcel at the regional hub.",
        )
    ]


async def reset_and_seed(db: AsyncSession) -> None:
    await db.execute(
        __import__("sqlalchemy").text(f"TRUNCATE {TRUNCATE_ORDER} RESTART IDENTITY CASCADE")
    )
    db.add_all(build_customers())
    db.add_all(build_orders())
    await db.flush()
    db.add_all(build_items())
    db.add_all(build_tickets())
    await db.flush()
    db.add_all(build_comments())
    await db.flush()


async def seed_if_empty(db: AsyncSession) -> bool:
    count = await db.scalar(select(func.count()).select_from(Customer))
    if count:
        return False
    db.add_all(build_customers())
    db.add_all(build_orders())
    await db.flush()
    db.add_all(build_items())
    db.add_all(build_tickets())
    await db.flush()
    db.add_all(build_comments())
    await db.flush()
    return True


async def clear_runtime_state(db: AsyncSession) -> None:
    """Remove per-test rows without touching the seed data."""
    for model in (TicketComment, Ticket, ToolReceipt, PendingAction, AuditLog, Session):
        await db.execute(delete(model))
    await db.flush()
