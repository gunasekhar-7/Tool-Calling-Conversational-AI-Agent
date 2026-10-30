from __future__ import annotations

from datetime import datetime

from agent.db.models import Order, Ticket


def money(cents: int, currency: str = "USD") -> str:
    symbol = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}.get(currency, "")
    return f"{symbol}{cents / 100:,.2f}"


def iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def serialize_order(order: Order) -> dict[str, object]:
    return {
        "id": order.id,
        "status": order.status,
        "placed_at": iso(order.placed_at),
        "delivered_at": iso(order.delivered_at),
        "total_cents": order.total_cents,
        "total_display": money(order.total_cents, order.currency),
        "currency": order.currency,
        "items": [
            {
                "sku": item.sku,
                "qty": item.qty,
                "unit_price_cents": item.unit_price_cents,
                "unit_price_display": money(item.unit_price_cents, order.currency),
            }
            for item in order.items
        ],
    }


def serialize_ticket(ticket: Ticket) -> dict[str, object]:
    return {
        "id": ticket.id,
        "subject": ticket.subject,
        "status": ticket.status,
        "priority": ticket.priority,
        "order_id": ticket.order_id,
        "created_at": iso(ticket.created_at),
        "comments": len(ticket.comments),
    }
