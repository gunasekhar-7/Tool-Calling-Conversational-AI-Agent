from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from agent.db.models import (
    AuditLog,
    Customer,
    Order,
    PendingAction,
    Session,
    Ticket,
    ToolReceipt,
)
from agent.errors import ApprovalExpired, ApprovalMismatch, Conflict, NotFound


def hash_args(args: dict[str, Any]) -> str:
    payload = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    ticket_id: str | None
    replayed: bool = False


class CustomerRepo:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, customer_id: uuid.UUID) -> Customer:
        customer = await self.session.get(Customer, customer_id)
        if customer is None:
            raise NotFound("customer not found")
        return customer

    async def list_all(self) -> list[Customer]:
        result = await self.session.scalars(select(Customer).order_by(Customer.name))
        return list(result)


class SessionRepo:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, customer_id: uuid.UUID) -> Session:
        await CustomerRepo(self.session).get(customer_id)
        row = Session(id=uuid.uuid4(), customer_id=customer_id)
        self.session.add(row)
        await self.session.flush()
        return row

    async def get(self, session_id: uuid.UUID) -> Session:
        row = await self.session.get(Session, session_id)
        if row is None:
            raise NotFound("session not found")
        return row

    async def touch(self, session_id: uuid.UUID) -> None:
        row = await self.get(session_id)
        row.last_seen_at = datetime.now(UTC)
        await self.session.flush()


class OrdersRepo:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_for_customer(self, order_id: str, customer_id: uuid.UUID) -> Order:
        order = await self.session.scalar(
            select(Order).where(Order.id == order_id, Order.customer_id == customer_id)
        )
        if order is None:
            raise NotFound(f"order {order_id} not found")
        return order

    async def list_for_customer(
        self,
        customer_id: uuid.UUID,
        limit: int = 5,
        status: str | None = None,
    ) -> list[Order]:
        limit = max(1, min(limit, 10))
        stmt = select(Order).where(Order.customer_id == customer_id)
        if status:
            stmt = stmt.where(Order.status == status)
        stmt = stmt.order_by(Order.placed_at.desc()).limit(limit)
        result = await self.session.scalars(stmt)
        return list(result)


class TicketsRepo:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        customer_id: uuid.UUID,
        order_id: str | None,
        subject: str,
        body: str,
        priority: str,
        pending_action_id: uuid.UUID,
    ) -> Ticket:
        ticket = Ticket(
            id=new_ticket_id(),
            customer_id=customer_id,
            order_id=order_id,
            subject=subject,
            body=body,
            priority=priority,
            status="open",
            pending_action_id=pending_action_id,
        )
        self.session.add(ticket)
        await self.session.flush()
        return ticket

    async def get_for_customer(self, ticket_id: str, customer_id: uuid.UUID) -> Ticket:
        ticket = await self.session.scalar(
            select(Ticket).where(Ticket.id == ticket_id, Ticket.customer_id == customer_id)
        )
        if ticket is None:
            raise NotFound(f"ticket {ticket_id} not found")
        return ticket

    async def list_for_customer(self, customer_id: uuid.UUID, limit: int = 5) -> list[Ticket]:
        result = await self.session.scalars(
            select(Ticket)
            .where(Ticket.customer_id == customer_id)
            .order_by(Ticket.created_at.desc())
            .limit(max(1, min(limit, 10)))
        )
        return list(result)


class PendingActionRepo:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def stage(
        self,
        *,
        session_id: uuid.UUID,
        customer_id: uuid.UUID,
        tool_name: str,
        args: dict[str, Any],
        ttl_seconds: int,
        stage_key: str,
    ) -> PendingAction:
        """Get-or-create by ``stage_key``.

        Safe to call again when LangGraph replays a node after ``interrupt()``.
        """
        existing = await self.session.scalar(
            select(PendingAction).where(PendingAction.stage_key == stage_key)
        )
        if existing is not None:
            return existing

        action = PendingAction(
            id=uuid.uuid4(),
            session_id=session_id,
            customer_id=customer_id,
            tool_name=tool_name,
            args=args,
            args_hash=hash_args(args),
            status="proposed",
            expires_at=datetime.now(UTC) + timedelta(seconds=ttl_seconds),
            stage_key=stage_key,
        )
        try:
            async with self.session.begin_nested():
                self.session.add(action)
        except IntegrityError:
            existing = await self.session.scalar(
                select(PendingAction).where(PendingAction.stage_key == stage_key)
            )
            if existing is None:
                raise
            return existing
        return action

    async def find_open(
        self, session_id: uuid.UUID, now: datetime | None = None
    ) -> PendingAction | None:
        """The live staged write for a session, if one is still awaiting a human."""
        moment = now or datetime.now(UTC)
        return await self.session.scalar(
            select(PendingAction)
            .where(
                PendingAction.session_id == session_id,
                PendingAction.status == "proposed",
                PendingAction.expires_at > moment,
            )
            .order_by(PendingAction.created_at.desc())
            .limit(1)
        )

    async def get_for_session(self, pending_id: uuid.UUID, session_id: uuid.UUID) -> PendingAction:
        action = await self.session.scalar(
            select(PendingAction).where(
                PendingAction.id == pending_id,
                PendingAction.session_id == session_id,
            )
        )
        if action is None:
            raise NotFound("pending action not found")
        return action

    async def consume(
        self,
        *,
        pending_id: uuid.UUID,
        session_id: uuid.UUID,
        decision: str,
        consume_key: str,
        preview_hash: str,
        now: datetime | None = None,
    ) -> ExecutionResult:
        """Authorize a staged write exactly once.

        The row is locked, re-validated against the approved preview and only then
        mutated, all inside one transaction. Concurrent approvals serialize on the lock.
        """
        moment = now or datetime.now(UTC)
        action = await self.session.scalar(
            select(PendingAction)
            .where(
                PendingAction.id == pending_id,
                PendingAction.session_id == session_id,
            )
            .with_for_update()
        )
        if action is None:
            raise NotFound("pending action not found")

        if action.status == "executed":
            if action.consume_key == consume_key:
                ticket_id = (action.result or {}).get("ticket_id")
                return ExecutionResult(status="executed", ticket_id=ticket_id, replayed=True)
            raise Conflict("this action was already executed")

        if action.status == "rejected":
            raise Conflict("this action was rejected")

        if action.status == "expired" or action.expires_at <= moment:
            if action.status != "expired":
                action.status = "expired"
                action.decided_at = moment
                await self.session.flush()
            raise ApprovalExpired("this approval has expired")

        if not hmac.compare_digest(preview_hash, action.args_hash):
            raise ApprovalMismatch("approval does not match the staged action")

        if decision == "reject":
            action.status = "rejected"
            action.decided_at = moment
            action.consume_key = consume_key
            await self.session.flush()
            return ExecutionResult(status="rejected", ticket_id=None)

        # The write and the state transition share one transaction that this method
        # flushes but does not commit: the request boundary commits once, so a failure
        # after the INSERT cannot leave a ticket behind with the action still pending.
        try:
            async with self.session.begin_nested():
                ticket = await TicketsRepo(self.session).create(
                    customer_id=action.customer_id,
                    order_id=action.args.get("order_id"),
                    subject=action.args["subject"],
                    body=action.args["body"],
                    priority=action.args.get("priority", "normal"),
                    pending_action_id=action.id,
                )
                action.status = "executed"
                action.decided_at = moment
                action.consume_key = consume_key
                action.result = {"ticket_id": ticket.id}
        except IntegrityError as exc:
            raise Conflict("this action was already executed") from exc
        return ExecutionResult(status="executed", ticket_id=ticket.id)


class ReceiptRepo:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(
        self,
        *,
        session_id: uuid.UUID,
        tool_name: str,
        args: dict[str, Any],
        facts: dict[str, Any],
    ) -> ToolReceipt:
        receipt = ToolReceipt(
            id=uuid.uuid4(),
            session_id=session_id,
            tool_name=tool_name,
            args_hash=hash_args(args),
            facts=facts,
        )
        self.session.add(receipt)
        await self.session.flush()
        return receipt

    async def list_for_session(self, session_id: uuid.UUID) -> list[ToolReceipt]:
        result = await self.session.scalars(
            select(ToolReceipt)
            .where(ToolReceipt.session_id == session_id)
            .order_by(ToolReceipt.created_at.desc())
            .limit(200)
        )
        return list(result)


class AuditRepo:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def record(
        self,
        *,
        session_id: uuid.UUID,
        actor: str,
        action: str,
        target: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.session.add(
            AuditLog(
                id=uuid.uuid4(),
                session_id=session_id,
                actor=actor,
                action=action,
                target=target,
                detail=detail,
            )
        )
        await self.session.flush()


def new_ticket_id() -> str:
    return f"TCK-{uuid.uuid4().hex[:8].upper()}"
