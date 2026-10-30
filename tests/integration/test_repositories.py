from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy import func, select

from agent.db.models import PendingAction, Session, Ticket
from agent.db.repositories import (
    OrdersRepo,
    PendingActionRepo,
    SessionRepo,
    TicketsRepo,
    hash_args,
)
from agent.db.seed import CUSTOMER_A, CUSTOMER_B
from agent.errors import ApprovalExpired, ApprovalMismatch, Conflict, NotFound

pytestmark = pytest.mark.integration


async def _session(db, customer_id: UUID = CUSTOMER_A) -> Session:
    row = await SessionRepo(db).create(customer_id)
    await db.commit()
    return row


async def _stage(db, session_id: UUID, key: str = "stage-1", ttl: int = 900) -> PendingAction:
    args = {
        "order_id": "ORD-1001",
        "subject": "Damaged cable",
        "body": "The USB-C cable arrived bent.",
        "priority": "high",
    }
    return await PendingActionRepo(db).stage(
        session_id=session_id,
        customer_id=CUSTOMER_A,
        tool_name="create_ticket",
        args=args,
        ttl_seconds=ttl,
        stage_key=key,
    )


async def test_foreign_order_raises_not_found(db) -> None:
    with pytest.raises(NotFound):
        await OrdersRepo(db).get_for_customer("ORD-2001", CUSTOMER_A)


async def test_list_orders_respects_limit(db) -> None:
    orders = await OrdersRepo(db).list_for_customer(CUSTOMER_A, limit=50)
    assert len(orders) == 4


async def test_seed_is_idempotent(db) -> None:
    from agent.db.seed import reset_and_seed

    await reset_and_seed(db)
    await db.commit()
    count = await db.scalar(select(func.count()).select_from(Ticket))
    assert count == 2


async def test_stage_is_idempotent_by_stage_key(db) -> None:
    session = await _session(db)
    first = await _stage(db, session.id, key="same")
    second = await _stage(db, session.id, key="same")
    await db.commit()
    assert first.id == second.id
    total = await db.scalar(
        select(func.count()).select_from(PendingAction).where(PendingAction.stage_key == "same")
    )
    assert total == 1


async def test_approve_creates_exactly_one_ticket(db) -> None:
    session = await _session(db)
    action = await _stage(db, session.id)
    await db.commit()

    result = await PendingActionRepo(db).consume(
        pending_id=action.id,
        session_id=session.id,
        decision="approve",
        consume_key="approval-1",
        preview_hash=action.args_hash,
    )
    assert result.status == "executed"
    assert result.ticket_id and result.ticket_id.startswith("TCK-")
    count = await db.scalar(select(func.count()).select_from(Ticket))
    assert count == 3


async def test_repeat_approval_key_replays_without_second_ticket(db) -> None:
    session = await _session(db)
    action = await _stage(db, session.id)
    await db.commit()

    first = await PendingActionRepo(db).consume(
        pending_id=action.id,
        session_id=session.id,
        decision="approve",
        consume_key="same-key",
        preview_hash=action.args_hash,
    )
    second = await PendingActionRepo(db).consume(
        pending_id=action.id,
        session_id=session.id,
        decision="approve",
        consume_key="same-key",
        preview_hash=action.args_hash,
    )
    assert first.ticket_id == second.ticket_id
    assert second.replayed is True
    assert await db.scalar(select(func.count()).select_from(Ticket)) == 3


async def test_second_approval_with_new_key_is_rejected(db) -> None:
    session = await _session(db)
    action = await _stage(db, session.id)
    await db.commit()
    await PendingActionRepo(db).consume(
        pending_id=action.id,
        session_id=session.id,
        decision="approve",
        consume_key="key-one",
        preview_hash=action.args_hash,
    )
    with pytest.raises(Conflict):
        await PendingActionRepo(db).consume(
            pending_id=action.id,
            session_id=session.id,
            decision="approve",
            consume_key="key-two",
            preview_hash=action.args_hash,
        )


async def test_reject_creates_no_ticket(db) -> None:
    session = await _session(db)
    action = await _stage(db, session.id)
    await db.commit()
    result = await PendingActionRepo(db).consume(
        pending_id=action.id,
        session_id=session.id,
        decision="reject",
        consume_key="reject-1",
        preview_hash=action.args_hash,
    )
    assert result.status == "rejected"
    assert result.ticket_id is None
    assert await db.scalar(select(func.count()).select_from(Ticket)) == 2


async def test_expired_approval_is_refused(db) -> None:
    session = await _session(db)
    action = await _stage(db, session.id)
    action.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    await db.commit()

    with pytest.raises(ApprovalExpired):
        await PendingActionRepo(db).consume(
            pending_id=action.id,
            session_id=session.id,
            decision="approve",
            consume_key="late-key",
            preview_hash=action.args_hash,
        )
    await db.refresh(action)
    assert action.status == "expired"


async def test_changed_preview_hash_is_refused(db) -> None:
    session = await _session(db)
    action = await _stage(db, session.id)
    await db.commit()
    tampered = {"subject": "Something else"}
    with pytest.raises(ApprovalMismatch):
        await PendingActionRepo(db).consume(
            pending_id=action.id,
            session_id=session.id,
            decision="approve",
            consume_key="tamper-key",
            preview_hash=hash_args(tampered),
        )
    assert await db.scalar(select(func.count()).select_from(Ticket)) == 2


async def test_pending_action_is_scoped_to_session(db) -> None:
    session = await _session(db)
    other = await SessionRepo(db).create(CUSTOMER_A)
    await db.commit()
    action = await _stage(db, session.id)
    await db.commit()

    with pytest.raises(NotFound):
        await PendingActionRepo(db).consume(
            pending_id=action.id,
            session_id=other.id,
            decision="approve",
            consume_key="cross-key",
            preview_hash=action.args_hash,
        )


async def test_concurrent_approvals_create_one_ticket(factory) -> None:
    async with factory() as setup:
        session = await _session(setup)
        action = await _stage(setup, session.id, key=f"concurrent-{uuid.uuid4()}")
        await setup.commit()
        pending_id, session_id, args_hash = action.id, session.id, action.args_hash

    async def approve(key: str) -> str:
        async with factory() as own:
            result = await PendingActionRepo(own).consume(
                pending_id=pending_id,
                session_id=session_id,
                decision="approve",
                consume_key=key,
                preview_hash=args_hash,
            )
            return result.ticket_id or ""

    first, second = await asyncio.gather(approve("concurrent-key"), approve("concurrent-key"))
    assert first == second
    async with factory() as check:
        assert await check.scalar(select(func.count()).select_from(Ticket)) == 3


async def test_ticket_is_unique_per_pending_action(db) -> None:
    session = await _session(db)
    action = await _stage(db, session.id)
    await db.commit()
    await PendingActionRepo(db).consume(
        pending_id=action.id,
        session_id=session.id,
        decision="approve",
        consume_key="unique-key",
        preview_hash=action.args_hash,
    )
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError):
        await TicketsRepo(db).create(
            customer_id=CUSTOMER_B,
            order_id=None,
            subject="Duplicate",
            body="Should fail",
            priority="low",
            pending_action_id=action.id,
        )
    await db.rollback()
