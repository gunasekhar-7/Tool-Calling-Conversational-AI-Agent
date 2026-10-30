from __future__ import annotations

import uuid

import httpx
import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import func, select, update

from agent.db.models import Ticket
from agent.db.seed import CUSTOMER_A, CUSTOMER_B
from tests.conftest import AUTH, collect_events, create_session
from tests.fakes import ScriptedChatModel, tool_call

pytestmark = pytest.mark.integration

TICKET_ARGS = {
    "order_id": "ORD-1001",
    "subject": "Damaged cable",
    "body": "The cable arrived bent.",
    "priority": "high",
}


def _writer_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        responses=[
            tool_call("create_ticket", TICKET_ARGS, "call-write"),
            AIMessage(content="Ticket is open."),
        ]
    )


async def _propose(client: httpx.AsyncClient, message: str = "Open a ticket please.") -> dict:
    session_id = await create_session(client, str(CUSTOMER_A))
    events = await collect_events(client, session_id, message)
    payload = dict(events)
    assert "confirmation_required" in payload, events
    return {"session_id": session_id, **payload["confirmation_required"]}


async def _ticket_count(factory, customer_id=CUSTOMER_A) -> int:
    async with factory() as db:
        total = await db.scalar(
            select(func.count()).select_from(Ticket).where(Ticket.customer_id == customer_id)
        )
        return int(total or 0)


async def test_approve_creates_exactly_one_ticket(app_client, factory) -> None:
    async with app_client(_writer_model()) as client:
        proposal = await _propose(client)
        before = await _ticket_count(factory)
        response = await client.post(
            f"/v1/sessions/{proposal['session_id']}/approvals",
            json={
                "pending_action_id": proposal["pending_action_id"],
                "decision": "approve",
                "idempotency_key": "approve-key-0001",
                "preview_hash": proposal["preview_hash"],
            },
            headers=AUTH,
        )
        after = await _ticket_count(factory)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "executed"
    assert body["ticket_id"].startswith("TCK-")
    assert body["replayed"] is False
    assert after == before + 1


async def test_same_idempotency_key_does_not_create_a_second_ticket(app_client, factory) -> None:
    async with app_client(_writer_model()) as client:
        proposal = await _propose(client)
        payload = {
            "pending_action_id": proposal["pending_action_id"],
            "decision": "approve",
            "idempotency_key": "approve-key-0002",
            "preview_hash": proposal["preview_hash"],
        }
        before = await _ticket_count(factory)
        first = await client.post(
            f"/v1/sessions/{proposal['session_id']}/approvals", json=payload, headers=AUTH
        )
        second = await client.post(
            f"/v1/sessions/{proposal['session_id']}/approvals", json=payload, headers=AUTH
        )
        created = await _ticket_count(factory) - before

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["replayed"] is True
    assert second.json()["ticket_id"] == first.json()["ticket_id"]
    assert created == 1


async def test_different_idempotency_key_after_execution_conflicts(app_client) -> None:
    async with app_client(_writer_model()) as client:
        proposal = await _propose(client)
        base = {
            "pending_action_id": proposal["pending_action_id"],
            "decision": "approve",
            "preview_hash": proposal["preview_hash"],
        }
        await client.post(
            f"/v1/sessions/{proposal['session_id']}/approvals",
            json={**base, "idempotency_key": "approve-key-0003"},
            headers=AUTH,
        )
        response = await client.post(
            f"/v1/sessions/{proposal['session_id']}/approvals",
            json={**base, "idempotency_key": "approve-key-0004"},
            headers=AUTH,
        )

    assert response.status_code == 409
    assert response.json()["code"] == "conflict"


async def test_reject_creates_no_ticket(app_client, factory) -> None:
    async with app_client(_writer_model()) as client:
        proposal = await _propose(client)
        before = await _ticket_count(factory)
        response = await client.post(
            f"/v1/sessions/{proposal['session_id']}/approvals",
            json={
                "pending_action_id": proposal["pending_action_id"],
                "decision": "reject",
                "idempotency_key": "reject-key-0001",
                "preview_hash": proposal["preview_hash"],
            },
            headers=AUTH,
        )
        created = await _ticket_count(factory) - before

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    assert response.json()["ticket_id"] is None
    assert created == 0


async def test_tampered_preview_hash_is_rejected(app_client) -> None:
    async with app_client(_writer_model()) as client:
        proposal = await _propose(client)
        response = await client.post(
            f"/v1/sessions/{proposal['session_id']}/approvals",
            json={
                "pending_action_id": proposal["pending_action_id"],
                "decision": "approve",
                "idempotency_key": "approve-key-0005",
                "preview_hash": "0" * 64,
            },
            headers=AUTH,
        )

    assert response.status_code == 409
    assert response.json()["code"] == "approval_mismatch"


async def test_approval_from_another_session_is_not_found(app_client) -> None:
    async with app_client(_writer_model()) as client:
        proposal = await _propose(client)
        other_session = await create_session(client, str(CUSTOMER_B))
        response = await client.post(
            f"/v1/sessions/{other_session}/approvals",
            json={
                "pending_action_id": proposal["pending_action_id"],
                "decision": "approve",
                "idempotency_key": "approve-key-0006",
                "preview_hash": proposal["preview_hash"],
            },
            headers=AUTH,
        )

    assert response.status_code == 404


async def test_expired_approval_is_refused(app_client, factory) -> None:
    from datetime import UTC, datetime, timedelta

    from agent.db.models import PendingAction

    async with app_client(_writer_model()) as client:
        proposal = await _propose(client)
        async with factory() as db:
            await db.execute(
                update(PendingAction)
                .where(PendingAction.id == uuid.UUID(proposal["pending_action_id"]))
                .values(expires_at=datetime.now(UTC) - timedelta(seconds=1))
            )
            await db.commit()

        response = await client.post(
            f"/v1/sessions/{proposal['session_id']}/approvals",
            json={
                "pending_action_id": proposal["pending_action_id"],
                "decision": "approve",
                "idempotency_key": "approve-key-0007",
                "preview_hash": proposal["preview_hash"],
            },
            headers=AUTH,
        )

    assert response.status_code == 409
    assert response.json()["code"] == "approval_expired"
