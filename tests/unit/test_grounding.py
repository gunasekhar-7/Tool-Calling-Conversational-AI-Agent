from __future__ import annotations

import uuid

from langchain_core.messages import AIMessage

from agent.agent.guardrails.claims import extract_claims
from agent.agent.guardrails.grounding import SAFE_TEMPLATE, repair_or_template, verify
from agent.db.models import ToolReceipt
from tests.fakes import ScriptedChatModel


def _receipt(facts: dict) -> ToolReceipt:
    return ToolReceipt(
        id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        tool_name="get_order_status",
        args_hash="abc",
        facts=facts,
    )


RECEIPT = _receipt(
    {
        "order": {
            "id": "ORD-1001",
            "status": "delivered",
            "total_cents": 12999,
            "total_display": "$129.99",
            "delivered_at": "2026-10-20T10:00:00+00:00",
        }
    }
)


def test_claims_are_extracted() -> None:
    claims = extract_claims("Order ORD-1001 cost $129.99 on 2026-10-20.")
    kinds = {claim.kind for claim in claims}
    assert kinds == {"order_id", "money", "date"}


def test_verified_answer_passes() -> None:
    report = verify("Order ORD-1001 was delivered and cost $129.99.", [RECEIPT])
    assert report.ok is True
    assert report.unsupported == []


def test_fabricated_order_id_is_flagged() -> None:
    report = verify("Order ORD-9999 is on its way.", [RECEIPT])
    assert report.ok is False
    assert [claim.text for claim in report.unsupported] == ["ORD-9999"]


def test_wrong_amount_is_flagged() -> None:
    report = verify("Order ORD-1001 cost $19.99.", [RECEIPT])
    assert report.ok is False
    assert report.unsupported[0].kind == "money"


def test_amount_without_symbol_is_not_a_claim() -> None:
    report = verify("Order ORD-1001 cost 129.99.", [RECEIPT])
    assert report.ok is True


def test_numeric_claim_with_no_receipts_is_flagged() -> None:
    report = verify("Your refund of $54.00 is on the way.", [])
    assert report.ok is False


def test_date_must_be_present_in_a_receipt() -> None:
    report = verify("It was delivered on 2026-10-20.", [RECEIPT])
    assert report.ok is True
    report = verify("It was delivered on 2026-01-01.", [RECEIPT])
    assert report.ok is False


# --------------------------------------------------------------------------- #
# dates inside timestamps and prose dates
# --------------------------------------------------------------------------- #


def test_iso_date_inside_a_receipted_timestamp_is_extracted() -> None:
    claims = extract_claims("Delivered on 2026-10-20T10:00:00+00:00.")
    assert [claim.normalized for claim in claims] == ["2026-10-20"]


def test_prose_date_matches_a_receipted_timestamp() -> None:
    report = verify("Order ORD-1001 was delivered on 20 October.", [RECEIPT])
    assert report.ok is True


def test_invented_prose_date_is_unsupported() -> None:
    report = verify("Order ORD-1001 was delivered on 3 November.", [RECEIPT])
    assert report.ok is False
    assert [claim.text for claim in report.unsupported] == ["3 November"]


def test_invented_iso_date_inside_a_sentence_is_unsupported() -> None:
    report = verify("Delivered on 2025-01-01.", [RECEIPT])
    assert report.ok is False


def test_a_denial_receipt_does_not_ground_a_delivery_date() -> None:
    denial = _receipt({"order_id": "ORD-1001", "found": False})
    report = verify("Order ORD-1001 was delivered on 20 October.", [denial])
    assert report.ok is False


async def test_repair_rewrites_ungrounded_answer() -> None:
    model = ScriptedChatModel(
        responses=[AIMessage(content="Order ORD-1001 cost $129.99 and was delivered.")]
    )
    draft = "Order ORD-9999 cost $129.99 and was delivered."
    report = verify(draft, [RECEIPT])
    repaired = await repair_or_template(
        answer=draft, report=report, receipts=[RECEIPT], model=model
    )
    assert "ORD-9999" not in repaired
    assert "ORD-1001" in repaired


async def test_failed_repair_falls_back_to_template() -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="Order ORD-8888 is fine.")])
    draft = "Order ORD-9999 cost $129.99."
    report = verify(draft, [RECEIPT])
    repaired = await repair_or_template(
        answer=draft, report=report, receipts=[RECEIPT], model=model
    )
    assert repaired == SAFE_TEMPLATE
