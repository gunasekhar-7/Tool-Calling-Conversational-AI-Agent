from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.agent.guardrails.policy import authorize
from agent.tools import ALL_TOOL_SPECS
from agent.tools.orders import ListOrdersArgs, OrderIdArgs


def test_tool_names_are_unique() -> None:
    names = [spec.name for spec in ALL_TOOL_SPECS]
    assert len(names) == len(set(names))


def test_registry_covers_read_and_write() -> None:
    kinds = {spec.kind for spec in ALL_TOOL_SPECS}
    assert kinds == {"read", "write"}


def test_order_args_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        OrderIdArgs(order_id="ORD-1001", customer_id="someone-else")


def test_order_args_shape_is_enforced() -> None:
    with pytest.raises(ValidationError):
        OrderIdArgs(order_id="1001")


def test_list_limit_is_bounded() -> None:
    with pytest.raises(ValidationError):
        ListOrdersArgs(limit=11)


def test_unknown_tool_fails_closed() -> None:
    assert authorize("delete_everything", {}).allowed is False


def test_identity_parameter_is_rejected() -> None:
    decision = authorize("get_order_status", {"order_id": "ORD-1001", "customer_id": "x"})
    assert decision.allowed is False
    assert decision.reason == "identity_parameter_not_allowed"


def test_failure_reason_is_not_echoed_to_the_browser() -> None:
    decision = authorize("nope", {})
    assert "exception" not in decision.reason


def test_read_tool_is_allowed() -> None:
    assert authorize("get_order_status", {"order_id": "ORD-1001"}).allowed is True
