from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REGISTERED_TOOLS = {
    "get_order_status",
    "list_customer_orders",
    "get_ticket",
    "check_refund_eligibility",
    "create_ticket",
}

WRITE_TOOLS = {"create_ticket"}

# The model must never be able to name the actor it acts as.
IDENTITY_KEYS = {"customer_id", "session_id", "actor", "user_id"}


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""


def authorize(tool_name: str, args: dict[str, Any]) -> Decision:
    """Structural authorization, enforced in the executor rather than the prompt."""
    if tool_name not in REGISTERED_TOOLS:
        return Decision(False, "unknown_tool")

    for key in args:
        if key in IDENTITY_KEYS:
            return Decision(False, "identity_parameter_not_allowed")

    for value in args.values():
        if isinstance(value, str) and len(value) > 4000:
            return Decision(False, "argument_too_long")
        if isinstance(value, str) and "\x00" in value:
            return Decision(False, "invalid_characters")

    return Decision(True)
