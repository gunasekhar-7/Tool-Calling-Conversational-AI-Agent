from __future__ import annotations

from langgraph.graph import MessagesState


class AgentState(MessagesState, total=False):
    """Messages plus the per-turn bookkeeping the router and UI need."""

    customer_id: str
    pending_action_id: str | None
    tool_calls: int
    limit_reached: bool
    degraded: bool
