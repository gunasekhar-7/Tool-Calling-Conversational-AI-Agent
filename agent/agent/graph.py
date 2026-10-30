from __future__ import annotations

import logging
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.agent.prompts import SYSTEM_PROMPT
from agent.agent.state import AgentState
from agent.agent.write_flow import WRITE_TOOL_NAMES, make_write_node
from agent.errors import ModelUnavailable
from agent.tools.context import ToolContext

logger = logging.getLogger(__name__)


def _handle_tool_error(exc: Exception) -> str:
    logger.warning("tool_error", extra={"extra_fields": {"error": type(exc).__name__}})
    return f"Tool failed: {type(exc).__name__}. Try a different approach or tell the user."


def build_graph(
    *,
    model: BaseChatModel | None,
    tools: list[BaseTool],
    checkpointer: Any,
    ctx: ToolContext,
    max_tool_calls: int,
    approval_ttl_seconds: int,
) -> Any:
    bound = model.bind_tools(tools) if model is not None and tools else model

    async def agent_node(state: AgentState) -> dict[str, Any]:
        if bound is None:
            raise ModelUnavailable("No model is configured")
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        # Consume the model's stream so LangGraph can forward tokens to SSE while
        # the node still returns a complete message.
        chunks: list[AIMessage] = []
        async for chunk in bound.astream(messages):
            chunks.append(chunk)
        response: AIMessage
        if not chunks:
            response = AIMessage(content="")
        else:
            response = chunks[0]
            for chunk in chunks[1:]:
                response = response + chunk  # type: ignore[assignment]
        produced = len(getattr(response, "tool_calls", None) or [])
        return {
            "messages": [response],
            "tool_calls": state.get("tool_calls", 0) + produced,
        }

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", None) or [])
        if state.get("tool_calls", 0) > max_tool_calls:
            return "limit"
        if not calls:
            return END
        if any(call["name"] in WRITE_TOOL_NAMES for call in calls):
            return "write"
        return "tools"

    async def limit_node(state: AgentState) -> dict[str, Any]:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I stopped after several tool lookups without reaching an answer. "
                        "Could you narrow the request to one order or one issue?"
                    )
                )
            ],
            "limit_reached": True,
        }

    write_node = make_write_node(ctx, ttl_seconds=approval_ttl_seconds)

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=_handle_tool_error))
    # A node that returns Command is supported at runtime; LangGraph's add_node
    # overloads only describe plain state dictionaries.
    builder.add_node("write", cast(Any, write_node))
    builder.add_node("limit", limit_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "write": "write", "limit": "limit", END: END},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("limit", END)

    return builder.compile(checkpointer=checkpointer)


def graph_config(thread_id: str, max_tool_calls: int) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": max(25, max_tool_calls * 2 + 10),
    }
