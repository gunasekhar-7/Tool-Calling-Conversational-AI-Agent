from __future__ import annotations

import json
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from agent.agent.graph import build_graph, graph_config
from agent.tools.context import ToolContext
from agent.tools.orders import OrderIdArgs
from tests.fakes import ScriptedChatModel, tool_call


def _dummy_tool() -> StructuredTool:
    async def run(order_id: str) -> str:
        return json.dumps({"ok": True, "order": {"id": order_id}})

    return StructuredTool.from_function(
        coroutine=run,
        name="get_order_status",
        description="Look up an order",
        args_schema=OrderIdArgs,
    )


def _ctx() -> ToolContext:
    return ToolContext(
        customer_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        db=None,  # type: ignore[arg-type]
        request_id="test",
    )


def _graph(model: ScriptedChatModel, *, max_tool_calls: int = 8):
    return build_graph(
        model=model,
        tools=[_dummy_tool()],
        checkpointer=InMemorySaver(),
        ctx=_ctx(),
        max_tool_calls=max_tool_calls,
        approval_ttl_seconds=900,
    )


async def _run(graph, text: str = "Where is ORD-1001?") -> dict:
    return await graph.ainvoke(
        {"messages": [HumanMessage(content=text)], "customer_id": "c1"},
        config=graph_config("thread-1", 8),
    )


async def test_tool_call_then_final_answer() -> None:
    model = ScriptedChatModel(
        responses=[
            tool_call("get_order_status", {"order_id": "ORD-1001"}, "call-1"),
            AIMessage(content="It shipped yesterday."),
        ]
    )
    state = await _run(_graph(model))
    assert state["messages"][-1].content == "It shipped yesterday."
    assert state["tool_calls"] == 1
    assert any(isinstance(m, ToolMessage) for m in state["messages"])
    assert model.calls == 2


async def test_no_tool_call_ends_immediately() -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="Hello, how can I help?")])
    state = await _run(_graph(model))
    assert len([m for m in state["messages"] if isinstance(m, AIMessage)]) == 1
    assert state["tool_calls"] == 0


async def test_unknown_tool_error_is_returned_to_the_model() -> None:
    model = ScriptedChatModel(
        responses=[
            tool_call("delete_everything", {}, "call-bad"),
            AIMessage(content="I can't do that."),
        ]
    )
    state = await _run(_graph(model))
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages
    assert tool_messages[0].status == "error"
    assert "not a valid tool" in str(tool_messages[0].content)
    assert state["messages"][-1].content == "I can't do that."


async def test_tool_calls_are_capped() -> None:
    model = ScriptedChatModel(
        responses=[tool_call("get_order_status", {"order_id": "ORD-1001"}, "call-loop")]
    )
    state = await _run(_graph(model, max_tool_calls=2))
    assert state.get("limit_reached") is True
    assert "stopped" in state["messages"][-1].content.lower()
    assert state["tool_calls"] <= 3


async def test_answer_after_the_cap_is_returned() -> None:
    model = ScriptedChatModel(
        responses=[
            tool_call("get_order_status", {"order_id": "ORD-1001"}, "c1"),
            tool_call("get_order_status", {"order_id": "ORD-1002"}, "c2"),
            tool_call("get_order_status", {"order_id": "ORD-1003"}, "c3"),
            AIMessage(content="Here is the summary."),
        ]
    )
    state = await _run(_graph(model, max_tool_calls=2))
    # The cap ends the tool loop, not the conversation: the model gets one
    # tool-free turn so a completed investigation is not replaced by an apology.
    assert state["messages"][-1].content == "Here is the summary."
    assert state["limit_reached"] is True
    assert "stopped after several" not in state["messages"][-1].content


async def test_cap_without_a_usable_answer_falls_back() -> None:
    model = ScriptedChatModel(
        responses=[
            tool_call("get_order_status", {"order_id": "ORD-1001"}, "c1"),
            tool_call("get_order_status", {"order_id": "ORD-1002"}, "c2"),
            tool_call("get_order_status", {"order_id": "ORD-1003"}, "c3"),
            AIMessage(content=""),
        ]
    )
    state = await _run(_graph(model, max_tool_calls=1))
    assert "stopped after several" in state["messages"][-1].content


async def test_system_prompt_is_prepended() -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="hi")])
    await _run(_graph(model))
    first_call = model.seen[0]
    assert first_call[0].type == "system"
    assert "tools provided" in first_call[0].content


async def test_tools_are_bound_to_the_model() -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="hi")])
    _graph(model)
    assert model.bound_tool_names == ["get_order_status"]


async def test_graph_requires_thread_id() -> None:
    model = ScriptedChatModel(responses=[AIMessage(content="hi")])
    graph = _graph(model)
    with pytest.raises(ValueError, match="thread_id"):
        await graph.ainvoke({"messages": [HumanMessage(content="hi")]})
