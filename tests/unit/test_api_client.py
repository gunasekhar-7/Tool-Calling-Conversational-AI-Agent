from __future__ import annotations

import httpx
import pytest

from dashboard.api_client import AgentAPI, ApiError, parse_sse


def test_parse_sse_ignores_keepalives_and_groups_events() -> None:
    lines = iter(
        [
            ": keepalive",
            "event: tool_call",
            'data: {"name": "get_order_status"}',
            "",
            "event: done",
            'data: {"text": "It shipped."}',
            "",
        ]
    )
    events = list(parse_sse(lines))
    assert [event.name for event in events] == ["tool_call", "done"]
    assert events[1].data["text"] == "It shipped."


def test_error_body_becomes_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"code": "policy_denied", "message": "message rejected", "request_id": "abc"},
        )

    api = AgentAPI("http://test", "key", transport=httpx.MockTransport(handler))
    with pytest.raises(ApiError) as caught:
        api.customers()
    assert caught.value.code == "policy_denied"
    assert caught.value.request_id == "abc"
    assert "request abc" in caught.value.user_message()


def test_stream_message_maps_events() -> None:
    body = (
        'event: token\ndata: {"text": "Hello "}\n\n'
        'event: token\ndata: {"text": "there"}\n\n'
        'event: done\ndata: {"text": "Hello there"}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer key"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    api = AgentAPI("http://test", "key", transport=httpx.MockTransport(handler))
    events = list(api.stream_message("00000000-0000-0000-0000-000000000000", "hi"))
    assert "".join(event.data["text"] for event in events if event.name == "token") == "Hello there"
    assert events[-1].name == "done"
