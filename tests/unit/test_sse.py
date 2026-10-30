from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from agent.api.sse import encode, with_keepalive
from agent.runtime import TurnEvent


async def test_encode_uses_event_and_data_frames() -> None:
    frame = encode("token", {"text": "hi"})
    assert frame.startswith("event: token\ndata: ")
    assert frame.endswith("\n\n")


async def test_keepalive_is_emitted_while_waiting() -> None:
    async def slow() -> AsyncIterator[TurnEvent]:
        await asyncio.sleep(0.12)
        yield TurnEvent("done", {"text": "ok"})

    frames = [frame async for frame in with_keepalive(slow(), interval=0.03)]
    assert frames[0] == ": keepalive\n\n"
    assert any("event: done" in frame for frame in frames)


async def test_keepalive_propagates_events() -> None:
    async def quick() -> AsyncIterator[TurnEvent]:
        yield TurnEvent("token", {"text": "a"})
        yield TurnEvent("done", {"text": "a"})

    frames = [frame async for frame in with_keepalive(quick(), interval=5)]
    assert len(frames) == 2
    assert "event: token" in frames[0]


async def test_closing_the_stream_cancels_the_producer() -> None:
    cancelled = asyncio.Event()

    async def endless() -> AsyncIterator[TurnEvent]:
        try:
            while True:
                await asyncio.sleep(0.01)
                yield TurnEvent("token", {"text": "."})
        finally:
            cancelled.set()

    stream = with_keepalive(endless(), interval=5)
    await anext(stream)
    await stream.aclose()
    await asyncio.wait_for(cancelled.wait(), timeout=1)
