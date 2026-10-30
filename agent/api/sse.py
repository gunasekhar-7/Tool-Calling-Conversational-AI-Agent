from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any


def encode(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


async def with_keepalive(
    source: AsyncIterator[Any], *, interval: float = 15.0
) -> AsyncIterator[str]:
    """Pump the runtime into a queue so a slow model never leaves the socket silent."""
    queue: asyncio.Queue[Any] = asyncio.Queue()

    async def pump() -> None:
        try:
            async for item in source:
                await queue.put(item)
        except Exception as exc:  # pragma: no cover - defensive
            await queue.put(exc)
        finally:
            await queue.put(None)

    task = asyncio.create_task(pump())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=interval)
            except TimeoutError:
                yield ": keepalive\n\n"
                continue
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield encode(item.event, item.data)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
