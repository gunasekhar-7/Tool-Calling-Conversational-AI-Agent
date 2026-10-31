from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


@dataclass(frozen=True)
class ToolContext:
    """Everything a tool may know about the caller.

    ``customer_id`` is injected by the server. It is never a tool argument, so the
    model cannot choose whose data it reads or writes.

    Tools borrow a short-lived session from ``session_factory`` for the operation
    they are performing and commit it themselves. A turn never holds a connection
    while the model is thinking: the pool would be exhausted by concurrency long
    before the database was.
    """

    customer_id: UUID
    session_id: UUID
    session_factory: async_sessionmaker[AsyncSession]
    request_id: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, compare=False, repr=False)
    # Optional so the evaluation harness and unit tests can build a context without
    # a metrics registry.
    metrics: Any = field(default=None, compare=False, repr=False)

    def session(self) -> AsyncSession:
        return self.session_factory()
