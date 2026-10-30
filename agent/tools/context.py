from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class ToolContext:
    """Everything a tool may know about the caller.

    ``customer_id`` is injected by the server. It is never a tool argument, so the
    model cannot choose whose data it reads or writes.
    """

    customer_id: UUID
    session_id: UUID
    db: AsyncSession
    request_id: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, compare=False, repr=False)
