from __future__ import annotations

from typing import Any

from agent.db.repositories import ReceiptRepo
from agent.tools.context import ToolContext


async def record_receipt(
    ctx: ToolContext,
    tool_name: str,
    args: dict[str, Any],
    facts: dict[str, Any],
) -> None:
    await ReceiptRepo(ctx.db).add(
        session_id=ctx.session_id,
        tool_name=tool_name,
        args=args,
        facts=facts,
    )
