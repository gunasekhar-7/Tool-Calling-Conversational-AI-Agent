from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from agent.agent.guardrails.policy import authorize
from agent.tools.context import ToolContext

logger = logging.getLogger(__name__)

ToolHandler = Callable[[Any, ToolContext], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    kind: Literal["read", "write"]
    handler: ToolHandler | None = None


def _build_tool(spec: ToolSpec, ctx: ToolContext) -> BaseTool:
    async def _invoke(**kwargs: Any) -> str:
        args = spec.args_model(**kwargs)
        decision = authorize(spec.name, args.model_dump())
        if not decision.allowed:
            logger.warning(
                "tool_policy_denied",
                extra={"extra_fields": {"tool": spec.name, "reason": decision.reason}},
            )
            return json.dumps({"ok": False, "error": "policy_denied", "message": decision.reason})
        if spec.handler is None:
            return json.dumps({"ok": False, "error": "tool_unavailable"})
        async with ctx.lock:
            result = await spec.handler(args, ctx)
        return json.dumps(result, default=str)

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=spec.name,
        description=spec.description,
        args_schema=spec.args_model,
    )


def to_langchain_tools(specs: list[ToolSpec], ctx: ToolContext) -> list[BaseTool]:
    return [_build_tool(spec, ctx) for spec in specs]


def tool_map(specs: list[ToolSpec]) -> dict[str, ToolSpec]:
    return {spec.name: spec for spec in specs}
