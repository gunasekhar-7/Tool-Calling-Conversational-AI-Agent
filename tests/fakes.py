from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    """Deterministic stand-in for a tool-calling model, with no network access."""

    responses: list[AIMessage] = Field(default_factory=list)
    repeat_last: bool = True
    calls: int = 0
    bound_tool_names: list[str] = Field(default_factory=list)
    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        self.bound_tool_names = [getattr(tool, "name", str(tool)) for tool in tools]
        return self

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        self.seen.append(list(messages))
        if not self.responses:
            return AIMessage(content="I have no answer configured.")
        index = min(self.calls, len(self.responses) - 1)
        if self.calls >= len(self.responses) and not self.repeat_last:
            index = len(self.responses) - 1
        self.calls += 1
        # LangGraph's message reducer de-duplicates by id, so each turn needs a
        # fresh id exactly like a real provider response.
        return self.responses[index].model_copy(update={"id": str(uuid.uuid4())})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message = self._next(messages)
        chunk = AIMessageChunk(
            content=message.content,
            tool_calls=message.tool_calls,
            tool_call_chunks=[],
            id=message.id,
        )
        yield ChatGenerationChunk(message=chunk)


class ExplodingChatModel(ScriptedChatModel):
    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        raise RuntimeError("provider exploded")

    def _stream(self, messages: list[BaseMessage], *args: Any, **kwargs: Any):
        raise RuntimeError("provider exploded")
        yield  # pragma: no cover


def tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )
