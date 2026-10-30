from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


class ScriptedModel(BaseChatModel):
    """Deterministic stand-in for a tool-calling model.

    The evaluation harness replays a fixed plan so a failing run always means a
    change in the pipeline, not in a provider's sampling. Each reply gets a fresh
    id because LangGraph de-duplicates messages by id.
    """

    responses: list[AIMessage] = Field(default_factory=list)
    replay_last: bool = True
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        return self

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        if not self.responses:
            return AIMessage(content="No scripted response was provided.")
        index = min(self.calls, len(self.responses) - 1)
        if self.replay_last and index == len(self.responses) - 1:
            self.calls += 1
        else:
            self.calls += 1
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
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=message.content,
                tool_calls=message.tool_calls,
                tool_call_chunks=[],
                id=message.id,
            )
        )


def call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def say(text: str) -> AIMessage:
    return AIMessage(content=text)
