from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CreateSessionResponse(BaseModel):
    session_id: UUID
    customer_id: UUID


class MessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=2000)


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pending_action_id: UUID
    decision: Literal["approve", "reject"]
    idempotency_key: str = Field(min_length=8, max_length=128)
    preview_hash: str = Field(min_length=16, max_length=64)


class ApprovalResponse(BaseModel):
    status: str
    ticket_id: str | None
    replayed: bool
    reply: str | None = None


class CustomerOut(BaseModel):
    id: UUID
    name: str


class SessionOut(BaseModel):
    session_id: UUID
    customer_id: UUID


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class TranscriptMessage(BaseModel):
    role: str
    content: str
    tool_calls: list[str] = []


class TranscriptOut(BaseModel):
    session_id: UUID
    messages: list[TranscriptMessage]
