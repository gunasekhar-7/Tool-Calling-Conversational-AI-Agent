from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Request

from agent.api.deps import get_runtime, require_api_key
from agent.db.repositories import CustomerRepo, SessionRepo
from agent.logging import request_id_var
from agent.runtime import AgentRuntime
from agent.schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    CustomerOut,
    SessionOut,
    TranscriptOut,
)

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/v1/customers", response_model=list[CustomerOut])
async def list_customers(request: Request) -> list[CustomerOut]:
    runtime: AgentRuntime = get_runtime(request)
    async with runtime.session_factory() as db:
        customers = await CustomerRepo(db).list_all()
        return [CustomerOut(id=c.id, name=c.name) for c in customers]


@router.post("/v1/sessions", response_model=CreateSessionResponse, status_code=201)
async def create_session(request: Request, body: CreateSessionRequest) -> CreateSessionResponse:
    runtime: AgentRuntime = get_runtime(request)
    async with runtime.session_factory() as db:
        row = await SessionRepo(db).create(body.customer_id)
        await db.commit()
        return CreateSessionResponse(session_id=row.id, customer_id=row.customer_id)


@router.get("/v1/sessions/{session_id}", response_model=SessionOut)
async def get_session(request: Request, session_id: UUID) -> SessionOut:
    runtime: AgentRuntime = get_runtime(request)
    async with runtime.session_factory() as db:
        row = await SessionRepo(db).get(session_id)
        return SessionOut(session_id=row.id, customer_id=row.customer_id)


@router.get("/v1/sessions/{session_id}/transcript", response_model=TranscriptOut)
async def get_transcript(request: Request, session_id: UUID) -> TranscriptOut:
    runtime: AgentRuntime = get_runtime(request)
    messages = await runtime.get_transcript(session_id, request_id_var.get() or "-")
    return TranscriptOut(session_id=session_id, messages=messages)
