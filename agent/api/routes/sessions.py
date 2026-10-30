from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Request

from agent.api.deps import Principal, get_runtime, require_principal
from agent.db.repositories import CustomerRepo, SessionRepo
from agent.logging import request_id_var
from agent.runtime import AgentRuntime
from agent.schemas import CreateSessionResponse, CustomerOut, SessionOut, TranscriptOut

router = APIRouter()


@router.get("/v1/customers", response_model=list[CustomerOut])
async def list_customers(
    request: Request, principal: Principal = Depends(require_principal)
) -> list[CustomerOut]:
    """The caller's own customer record. There is no way to enumerate the others."""
    runtime: AgentRuntime = get_runtime(request)
    async with runtime.session_factory() as db:
        customer = await CustomerRepo(db).get(principal.customer_id)
        return [CustomerOut(id=customer.id, name=customer.name)]


@router.post("/v1/sessions", response_model=CreateSessionResponse, status_code=201)
async def create_session(
    request: Request, principal: Principal = Depends(require_principal)
) -> CreateSessionResponse:
    """Start a conversation.

    There is deliberately no body: the session is bound to whoever the bearer
    token authenticates, so a client cannot ask to be someone else.
    """
    runtime: AgentRuntime = get_runtime(request)
    async with runtime.session_factory() as db:
        row = await SessionRepo(db).create(principal.customer_id)
        await db.commit()
        return CreateSessionResponse(session_id=row.id, customer_id=row.customer_id)


@router.get("/v1/sessions/{session_id}", response_model=SessionOut)
async def get_session(
    request: Request, session_id: UUID, principal: Principal = Depends(require_principal)
) -> SessionOut:
    runtime: AgentRuntime = get_runtime(request)
    await runtime.ensure_session(session_id, customer_id=principal.customer_id)
    async with runtime.session_factory() as db:
        row = await SessionRepo(db).get(session_id)
        return SessionOut(session_id=row.id, customer_id=row.customer_id)


@router.get("/v1/sessions/{session_id}/transcript", response_model=TranscriptOut)
async def get_transcript(
    request: Request, session_id: UUID, principal: Principal = Depends(require_principal)
) -> TranscriptOut:
    runtime: AgentRuntime = get_runtime(request)
    messages = await runtime.get_transcript(
        session_id,
        request_id_var.get() or "-",
        customer_id=principal.customer_id,
    )
    return TranscriptOut(session_id=session_id, messages=messages)
