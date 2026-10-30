from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Request

from agent.api.deps import Principal, get_runtime, require_principal
from agent.logging import request_id_var
from agent.runtime import AgentRuntime
from agent.schemas import ApprovalRequest, ApprovalResponse

router = APIRouter()


@router.post("/v1/sessions/{session_id}/approvals", response_model=ApprovalResponse)
async def decide(
    request: Request,
    session_id: UUID,
    body: ApprovalRequest,
    principal: Principal = Depends(require_principal),
) -> ApprovalResponse:
    runtime: AgentRuntime = get_runtime(request)
    return await runtime.resolve_approval(
        session_id=session_id,
        pending_action_id=body.pending_action_id,
        decision=body.decision,
        idempotency_key=body.idempotency_key,
        preview_hash=body.preview_hash,
        request_id=request_id_var.get() or "-",
        customer_id=principal.customer_id,
    )
