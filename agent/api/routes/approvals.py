from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Request

from agent.api.deps import get_runtime, require_api_key
from agent.logging import request_id_var
from agent.runtime import AgentRuntime
from agent.schemas import ApprovalRequest, ApprovalResponse

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/v1/sessions/{session_id}/approvals", response_model=ApprovalResponse)
async def decide(
    request: Request,
    session_id: UUID,
    body: ApprovalRequest,
) -> ApprovalResponse:
    runtime: AgentRuntime = get_runtime(request)
    return await runtime.resolve_approval(
        session_id=session_id,
        pending_action_id=body.pending_action_id,
        decision=body.decision,
        idempotency_key=body.idempotency_key,
        preview_hash=body.preview_hash,
        request_id=request_id_var.get() or "-",
    )
