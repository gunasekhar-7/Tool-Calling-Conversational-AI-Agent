from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from agent import __version__
from agent.api.deps import Principal, require_principal

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(status_code=503, content={"status": "starting"})
    from sqlalchemy import text

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        logger.warning("readyz_db_unavailable", extra={"extra_fields": {"error": str(exc)}})
        return JSONResponse(status_code=503, content={"status": "db_unavailable"})

    # Check model seam readiness (relevant for degraded mode)
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        return JSONResponse(status_code=503, content={"status": "runtime_unavailable"})

    model_ready = runtime.model is not None
    degraded = runtime.is_degraded
    db_circuit_state = runtime.db_circuit.state.value
    provider_circuit_state = runtime.provider_circuit.state.value

    return JSONResponse(
        status_code=200,
        content={
            "status": "ready",
            "model_ready": model_ready,
            "degraded": degraded,
            "db_circuit": db_circuit_state,
            "provider_circuit": provider_circuit_state,
        },
    )


@router.get("/metrics")
async def metrics(request: Request, principal: Principal = Depends(require_principal)) -> Response:
    """Prometheus scrape target.

    Bearer-protected like every other endpoint: series names and customer-level
    throughput are not public, and an open scrape endpoint is an easy amplifier.
    """
    body, content_type = request.app.state.metrics.render()
    return Response(content=body, headers={"Content-Type": content_type})
