from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from agent import __version__

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
    return JSONResponse(status_code=200, content={"status": "ready"})
