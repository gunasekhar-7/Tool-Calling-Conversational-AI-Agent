from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent.errors import DomainError
from agent.logging import request_id_var

logger = logging.getLogger(__name__)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def _domain_error(request: Request, exc: DomainError) -> JSONResponse:
        headers = {}
        retry_after = getattr(exc, "retry_after", None)
        if retry_after:
            headers["Retry-After"] = str(retry_after)
        return JSONResponse(
            status_code=exc.http_status,
            headers=headers,
            content={
                "code": exc.code,
                "message": exc.message,
                "request_id": request_id_var.get() or "",
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error")
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_error",
                "message": "Something went wrong. Quote the request id to support.",
                "request_id": request_id_var.get() or "",
            },
        )
