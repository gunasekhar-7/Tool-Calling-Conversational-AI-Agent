from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
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

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Give schema failures the same envelope as every other error.

        A client should never have to parse two error shapes to find `code`.
        """
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
        detail = first.get("msg", "request did not match the expected schema")
        message = f"{location}: {detail}" if location else detail
        return JSONResponse(
            status_code=422,
            content={
                "code": "invalid_input",
                "message": message,
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
