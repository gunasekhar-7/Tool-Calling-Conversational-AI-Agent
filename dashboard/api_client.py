"""Thin HTTP client for the agent API.

Kept free of Streamlit imports so the transport can be tested on its own.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx


class ApiError(Exception):
    """An error the API reported with a machine code and a request id."""

    def __init__(
        self, message: str, *, code: str = "internal_error", request_id: str | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.request_id = request_id

    def user_message(self) -> str:
        suffix = f" (request {self.request_id})" if self.request_id else ""
        return f"{self.message}{suffix}"


@dataclass(frozen=True)
class StreamEvent:
    name: str
    data: dict[str, Any]


def parse_sse(lines: Iterator[str]) -> Iterator[StreamEvent]:
    """Turn raw SSE lines into events, ignoring keepalive comments."""
    name = "message"
    for line in lines:
        line = line.rstrip("\r")
        if not line or line.startswith(":"):
            continue
        if line.startswith("event: "):
            name = line[len("event: ") :]
        elif line.startswith("data: "):
            raw = line[len("data: ") :]
            try:
                payload = json.loads(raw)
            except ValueError:
                payload = {"text": raw}
            yield StreamEvent(name, payload if isinstance(payload, dict) else {"value": payload})


class AgentAPI:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout
        self._transport = transport

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            headers=self._headers,
            timeout=self._timeout,
            transport=self._transport,
        )

    def _error(self, response: httpx.Response) -> ApiError:
        try:
            body = response.json()
        except ValueError:
            body = {}
        return ApiError(
            body.get("message") or f"HTTP {response.status_code}",
            code=body.get("code") or "http_error",
            request_id=body.get("request_id") or response.headers.get("x-request-id"),
        )

    def health(self) -> dict[str, Any]:
        with self._client() as client:
            response = client.get("/healthz")
            if response.status_code != 200:
                raise self._error(response)
            return response.json()

    def customers(self) -> list[dict[str, Any]]:
        with self._client() as client:
            response = client.get("/v1/customers")
            if response.status_code != 200:
                raise self._error(response)
            return list(response.json())

    def create_session(self, customer_id: str) -> str:
        with self._client() as client:
            response = client.post("/v1/sessions", json={"customer_id": customer_id})
            if response.status_code != 201:
                raise self._error(response)
            return str(response.json()["session_id"])

    def transcript(self, session_id: str) -> list[dict[str, Any]]:
        with self._client() as client:
            response = client.get(f"/v1/sessions/{session_id}/transcript")
            if response.status_code != 200:
                raise self._error(response)
            return list(response.json()["messages"])

    def stream_message(self, session_id: str, message: str) -> Iterator[StreamEvent]:
        with (
            self._client() as client,
            client.stream(
                "POST",
                f"/v1/sessions/{session_id}/messages",
                json={"message": message},
            ) as response,
        ):
            if response.status_code != 200:
                response.read()
                raise self._error(response)
            yield from parse_sse(response.iter_lines())

    def decide(
        self,
        session_id: str,
        *,
        pending_action_id: str,
        decision: str,
        idempotency_key: str,
        preview_hash: str,
    ) -> dict[str, Any]:
        with self._client() as client:
            response = client.post(
                f"/v1/sessions/{session_id}/approvals",
                json={
                    "pending_action_id": pending_action_id,
                    "decision": decision,
                    "idempotency_key": idempotency_key,
                    "preview_hash": preview_hash,
                },
            )
            if response.status_code != 200:
                raise self._error(response)
            return dict(response.json())
