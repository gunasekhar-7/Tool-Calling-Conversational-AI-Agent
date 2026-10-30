"""Run the API locally.

uvicorn's default loop factory on Windows is a Proactor loop, which psycopg's
async mode cannot use, so this entry point asks for the selector loop instead.
Containers use the plain uvicorn command and never take this branch.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

import uvicorn

# uvicorn accepts a loop factory at runtime; its annotations only describe names.
LOOP: Any = asyncio.SelectorEventLoop if sys.platform == "win32" else "auto"


def main() -> None:
    uvicorn.run(
        "agent.api.main:create_app",
        factory=True,
        host=os.environ.get("API_HOST", "127.0.0.1"),
        port=int(os.environ.get("API_PORT", "8000")),
        loop=LOOP,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
