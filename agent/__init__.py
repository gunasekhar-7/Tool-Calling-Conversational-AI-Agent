"""Tool-calling support agent."""

from __future__ import annotations

import asyncio
import sys

__version__ = "0.1.0"

# psycopg's async mode cannot run on Windows' default Proactor loop. The Linux
# containers used in deployment are unaffected.
if sys.platform == "win32":  # pragma: no cover - platform bootstrap
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
