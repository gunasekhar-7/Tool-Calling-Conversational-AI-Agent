from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agent.agent.guardrails.claims import Claim, extract_claims
from agent.db.models import ToolReceipt

logger = logging.getLogger(__name__)

SAFE_TEMPLATE = (
    "I couldn't verify every detail in that answer, so I won't repeat it. "
    "I can look it up again — which order reference should I check?"
)


@dataclass(frozen=True)
class GroundingReport:
    ok: bool
    claims: list[Claim]
    unsupported: list[Claim]


def _collect_numbers(value: Any, out: set[str]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        out.add(str(int(value)))
    elif isinstance(value, str):
        out.add(value.lower())
    elif isinstance(value, dict):
        for key, item in value.items():
            out.add(str(key).lower())
            _collect_numbers(item, out)
    elif isinstance(value, list | tuple):
        for item in value:
            _collect_numbers(item, out)


def _corpus(receipts: list[ToolReceipt]) -> tuple[str, set[str]]:
    chunks: list[str] = []
    tokens: set[str] = set()
    for receipt in receipts:
        chunks.append(json.dumps(receipt.facts, default=str).lower())
        _collect_numbers(receipt.facts, tokens)
    return "\n".join(chunks), tokens


_ISO_DATE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")


def _supported(claim: Claim, corpus: str, tokens: set[str], dates: set[str]) -> bool:
    if claim.kind == "money":
        cents = claim.normalized
        if cents in tokens or cents in corpus:
            return True
        decimal = f"{int(cents) / 100:.2f}"
        return decimal in corpus
    if claim.kind == "date":
        if len(claim.normalized) == 10:
            return claim.normalized in corpus
        # A prose date states no year; it is supported when some receipted
        # timestamp has the same month and day.
        return any(date.endswith(f"-{claim.normalized}") for date in dates)
    return claim.normalized in corpus


def verify(text: str, receipts: list[ToolReceipt]) -> GroundingReport:
    claims = extract_claims(text)
    corpus, tokens = _corpus(receipts)
    dates = set(_ISO_DATE.findall(corpus))
    unsupported = [claim for claim in claims if not _supported(claim, corpus, tokens, dates)]
    return GroundingReport(ok=not unsupported, claims=claims, unsupported=unsupported)


def _message_text(message: Any) -> str:
    if isinstance(message.content, str):
        return message.content
    parts: list[str] = []
    for block in message.content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


async def repair_or_template(
    *,
    answer: str,
    report: GroundingReport,
    receipts: list[ToolReceipt],
    model: BaseChatModel,
) -> str:
    """One repair attempt using only receipted facts, then a safe fallback."""
    facts = "\n".join(json.dumps(r.facts, default=str) for r in receipts)[:8000]
    unsupported = ", ".join(claim.text for claim in report.unsupported)
    messages = [
        SystemMessage(
            content=(
                "You rewrite support replies so every concrete detail is supported by the "
                "tool results. Remove or correct anything not present in them. Do not add "
                "new facts. Answer in one or two sentences."
            )
        ),
        HumanMessage(
            content=(
                f"Tool results:\n{facts or '(none)'}\n\n"
                f"Draft answer:\n{answer}\n\n"
                f"Unsupported details: {unsupported}\n\n"
                "Rewrite the draft without the unsupported details."
            )
        ),
    ]
    try:
        repaired = await model.ainvoke(messages)
    except Exception:  # pragma: no cover - provider failure path
        logger.exception("grounding_repair_failed")
        return SAFE_TEMPLATE

    text = _message_text(repaired).strip()
    if not text:
        return SAFE_TEMPLATE
    if verify(text, receipts).ok:
        return text
    return SAFE_TEMPLATE
