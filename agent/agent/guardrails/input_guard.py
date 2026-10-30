from __future__ import annotations

import re
from dataclasses import dataclass

MAX_INPUT_CHARS = 2000

_OVERRIDE = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+|the\s+)?(previous|prior|above)\s+"
    r"(instructions|rules|prompts|context)",
    re.IGNORECASE,
)
_EXFILTRATE = re.compile(
    r"(system|developer)\s+(prompt|message|instructions)|"
    r"(reveal|print|repeat|show)\s+(me\s+)?your\s+(system\s+)?"
    r"(prompt|instructions|rules)",
    re.IGNORECASE,
)
_ROLE_HIJACK = re.compile(r"you\s+are\s+now\s+(a|an|the)\s+\w+", re.IGNORECASE)
_TAG = re.compile(r"<[^>]{0,300}>")


@dataclass(frozen=True)
class ScreenResult:
    allowed: bool
    reason: str | None
    sanitized: str


def _strip_control_chars(text: str) -> str:
    return "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)


def screen_input(text: str, max_len: int = MAX_INPUT_CHARS) -> ScreenResult:
    """Cheap, deterministic screening. Ambiguous text is allowed and logged."""
    sanitized = _TAG.sub("", _strip_control_chars(text))
    sanitized = re.sub(r"[ \t]{2,}", " ", sanitized).strip()

    if not sanitized:
        return ScreenResult(False, "empty_message", sanitized)

    if len(sanitized) > max_len:
        return ScreenResult(False, "message_too_long", sanitized[:max_len])

    if _OVERRIDE.search(sanitized):
        return ScreenResult(False, "instruction_override", sanitized)
    if _EXFILTRATE.search(sanitized):
        return ScreenResult(False, "prompt_exfiltration", sanitized)
    if _ROLE_HIJACK.search(sanitized):
        return ScreenResult(False, "role_hijack", sanitized)

    return ScreenResult(True, None, sanitized)
