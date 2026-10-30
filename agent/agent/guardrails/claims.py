from __future__ import annotations

import re
from dataclasses import dataclass

ORDER_ID = re.compile(r"\bORD-\d{4}\b")
TICKET_ID = re.compile(r"\bTCK-[0-9A-F]{8}\b")
MONEY = re.compile(r"(?:USD|\$|€|£|₹)\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


@dataclass(frozen=True)
class Claim:
    kind: str
    text: str
    normalized: str


def extract_claims(text: str) -> list[Claim]:
    claims: list[Claim] = []
    for match in ORDER_ID.findall(text):
        claims.append(Claim("order_id", match, match.lower()))
    for match in TICKET_ID.findall(text):
        claims.append(Claim("ticket_id", match, match.lower()))
    for match in MONEY.findall(text):
        normalized = match.replace(",", "")
        try:
            cents = round(float(normalized) * 100)
        except ValueError:
            continue
        claims.append(Claim("money", f"${match}", str(cents)))
    for match in ISO_DATE.findall(text):
        claims.append(Claim("date", match, match))
    return claims
