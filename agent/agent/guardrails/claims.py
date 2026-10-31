from __future__ import annotations

import re
from dataclasses import dataclass

ORDER_ID = re.compile(r"\bORD-\d{4}\b")
TICKET_ID = re.compile(r"\bTCK-[0-9A-F]{8}\b")
MONEY = re.compile(r"(?:USD|\$|€|£|₹)\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")
# Receipts store timestamps ("2026-10-12T09:23:00+00:00"), so the date must not
# require a word boundary after it.
ISO_DATE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MONTHS.update({name[:3]: number for name, number in _MONTHS.items()})
_MONTHS["sept"] = 9

# "12 October", "12 Oct", "October 12" — a model that invents a delivery date
# rarely invents it in ISO form, so prose dates are claims too.
NATURAL_DATE = re.compile(
    r"\b(?:(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]{3,9})"
    r"|(?P<month_first>[A-Za-z]{3,9})\s+(?P<day_last>\d{1,2}))\b"
)


@dataclass(frozen=True)
class Claim:
    kind: str
    text: str
    normalized: str


def _month_number(name: str) -> int | None:
    return _MONTHS.get(name.lower())


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
    for match in NATURAL_DATE.finditer(text):
        day_text = match.group("day") or match.group("day_last") or ""
        month_text = match.group("month") or match.group("month_first") or ""
        number = _month_number(month_text)
        if number is None or not day_text.isdigit():
            continue
        day = int(day_text)
        if not 1 <= day <= 31:
            continue
        # No year is stated, so support is checked against the month and day of
        # a receipted timestamp.
        claims.append(Claim("date", match.group(0), f"{number:02d}-{day:02d}"))
    return claims
