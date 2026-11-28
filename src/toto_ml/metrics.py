"""Axis & Mirror number metrics (from the "Axis & Mirror Numbers Reference Chart (01 - 89)").

For a number written with its digits (TOTO: 2 digits "07", 4D: 4 digits "0725"):

* root   - digital root: add the digits until one digit is left (27 -> 9, 55 -> 10 -> 1, 0000 -> 0)
* axis   - every digit replaced by its axis partner 0<->5, 1<->6, 2<->7, 3<->8, 4<->9 (27 -> 72, 01 -> 56)
* mirror - the digits in reverse order (27 -> 72, 14 -> 41, 0725 -> 5270)

The rules work for any digit count, so the same three metrics are stored for TOTO numbers and for every
4D prize number. A 4D number is also split into its front and back pairs ("0725" -> "07", "25"), which are
looked up in the 2-digit chart.
"""
from __future__ import annotations

AXIS = {"0": "5", "1": "6", "2": "7", "3": "8", "4": "9", "5": "0", "6": "1", "7": "2", "8": "3", "9": "4"}
CHART_RANGE = range(1, 90)  # the document's chart covers 01 - 89


def as_digits(number: int | str, width: int = 2) -> str:
    """7 -> '07', '725' -> '0725' (width 4). Keeps leading zeros, which matter for 4D."""
    s = str(number).strip()
    if not s.isdigit():
        raise ValueError(f"Not a number: {number!r}")
    return s.zfill(width)


def digit_sum(number: str) -> int:
    return sum(int(c) for c in number)


def root(number: str) -> int:
    s = digit_sum(number)
    return 0 if s == 0 else 1 + (s - 1) % 9


def axis(number: str) -> str:
    return "".join(AXIS[c] for c in number)


def mirror(number: str) -> str:
    return number[::-1]


def metrics(number: int | str, width: int = 2) -> dict:
    """All metrics of one number, e.g. metrics(27) -> {'number': '27', 'root': 9, 'axis': '72', 'mirror': '72', ...}."""
    n = as_digits(number, width)
    out = {"number": n, "digit_sum": digit_sum(n), "root": root(n), "axis": axis(n), "mirror": mirror(n)}
    if len(n) == 4:
        out["front"], out["back"] = n[:2], n[2:]
    return out


def reference_chart(include_all: bool = True) -> list[dict]:
    """The 2-digit chart. 00-99 when include_all (so every 4D pair has a row), flagged in_chart for 01-89."""
    nums = range(100) if include_all else CHART_RANGE
    return [{**metrics(i), "value": i, "in_chart": i in CHART_RANGE} for i in nums]
