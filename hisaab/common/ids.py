"""Identifier formats — the one place ID widths are defined.

Decision #8 (see .plan/phase1.md): fixed width 4 for every entity, so IDs sort
lexicographically at n=200 and beyond. Appendix A of the track spec mixes
setl_01 / C001 widths; we pick one width and never revisit it.

The LEAK_PATTERNS regexes are load-bearing: invariant I7 uses them to prove no
gateway identifier reached the bank statement. They match the ID *pattern* with
word boundaries rather than a bare prefix, because a naive "'C' in narration"
check both fires on IMPS-JOHNDOE-DIRECTTRANSFER and misses things it should catch.
"""

from __future__ import annotations

import re

ID_WIDTH = 4

PAYMENT_PREFIX = "pay_"
ORDER_PREFIX = "ord_"
SETTLEMENT_PREFIX = "setl_"
REFUND_PREFIX = "rfnd_"
CREDIT_PREFIX = "C"


def payment_id(n: int) -> str:
    return f"{PAYMENT_PREFIX}{n:0{ID_WIDTH}d}"


def order_id(n: int) -> str:
    return f"{ORDER_PREFIX}{n:0{ID_WIDTH}d}"


def settlement_id(n: int) -> str:
    return f"{SETTLEMENT_PREFIX}{n:0{ID_WIDTH}d}"


def refund_id(n: int) -> str:
    return f"{REFUND_PREFIX}{n:0{ID_WIDTH}d}"


def credit_id(n: int) -> str:
    return f"{CREDIT_PREFIX}{n:0{ID_WIDTH}d}"


# Invariant I7: none of these may appear in a bank narration, ever.
LEAK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("payment_id", re.compile(r"\bpay_\d{%d}\b" % ID_WIDTH)),
    ("order_id", re.compile(r"\bord_\d{%d}\b" % ID_WIDTH)),
    ("settlement_id", re.compile(r"\bsetl_\d{%d}\b" % ID_WIDTH)),
    ("refund_id", re.compile(r"\brfnd_\d{%d}\b" % ID_WIDTH)),
    ("credit_id", re.compile(r"\bC\d{%d}\b" % ID_WIDTH)),
)

DIGIT_RUN = re.compile(r"\d+")


def leaked_identifiers(text: str) -> list[str]:
    """Return the names of any gateway identifier patterns found in `text`."""
    return [name for name, pat in LEAK_PATTERNS if pat.search(text)]


def digit_runs(text: str) -> list[str]:
    """Maximal runs of digits in `text`.

    Used by I7 to compare against amounts by whole-run equality rather than
    substring containment: a UTR tail of "1004" must not be read as containing
    the rupee amount "100".
    """
    return DIGIT_RUN.findall(text)


if __name__ == "__main__":
    assert payment_id(1) == "pay_0001"
    assert credit_id(42) == "C0042"
    assert settlement_id(200) == "setl_0200"
    assert leaked_identifiers("NEFT-RAZORPAYSOFT-XXXX4471") == []
    assert leaked_identifiers("IMPS-JOHNDOE-DIRECTTRANSFER") == []
    assert leaked_identifiers("NEFT-pay_0042-XXXX4471") == ["payment_id"]
    assert leaked_identifiers("CR/C0001/SETTLEMENT") == ["credit_id"]
    assert digit_runs("NEFT CR/RAZORPAY/4471") == ["4471"]
    assert "100" not in digit_runs("IMPS-RZRPAY-1004")
    print("ids.py self-check ok")
