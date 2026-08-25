"""Money is integer paise. Always. Everywhere.

Decision #2 and #4 (see .plan/phase1.md):

  * Every monetary value in this codebase is an ``int`` number of paise. A single
    float anywhere costs an evening in Phase 4 chasing off-by-one-paisa failures
    that came from our own arithmetic rather than from the scenario under test.
  * Rates are **integer basis points**, never percentages as floats.
    2% = 200 bps, 18% GST = 1800 bps.
  * The rounding rule is **half-up at the paisa**, declared here and stated in
    ASSUMPTIONS.md. It is declared in Phase 1 even though nothing rounds yet,
    because deciding it under pressure in Phase 4 is exactly how the generator
    and the matcher end up disagreeing by one paisa.

``mul_bps`` is not called anywhere in Phase 1 (clean mode has zero fees). It is
written and tested now so the rule exists before two components can disagree
about it.
"""

from __future__ import annotations

RUPEE = 100  # paise per rupee


def paise(x: int) -> int:
    """Boundary guard for money.

    Rejects ``float`` and ``bool``. Looks paranoid; catches a real bug the first
    time Phase 4 writes ``gross * 0.02`` instead of ``mul_bps(gross, 200)``.
    """
    if isinstance(x, bool) or not isinstance(x, int):
        raise TypeError(f"money must be int paise, got {type(x).__name__}: {x!r}")
    return x


def rupees(n: int) -> int:
    """Rupees -> paise, for readability in fixtures and band definitions."""
    return paise(n) * RUPEE


def mul_bps(amount_paise: int, bps: int) -> int:
    """``amount_paise * bps / 10_000``, rounded **half-up** at the paisa.

    Exact integer arithmetic — no float, no Decimal context to misconfigure.

    Defined for non-negative operands only. Half-up on negatives is ambiguous
    (is -0.5 -> 0 or -1?), and every rate in this system applies to a positive
    amount, so we assert rather than guess. If Phase 6 ever needs a negative
    rate, negate the result of a positive call and state it there.

    The track spec's worked example: a Rs 1,111 sale at 2% is a Rs 22.22 fee, and
    18% GST on that fee is 399.96 paise, which must round to 400.
    """
    paise(amount_paise)
    if isinstance(bps, bool) or not isinstance(bps, int):
        raise TypeError(f"bps must be int, got {type(bps).__name__}: {bps!r}")
    assert amount_paise >= 0, f"half-up is defined here for non-negatives: {amount_paise}"
    assert bps >= 0, f"half-up is defined here for non-negatives: {bps}"
    return (amount_paise * bps + 5_000) // 10_000


def fmt(amount_paise: int) -> str:
    """Paise -> ``'Rs 13,598.80'`` with Indian digit grouping, for reports.

    Display only. Never compare formatted strings; compare ints.
    """
    paise(amount_paise)
    sign = "-" if amount_paise < 0 else ""
    whole, frac = divmod(abs(amount_paise), RUPEE)
    digits = str(whole)
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            head, group = head[:-2], head[-2:]
            groups.insert(0, group)
        if head:
            groups.insert(0, head)
        digits = ",".join(groups) + "," + tail
    return f"{sign}₹{digits}.{frac:02d}"


if __name__ == "__main__":
    # The rounding rule, on the track spec's own worked example.
    fee = mul_bps(rupees(1111), 200)
    assert fee == 2222, fee                      # Rs 22.22
    assert mul_bps(fee, 1800) == 400             # 399.96 paise, half-up -> 400
    # Exactly half a paisa must round up, not to-even.
    assert mul_bps(25, 200) == 1                 # 0.5 -> 1
    assert mul_bps(75, 200) == 2                 # 1.5 -> 2  (banker's would give 2)
    assert mul_bps(125, 200) == 3                # 2.5 -> 3  (banker's would give 2)
    # Degenerate cases.
    assert mul_bps(0, 1800) == 0
    assert mul_bps(rupees(10_000), 0) == 0
    # The float guard.
    for bad in (1.0, 0.5, True, "100", None):
        try:
            paise(bad)  # type: ignore[arg-type]
        except TypeError:
            pass
        else:
            raise AssertionError(f"paise() accepted {bad!r}")
    # Indian grouping.
    assert fmt(1_359_880) == "₹13,598.80", fmt(1_359_880)
    assert fmt(0) == "₹0.00"
    assert fmt(99) == "₹0.99"
    assert fmt(rupees(100)) == "₹100.00"
    assert fmt(rupees(1_00_00_000)) == "₹1,00,00,000.00", fmt(rupees(1_00_00_000))
    assert fmt(-1_359_880) == "-₹13,598.80"
    print("money.py self-check ok")
