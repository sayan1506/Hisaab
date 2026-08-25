"""Exception reason codes — shared by the generator and the matcher.

Declared in Phase 1 so that Phase 8 can compare *generator intent* against
*matcher verdict* mechanically: when the generator plants an unresolvable case it
records the reason here, and when the matcher abstains it emits a code from the
same enum. Two different vocabularies would make "correct abstention" a
judgement call instead of a count.

Taxonomy is section 16 of .response/razorpay-track-04-finance-controller.md.
Nothing in Phase 1 emits any of these — every clean-mode credit is resolvable.
"""

from __future__ import annotations

from enum import Enum


class Reason(str, Enum):
    """Why a record could not be resolved."""

    NO_CANDIDATE = "NO_CANDIDATE"
    AMBIGUOUS_MULTI_SUBSET = "AMBIGUOUS_MULTI_SUBSET"
    AMBIGUOUS_DUPLICATE_AMOUNT = "AMBIGUOUS_DUPLICATE_AMOUNT"
    UNEXPLAINED_RESIDUAL = "UNEXPLAINED_RESIDUAL"
    PARTIAL_SETTLEMENT_PENDING = "PARTIAL_SETTLEMENT_PENDING"
    REFUND_UNLINKED = "REFUND_UNLINKED"
    FX_RATE_GAP = "FX_RATE_GAP"
    NON_GATEWAY_CREDIT = "NON_GATEWAY_CREDIT"
    CREDIT_MISSING = "CREDIT_MISSING"
    SETTLEMENT_MISSING = "SETTLEMENT_MISSING"
    ROUNDING_DRIFT = "ROUNDING_DRIFT"

    def __str__(self) -> str:  # so f-strings and json.dump give the bare code
        return self.value


#: Codes that are *never* auto-resolvable — abstaining on these is the correct
#: answer, not a limitation. Phase 8's correct-abstention count reads this.
CORRECT_ABSTENTION_CODES = frozenset(
    {Reason.AMBIGUOUS_MULTI_SUBSET, Reason.AMBIGUOUS_DUPLICATE_AMOUNT}
)


if __name__ == "__main__":
    assert Reason("NO_CANDIDATE") is Reason.NO_CANDIDATE
    assert f"{Reason.FX_RATE_GAP}" == "FX_RATE_GAP"
    assert len(Reason) == 11
    print("reasons.py self-check ok")
