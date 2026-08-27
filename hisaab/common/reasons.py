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
    #: **A search that found two or more subsets summing to the target, and nothing else.**
    #: Phase 5 decision 8. It is inside ``ABSTENTION_REASONS``, so any looser meaning lets a
    #: *missing capability* score as an honest refusal -- which is why "membership is not
    #: declared, so the set would have to be searched" got its own code below rather than
    #: borrowing this one.
    AMBIGUOUS_MULTI_SUBSET = "AMBIGUOUS_MULTI_SUBSET"
    #: The settlement matched, but no membership is declared for it and no search has run.
    #: Deliberately **not** an honest abstention: the matcher did not look and could not.
    #: Phase 5 step 6 turns this branch into a call into Tier 2, after which this code should
    #: appear only if the search is unavailable -- so a run reporting it is reporting a gap.
    MEMBERSHIP_UNDECLARED = "MEMBERSHIP_UNDECLARED"
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


# ``CORRECT_ABSTENTION_CODES`` was here, and Phase 5 decision 9 **deleted it rather than
# wiring it up.** It listed the two codes where abstaining is the right answer, and it was
# imported by ``scoring/metrics.py`` and referenced nowhere in the tree -- dead for three
# phases. ``metrics._classify`` keys ``CORRECT_ABSTENTION`` on truth's ``resolvable`` field
# and never reads a reason code, which is correct: whether abstaining was right is a property
# of the *data*, not of the excuse the matcher gave.
#
# It is deleted instead of kept because a dead frozenset named for a metric cell is an
# invitation -- the obvious "fix" is to make the cell read it, which would let a matcher
# choose its own grade by choosing its wording. If a later phase wants to cross-check the
# reason a row abstained against the reason it was planted, that belongs in a *new*,
# separately named assertion, not in a constant that looks like it is already load-bearing.


if __name__ == "__main__":
    assert Reason("NO_CANDIDATE") is Reason.NO_CANDIDATE
    assert f"{Reason.FX_RATE_GAP}" == "FX_RATE_GAP"
    # 12 with Phase 5's MEMBERSHIP_UNDECLARED. Asserted so that adding a code is a
    # deliberate act: the vocabulary is shared with the generator, and a code the two sides
    # do not both know about makes "did it abstain for the reason we planted?" unanswerable.
    assert len(Reason) == 12
    assert Reason.MEMBERSHIP_UNDECLARED is not Reason.AMBIGUOUS_MULTI_SUBSET, (
        "decision 8: 'nothing declares the members' and 'the search found two' are "
        "different facts and must not share a code"
    )
    assert not hasattr(Reason, "CORRECT_ABSTENTION_CODES")
    print("reasons.py self-check ok")
