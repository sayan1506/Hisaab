"""How long one exception costs a person, by reason code. One table, two readers.

**Assumptions, not measurements** -- stated in ASSUMPTIONS.md row 34 so a judge can challenge
the numbers rather than discover they were invented at demo time. Gate 7 compares that row
against this table and fails on a disagreement, so the prose and the code cannot drift.

**Why this lives in ``hisaab/common/`` from Phase 9.** It was in ``scoring/metrics.py``, which
was fine while the scorer was the only reader. ``hisaab/triage`` is in ``MATCHER_PACKAGES``, so
``tools/check_isolation.py`` check 1 forbids it from importing ``hisaab.scoring`` -- and the
queue's whole job is telling a person how much work a group is. The alternative was a second
copy of the numbers in triage, which is the defect Phase 9 step 1 had just finished deleting
(``metrics`` and ``report`` read one table through two ``.get`` calls with defaults that
disagreed by ten minutes), reappearing one level up as two tables for one number.

Note which kind of thing moved. ``matcher/load.py`` duplicates the CSV *headers* on purpose and
``triage/read.py`` duplicates the *act of parsing* three keys, because a schema drift must fail
loudly. This is neither: it is a declared quantity, where two copies that disagree produce a
plausible wrong answer nothing detects -- the same reasoning that keeps the business-day
calendar shared in ``bizdays.py`` rather than copied.

**What did NOT move: the accessor.** ``scoring.metrics.minutes_for`` raises ``MetricsError`` so
``scoring/cli.py`` can report a bad table as a clean exit, and triage raises ``TriageError`` for
the same reason on its side. Two accessors over one table is only a defect when they can
disagree about a *value*; both of these refuse an unpriced code outright, and each self-check
asserts its agreement with the table here. The rule from step 1 stands -- no site may default.
"""

from __future__ import annotations

from .reasons import Reason

#: Minutes to clear one exception, by reason code.
MINUTES_PER_EXCEPTION: dict[Reason, int] = {
    Reason.NO_CANDIDATE: 10,
    Reason.AMBIGUOUS_MULTI_SUBSET: 15,
    # Strictly more work than the code above, which is why it is priced higher: there, the
    # search ran and the human picks between candidates it already found; here nothing
    # declares the members, so the human has to *construct* the candidate set by hand before
    # they can choose. Once Tier 2 exists this code should be rare -- a run full of them is
    # reporting that the search was unavailable, not that the data was hard.
    Reason.MEMBERSHIP_UNDECLARED: 20,
    Reason.AMBIGUOUS_DUPLICATE_AMOUNT: 8,
    # Cheaper than either ambiguity above, because less is unknown: the settlement and the
    # payment set are both already matched, and the only open question is *which declared
    # rate schedule applied*. That is looked up in a contract rather than reconstructed from
    # the data -- unlike AMBIGUOUS_MULTI_SUBSET (15), where the human still has to choose
    # between candidate payment sets, or MEMBERSHIP_UNDECLARED (20), where they build one.
    Reason.AMBIGUOUS_ADJUSTMENT: 10,
    Reason.UNEXPLAINED_RESIDUAL: 12,
    Reason.PARTIAL_SETTLEMENT_PENDING: 5,
    Reason.REFUND_UNLINKED: 10,
    Reason.FX_RATE_GAP: 20,
    Reason.NON_GATEWAY_CREDIT: 3,
    Reason.CREDIT_MISSING: 15,
    Reason.SETTLEMENT_MISSING: 15,
    Reason.ROUNDING_DRIFT: 5,
}

assert set(MINUTES_PER_EXCEPTION) == set(Reason), (
    "every reason code needs an effort estimate, or the human-time figure silently "
    "under-reports: missing " + str(sorted(str(r) for r in set(Reason) - set(MINUTES_PER_EXCEPTION)))
)

#: Minutes charged for one *dismissal* -- a row the matcher set aside as not gateway money
#: (``Outcome.IGNORED``). **Phase 9 charges these, where before they were priced and never
#: billed:** minutes accrued only for ``Outcome.EXCEPTION``, and ``NON_GATEWAY_CREDIT`` is
#: always ``IGNORED`` (``tier1.py:776``), so its price satisfied the exhaustiveness assertion
#: above without ever reaching a total.
#:
#: Sourced from the table rather than restated, so there is one number to challenge.
#:
#: Charged per *row*, not per code, because the verdict contract requires a reason only on
#: ``EXCEPTION`` (``verdict.py:262``): a legal file may hold an ``IGNORED`` row with no code,
#: and pricing dismissals by code would make the scorer raise on a file it must be able to
#: score. The work is the same either way -- a person glances at the row and moves on.
#:
#: **Measured before it was adopted** (`.plan/probe_phase9_nongateway_charge.py`). Charging
#: adds ``3K`` to the tool total while leaving the by-hand side alone, since those rows already
#: sit in the easy bucket -- so each charged row is worth -1 min against the tool and the
#: break-even hard rate rises by ``3K/gateway_exceptions``. Across six cells K is 3 at n=200
#: and 14-15 at n=1000: break-even 13.17 -> 13.34, leaving 1.66 min of the 1.83-minute ROI
#: window. At K >= 33 on the binding cell it would have inverted the claim instead.
DISMISSAL_MINUTES = MINUTES_PER_EXCEPTION[Reason.NON_GATEWAY_CREDIT]


if __name__ == "__main__":
    assert MINUTES_PER_EXCEPTION[Reason.NO_CANDIDATE] == 10
    assert DISMISSAL_MINUTES == 3
    assert all(v > 0 for v in MINUTES_PER_EXCEPTION.values()), (
        "a code priced at zero is a row the queue would show as free work"
    )
    # The assertion above guards the table; this guards the assertion. If a code were dropped
    # from Reason itself the set comparison would still pass, so the count is pinned too --
    # the same reasoning as reasons.py's own `len(Reason) == 13`.
    assert len(MINUTES_PER_EXCEPTION) == 13
    print("effort.py self-check ok")
