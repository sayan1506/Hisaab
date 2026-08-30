"""What to do about each reason code. One hint per code, written by hand, never generated.

**Templated on purpose.** A hint is advice a person acts on, and a sentence that was composed at
demo time cannot be reviewed before it is read. These thirteen are fixed text, checked into the
repo, and each one was written against the branch that actually raises its code -- so a hint that
misdescribes its cause is a diff, not a surprise. No model writes here, in this phase or a later
one; if Phase 10 adds an explainer it goes beside this table, not into it.

**This is not the verdict's ``note``, and does not repeat it.** The matcher already writes a
per-row note naming the settlement, the gap in paise, the candidate distances -- *what happened
to this row*. A hint says *what a person does about this class of row*: which fact is missing,
and who has it. The note is evidence; the hint is the next action. A queue that printed only
notes would be a list of well-explained dead ends.

**Why this table stays in ``hisaab/triage/`` when the effort table had to move.** The effort
numbers are read by two packages, so two copies could disagree about a quantity and nothing
would notice (``hisaab/common/effort.py`` says so at length). Nothing outside triage prints a
hint: the scorer reports rates and cells, not advice. Moving this to ``common/`` would widen the
shared surface to buy nothing.

**``unblocks`` is the honest half.** For most codes there is a specific input that, if it were in
the files, would let the matcher resolve the row unattended next month -- the settlement's
membership, the settlement-day FX rate, the refund's link. Naming it turns the queue into a list
of feed requests rather than a list of chores. Where nothing would help, it says ``None`` rather
than inventing an ask: ``NON_GATEWAY_CREDIT`` is a row the matcher got *right*, and there is no
missing input behind it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..common.reasons import Reason
from .read import TriageError


@dataclass(frozen=True, slots=True)
class Hint:
    """What to do, and what would stop it recurring."""

    #: Imperative, and addressed to the person clearing the queue.
    action: str
    #: The input that would let the matcher resolve this class unattended -- or ``None`` where
    #: no input would, because the verdict was already correct.
    unblocks: str | None


#: One hint per reason code. Each was written against the branch in ``matcher/tier1.py`` that
#: raises it, cited so the pairing can be re-checked rather than trusted.
HINTS: dict[Reason, Hint] = {
    # tier1.py:613 (subset search found nothing, no foreign payment in the pool) and
    # tier1.py:876 (no settlement within the window pays this amount at all).
    # Two branches, and they send a person to different files, so the hint has to say which
    # note means which. At :876 no settlement in the window pays the amount at all -- no subset
    # search ran. At :613 a settlement did match and no subset of its payments sums to the
    # credit. An earlier draft described only the second and would have sent half of this
    # group's rows to look for a payment when a settlement was what was missing.
    Reason.NO_CANDIDATE: Hint(
        action=(
            "Something is absent from this export, and the note says which side: if no "
            "settlement in the window pays this amount, chase the payout; if a settlement "
            "matched but no subset of its payments sums to the credit at gross or net of the "
            "declared rates, chase the payments. If both files are complete, an undeclared "
            "deduction is in play and the rate schedule needs confirming."
        ),
        unblocks="the missing settlement or payment row in settlements.csv / payments.csv",
    ),
    # tier1.py:481 -- two or more subsets sum to the target, or the gross and net-of-fees
    # readings resolve differently.
    # Both of that branch's readings, not just the first: the note says either "two different
    # payment sets sum to it" or "the gross and net-of-fees readings of the credit resolve
    # differently". A hint naming only payment sets would misdescribe half the rows it heads.
    Reason.AMBIGUOUS_MULTI_SUBSET: Hint(
        action=(
            "The search found more than one answer -- either two payment sets that both sum to "
            "this credit, or a set that fits when read gross and a different one when read net "
            "of fees. The data cannot say which settled, so read the settlement's own payment "
            "list off the gateway dashboard and confirm the membership by hand."
        ),
        unblocks="settlement_items.csv rows for this settlement, so membership is declared "
        "rather than searched",
    ),
    # tier1.py:463 -- membership undeclared *and* the pool exceeds the Tier 2 cap, so the
    # search was refused rather than run. The note names the bound.
    Reason.MEMBERSHIP_UNDECLARED: Hint(
        action=(
            "The matcher declined to search here because the candidate pool is larger than its "
            "stated bound, not because the data is ambiguous -- so there is nothing to "
            "adjudicate. Supply the settlement's membership and the row resolves without "
            "judgement."
        ),
        unblocks="settlement_items.csv rows for this settlement (the search is unnecessary once "
        "membership is declared)",
    ),
    # tier1.py:895 -- two or more settlements match the amount inside the window; the note
    # names each one's distance in business days.
    #
    # **The ask here is the posting lag, deliberately not the UTR.** The first version of this
    # hint asked for "the full UTR on this bank row", which is wrong twice: the reference is
    # usually already present, and ``tier1.py:196`` states that nothing on the resolution path
    # reads it *on purpose* -- an independent signal that corroborates the join is evidence,
    # while the same signal used *as* the join would have hidden every missing capability
    # through Phase 4. So a fuller UTR would not let the matcher resolve this unattended, and
    # naming it as the unblocker would have sent someone to fix a feed that is not the problem.
    # A human may still use the reference, which is why the action says so and the ask does not.
    Reason.AMBIGUOUS_DUPLICATE_AMOUNT: Hint(
        action=(
            "Several settlements are for this same amount inside the posting window, so the "
            "amount alone cannot pick one. The note names each candidate and how many business "
            "days away it settled; where the statement line carries a reference, compare it "
            "against those settlements' UTRs to see which one actually landed."
        ),
        unblocks="the posting lag between a settlement and its credit, which no input file "
        "states -- given it, the candidate at that distance is the answer",
    ),
    # tier1.py:1017 -- two declared rules close the same gap with different component splits.
    Reason.AMBIGUOUS_ADJUSTMENT: Hint(
        action=(
            "The total is agreed and only the split is not: two declared rate rules both "
            "account for the deduction, with different fee/GST/TDS components. Confirm which "
            "schedule applied from the contract, then book the components accordingly."
        ),
        unblocks="the fee, GST and TDS components on the settlement record, rather than the "
        "net alone",
    ),
    # tier1.py:1138 -- the settlement agrees on date and amount, but the gap is larger or
    # smaller than the declared rates predict.
    # The gap runs in **both** directions and the two mean opposite things, so the hint cannot
    # name just one: either the rates predict a *larger* deduction than was actually withheld
    # (less was taken than the schedule says -- often a waiver or a rate that changed), or they
    # account for only part of the shortfall and a remainder is unexplained (a charge nobody
    # declared). An earlier draft described only the second and told a person to go looking for
    # an extra charge on rows where too little had been deducted.
    Reason.UNEXPLAINED_RESIDUAL: Hint(
        action=(
            "The settlement matches but the money does not, and the note says which way: either "
            "less was withheld than the declared rates predict -- check for a waiver or a rate "
            "that changed mid-month -- or part of the shortfall is unaccounted for, which is "
            "usually a charge nobody declared. Pull the settlement's own deduction breakup and "
            "compare it against the note's figures."
        ),
        unblocks="the settlement's itemised deductions, so the residual is declared rather "
        "than inferred",
    ),
    # tier1.py:855 -- no settlement pays the amount exactly, but one or more fall short by a
    # plausible reserve share.
    Reason.PARTIAL_SETTLEMENT_PENDING: Hint(
        action=(
            "This looks like a settlement with part of it held back, not a mismatch: the "
            "shortfall is within the reserve band. Confirm the rolling-reserve release "
            "schedule, then match the credit against the settlement net of the held amount."
        ),
        unblocks="the reserve held and its release date, which no input file states today",
    ),
    # tier1.py:549 and tier1.py:1091 -- a refund cites a payment but is linked to no
    # settlement, and it closes this credit's gap.
    Reason.REFUND_UNLINKED: Hint(
        action=(
            "A refund accounts for the difference but is not tied to any settlement, so the "
            "matcher would have had to assume the link. Confirm the refund named in the note "
            "was netted off this payout, then attach it."
        ),
        unblocks="a settlement reference on the refund record in refunds.csv",
    ),
    # tier1.py:595 and tier1.py:671 -- a foreign-currency payment sits in the pool and no
    # input file carries the settlement-day rate.
    Reason.FX_RATE_GAP: Hint(
        action=(
            "A payment here was captured in another currency, and its gross is fixed at the "
            "capture-day rate while the payout is not. The rate that reconciles them is in no "
            "input file, so supply the settlement-day rate; the matcher will not fit one, "
            "because a rate chosen to make the arithmetic work is not evidence."
        ),
        unblocks="the settlement-day conversion rate, or the payout's own foreign-currency "
        "amount alongside its INR value",
    ),
    # tier1.py:776 -- the narration matches no settlement's UTR tail and names no gateway
    # counterparty. Always IGNORED, so this is a dismissal rather than an exception.
    Reason.NON_GATEWAY_CREDIT: Hint(
        action=(
            "Nothing here points at the gateway: the narration names no gateway counterparty "
            "and its reference matches no settlement. Glance at it to agree it is other "
            "business, then take it out of the reconciliation scope."
        ),
        # Deliberately None. Every other hint asks for a fact the files lack, but this row was
        # judged correctly -- the only "missing" input would be the answer key, and a queue
        # that could read one would not be a queue.
        unblocks=None,
    ),
    # --- the three codes with no producer -----------------------------------------------
    # Priced and hinted anyway, because the table must be exhaustive over ``Reason`` and a
    # code left out would fail at the moment it first fired. They are documented in
    # ``PRODUCERLESS`` below; the queue never shows these headings today.
    Reason.CREDIT_MISSING: Hint(
        action=(
            "A settlement was paid out with no bank credit against it. Check whether the "
            "statement export covers the full period, then chase the payout with the bank."
        ),
        unblocks="a complete bank_statement.csv for the period",
    ),
    # tier1.py:948 -- reachable only if load.py's referential check is weakened, since the
    # loader refuses a settlement citing an unknown payment before the matcher runs.
    Reason.SETTLEMENT_MISSING: Hint(
        action=(
            "A settlement cites payments that are not in payments.csv, so its total cannot be "
            "rebuilt. Re-export payments for the period -- this is an incomplete extract "
            "rather than a reconciliation problem."
        ),
        unblocks="the cited payment rows in payments.csv",
    ),
    Reason.ROUNDING_DRIFT: Hint(
        action=(
            "The difference is sub-rupee and consistent with rounding at a different step than "
            "the matcher assumes. Confirm where the gateway rounds, then write the remainder "
            "off rather than investigating each row."
        ),
        unblocks="the rounding convention behind the declared rates",
    ),
}

assert set(HINTS) == set(Reason), (
    "every reason code needs a resolution hint, or the queue shows a group with a heading and "
    "no next action: missing " + str(sorted(str(r) for r in set(Reason) - set(HINTS)))
)

#: Codes no run can currently produce, for three different reasons -- documented so an empty
#: heading is never mistaken for a clean result. ``CREDIT_MISSING`` has no construction site at
#: all; ``SETTLEMENT_MISSING``'s branch (``tier1.py:948``) is unreachable because ``load.py``
#: refuses a settlement citing an unknown payment before the matcher runs, so it is defence
#: against that check being weakened; ``ROUNDING_DRIFT`` lost its producer when
#: ``--rounding-edge`` was declined. Their hints exist because ``HINTS`` must be exhaustive,
#: and because the day one of them fires is the wrong day to be writing advice.
PRODUCERLESS: frozenset[Reason] = frozenset(
    {Reason.CREDIT_MISSING, Reason.SETTLEMENT_MISSING, Reason.ROUNDING_DRIFT}
)

#: The dismissal group's hint. **Not ``HINTS[NON_GATEWAY_CREDIT]``**, even though that is the
#: only code the matcher ever dismisses with (``tier1.py:776``): the verdict contract requires a
#: reason only on ``EXCEPTION``, so a legal file may dismiss a row with no code or with another
#: one, and ``group.py`` files every dismissal under one heading regardless. This text therefore
#: has to hold for the whole group rather than for one code's evidence.
DISMISSAL_HINT = Hint(
    action=(
        "These were set aside as not gateway money. Scan the list to agree none of them belongs "
        "in the payout reconciliation, then leave them to the accounts they came from."
    ),
    unblocks=None,
)


def hint_for(reason: Reason | None) -> Hint:
    """The hint for one code. No default, on the same rule as ``group.minutes_for``.

    A missing hint raises rather than printing a blank line: a group heading with no next action
    is worse than no group, because it reads as "nothing can be done" instead of "nobody wrote
    this yet". ``None`` is refused too -- the dismissal group has ``DISMISSAL_HINT`` and callers
    must choose it deliberately, so that a code accidentally arriving as ``None`` cannot silently
    collect the dismissal advice.
    """
    if reason is None:
        raise TriageError(
            "no reason code to look up a hint for -- the dismissal group uses DISMISSAL_HINT, "
            "which callers select on Kind rather than by passing None here"
        )
    try:
        return HINTS[reason]
    except KeyError:
        raise TriageError(
            f"{reason} has no resolution hint in hisaab/triage/hint.py. Write one against the "
            f"branch that raises it -- a group heading with no next action reads as 'nothing "
            f"can be done about these'."
        ) from None


if __name__ == "__main__":
    # --- the table is complete, and every entry says something -------------------------
    assert len(HINTS) == len(Reason) == 13
    for reason, h in HINTS.items():
        assert h.action.strip(), f"{reason}: empty action"
        # A floor, so a placeholder cannot pass the completeness assertion above. The
        # shortest real hint here is well over this.
        assert len(h.action) > 60, f"{reason}: action too short to be advice: {h.action!r}"
        assert h.action[0].isupper(), f"{reason}: action should read as a sentence"
        if h.unblocks is not None:
            assert h.unblocks.strip() and len(h.unblocks) > 10, f"{reason}: thin unblocks"

    # --- no two codes share advice ----------------------------------------------------
    # The failure mode of a hand-written table of thirteen: a copy-paste that leaves two
    # different causes with the same next action, which nothing else here would notice.
    actions = [h.action for h in HINTS.values()]
    assert len(set(actions)) == len(actions), "two codes share the same action text"
    unblocks = [h.unblocks for h in HINTS.values() if h.unblocks is not None]
    assert len(set(unblocks)) == len(unblocks), "two codes ask for the same missing input"
    assert DISMISSAL_HINT.action not in actions, (
        "the dismissal group's advice must not be a copy of a single code's -- it has to hold "
        "for a group that may mix codes, including rows carrying none"
    )

    # --- exactly one code has nothing to ask for, and it is the correct one -------------
    no_ask = {r for r, h in HINTS.items() if h.unblocks is None}
    assert no_ask == {Reason.NON_GATEWAY_CREDIT}, (
        f"only a code the matcher got right has no missing input behind it, got {no_ask}"
    )
    assert DISMISSAL_HINT.unblocks is None

    # --- the producerless three ---------------------------------------------------------
    assert PRODUCERLESS <= set(Reason) and len(PRODUCERLESS) == 3
    # They are in the table like everything else. This is the assertion that fails if someone
    # "tidies up" by dropping advice for codes that cannot fire -- which would turn the day one
    # of them first fires into a crash in the queue.
    assert PRODUCERLESS <= set(HINTS)
    assert Reason.NON_GATEWAY_CREDIT not in PRODUCERLESS, (
        "NON_GATEWAY_CREDIT is the most common code in a noisy run, not a dead one"
    )

    # --- the accessor -------------------------------------------------------------------
    assert hint_for(Reason.FX_RATE_GAP) is HINTS[Reason.FX_RATE_GAP]
    assert all(hint_for(r) is HINTS[r] for r in Reason), "the accessor diverges from the table"
    assert "settlement-day rate" in hint_for(Reason.FX_RATE_GAP).action

    try:
        hint_for(None)
    except TriageError as e:
        assert "DISMISSAL_HINT" in str(e), e
    else:
        raise AssertionError("looked up a hint for no code at all")

    victim = Reason.REFUND_UNLINKED
    held = HINTS.pop(victim)
    try:
        assert victim not in HINTS, "the mutation did not take"
        try:
            hint_for(victim)
        except TriageError as e:
            assert "no resolution hint" in str(e) and "hint.py" in str(e), e
        else:
            raise AssertionError(f"{victim} returned a hint with no entry in the table")
    finally:
        HINTS[victim] = held
    assert set(HINTS) == set(Reason), "the table was left mutated"

    print("triage/hint.py self-check ok")
