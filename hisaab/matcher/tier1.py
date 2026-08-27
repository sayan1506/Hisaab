"""Tier 1: resolve a credit against exactly one settlement, or abstain and say why.

The decision, in full::

    exactly one candidate  -> RESOLVED, tier 1, residual computed
    two or more candidates -> EXCEPTION, AMBIGUOUS_DUPLICATE_AMOUNT
    no candidates          -> EXCEPTION, NO_CANDIDATE

**Candidates are counted, never taken first.** ``blocking`` returns a list and this
module branches on its length: no ``next()``, no ``[0]`` without a length check ahead of
it. In clean mode the pool is always a singleton, so first-wins would score identically
and be wrong -- Phase 5's subset-sum needs "a subset exists" and "exactly one subset
exists" to be different facts, and the discipline is cheaper to build now, while the
counting is trivial, than to retrofit once it matters.

**A credit-to-settlement link is not a match.** Correctness is set equality on
``payment_ids`` (``hisaab/scoring/metrics._classify``), and ``Verdict.__post_init__``
refuses a ``RESOLVED`` verdict with an empty payment set outright. The payment set is
read from ``settlement_items.csv`` -- the membership declaration -- and never inferred.
That file is exactly what ``--settlement-report-late`` withholds in Phase 8, which is
the reason Phase 5's subset-sum has to exist: without it, the set must be *found*.

**The residual is computed, never assumed.** ``credit.amount_paise`` minus the gross of
the matched payments. It is zero for every row in clean mode, and computing it anyway is
what makes Phase 4's fee model move a number that already exists rather than introduce
one. A hard-coded zero would pass every check in this phase and be a silent bug in the
next.

**And the arithmetic behind it is published, not summarised.** Every resolved row carries
a ``Decomposition`` -- gross, fee, GST, and Phase 6's three terms at zero -- plus the
credit amount it was measured against, so ``residual == credit - expected`` is a sum any
reader can re-run and the scorer can check term by term against ``truth.json``'s own
decomposition block. A residual on its own was one number asserted about arithmetic nobody
else could see; a fee wrong in one direction and a GST wrong in the other land on the same
total, and only the term-by-term comparison separates them.

**Nothing here reads the narration to make a decision.** The parsed tail is recorded in
the verdict's ``note`` as corroboration only. See ``normalize.py`` for why that
separation is load-bearing, and note the consequence for testing: blanking every
narration must leave every *decision field* untouched, while ``note`` legitimately
changes. ``tools/acceptance.py`` gate 9 compares the former and excludes the latter.
"""

from __future__ import annotations

from datetime import date

from ..common.bizdays import BusinessCalendar
from ..common.reasons import Reason
from ..common.verdict import Decomposition, Outcome, Verdict
from .blocking import Candidate, SettlementIndex
from .fees import Explanation, FeeSchedule, derive, explain_gap
from .load import Credit, Dataset, Payment
from .normalize import Narration, parse
from .tier2 import ExactlyOne, Member, PoolTooLarge, TwoOrMore
from .tier2 import MAX_POOL as TIER2_MAX_POOL
from .tier2 import resolve as tier2_resolve

#: The tier this module speaks for. Carried onto every verdict it resolves so a report
#: can say *which* strategy earned a match, and so Phase 5's tier 2 is distinguishable
#: in the output rather than only in the code.
TIER = 1

# The tier a searched membership is credited to. Separate from ``TIER`` because the scorer
# reports the distribution, and a search that resolved a row must not be indistinguishable
# from a join that did -- gate 12 asserts both are non-zero for exactly that reason.
TIER_2 = 2

# How many business days after a payment is captured its settlement can land. This is the
# **settlement cycle**, and it is a declared assumption in the same sense as the fee rates
# in ``fees.py``: the matcher reads it off a contract, not off the generator, and is wrong
# when the counterparty settles on a different rhythm. It bounds the Tier 2 pool, so an
# understated value loses true members and an overstated one multiplies subsets.
#
# Measured on this data (``--settlement-delay``): every declared member sits at exactly
# +2 business days from capture, and without the flag at 0. A range of 0..2 covers both, so
# the bound is exact here rather than generous -- which is why the pool at n=1000 tops out
# near 60 and the cap is close to binding. Widening it is the trap the plan names: a wider
# pool manufactures ambiguity that costs coverage while looking generous.
SETTLEMENT_CYCLE_DAYS = 2

# The calendar the window is measured on. Weekends and holidays are not settlement days, so
# a calendar-day window would have to be wide enough to straddle a weekend -- measured at
# 4 calendar days to catch what 2 business days catches, and that widening tripled the pool
# (92 members against 63) for no additional true member.
_CALENDAR = BusinessCalendar()


def _note_for_match(
    candidate: Candidate,
    narration: Narration,
    payment_count: int,
    explanation: Explanation,
    tier: int = TIER,
) -> str:
    """Human-readable evidence for a resolved row.

    Names the **rule** that closed the gap, not merely that it closed. With one deduction
    rule that is nearly redundant; by Phase 6 there are several, and a row that balanced by
    coincidence rather than by truth is only visible if the output says which rule it
    credited. Phase 4 step 5 emits the components alongside it.

    Includes whether the settlement's UTR tail agrees with the narration's, which on
    this data it always does -- and which is precisely why nothing above reads it. An
    independent signal that corroborates the join is evidence; the same signal used *as*
    the join is a shortcut that would hide every missing capability through Phase 4.
    """
    settlement = candidate.settlement
    tail = settlement.utr.removeprefix("XXXX")
    if narration.ref_tail is None:
        utr = "utr=unparsed"
    elif narration.ref_tail == tail:
        utr = f"utr={tail} agrees"
    else:
        utr = f"utr={tail} vs narration {narration.ref_tail} DISAGREES"
    if explanation.total_paise:
        # Deliberately the *derived* components, never the settlement's declared columns:
        # this line is the arithmetic that earned the match, so it has to be the arithmetic
        # this matcher did.
        accounted = (
            f"{explanation.rule} accounts for {explanation.total_paise}p "
            f"(fee {explanation.fee_paise}p + GST {explanation.gst_paise}p "
            f"+ TDS {explanation.tds_paise}p)"
        )
    else:
        accounted = f"{explanation.rule}, credit equals gross"
    # Tier 2 rows must not read like Tier 1 rows. The join is identical -- same date, same
    # amount -- but the membership was *searched* rather than read, and a reader auditing a
    # match needs to know which of those they are looking at before they trust the payment
    # list. The tier is on the verdict as a field too; this is the sentence a human sees.
    if tier == TIER_2:
        how = (
            f"tier 2 subset: {settlement.settlement_id}, membership undeclared and searched "
            f"to {payment_count} payment(s) within {SETTLEMENT_CYCLE_DAYS}bd, uniquely"
        )
    else:
        how = (
            f"tier 1 exact: {settlement.settlement_id}, {payment_count} payment(s)"
        )
    return (
        f"{how}, "
        f"date distance {candidate.date_distance_days}bd, "
        f"amount delta {candidate.amount_delta_paise}p; "
        f"{accounted}; "
        f"corroboration only: {utr}"
    )


def _tier2_pool(settled_on: date, dataset: Dataset) -> list[Payment]:
    """The payments that could compose a settlement landing on ``settled_on``.

    Two filters, and the second is the one that needs justifying.

    **The window.** A payment can only be in this settlement if it was captured within
    ``SETTLEMENT_CYCLE_DAYS`` business days before the settlement landed. Measured exactly on
    this data, so it is a bound rather than a guess.

    **Payments another settlement already claims are excluded, and that is a partition fact
    rather than a hint.** The distinction is the whole argument, because reading
    ``settlement_items.csv`` at all is what the plan warns against:

      * A *hint* would be reading settlement X's own membership rows to answer "what is in
        X". Those rows are withheld -- that is the premise of the entire problem, and nothing
        here reads them.
      * A *partition fact* is reading settlement Y's rows to conclude that payment P, which Y
        declares, is therefore not in X. That is different information, and it is what a human
        reconciler does: money already settled elsewhere is not available to settle here.

    Verified rather than argued: across seeds 1-2 at n up to 1000, no payment appears in more
    than one settlement, and the exclusion never removed a true member of a withheld
    settlement (``true_member_lost=0`` at withholding shares from 30% to 100%).

    What it *does* affect is pool size, and therefore how often the cap binds -- so coverage
    depends on the withholding share, which is a dependency worth stating rather than hiding.
    Measured at n=1000: pool max 57-62 at a 30% share against a cap of 64, rising to 157-168
    at 100%, where the cap refuses ~95% of rows. The Phase 5 default share is 30%.

    Deliberately **not** excluding payments that an earlier Tier 2 search in this same run
    already claimed. It would raise coverage, and it would make the answer depend on the order
    credits happen to be processed in -- a matcher whose verdict for row 40 changes because
    row 12 was resolved first is not reproducible, and ``engine.py`` guarantees that it is.

    Nothing filters on ``status`` here. Every payment in this data is ``captured``, so a status
    guard would be a check never seen to fail; Phase 7's noise rows are where it earns a place.
    """
    claimed = {pid for pids in dataset.items.values() for pid in pids}
    return [
        payment
        for payment in dataset.payments
        if payment.payment_id not in claimed
        and 0
        <= _CALENDAR.business_days_between(payment.captured_at.date(), settled_on)
        <= SETTLEMENT_CYCLE_DAYS
    ]


def _search_membership(
    credit: Credit,
    candidate: Candidate,
    dataset: Dataset,
    schedule: FeeSchedule | None = None,
) -> tuple[str, ...] | Verdict:
    """Tier 2. Find the payments composing an undeclared settlement, or abstain.

    Returns the payment ids on success, and a finished ``Verdict`` on every abstention, so the
    caller can hand a found set straight into the same arithmetic that proves a Tier 1 match.
    That sharing is the point: Tier 2 finds a *set*, and the money proof below --
    ``explain_gap``, the decomposition, the residual assertion -- is the same code either way.
    A second copy of it would be a second chance to close a gap by coincidence.

    **Two hypotheses about the target, for the reason ``explain_gap`` has two rules.** The
    search needs a per-member amount, and which one is right depends on whether anything was
    deducted:

      * ``gross`` -- the bank credited the gross, which is what clean mode does and what any
        run without ``--fees`` does. A search that always subtracted a fee would find nothing
        on that data while the rate table sat there looking correct.
      * ``gross - fee - GST`` -- the declared rates were charged.

    Both are tried and the *distinct* sets are counted, exactly as ``explain_gap`` counts its
    rules. When the derived fee is zero the two hypotheses are the same number and collapse to
    one set, which must not read as an ambiguity. When they differ and both hit, the data
    genuinely does not say which, and that is an abstention.

    Per-member netting is sound here because the deduction is additive over members: measured
    across every declared settlement on seeds 1 and 42 at n=200 and n=1000, a batch net equals
    the sum of its members individual nets to the paisa, with zero drift. That is ``derive()``
    per-member-then-sum rule (see ``fees.py``) rather than a rate applied to a batch total, and
    it is what makes the target a plain integer subset-sum.
    """
    settlement_id = candidate.settlement_id
    pool = _tier2_pool(candidate.settlement.settled_on, dataset)

    # One list per hypothesis, and Phase 6 takes the count from two to **four** -- the same
    # four combinations ``explain_gap`` enumerates, for the same reason. The credit may have
    # been paid at the gross, net of the gateway's cut, net of the tax, or net of both, and
    # nothing in the inputs says which. This is not optional bookkeeping: under ``--tds`` the
    # settlement's net is ``gross - fee - gst - tds``, so a search offering only the first two
    # readings would find **nothing at all** on the deliverable's own flag set, while the rate
    # table sat there looking correct. That is the failure the gross hypothesis was added to
    # prevent one phase earlier, arriving from the other side.
    #
    # The cost is real and is measured rather than waved at: every extra reading is another
    # chance for a coincidental subset to hit the target, which raises the ambiguity rate
    # Phase 5 recorded at 6.59% (n=1000). Identical readings are collapsed below so a run
    # without a tax or without a fee does not pay for hypotheses that duplicate each other.
    subtractions: list[tuple[bool, bool]] = [
        (False, False),  # gross            -- NO_DEDUCTION
        (False, True),   # gross - tds      -- TDS_ONLY
        (True, False),   # gross - fee - gst -- FEE_AND_GST
        (True, True),    # gross - all       -- FEE_GST_TDS
    ]
    hypotheses: list[list[Member]] = []
    for take_fee, take_tds in subtractions:
        reading: list[Member] = []
        for payment in pool:
            deduction = derive([(payment.gross_paise, payment.method)], schedule)
            if take_fee and not deduction.is_complete:
                # No declared rate for this method, so its net cannot be computed. Dropping it
                # from a fee-bearing hypothesis is right -- the gross readings still carry it,
                # and a member priced at its gross under a schedule that does charge it would
                # be a wrong amount silently entering the search.
                continue
            amount = payment.gross_paise
            if take_fee:
                amount -= deduction.fee_and_gst_paise
            if take_tds:
                amount -= deduction.tds_paise
            if amount > 0:
                reading.append(Member(payment.payment_id, amount))
        # Collapse duplicates: with no fee charged, "gross" and "net of fee" are the same
        # list, and with the tax rate at zero so are the two TDS readings. Searching an
        # identical list twice cannot find a new set -- it can only spend the work bound --
        # and it must not read as an ambiguity either, which the set-level dedup below
        # already guarantees.
        if reading and reading not in hypotheses:
            hypotheses.append(reading)

    # Count the distinct sets across **both** hypotheses together, because the question is
    # not "does this reading resolve?" but "can anything I know explain this credit in more
    # than one way?". A hypothesis that reports two or more contributes both of them, so an
    # ambiguity under either reading is an ambiguity full stop.
    #
    # Getting that precedence wrong is measured, not hypothetical. The first version of this
    # function kept a single found-set and the ambiguity separately, and abstained on the
    # ambiguity only when nothing had been found. On seed 1 at n=1000, credit C0277 has a true
    # membership of the single payment pay_0439 whose net is exactly the credit; the net
    # hypothesis correctly reported two or more, while the gross hypothesis -- a reading that
    # is simply false on a fee-bearing run -- turned up one coincidental four-member set. The
    # unique-but-wrong answer overrode the known ambiguity and the row resolved wrongly. One
    # wrong match in 627, and the whole point of the third axis is that this is the number that
    # may not be traded.
    found: list[frozenset[str]] = []
    split = False
    over_cap: PoolTooLarge | None = None
    for members in hypotheses:
        result = tier2_resolve(members, credit.amount_paise)
        if isinstance(result, PoolTooLarge):
            over_cap = result
        elif isinstance(result, TwoOrMore):
            split = True
            for candidate_set in (result.first, result.second):
                if candidate_set not in found:
                    found.append(candidate_set)
        elif isinstance(result, ExactlyOne) and result.payment_ids not in found:
            found.append(result.payment_ids)

    if over_cap is not None:
        # A bounded refusal, and deliberately **not** an ``ABSTENTION_REASONS`` code: the
        # search never ran, so this is a capability limit and must score as a miss rather than
        # as an honest "I looked and could not tell". The note names the bound, so the answer
        # to "what happens at 10,000 records?" is a number rather than a shrug.
        #
        # Checked before the results below even when the other hypothesis did find something:
        # half a search is not a search, and committing on it would make coverage depend on
        # which reading happened to survive the cap.
        return Verdict(
            credit.credit_id,
            Outcome.EXCEPTION,
            reason=Reason.MEMBERSHIP_UNDECLARED,
            note=(
                f"matched {settlement_id} on date and amount, but its membership is "
                f"undeclared and the candidate pool of {over_cap.pool_size} payments within "
                f"{SETTLEMENT_CYCLE_DAYS}bd exceeds the tier 2 cap of {over_cap.cap} -- "
                f"refused rather than searched, because the work is not bounded above it"
            ),
        )

    if len(found) > 1:
        how = (
            "two different payment sets sum to it"
            if split
            else "the gross and net-of-fees readings of the credit resolve differently"
        )
        return Verdict(
            credit.credit_id,
            Outcome.EXCEPTION,
            reason=Reason.AMBIGUOUS_MULTI_SUBSET,
            note=(
                f"matched {settlement_id} on date and amount, but its membership is "
                f"undeclared and {how}: {sorted(found[0])} and {sorted(found[1])} -- the "
                f"inputs cannot separate them, so a human decides"
            ),
        )

    if not found:
        # Not a cue to widen the pool. Either a payment is missing from the data or a
        # deduction this matcher does not model was taken; widening the window until
        # something adds up would convert both into a match, which is the failure mode
        # Phase 4 measured for the nearest-date tie-break.
        return Verdict(
            credit.credit_id,
            Outcome.EXCEPTION,
            reason=Reason.NO_CANDIDATE,
            note=(
                f"matched {settlement_id} on date and amount, but no subset of the "
                f"{len(pool)} payment(s) captured within {SETTLEMENT_CYCLE_DAYS}bd sums to "
                f"{credit.amount_paise}p, at gross or net of the declared rates -- a payment "
                f"is missing or a deduction is unmodelled, and the window stays fixed"
            ),
        )

    # Sorted so the verdict is stable: the search returns a frozenset, and nothing on the
    # path from input to output may iterate one.
    return tuple(sorted(found[0]))


def resolve_credit(
    credit: Credit,
    index: SettlementIndex,
    dataset: Dataset,
    window_days: int = 0,
    max_adjustment_paise: int = 0,
    schedule: FeeSchedule | None = None,
) -> Verdict:
    """One credit in, exactly one ``Verdict`` out. Never returns ``None``, never skips.

    A missing verdict is not an option: ``verdict_io.reconcile`` refuses a file that
    drops a row, and discovering that in this loop is much cheaper than discovering it
    at review time.

    ``schedule`` is the fee rates this matcher believes are in force (``fees.FeeSchedule``,
    defaulted there). It is a parameter rather than a constant because the rates are an
    *assumption* about a counterparty, and a run should be able to state a different one
    without editing the matcher.
    """
    narration = parse(credit.narration)
    candidates = index.candidates_for(
        credit, window_days=window_days, max_adjustment_paise=max_adjustment_paise
    )

    # There is deliberately NO date-proximity tie-break here. Phase 3 shipped one
    # (``blocking.narrow_by_date_distance``) that could not fire at ``W = 0``; Phase 4's
    # posting lag made it fire, and it turned out to pick the *wrong* candidate
    # systematically -- the true settlement sits at a constant +1 business days, so the
    # closest candidate is never the right one. It cost 5-10 wrong matches per 1000 rows
    # while coverage stayed at ~99.5%. See that function's docstring for the measurement.
    #
    # Two settlements sharing an amount inside the window are not separable from these
    # inputs without knowing the lag, so the pool falls through to the abstention below.

    if not candidates:
        return Verdict(
            credit.credit_id,
            Outcome.EXCEPTION,
            reason=Reason.NO_CANDIDATE,
            note=(
                f"no settlement within {window_days}bd of {credit.value_date.isoformat()} "
                f"at {credit.amount_paise}p +/-{max_adjustment_paise}p"
            ),
        )

    if len(candidates) > 1:
        # Name the distances, not just the ids. At W=0 they genuinely share a date, but
        # under a posting lag they need not -- they share an *amount* while both sitting
        # inside the window, and which one is which is exactly what a human has to decide.
        # Printing "share this date" there would have been wrong, and printing the
        # distances is what makes the exception actionable rather than merely honest.
        ids = ", ".join(
            f"{c.settlement_id} at {c.date_distance_days:+d}bd" for c in candidates
        )
        return Verdict(
            credit.credit_id,
            Outcome.EXCEPTION,
            reason=Reason.AMBIGUOUS_DUPLICATE_AMOUNT,
            note=(
                f"{len(candidates)} settlements match {credit.amount_paise}p within "
                f"{window_days}bd of {credit.value_date.isoformat()} ({ids}) -- the inputs "
                f"cannot separate them without knowing the posting lag, so a human decides"
            ),
        )

    winner = candidates[0]
    settlement_id = winner.settlement_id
    payment_ids = dataset.items.get(settlement_id, ())
    tier = TIER

    if not payment_ids:
        # The settlement matched, but nothing declares which payments compose it.
        # ``--settlement-report-late`` creates exactly this state by withholding
        # settlement_items.csv, and the payment set has to be *searched*: Tier 2.
        #
        # Phase 5 step 6 replaced an abstention here with that call. What the search returns
        # is only a **set**; every line below this point is unchanged and runs on it exactly
        # as it runs on a declared membership, so the arithmetic that proves a Tier 2 match is
        # the same arithmetic that proves a Tier 1 one. A separate proof for the searched case
        # would be a second chance to close a gap by coincidence, and the residual assertion
        # below is the guard that would be duplicated -- and therefore weakened.
        #
        # That the proof cannot fail here is worth stating, because it is a property of the
        # search rather than luck: the set was found because it sums to the credit either at
        # gross (so the gap is zero and ``NO_DEDUCTION`` closes it) or at net of the declared
        # rates (so the gap equals ``derive()``'s per-member fee-and-GST sum to the paisa,
        # which is what ``FEE_AND_GST`` accounts for). The assertion still runs, because a
        # property that holds by argument and is never checked is how the argument stops being
        # true.
        #
        # Every abstention comes back as a finished ``Verdict``, and the reason codes differ by
        # what actually happened -- ``AMBIGUOUS_MULTI_SUBSET`` when the search found two
        # answers, ``NO_CANDIDATE`` when it found none, ``MEMBERSHIP_UNDECLARED`` when the pool
        # was over the cap and it never ran. Only the first two are honest refusals; see
        # ``_search_membership``.
        searched = _search_membership(credit, winner, dataset, schedule)
        if isinstance(searched, Verdict):
            return searched
        payment_ids = searched
        tier = TIER_2

    by_payment = dataset.payments_by_id()
    missing = [pid for pid in payment_ids if pid not in by_payment]
    if missing:
        # load.py already refuses this, so reaching it means the loader's referential
        # check was weakened. Abstain rather than emit a residual computed from a
        # partial sum, which would look precise and be wrong.
        return Verdict(
            credit.credit_id,
            Outcome.EXCEPTION,
            reason=Reason.SETTLEMENT_MISSING,
            note=f"{settlement_id} cites payments absent from payments.csv: {missing}",
        )

    # The fee model needs the method as well as the amount: the rate depends on it, so a
    # gap cannot be priced from the money alone. Already the Phase 5 shape -- batching
    # lengthens this list and changes nothing below.
    members = [(by_payment[pid].gross_paise, by_payment[pid].method) for pid in payment_ids]
    gross_total = sum(gross for gross, _ in members)

    # The residual is what no rule this matcher can apply accounts for.
    #
    # ``gap`` is positive when the bank credited *less* than the payments grossed, which is
    # the only direction a deduction can explain. ``explain_gap`` accounts for it exactly or
    # not at all -- there is no tolerance band, so there is nothing to tune and no width to
    # widen under pressure.
    #
    # It deliberately does NOT read ``winner.fee_paise``/``gst_paise``/``tds_paise``, even
    # though load.py parses those columns and they are legitimate input. Trusting a declared
    # number is not explaining a gap: reading the fee off settlements.csv would have closed
    # the residual the instant ``--fees`` populated it, so coverage would have held at 100%
    # with no fee model ever written and no number moving to say one was missing. Because
    # the rule (rate x gross, GST on the fee, half-up at the paisa) is independent of the
    # declared column, a settlement whose stated fee disagrees with its own published rate
    # now fails to close -- a real finding rather than a rounding error nobody looks at.
    # The self-check below pins this with a settlement whose ``fee_paise`` would close the
    # gap if anything here read it.
    gap = gross_total - credit.amount_paise
    closing, derived = explain_gap(gap, members, schedule)

    if len(closing) > 1:
        # **Two declared rules close this gap with different component splits.** Phase 6
        # decision 7, and it lands here rather than inside ``explain_gap`` because the module
        # that knows the rules should report all of them and the caller should decide -- the
        # same division Phase 5 settled for the subset search.
        #
        # Committing to one would be a coin flip on the *decomposition*, which the scorer
        # grades term by term rather than on the total (ASSUMPTIONS.md #25). So a row that
        # balances under two different splits is genuinely undetermined even though its
        # settlement, its payment set and its total are all known -- which is why this is an
        # honest abstention and not a missing capability.
        #
        # Unreachable at the declared rates: see ``Reason.AMBIGUOUS_ADJUSTMENT`` for the
        # measurement. It is reachable under a ``--fee-bps`` override, which is exactly the
        # case that flag exists for, so the branch is live rather than defensive.
        rules = " / ".join(
            f"{e.rule} (fee {e.fee_paise}p + GST {e.gst_paise}p + TDS {e.tds_paise}p)"
            for e in closing
        )
        return Verdict(
            credit.credit_id,
            Outcome.EXCEPTION,
            reason=Reason.AMBIGUOUS_ADJUSTMENT,
            note=(
                f"{settlement_id} agrees on date and amount and the {gap}p gap closes "
                f"exactly under {len(closing)} different declared rules -- {rules} -- so "
                f"the components are undetermined even though the total is not, and a "
                f"human decides which schedule applied"
            ),
        )

    explanation = closing[0] if closing else None
    if explanation is None:
        # ASSUMPTIONS.md #25: matched-but-unproven is an exception, not a match. A
        # decomposition that does not close to zero paise means money is unaccounted
        # for, and calling that a match is how a reconciliation tool issues a false
        # clean bill of health -- the error then surfaces at audit instead of here.
        #
        # The verdict contract (common/verdict.py) forbids an EXCEPTION from carrying
        # settlement_ids, payment_ids or a residual, so an abstention can never be
        # mistaken for a match by the scorer's set-equality join. That is the right
        # trade, and it puts the whole diagnostic in ``note``.
        #
        # The note quantifies how *close* the model got, because that is the difference
        # between a triage-able exception and a shrug: a 2,622p model against a 3,000p gap
        # points at a missing rule, while an unpriced method points at the rate table, and
        # "unexplained" alone points at nothing.
        if gap < 0:
            shortfall = (
                f"the credit exceeds the {gross_total}p gross of {len(payment_ids)} "
                f"payment(s) by {-gap}p, and no deduction adds money"
            )
        elif not derived.is_complete:
            shortfall = (
                f"the credit falls {gap}p short of the {gross_total}p gross of "
                f"{len(payment_ids)} payment(s), and no rate is declared for "
                f"{', '.join(derived.unpriced)} -- the gap cannot be priced at all"
            )
        elif derived.total_paise > gap:
            # The model predicts a *bigger* deduction than actually happened. Worth its own
            # sentence rather than a negative remainder: an under-prediction points at a
            # rule this matcher has not built yet, while an over-prediction points at the
            # rate table being wrong for this row -- opposite investigations, and a
            # "leaving -1514p unexplained" would send a reader down neither.
            shortfall = (
                f"the credit falls {gap}p short of the {gross_total}p gross of "
                f"{len(payment_ids)} payment(s), but the declared rates predict a larger "
                f"deduction of {derived.total_paise}p ({derived.describe()}) -- the rates "
                f"assumed here do not match what was actually withheld"
            )
        else:
            shortfall = (
                f"the credit falls {gap}p short of the {gross_total}p gross of "
                f"{len(payment_ids)} payment(s), and the declared rates account for "
                f"{derived.total_paise}p of it ({derived.describe()}), leaving "
                f"{gap - derived.total_paise}p unexplained"
            )
        return Verdict(
            credit.credit_id,
            Outcome.EXCEPTION,
            reason=Reason.UNEXPLAINED_RESIDUAL,
            note=(
                f"{settlement_id} agrees on date and amount, but {shortfall} -- "
                f"matched is not proven, so a human decides"
            ),
        )

    # Phase 4 step 5: the proof goes in the output, not only in the note.
    #
    # ``tds_paise``, ``refunds_paise`` and ``reserve_paise`` stay at their zero defaults
    # because tier 1 has no rule that produces them yet. Phase 6 fills them in, and doing
    # so is then a change of *value* -- the scorer already compares all six terms, so a
    # refund modelled as a fee is a mismatch on the very run that introduces it, rather
    # than a total that happens to agree.
    decomposition = Decomposition(
        gross_paise=gross_total,
        fee_paise=explanation.fee_paise,
        gst_paise=explanation.gst_paise,
        # Phase 6 step 2. Non-zero only under a rule that includes it, and **derived from the
        # declared rate rather than read from ``winner.tds_paise``** -- decision 2. The column
        # is parsed and one field away, and reading it would score 100% while modelling
        # nothing: unlike the Tier 1/Tier 2 split there is no distribution downstream that
        # would look wrong if it were copied, so the discipline has no backstop here beyond
        # being stated. The derived-versus-declared comparison is *reported* by the CLI for
        # exactly that reason (step 2).
        tds_paise=explanation.tds_paise,
        rule=explanation.rule,
    )

    # Zero by construction: ``explain_gap`` returned an explanation only because it
    # accounts for the gap exactly. Computed from the decomposition rather than from
    # ``explanation.total_paise`` so the number on the verdict is the remainder of the
    # arithmetic the verdict actually *publishes* -- if the two ever diverge, the published
    # decomposition is the one a reader can check, so it must be the one the residual is
    # measured against.
    #
    # Asserted rather than trusted, and the assertion is not redundant with the contract's
    # own balance check: that one proves the residual matches the decomposition, while this
    # one proves the decomposition closes to *nothing*. A future rule that closes a gap
    # only approximately would satisfy the first and fail here, which is the whole point --
    # a resolved row must be fully accounted for, not merely self-consistent.
    residual = credit.amount_paise - decomposition.expected_credit_paise
    if residual != 0:
        raise AssertionError(
            f"internal: {credit.credit_id} was explained by {explanation.rule} but "
            f"residuals to {residual}p -- an explanation must close the gap exactly"
        )

    return Verdict(
        credit.credit_id,
        Outcome.RESOLVED,
        settlement_ids=(settlement_id,),
        payment_ids=tuple(payment_ids),
        tier=tier,
        residual_paise=residual,
        credit_amount_paise=credit.amount_paise,
        decomposition=decomposition,
        note=_note_for_match(winner, narration, len(payment_ids), explanation, tier),
    )


if __name__ == "__main__":
    from datetime import date, datetime, timezone

    from .load import Credit as C
    from .load import Dataset as D
    from .load import Payment as P
    from .load import Settlement as S

    mon, tue = date(2026, 8, 10), date(2026, 8, 11)
    when = datetime(2026, 8, 10, 5, 34, 22, tzinfo=timezone.utc)

    def payment(pid: str, gross: int, method: str = "card") -> P:
        # The method is now load-bearing, not decoration: the fee rate is looked up by it,
        # so a residual cannot be explained from the amount alone.
        return P(pid, f"ord_{pid[4:]}", when, gross, method, "INR", "captured")

    def settlement(
        sid: str, on: date, net: int, tail: str = "0000", fee_paise: int = 0
    ) -> S:
        # fee_paise is settable so a test can prove the gate does NOT trust the declared
        # column. Phase 4 populates it for real; nothing on the match path reads it.
        return S(sid, on, net, fee_paise, 0, 0, f"XXXX{tail}")

    def credit(cid: str, on: date, amount: int, narration: str = "NEFT-RAZORPAYSOFT-XXXX8104") -> C:
        return C(cid, on, amount, narration)

    def dataset(payments, settlements, credits, items) -> D:
        return D(tuple(payments), tuple(settlements), tuple(credits), (), items)

    def verdict_of(ds: D, cid: str, **kwargs) -> Verdict:
        index = SettlementIndex(ds.settlements)
        target = next(c for c in ds.credits if c.credit_id == cid)
        return resolve_credit(target, index, ds, **kwargs)

    # --- the happy path: exactly one candidate ----------------------------
    ds = dataset(
        [payment("pay_0001", 85358)],
        [settlement("setl_0005", mon, 85358, "8104")],
        [credit("C0001", mon, 85358)],
        {"setl_0005": ("pay_0001",)},
    )
    v = verdict_of(ds, "C0001")
    assert v.outcome is Outcome.RESOLVED
    assert v.settlement_ids == ("setl_0005",)
    assert v.payment_ids == ("pay_0001",), "the payment set comes from settlement_items"
    assert v.payment_set == frozenset({"pay_0001"})
    assert v.tier == TIER
    assert v.residual_paise == 0, "an exact match residuals to zero"
    assert v.reason is None
    assert "agrees" in (v.note or ""), v.note
    # Step 5: the proof is in the output. A zero-deduction row still decomposes -- it says
    # the credit equals the gross, which is a claim, not the absence of one.
    assert v.credit_amount_paise == 85358, "the credit amount must be carried"
    assert v.decomposition is not None
    assert v.decomposition.gross_paise == 85358
    assert v.decomposition.deductions_paise == 0
    assert v.decomposition.expected_credit_paise == 85358
    assert v.decomposition.rule == "no deduction", v.decomposition.rule

    # --- the residual gate: matched is not proven (Phase 4 step 2) ---------
    # A credit 500 paise short of its payment's gross agrees on date and amount, so
    # Phase 3 resolved it and reported residual -500. Phase 4 refuses it: money is
    # unaccounted for, and ASSUMPTIONS.md #25 makes matched-but-unproven an exception.
    #
    # This assertion is inverted from the Phase 3 version on purpose. Until step 4
    # derives a fee, *any* gap is unexplained, so the gate is strict by construction
    # rather than by tolerance -- there is no band to widen and nothing to tune.
    short = dataset(
        [payment("pay_0001", 85358)],
        [settlement("setl_0005", mon, 84858, "8104")],
        [credit("C0001", mon, 84858)],
        {"setl_0005": ("pay_0001",)},
    )
    v = verdict_of(short, "C0001")
    assert v.outcome is Outcome.EXCEPTION, (
        f"a 500p gap must not resolve, got {v.outcome} -- an unproven match is how a "
        f"reconciliation tool issues a false clean bill of health"
    )
    assert v.reason is Reason.UNEXPLAINED_RESIDUAL, v.reason
    # The contract forbids ids and a residual on an abstention, so the whole diagnostic
    # lives in the note -- and it must still name the size and direction of the gap, or
    # the exception is unactionable and a human cannot triage it.
    assert v.settlement_ids == () and v.payment_ids == (), "an abstention is not a match"
    assert v.residual_paise is None, "the contract forbids a residual on an EXCEPTION"
    assert "500p short" in (v.note or ""), v.note
    assert "setl_0005" in (v.note or ""), "the note must name the settlement it matched"
    # Step 4: the note must say how the model *failed*, not merely that it did. A 2% card
    # fee on 85,358p is 1,707p plus 307p GST, and Phase 6 adds 85p of TDS at 10bps -- 2,099p
    # in all, far more than the 500p actually withheld. So this row is the rate table being
    # wrong for it, and the note has to say that rather than report a negative remainder.
    #
    # The number moved from 2,014p to 2,099p when ``--tds`` arrived, and the diagnostic quotes
    # ``derived.total_paise`` -- *everything* the declared rates say was withheld. That is the
    # right quantity for this sentence: the reader is being told the model over-predicts, so
    # the model has to be reported in full rather than net of the term that grew it.
    assert "predict a larger deduction of 2099p" in (v.note or ""), v.note
    assert "-1599p" not in (v.note or ""), (
        f"a negative remainder leaked into the diagnostic instead of the over-prediction "
        f"wording: {v.note}"
    )

    # The other failure direction: a gap *bigger* than the model accounts for. This one
    # points at a missing rule (Phase 6's refunds, reserves and TDS all look like this),
    # and the note must quantify the leftover so a reader can tell the two cases apart.
    under = dataset(
        [payment("pay_0001", 85358)],
        [settlement("setl_0005", mon, 82358, "8104")],
        [credit("C0001", mon, 82358)],
        {"setl_0005": ("pay_0001",)},
    )
    v = verdict_of(under, "C0001")
    assert v.outcome is Outcome.EXCEPTION and v.reason is Reason.UNEXPLAINED_RESIDUAL
    assert "3000p short" in (v.note or ""), v.note
    assert "account for 2099p" in (v.note or ""), v.note
    assert "leaving 901p unexplained" in (v.note or ""), v.note

    # And in the other direction: an over-credit is equally unproven, not a bonus.
    over = dataset(
        [payment("pay_0001", 85358)],
        [settlement("setl_0005", mon, 85858, "8104")],
        [credit("C0001", mon, 85858)],
        {"setl_0005": ("pay_0001",)},
    )
    v = verdict_of(over, "C0001")
    assert v.outcome is Outcome.EXCEPTION and v.reason is Reason.UNEXPLAINED_RESIDUAL
    assert "500p" in (v.note or "") and "exceeds" in (v.note or ""), v.note

    # A declared fee column must NOT close the gap. This is the trap the gate exists to
    # keep shut: if the residual subtracted settlements.csv's own fee_paise, then
    # --fees would resolve at 100% in Phase 4 with no fee model written and no number
    # moving to say one was missing. Explaining a gap means applying a rule, not
    # trusting the counterparty's arithmetic.
    #
    # The declared 500p here matches the gap exactly, so reading the column would resolve
    # this row. The derived rates say 2,099p, so nothing closes and the row abstains.
    declared = dataset(
        [payment("pay_0001", 85358)],
        [settlement("setl_0005", mon, 84858, "8104", fee_paise=500)],
        [credit("C0001", mon, 84858)],
        {"setl_0005": ("pay_0001",)},
    )
    v = verdict_of(declared, "C0001")
    assert v.outcome is Outcome.EXCEPTION, (
        "a declared fee_paise must not explain the residual -- only a derived rule can"
    )

    # --- step 4: a derived fee closes the gap and the row resolves ----------
    # 2% of 85,358p is 1,707.16p -> 1,707p half-up, and 18% GST on that is 307.26p -> 307p.
    # A credit of 83,344p is exactly 2,014p short, so the declared rule accounts for all of
    # it. These are the numbers ASSUMPTIONS.md #5-#9 commit to; the arithmetic is
    # money.mul_bps, shared with the generator so the two cannot disagree by a paisa.
    #
    # **This fixture withholds no tax, and it still resolves under exactly one rule** -- which
    # is what makes the four-rule set safe rather than merely bigger. The gap is fee-and-GST,
    # so ``FEE_AND_GST`` closes it while ``FEE_GST_TDS`` over-predicts by the 85p of tax and
    # ``TDS_ONLY`` under-predicts by everything else. A rule set that closed this gap two ways
    # would have made every ``--fees``-without-``--tds`` row ambiguous, and the assertion below
    # that ``dec.tds_paise == 0`` is what pins the term to the rule that claimed it.
    CARD_FEE, CARD_GST = 1707, 307
    priced = dataset(
        [payment("pay_0001", 85358)],
        # net and fee_paise are deliberately *wrong* here -- 0 fee on a settlement whose
        # credit is plainly 2,014p short. The row must still resolve, which proves the
        # declared columns are neither trusted nor required: the rule is the evidence.
        [settlement("setl_0005", mon, 83344, "8104", fee_paise=0)],
        [credit("C0001", mon, 83344)],
        {"setl_0005": ("pay_0001",)},
    )
    v = verdict_of(priced, "C0001")
    assert v.outcome is Outcome.RESOLVED, (
        f"a gap equal to the derived fee must resolve, got {v.reason} -- otherwise the fee "
        f"model exists and explains nothing"
    )
    assert v.residual_paise == 0, "an explained gap residuals to zero, by construction"
    assert v.payment_ids == ("pay_0001",)
    # The note names the rule and its components, so a coincidental balance is visible in
    # the output rather than hidden inside a coverage percentage.
    assert "gateway fee + GST" in (v.note or ""), v.note
    assert f"fee {CARD_FEE}p" in (v.note or "") and f"GST {CARD_GST}p" in (v.note or ""), v.note
    assert f"{CARD_FEE + CARD_GST}p" in (v.note or ""), v.note

    # --- step 5: the components, term by term -----------------------------
    # Not the total. A fee that is right in sum because the GST was wrong in the opposite
    # direction is a real arithmetic error, and it lands on exactly the same total -- so a
    # check on ``deductions_paise`` alone would pass on it. This is why the verdict carries
    # six named terms and why the scorer compares them individually.
    dec = v.decomposition
    assert dec is not None
    assert dec.gross_paise == 85358
    assert dec.fee_paise == CARD_FEE, dec
    assert dec.gst_paise == CARD_GST, dec
    assert dec.deductions_paise == CARD_FEE + CARD_GST
    assert dec.expected_credit_paise == 83344
    assert dec.rule == "gateway fee + GST at declared rates", dec.rule
    # Phase 6's terms are present at zero, so filling them is a change of value not shape.
    assert (dec.tds_paise, dec.refunds_paise, dec.reserve_paise) == (0, 0, 0)
    # The published arithmetic closes against the credit the matcher actually read. This is
    # the sum a reader re-runs, and it is what gate 10 checks on every resolved row.
    assert v.credit_amount_paise == 83344
    assert v.credit_amount_paise - dec.expected_credit_paise == v.residual_paise == 0
    # The declared columns took no part in it: this settlement states a 0p fee while the
    # decomposition publishes 1,707p. The proof is the rule, not the counterparty's number.
    assert dec.fee_paise != 0 and priced.settlements[0].fee_paise == 0

    # --- Phase 6 step 3: two rules, one gap, and the old code resolved it ---
    # **A reproduction, not a description.** This fixture is the reason the arity change landed
    # before any new rule: under the old single-optional ``explain_gap`` it RESOLVED, because
    # first-hit returned whichever rule was tried first and silently discarded the other. It
    # now abstains, and the reason code distinguishes it from a subset-search ambiguity.
    #
    # The rates are overridden to reach it, and that is a property of the arithmetic rather
    # than a contrivance: a genuine collision needs ``fee + gst == tds`` with both non-zero,
    # and at the declared rates fee-and-GST is 236bps effective against TDS's 10, so they can
    # never meet. 9bps on a Rs 105 gross gives a 9p fee plus 2p GST against 11p of tax --
    # two different splits of one 11p gap. ``--fee-bps`` exists precisely so a run can be
    # re-pointed at a negotiated rate, so this is a live branch rather than a defensive one.
    cheap = FeeSchedule(fee_bps_by_method={"card": 9})
    collide = dataset(
        [payment("pay_0001", 10_500)],
        [settlement("setl_0005", mon, 10_489, "8104")],
        [credit("C0001", mon, 10_489)],
        {"setl_0005": ("pay_0001",)},
    )
    v = verdict_of(collide, "C0001", schedule=cheap)
    assert v.outcome is Outcome.EXCEPTION, (
        f"two rules close this 11p gap with different splits, so committing to one is a coin "
        f"flip on the decomposition -- got {v.outcome} with {v.decomposition}"
    )
    assert v.reason is Reason.AMBIGUOUS_ADJUSTMENT, (
        f"a rule collision must not borrow the subset-search code: {v.reason}"
    )
    # The note has to name *both* readings and their components, or a reader cannot tell which
    # two rules disagreed -- and with the settlement, the payment set and the total all known,
    # the components are the only thing left in question.
    assert "TDS at the declared rate" in (v.note or ""), v.note
    assert "gateway fee + GST at declared rates" in (v.note or ""), v.note
    assert "TDS 11p" in (v.note or "") and "fee 9p" in (v.note or ""), v.note
    # An abstention is not a match, and the contract forbids it carrying either.
    assert v.settlement_ids == () and v.payment_ids == () and v.residual_paise is None
    # ...and the same fixture at the *declared* rates resolves, which is what proves the
    # abstention above is caused by the collision rather than by the amounts.
    v = verdict_of(collide, "C0001")
    assert v.outcome is Outcome.RESOLVED and v.decomposition is not None, v.reason
    assert v.decomposition.tds_paise == 11 and v.decomposition.rule == "TDS at the declared rate"

    # One paisa either side of the derived fee must not resolve. There is no tolerance band
    # to widen -- that is the whole point of the gate being exact.
    for off in (-1, 1):
        near = dataset(
            [payment("pay_0001", 85358)],
            [settlement("setl_0005", mon, 83344 + off, "8104")],
            [credit("C0001", mon, 83344 + off)],
            {"setl_0005": ("pay_0001",)},
        )
        v = verdict_of(near, "C0001")
        assert v.outcome is Outcome.EXCEPTION, (
            f"a gap {-off:+d}p off the derived fee resolved -- the gate has a tolerance "
            f"band, and a tolerance band is a place for a wrong match to hide"
        )

    # A zero-rated method settles at its gross, so it resolves with no fee model at all.
    # This is why "the residual moved" has to be measured per method: on a run where every
    # payment were pos_upi, the fee model would be dead code and coverage would look perfect.
    zero_rated = dataset(
        [payment("pay_0001", 197600, "pos_upi")],
        [settlement("setl_0005", mon, 197600, "8104")],
        [credit("C0001", mon, 197600)],
        {"setl_0005": ("pay_0001",)},
    )
    v = verdict_of(zero_rated, "C0001")
    assert v.outcome is Outcome.RESOLVED and v.residual_paise == 0
    assert "no deduction" in (v.note or ""), v.note
    assert "credit equals gross" in (v.note or ""), v.note
    # A zero-rated row's decomposition is all zeros below the gross, and its rule says so
    # by name. Worth asserting because this is the row that would still resolve if the fee
    # model were deleted -- the rule name is what distinguishes "nothing was withheld" from
    # "nothing was checked", and a report that cannot tell them apart cannot claim the fee
    # model earned any coverage at all.
    assert v.decomposition is not None
    assert v.decomposition.deductions_paise == 0
    assert v.decomposition.rule == "no deduction", v.decomposition.rule
    assert v.decomposition.gross_paise == v.credit_amount_paise == 197600

    # A method the rate table does not price cannot be reconciled at all, and the note must
    # point at the rate table rather than at a missing rule. An unknown method defaulting
    # to free would silently model it as zero-rated, which is a wrong answer wearing a
    # match's label.
    unknown = dataset(
        [payment("pay_0001", 85358, "crypto")],
        [settlement("setl_0005", mon, 83344, "8104")],
        [credit("C0001", mon, 83344)],
        {"setl_0005": ("pay_0001",)},
    )
    v = verdict_of(unknown, "C0001")
    assert v.outcome is Outcome.EXCEPTION and v.reason is Reason.UNEXPLAINED_RESIDUAL
    assert "no rate is declared for crypto" in (v.note or ""), v.note
    # ...but an unpriced method whose credit equals its gross needs no rate: nothing was
    # withheld, so there is no gap to price. Deliberate ordering inside fees.explain_gap.
    unknown_exact = dataset(
        [payment("pay_0001", 85358, "crypto")],
        [settlement("setl_0005", mon, 85358, "8104")],
        [credit("C0001", mon, 85358)],
        {"setl_0005": ("pay_0001",)},
    )
    assert verdict_of(unknown_exact, "C0001").outcome is Outcome.RESOLVED

    # A batched settlement's fee is the sum of per-member fees, each rounded at the paisa,
    # not the rate applied to the batch total. Here the two genuinely differ: 25p at 200bps
    # is half a paisa, which rounds up to 1p *per member* (2p total), while the 50p batch
    # total would round to 1p. The credit that resolves is the one the per-member rule
    # predicts, and a paisa is exactly the size of error that gets waved away as noise.
    split = dataset(
        [payment("pay_0001", 25), payment("pay_0002", 25)],
        [settlement("setl_0005", mon, 48, "8104")],
        [credit("C0001", mon, 48)],
        {"setl_0005": ("pay_0001", "pay_0002")},
    )
    v = verdict_of(split, "C0001")
    assert v.outcome is Outcome.RESOLVED, (
        f"got {v.reason} -- per-member rounding is the declared rule (ASSUMPTIONS.md), "
        f"and this is the case where it differs from pricing the batch total"
    )
    assert v.payment_set == frozenset({"pay_0001", "pay_0002"})
    batch_rounded = dataset(
        [payment("pay_0001", 25), payment("pay_0002", 25)],
        [settlement("setl_0005", mon, 49, "8104")],
        [credit("C0001", mon, 49)],
        {"setl_0005": ("pay_0001", "pay_0002")},
    )
    assert verdict_of(batch_rounded, "C0001").outcome is Outcome.EXCEPTION, (
        "a batch-total rounding resolved -- the two rules are no longer distinguishable "
        "and one of them is wrong"
    )

    # A run may state different rates: they are an assumption about a counterparty, not a
    # constant of the universe. At 0bps for card, the priced row above stops balancing.
    from .fees import FeeSchedule

    assert verdict_of(priced, "C0001", schedule=FeeSchedule({"card": 0})).outcome is (
        Outcome.EXCEPTION
    ), "the schedule is not being threaded through -- resolve_credit ignored it"

    # A batched settlement residuals against the *sum* of its members.
    batched = dataset(
        [payment("pay_0001", 60000), payment("pay_0002", 25358)],
        [settlement("setl_0005", mon, 85358, "8104")],
        [credit("C0001", mon, 85358)],
        {"setl_0005": ("pay_0001", "pay_0002")},
    )
    v = verdict_of(batched, "C0001")
    assert v.payment_set == frozenset({"pay_0001", "pay_0002"})
    assert v.residual_paise == 0
    # The published gross is the sum of the *members*, not the settlement's declared net.
    # They agree here, which is exactly why it is worth asserting the provenance: a
    # decomposition built from settlements.csv would be indistinguishable on this row and
    # wrong on any row where the counterparty's own arithmetic is off.
    assert v.decomposition is not None
    assert v.decomposition.gross_paise == 60000 + 25358

    # --- no candidate ------------------------------------------------------
    none_ds = dataset(
        [payment("pay_0001", 85358)],
        [settlement("setl_0005", mon, 85358, "8104")],
        [credit("C0002", tue, 85358)],   # right amount, wrong day
        {"setl_0005": ("pay_0001",)},
    )
    v = verdict_of(none_ds, "C0002")
    assert v.outcome is Outcome.EXCEPTION and v.reason is Reason.NO_CANDIDATE
    assert v.payment_ids == () and v.settlement_ids == ()
    assert v.tier is None and v.residual_paise is None, (
        "an abstention must carry no tier and no residual -- there is no claimed "
        "decomposition for a residual to be the remainder of"
    )

    # --- two candidates: ambiguous, and it must not pick one ---------------
    ambiguous = dataset(
        [payment("pay_0001", 85358), payment("pay_0002", 85358)],
        [settlement("setl_0005", mon, 85358, "8104"),
         settlement("setl_0009", mon, 85358, "4451")],
        [credit("C0001", mon, 85358)],
        {"setl_0005": ("pay_0001",), "setl_0009": ("pay_0002",)},
    )
    v = verdict_of(ambiguous, "C0001")
    assert v.outcome is Outcome.EXCEPTION
    assert v.reason is Reason.AMBIGUOUS_DUPLICATE_AMOUNT
    assert "setl_0005" in (v.note or "") and "setl_0009" in (v.note or "")
    assert v.payment_ids == (), "an abstention that names an answer is a match"

    # --- two candidates under a posting lag: abstain, never prefer the closer ---
    # Phase 4's geometry. The true settlement sits one business day behind its credit;
    # a same-day settlement sharing the amount is an impostor. The retired tie-break
    # (blocking.narrow_by_date_distance) kept the *minimum* distance and so committed to
    # the impostor at full confidence -- 5-10 wrong matches per 1000 rows, with coverage
    # unmoved. This asserts tier1 no longer calls it.
    lagged = dataset(
        [payment("pay_0001", 85358), payment("pay_0002", 85358)],
        [settlement("setl_0005", tue, 85358, "8104"),   # same day as the credit: impostor
         settlement("setl_0009", mon, 85358, "4451")],  # one day behind: the true one
        [credit("C0001", tue, 85358)],
        {"setl_0005": ("pay_0001",), "setl_0009": ("pay_0002",)},
    )
    v = verdict_of(lagged, "C0001", window_days=1)
    assert v.outcome is Outcome.EXCEPTION, (
        f"got {v.outcome} -- a same-day settlement is not more likely than one at the "
        f"posting lag, and committing to it is a wrong match wearing a match's label"
    )
    assert v.reason is Reason.AMBIGUOUS_DUPLICATE_AMOUNT, v.reason
    assert v.payment_ids == () and v.settlement_ids == ()
    # The note must name both distances, or a human cannot triage what the matcher saw.
    assert "+0bd" in (v.note or "") and "+1bd" in (v.note or ""), v.note
    assert "posting lag" in (v.note or ""), v.note
    # At window 0 the same data resolves cleanly: only the same-day settlement is in
    # range, so widening the window is what created the ambiguity -- not the lag itself.
    v0 = verdict_of(lagged, "C0001", window_days=0)
    assert v0.outcome is Outcome.RESOLVED and v0.settlement_ids == ("setl_0005",)

    # A settlement dated AFTER its credit is physically impossible and must not be a
    # candidate at any width -- money moves forward. Without the forward-only window this
    # returned a candidate, and paired with a real one it produced an unresolvable tie.
    backward = dataset(
        [payment("pay_0001", 85358)],
        [settlement("setl_0005", tue, 85358, "8104")],   # settles the day AFTER the credit
        [credit("C0001", mon, 85358)],
        {"setl_0005": ("pay_0001",)},
    )
    v = verdict_of(backward, "C0001", window_days=5)
    assert v.outcome is Outcome.EXCEPTION and v.reason is Reason.NO_CANDIDATE, (
        f"got {v.reason} -- a credit cannot be paid by a settlement dated after it, "
        f"however wide the window"
    )

    # --- a matched settlement with no declared membership: tier 2 ---------
    # The shape --settlement-report-late creates, and the four outcomes it can have. Before
    # Phase 5 step 6 every one of these abstained with MEMBERSHIP_UNDECLARED; the search is
    # what turns the first into a match, and the codes below are what keeps the other three
    # from being scored as the same thing.

    # (1) exactly one subset sums to the credit -> a tier 2 match. The membership was never
    # declared, so the payment list on this verdict was *derived*, and the tier field is the
    # only thing distinguishing it from a tier 1 row that read the list off a file.
    batched = dataset(
        [payment("pay_0001", 40_000), payment("pay_0002", 60_000)],
        [settlement("setl_0005", mon, 100_000, "8104")],
        [credit("C0001", mon, 100_000)],
        {},   # settlement_items.csv withheld
    )
    v = verdict_of(batched, "C0001")
    assert v.outcome is Outcome.RESOLVED, f"{v.reason}: {v.note}"
    assert v.tier == TIER_2 == 2, v.tier
    assert v.payment_ids == ("pay_0001", "pay_0002"), v.payment_ids
    assert v.settlement_ids == ("setl_0005",)
    assert v.residual_paise == 0, "a searched membership is proven by the same arithmetic"
    assert v.decomposition is not None and v.decomposition.gross_paise == 100_000
    assert "tier 2 subset" in (v.note or ""), v.note
    assert "searched" in (v.note or ""), v.note
    # The note must not claim the ids were declared, and must not read as a tier 1 row.
    assert "tier 1" not in (v.note or ""), v.note

    # (2) two subsets sum to it -> AMBIGUOUS_MULTI_SUBSET, which *is* an honest refusal: the
    # search ran and the data could not separate two answers. Two payments of equal value are
    # the simplest case and a real one -- nothing in these inputs can tell them apart.
    twins = dataset(
        [payment("pay_0001", 50_000), payment("pay_0002", 50_000)],
        [settlement("setl_0005", mon, 50_000, "8104")],
        [credit("C0001", mon, 50_000)],
        {},
    )
    v = verdict_of(twins, "C0001")
    assert v.outcome is Outcome.EXCEPTION
    assert v.reason is Reason.AMBIGUOUS_MULTI_SUBSET, v.reason
    assert v.tier is None, "an abstention claims no tier"
    assert v.payment_ids == () and v.settlement_ids == (), (
        "the verdict contract forbids an abstention from carrying ids, so it can never be "
        "mistaken for a match by the scorer's set-equality join"
    )
    assert "pay_0001" in (v.note or "") and "pay_0002" in (v.note or ""), (
        f"both candidate sets belong in the note -- an ambiguity a human can see is one they "
        f"can resolve from a source this matcher does not have: {v.note}"
    )

    # (3) no subset sums to it -> NO_CANDIDATE, and emphatically not a wider search. The
    # window stays fixed at the settlement cycle; widening it until something adds up is how
    # a missing payment becomes a false match.
    unreachable = dataset(
        [payment("pay_0001", 40_000)],
        [settlement("setl_0005", mon, 99_999, "8104")],
        [credit("C0001", mon, 99_999)],
        {},
    )
    v = verdict_of(unreachable, "C0001")
    assert v.outcome is Outcome.EXCEPTION
    assert v.reason is Reason.NO_CANDIDATE, v.reason
    assert "window stays fixed" in (v.note or ""), v.note

    # (4) the pool is over the cap -> refused before the search runs, and the code is
    # MEMBERSHIP_UNDECLARED rather than an ABSTENTION_REASONS one (Phase 5 decision 8). The
    # difference is worth more than a wording slip: AMBIGUOUS_MULTI_SUBSET is inside the set
    # the acceptance gates accept as *honest* refusal, so emitting it for a row that was
    # never searched would score a missing capability as a correct abstention. "I looked and
    # could not separate two answers" and "I never looked" are different facts.
    #
    # A cap never exceeded on dev seeds is a cap untested, which is why this is a unit
    # fixture: measured pool maxima are 20 at n=200 and 57-63 at n=1000, so nothing in the
    # dev range reaches it.
    crowd = [payment(f"pay_{i:04d}", 1_000 + i) for i in range(TIER2_MAX_POOL + 1)]
    over = dataset(
        crowd,
        [settlement("setl_0005", mon, 77_777, "8104")],
        [credit("C0001", mon, 77_777)],
        {},
    )
    v = verdict_of(over, "C0001")
    assert v.outcome is Outcome.EXCEPTION
    assert v.reason is Reason.MEMBERSHIP_UNDECLARED, v.reason
    assert v.reason is not Reason.AMBIGUOUS_MULTI_SUBSET, (
        "decision 8: a refusal to search must not borrow the code reserved for a search that "
        "found two answers -- that one is graded as an honest abstention"
    )
    assert str(TIER2_MAX_POOL) in (v.note or ""), (
        f"the note must name the bound, so that 'what happens at 10,000 records?' has a "
        f"number for an answer rather than a shrug: {v.note}"
    )
    assert f"{len(crowd)} payments" in (v.note or ""), v.note
    # One payment fewer and the same shape searches rather than refusing, so it is the cap
    # that refused it and not something else.
    assert len(crowd) - 1 == TIER2_MAX_POOL
    under = dataset(
        crowd[:-1],
        [settlement("setl_0005", mon, 77_777, "8104")],
        [credit("C0001", mon, 77_777)],
        {},
    )
    assert verdict_of(under, "C0001").reason is not Reason.MEMBERSHIP_UNDECLARED

    # --- the pool excludes what another settlement claims, and that is a partition fact ---
    # Reading settlement_items.csv at all is the trap here, so the distinction has to be
    # sharp. Settlement A *declares* pay_0001; settlement B declares nothing. Both payments
    # have the same gross, so without the exclusion B has two candidate subsets and must
    # abstain; with it, pay_0001 is unavailable -- already settled elsewhere -- and B resolves
    # to the one payment left.
    #
    # That is a fact about a partition, not a hint about B's membership: nothing here reads
    # B's own rows, which is the premise of the problem. Verified on real data at n up to
    # 1000: no payment appears in more than one settlement, and the exclusion never removed a
    # true member of a withheld settlement.
    partitioned = dataset(
        [payment("pay_0001", 50_000), payment("pay_0002", 50_000)],
        [
            settlement("setl_0004", mon, 12_345, "7777"),
            settlement("setl_0005", mon, 50_000, "8104"),
        ],
        [credit("C0001", mon, 50_000)],
        {"setl_0004": ("pay_0001",)},   # A declares its member; B is withheld
    )
    v = verdict_of(partitioned, "C0001")
    assert v.outcome is Outcome.RESOLVED, f"{v.reason}: {v.note}"
    assert v.tier == TIER_2 and v.payment_ids == ("pay_0002",), v.payment_ids
    # ... and with A's declaration removed the exclusion has nothing to act on, so the same
    # data becomes genuinely ambiguous. This is the exclusion being load-bearing, which is why
    # its justification has to be a partition argument rather than a convenience.
    v = verdict_of(
        dataset(partitioned.payments, partitioned.settlements, partitioned.credits, {}),
        "C0001",
    )
    assert v.outcome is Outcome.EXCEPTION and v.reason is Reason.AMBIGUOUS_MULTI_SUBSET, (
        f"got {v.reason} -- with nothing declared anywhere, two equal payments are not "
        f"separable and the honest answer is an abstention"
    )

    # --- tier 2 must not read the settlement's declared fee_paise ---------
    # The same discipline tier 1 is already held to, extended to a searched membership. The
    # declared column carries a number that would close a gap if anything read it; the search
    # resolves on the gross hypothesis instead, and the decomposition it publishes must show
    # the fee this matcher *derived* (zero here), never the one the file stated.
    poisoned = dataset(
        [payment("pay_0001", 40_000), payment("pay_0002", 60_000)],
        [settlement("setl_0005", mon, 100_000, "8104", fee_paise=25_000)],
        [credit("C0001", mon, 100_000)],
        {},
    )
    v = verdict_of(poisoned, "C0001")
    assert v.outcome is Outcome.RESOLVED and v.tier == TIER_2
    assert v.decomposition is not None
    assert v.decomposition.fee_paise == 0 and v.decomposition.gst_paise == 0, (
        f"the declared fee_paise leaked into a tier 2 decomposition: {v.decomposition}"
    )
    assert v.residual_paise == 0

    # --- the net-of-fees hypothesis, which is why there are two ----------
    # A run without --fees credits the gross; a run with it credits the net. The search cannot
    # know which, so it tries both and counts the distinct sets -- the same shape explain_gap
    # uses for its two rules. Without the second hypothesis every fee-bearing batch would find
    # nothing while the rate table sat there looking correct.
    _members = [(40_000, "card"), (60_000, "card")]
    _deduction = derive(_members)
    _net = 100_000 - _deduction.fee_paise - _deduction.gst_paise
    assert _deduction.fee_paise > 0, "the fixture needs a schedule that actually charges"
    netted = dataset(
        [payment("pay_0001", 40_000), payment("pay_0002", 60_000)],
        [settlement("setl_0005", mon, _net, "8104")],
        [credit("C0001", mon, _net)],
        {},
    )
    v = verdict_of(netted, "C0001")
    assert v.outcome is Outcome.RESOLVED, f"{v.reason}: {v.note}"
    assert v.tier == TIER_2 and v.payment_ids == ("pay_0001", "pay_0002"), v.payment_ids
    assert v.decomposition is not None
    assert v.decomposition.fee_paise == _deduction.fee_paise, v.decomposition
    assert v.decomposition.gst_paise == _deduction.gst_paise, v.decomposition
    assert v.residual_paise == 0, (
        "the searched set went through the same explain_gap path as a declared one, so the "
        "residual must close to nothing"
    )

    # --- an ambiguity under EITHER hypothesis is an ambiguity -------------
    # The regression guard for the precedence bug step 6 shipped and then measured. This is
    # credit C0277 (seed 1, n=1000) in miniature, and it is the case where the two readings of
    # the credit disagree about whether an answer even exists:
    #
    #   * pay_0001 grosses 100,000p and nets to exactly the credit, so it is the true single
    #     member. pay_0002 and pay_0003 net to the same total, so the NET reading sees two
    #     answers and cannot separate them.
    #   * pay_0004 and pay_0005 GROSS to the credit, so the gross reading -- which is simply
    #     false on a run where fees were charged -- finds exactly one set.
    #
    # The first version of ``_search_membership`` kept the found set and the ambiguity apart
    # and abstained on ambiguity only when nothing had been found, so that unique-but-wrong
    # gross answer overrode the known ambiguity and the row resolved. One wrong match in 627
    # end to end, invisible in every unit test, and correctness is the axis this project
    # refuses to trade -- hence a fixture rather than a comment.
    #
    # The amounts were searched for rather than derived: per-paisa rounding is what makes the
    # two readings disagree, so a hand-picked set would not reproduce it.
    AMBIGUOUS_TOTAL = 97_640
    two_readings = dataset(
        [
            payment("pay_0001", 100_000),   # true single member: its NET is the credit
            payment("pay_0002", 20_000),    # pay_0002 + pay_0003 nets also reach the credit
            payment("pay_0003", 80_000),
            payment("pay_0004", 40_000),    # pay_0004 + pay_0005 GROSSES reach the credit
            payment("pay_0005", 57_640),
        ],
        [settlement("setl_0005", mon, AMBIGUOUS_TOTAL, "8104")],
        [credit("C0001", mon, AMBIGUOUS_TOTAL)],
        {},
    )
    # The premise, asserted so the fixture cannot rot into a different shape silently.
    _one = derive([(100_000, "card")])
    assert 100_000 - _one.fee_paise - _one.gst_paise == AMBIGUOUS_TOTAL
    v = verdict_of(two_readings, "C0001")
    assert v.outcome is Outcome.EXCEPTION, (
        f"the net reading sees two answers, so the only honest verdict is an abstention -- a "
        f"unique answer under the gross reading must not override it. Got {v.outcome} "
        f"claiming {v.payment_ids}"
    )
    assert v.reason is Reason.AMBIGUOUS_MULTI_SUBSET, v.reason
    assert v.tier is None and v.payment_ids == ()
    # ... and the wrong answer the old precedence would have committed to is specifically the
    # gross-only set, which is what makes this a precedence bug rather than a search bug.
    assert isinstance(
        tier2_resolve([Member("pay_0004", 40_000), Member("pay_0005", 57_640)],
                      AMBIGUOUS_TOTAL),
        ExactlyOne,
    ), "the gross pair must resolve on its own, or the fixture is not reproducing the bug"

    # --- tier 2 is deterministic and order-independent --------------------
    # The search returns a frozenset and nothing on the path to output may iterate one. Two
    # runs over the same data, and the same data with its payment rows reversed, must produce
    # the identical verdict.
    first = verdict_of(batched, "C0001")
    assert first.payment_ids == verdict_of(batched, "C0001").payment_ids
    reversed_rows = dataset(
        tuple(reversed(batched.payments)), batched.settlements, batched.credits, {}
    )
    assert verdict_of(reversed_rows, "C0001").payment_ids == first.payment_ids, (
        "the verdict moved when the payment rows were reordered"
    )

    # --- the narration must not influence any decision field --------------
    # The guard behind decision 2, asserted at the unit level as well as in gate 9.
    DECISION = ("credit_id", "outcome", "settlement_ids", "payment_ids", "tier",
                "reason", "residual_paise")

    def decision_of(v: Verdict) -> tuple:
        return tuple(getattr(v, f) for f in DECISION)

    base = dataset(
        [payment("pay_0001", 85358)],
        [settlement("setl_0005", mon, 85358, "8104")],
        [credit("C0001", mon, 85358, "NEFT-RAZORPAYSOFT-XXXX8104")],
        {"setl_0005": ("pay_0001",)},
    )
    reference = decision_of(verdict_of(base, "C0001"))
    for narration in ("X", "", "IMPS CR/RAZORPAY SOFTWARE/9999",
                      "NEFT-RZRPAY-1111", "TOTALLY UNRELATED TEXT"):
        altered = dataset(
            base.payments, base.settlements,
            [credit("C0001", mon, 85358, narration)], base.items,
        )
        assert decision_of(verdict_of(altered, "C0001")) == reference, (
            f"narration {narration!r} changed a decision field -- the UTR shortcut got "
            f"in, and this matcher would score 100% while never doing the arithmetic"
        )
    # The note *does* change, and that is correct: it is corroboration, not a decision.
    blanked = dataset(base.payments, base.settlements,
                      [credit("C0001", mon, 85358, "X")], base.items)
    assert verdict_of(blanked, "C0001").note != verdict_of(base, "C0001").note
    assert "unparsed" in (verdict_of(blanked, "C0001").note or "")

    # --- a disagreeing UTR is reported, and still changes nothing ----------
    mismatch = dataset(
        [payment("pay_0001", 85358)],
        [settlement("setl_0005", mon, 85358, "9999")],
        [credit("C0001", mon, 85358, "NEFT-RAZORPAYSOFT-XXXX8104")],
        {"setl_0005": ("pay_0001",)},
    )
    v = verdict_of(mismatch, "C0001")
    assert v.outcome is Outcome.RESOLVED, "a UTR disagreement must not block a match"
    assert "DISAGREES" in (v.note or ""), v.note

    # --- the committed run, end to end through this module -----------------
    from pathlib import Path

    from .load import load

    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    if (data_dir / "bank_statement.csv").exists():
        real = load(data_dir)
        index = SettlementIndex(real.settlements)
        verdicts = [resolve_credit(c, index, real) for c in real.credits]
        resolved = [v for v in verdicts if v.outcome is Outcome.RESOLVED]
        assert len(verdicts) == len(real.credits), "one verdict per bank row, always"
        assert len(resolved) == len(real.credits), (
            f"only {len(resolved)}/{len(real.credits)} resolved -- clean mode must be "
            f"fully resolvable, so this is a matcher bug, not a property of the data"
        )
        assert all(v.residual_paise == 0 for v in resolved), "clean mode has no fees"
        assert all(v.tier == TIER for v in resolved)
        # Every settlement claimed at most once: a 1:1:1 month cannot reuse one.
        claimed = [sid for v in resolved for sid in v.settlement_ids]
        assert len(set(claimed)) == len(claimed), "a settlement was matched twice"
        print(
            f"tier1.py self-check ok  ({len(resolved)}/{len(real.credits)} resolved on "
            f"the committed run, all residuals zero)"
        )
    else:
        print("tier1.py self-check ok  (no committed data/ to cross-read)")
