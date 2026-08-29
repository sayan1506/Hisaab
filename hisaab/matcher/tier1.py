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
from ..common.money import mul_bps
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

#: The currency these books are kept in. A payment captured in anything else has a
#: ``gross_paise`` fixed at the **capture-day** rate, while its settlement's net and its bank
#: credit reflect the **settlement-day** rate -- so the recorded gross is stale by an amount no
#: input file declares. Phase 8's ``--fx``.
#:
#: A *declared assumption* in the same sense as ``SETTLEMENT_CYCLE_DAYS`` and the fee rates in
#: ``fees.py``: read off the merchant's own reporting currency, and wrong for a merchant who
#: keeps books elsewhere.
#:
#: **Deliberately not derived from the data.** "Whatever most payments use" would be a majority
#: vote that agrees with the file by construction -- and on a file where every payment is
#: foreign it would report no foreign payments at all, which is the failure mode that makes a
#: derived constant worse than a wrong declared one: it cannot be contradicted.
#:
#: Read only where the subset search has already found nothing, and only to *name* a cause --
#: it can never turn an abstention into a match. What the column supplies is that a rate
#: applies, never its magnitude, which is why ``FX_RATE_GAP`` stays an honest refusal rather
#: than a gap this matcher could close if it tried harder.
HOME_CURRENCY = "INR"

#: How far above a credit this matcher will look for a settlement when the exact join found
#: **nothing**, expressed in basis points of the credit. A *declared assumption* about the
#: counterparty's rolling-reserve policy, in exactly the sense the fee rates in ``fees.py`` and
#: ``SETTLEMENT_CYCLE_DAYS`` above are: read off a contract, wrong if the counterparty holds a
#: different share, and never fitted to a score.
#:
#: **Derived, not chosen.** A reserve of ``r`` leaves ``credit = net(1 - r)``, so the shortfall
#: is ``credit x r/(1 - r)``. Against a published reserve of up to 20% that is 2,500 bps of the
#: credit; 2,600 adds headroom for the paise-level nudge the generator applies. Proportional
#: rather than absolute so it is size-independent -- an absolute paise band would be wrong at
#: both ends of an amount distribution spanning four orders of magnitude.
#:
#: **This band is used only where the exact join already returned nothing, and it can only ever
#: produce an abstention.** That is the whole reason it is admissible, and it is a sharper
#: position than ``.plan/phase6.md`` correction (c) reached. The plan concluded that
#: ``--max-adjustment`` must widen globally for a reserved row to be reachable at all, and
#: measured the cost: at n=1000 even a 100p band loses 6 rows and 1,000p loses 41, so a band
#: wide enough for a reserve would gut the file. That cost is real but it is a cost of widening
#: the **resolution** path. Widening only the *diagnostic* path costs nothing measurable:
#: ``candidates_for`` is called a second time, on rows that were already abstaining, and its
#: result is never allowed to resolve. So the resolution path stays byte-identical -- same
#: coverage, same ambiguity rate, same correctness argument -- and ``--max-adjustment`` keeps
#: its default of 0. The free parameter sits where it provably cannot buy a match.
RESERVE_PROBE_BPS = 2_600

#: The shortfall fractions this matcher is willing to call a reserve, in basis points of the
#: settlement's declared net. Below the floor a shortfall is more likely a rate or rounding
#: disagreement than money deliberately held, and saying "reserve" there would be a confident
#: wrong diagnosis; above the ceiling it is not a partial payout in any published sense.
#:
#: The floor is the load-bearing end and it is what keeps ``PARTIAL_SETTLEMENT_PENDING``
#: distinct from ``UNEXPLAINED_RESIDUAL``: the largest rate-model error this data can produce
#: is the fee's own rounding divergence, a couple of paise on a batch, which is orders of
#: magnitude below 100 bps of a net. A diagnostic that fired on a few paise would relabel every
#: rounding bug as a business explanation, which is worse than declining to explain it.
#:
#: **Phase 7 step 6 re-measured this against a population it was not set against.** The band was
#: chosen when every bank row was a gateway credit; ``--noise-rows`` puts rows on the file that
#: are not gateway money at all, and a threshold left unexamined after its population changed is
#: the stale-population defect this phase found twice elsewhere. Measured both directions at
#: n=1000 seed 42 (``.plan/probe_phase7_band.py``, gitignored -- the numbers are here because
#: the probe is not): **the band has no setting that trades favourably, so it does not move.**
#:
#: The ceiling is inert, and provably so rather than incidentally. The probe admits a settlement
#: only within ``RESERVE_PROBE_BPS`` of the *credit*, so ``net <= 1.26 x credit`` and the
#: shortfall share ``(net - credit) / net`` cannot exceed ``0.26 / 1.26`` = 2,063 bps. Every
#: ceiling from 2,063 to 5,000 measured identically (80/80 reserves, 17 noise). The ceiling
#: therefore filters nothing today; it states the intent, and the self-check below asserts the
#: relationship so that raising the probe cannot silently make it binding.
#:
#: The floor is where a trade exists and it is a losing one. At 1,000 bps the band gives up
#: **10 of 80** credits truth records as reserved to remove **5 of 17** misdiagnosed noise rows,
#: and the lost ten fall to ``NO_CANDIDATE`` -- trading gate 13's characterised abstention back
#: into the undifferentiated pile Phase 7 exists to break up. 500 bps is the only free move
#: measured and it is worth one row of 17, which is not a reason to replace a principled
#: boundary with a seed-fitted one: the lowest genuine reserve sits between 500 and 1,000 bps,
#: so a floor tuned to today's draw silently costs true positives the moment the reserve range
#: or the seed changes. That is the same fitting-to-the-gap the reserve design refuses outright.
#:
#: What remains is a real limit, not a tuning failure: **8/18 ``gateway_plausible`` and 9/18
#: ``look_alike`` rows are diagnosed as pending reserves.** Both strata carry a gateway
#: counterparty by construction, so step 5's gate cannot set them aside, and a noise row drawn
#: in the gateway amount band genuinely does fall a plausible-reserve share short of some
#: settlement's net. The geometry is identical to a real reserve and no width of this band
#: separates them; they score ``NOISE_MISHANDLED`` and are reported as such. All 24
#: ``plainly_foreign`` rows are ignored before reaching here.
RESERVE_PLAUSIBLE_BPS: tuple[int, int] = (100, 3_000)

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
    Measured at n=1000: pool max 57-62 at a 30% share, rising to 157-168 at 100%, where the
    cap refuses ~95% of rows (that share was measured against the cap of 64 that stood before
    Phase 7 raised it to 80; the pool maxima are properties of the data and did not move). The
    Phase 5 default share is 30%.

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


def _count_distinct_sets(
    hypotheses: list[list[Member]], target: int
) -> tuple[list[frozenset[str]], bool, PoolTooLarge | None]:
    """Search every reading at one target, counting the *distinct* sets across all of them.

    **Extracted in Phase 8 with no behaviour change.** The body is the loop that sat inline in
    ``_search_membership``; it is lifted out because the refunds-first ordering below has to run
    the same readings against a **second** target -- the credit plus a declared orphan refund.
    Re-searching by copying the loop would make two copies of the precedence rule that
    ``_search_membership`` documents as a measured wrong-match bug (one wrong match in 627 on
    seed 1 at n=1000, from letting a unique-but-wrong answer override a known ambiguity), and a
    duplicated rule is the copy that drifts.

    Returns ``(found, split, over_cap)``:

      * ``found`` -- the distinct payment-id sets, in discovery order. A reading reporting two
        or more contributes **both**, because the question is not "does this reading resolve?"
        but "can anything I know explain this credit in more than one way?".
      * ``split`` -- whether any single reading itself reported two or more. Carried separately
        from ``len(found)``: two readings each finding one *different* set is also an ambiguity,
        but it is a different sentence in the note, and the caller says which.
      * ``over_cap`` -- a pool refusal, if any reading was refused before searching. The last
        one wins, which is deliberate rather than careless: the caller treats *any* refusal as
        fatal ("half a search is not a search"), so which reading reported it cannot change the
        verdict, and overwriting is what keeps this extraction behaviour-identical.
    """
    found: list[frozenset[str]] = []
    split = False
    over_cap: PoolTooLarge | None = None
    for members in hypotheses:
        result = tier2_resolve(members, target)
        if isinstance(result, PoolTooLarge):
            over_cap = result
        elif isinstance(result, TwoOrMore):
            split = True
            for candidate_set in (result.first, result.second):
                if candidate_set not in found:
                    found.append(candidate_set)
        elif isinstance(result, ExactlyOne) and result.payment_ids not in found:
            found.append(result.payment_ids)
    return found, split, over_cap


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
    # **A linked refund is subtracted in every reading, and it is deliberately *not* a fifth
    # axis.** ``refunds.csv`` names the ``payment_id`` each refund cites, so whether a refund
    # was taken is *declared* rather than hypothesised -- ``refunds_by_payment`` is a lookup,
    # never a search (``load.py`` decision 9). Making it an axis would double the readings to
    # eight to recover information the file already states, and every extra reading is another
    # chance for a coincidental subset to hit the target.
    #
    # **This was a wrong-match bug, not a tidy-up.** Withheld membership plus netted refunds
    # was first run together by gate 13, and the true subset was *not in the search space at
    # all*: the settlement's net is ``gross - fee - gst - tds - refunds``, and a reading that
    # stopped at TDS priced every refunded member too high. So the search saw only
    # coincidences. Usually none, and the row abstained as ``NO_CANDIDATE`` -- 22 of them on
    # seed 1 at n=1000. Occasionally one unrelated subset hit the shrunken target exactly, and
    # the row **resolved wrongly**: 2 per run on seeds 1 and 2 at n=1000, correctness 0.9962.
    # Gate 12 could not see it because it withholds membership without netting refunds, so the
    # true set was always reachable there.
    #
    # Zero-cost on a run without ``--netted-refunds``: the map is empty, every lookup returns
    # 0, and the amounts are the ones Phase 5 measured.
    linked_refunds = dataset.refunds_by_payment()

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
            amount -= linked_refunds.get(payment.payment_id, 0)
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
    found, split, over_cap = _count_distinct_sets(hypotheses, credit.amount_paise)

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

    # **Read off the POOL, not off any found subset, and that is the whole subtlety.** A stale
    # gross belongs to a payment in the *true* membership -- which is precisely what this search
    # does not know. Measured at seed 1, n=1000: all five wrong matches chose an entirely
    # domestic subset (``got_fgn=0``) while the true set held one foreign member, so a test on
    # the found rows would have caught none of them. The pool is the only set the matcher can
    # ask about before it has committed.
    #
    # Hoisted out of the ``not found`` branch below because Phase 8 step 5 gave it a second
    # reader: the same fact now also voids a *successful* search. Zero cost when ``--fx`` is
    # off -- no foreign payment, empty list, both readers fall through.
    foreign = sorted({p.currency for p in pool if p.currency != HOME_CURRENCY})
    n_foreign = sum(1 for p in pool if p.currency != HOME_CURRENCY)

    if not found:
        # Nothing sums to the credit under any reading. **Three different facts produce that,
        # and Phase 8 separates them here in a fixed order -- declared causes before
        # undeclared ones.** Before this branch existed the row abstained as ``NO_CANDIDATE``
        # for all three, which is honest about the search and silent about the cause.
        #
        # Still not a cue to widen the pool, in any of the three: widening the window until
        # something adds up converts a missing payment into a match, which is the failure mode
        # Phase 4 measured for the nearest-date tie-break.

        # --- 1. an orphan refund, which the inputs DECLARE -------------------------------
        # ``refunds_by_payment`` keys only payments present in ``payments.csv`` (``load.py``
        # decision 9, so unattributable money cannot close a gap), so an orphan refund is
        # subtracted from **no** member and every reading above sums to a target the credit
        # sits below by exactly that refund. Offering the shortfall back is therefore a
        # *lookup* on a declared amount, not a fitted magnitude -- the same evidence the
        # declared-membership path uses one level up, which until Phase 8 the withheld path
        # ignored. That asymmetry was the bug: the same orphan refund named on a declared
        # settlement was invisible on a withheld one.
        #
        # **Measured before it was written** (`.plan/probe_phase8_refunds_first.py`, seeds
        # 1/2/3/42 at n=200 and n=1000, both the Phase 6 and Phase 7 flag sets): of the 4 rows
        # that reach here on a non-FX run, the bump reveals **truth's own membership on 3 and
        # an ambiguity on the 4th**, with **zero coincidences** -- and on every row whose truth
        # nets no orphan refund, the bump reveals nothing at all. So this test does not fire
        # where it must not.
        #
        # **It names the cause and still abstains.** The set the bump reveals is deliberately
        # discarded rather than returned: resolving on it would mean subtracting money that
        # the inputs say left *some* settlement without saying it left *this* one -- exactly
        # what the declared path refuses two branches down, and the withheld path must not be
        # more confident than the path that can see its membership.
        for orphan in dataset.orphan_refunds():
            # ``over_cap`` cannot newly fire here: the hypotheses are the same lists, so the
            # pool size is unchanged and a refusal would already have returned above.
            revealed, _split, _cap = _count_distinct_sets(
                hypotheses, credit.amount_paise + orphan.amount_paise
            )
            if revealed:
                how = (
                    f"{len(revealed)} different subsets do"
                    if len(revealed) > 1
                    else f"a subset of {len(sorted(revealed[0]))} payment(s) does"
                )
                return Verdict(
                    credit.credit_id,
                    Outcome.EXCEPTION,
                    reason=Reason.REFUND_UNLINKED,
                    note=(
                        f"matched {settlement_id} on date and amount, its membership is "
                        f"undeclared, and no subset sums to {credit.amount_paise}p -- but "
                        f"{how} sum to {credit.amount_paise + orphan.amount_paise}p, which is "
                        f"this credit plus {orphan.refund_id} ({orphan.amount_paise}p). That "
                        f"refund cites {orphan.payment_id}, which is not in payments.csv, so "
                        f"the money left a settlement and nothing in the input files says it "
                        f"left this one. The amount is declared, so the shortfall is named "
                        f"rather than fitted -- and the membership is still not proved, so a "
                        f"human decides"
                    ),
                )

        # --- 2. a foreign-currency payment, whose rate the inputs DO NOT declare ----------
        # Phase 8's ``--fx``: a payment captured in a foreign currency carries a
        # ``gross_paise`` fixed at the **capture-day** rate, while the settlement's net and the
        # bank credit both reflect the **settlement-day** rate. Every reading above is built
        # from the recorded gross, so a stale figure puts the true membership outside the
        # search space -- the same shape as the refund bug, minus the fix, because the
        # settlement-day rate is declared in **no input file**. There is nothing to look up and
        # no fifth hypothesis that is not a free parameter.
        #
        # **The witness is the currency column, and it says only *that* a rate applies.** It is
        # read rather than the *magnitude* inferred, which is the whole distinction from
        # ``UNEXPLAINED_RESIDUAL``: that code means "the gap is measurable and no rule I have
        # accounts for it" -- reachable, and the honest answer where membership *is* declared,
        # because there the recorded grosses can be summed and the gap computed. Here there is
        # no declared set to sum, so the gap has no measurable value at all. Different facts,
        # different codes; sharing one would make a measurable gap and an unmeasurable one
        # indistinguishable in every report.
        #
        # **Keyed on ``currency``, never on ``method``** -- decision 7. ``config.py``'s
        # ``international_card`` is about where a card was issued, not what currency was
        # charged, so keying on the method would make FX-ness readable from a column the
        # generator sets for unrelated reasons.
        #
        # **And it is deliberately not a catch-all.** With no foreign-currency payment in the
        # pool the row falls through to ``NO_CANDIDATE`` below, which keeps its own meaning: a
        # payment is genuinely missing from the file, or a deduction is unmodelled. Admitting
        # every failed search here would let a missing capability score as an honest refusal,
        # and separating those two is what Phase 7 was for.
        if foreign:
            return Verdict(
                credit.credit_id,
                Outcome.EXCEPTION,
                reason=Reason.FX_RATE_GAP,
                note=(
                    f"matched {settlement_id} on date and amount, but no subset of the "
                    f"{len(pool)} payment(s) captured within {SETTLEMENT_CYCLE_DAYS}bd sums to "
                    f"{credit.amount_paise}p at gross or net of the declared rates -- and "
                    f"{n_foreign} of them were captured in {', '.join(foreign)}, whose "
                    f"gross_paise is fixed at the capture-day rate while this settlement's net "
                    f"and its credit are not. No input file carries the settlement-day rate, "
                    f"so the gap cannot be closed from these inputs -- only recognised. "
                    f"Fitting a rate that makes the arithmetic work would be choosing a free "
                    f"parameter, so a human supplies the rate"
                ),
            )

        # --- 3. neither: the search looked and found nothing ------------------------------
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

    # --- exactly one subset hit, and a foreign payment makes that stop being proof ---------
    # **Phase 8 step 5, and it is a correctness fix rather than a coverage choice.** Everything
    # above rests on one inference: exactly one subset of the pool sums to the credit, therefore
    # it is the membership. That holds only while the true subset is *in* the search space, and
    # a foreign payment is exactly the condition under which it is not -- design (b) leaves
    # ``payments.csv``'s gross at the capture-day rate while the settlement's net and the credit
    # carry the settlement-day one, so a true member is priced wrong in every reading here.
    #
    # Measured rather than argued (`.plan/probe_phase8_fx_withheld_real.py`, seeds 1/2/3/42 at
    # n=1000, six flags): **7 wrong matches** -- 5 at seed 1, 2 at seed 3, none on the same
    # seeds with ``--fx`` off. At seed 1 the arithmetic is exact on all five, and it names the
    # mechanism with no room for interpretation::
    #
    #     sum(net over the TRUE members) + fx_paise == credit_amount
    #
    # So the true subset sums to ``credit - fx``, the search targets ``credit``, the true set is
    # provably absent, and an unrelated all-domestic subset hit the target by coincidence. Seed
    # 1's 21 FX-bearing withheld credits went 16 ``FX_RATE_GAP`` + 5 wrong + **0 correct**.
    # ``AMBIGUOUS_MULTI_SUBSET`` also *fell* (-1/-2/-4 across seeds), which is the same fact
    # from the other side: removing the true subset turns honest ambiguity into false certainty.
    #
    # **Why the check cannot be narrower.** Three sharper tests were measured and refused:
    #
    #   * *Test the found subset for foreign members* -- catches none of the seven. Every wrong
    #     subset was entirely domestic; the foreign payment is in the true set, which is the set
    #     the search failed to find.
    #   * *Compare the re-derived fee/GST/TDS against ``settlements.csv``'s declared columns* --
    #     a perfect discriminator on the runs that have ``--tds`` (1,220 correct subsets kept,
    #     all 7 wrong ones caught) and **catastrophic without it**: on ``--fees`` alone the
    #     generator withholds no tax while this schedule derives 10 bps, so the tuple never
    #     agrees and it rejects 163 of 163 *correct* subsets. ``adjustments.py`` documents that
    #     divergence and I measured around it. It is also refused structurally --
    #     ``check_isolation.py`` check 7 keeps those three columns out of the resolution path,
    #     for the reason ``fees.py`` gives: consuming a declared deduction closes the residual
    #     the instant ``--fees`` populates it, and coverage becomes a tautology.
    #   * *Bound the FX magnitude and re-search* -- fitting a free parameter, which is the one
    #     thing this row may not do. No input file carries the settlement-day rate.
    #
    # **The cost is real, stated, and paid deliberately.** At seed 1, 122 of the 141 withheld
    # credits that resolve correctly today have a foreign payment somewhere in their pool (~19%
    # of all credits), and they become abstentions. The pool is a date window, not a settlement,
    # so 80 foreign payments in a 1,000-payment month contaminate nearly every window -- the
    # coarseness is inherent to what the matcher is allowed to know, not to this check. What is
    # bought is that correctness stops depending on whether a coincidence happened to be unique,
    # and every lost row carries an ``ABSTENTION_REASONS`` code instead of a wrong answer.
    if foreign:
        return Verdict(
            credit.credit_id,
            Outcome.EXCEPTION,
            reason=Reason.FX_RATE_GAP,
            note=(
                f"matched {settlement_id} on date and amount, and exactly one subset of the "
                f"{len(pool)} payment(s) captured within {SETTLEMENT_CYCLE_DAYS}bd sums to "
                f"{credit.amount_paise}p -- but {n_foreign} payment(s) in that pool were "
                f"captured in {', '.join(foreign)}, whose gross_paise is fixed at the "
                f"capture-day rate while this settlement's net and its credit are not. A "
                f"member priced at a stale rate cannot be summed to the target, so the true "
                f"membership need not be in the search space at all and this single hit is not "
                f"evidence that it was found: a subset drawn entirely from the other payments "
                f"can sum to the same number by coincidence, which is measured at 7 wrong "
                f"matches across four seeds when this row resolves. No input file carries the "
                f"settlement-day rate, so the uniqueness cannot be re-established -- only "
                f"reported. A human supplies the rate"
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
        # **Phase 7 step 5, and it runs before the reserve probe on purpose.** The exact join
        # found nothing, and there are two very different reasons that happens: the credit is
        # gateway money whose settlement this matcher cannot pin down, or it is **not gateway
        # money at all** -- a vendor payment, a salary run, an interest credit, sitting in the
        # same bank statement because a real statement is not filtered for us. Deciding which
        # comes first, because every diagnosis below assumes the row is ours.
        #
        # Ordering is the entire fix (`.plan/phase7.md` decision 4). Left after the reserve
        # probe, a non-gateway row that happens to sit a plausible percentage below some
        # settlement's net gets diagnosed as a partially-paid gateway settlement -- a confident
        # wrong answer about a row that was never ours. Measured on real generated noise before
        # this gate existed (`.plan/probe_phase7_gate_baseline.py`, seed 42, n=1000): **16 of 24
        # plainly-foreign rows collected ``PARTIAL_SETTLEMENT_PENDING``**, and 17 of 24 under
        # ``--fees --settlement-delay``. The mechanism, not the exact count, is the finding --
        # the plan's own 72.5% came from synthetic rows drawn by a probe's RNG, and a number that
        # moves with the probe's seed is a property of the probe.
        #
        # **Positive evidence, and both tests must fail before a row is ignored:**
        #
        #   * the narration's 4-digit tail hits some settlement's UTR, **or**
        #   * the narration carries a gateway counterparty spelling.
        #
        # Requiring *both* to fail is what keeps ``WRONG_IGNORE`` at zero, and it is written
        # this way now for Phase 8's benefit rather than this phase's. ``--utr-patchy`` strips
        # UTRs from **gateway** credits, so a rule where "no resolvable tail" alone sufficed
        # would convert this phase's ``noise_recall`` into next phase's ``WRONG_IGNORE`` --
        # real money dropped out of the books instead of merely left unexplained. The two are
        # not comparable failures: an abstention costs a human a look, and this is the one
        # verdict in the file that can lose a credit silently.
        #
        # The honest consequence, stated because it looks like a shortfall from outside: the
        # generator's ``gateway_plausible`` and ``look_alike`` strata both carry a gateway
        # counterparty *by construction*, so **neither can ever be ignored here** and both fall
        # through to the diagnosis below as ``NOISE_MISHANDLED``. That is the rule working, not
        # failing. ``noise_recall`` is therefore expected near the plainly-foreign share, and a
        # recall of 100% would be evidence the strata are too easy rather than a win.
        #
        # Nothing here reads truth, and nothing here reads an amount. The gate asks only what
        # the narration says and whether any settlement claims that tail -- which is the same
        # information a human doing this by hand would use, and the reason a bank statement's
        # narration column is worth parsing at all.
        tail_hits_a_settlement = (
            narration.ref_tail is not None and narration.ref_tail in index.utr_tails
        )
        if not tail_hits_a_settlement and not narration.is_gateway_counterparty:
            return Verdict(
                credit.credit_id,
                Outcome.IGNORED,
                reason=Reason.NON_GATEWAY_CREDIT,
                credit_amount_paise=credit.amount_paise,
                note=(
                    f"no settlement pays {credit.amount_paise}p within {window_days}bd of "
                    f"{credit.value_date.isoformat()}, and this row offers no evidence of "
                    f"being gateway money: its narration names no gateway counterparty"
                    + (
                        f" and its reference {narration.ref_tail} matches no settlement's UTR"
                        if narration.ref_tail is not None
                        else " and carries no reference this matcher can read"
                    )
                    + f". Read as income from another source -- {narration.raw!r} -- so it is "
                    f"out of scope for this reconciliation rather than an unexplained "
                    f"gateway credit. Both evidence tests have to fail before a row is set "
                    f"aside: a gateway credit whose UTR is merely unreadable still names the "
                    f"counterparty, and dropping one would lose real money from the books"
                ),
            )

        # Phase 6 step 7: **the exact join found nothing, so before reporting "nothing
        # matches" this asks the one further question the data can answer** -- is there a
        # settlement this credit is *short of* by a plausible reserve?
        #
        # Why this is a second, separate lookup rather than a wider first one. A reserved
        # credit is short of its settlement's net, so at ``max_adjustment_paise=0`` its true
        # settlement is invisible and the row lands here. ``.plan/phase6.md`` correction (c)
        # concluded from that ``--max-adjustment`` must widen globally, and measured the price
        # at n=1000: a 100p band costs 6 rows and 1,000p costs 41, so a band wide enough for a
        # 5-20% reserve would gut the file. That price is real -- but it is the price of
        # widening the **resolution** path. Widening only the *diagnostic* path costs nothing:
        # this call happens only on rows that were already abstaining, and its result can only
        # ever produce another abstention. The resolution path above is untouched, so coverage,
        # the ambiguity rate and the correctness argument are all exactly what they were.
        #
        # **It diagnoses and never resolves, and that is not a stylistic choice.** Decision 4
        # forbids modelling the reserve, because a deduction whose magnitude is free closes
        # every gap by construction -- it would convert ``UNEXPLAINED_RESIDUAL`` rows into
        # resolved ones while every arithmetic gate stayed green, which is the single most
        # dangerous thing this phase could build. The held amount is declared in **no input
        # file**, so there is nothing to verify a fitted magnitude against. What the matcher
        # can honestly say is "this settlement is short by an amount consistent with a rolling
        # reserve, and a human should confirm it", which is what ``PARTIAL_SETTLEMENT_PENDING``
        # means. Gate 13 asserts no resolved row ever carries a non-zero ``reserve_paise``.
        #
        # The plausibility band is what keeps this distinct from ``UNEXPLAINED_RESIDUAL``: a
        # shortfall of a few paise is a rate or rounding disagreement, not a business decision,
        # and calling it a reserve would relabel a rounding bug as an explanation.
        lo_bps, hi_bps = RESERVE_PLAUSIBLE_BPS
        short_of: list[tuple[Candidate, int, int]] = []
        for cand in index.candidates_for(
            credit,
            window_days=window_days,
            max_adjustment_paise=mul_bps(credit.amount_paise, RESERVE_PROBE_BPS),
        ):
            net = cand.settlement.net_paise
            shortfall = net - credit.amount_paise
            # Strictly positive only. ``amount_band`` is symmetric, so this probe also returns
            # settlements the credit *exceeds* -- and a reserve can only ever make a credit
            # smaller. A credit above a settlement's net is a different finding entirely
            # (``UNEXPLAINED_RESIDUAL``'s negative-gap branch says "no deduction adds money"),
            # and letting it in here would report money appearing as money withheld.
            if shortfall <= 0:
                continue
            # As a share of the **declared net**, which is the base a reserve is actually a
            # percentage of. Integer division: this is a plausibility test, so a floored basis
            # point is exact enough and keeps the financial path free of floats.
            share_bps = shortfall * 10_000 // net
            if lo_bps <= share_bps <= hi_bps:
                short_of.append((cand, shortfall, share_bps))

        if short_of:
            cited = "; ".join(
                f"{c.settlement_id} (net {c.settlement.net_paise}p, short {short}p "
                f"= {share / 100:.2f}% at {c.date_distance_days:+d}bd)"
                for c, short, share in short_of[:3]
            )
            return Verdict(
                credit.credit_id,
                Outcome.EXCEPTION,
                reason=Reason.PARTIAL_SETTLEMENT_PENDING,
                note=(
                    f"no settlement pays {credit.amount_paise}p exactly, but "
                    f"{len(short_of)} within {window_days}bd declare a net this credit falls "
                    f"plausibly short of: {cited}. A rolling reserve withheld from the payout "
                    f"would look exactly like this, and the held amount is declared in no "
                    f"input file -- so the shortfall cannot be proved, only recognised. "
                    f"Committing to a settlement on a shortfall this matcher cannot verify "
                    f"would be fitting a free magnitude to a gap, so a human confirms the "
                    f"release schedule"
                    + (
                        f" -- {len(short_of)} settlements fit, which is the ambiguity itself"
                        if len(short_of) > 1
                        else ""
                    )
                ),
            )

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

    # Phase 6 step 6: the refunds netted off this settlement, **looked up and not searched**
    # (decision 9 -- ``refunds.csv`` states the payment each refund belongs to, so the link is
    # given). Subtracted from the gap *before* the rules are consulted, which is the design
    # decision worth stating: a refund is declared data, not a hypothesis, so it belongs on the
    # known side of the arithmetic rather than as a fifth rule.
    #
    # Two consequences. The rule count stays at **four**, because a constant subtracted from
    # the gap shifts every rule's target equally and cannot make two rules collide that did not
    # collide before -- so ``AMBIGUOUS_ADJUSTMENT``'s reachability argument is untouched by this
    # step. And the refund can never *close* a gap on its own: it moves the target, and one of
    # the four rules still has to account for what remains, exactly. A term that could close a
    # gap by itself with a magnitude read from a file is decision 4's trap, which is why the
    # reserve is not modelled at all and this one is subtracted rather than fitted.
    refunds_by_payment = dataset.refunds_by_payment()
    refunds_total = sum(refunds_by_payment.get(pid, 0) for pid in payment_ids)
    rate_gap = gap - refunds_total
    closing, derived = explain_gap(rate_gap, members, schedule)

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
        # Phase 6 step 6: **an orphan refund gets its own reason code before the generic
        # residual one.** A refund citing a payment outside this month's file took money off a
        # settlement that nothing in these three files attributes, so the remainder is not
        # "unexplained" in the sense ``UNEXPLAINED_RESIDUAL`` means -- it is explained in kind
        # and unattributable in fact, which is a different thing for whoever has to work the
        # exception. Distinguished by amount: the leftover must equal one orphan refund exactly.
        #
        # **This diagnoses the row and deliberately does not resolve it.** Matching the residual
        # against a declared refund amount would be enough to *attribute* the refund, and that
        # capability is real -- truth marks these rows ``resolvable: true`` precisely because an
        # unbounded matcher could do it (Phase 4b's standard, see ``story.build``). Doing it here
        # would mean resolving on an amount coincidence with no independent confirmation, which
        # is how a one-in-many collision becomes a wrong match. So the row abstains, coverage
        # falls, correctness holds, and the capability is declared as available to a later phase
        # rather than quietly taken.
        # **Each candidate orphan is subtracted and the gap re-offered to the same four rules**,
        # rather than compared against a hand-computed leftover. The first version of this
        # branch did the latter -- ``leftover = gap - derived.total_paise`` -- and it silently
        # failed to fire on the very run that introduced the flag. Worth recording because the
        # mistake is a familiar one wearing new clothes: ``derived`` is what this schedule
        # *predicts*, not what was withheld, and a ``--netted-refunds`` run without ``--fees``
        # withholds no fee at all. So the leftover was measured against a deduction that never
        # happened (searching for 14,982p while the orphan refund was 16,241p), and the row fell
        # through to ``UNEXPLAINED_RESIDUAL``.
        #
        # Re-offering the remainder to ``explain_gap`` is the fix *and* the cheaper design: the
        # four rules already enumerate which deductions were actually applied, which is exactly
        # the unknown that broke the subtraction, and the unpriced-method discipline comes along
        # for free instead of being re-implemented here.
        hits: list[tuple[object, str]] = []
        if gap > 0:
            for orphan in dataset.orphan_refunds():
                if orphan.amount_paise > gap:
                    continue
                also, _ = explain_gap(gap - orphan.amount_paise, members, schedule)
                # Any rule closing the remainder is enough to *name* this orphan as the likely
                # cause. Two rules closing it is not an ``AMBIGUOUS_ADJUSTMENT``: that code is
                # about a resolved row's components being undetermined, and nothing is being
                # resolved here.
                if also:
                    hits.append((orphan, also[0].rule))
        if hits:
            cited = ", ".join(
                f"{o.refund_id} ({o.amount_paise}p, cites {o.payment_id})"  # type: ignore[attr-defined]
                for o, _rule in hits[:3]
            )
            return Verdict(
                credit.credit_id,
                Outcome.EXCEPTION,
                reason=Reason.REFUND_UNLINKED,
                note=(
                    f"{settlement_id} agrees on date and amount, and its {gap}p gap closes "
                    f"only if an out-of-scope refund is assumed: {cited}. Each cites a payment "
                    f"that is not in payments.csv, so the money left a settlement and nothing "
                    f"in the input files says it left *this* one. Attributing it on an amount "
                    f"coincidence would be a guess rather than a proof, so a human decides"
                    + (
                        f" -- {len(hits)} orphan refunds fit, which is the ambiguity itself"
                        if len(hits) > 1
                        else ""
                    )
                ),
            )
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
        # Phase 6 step 6, and the **one term here that is read rather than derived** -- which
        # is legitimate for a reason the fee and TDS columns do not share. A refund is not
        # priced by any rate: ``refunds.csv`` is the only statement of it that exists, so
        # looking it up is modelling it, not copying an answer. The distinction that keeps
        # decision 2 intact: reading ``settlements.csv``'s ``fee_paise`` would substitute a
        # declared result for an arithmetic the matcher is supposed to reproduce, whereas
        # reading a refund substitutes nothing -- there is no independent derivation of it to
        # skip. What stops it closing gaps by construction is that it cannot close one at all:
        # it shifts the target and a declared rule still has to account for the remainder
        # exactly (see the ``rate_gap`` block above).
        refunds_paise=refunds_total,
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
    from .load import Refund as R
    from .load import Settlement as S

    mon, tue = date(2026, 8, 10), date(2026, 8, 11)
    when = datetime(2026, 8, 10, 5, 34, 22, tzinfo=timezone.utc)

    def payment(
        pid: str, gross: int, method: str = "card", currency: str = HOME_CURRENCY
    ) -> P:
        # The method is now load-bearing, not decoration: the fee rate is looked up by it,
        # so a residual cannot be explained from the amount alone.
        #
        # ``currency`` is settable for Phase 8's ``FX_RATE_GAP`` fixtures, and it defaults to
        # ``HOME_CURRENCY`` rather than to the literal ``"INR"`` so that re-pointing the
        # constant cannot leave the fixtures asserting against a currency the matcher no longer
        # treats as domestic -- which would turn every fixture below into an FX row at once.
        return P(pid, f"ord_{pid[4:]}", when, gross, method, currency, "captured")

    def settlement(
        sid: str, on: date, net: int, tail: str = "0000", fee_paise: int = 0
    ) -> S:
        # fee_paise is settable so a test can prove the gate does NOT trust the declared
        # column. Phase 4 populates it for real; nothing on the match path reads it.
        return S(sid, on, net, fee_paise, 0, 0, f"XXXX{tail}")

    def credit(cid: str, on: date, amount: int, narration: str = "NEFT-RAZORPAYSOFT-XXXX8104") -> C:
        return C(cid, on, amount, narration)

    def dataset(payments, settlements, credits, items, refunds=()) -> D:
        # ``refunds`` defaults to empty, which is what every pre-Phase-8 fixture passed
        # positionally. Phase 8's refunds-first ordering needs an *orphan* refund on the file
        # (one citing a payment absent from ``payments``), so the parameter is exposed rather
        # than the tuple being hard-coded.
        return D(tuple(payments), tuple(settlements), tuple(credits), tuple(refunds), items)

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

    # --- Phase 6 step 7: the reserve probe, and what it must refuse to say ------
    # A credit short of its settlement's net by 10% -- the shape ``--reserve`` produces. The
    # exact join finds nothing (the band is [credit, credit] and no settlement pays that), so
    # the probe runs and recognises the shortfall.
    reserved_ds = dataset(
        [payment("pay_0001", 100_000)],
        [settlement("setl_0005", mon, 100_000, "8104")],
        [credit("C0001", mon, 90_000)],
        {"setl_0005": ("pay_0001",)},
    )
    v = verdict_of(reserved_ds, "C0001")
    assert v.outcome is Outcome.EXCEPTION, (
        "a reserved credit must abstain -- the held amount is in no input file, so resolving "
        "it means fitting a free magnitude to a gap (decision 4)"
    )
    assert v.reason is Reason.PARTIAL_SETTLEMENT_PENDING, v.reason
    # **The assertion the whole step exists for.** Correction (c)'s regression test: this row
    # must not arrive as ``NO_CANDIDATE``, because Phase 7's entire job is telling a
    # non-gateway credit from a gateway credit it cannot explain -- and a reserved row hiding
    # inside ``NO_CANDIDATE`` would poison that distinction before Phase 7 begins.
    assert v.reason is not Reason.NO_CANDIDATE
    # And it names the settlement it suspects *in the note only*. Nothing is claimed: no tier,
    # no residual, no payment set, no decomposition. This is what "diagnose, never resolve"
    # means mechanically, and it is what gate 13 re-asserts over a whole run.
    assert v.payment_ids == () and v.settlement_ids == ()
    assert v.tier is None and v.residual_paise is None
    assert v.decomposition is None, (
        "a diagnosed reserve must publish no decomposition -- a reserve_paise term on a "
        "verdict would be a magnitude this matcher cannot verify against any input"
    )
    assert "setl_0005" in (v.note or "") and "10.00%" in (v.note or ""), v.note

    # A shortfall of 50p on a 100,000p net -- 5 bps, below ``RESERVE_PLAUSIBLE_BPS``' floor.
    # The probe *finds* this settlement (it is well inside the band) and must still decline to
    # call it a reserve: a few paise is a rate or rounding disagreement, and answering
    # "rolling reserve" there would relabel a rounding bug as a business explanation. This is
    # the case that keeps ``PARTIAL_SETTLEMENT_PENDING`` a meaningful code rather than a
    # catch-all for every unmatched short credit.
    tiny_ds = dataset(
        [payment("pay_0001", 100_000)],
        [settlement("setl_0005", mon, 100_000, "8104")],
        [credit("C0001", mon, 99_950)],
        {"setl_0005": ("pay_0001",)},
    )
    v = verdict_of(tiny_ds, "C0001")
    assert v.reason is Reason.NO_CANDIDATE, (
        f"a 5bps shortfall must not be diagnosed as a reserve, got {v.reason}"
    )

    # A credit that *exceeds* a settlement's net. ``amount_band`` is symmetric, so the probe
    # sees this settlement -- and a reserve can only ever make a credit smaller. Reporting
    # money appearing as money withheld would be a confident wrong diagnosis in the opposite
    # direction, so the shortfall test is strictly positive.
    over_ds = dataset(
        [payment("pay_0001", 100_000)],
        [settlement("setl_0005", mon, 100_000, "8104")],
        [credit("C0001", mon, 110_000)],
        {"setl_0005": ("pay_0001",)},
    )
    v = verdict_of(over_ds, "C0001")
    assert v.reason is Reason.NO_CANDIDATE, (
        f"a credit above a settlement's net is not a reserve, got {v.reason}"
    )

    # The probe respects the date window like every other lookup: the "right amount, wrong
    # day" fixture above stays ``NO_CANDIDATE``, and so does a *reserved* row on the wrong day.
    # Without this, the probe would silently widen the window as well as the band.
    off_day = dataset(
        [payment("pay_0001", 100_000)],
        [settlement("setl_0005", mon, 100_000, "8104")],
        [credit("C0001", tue, 90_000)],
        {"setl_0005": ("pay_0001",)},
    )
    assert verdict_of(off_day, "C0001").reason is Reason.NO_CANDIDATE, (
        "the reserve probe must not widen the date window, only the amount band"
    )
    # ...and the same row *inside* a 1-day window is diagnosed, or the assertion above is
    # passing for the wrong reason (a probe that never fires would also satisfy it).
    assert verdict_of(off_day, "C0001", window_days=1).reason is (
        Reason.PARTIAL_SETTLEMENT_PENDING
    ), "the probe does not fire even inside the window -- the case above proves nothing"

    # **The band is the binding bound, and the plausibility ceiling is deliberately looser.**
    # Recorded because the ceiling is therefore *unreachable* on this data and that is a fact
    # worth stating rather than discovering later: a probe band of ``b`` bps of the credit
    # admits nets up to ``credit x (1 + b/10000)``, so the largest share of the *net* it can
    # ever see is ``b / (10000 + b)`` -- 2,063 bps at b=2,600, below the 3,000 bps ceiling. So
    # the ceiling cannot fire while the band stands, and it is kept as a guard against a future
    # band widening rather than as a live filter. Asserted so the two constants cannot drift
    # into disagreeing about which one is in charge.
    _max_share = RESERVE_PROBE_BPS * 10_000 // (10_000 + RESERVE_PROBE_BPS)
    assert _max_share <= RESERVE_PLAUSIBLE_BPS[1], (
        f"the probe band admits shortfalls up to {_max_share}bps of net, above the "
        f"{RESERVE_PLAUSIBLE_BPS[1]}bps plausibility ceiling -- the ceiling would then be "
        f"doing real filtering and needs its own fixture"
    )
    # The band must still cover the reserve the generator actually draws: config's
    # RESERVE_BPS_BAND tops out at 2,000 bps of net, which is 2,500 bps of the short credit.
    assert RESERVE_PROBE_BPS >= 2_500, (
        "the probe band no longer covers a 20% reserve, so the reserved rows this matcher is "
        "built to recognise would come back as NO_CANDIDATE"
    )

    # --- Phase 7 step 5: the IGNORED gate, and the four ways it must not misfire ------
    # First use of ``Outcome.IGNORED`` in the project. Every fixture below pairs the case with
    # the control that could make it pass for the wrong reason -- a gate that never fires and a
    # gate that always fires both satisfy a one-sided test.
    foreign = "NEFT-ACME SUPPLIES LTD-XXXX9999"

    # (1) Both evidence tests fail, and no settlement is even close: IGNORED, and the verdict
    # claims nothing at all.
    noise_ds = dataset(
        [payment("pay_0001", 85358)],
        [settlement("setl_0005", mon, 85358, "8104")],
        [credit("C0001", mon, 12_345, foreign)],
        {"setl_0005": ("pay_0001",)},
    )
    v = verdict_of(noise_ds, "C0001")
    assert v.outcome is Outcome.IGNORED, v.outcome
    assert v.reason is Reason.NON_GATEWAY_CREDIT, v.reason
    assert not v.outcome.is_committal, "IGNORED must not be scored as an asserted answer"
    assert v.payment_ids == () and v.settlement_ids == (), (
        "a row set aside as out of scope names nothing -- it is not a match"
    )
    assert v.tier is None and v.residual_paise is None and v.decomposition is None
    assert v.credit_amount_paise == 12_345, (
        "the amount is a bank column, not a claim, and an out-of-scope row is more useful "
        "quoting the figure it declined to explain"
    )
    assert "ACME SUPPLIES LTD" in (v.note or ""), (
        f"the note must quote the narration it read, so the decision is reviewable: {v.note}"
    )

    # (2) **The assertion this step exists for.** The same foreign row, positioned so the
    # reserve probe *would* have diagnosed it: 10% below a real settlement's net, which is
    # squarely inside RESERVE_PLAUSIBLE_BPS. Before the gate this returned
    # PARTIAL_SETTLEMENT_PENDING -- a confident wrong answer about a row that was never ours,
    # measured at 16 of 24 plainly-foreign rows on real data.
    tempting = dataset(
        [payment("pay_0001", 100_000)],
        [settlement("setl_0005", mon, 100_000, "8104")],
        [credit("C0001", mon, 90_000, foreign)],
        {"setl_0005": ("pay_0001",)},
    )
    v = verdict_of(tempting, "C0001")
    assert v.outcome is Outcome.IGNORED and v.reason is Reason.NON_GATEWAY_CREDIT, (
        f"a non-gateway row sitting a plausible reserve below a settlement must be set aside, "
        f"not diagnosed as a partially-paid gateway settlement: {v.reason}"
    )
    # ...and the control, without which the fixture above proves nothing: the *identical*
    # geometry with a gateway narration must still reach the probe. If this were also IGNORED,
    # the gate would be swallowing reserved rows and gate 13 would fail at n=1000.
    assert verdict_of(
        dataset(
            [payment("pay_0001", 100_000)],
            [settlement("setl_0005", mon, 100_000, "8104")],
            [credit("C0001", mon, 90_000, "NEFT-RAZORPAYSOFT-XXXX9999")],
            {"setl_0005": ("pay_0001",)},
        ),
        "C0001",
    ).reason is Reason.PARTIAL_SETTLEMENT_PENDING, (
        "the gate is eating reserved gateway rows -- the probe never fires, so the fixture "
        "above passes for the wrong reason"
    )

    # (3) **Either test passing is enough to keep a row in scope.** Both halves are asserted,
    # because "both must fail" and "either must fail" differ on exactly these two rows -- and
    # the second is the one Phase 8 turns into a live case.
    #
    # (3a) No gateway counterparty, but the reference hits a real settlement's UTR.
    assert verdict_of(
        dataset(
            [payment("pay_0001", 100_000)],
            [settlement("setl_0005", mon, 100_000, "8104")],
            [credit("C0001", mon, 90_000, "NEFT-ACME SUPPLIES LTD-XXXX8104")],
            {"setl_0005": ("pay_0001",)},
        ),
        "C0001",
    ).reason is Reason.PARTIAL_SETTLEMENT_PENDING, (
        "a row whose reference matches a settlement's UTR is evidence of gateway money even "
        "with an unrecognised counterparty, and must not be set aside"
    )
    # (3b) A gateway counterparty with **no readable reference at all** -- the masked form.
    # This is ``--utr-patchy``'s shape (Phase 8) arriving early: a genuine gateway credit whose
    # UTR is gone still names the counterparty, and ignoring it would drop real money from the
    # books. A rule where "no resolvable tail" alone sufficed would fail right here.
    for spelling in ("RAZORPAYSOFT", "RAZORPAY SOFTWARE", "RZRPAY"):
        masked = dataset(
            [payment("pay_0001", 100_000)],
            [settlement("setl_0005", mon, 100_000, "8104")],
            [credit("C0001", mon, 90_000, f"NEFT-{spelling}-XXXX")],
            {"setl_0005": ("pay_0001",)},
        )
        mv = verdict_of(masked, "C0001")
        assert mv.outcome is not Outcome.IGNORED, (
            f"{spelling!r} names the gateway, so a missing UTR alone must never be sufficient "
            f"to ignore the row -- that is how Phase 7's noise_recall becomes Phase 8's "
            f"WRONG_IGNORE: {mv.reason}"
        )
        assert mv.reason is Reason.PARTIAL_SETTLEMENT_PENDING, mv.reason

    # (4) **The gate sits inside the no-candidate branch, so it never pre-empts a match.** A
    # row failing both evidence tests that nonetheless has an exact settlement still RESOLVES.
    #
    # That ordering is deliberate and it cuts both ways, which is worth stating rather than
    # discovering: a genuinely non-gateway row landing exactly on some settlement's net inside
    # the window would be wrong-matched here, and no narration test would save it. The
    # alternative -- gating before the join -- would ignore a *gateway* credit whose narration
    # is unreadable even though its settlement matches exactly, and that loses money instead of
    # merely misattributing a row the generator refuses to create. Measured at 0 occurrences
    # across seven flag sets at n=1000 (`.plan/probe_phase7_gate_baseline.py`), because noise
    # amounts are drawn clear of every credit key; it is a coincidence, not a mechanism.
    exact_ds = dataset(
        [payment("pay_0001", 85358)],
        [settlement("setl_0005", mon, 85358, "8104")],
        [credit("C0001", mon, 85358, foreign)],
        {"setl_0005": ("pay_0001",)},
    )
    ev = verdict_of(exact_ds, "C0001")
    assert ev.outcome is Outcome.RESOLVED, (
        f"the gate must not run ahead of the exact join -- a readable settlement match is "
        f"stronger evidence than any narration test: {ev.outcome}/{ev.reason}"
    )

    # (5) The gate reads the index's tail set, so that set has to be populated. A guard against
    # the failure mode where ``utr_tails`` is empty and *every* unmatched row looks foreign.
    assert SettlementIndex(exact_ds.settlements).utr_tails == {"8104"}, (
        "the settlement tail set is empty or mis-stripped, which would make the first "
        "evidence test unfalsifiable and ignore every unmatched gateway credit"
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

    # (3a-3d) Phase 8 splits that empty-search branch three ways, **declared causes before
    # undeclared ones**, and these four fixtures are deliberately a matched set: (3) above and
    # (3a) differ in *one field*, (3b) and (3c) differ in the same one field, and (3d) is the
    # control that stops the refund test from being a rubber stamp. A fixture suite that only
    # showed each new code firing somewhere would not distinguish "the branch works" from
    # "the branch fires on everything", which is the failure the ordering exists to prevent.

    # (3a) The same numbers as (3), with the member captured in a foreign currency ->
    # FX_RATE_GAP. The pairing is the argument: identical pool, identical target, identical
    # settlement, and the *only* difference is a column whose whole content is "a rate applies
    # here". So a pass cannot be attributed to the amounts.
    fx_row = dataset(
        [payment("pay_0001", 40_000, currency="USD")],
        [settlement("setl_0005", mon, 99_999, "8104")],
        [credit("C0001", mon, 99_999)],
        {},
    )
    v = verdict_of(fx_row, "C0001")
    assert v.outcome is Outcome.EXCEPTION
    assert v.reason is Reason.FX_RATE_GAP, v.reason
    assert v.tier is None, "an abstention claims no tier"
    assert v.payment_ids == () and v.settlement_ids == (), (
        "the verdict contract forbids an abstention from carrying ids -- and here it matters "
        "twice, because a row this matcher cannot price must not look like a match to the "
        "scorer's set-equality join"
    )
    assert "USD" in (v.note or ""), (
        f"the note must name the currency it read, so a human can check the claim against "
        f"payments.csv rather than taking the code's word for it: {v.note}"
    )
    assert "capture-day rate" in (v.note or ""), v.note
    # The distinction from UNEXPLAINED_RESIDUAL has to be *in the note*, or the two codes are
    # one code with two spellings: this one says the gap has no measurable value at all,
    # because no declared set exists to sum. See the branch comment.
    assert "No input file carries the settlement-day rate" in (v.note or ""), v.note
    # Wording unique to THIS branch, because step 5 gave ``FX_RATE_GAP`` a second call site
    # whose note also says "capture-day rate" and also says "No input file carries the
    # settlement-day rate". Without an assertion on the half that differs, this fixture would
    # pass against a mutant that reached the *other* site -- the identical failure Phase 8 step
    # 1 hit with ``REFUND_UNLINKED``, where one code from two sites made a wrong verdict look
    # right because the code and its cited evidence both matched.
    assert "no subset of the" in (v.note or ""), (
        f"this fixture pins the EMPTY-SEARCH site, whose note must say nothing summed: {v.note}"
    )

    # (3a2) The same code from the **other** site, and the pairing is again the argument:
    # ``pay_0001``'s gross is now exactly the credit, so exactly one subset hits and the search
    # *succeeds*. A foreign payment in the pool voids the uniqueness inference anyway -- the true
    # membership need not be in the search space at all, so one hit is not evidence it was found.
    # Phase 8 step 5, measured at 7 wrong matches across seeds 1/2/3/42 at n=1000 before this
    # branch existed.
    fx_unique = dataset(
        [payment("pay_0001", 99_999, currency="USD")],
        [settlement("setl_0005", mon, 99_999, "8104")],
        [credit("C0001", mon, 99_999)],
        {},
    )
    v = verdict_of(fx_unique, "C0001")
    assert v.outcome is Outcome.EXCEPTION, (
        f"a unique subset hit with a foreign payment in the pool must not resolve: {v}"
    )
    assert v.reason is Reason.FX_RATE_GAP, v.reason
    assert v.payment_ids == () and v.settlement_ids == (), (
        "the abstention must carry no ids, even though a subset WAS found -- the found set is "
        "deliberately discarded rather than reported, exactly as the orphan-refund branch "
        "discards the set its bump reveals"
    )
    assert "exactly one subset of the" in (v.note or ""), (
        f"the note must say the search succeeded and was overruled, or it is indistinguishable "
        f"from the empty-search site above: {v.note}"
    )
    assert "coincidence" in (v.note or ""), v.note

    # (3a3) **The control that could embarrass (3a2)**, and it is the one that matters: the same
    # numbers with the payment domestic must RESOLVE. Without it, (3a2) would pass just as well
    # against a matcher that had stopped resolving withheld settlements altogether -- an
    # abstention-on-everything is trivially free of wrong matches, and the coverage cost of the
    # veto is only defensible if it is confined to pools that actually hold a foreign payment.
    domestic_unique = dataset(
        [payment("pay_0001", 99_999)],
        [settlement("setl_0005", mon, 99_999, "8104")],
        [credit("C0001", mon, 99_999)],
        {},
    )
    v = verdict_of(domestic_unique, "C0001")
    assert v.outcome is Outcome.RESOLVED, (
        f"an all-domestic pool with one exact subset must still resolve -- the veto is keyed on "
        f"the currency column, not on the search succeeding: {v.outcome} / {v.reason}"
    )
    assert v.payment_ids == ("pay_0001",), v.payment_ids

    # (3b) A withheld settlement whose shortfall is a **declared** orphan refund ->
    # REFUND_UNLINKED, not FX_RATE_GAP and not NO_CANDIDATE. ``refunds_by_payment`` keys only
    # payments present in the file, so rfnd_0001 is subtracted from no member and every reading
    # sums to 50_000 while the credit is 30_000 -- short by exactly the refund.
    orphan_withheld = dataset(
        [payment("pay_0001", 50_000)],
        [settlement("setl_0005", mon, 30_000, "8104")],
        [credit("C0001", mon, 30_000)],
        {},
        refunds=[R("rfnd_0001", "pay_9999", when, 20_000)],
    )
    v = verdict_of(orphan_withheld, "C0001")
    assert v.outcome is Outcome.EXCEPTION
    assert v.reason is Reason.REFUND_UNLINKED, v.reason
    assert "rfnd_0001" in (v.note or "") and "pay_9999" in (v.note or ""), (
        f"the note must name the refund and the payment it cites, which is what makes this a "
        f"lookup a human can verify rather than a magnitude the matcher fitted: {v.note}"
    )
    # **Which branch produced this, asserted rather than assumed.** ``REFUND_UNLINKED`` is
    # raised from *two* sites -- here (membership withheld, the bump revealed a set) and the
    # declared-membership path further down (a gap that closes only if an orphan is assumed) --
    # and both notes name the refund and the payment it cites. So the assertions above cannot
    # tell them apart, and this fixture was **measured passing against a mutant that returned
    # the revealed set** (`.plan/probe_phase8_branch_mutants.py`): the row resolved, the money
    # proof failed to close the 20,000p gap, the declared path then named the same orphan, and
    # every assertion above still held. The verdict was right by accident.
    #
    # This is the ``check-must-fire-its-own-assertion`` failure exactly: one code from many
    # sites hides which one fired. The withheld-path note is the only thing that distinguishes
    # them, so it is what gets asserted.
    assert "its membership is undeclared" in (v.note or ""), (
        f"this must be the withheld-membership branch, not the declared-membership orphan "
        f"branch that emits the same code: {v.note}"
    )
    assert "no subset sums to" in (v.note or ""), v.note
    # **The bump revealed pay_0001, and the verdict must still carry nothing.** This is the
    # assertion that keeps the withheld path from being more confident than the declared path,
    # which names the same orphan and also refuses to resolve on it. Resolving here would mean
    # subtracting money the inputs say left *some* settlement without saying it left this one.
    assert v.payment_ids == () and v.settlement_ids == (), (
        f"the revealed set must be discarded, not returned: {v.payment_ids}"
    )
    assert v.tier is None, "an abstention claims no tier"

    # (3c) The ordering itself, and it is the fixture the whole decision rests on: the same row
    # as (3b) with the member *also* foreign, so **both** causes are present. Declared wins --
    # an amount ``refunds.csv`` states beats a rate no file carries. Reversed, every
    # orphan-refund row on an FX run would be relabelled as an unpriceable currency gap and the
    # named, checkable cause would be lost.
    both_causes = dataset(
        [payment("pay_0001", 50_000, currency="USD")],
        [settlement("setl_0005", mon, 30_000, "8104")],
        [credit("C0001", mon, 30_000)],
        {},
        refunds=[R("rfnd_0001", "pay_9999", when, 20_000)],
    )
    v = verdict_of(both_causes, "C0001")
    assert v.reason is Reason.REFUND_UNLINKED, (
        f"a declared orphan refund must outrank an undeclared FX rate, got {v.reason}"
    )

    # (3d) The control on the refund test: an orphan refund whose amount closes *nothing*.
    # Bumping by 777p makes no reading sum, so the branch must decline and fall through --
    # here to NO_CANDIDATE, because the member is domestic. Without this, "an orphan refund
    # exists on the file" would be indistinguishable from "an orphan refund explains this gap",
    # and the refunds branch would be a rubber stamp on every run that has one.
    orphan_irrelevant = dataset(
        [payment("pay_0001", 50_000)],
        [settlement("setl_0005", mon, 30_000, "8104")],
        [credit("C0001", mon, 30_000)],
        {},
        refunds=[R("rfnd_0001", "pay_9999", when, 777)],
    )
    v = verdict_of(orphan_irrelevant, "C0001")
    assert v.reason is Reason.NO_CANDIDATE, (
        f"an orphan refund that closes nothing must not be named as the cause, got {v.reason}"
    )
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
