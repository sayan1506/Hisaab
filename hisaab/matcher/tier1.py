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

from ..common.reasons import Reason
from ..common.verdict import Decomposition, Outcome, Verdict
from .blocking import Candidate, SettlementIndex
from .fees import Explanation, FeeSchedule, explain_gap
from .load import Credit, Dataset
from .normalize import Narration, parse

#: The tier this module speaks for. Carried onto every verdict it resolves so a report
#: can say *which* strategy earned a match, and so Phase 5's tier 2 is distinguishable
#: in the output rather than only in the code.
TIER = 1


def _note_for_match(
    candidate: Candidate,
    narration: Narration,
    payment_count: int,
    explanation: Explanation,
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
            f"(fee {explanation.fee_paise}p + GST {explanation.gst_paise}p)"
        )
    else:
        accounted = f"{explanation.rule}, credit equals gross"
    return (
        f"tier 1 exact: {settlement.settlement_id}, "
        f"{payment_count} payment(s), "
        f"date distance {candidate.date_distance_days}bd, "
        f"amount delta {candidate.amount_delta_paise}p; "
        f"{accounted}; "
        f"corroboration only: {utr}"
    )


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

    if not payment_ids:
        # The settlement matched, but nothing declares which payments compose it. Phase
        # 8's --settlement-report-late creates exactly this state by withholding
        # settlement_items.csv, and the honest Phase 3 answer is to abstain: the payment
        # set would have to be *searched*, which is Phase 5's subset-sum, not this tier.
        # Unreachable in clean mode -- load.py proves every settlement has members.
        return Verdict(
            credit.credit_id,
            Outcome.EXCEPTION,
            reason=Reason.AMBIGUOUS_MULTI_SUBSET,
            note=(
                f"matched {settlement_id} on date and amount, but no membership is "
                f"declared for it -- the payment set would have to be searched (tier 2)"
            ),
        )

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
    explanation, derived = explain_gap(gap, members, schedule)

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
        tier=TIER,
        residual_paise=residual,
        credit_amount_paise=credit.amount_paise,
        decomposition=decomposition,
        note=_note_for_match(winner, narration, len(payment_ids), explanation),
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
    # fee on 85,358p is 1,707p plus 307p GST = 2,014p, which is far more than the 500p
    # actually withheld -- so this row is the rate table being wrong for it, and the note
    # has to say that rather than report a negative remainder.
    assert "predict a larger deduction of 2014p" in (v.note or ""), v.note
    assert "-1514p" not in (v.note or ""), (
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
    assert "account for 2014p" in (v.note or ""), v.note
    assert "leaving 986p unexplained" in (v.note or ""), v.note

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
    # this row. The derived rate says 2,014p, so nothing closes and the row abstains.
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

    # --- a matched settlement with no declared membership (Phase 8 shape) --
    undeclared = dataset(
        [payment("pay_0001", 85358)],
        [settlement("setl_0005", mon, 85358, "8104")],
        [credit("C0001", mon, 85358)],
        {},   # settlement_items.csv withheld
    )
    v = verdict_of(undeclared, "C0001")
    assert v.outcome is Outcome.EXCEPTION
    assert v.reason is Reason.AMBIGUOUS_MULTI_SUBSET, v.reason
    assert "searched" in (v.note or "")

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
