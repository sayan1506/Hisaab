"""The matcher's own fee model: what a declared rule can account for in a gap.

Phase 4 step 4. Tier 1 finds a settlement that agrees on date and amount; this module
answers the separate question of whether the money adds up, and the residual gate
(``tier1.py``) refuses the match when it does not.

**The rates here are the matcher's assumption about the world, deliberately not the
generator's number.** ``hisaab/generator/config.py`` has a ``FeeConfig`` with the same
shape, and importing it would be a shorter route to the same table -- which is exactly why
``tools/check_isolation.py`` check 6 forbids it. The point is not the import graph. It is
that a reconciliation engine in the real world reads its rates off a pricing page and a
contract, and is *wrong* when the counterparty charged something else. Re-deriving the fee
from an independently declared rate is what makes the residual a test; taking the rate
from whoever produced the data would make it a tautology.

Which is also why nothing here reads ``settlements.csv``'s ``fee_paise``, ``gst_paise`` or
``tds_paise`` columns, though ``load.py`` parses them and they are legitimate input.
Trusting a declared number is not explaining a gap: subtracting the stated fee would close
the residual the instant ``--fees`` populated it, so coverage would hold at 100% with no
model ever written and no number moving to say one was missing. A settlement whose stated
fee disagrees with its own published rate is then a finding rather than a rounding error
nobody looks at. ``tier1.py``'s self-check pins that with a settlement carrying a
``fee_paise`` that would close the gap if anyone read it.

What *is* shared, from ``hisaab/common/money.py``, is ``mul_bps`` -- the half-up-at-the-
paisa rounding rule. That is a deliberate opposite: two independent implementations of a
rounding rule that disagree on 399.96 paise produce an off-by-one-paisa residual, which is
a plausible wrong answer that nothing detects. Rates are duplicated so drift is *found*;
the rounding rule is shared so drift is *impossible*. Same reasoning as ``load.py``'s CSV
headers, in the other direction.

**Composition, and the order that matters.** The fee is a share of the gross; GST is a
share of the **fee**, never of the gross. Backwards, an 18% GST on a ₹10,000 gross is nine
times a 2% fee -- roughly fifty times the correct GST -- so the two rates are applied in
one function and one order on both sides of this codebase.

**Per payment, then summed.** A batched settlement's fee is the sum of its members' fees,
each rounded at the paisa, and not the rate applied to the batch total. The two differ by
a paisa or two whenever rounding does not distribute, and that difference is precisely the
kind of residual that gets waved away as noise. Already the Phase 5 shape: batching
changes the length of the member list and nothing here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..common.money import mul_bps

#: Gateway fee in integer basis points, by payment method. Verified 2026-08-26 against
#: https://razorpay.com/pricing/ and recorded in ASSUMPTIONS.md #5-#9. Still an assumption
#: rather than a fact: these are **list prices**, real rates are negotiated at volume, and
#: Indian MDR structures move with NPCI/RBI policy. Overridable per method with
#: ``--fee-bps METHOD=BPS`` precisely so a run can be re-pointed without a code change.
#:
#: The rail to read twice is ``pos_upi`` at 0 bps, and it is worth being exact about why,
#: because an earlier version of this table had **``upi``** at zero and that was wrong.
#: UPI carries zero *MDR* by mandate, which is true and widely repeated, but Razorpay's own
#: 2% platform fee still applies on the standard payment-gateway rail -- so a UPI sale is
#: not free to the merchant. Zero MDR and zero fee are different claims. POS terminals do
#: price UPI and RuPay debit at 0.00%, so the free rail is real; it is just a different one.
#:
#: What the free rail costs the *evidence*: a ``pos_upi`` settlement pays out at its gross,
#: so its residual is zero even under ``--fees`` and it resolves with no fee model at all.
#: Any claim that "the fee model moved the coverage number" has to be measured per method,
#: or the zero-rated rows quietly carry the result. Under the old table that was 36% of
#: rows; it is now ~6%, which makes ``--fees`` a considerably sharper test than it was.
DEFAULT_FEE_BPS: dict[str, int] = {
    "card": 200,
    "upi": 200,
    "netbanking": 200,
    "wallet": 200,
    "corporate_card": 215,
    "international_card": 300,
    "pos_upi": 0,
}

#: GST on the fee. 18%, and charged on the fee alone -- see the module docstring.
DEFAULT_GST_BPS = 1800

#: TDS withheld under §194-O, in basis points. **10, not 100** -- the section was cut from
#: 1% to 0.1% by the Finance (No. 2) Act 2024, effective 2024-10-01. ASSUMPTIONS.md #9 carries
#: the citation and #9a the caveat that matters more than the rate: whether a payment
#: aggregator withholds this at all is a modelling assumption, not a verified fact.
#:
#: Re-declared here rather than imported, for this module's whole reason to exist -- the
#: generator has its own copy and ``tools/check_isolation.py`` check 6 forbids reaching for
#: it. The duplication is what makes the residual a test.
#:
#: **Two properties set this rate apart from every other number in this file, and both matter
#: downstream.** It takes no method argument, because a tax rate is not a price -- so unlike
#: ``fee_bps`` there is no "unpriced method" case and no ``None`` return to abstain on. And it
#: applies to the **gross**, like the fee and unlike GST. The consequence is that ``pos_upi``
#: stops being a zero-deduction rail the moment ``--tds`` is on: its fee is 0 and its TDS is
#: not, so the row that used to settle at its gross no longer does.
DEFAULT_TDS_BPS = 10


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """The rates this matcher believes are in force. Integer basis points, never floats."""

    fee_bps_by_method: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_FEE_BPS)
    )
    gst_bps: int = DEFAULT_GST_BPS
    tds_bps: int = DEFAULT_TDS_BPS

    def fee_bps(self, method: str) -> int | None:
        """The rate for ``method``, or ``None`` if this schedule does not price it.

        ``None`` rather than a ``0`` default, and the distinction is the whole reason this
        returns an optional: "this method is free" and "we do not know what this method
        costs" are different facts, and defaulting the unknown one to zero turns a gap the
        matcher cannot model into a gap it silently mis-models. The caller must abstain,
        which is what ``derive`` does.
        """
        return self.fee_bps_by_method.get(method)

    def describe(self) -> str:
        """One line for a report header, so a run states the rates it assumed."""
        rates = ", ".join(
            f"{method} {bps}bps" for method, bps in sorted(self.fee_bps_by_method.items())
        )
        return (
            f"{rates}; GST {self.gst_bps}bps on the fee; "
            f"TDS {self.tds_bps}bps on the gross"
        )


@dataclass(frozen=True, slots=True)
class Deduction:
    """What a declared rule says was withheld from a set of payments.

    ``unpriced`` names the methods the schedule could not price. When it is non-empty the
    fee and GST below are computed over the *remaining* members only, so the total is a
    lower bound rather than a model -- and the caller must not treat it as an explanation.
    ``is_complete`` is the flag to branch on.
    """

    fee_paise: int
    gst_paise: int
    tds_paise: int = 0
    unpriced: tuple[str, ...] = ()

    @property
    def fee_and_gst_paise(self) -> int:
        """The gateway's own cut, without the tax withholding.

        Kept separate from ``total_paise`` because the two are *different rules*, not two
        views of one number: a run without ``--tds`` has a gap of exactly this, and a run
        with it has a gap of ``total_paise``. Collapsing them into one attribute is how a
        rule set silently stops being able to tell those two runs apart.
        """
        return self.fee_paise + self.gst_paise

    @property
    def total_paise(self) -> int:
        """Everything the declared rates say was withheld, tax included."""
        return self.fee_paise + self.gst_paise + self.tds_paise

    @property
    def is_complete(self) -> bool:
        """True when every member's method had a declared rate.

        Unchanged by TDS, and deliberately: the TDS rate takes no method, so it is always
        computable. But an unpriced member still makes the *fee* unknown, and every rule
        that includes TDS also includes the fee -- so a partial model stays unusable and
        this flag keeps meaning what it meant.
        """
        return not self.unpriced

    def describe(self) -> str:
        return (
            f"fee {self.fee_paise}p + GST {self.gst_paise}p + TDS {self.tds_paise}p "
            f"= {self.total_paise}p"
        )


def derive(
    members: Sequence[tuple[int, str]], schedule: FeeSchedule | None = None
) -> Deduction:
    """``(gross_paise, method)`` per payment -> the deduction the rates imply.

    Each member is priced and rounded on its own, then summed: see the module docstring on
    why that is not the same as pricing the batch total.
    """
    rates = schedule or FeeSchedule()
    fee_total = gst_total = tds_total = 0
    unpriced: list[str] = []
    for gross_paise, method in members:
        # TDS first, and **outside** the unpriced branch. It takes no method, so it is
        # computable for every member including one whose rail this schedule cannot price --
        # putting it after the ``continue`` would silently drop the tax on exactly the members
        # a partial model is already struggling with.
        tds_total += mul_bps(gross_paise, rates.tds_bps)
        bps = rates.fee_bps(method)
        if bps is None:
            unpriced.append(method)
            continue
        fee = mul_bps(gross_paise, bps)
        fee_total += fee
        gst_total += mul_bps(fee, rates.gst_bps)
    # Sorted and deduplicated so the note a caller builds from this is stable across runs
    # -- verdict files are compared byte for byte.
    return Deduction(fee_total, gst_total, tds_total, tuple(sorted(set(unpriced))))


@dataclass(frozen=True, slots=True)
class Explanation:
    """A gap, and the rule that accounts for all of it.

    ``rule`` is the name that goes in the verdict note, so a resolved row says *why* it
    balanced rather than only that it did. Phase 4 step 5 emits the components.
    """

    rule: str
    fee_paise: int
    gst_paise: int
    tds_paise: int = 0

    @property
    def total_paise(self) -> int:
        return self.fee_paise + self.gst_paise + self.tds_paise

    @property
    def terms(self) -> tuple[int, int, int]:
        """The component split, without the rule name.

        **This, not the rule name, is what two explanations are compared on** -- see
        ``explain_gap``. Two rules that close one gap with the same components have not
        produced an ambiguity; they have produced one answer twice.
        """
        return (self.fee_paise, self.gst_paise, self.tds_paise)


#: The rule that explains a gap of nothing. Named rather than special-cased so a clean-mode
#: row and a fee-bearing row come back through the same seam, and a report can say which
#: rule closed each one.
NO_DEDUCTION = "no deduction"

#: The rule that explains a gap equal to the derived fee plus its GST.
FEE_AND_GST = "gateway fee + GST at declared rates"

#: Phase 6. The two rules the TDS term adds, and the second one is easy to overlook.
#:
#: ``FEE_GST_TDS`` is the obvious one: a run with both ``--fees`` and ``--tds`` has a gap of
#: all three terms.
#:
#: ``TDS_ONLY`` is required for the *same reason* ``NO_DEDUCTION`` is, and leaving it out was
#: the bug this pair was written to avoid. ``--tds`` without ``--fees`` is a legal run: the
#: generator withholds tax and charges no gateway fee, so the true gap is the TDS alone. But
#: this schedule prices card at 200 bps whether or not anyone charged it, so ``FEE_GST_TDS``
#: over-predicts while ``NO_DEDUCTION`` under-predicts -- and without a rule for the tax by
#: itself, **every row on such a run would land in ``UNEXPLAINED_RESIDUAL``**. That is exactly
#: the failure ``NO_DEDUCTION`` exists to prevent one phase earlier, so the rule set has to be
#: the *combinations of the two independent withholdings* rather than a list of the flags
#: someone happened to think of.
TDS_ONLY = "TDS at the declared rate"
FEE_GST_TDS = "gateway fee + GST + TDS at declared rates"


def explain_gap(
    gap_paise: int,
    members: Sequence[tuple[int, str]],
    schedule: FeeSchedule | None = None,
) -> tuple[list[Explanation], Deduction]:
    """Every declared rule that accounts for ``gap_paise`` exactly. Also what was derived.

    ``gap_paise`` is ``gross of the members - what the bank credited``: positive when money
    is missing, which is the only direction a deduction can explain. The second element of
    the tuple is the derived deduction whether or not anything matched, because the caller
    needs it for the note on an abstention -- "the model accounts for 2,622p of a 3,000p gap"
    is triage-able, and "unexplained" alone is not.

    **Returns a list, and Phase 6 changed that from a single optional.** The old signature
    returned the *first* rule that closed the gap, which its own docstring called safe only
    while there were two rules -- because with two, both matching implies the derived fee is
    zero, and the two are then the same number. Phase 6 takes the count to four, so first-hit
    would silently pick a winner among rules that disagree. The caller resolves on exactly one
    and abstains on two or more; see ``tier1.py``.

    Phase 5 measured the identical precedence bug one layer down, in the Tier 2 subset search,
    and that number is why this landed **before** any new rule rather than alongside one: a
    unique-but-wrong answer under one hypothesis overrode a known ambiguity under another, for
    **one wrong match in 627** at seed 1, n=1000 -- invisible in every unit test.

    **The four rules are the combinations of two independent withholdings**, the gateway's cut
    and the tax, rather than a list of flags:

    ==================  ===================  ======================
    rule                closes a gap of      the run it belongs to
    ==================  ===================  ======================
    ``NO_DEDUCTION``    0                    clean mode
    ``FEE_AND_GST``     fee + GST            ``--fees``
    ``TDS_ONLY``        TDS                  ``--tds``
    ``FEE_GST_TDS``     fee + GST + TDS      ``--fees --tds``
    ==================  ===================  ======================

    Both zero-bearing rules are load-bearing for one reason: this schedule prices card at
    200 bps and withholds 10 bps of tax *whether or not anyone charged either*, so a run that
    charged neither needs a rule saying so, or every row becomes an unexplained residual.

    **Two explanations are compared on their components, never on their rule names**, and that
    distinction is the difference between working code and a 5% coverage regression. Measured
    on the committed clean-mode run: **3 of 60** settlements are zero-rated ``pos_upi`` rows
    where the gap is 0 and the derived deduction is also 0, so ``NO_DEDUCTION`` and
    ``FEE_AND_GST`` *both* match -- and with ``--tds`` off, all four rules do. Counting rule
    names would make every such row abstain as ambiguous and drop clean mode from 60/60 to
    57/60: an honest-looking abstention caused entirely by how the rules were counted. They are
    not two answers, they are one answer reached twice, and ``Explanation.terms`` is what says
    so. Deduplicating on the components collapses them, which leaves a genuine disagreement --
    two rules closing one gap with *different* splits -- as the only thing able to return a
    list longer than one.

    A **negative** gap -- the bank credited more than the payments grossed -- is never
    explained here. No deduction adds money, so this returns an empty list and lets the caller
    say so; treating it as a small residual to absorb would be how an over-credit becomes a
    clean bill of health.

    An **incomplete** model -- one with a member whose method this schedule cannot price --
    explains nothing that requires a rate, even when the arithmetic coincides: its fee is a
    lower bound over the members it could price, so an equality would be an accident rather
    than a rule.

    **``NO_DEDUCTION`` is exempt, and the ordering that makes it exempt is deliberate.** A
    credit equal to its members' gross is explained by the amounts alone; no rate is consulted,
    so an unpriced method cannot invalidate it. Nothing was withheld, so there is nothing to
    price. ``TDS_ONLY`` is *not* exempt despite the tax rate taking no method: firing it on an
    incomplete model would assert the unknown member's fee was zero, which is precisely what
    ``fee_bps`` returning ``None`` exists to prevent.
    """
    derived = derive(members, schedule)
    if gap_paise < 0:
        return [], derived

    fee, gst, tds = derived.fee_paise, derived.gst_paise, derived.tds_paise
    # Declared in ascending order of what they claim, which is what makes the survivor of a
    # duplicate the *weakest* rule accounting for the gap -- the honest one to publish.
    #
    # The flag is **whether the rule needs a rate at all**, and only ``NO_DEDUCTION`` does not.
    # That distinction is load-bearing rather than tidy: a credit equal to its members' gross
    # is explained by the amounts themselves, so an unpriced method cannot block it -- nothing
    # was withheld, so there is no gap to price. Gating the whole function on completeness
    # instead of gating each rule broke exactly that case, and ``tier1.py``'s ``unknown_exact``
    # fixture caught it: an unpriced ``crypto`` payment whose credit equals its gross went from
    # RESOLVED to UNEXPLAINED_RESIDUAL.
    #
    # ``TDS_ONLY`` sits on the other side of the line even though the tax rate takes no method
    # and is therefore always computable. Firing it on an incomplete model would assert that
    # the *fee* was zero on a member whose rate is unknown -- pricing the unpriced by the back
    # door, which is the one thing ``fee_bps`` returning ``None`` exists to prevent.
    candidates: tuple[tuple[Explanation, bool], ...] = (
        (Explanation(NO_DEDUCTION, 0, 0, 0), False),
        (Explanation(TDS_ONLY, 0, 0, tds), True),
        (Explanation(FEE_AND_GST, fee, gst, 0), True),
        (Explanation(FEE_GST_TDS, fee, gst, tds), True),
    )

    closing: list[Explanation] = []
    seen: set[tuple[int, int, int]] = set()
    for candidate, needs_rates in candidates:
        if needs_rates and not derived.is_complete:
            continue
        if candidate.total_paise != gap_paise or candidate.terms in seen:
            continue
        seen.add(candidate.terms)
        closing.append(candidate)
    return closing, derived


def unpriced_methods(methods: Iterable[str], schedule: FeeSchedule | None = None) -> list[str]:
    """Methods present in the data that this schedule cannot price.

    For the CLI to warn *once* at load time rather than once per row. A method the rates do
    not cover makes every row that uses it unexplainable, which is worth saying up front
    instead of leaving a reader to infer it from a pile of identical exceptions.
    """
    rates = schedule or FeeSchedule()
    return sorted({m for m in methods if rates.fee_bps(m) is None})


if __name__ == "__main__":
    from ..common.money import rupees

    schedule = FeeSchedule()
    assert schedule.fee_bps("card") == 200
    assert schedule.fee_bps("pos_upi") == 0, "zero-rated is a rate, not a missing entry"
    assert schedule.fee_bps("crypto") is None, "an unknown method must not default to free"
    assert "card 200bps" in schedule.describe() and "GST 1800bps" in schedule.describe()

    # The three-way distinction this whole optional-return exists for, in one line: a rate,
    # a zero rate, and no rate are three different answers and only the middle one is free.
    assert schedule.fee_bps("upi") == 200, (
        "UPI carries zero MDR but Razorpay's 2% platform fee still applies on the standard "
        "PG rail -- see DEFAULT_FEE_BPS. This assertion is here because the table once said 0."
    )
    # Shape, not values: the schedule has to price at least two methods *differently* or
    # ``fee_bps`` is a constant dressed up as a lookup and nothing downstream is testing
    # method-dependence. Kept as a count so correcting a rate does not break it.
    assert len({b for b in DEFAULT_FEE_BPS.values() if b}) >= 2, (
        "a uniform non-zero table makes the per-method model decorative"
    )

    # The track spec's worked example, end to end through this module: a Rs 1,111 card sale
    # at 2% is a Rs 22.22 fee, and 18% GST on that fee is 399.96 paise -> 400 half-up.
    d = derive([(rupees(1111), "card")])
    assert (d.fee_paise, d.gst_paise) == (2222, 400), d
    # Two totals now, and keeping them apart is the point: the gateway's own cut is what the
    # track spec's worked example computes, while ``total_paise`` is everything the declared
    # rates withhold. A single attribute would make a ``--fees`` run and a ``--fees --tds``
    # run indistinguishable at exactly the layer that has to tell them apart.
    assert d.fee_and_gst_paise == 2622 and d.is_complete
    assert d.tds_paise == 111, f"10bps of a Rs 1,111 gross, got {d.tds_paise}"
    assert d.total_paise == 2733 == 2622 + 111
    assert "2733p" in d.describe() and "TDS 111p" in d.describe()

    # GST sits on the fee. Applied to the gross instead it would be 20,000 paise here --
    # fifty times larger -- so this comparison is the composition check, not a spot value.
    assert d.gst_paise < d.fee_paise, "GST on the fee cannot approach the fee"
    assert mul_bps(rupees(1111), DEFAULT_GST_BPS) > 40 * d.gst_paise

    # **The zero-rated rail stops being a zero-*deduction* rail, and this is the assertion
    # that says so.** ``pos_upi`` is 0 bps, so its fee and GST are nothing -- but the tax rate
    # takes no method, so TDS is withheld all the same. Since Phase 4 a POS settlement paid out
    # at its gross and resolved with no fee model at all; under ``--tds`` it does not. That
    # matters beyond bookkeeping: every "the residual moved" claim in this codebase is measured
    # per method precisely because the free rows used to carry the result silently, and this is
    # the phase where there are no free rows left.
    z = derive([(rupees(5000), "pos_upi")])
    assert (z.fee_paise, z.gst_paise, z.fee_and_gst_paise) == (0, 0, 0)
    assert z.tds_paise == 500 and z.total_paise == 500, (
        "a zero-rated method still has tax withheld -- the rate takes no method"
    )
    assert z.is_complete, "zero-rated is fully modelled, unlike unpriced"

    # Per member, then summed -- not the rate on the batch total. Two Rs 1,111 card
    # payments must give exactly twice the single-payment fee.
    pair = derive([(rupees(1111), "card"), (rupees(1111), "card")])
    assert (pair.fee_paise, pair.gst_paise) == (4444, 800), pair
    # ...and a case where rounding genuinely does not distribute, so the discipline earns
    # its keep: 25p at 200bps is half a paisa, which rounds up to 1p *each*, while the
    # 50p total would round to 1p for the pair.
    split = derive([(25, "card"), (25, "card")])
    assert split.fee_paise == 2, f"per-member rounding, got {split.fee_paise}"
    assert derive([(50, "card")]).fee_paise == 1, "the batch total rounds differently"

    # An unpriced method poisons the model rather than being counted as free.
    u = derive([(rupees(1111), "card"), (rupees(500), "crypto")])
    assert not u.is_complete and u.unpriced == ("crypto",)
    assert u.fee_paise == 2222, "the priced member is still priced, as a lower bound"
    assert unpriced_methods(["card", "upi", "crypto", "crypto"]) == ["crypto"]
    assert unpriced_methods(["card", "upi"]) == []

    # --- explain_gap ------------------------------------------------------
    # Returns a *list* since Phase 6. Every assertion below is written against the arity,
    # because the old single-optional shape is what first-hit precedence was hiding behind.
    members = [(rupees(1111), "card")]

    # Each of the four rules, hit exactly. The gaps are different numbers, so exactly one
    # rule can close each -- which is what makes these four separate facts rather than one.
    for gap, rule, terms in (
        (0, NO_DEDUCTION, (0, 0, 0)),
        (111, TDS_ONLY, (0, 0, 111)),
        (2622, FEE_AND_GST, (2222, 400, 0)),
        (2733, FEE_GST_TDS, (2222, 400, 111)),
    ):
        closing, derived = explain_gap(gap, members)
        assert len(closing) == 1, f"gap {gap}: expected one rule, got {[e.rule for e in closing]}"
        assert closing[0].rule == rule and closing[0].terms == terms, closing[0]
        assert closing[0].total_paise == gap
        assert derived.total_paise == 2733, "the derivation comes back either way"

    # ``TDS_ONLY`` is the rule that keeps a ``--tds``-without-``--fees`` run alive, and this is
    # the assertion that would fail if someone pruned it as redundant. The schedule prices card
    # at 200bps whether or not anyone charged it, so without this rule the 111p gap above would
    # be unexplainable and *every* row on such a run would land in UNEXPLAINED_RESIDUAL --
    # the same failure NO_DEDUCTION prevents one phase earlier.
    assert explain_gap(111, members)[0][0].rule == TDS_ONLY

    # Off by a single paisa in either direction is not explained. No tolerance band exists
    # to widen: that is the point of the gate.
    for gap in (2732, 2734):
        closing, derived = explain_gap(gap, members)
        assert not closing, f"gap {gap} must not be explained by a {derived.total_paise}p model"
    # ...and the derivation is still returned, so the note can quantify the shortfall.
    closing, derived = explain_gap(3000, members)
    assert not closing and derived.total_paise == 2733

    # An over-credit is never explained: deductions do not add money.
    assert not explain_gap(-500, members)[0], "a bank crediting more than the gross is not a fee"

    # An unpriced member cannot explain a gap even when the arithmetic coincides. Note this
    # holds for the *tax-only* rule too, though TDS is computable without a rate for the
    # method: admitting it would price the crypto member's fee at zero by the back door.
    mixed_pool = [(rupees(1111), "card"), (rupees(500), "crypto")]
    closing, derived = explain_gap(2622, mixed_pool)
    assert not closing, "an incomplete model must not close a gap by coincidence"
    assert derived.unpriced == ("crypto",)
    assert not explain_gap(derived.tds_paise, mixed_pool)[0], (
        "TDS is computable for an unpriced member, but no rule may fire while the fee is not"
    )
    # ...while a **zero** gap is still explained, unpriced member and all, because that rule
    # consults no rate. Pinned here as well as in ``tier1.py``'s ``unknown_exact`` fixture:
    # gating the whole function on completeness rather than per rule broke exactly this, and
    # only the tier-1 fixture noticed.
    closing, _ = explain_gap(0, mixed_pool)
    assert len(closing) == 1 and closing[0].rule == NO_DEDUCTION, [e.rule for e in closing]

    # **Two rules matching is not an ambiguity when their components agree**, and this is the
    # case that would have cost 5% of clean-mode coverage if the dedup keyed on rule names.
    # A zero-rated member with no tax charged: the gap is 0 and all four rules predict 0, so
    # they collapse to the single weakest one. Measured on the committed clean-mode run --
    # 3 of 60 settlements are exactly this shape.
    free = FeeSchedule(tds_bps=0)
    closing, derived = explain_gap(0, [(rupees(5000), "pos_upi")], free)
    assert len(closing) == 1 and closing[0].rule == NO_DEDUCTION, [e.rule for e in closing]
    assert derived.total_paise == 0
    # Same collapse one step up: a fee-bearing member on a run with no tax. FEE_AND_GST and
    # FEE_GST_TDS are then the same three numbers, and the weaker name survives.
    closing, _ = explain_gap(2622, members, free)
    assert len(closing) == 1 and closing[0].rule == FEE_AND_GST, [e.rule for e in closing]

    # **A genuine ambiguity: two rules, one gap, different splits.** Of the six pairs the four
    # rules admit, this is the only one that can disagree -- it needs ``fee + gst == tds`` with
    # both non-zero. At the declared rates that is unreachable (fee-and-GST is 236bps effective
    # against TDS's 10, so they never meet), which is why the branch is exercised through a
    # ``--fee-bps`` override: 9bps on a Rs 105 gross gives a 9p fee plus 2p GST against 11p of
    # tax. That is not a contrived number for its own sake -- ``--fee-bps`` exists so a run can
    # be re-pointed at a negotiated rate, and this is what happens when one lands here.
    cheap = FeeSchedule(fee_bps_by_method={"card": 9})
    closing, derived = explain_gap(11, [(10_500, "card")], cheap)
    assert len(closing) == 2, [e.rule for e in closing]
    assert {e.rule for e in closing} == {TDS_ONLY, FEE_AND_GST}
    assert {e.terms for e in closing} == {(0, 0, 11), (9, 2, 0)}
    # ...and the weakest-first declaration order is what the caller publishes, so the note is
    # stable rather than dependent on dict or set iteration.
    assert closing[0].rule == TDS_ONLY, "rules are tried in ascending order of what they claim"

    # A mixed-rail batch prices each member at its own rate. This is the case that would
    # have passed silently under a uniform table, so it is worth pinning: 215bps on the
    # corporate member and 200 on the card member cannot be recovered from one blended rate.
    mixed = derive([(rupees(1000), "card"), (rupees(1000), "corporate_card")])
    assert mixed.fee_paise == mul_bps(rupees(1000), 200) + mul_bps(rupees(1000), 215)
    assert mixed.fee_paise != 2 * mul_bps(rupees(1000), 200), "the rails must not blend"
    # The tax, by contrast, *is* blended across the two rails, and that asymmetry is the
    # whole character of the term: the rate takes no method, so it is the one deduction a
    # mixed batch cannot disagree about.
    assert mixed.tds_paise == 2 * mul_bps(rupees(1000), DEFAULT_TDS_BPS)

    # **The rounding convention, pinned by a pair that separates the two readings.** The
    # assertion above cannot do it: 10bps of a Rs 1,000 gross is exactly 100p, so
    # per-member-then-sum and rate-on-the-total both give 200p and either implementation
    # passes. Two Rs 102.50 members do separate them -- 10.25p rounds to 10p each for 20p
    # summed, while 10bps of the 20,500p total is 20.5p and rounds to 21p.
    #
    # One paisa, and it decides a term the scorer grades on its own (ASSUMPTIONS.md #25),
    # so it cannot be absorbed by a correct total. The generator withholds per member --
    # ``story.build`` sums ``_tds`` over the batch one payment at a time -- and the two
    # sides derive independently by design, so nothing but this assertion would notice the
    # matcher switching. Measured on ``--fees --tds --batching`` at n=200: 19 of 49
    # multi-member batches separate the conventions, always by 1p. The common case, not a
    # corner -- which is why agreement at 100% today is not evidence the rule is right.
    pair = derive([(10_250, "card"), (10_250, "card")])
    assert pair.tds_paise == 20 == 2 * mul_bps(10_250, DEFAULT_TDS_BPS)
    assert mul_bps(20_500, DEFAULT_TDS_BPS) == 21, "the fixture must actually separate them"
    assert pair.tds_paise != mul_bps(20_500, DEFAULT_TDS_BPS), (
        "TDS is withheld per member and summed, never taken on the batch total"
    )

    print("fees.py self-check ok  (rates are ASSUMPTIONS -- see ASSUMPTIONS.md #5-#9)")
