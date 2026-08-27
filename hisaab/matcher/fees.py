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


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """The rates this matcher believes are in force. Integer basis points, never floats."""

    fee_bps_by_method: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_FEE_BPS)
    )
    gst_bps: int = DEFAULT_GST_BPS

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
        return f"{rates}; GST {self.gst_bps}bps on the fee"


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
    unpriced: tuple[str, ...] = ()

    @property
    def total_paise(self) -> int:
        return self.fee_paise + self.gst_paise

    @property
    def is_complete(self) -> bool:
        """True when every member's method had a declared rate."""
        return not self.unpriced

    def describe(self) -> str:
        return f"fee {self.fee_paise}p + GST {self.gst_paise}p = {self.total_paise}p"


def derive(
    members: Sequence[tuple[int, str]], schedule: FeeSchedule | None = None
) -> Deduction:
    """``(gross_paise, method)`` per payment -> the deduction the rates imply.

    Each member is priced and rounded on its own, then summed: see the module docstring on
    why that is not the same as pricing the batch total.
    """
    rates = schedule or FeeSchedule()
    fee_total = gst_total = 0
    unpriced: list[str] = []
    for gross_paise, method in members:
        bps = rates.fee_bps(method)
        if bps is None:
            unpriced.append(method)
            continue
        fee = mul_bps(gross_paise, bps)
        fee_total += fee
        gst_total += mul_bps(fee, rates.gst_bps)
    # Sorted and deduplicated so the note a caller builds from this is stable across runs
    # -- verdict files are compared byte for byte.
    return Deduction(fee_total, gst_total, tuple(sorted(set(unpriced))))


@dataclass(frozen=True, slots=True)
class Explanation:
    """A gap, and the rule that accounts for all of it.

    ``rule`` is the name that goes in the verdict note, so a resolved row says *why* it
    balanced rather than only that it did. Phase 4 step 5 emits the components.
    """

    rule: str
    fee_paise: int
    gst_paise: int

    @property
    def total_paise(self) -> int:
        return self.fee_paise + self.gst_paise


#: The rule that explains a gap of nothing. Named rather than special-cased so a clean-mode
#: row and a fee-bearing row come back through the same seam, and a report can say which
#: rule closed each one.
NO_DEDUCTION = "no deduction"

#: The rule that explains a gap equal to the derived fee plus its GST.
FEE_AND_GST = "gateway fee + GST at declared rates"


def explain_gap(
    gap_paise: int,
    members: Sequence[tuple[int, str]],
    schedule: FeeSchedule | None = None,
) -> tuple[Explanation | None, Deduction]:
    """Account for ``gap_paise`` exactly, or return ``None``. Also returns what was derived.

    ``gap_paise`` is ``gross of the members - what the bank credited``: positive when money
    is missing, which is the only direction a deduction can explain. The second element of
    the tuple is the derived deduction whether or not it matched, because the caller needs
    it for the note on an abstention -- "the model accounts for 2,622p of a 3,000p gap" is
    triage-able, and "unexplained" alone is not.

    **Two candidate rules, tried in order**, and the count is the thing to watch. Zero
    deduction has to be one of them or clean mode -- where the true fee genuinely is zero
    and the data is byte-identical to Phase 1 -- would fail every row the moment this
    module existed, since the schedule prices card at 200bps regardless of whether anyone
    charged it. So "explained" means *some* declared rule closes the gap to the paisa.

    That framing has a cost, and it is worth stating plainly before Phase 6 makes it
    bigger: every rule added is another chance to close a gap by coincidence rather than by
    truth. With two rules the risk is nil -- they can only both match when the derived fee
    is itself zero, in which case they are the same number and the decomposition is
    identical. With refunds, TDS and reserves as further candidates the combinations
    multiply, and the defence is that a resolved row must name the rule that closed it
    (``Explanation.rule``) and emit the components, so a coincidence is visible in the
    output instead of hiding inside a coverage percentage.

    A **negative** gap -- the bank credited more than the payments grossed -- is never
    explained here. No deduction adds money, so this returns ``None`` and lets the caller
    say so; treating it as a small residual to absorb would be how an over-credit becomes
    a clean bill of health.
    """
    derived = derive(members, schedule)
    if gap_paise == 0:
        return Explanation(NO_DEDUCTION, 0, 0), derived
    if gap_paise < 0:
        return None, derived
    # An incomplete model cannot explain anything: its total is a lower bound over the
    # members it could price, so a coincidental equality would be an accident, not a rule.
    if derived.is_complete and derived.total_paise == gap_paise:
        return Explanation(FEE_AND_GST, derived.fee_paise, derived.gst_paise), derived
    return None, derived


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
    assert d.total_paise == 2622 and d.is_complete
    assert "2622p" in d.describe()

    # GST sits on the fee. Applied to the gross instead it would be 20,000 paise here --
    # fifty times larger -- so this comparison is the composition check, not a spot value.
    assert d.gst_paise < d.fee_paise, "GST on the fee cannot approach the fee"
    assert mul_bps(rupees(1111), DEFAULT_GST_BPS) > 40 * d.gst_paise

    # A zero-rated method deducts nothing at all, so its gap is zero and it needs no model.
    z = derive([(rupees(5000), "pos_upi")])
    assert (z.fee_paise, z.gst_paise, z.total_paise) == (0, 0, 0)
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
    members = [(rupees(1111), "card")]
    ex, derived = explain_gap(2622, members)
    assert ex is not None and ex.rule == FEE_AND_GST
    assert (ex.fee_paise, ex.gst_paise, ex.total_paise) == (2222, 400, 2622)
    assert derived.total_paise == 2622, "the derivation comes back either way"

    # A gap of nothing is explained by the rule that says nothing was deducted -- this is
    # what keeps clean mode at 100% now that the schedule prices card at 200bps whether or
    # not anyone charged it.
    ex, _ = explain_gap(0, members)
    assert ex is not None and ex.rule == NO_DEDUCTION and ex.total_paise == 0

    # Off by a single paisa in either direction is not explained. No tolerance band exists
    # to widen: that is the point of the gate.
    for gap in (2621, 2623):
        ex, derived = explain_gap(gap, members)
        assert ex is None, f"gap {gap} must not be explained by a {derived.total_paise}p model"
    # ...and the derivation is still returned, so the note can quantify the shortfall.
    ex, derived = explain_gap(3000, members)
    assert ex is None and derived.total_paise == 2622

    # An over-credit is never explained: deductions do not add money.
    ex, _ = explain_gap(-500, members)
    assert ex is None, "a bank crediting more than the gross is not a fee"

    # An unpriced member cannot explain a gap even when the arithmetic coincides. The
    # 2622p below is the whole model for the card member, and it happens to equal the gap
    # -- accepting it would silently price the crypto member at zero.
    ex, derived = explain_gap(2622, [(rupees(1111), "card"), (rupees(500), "crypto")])
    assert ex is None, "an incomplete model must not close a gap by coincidence"
    assert derived.unpriced == ("crypto",)

    # A zero-rated batch: both rules give the same number, so there is no ambiguity to
    # resolve -- the decomposition is identical either way.
    ex, derived = explain_gap(0, [(rupees(5000), "pos_upi")])
    assert ex is not None and derived.total_paise == 0

    # A mixed-rail batch prices each member at its own rate. This is the case that would
    # have passed silently under a uniform table, so it is worth pinning: 215bps on the
    # corporate member and 200 on the card member cannot be recovered from one blended rate.
    mixed = derive([(rupees(1000), "card"), (rupees(1000), "corporate_card")])
    assert mixed.fee_paise == mul_bps(rupees(1000), 200) + mul_bps(rupees(1000), 215)
    assert mixed.fee_paise != 2 * mul_bps(rupees(1000), 200), "the rails must not blend"

    print("fees.py self-check ok  (rates are ASSUMPTIONS -- see ASSUMPTIONS.md #5-#9)")
