"""Derived deductions against the ones ``settlements.csv`` declares -- **reported, never consumed**.

Phase 6 step 2. Every other module in ``hisaab/matcher/`` re-derives what was withheld from an
independently declared rate table and never looks at the ``fee_paise``, ``gst_paise`` or
``tds_paise`` columns, though ``load.py`` parses all three. This module is the one place that
reads them, and it exists to make that duplication *pay*: two tables that are never compared
drift silently, and the whole argument for duplicating them (``fees.py``'s docstring,
``tools/check_isolation.py`` check 6) is that drift should be **found**.

**The output is a report. Nothing here returns a ``Verdict``, and nothing on the resolution
path may import it.** That is not stylistic. Subtracting a declared fee would close the
residual the instant ``--fees`` populated it, so coverage would read 100% with no model ever
written -- the tautology ``fees.py`` was built to avoid. A comparison that only *reports*
cannot do that, and ``check_isolation.py`` check 7 enforces the direction rather than trusting
this paragraph, because a stated discipline with no backstop is the kind of rule that survives
exactly until someone needs a number.

**Per term, and that is measured rather than tidy.** Comparing one blended total would hide
real drift: with the card rate set 1 bp off on a ``--fees --tds --batching`` run at n=200, the
fee term diverges on 53 of 120 settlements but GST on only 49 -- on 4 rows an 18% share of a
small fee delta rounds to the very same paisa, so the fee is wrong and the GST agrees. Those 4
rows are invisible to any check that does not look at the fee on its own. Same reasoning as the
scorer grading a decomposition term by term instead of on its total (ASSUMPTIONS.md #25).

**Divergence is not an error, and reading it as one would make this useless on the
deliverable's own flag set.** A run with ``--fees`` but no ``--tds`` withholds no tax, while
this matcher's schedule says 10 bps -- so every row diverges on TDS, and *both sides are
right*. Measured at n=200, seed 42:

    clean                     fee 184/200, gst 184/200, tds 200/200 diverge
    --fees                    tds 200/200 diverge; fee and gst agree everywhere
    --tds                     fee 184/200, gst 184/200 diverge (the 16 that agree are
                              pos_upi, zero-rated, so the derivation is 0 too)
    --fees --tds              every term agrees on every row
    --fees --tds --batching   every term agrees on every row

So the interesting signal is not *whether* a term diverges but *how*. A term where every
disagreeing row declares zero against a non-zero derivation says the counterparty did not apply
that deduction at all -- a flag mismatch. A term where both sides are non-zero and still
disagree is the rate being wrong, which is the finding this module is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .fees import FeeSchedule, derive
from .load import Dataset, Settlement

#: The three terms, in the order they compose: the fee is a share of the gross, GST a share of
#: the fee, TDS a share of the gross again. Reported in this order so a reader meets a wrong
#: fee before the GST it propagates into.
TERMS: tuple[str, ...] = ("fee", "gst", "tds")


class Shape(str, Enum):
    """How a term's divergence looks across the run -- see the module docstring."""

    #: Every compared row agrees, term by term and paisa for paisa.
    AGREES = "AGREES"
    #: Every disagreeing row declares 0 against a non-zero derivation. The counterparty did
    #: not apply this deduction; this schedule says it should have. Expected on a run whose
    #: flags do not match the assumed table, and **not** a finding on its own.
    NOT_WITHHELD = "NOT_WITHHELD"
    #: The mirror: the counterparty withheld something this schedule prices at zero. Worth
    #: more attention than ``NOT_WITHHELD``, because money left the merchant that no declared
    #: rate accounts for.
    NOT_MODELLED = "NOT_MODELLED"
    #: Both sides are non-zero and still disagree, on at least one row. The rate table is
    #: wrong for those rows -- the drift the duplication exists to surface.
    RATE_DRIFT = "RATE_DRIFT"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class TermReport:
    """One term's comparison across every settlement whose membership is declared."""

    term: str
    compared: int
    diverging: int
    shape: Shape
    #: Smallest and largest absolute disagreement in paise, over diverging rows only.
    min_delta: int
    max_delta: int
    #: Signed total, derived minus declared. The sign is the direction of the model's error:
    #: positive means this schedule predicts more was withheld than the file says.
    net_delta: int
    #: Up to three diverging settlement ids, in file order, so a reader can go and look.
    examples: tuple[str, ...]

    @property
    def is_finding(self) -> bool:
        """Whether this term is a *finding* rather than a flag mismatch.

        ``NOT_WITHHELD`` is excluded deliberately: it is the expected shape whenever the run's
        flags and the assumed rates disagree, which includes every ``--fees``-without-``--tds``
        run. Calling it a finding would put 200 alarms on a correct run and teach the reader to
        skip the line.
        """
        return self.shape in (Shape.RATE_DRIFT, Shape.NOT_MODELLED)

    def describe(self) -> str:
        if self.shape is Shape.AGREES:
            return f"{self.term}: {self.compared}/{self.compared} agree"
        sign = "+" if self.net_delta >= 0 else "-"
        return (
            f"{self.term}: {self.diverging}/{self.compared} diverge, "
            f"{self.min_delta}-{self.max_delta}p each, net {sign}{abs(self.net_delta)}p "
            f"({self.shape})"
        )


@dataclass(frozen=True, slots=True)
class AdjustmentReport:
    """Every term's comparison, plus the rows that could not be compared at all."""

    terms: tuple[TermReport, ...]
    #: Settlements with no declared membership. Nothing can be derived for them, so they are
    #: excluded from every count above rather than silently scored as agreeing --
    #: ``--settlement-report-late`` withholds membership, and a withheld row is unknown, not
    #: clean. The same distinction ``fees.fee_bps`` draws by returning ``None``.
    unmeasurable: int
    #: Methods this schedule cannot price. Their members are dropped from the derivation, so
    #: any term touching them is a lower bound and no divergence on them means anything.
    unpriced: tuple[str, ...]

    @property
    def findings(self) -> tuple[TermReport, ...]:
        return tuple(t for t in self.terms if t.is_finding)

    def lines(self) -> tuple[str, ...]:
        """Report lines for the CLI."""
        if self.unpriced:
            # An unpriced method makes every term a lower bound, so reporting deltas would
            # invite reading a modelling gap as counterparty drift.
            return (
                f"  declared-vs-derived: not compared -- no rate for "
                f"{', '.join(self.unpriced)}, so every derived term is a lower bound",
            )
        out = [f"  declared-vs-derived: {t.describe()}" for t in self.terms]
        if self.unmeasurable:
            out.append(
                f"  {self.unmeasurable} settlement(s) have no declared membership, so their "
                f"deductions cannot be derived and are excluded above"
            )
        if self.findings:
            out.append(
                f"  ^ {len(self.findings)} term(s) disagree where both sides are non-zero -- "
                f"the assumed rate is wrong for those rows, which is a finding rather than a "
                f"rounding artefact"
            )
        return tuple(out)


def _declared(settlement: Settlement) -> dict[str, int]:
    """The three declared columns. **The only read of them in the whole matcher.**"""
    return {
        "fee": settlement.fee_paise,
        "gst": settlement.gst_paise,
        "tds": settlement.tds_paise,
    }


def compare(dataset: Dataset, schedule: FeeSchedule | None = None) -> AdjustmentReport:
    """Derive each settlement's deductions and compare them, term by term, to its own row.

    Reads ``settlements.csv``'s declared columns -- and returns a report. See the module
    docstring on why that direction is the entire design.
    """
    rates = schedule or FeeSchedule()
    by_payment = dataset.payments_by_id()

    deltas: dict[str, list[tuple[str, int, int]]] = {term: [] for term in TERMS}
    compared = 0
    unmeasurable = 0
    unpriced: set[str] = set()

    for settlement in dataset.settlements:
        members: list[tuple[int, str]] = []
        for pid in dataset.items.get(settlement.settlement_id, ()):
            payment = by_payment.get(pid)
            if payment is not None:
                members.append((payment.gross_paise, payment.method))
        if not members:
            # No declared membership, or none of it resolves to a payment. Either way nothing
            # can be derived, and an underived row is unknown rather than agreeing.
            unmeasurable += 1
            continue

        deduction = derive(members, rates)
        unpriced.update(deduction.unpriced)
        declared = _declared(settlement)
        derived = {
            "fee": deduction.fee_paise,
            "gst": deduction.gst_paise,
            "tds": deduction.tds_paise,
        }
        compared += 1
        for term in TERMS:
            if derived[term] != declared[term]:
                deltas[term].append(
                    (settlement.settlement_id, derived[term], declared[term])
                )

    reports: list[TermReport] = []
    for term in TERMS:
        rows = deltas[term]
        if not rows:
            reports.append(TermReport(term, compared, 0, Shape.AGREES, 0, 0, 0, ()))
            continue
        magnitudes = [abs(d - c) for _sid, d, c in rows]
        # The order of these tests is the honest one: a term is only called drift when neither
        # side is zero on some row, because a zero on either side is a *structural* difference
        # (the deduction was not applied, or is not modelled) rather than a wrong rate.
        if all(c == 0 for _sid, _d, c in rows):
            shape = Shape.NOT_WITHHELD
        elif all(d == 0 for _sid, d, _c in rows):
            shape = Shape.NOT_MODELLED
        else:
            shape = Shape.RATE_DRIFT
        reports.append(
            TermReport(
                term=term,
                compared=compared,
                diverging=len(rows),
                shape=shape,
                min_delta=min(magnitudes),
                max_delta=max(magnitudes),
                net_delta=sum(d - c for _sid, d, c in rows),
                examples=tuple(sid for sid, _d, _c in rows[:3]),
            )
        )
    return AdjustmentReport(tuple(reports), unmeasurable, tuple(sorted(unpriced)))


if __name__ == "__main__":
    from datetime import date, datetime, timezone

    from .load import Credit as C
    from .load import Dataset as D
    from .load import Payment as P
    from .load import Settlement as S

    mon = date(2026, 8, 3)

    def payment(pid: str, gross: int, method: str = "card") -> P:
        return P(pid, f"order_{pid}", datetime(2026, 8, 3, 9, tzinfo=timezone.utc),
                 gross, method, "INR", "captured")

    def settlement(sid: str, net: int, fee: int, gst: int, tds: int) -> S:
        return S(sid, mon, net, fee, gst, tds, "8104")

    def dataset(payments, settlements, items) -> D:
        return D(tuple(payments), tuple(settlements), (C("C0001", mon, 1, "x"),), (), items)

    # 2% of 85,358p is 1,707p, 18% GST on that is 307p, 10bps TDS on the gross is 85p.
    honest = dataset(
        [payment("pay_0001", 85_358)],
        [settlement("setl_0001", 83_259, 1_707, 307, 85)],
        {"setl_0001": ("pay_0001",)},
    )
    rep = compare(honest)
    assert [t.shape for t in rep.terms] == [Shape.AGREES] * 3, rep
    assert not rep.findings and rep.unmeasurable == 0
    assert all("1/1 agree" in t.describe() for t in rep.terms)

    # A run that withheld no tax, against a schedule that says 10bps. **Every row diverges on
    # TDS and nothing is wrong** -- this is `--fees` without `--tds`, measured at 200/200. The
    # shape has to come back NOT_WITHHELD rather than drift, or the deliverable's own flag set
    # reports 200 findings and the line becomes noise a reader learns to skip.
    no_tax = dataset(
        [payment("pay_0001", 85_358)],
        [settlement("setl_0001", 83_344, 1_707, 307, 0)],
        {"setl_0001": ("pay_0001",)},
    )
    rep = compare(no_tax)
    tds = next(t for t in rep.terms if t.term == "tds")
    assert tds.shape is Shape.NOT_WITHHELD and tds.diverging == 1, tds
    assert not tds.is_finding, "a flag mismatch is not a finding"
    assert not rep.findings, "and it must not reach the findings list either"
    assert tds.net_delta == 85, "derived minus declared: this schedule predicts 85p more"

    # The mirror, and it *is* a finding: money left the merchant that no rate accounts for.
    # pos_upi is priced at 0bps, so a fee on it is not drift -- it is unmodelled withholding.
    unmodelled = dataset(
        [payment("pay_0001", 85_358, "pos_upi")],
        [settlement("setl_0001", 83_273, 2_000, 0, 85)],
        {"setl_0001": ("pay_0001",)},
    )
    rep = compare(unmodelled)
    fee = next(t for t in rep.terms if t.term == "fee")
    assert fee.shape is Shape.NOT_MODELLED and fee.is_finding, fee
    assert fee.net_delta == -2_000, "negative: they withheld more than the model predicts"

    # **A wrong rate: both sides non-zero, disagreeing.** The finding this module is for.
    drift = dataset(
        [payment("pay_0001", 85_358)],
        [settlement("setl_0001", 83_366, 1_600, 307, 85)],
        {"setl_0001": ("pay_0001",)},
    )
    rep = compare(drift)
    fee = next(t for t in rep.terms if t.term == "fee")
    assert fee.shape is Shape.RATE_DRIFT and fee.is_finding, fee
    assert (fee.min_delta, fee.max_delta) == (107, 107) and fee.examples == ("setl_0001",)
    assert any("finding" in line for line in rep.lines())

    # **The case that forces per-term comparison.** 2% of 160,000p is 3,200p and 18% of that
    # is 576p -- but 18% of 3,201p is 576.18p, which is also 576p half-up. So a counterparty
    # one paisa off on the fee declares a GST identical to the derived one: the row is
    # internally consistent, the fee is wrong, and the GST agrees. A blended total or a
    # GST-only check calls this run clean. Measured on real data: 4 of 120 settlements have
    # exactly this shape at card 199bps.
    hidden = dataset(
        [payment("pay_0001", 160_000)],
        [settlement("setl_0001", 156_063, 3_201, 576, 160)],
        {"setl_0001": ("pay_0001",)},
    )
    rep = compare(hidden)
    fee = next(t for t in rep.terms if t.term == "fee")
    gst = next(t for t in rep.terms if t.term == "gst")
    assert fee.diverging == 1 and fee.shape is Shape.RATE_DRIFT, fee
    assert (fee.min_delta, fee.net_delta) == (1, -1), fee
    assert gst.diverging == 0 and gst.shape is Shape.AGREES, (
        "the whole reason the comparison is per term: this GST agrees while its fee does not"
    )

    # Withheld membership is *unknown*, not agreeing. Counting it as a match would let
    # --settlement-report-late improve the agreement rate by removing evidence.
    withheld = dataset(
        [payment("pay_0001", 85_358)],
        [settlement("setl_0001", 83_259, 1_707, 307, 85),
         settlement("setl_0002", 50_000, 999, 999, 999)],
        {"setl_0001": ("pay_0001",)},
    )
    rep = compare(withheld)
    assert rep.unmeasurable == 1 and rep.terms[0].compared == 1, rep
    assert all(t.shape is Shape.AGREES for t in rep.terms), (
        "the withheld row must not be compared -- its declared 999s are not disagreements"
    )
    assert any("no declared membership" in line for line in rep.lines())

    # An unpriced method makes every term a lower bound, so the whole comparison stands down
    # rather than reporting a modelling gap as counterparty drift.
    exotic = dataset(
        [payment("pay_0001", 85_358, "crypto")],
        [settlement("setl_0001", 85_358, 0, 0, 0)],
        {"setl_0001": ("pay_0001",)},
    )
    rep = compare(exotic)
    assert rep.unpriced == ("crypto",), rep
    assert len(rep.lines()) == 1 and "lower bound" in rep.lines()[0], rep.lines()

    print("adjustments.py self-check ok  (declared columns are read here and nowhere else)")
