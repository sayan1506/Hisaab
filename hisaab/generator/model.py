"""The story's entities, and the frozen shape of every file they emit.

Two design decisions live here, and both are load-bearing:

**Decision #10 — cardinality.** ``Settlement.payment_ids`` and
``Credit.settlement_ids`` are ``list[str]`` from day one, even though both hold
exactly one element in clean mode. Modelling them as scalars would force a
data-model rewrite in Phase 5, which is the phase you least want to be
refactoring in.

**Generate forward, strip backward.** A ``Credit`` object knows exactly which
settlements and payments produced it -- that is how the generator built it. Its
``csv_row()`` emits **four fields**: row_id, value_date, amount_paise, narration.
The linkage exists in memory and lands in ``truth.json``; it never reaches
``bank_statement.csv``. Keeping ``csv_header()`` next to the dataclass is what
makes that discipline visible at review time rather than buried in a writer.

Every ``csv_row()`` returns strings already, so the CSV writer does no formatting
and two runs cannot differ by a repr change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from ..common.money import paise

# --------------------------------------------------------------------------
# Frozen file schemas. Invariant I9 asserts each written header equals these,
# in order. Appendix A of the track spec is the source; do not widen them --
# every field added to the bank statement is difficulty deleted from the
# submission.
# --------------------------------------------------------------------------

PAYMENTS_HEADER = (
    "payment_id", "order_id", "captured_at", "gross_paise", "method", "currency", "status",
)
SETTLEMENTS_HEADER = (
    "settlement_id", "settled_on", "net_paise", "fee_paise", "gst_paise", "tds_paise", "utr",
)
SETTLEMENT_ITEMS_HEADER = ("settlement_id", "payment_id")
BANK_HEADER = ("row_id", "value_date", "amount_paise", "narration")
REFUNDS_HEADER = ("refund_id", "payment_id", "created_at", "amount_paise")

#: The currency these books are kept in, and the default every payment carries unless ``--fx``
#: moves it. Named here rather than repeated as a literal because Phase 8 gave it a second
#: reader (``story._draw_fx`` needs "the value that is *not* foreign"), and two spellings of one
#: fact is how a later widening changes one of them.
#:
#: ``matcher/tier1.py`` declares its **own** ``HOME_CURRENCY`` and must keep doing so: the
#: matcher cannot import the generator (``tools/check_isolation.py`` check 6), so this is not a
#: shared constant but the same assumption stated independently on both sides. Their agreement
#: is a fact about the data rather than a fact about the code, which is what makes the matcher's
#: reading of the column a real inference instead of a lookup.
HOME_CURRENCY = "INR"


def iso_utc(dt: datetime) -> str:
    """Aware datetime -> ``2026-08-10T11:04:22Z``.

    Refuses a naive datetime: an unlabelled local time written as ``Z`` is a
    silent one-day error in Phase 4's date window (trap 3).
    """
    if dt.tzinfo is None:
        raise ValueError(f"refusing to serialise a naive datetime: {dt!r}")
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, slots=True)
class Payment:
    """A captured payment, as the merchant's gateway export shows it."""

    payment_id: str
    order_id: str
    captured_at: datetime      # IST-aware; emitted as UTC Z
    gross_paise: int
    method: str
    currency: str = HOME_CURRENCY   # only --fx ever changes this (story._draw_fx)
    status: str = "captured"

    def __post_init__(self) -> None:
        paise(self.gross_paise)
        assert self.gross_paise > 0, f"{self.payment_id}: gross must be positive"
        assert self.captured_at.tzinfo is not None, f"{self.payment_id}: naive captured_at"

    @property
    def business_date(self) -> date:
        """The IST calendar date. *The* date for all business logic.

        Never use ``captured_at.date()`` on the UTC projection -- see trap 3.
        """
        return self.captured_at.date()

    @staticmethod
    def csv_header() -> tuple[str, ...]:
        return PAYMENTS_HEADER

    def csv_row(self) -> tuple[str, ...]:
        return (
            self.payment_id,
            self.order_id,
            iso_utc(self.captured_at),
            str(self.gross_paise),
            self.method,
            self.currency,
            self.status,
        )


@dataclass(frozen=True, slots=True)
class Refund:
    """A refund. Phase 1 emits none -- refunds.csv is header-only."""

    refund_id: str
    payment_id: str
    created_at: datetime
    amount_paise: int

    def __post_init__(self) -> None:
        paise(self.amount_paise)
        assert self.amount_paise > 0, f"{self.refund_id}: refund must be positive"

    @staticmethod
    def csv_header() -> tuple[str, ...]:
        return REFUNDS_HEADER

    def csv_row(self) -> tuple[str, ...]:
        return (
            self.refund_id,
            self.payment_id,
            iso_utc(self.created_at),
            str(self.amount_paise),
        )


@dataclass(frozen=True, slots=True)
class Settlement:
    """A gateway settlement. ``payment_ids`` is a list from day one (decision #10)."""

    settlement_id: str
    settled_on: date
    payment_ids: list[str]
    net_paise: int
    fee_paise: int = 0
    gst_paise: int = 0
    tds_paise: int = 0
    utr: str = ""

    def __post_init__(self) -> None:
        for amount in (self.net_paise, self.fee_paise, self.gst_paise, self.tds_paise):
            paise(amount)
        assert self.payment_ids, f"{self.settlement_id}: a settlement needs payments"
        assert len(set(self.payment_ids)) == len(self.payment_ids), (
            f"{self.settlement_id}: duplicate payment_ids"
        )
        assert self.net_paise > 0, f"{self.settlement_id}: net must be positive"

    @staticmethod
    def csv_header() -> tuple[str, ...]:
        return SETTLEMENTS_HEADER

    def csv_row(self) -> tuple[str, ...]:
        return (
            self.settlement_id,
            self.settled_on.isoformat(),
            str(self.net_paise),
            str(self.fee_paise),
            str(self.gst_paise),
            str(self.tds_paise),
            self.utr,
        )

    def item_rows(self) -> list[tuple[str, str]]:
        """Rows for ``settlement_items.csv`` -- the membership declaration.

        This is what Tier 2 reads and what ``--settlement-report-late`` withholds
        in Phase 8, forcing real subset-sum instead of reading the answer off the
        settlement report. It links payments to settlements only; the bank link
        stays the hard part, which is the whole task.
        """
        return [(self.settlement_id, pid) for pid in self.payment_ids]


@dataclass(frozen=True, slots=True)
class Decomposition:
    """The expected balance for one credit, to the paisa.

    Lands in ``truth.json`` so Phase 4's *prove* stage can be scored against
    truth's arithmetic, not just against ID sets. In clean mode every deduction
    is zero and ``expected_credit_paise == gross_paise``.
    """

    gross_paise: int
    fee_paise: int = 0
    gst_paise: int = 0
    tds_paise: int = 0
    refunds_paise: int = 0
    reserve_paise: int = 0
    #: Phase 8 step 2b (``--fx``): how far the rate moved between capture and settlement, in
    #: paise, on this credit's members. **The only SIGNED term here, and the only one that is
    #: added rather than subtracted.** Under design (b) the payout is correct at the
    #: settlement-day rate while ``payments.csv`` keeps the stale capture-rate gross, so the
    #: credit really is ``gross + fx - fee - gst - tds - refunds - reserve`` and truth has to
    #: carry the term for its own arithmetic to close on an FX row.
    #:
    #: Signed because a rate moves both ways. That makes this the one term ``verdict.py``'s
    #: negativity guard could not accept, which is the right outcome for a second reason: the
    #: **matcher's** copy of this shape deliberately has no ``fx_paise`` at all. A field the
    #: matcher could populate is a field it could fit any residual into, closing the gap by
    #: construction -- the failure ``.plan/phase8.md`` decision 8 names. So the asymmetry
    #: between the two shapes is load-bearing rather than an oversight, and it is safe only
    #: because an FX row is never ``RESOLVED``: the residual is non-zero, so ``tier1`` abstains
    #: and the term is never compared. Measured rather than assumed.
    #:
    #: Like ``reserve_paise``, no input file declares it -- but unlike the reserve, it is not
    #: even *bounded* by anything the inputs carry, since it hides inside a gross the matcher
    #: reads as authoritative.
    fx_paise: int = 0

    def __post_init__(self) -> None:
        for amount in self.as_dict().values():
            # ``paise`` rejects floats and bools and says nothing about sign, which is exactly
            # what this needs: ``fx_paise`` is legitimately negative when the rate moved down.
            # The six other terms are non-negative by construction rather than by assertion
            # here (``verdict.Decomposition`` is where that guard lives, on the shape that has
            # no signed term to exempt).
            paise(amount)

    @property
    def expected_credit_paise(self) -> int:
        return (
            self.gross_paise
            + self.fx_paise
            - self.fee_paise
            - self.gst_paise
            - self.tds_paise
            - self.refunds_paise
            - self.reserve_paise
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "gross_paise": self.gross_paise,
            "fee_paise": self.fee_paise,
            "gst_paise": self.gst_paise,
            "tds_paise": self.tds_paise,
            "refunds_paise": self.refunds_paise,
            "reserve_paise": self.reserve_paise,
            "fx_paise": self.fx_paise,
        }

    def as_truth(self) -> dict[str, int]:
        return {**self.as_dict(), "expected_credit_paise": self.expected_credit_paise}


@dataclass(frozen=True, slots=True)
class Credit:
    """A bank credit.

    Carries its full provenance in memory. ``csv_row()`` emits four fields: a
    date, an amount and a junk narration string. Real bank statements are exactly
    this impoverished, and the moment a gateway identifier lands here the task
    being scored has evaporated.
    """

    credit_id: str
    value_date: date
    amount_paise: int
    narration: str
    # --- provenance: truth.json only, never a CSV column ---
    settlement_ids: list[str]
    payment_ids: list[str]
    decomposition: Decomposition
    refunds_netted: list[str] = field(default_factory=list)
    reserve_held_paise: int = 0
    resolvable: bool = True
    reason: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        paise(self.amount_paise)
        paise(self.reserve_held_paise)
        assert self.amount_paise > 0, f"{self.credit_id}: credit must be positive"
        assert self.settlement_ids, f"{self.credit_id}: a gateway credit needs settlements"
        assert self.narration, f"{self.credit_id}: narration must not be empty"
        if self.resolvable:
            assert self.reason is None, f"{self.credit_id}: resolvable row carries a reason"

    @staticmethod
    def csv_header() -> tuple[str, ...]:
        return BANK_HEADER

    def csv_row(self) -> tuple[str, ...]:
        """Four fields. Do not extend this method."""
        return (
            self.credit_id,
            self.value_date.isoformat(),
            str(self.amount_paise),
            self.narration,
        )

    def as_truth(self) -> dict[str, object]:
        """The answer-key view. ``reason``/``note`` are present and explicitly
        ``null`` in clean mode so Phase 8 needs no schema migration and Phase 2's
        reader never branches on key presence."""
        return {
            "credit_id": self.credit_id,
            "settlement_ids": list(self.settlement_ids),
            "payment_ids": list(self.payment_ids),
            "refunds_netted": list(self.refunds_netted),
            "reserve_held_paise": self.reserve_held_paise,
            "decomposition": self.decomposition.as_truth(),
            "resolvable": self.resolvable,
            "reason": self.reason,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class NoiseRow:
    """A bank row that is **not gateway money at all** (``--noise-rows``, Phase 7).

    Deliberately **not** a ``Credit`` with empty provenance, and the reason is that two
    existing checks would have to be weakened to allow that:

      * ``Credit.__post_init__`` asserts ``settlement_ids`` is non-empty -- "a gateway credit
        needs settlements". A noise row has none, by definition. Relaxing that assertion would
        retire the check that catches a genuine credit losing its provenance.
      * I8b reads ``c.payment_ids[0]`` on every credit in ``story.credits`` to test that
        within-day row position carries no information. A member with no payments would either
        crash it or have to be filtered out of it -- and a check with a filter for the rows
        that cannot satisfy it is a check that has quietly stopped covering them.

    So a noise row is its own type, ``story.credits`` stays gateway-only, and every invariant
    written against it keeps its exact meaning. What the two share is the **CSV shape**: four
    fields, indistinguishable on disk from a real credit, which is the entire point.

    ``stratum`` is answer-key data (it names *why* this row is hard) and reaches
    ``run_manifest.json`` only -- never a CSV column, and not ``truth.json``'s scored section
    either. ``truth.json`` publishes the row **ids** under ``orphans.non_gateway_credit_ids``,
    which is what the scorer keys on.
    """

    row_id: str
    value_date: date
    amount_paise: int
    narration: str
    #: Which of ``config.NOISE_STRATA_SPLIT``'s three shapes this row was drawn as.
    stratum: str

    def __post_init__(self) -> None:
        paise(self.amount_paise)
        assert self.amount_paise > 0, f"{self.row_id}: a bank credit must be positive"
        assert self.narration, f"{self.row_id}: narration must not be empty"

    @staticmethod
    def csv_header() -> tuple[str, ...]:
        return BANK_HEADER

    def csv_row(self) -> tuple[str, ...]:
        """Four fields, the same shape a ``Credit`` emits. Do not extend this method."""
        return (
            self.row_id,
            self.value_date.isoformat(),
            str(self.amount_paise),
            self.narration,
        )


@dataclass(frozen=True, slots=True)
class Story:
    """One complete generated month, before it is split into impoverished views.

    Held in memory so ``invariants.check_story`` can validate the whole thing
    *before* anything touches disk.
    """

    payments: list[Payment]
    settlements: list[Settlement]
    credits: list[Credit]
    refunds: list[Refund] = field(default_factory=list)
    unsettled_payment_ids: list[str] = field(default_factory=list)
    settlements_without_credit: list[str] = field(default_factory=list)

    #: Bank rows that are not gateway money (``--noise-rows``). Held as **rows**, not as a list
    #: of ids: ``non_gateway_credit_ids`` below is derived from them, so the answer key's id
    #: list cannot drift from the rows it names. ``credits`` stays gateway-only -- see
    #: ``NoiseRow`` for the two invariants that depend on that separation.
    noise_rows: list[NoiseRow] = field(default_factory=list)

    #: Settlements whose rows ``emit`` omits from ``settlement_items.csv``
    #: (``--settlement-report-late``). **The membership itself is not removed** -- every
    #: ``Settlement`` here still lists its payments and truth still publishes them. What is
    #: withheld is the *declaration a judge's files carry*, which is what forces the matcher
    #: to search for a payment set instead of reading it off the settlement report. Keeping
    #: the two apart is why every in-memory invariant still runs unchanged under the flag.
    membership_withheld: list[str] = field(default_factory=list)

    @property
    def non_gateway_credit_ids(self) -> list[str]:
        """The ids of the noise rows, in bank-file order.

        **Derived, not stored.** It was a field until Phase 7 step 4 and became a property the
        moment real rows existed: a stored copy is a second place for the same fact, and the
        failure it invites is the answer key naming a row the statement does not carry (or
        missing one it does). The scorer keys ``noise_recall`` on exactly this list, so a drift
        here would mis-grade silently rather than crash.
        """
        return [r.row_id for r in self.noise_rows]

    def counts(self) -> dict[str, int]:
        return {
            "payments": len(self.payments),
            "settlements": len(self.settlements),
            "credits": len(self.credits),
            "refunds": len(self.refunds),
            "noise_rows": len(self.non_gateway_credit_ids),
        }

    def total_gross_paise(self) -> int:
        return sum(p.gross_paise for p in self.payments)

    def total_net_paise(self) -> int:
        return sum(s.net_paise for s in self.settlements)

    def total_credited_paise(self) -> int:
        return sum(c.amount_paise for c in self.credits)


if __name__ == "__main__":
    from datetime import timedelta
    from .config import IST

    when = datetime(2026, 8, 10, 11, 4, 22, tzinfo=IST)
    p = Payment("pay_0001", "ord_0001", when, 1_000_000, "card")
    assert p.business_date == date(2026, 8, 10)
    assert p.csv_row()[2] == "2026-08-10T05:34:22Z", p.csv_row()[2]
    assert len(p.csv_row()) == len(PAYMENTS_HEADER) == 7

    s = Settlement("setl_0001", date(2026, 8, 10), ["pay_0001"], 1_000_000, utr="XXXX4471")
    assert s.item_rows() == [("setl_0001", "pay_0001")]
    assert len(s.csv_row()) == len(SETTLEMENTS_HEADER) == 7

    d = Decomposition(gross_paise=1_000_000)
    assert d.expected_credit_paise == 1_000_000
    assert Decomposition(1_700_000, 34_000, 6_120, 0, 300_000).expected_credit_paise == 1_359_880

    c = Credit(
        "C0001", date(2026, 8, 10), 1_000_000, "NEFT-RAZORPAYSOFT-XXXX4471",
        settlement_ids=["setl_0001"], payment_ids=["pay_0001"], decomposition=d,
    )
    # The impoverishment check: four columns, and no linkage among them.
    assert c.csv_row() == ("C0001", "2026-08-10", "1000000", "NEFT-RAZORPAYSOFT-XXXX4471")
    assert len(c.csv_row()) == len(BANK_HEADER) == 4
    assert not any("setl_" in cell or "pay_" in cell for cell in c.csv_row())
    # truth.json keeps reason/note present-but-null.
    t = c.as_truth()
    assert t["reason"] is None and t["note"] is None and "note" in t
    assert t["decomposition"]["expected_credit_paise"] == 1_000_000

    st = Story([p], [s], [c])
    assert st.counts() == {"payments": 1, "settlements": 1, "credits": 1,
                           "refunds": 0, "noise_rows": 0}
    assert st.total_gross_paise() == st.total_net_paise() == st.total_credited_paise()

    # Guards must actually fire.
    try:
        iso_utc(datetime(2026, 8, 10, 11, 0))
    except ValueError:
        pass
    else:
        raise AssertionError("iso_utc accepted a naive datetime")
    try:
        Payment("pay_0002", "ord_0002", when, 1_000_00, "card", status="captured")._replace  # type: ignore[attr-defined]
    except AttributeError:
        pass  # frozen dataclass, no _replace -- fine, just confirming immutability shape
    try:
        Settlement("setl_0002", date(2026, 8, 10), [], 100)
    except AssertionError:
        pass
    else:
        raise AssertionError("Settlement accepted an empty payment list")
    try:
        Payment("pay_0003", "ord_0003", when, 1.0, "card")  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("Payment accepted a float amount")
    # UTC-date equals IST-date across the whole clamped hour range (trap 3).
    for hour in range(9, 22):
        t2 = datetime(2026, 8, 10, hour, 30, tzinfo=IST)
        assert t2.astimezone(timezone.utc).date() == t2.date() == date(2026, 8, 10), hour
    # ... and demonstrably fails outside it, which is why the clamp exists.
    late = datetime(2026, 8, 10, 2, 0, tzinfo=IST)
    assert late.astimezone(timezone.utc).date() == date(2026, 8, 9)
    assert (late - timedelta(hours=3)).date() == date(2026, 8, 9)
    print("model.py self-check ok")
