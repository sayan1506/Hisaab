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

**Nothing here reads the narration to make a decision.** The parsed tail is recorded in
the verdict's ``note`` as corroboration only. See ``normalize.py`` for why that
separation is load-bearing, and note the consequence for testing: blanking every
narration must leave every *decision field* untouched, while ``note`` legitimately
changes. ``tools/acceptance.py`` gate 9 compares the former and excludes the latter.
"""

from __future__ import annotations

from ..common.reasons import Reason
from ..common.verdict import Outcome, Verdict
from .blocking import Candidate, SettlementIndex, narrow_by_date_distance
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
) -> str:
    """Human-readable evidence for a resolved row.

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
    return (
        f"tier 1 exact: {settlement.settlement_id}, "
        f"{payment_count} payment(s), "
        f"date distance {candidate.date_distance_days}bd, "
        f"amount delta {candidate.amount_delta_paise}p; "
        f"corroboration only: {utr}"
    )


def resolve_credit(
    credit: Credit,
    index: SettlementIndex,
    dataset: Dataset,
    window_days: int = 0,
    max_adjustment_paise: int = 0,
) -> Verdict:
    """One credit in, exactly one ``Verdict`` out. Never returns ``None``, never skips.

    A missing verdict is not an option: ``verdict_io.reconcile`` refuses a file that
    drops a row, and discovering that in this loop is much cheaper than discovering it
    at review time.
    """
    narration = parse(credit.narration)
    candidates = index.candidates_for(
        credit, window_days=window_days, max_adjustment_paise=max_adjustment_paise
    )

    # The tie-break. A no-op at window_days=0 (every candidate is at distance 0), and
    # unit-tested at a wide window in blocking.py because no run here reaches it.
    if len(candidates) > 1:
        candidates = narrow_by_date_distance(candidates)

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
        ids = ", ".join(c.settlement_id for c in candidates)
        return Verdict(
            credit.credit_id,
            Outcome.EXCEPTION,
            reason=Reason.AMBIGUOUS_DUPLICATE_AMOUNT,
            note=(
                f"{len(candidates)} settlements share this date and amount ({ids}) -- "
                f"the inputs cannot separate them, so a human decides"
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

    gross = dataset.gross_by_payment()
    missing = [pid for pid in payment_ids if pid not in gross]
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

    residual = credit.amount_paise - sum(gross[pid] for pid in payment_ids)

    return Verdict(
        credit.credit_id,
        Outcome.RESOLVED,
        settlement_ids=(settlement_id,),
        payment_ids=tuple(payment_ids),
        tier=TIER,
        residual_paise=residual,
        note=_note_for_match(winner, narration, len(payment_ids)),
    )


if __name__ == "__main__":
    from datetime import date, datetime, timezone

    from .load import Credit as C
    from .load import Dataset as D
    from .load import Payment as P
    from .load import Settlement as S

    mon, tue = date(2026, 8, 10), date(2026, 8, 11)
    when = datetime(2026, 8, 10, 5, 34, 22, tzinfo=timezone.utc)

    def payment(pid: str, gross: int) -> P:
        return P(pid, f"ord_{pid[4:]}", when, gross, "card", "INR", "captured")

    def settlement(sid: str, on: date, net: int, tail: str = "0000") -> S:
        return S(sid, on, net, 0, 0, 0, f"XXXX{tail}")

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

    # --- the residual is computed, not hard-coded (acceptance item 6) ------
    # A credit 500 paise short of its payment's gross must report -500, not 0. This is
    # the whole reason the field exists before Phase 4 has anything to put in it.
    short = dataset(
        [payment("pay_0001", 85358)],
        [settlement("setl_0005", mon, 84858, "8104")],
        [credit("C0001", mon, 84858)],
        {"setl_0005": ("pay_0001",)},
    )
    v = verdict_of(short, "C0001")
    assert v.outcome is Outcome.RESOLVED
    assert v.residual_paise == -500, (
        f"expected -500, got {v.residual_paise} -- the residual must be computed from "
        f"the payment gross, not assumed zero"
    )
    # And in the other direction: an over-credit is a real Phase 4/6 finding.
    over = dataset(
        [payment("pay_0001", 85358)],
        [settlement("setl_0005", mon, 85858, "8104")],
        [credit("C0001", mon, 85858)],
        {"setl_0005": ("pay_0001",)},
    )
    assert verdict_of(over, "C0001").residual_paise == 500

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
