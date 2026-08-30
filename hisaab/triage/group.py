"""Groups the queue by cause, and prices each group.

**Groups come from the codes that occur, never from the vocabulary.** Thirteen codes are
declared and priced; six appear in an eleven-flag run, and three are producerless in three
different senses (`CREDIT_MISSING` has no construction site, `SETTLEMENT_MISSING`'s branch is
unreachable through the loader, `ROUNDING_DRIFT` lost its flag). A queue with a heading for each
declared code would show a person seven empty sections and one real one, and the empty sections
would look like results -- "nothing wrong here" -- rather than like categories that cannot occur.

**Two kinds of work, and they are charged differently because they are different work.**

  * An **exception** is a row the matcher could not resolve. It is grouped by its reason code
    and charged at that code's rate (``effort.MINUTES_PER_EXCEPTION``), because what a person
    does about an FX gap is not what they do about a duplicate amount.
  * A **dismissal** is a row the matcher set aside as not gateway money at all. Every dismissal
    is the same work -- glance, agree, move on -- so they form **one** group charged per row at
    ``effort.DISMISSAL_MINUTES``, whatever code each carries. Phase 9 charges these for the
    first time: they were priced from Phase 2 and never billed, because the scorer accrued
    minutes only for exceptions.

**What triage cannot see, and why that is right.** `.plan/phase9.md` §1(7) distinguishes two
non-gateway populations: the ~82 rows the matcher raised as exceptions on money that was not
gateway money (charged, at their own codes' rates), and the rows it correctly dismissed (charged
here at 3 min). Only the first is a mistake. **Triage cannot tell them apart**, because
``non_gateway_credit_ids`` exists only in ``truth.json`` and the queue reads ``matches.json``
plus ``data/``.

That is not a gap to work around. An operator cannot tell either -- a wrongly-raised noise row
looks exactly like a real unresolved credit, which is *why* it is in their queue costing them
time. The distinction is real on the **scoring** side, where the answer key is available, and
``report.roi`` keeps it: those rows sit in the by-hand easy bucket while remaining in the tool's
total, which is the harsher comparison and the one a judge can reconstruct. Importing that
distinction here would mean importing the answer key, and the queue would stop being something
an operator could run.

Resolution hints are Phase 9 step 5 and live in ``hint.py``; money is step 4 and comes from the
bank statement. This module counts and prices.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..common.effort import DISMISSAL_MINUTES, MINUTES_PER_EXCEPTION
from ..common.reasons import Reason
from ..common.verdict import Outcome
from .read import Ruling, TriageError


class Kind(str, Enum):
    """Whether a group is work the matcher *could not* do or work it *declined* to do."""

    EXCEPTION = "exception"
    DISMISSAL = "dismissal"

    def __str__(self) -> str:
        return self.value


def minutes_for(reason: Reason | None) -> int:
    """Minutes for one exception, by reason code -- triage's accessor over the shared table.

    Deliberately a second accessor over **one** table rather than a second table. The numbers
    live in ``hisaab/common/effort.py``; this raises ``TriageError`` where the scorer's
    ``metrics.minutes_for`` raises ``MetricsError``, so each side can report a bad price table
    as its own clean failure instead of leaking the other's exception type through.

    **No default, at either site.** Phase 9 step 1 deleted two ``.get`` fallbacks that
    disagreed by ten minutes -- an unpriced code cost 10 in the reported total and displayed 0
    against its own row -- so an unpriced code now stops the queue rather than pricing it at
    somebody's guess. A defaulted estimate in a *ranked* queue is worse than a wrong one: it
    sorts to the bottom and is never looked at again.
    """
    if reason is None:
        raise TriageError(
            "cannot price an exception with no reason code -- read.load_rulings refuses "
            "one, so this is a triage bug rather than bad input"
        )
    try:
        return MINUTES_PER_EXCEPTION[reason]
    except KeyError:
        raise TriageError(
            f"{reason} has no effort estimate in hisaab/common/effort.py. Add one, and the "
            f"matching entry in ASSUMPTIONS.md row 34 that gate 7 compares against it -- a "
            f"code with no price must not be shown as free work."
        ) from None


@dataclass(frozen=True, slots=True)
class Group:
    """One cause, the rows it holds, and what clearing them costs."""

    kind: Kind
    #: The reason code for an exception group. ``None`` for the dismissal group, which holds
    #: every dismissal regardless of the code each row carries -- the contract allows an
    #: ``IGNORED`` row to carry none at all.
    reason: Reason | None
    #: In the order they appeared in ``matches.json``, which is the bank statement's order.
    #: Step 4 re-orders by value; until then a group reads in statement order.
    credit_ids: tuple[str, ...]
    minutes_per_row: int

    @property
    def count(self) -> int:
        return len(self.credit_ids)

    @property
    def total_minutes(self) -> int:
        return self.count * self.minutes_per_row

    @property
    def label(self) -> str:
        """What a person reads. The code itself for an exception; dismissals say what they are."""
        if self.kind is Kind.DISMISSAL:
            return "DISMISSED (not gateway money)"
        return str(self.reason)


def group(rulings: tuple[Ruling, ...]) -> tuple[Group, ...]:
    """Group rulings by cause, heaviest first.

    Resolved rows are absent, because they are not work: an operator never looks at them. So
    the groups partition the *queue*, not the file -- ``sum(g.count) + resolved == len(rulings)``
    -- and that is asserted below rather than assumed, since a row silently missing from every
    group is a row that quietly stops being somebody's job.

    Ordered by total minutes descending, then by label, so the ordering is total and does not
    depend on dict iteration. **This is not the final order**: step 4 ranks by money at risk,
    which is what a finance team actually clears by. Effort-descending is the honest fallback
    until the bank join exists, and the amendment's "degrade to a sort key" plan if it does not.
    """
    by_reason: dict[Reason, list[str]] = {}
    dismissed: list[str] = []

    for r in rulings:
        if r.outcome is Outcome.RESOLVED:
            continue
        if r.is_dismissal:
            dismissed.append(r.credit_id)
        elif r.is_exception:
            # ``load_rulings`` has already refused an exception with no code, so this
            # subscript is safe; ``minutes_for`` refuses ``None`` again anyway rather than
            # trusting a guarantee made in another module.
            assert r.reason is not None, f"{r.credit_id}: exception with no reason reached group()"
            by_reason.setdefault(r.reason, []).append(r.credit_id)
        else:  # pragma: no cover -- Outcome has three members and two are handled above
            raise TriageError(f"{r.credit_id}: unhandled outcome {r.outcome}")

    groups = [
        Group(
            kind=Kind.EXCEPTION,
            reason=reason,
            credit_ids=tuple(ids),
            minutes_per_row=minutes_for(reason),
        )
        for reason, ids in by_reason.items()
    ]
    if dismissed:
        groups.append(
            Group(
                kind=Kind.DISMISSAL,
                reason=None,
                credit_ids=tuple(dismissed),
                minutes_per_row=DISMISSAL_MINUTES,
            )
        )

    queued = sum(1 for r in rulings if r.outcome is not Outcome.RESOLVED)
    grouped = sum(g.count for g in groups)
    if grouped != queued:
        raise TriageError(
            f"internal: {grouped} rows in groups but {queued} rows need work -- a row that is "
            f"in no group is a row that quietly stops being somebody's job"
        )

    return tuple(sorted(groups, key=lambda g: (-g.total_minutes, g.label)))


def total_minutes(groups: tuple[Group, ...]) -> int:
    """What the whole queue costs. Summed from the groups, so it cannot disagree with them."""
    return sum(g.total_minutes for g in groups)


if __name__ == "__main__":
    from ..common.verdict import Verdict

    def _r(cid: str, outcome: Outcome, reason: Reason | None = None) -> Ruling:
        return Ruling(credit_id=cid, outcome=outcome, reason=reason)

    E, I, R = Outcome.EXCEPTION, Outcome.IGNORED, Outcome.RESOLVED

    # --- the empty queue, first: a month where everything resolved -----------------------
    # Step 7's first control, and it must not raise or invent a group.
    assert group(()) == ()
    assert group((_r("C0001", R), _r("C0002", R))) == ()
    assert total_minutes(()) == 0

    # --- the shape of a real queue -------------------------------------------------------
    rulings = (
        _r("C0001", R),
        _r("C0002", E, Reason.NO_CANDIDATE),          # 10
        _r("C0003", I, Reason.NON_GATEWAY_CREDIT),    # 3
        _r("C0004", E, Reason.FX_RATE_GAP),           # 20
        _r("C0005", E, Reason.NO_CANDIDATE),          # 10
        _r("C0006", I, None),                         # 3, and legal: IGNORED needs no code
        _r("C0007", E, Reason.AMBIGUOUS_MULTI_SUBSET),  # 15
        _r("C0008", R),
    )
    gs = group(rulings)

    # Resolved rows are in no group at all.
    assert all("C0001" not in g.credit_ids and "C0008" not in g.credit_ids for g in gs)
    # Every other row is in exactly one.
    placed = [cid for g in gs for cid in g.credit_ids]
    assert sorted(placed) == ["C0002", "C0003", "C0004", "C0005", "C0006", "C0007"]
    assert len(placed) == len(set(placed)), "a row landed in two groups"

    # Four groups: three codes that occurred, plus one for both dismissals.
    assert len(gs) == 4, [g.label for g in gs]
    labels = [g.label for g in gs]
    assert "DISMISSED (not gateway money)" in labels
    # ...and nothing for the ten codes that did not occur.
    assert "CREDIT_MISSING" not in labels and "ROUNDING_DRIFT" not in labels

    # Heaviest first: NO_CANDIDATE 2x10=20 ties FX_RATE_GAP 1x20=20, so the label breaks it.
    assert [(g.label, g.total_minutes) for g in gs] == [
        ("FX_RATE_GAP", 20),
        ("NO_CANDIDATE", 20),
        ("AMBIGUOUS_MULTI_SUBSET", 15),
        ("DISMISSED (not gateway money)", 6),
    ], [(g.label, g.total_minutes) for g in gs]

    # Both dismissals are one group and both are charged, including the one with no code.
    dismissal = next(g for g in gs if g.kind is Kind.DISMISSAL)
    assert dismissal.credit_ids == ("C0003", "C0006"), dismissal.credit_ids
    assert dismissal.reason is None and dismissal.minutes_per_row == 3
    assert dismissal.total_minutes == 6, "a dismissal with no reason code must still be charged"

    # Members keep file order, which is the bank statement's order.
    no_cand = next(g for g in gs if g.reason is Reason.NO_CANDIDATE)
    assert no_cand.credit_ids == ("C0002", "C0005"), no_cand.credit_ids

    # The total is the groups' total, and it is the number the ROI comparison charges.
    assert total_minutes(gs) == 20 + 20 + 15 + 6 == 61

    # --- prices agree with the shared table, not with a copy of it ----------------------
    # If this module ever grew its own numbers, this is the assertion that would fail.
    for g in gs:
        if g.kind is Kind.EXCEPTION:
            assert g.minutes_per_row == MINUTES_PER_EXCEPTION[g.reason]  # type: ignore[index]
        else:
            assert g.minutes_per_row == DISMISSAL_MINUTES
    assert minutes_for(Reason.FX_RATE_GAP) == 20
    assert all(minutes_for(r) == MINUTES_PER_EXCEPTION[r] for r in Reason)

    # --- the refusals, each fired ------------------------------------------------------
    try:
        minutes_for(None)
    except TriageError as e:
        assert "no reason code" in str(e), e
    else:
        raise AssertionError("priced an exception with no reason code")

    victim = Reason.NO_CANDIDATE
    price = MINUTES_PER_EXCEPTION.pop(victim)
    try:
        assert victim not in MINUTES_PER_EXCEPTION, "the mutation did not take"
        try:
            minutes_for(victim)
        except TriageError as e:
            assert "no effort estimate" in str(e) and "ASSUMPTIONS.md" in str(e), e
        else:
            raise AssertionError(f"{victim} priced with no entry in the table")
        # And the whole path, not just the accessor: grouping a row whose code has no price
        # must stop rather than show the group as free work.
        try:
            group((_r("C0009", E, victim),))
        except TriageError as e:
            assert "no effort estimate" in str(e), e
        else:
            raise AssertionError("grouped an unpriced code")
    finally:
        MINUTES_PER_EXCEPTION[victim] = price
    assert set(MINUTES_PER_EXCEPTION) == set(Reason), "the table was left mutated"

    # --- the partition assertion fires, which needs a Ruling the reader would refuse ----
    # Built directly rather than through ``load_rulings``: an EXCEPTION with no code cannot
    # come off disk, so the only way to test the guard is to construct one.
    try:
        group((Ruling(credit_id="C0010", outcome=E, reason=None),))
    except AssertionError as e:
        assert "exception with no reason" in str(e), e
    else:
        raise AssertionError("grouped an exception carrying no reason code")

    # The contract still refuses the same thing on write, which is why the above is unreachable
    # from a real file -- asserted here so this stays true rather than assumed.
    try:
        Verdict("C0011", Outcome.EXCEPTION)
    except ValueError as e:
        assert "reason" in str(e)
    else:
        raise AssertionError("the verdict contract accepted an EXCEPTION with no reason")

    print("triage/group.py self-check ok")
