"""Candidate generation: the amount band, the date window, and one predicate for each.

Given a bank credit, produce the settlements that could plausibly be it. Two
independent conditions, deliberately kept as two functions so that Phase 4 can widen
either one without touching the other:

  * **the amount band** -- ``[credit - max_adjustment, credit + max_adjustment]``.
    Exact in Phase 3 (``max_adjustment=0``), because clean mode has zero fees and a
    tolerance would be a parameter fitted to data nobody has generated yet.
  * **the date window** -- ``abs(business_days_between(settled_on, value_date)) <= W``,
    with ``W = 0`` in Phase 3.

**One window predicate, in one place.** ``bizdays.business_days_between`` counts the
half-open interval ``[start, end)`` and returns 0 for equal dates, so ``W = 0`` means
"no business days separate them". Never write a ``timedelta`` comparison next to this:
mixing calendar days and business days in two places is how Phase 4 acquires a one-day
bug that clean mode cannot see.

There is a real edge in that predicate worth stating rather than discovering. At
``W = 0`` the rule is *not* strictly "same calendar date" -- a Saturday settlement and
the following Monday's credit are also zero business days apart, because neither
Saturday nor Sunday is counted. That is the same semantics
``bizdays.add_business_days(sat, 0) == Monday`` already commits to, and it is right for
the domain: money that settles over a weekend moves on the next business day. Clean
mode never exercises it, since every ``settled_on`` is a business day and equals its
credit's ``value_date`` -- so it is asserted in this module's self-check instead, where
it is visible.

**Why a sorted amount array rather than the ``(settled_on, net_paise)`` dict
``.plan/phase3.md`` step 3 specifies.** The plan's reason for that index was O(1)
lookup, and this keeps it -- ``bisect`` is O(log n) plus the bucket. What it also does
is serve the *band* and any window through one code path. An exact-tuple index serves
neither: a band cannot be enumerated key-by-key, and a window wider than 0 forces an
enumeration of candidate dates that recomputes the business-day count at every step --
which goes quadratic in exactly the wide-window test correction (a) requires. Both
conditions are still enforced, so the effective key remains ``(date, amount)``; only
the lookup structure differs, and Phase 4's band arrives as an argument rather than a
rewrite.

Candidates come back as a **list sorted by ``settlement_id``**, never a set. Set
iteration order is a real source of run-to-run drift, and byte-identical
``matches.json`` across two runs is a Phase 3 acceptance item.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from ..common.bizdays import BusinessCalendar
from .load import Credit, Settlement

#: Phase 3's date window, in business days. T+0 is measured on the committed data:
#: ``value_date == settled_on`` on every credit, across seeds 1/2/3/42 at n=60/200/1000.
DEFAULT_WINDOW_DAYS = 0

#: Phase 3's amount tolerance, in paise. Exact, because clean mode has zero fees --
#: the wedge between gross and net is Phase 4's ``--fees``, and a tolerance declared
#: now would be a number fitted to data that does not exist yet.
DEFAULT_MAX_ADJUSTMENT_PAISE = 0


def amount_band(amount_paise: int, max_adjustment_paise: int = DEFAULT_MAX_ADJUSTMENT_PAISE
                ) -> tuple[int, int]:
    """The inclusive ``[lo, hi]`` band of settlement nets that could produce ``amount_paise``.

    Exact at ``max_adjustment_paise=0``: ``(a, a)``. This is the *interface* Phase 4
    widens, which is why Phase 3 has it at all -- a tolerance should arrive as an
    argument, not as a rewrite of the lookup.
    """
    if max_adjustment_paise < 0:
        raise ValueError(f"max_adjustment_paise must be >= 0, got {max_adjustment_paise}")
    return amount_paise - max_adjustment_paise, amount_paise + max_adjustment_paise


def date_distance(cal: BusinessCalendar, settled_on: object, value_date: object) -> int:
    """Signed business-day distance from ``settled_on`` to ``value_date``.

    Positive when the credit lands after the settlement, which is the normal direction
    (T+n). The single place this arithmetic happens; ``within_window`` and the
    tie-break both read it, so they cannot disagree.
    """
    return cal.business_days_between(settled_on, value_date)  # type: ignore[arg-type]


def inside_window(distance_days: int, window_days: int = DEFAULT_WINDOW_DAYS) -> bool:
    """Does a business-day ``distance_days`` fall inside ``window_days``?

    **The one place the window comparison is written.** Both ``within_window`` (which
    computes the distance for you) and ``SettlementIndex.candidates_for`` (which already
    has it) route through here, so the two cannot disagree about what "inside" means.

    That is not hypothetical tidiness. The first version of this module inlined
    ``abs(distance) > window_days`` in ``candidates_for`` and kept the validation in
    ``within_window``, which the lookup path never called -- so a negative window
    silently excluded *every* candidate and reported 0% coverage instead of raising.
    A wrong answer that looks like a real result is the exact failure this separation
    exists to prevent, and it is why the parameter is validated here, on the path, and
    not in a sibling.
    """
    if window_days < 0:
        raise ValueError(
            f"window_days must be >= 0, got {window_days} -- a negative window excludes "
            f"every candidate and would report 0% coverage rather than an error"
        )
    return abs(distance_days) <= window_days


def within_window(cal: BusinessCalendar, settled_on: object, value_date: object,
                  window_days: int = DEFAULT_WINDOW_DAYS) -> bool:
    """Is ``settled_on`` within ``window_days`` business days of ``value_date``?

    The one window predicate in the matcher. See the module docstring for what
    ``window_days=0`` does and does not mean.
    """
    return inside_window(date_distance(cal, settled_on, value_date), window_days)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One settlement that could be a given credit, with the evidence for why.

    ``date_distance_days`` and ``amount_delta_paise`` are carried because they are what
    the tie-break reads and what a verdict's ``note`` reports. Both are zero for every
    match in clean mode, which is precisely why they must be *computed* -- a hard-coded
    zero would still look correct here and be wrong the moment Phase 4 turns on fees.
    """

    settlement: Settlement
    date_distance_days: int
    amount_delta_paise: int

    @property
    def settlement_id(self) -> str:
        return self.settlement.settlement_id

    @property
    def is_exact(self) -> bool:
        """Same business day, same amount to the paisa."""
        return self.date_distance_days == 0 and self.amount_delta_paise == 0


class SettlementIndex:
    """Settlements indexed by ``net_paise`` for band lookup, then filtered by date.

    Built once per run. ``candidates_for`` is O(log n + k) where k is the number of
    settlements inside the amount band -- so n=200 to n=5000 needs no change, which is
    the Phase 12 scale answer arriving for free.
    """

    def __init__(self, settlements: tuple[Settlement, ...] | list[Settlement],
                 calendar: BusinessCalendar | None = None) -> None:
        self.calendar = calendar or BusinessCalendar()
        # Sorted by (net_paise, settlement_id): the amount orders the array for bisect,
        # and the id breaks ties so construction is deterministic rather than dependent
        # on input order.
        self._sorted: list[Settlement] = sorted(
            settlements, key=lambda s: (s.net_paise, s.settlement_id)
        )
        self._amounts: list[int] = [s.net_paise for s in self._sorted]

    def __len__(self) -> int:
        return len(self._sorted)

    def amount_collisions(self) -> int:
        """How many settlements share a ``net_paise`` with an earlier one.

        Diagnostic, not used on the match path. Reported by the CLI because it is the
        measurement behind decision 1: a bare net amount is unique at n=60 and collides
        from n=200 up (1-2 at n=200, 42-64 at n=1000), so the *pair* is the key and the
        date is doing real work even when the window is 0.
        """
        return sum(1 for a, b in zip(self._amounts, self._amounts[1:]) if a == b)

    def candidates_for(
        self,
        credit: Credit,
        window_days: int = DEFAULT_WINDOW_DAYS,
        max_adjustment_paise: int = DEFAULT_MAX_ADJUSTMENT_PAISE,
    ) -> list[Candidate]:
        """Every settlement inside both the amount band and the date window.

        Returns a list sorted by ``settlement_id``. Never a set, never a generator, and
        never truncated -- ``tier1`` must be able to *count* candidates, because
        "a candidate exists" and "exactly one candidate exists" are different facts and
        conflating them is how a matcher starts guessing.
        """
        # Both parameters are validated before any lookup: amount_band rejects a
        # negative tolerance, inside_window a negative window. Neither may be
        # re-implemented inline here -- see inside_window's docstring for the bug that
        # rule comes from.
        lo, hi = amount_band(credit.amount_paise, max_adjustment_paise)
        inside_window(0, window_days)
        left = bisect_left(self._amounts, lo)
        right = bisect_right(self._amounts, hi)

        found: list[Candidate] = []
        for settlement in self._sorted[left:right]:
            distance = date_distance(self.calendar, settlement.settled_on, credit.value_date)
            if not inside_window(distance, window_days):
                continue
            found.append(
                Candidate(
                    settlement=settlement,
                    date_distance_days=distance,
                    amount_delta_paise=credit.amount_paise - settlement.net_paise,
                )
            )
        found.sort(key=lambda c: c.settlement_id)
        return found


def narrow_by_date_distance(candidates: list[Candidate]) -> list[Candidate]:
    """Keep only the candidates closest in business days to the credit.

    The tie-break. **It cannot fire in any Phase 3 end-to-end run**: at ``W = 0`` every
    candidate is already at distance 0, so this is the identity function on real data.
    It is written and unit-tested here anyway, at an artificially wide window, because
    Phase 4 widens the window and would otherwise be the first thing to execute it --
    and untested code that only runs in Phase 4 is a Phase 4 bug with a Phase 3
    postmark.

    A still-tied result is returned **whole, not resolved**. Two settlements equally
    close with the same amount are genuinely indistinguishable from the inputs, and the
    honest verdict is ``AMBIGUOUS_DUPLICATE_AMOUNT`` rather than whichever one sorted
    first. Narrowing to a single element here would silently convert an abstention into
    a coin flip.
    """
    if len(candidates) <= 1:
        return list(candidates)
    best = min(abs(c.date_distance_days) for c in candidates)
    return [c for c in candidates if abs(c.date_distance_days) == best]


if __name__ == "__main__":
    from datetime import date

    from .load import Credit as C
    from .load import Settlement as S

    cal = BusinessCalendar()
    mon, tue, fri = date(2026, 8, 10), date(2026, 8, 11), date(2026, 8, 14)
    sat, sun, next_mon = date(2026, 8, 15), date(2026, 8, 16), date(2026, 8, 17)

    def settlement(sid: str, when: date, net: int) -> S:
        return S(sid, when, net, 0, 0, 0, f"XXXX{sid[-4:]}")

    def credit(cid: str, when: date, amount: int) -> C:
        return C(cid, when, amount, "NEFT-RAZORPAYSOFT-XXXX0000")

    # --- the amount band ---------------------------------------------------
    assert amount_band(85358) == (85358, 85358), "Phase 3's band is exact"
    assert amount_band(85358, 100) == (85258, 85458)
    try:
        amount_band(100, -1)
    except ValueError:
        pass
    else:
        raise AssertionError("amount_band accepted a negative adjustment")

    # --- the window predicate, including the edge clean mode cannot show ---
    assert within_window(cal, mon, mon, 0), "same business day is inside a 0 window"
    assert not within_window(cal, mon, tue, 0)
    assert within_window(cal, mon, tue, 1)
    assert date_distance(cal, mon, tue) == 1, "positive: the credit lands after"
    assert date_distance(cal, tue, mon) == -1, "negative: the credit lands before"
    assert date_distance(cal, fri, next_mon) == 1, "a weekend is not a business day"
    # The documented edge: a weekend settlement is 0 business days from Monday.
    assert date_distance(cal, sat, next_mon) == 0
    assert within_window(cal, sat, next_mon, 0), (
        "at W=0 a Saturday settlement matches Monday's credit -- 'no business days "
        "separate them', not 'same calendar date'. Stated in the docstring because no "
        "clean-mode run can reveal it."
    )
    assert date_distance(cal, sun, next_mon) == 0
    # The comparison itself, independent of any calendar.
    assert inside_window(0, 0) and inside_window(-2, 2) and inside_window(2, 2)
    assert not inside_window(3, 2)
    for guard, label in (
        (lambda: within_window(cal, mon, mon, -1), "within_window"),
        (lambda: inside_window(0, -1), "inside_window"),
    ):
        try:
            guard()
        except ValueError:
            continue
        raise AssertionError(f"{label} accepted a negative window")

    # --- the index: exactly one candidate, and it is the right one ---------
    index = SettlementIndex([
        settlement("setl_0005", mon, 85358),
        settlement("setl_0009", mon, 197600),
        settlement("setl_0002", tue, 85358),   # same amount, different day
    ])
    assert len(index) == 3
    assert index.amount_collisions() == 1, "two settlements share 85358"

    got = index.candidates_for(credit("C0001", mon, 85358))
    assert len(got) == 1, [c.settlement_id for c in got]
    assert got[0].settlement_id == "setl_0005"
    assert got[0].is_exact and got[0].date_distance_days == 0
    assert got[0].amount_delta_paise == 0

    # The bare amount collides; the date is what separates them. This is decision 1
    # measured on a fixture rather than asserted in prose.
    got = index.candidates_for(credit("C0002", tue, 85358))
    assert [c.settlement_id for c in got] == ["setl_0002"]
    # Widen the window and the collision becomes a genuine ambiguity -- which is why
    # a wider window is actively harmful rather than merely unnecessary.
    both = index.candidates_for(credit("C0003", mon, 85358), window_days=1)
    assert [c.settlement_id for c in both] == ["setl_0002", "setl_0005"], (
        [c.settlement_id for c in both]
    )
    assert len(both) == 2, "counted, not silently first-wins"

    # No candidate at all, in each direction.
    assert index.candidates_for(credit("C0004", mon, 999_999)) == []
    assert index.candidates_for(credit("C0005", date(2026, 8, 20), 85358)) == []

    # Regression: a bad parameter must raise on the *lookup* path, not just in the
    # sibling predicate. An earlier version validated only in within_window, which
    # candidates_for never calls -- so a negative window quietly excluded every
    # candidate and the matcher reported 0% coverage instead of failing.
    for bad, label in (
        ({"window_days": -1}, "a negative window"),
        ({"max_adjustment_paise": -1}, "a negative tolerance"),
    ):
        try:
            index.candidates_for(credit("C0001", mon, 85358), **bad)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"candidates_for accepted {label}")
    # An empty index must not crash.
    assert SettlementIndex([]).candidates_for(credit("C0006", mon, 1)) == []
    assert SettlementIndex([]).amount_collisions() == 0

    # The band is honoured, and only inside the window.
    near = SettlementIndex([settlement("setl_0001", mon, 85_000)])
    assert near.candidates_for(credit("C0007", mon, 85_358)) == []
    loose = near.candidates_for(credit("C0007", mon, 85_358), max_adjustment_paise=400)
    assert len(loose) == 1 and loose[0].amount_delta_paise == 358
    assert not loose[0].is_exact, "a match inside a tolerance is not an exact match"

    # --- determinism: sorted by settlement_id regardless of input order ----
    forward = SettlementIndex([settlement(f"setl_{i:04d}", mon, 500) for i in (1, 2, 3)])
    backward = SettlementIndex([settlement(f"setl_{i:04d}", mon, 500) for i in (3, 2, 1)])
    key = credit("C0008", mon, 500)
    assert [c.settlement_id for c in forward.candidates_for(key)] == [
        "setl_0001", "setl_0002", "setl_0003"
    ]
    assert [c.settlement_id for c in forward.candidates_for(key)] == [
        c.settlement_id for c in backward.candidates_for(key)
    ], "candidate order must not depend on input order"

    # --- the tie-break, at a width no Phase 3 run reaches ------------------
    wide = SettlementIndex([
        settlement("setl_0001", mon, 500),        # 2 business days before Wednesday
        settlement("setl_0002", date(2026, 8, 12), 500),   # the same day
        settlement("setl_0003", date(2026, 8, 13), 500),   # 1 business day after
    ])
    wed = credit("C0009", date(2026, 8, 12), 500)
    pool = wide.candidates_for(wed, window_days=5)
    assert len(pool) == 3, "all three are inside a 5-day window"
    best = narrow_by_date_distance(pool)
    assert [c.settlement_id for c in best] == ["setl_0002"], [c.settlement_id for c in best]
    assert best[0].date_distance_days == 0

    # A genuine tie stays a tie -- it must not be resolved to whichever sorted first.
    tied = SettlementIndex([
        settlement("setl_0001", date(2026, 8, 11), 500),   # 1 day before
        settlement("setl_0002", date(2026, 8, 13), 500),   # 1 day after
    ])
    pool = tied.candidates_for(wed, window_days=5)
    still_tied = narrow_by_date_distance(pool)
    assert len(still_tied) == 2, "equidistant candidates must remain ambiguous"
    assert {c.date_distance_days for c in still_tied} == {1, -1}

    # Degenerate inputs to the tie-break.
    assert narrow_by_date_distance([]) == []
    assert len(narrow_by_date_distance(pool[:1])) == 1

    # At W=0 the tie-break is provably the identity -- correction (a), asserted.
    for cid, when, amount in (("C0001", mon, 85358), ("C0002", tue, 85358)):
        pool = index.candidates_for(credit(cid, when, amount))
        assert narrow_by_date_distance(pool) == pool, (
            "at a 0 window the tie-break must be a no-op; if it is not, the window "
            "predicate is admitting candidates it should have excluded"
        )

    print("blocking.py self-check ok  (window predicate, band, index, tie-break)")
