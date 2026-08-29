"""Candidate generation: the amount band, the date window, and one predicate for each.

Given a bank credit, produce the settlements that could plausibly be it. Two
independent conditions, deliberately kept as two functions so that Phase 4 can widen
either one without touching the other:

  * **the amount band** -- ``[credit - max_adjustment, credit + max_adjustment]``.
    Exact in Phase 3 (``max_adjustment=0``), because clean mode has zero fees and a
    tolerance would be a parameter fitted to data nobody has generated yet.
  * **the date window** -- ``0 <= business_days_between(settled_on, value_date) <= W``,
    with ``W = 0`` in Phase 3 and ``W = 1`` once Phase 4's posting lag exists. One-sided,
    because money moves forward: a credit cannot precede the settlement that paid it.
    See ``FORWARD_ONLY``, and note that at ``W = 0`` the one-sided and symmetric forms are
    identical, so no Phase 3 number moved when this changed.

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
# ``MASK_PREFIX`` only -- the parser itself is not called here. ``normalize`` imports
# nothing from this module (only ``common.ids``), so the direction cannot cycle.
from .normalize import MASK_PREFIX

#: Phase 3's date window, in business days. T+0 is measured on the committed data:
#: ``value_date == settled_on`` on every credit, across seeds 1/2/3/42 at n=60/200/1000.
DEFAULT_WINDOW_DAYS = 0

#: Phase 3's amount tolerance, in paise. Exact, because clean mode has zero fees --
#: the wedge between gross and net is Phase 4's ``--fees``, and a tolerance declared
#: now would be a number fitted to data that does not exist yet.
DEFAULT_MAX_ADJUSTMENT_PAISE = 0

#: **A bank credit cannot precede the settlement that paid it.** Money moves forward, so
#: the window is one-sided: ``0 <= business_days(settled_on -> value_date) <= W``, not
#: ``|distance| <= W``.
#:
#: This is a physical constraint, not a fitted preference, and it is a *declared
#: assumption* -- recorded in ASSUMPTIONS.md, because a real bank can in principle post a
#: credit before the gateway's settlement report is dated, and if that ever happens this
#: predicate is what would have to be relaxed.
#:
#: Measured on Phase 4's delayed data, seeds 1/2/3/42 x n=60/200/1000: every one of 5,040
#: true (settlement, credit) pairs sits at distance **+1** and not one is negative. A
#: symmetric window therefore admits only impossible candidates on the negative side --
#: and those impostors were the entire source of the unresolvable ties measured at n=1000.
#: At ``W = 0`` this changes nothing (``0 <= 0 <= 0``), so clean mode is unaffected.
FORWARD_ONLY = True


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
    """Is a settlement ``distance_days`` before its credit inside ``window_days``?

    ``0 <= distance_days <= window_days``. **Forward-only, not symmetric** -- a bank credit
    cannot precede the settlement that paid it, so a negative distance names a candidate
    that could not physically be the answer. See ``FORWARD_ONLY``. At ``window_days = 0``
    this is identical to the old symmetric form, so no Phase 3 number moves.

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
    return 0 <= distance_days <= window_days


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
        #: Every settlement's UTR tail, as the 4-digit string a narration would carry.
        #:
        #: **Not on the match path** -- ``candidates_for`` does not read it, and that separation
        #: is the same one ``_note_for_match`` documents: a tail that corroborates a join is
        #: evidence, the same tail used *as* the join is a shortcut that hides every missing
        #: capability. What reads this is Phase 7's ``IGNORED`` gate, which asks the opposite
        #: question -- not "which settlement is this?" but "is this gateway money at all?".
        #:
        #: Built here rather than per credit because the gate runs inside a per-credit branch:
        #: recomputing it there would make an O(n) scan into O(n^2), and this object is
        #: constructed once per run. Eager rather than lazy since it is one pass over a list
        #: already being sorted.
        self.utr_tails: frozenset[str] = frozenset(
            s.utr.removeprefix(MASK_PREFIX) for s in self._sorted
        )

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

    **RETIRED FROM THE MATCH PATH IN PHASE 4. Do not reintroduce it without reading this.**

    Phase 3 shipped this unit-tested at an artificially wide window and noted it could not
    fire on real data. Phase 4's posting lag made it fire for the first time, and it was
    not merely untested -- it was **wrong**, in the direction that costs the most:

      * Every true (settlement, credit) pair sits at distance **+1**, measured across
        5,040 pairs, seeds 1/2/3/42 x n=60/200/1000. The lag is constant, so the true
        candidate is *never* the closest one.
      * This function keeps the **minimum** distance. When a same-day settlement (d=0)
        happened to share an amount with the true one (d=+1), it therefore preferred the
        impostor -- every time. Measured at n=1000: 10 wrong matches at seed 1, 9 at
        seed 2, 5 at seed 3, 7 at seed 42.
      * Coverage stayed at ~99.5% throughout. **Only correctness moved.** A tie-break that
        guesses confidently is indistinguishable from one that works, unless you score it.

    The lesson generalises past this function: proximity in time is not evidence of
    identity when the lag is unknown and constant. The closest candidate is the *least*
    likely one at any constant non-zero lag.

    ``tier1`` now abstains on a multi-candidate pool instead
    (``AMBIGUOUS_DUPLICATE_AMOUNT``), which is what the inputs actually support: two
    settlements sharing an amount inside the window cannot be separated without knowing
    the lag, and the honest verdict is that a human decides. That trades ~0.5% coverage at
    n=1000 for zero wrong matches, which is the trade this whole submission argues for.

    A legitimate non-leaking successor exists and is deliberately **not** built here:
    infer the modal lag from the rows that resolved unambiguously, then prefer that
    distance. That reads the lag off the *inputs* rather than importing the generator's
    T+n, so it would not be a leak -- but it is a fitted parameter, and Phase 4 is exact
    arithmetic. It is recorded as a Phase 5 candidate, not smuggled in as a tie-break.

    Kept, not deleted, because the self-check below is the evidence for all of the above.
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
    assert inside_window(0, 0) and inside_window(2, 2) and inside_window(1, 2)
    assert not inside_window(3, 2)
    # Forward-only: a credit cannot precede the settlement that paid it, so a negative
    # distance is out at every width. This is the assertion that would have caught the
    # symmetric window admitting impossible candidates -- see FORWARD_ONLY for the
    # measurement (5,040 true pairs, every one at +1, none negative).
    assert not inside_window(-1, 5), "a settlement dated after its own credit is impossible"
    assert not inside_window(-2, 2)
    assert not within_window(cal, tue, mon, 5), "backward in time, however wide the window"
    # ...and at W=0 forward-only and symmetric agree exactly, which is why no Phase 3
    # number moved when this changed.
    for d in (-2, -1, 0, 1, 2):
        assert inside_window(d, 0) == (d == 0)
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
    #
    # The credit is Tuesday's, so both colliding settlements are *behind* it: setl_0005
    # (Monday) at +1 and setl_0002 (Tuesday) at 0. Phase 3 asked this with Monday's credit,
    # which under the forward-only window now yields one candidate rather than two -- the
    # Tuesday settlement would have had to pay a Monday credit. The property being tested
    # is unchanged; the fixture just no longer relies on an impossible candidate to show it.
    both = index.candidates_for(credit("C0003", tue, 85358), window_days=1)
    assert [c.settlement_id for c in both] == ["setl_0002", "setl_0005"], (
        [c.settlement_id for c in both]
    )
    assert len(both) == 2, "counted, not silently first-wins"
    assert sorted(c.date_distance_days for c in both) == [0, 1], "both behind the credit"
    # Monday's credit sees only Monday's settlement: the forward-only window drops the
    # Tuesday one instead of handing tier1 a coin flip between them.
    assert [c.settlement_id for c in index.candidates_for(
        credit("C0003", mon, 85358), window_days=1)] == ["setl_0005"]

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

    # --- the tie-break: retired from the match path, kept as evidence -------
    # ``narrow_by_date_distance`` is no longer called by ``tier1``. What follows is the
    # measurement that retired it, written as assertions so the reasoning cannot be lost
    # and so the function cannot be quietly reinstated as the "obvious fix" for the
    # ambiguity its removal creates.
    #
    # The geometry is Phase 4's real one: a constant +1 business-day posting lag, so the
    # TRUE settlement is always one day behind its credit and any same-day settlement
    # sharing the amount is an impostor.
    wed = credit("C0009", date(2026, 8, 12), 500)
    lagged = SettlementIndex([
        settlement("setl_0002", date(2026, 8, 12), 500),   # same day     -> d=0, impostor
        settlement("setl_0007", date(2026, 8, 11), 500),   # 1 day before -> d=+1, TRUE
    ])
    pool = lagged.candidates_for(wed, window_days=1)
    assert len(pool) == 2, "both sit behind the credit, so both are physically possible"
    assert sorted(c.date_distance_days for c in pool) == [0, 1]
    best = narrow_by_date_distance(pool)
    assert [c.settlement_id for c in best] == ["setl_0002"], (
        "the tie-break keeps the MINIMUM distance, which at a constant +1 lag is always "
        "the impostor -- the bug, asserted rather than described"
    )
    assert len(best) == 1, (
        "...and it narrows to one, so tier1 would have committed to the wrong settlement "
        "at full confidence. Coverage would still read ~100%; only correctness moved, "
        "which is why this survived Phase 3 and every coverage-only check in Phase 4."
    )
    # On real delayed data at n=1000 this shape cost 10 wrong matches at seed 1, 9 at
    # seed 2, 5 at seed 3 and 7 at seed 42.

    # The forward-only window makes the *symmetric* tie unreachable: a candidate one day
    # AFTER the credit is excluded outright rather than left tied with the one before it.
    tied = SettlementIndex([
        settlement("setl_0001", date(2026, 8, 11), 500),   # 1 day before -> +1
        settlement("setl_0002", date(2026, 8, 13), 500),   # 1 day after  -> -1, impossible
    ])
    survivors = tied.candidates_for(wed, window_days=5)
    assert [c.settlement_id for c in survivors] == ["setl_0001"], (
        "Phase 3 asserted these two stayed ambiguous at {+1, -1}. Under the forward-only "
        "window the -1 candidate is gone, so this is a clean single match -- the "
        "impossible half was the entire ambiguity."
    )

    # Degenerate inputs, unchanged: the function still behaves, it is simply not called.
    assert narrow_by_date_distance([]) == []
    assert len(narrow_by_date_distance(survivors)) == 1
    # A genuine tie is still returned WHOLE rather than resolved -- the property that
    # mattered while it was on the path. Asserted on a hand-built pool, because the
    # forward-only window can no longer produce an equidistant one.
    equidistant = [
        Candidate(settlement=S("setl_0011", mon, 500, 0, 0, 0, "XXXX0011"),
                  date_distance_days=1, amount_delta_paise=0),
        Candidate(settlement=S("setl_0012", tue, 500, 0, 0, 0, "XXXX0012"),
                  date_distance_days=1, amount_delta_paise=0),
    ]
    assert len(narrow_by_date_distance(equidistant)) == 2, (
        "equidistant candidates must remain ambiguous, never resolved to whichever "
        "sorted first"
    )

    # At W=0 it is provably the identity, which is why Phase 3 could not see any of this.
    for cid, when, amount in (("C0001", mon, 85358), ("C0002", tue, 85358)):
        pool = index.candidates_for(credit(cid, when, amount))
        assert narrow_by_date_distance(pool) == pool, (
            "at a 0 window the tie-break must be a no-op; if it is not, the window "
            "predicate is admitting candidates it should have excluded"
        )

    print(
        "blocking.py self-check ok  (forward-only window, band, index; tie-break "
        "refuted and retired)"
    )
