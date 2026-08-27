"""Tier 2: the subset search, for a settlement that did not declare its members.

Phase 5 step 6. Tier 1 answers "which settlement is this bank credit?" by joining on date
and amount. When that settlement is a *batch* whose ``settlement_items.csv`` rows were never
published, the harder question is left over: **which payments made it up?**

This module answers only that, and its interface is deliberately narrow (decision 10)::

    resolve(members, target, cap) -> ExactlyOne | TwoOrMore | PoolTooLarge | None

It takes a pool of candidate members, an integer target, and a cap. **It never receives a
``Settlement``.** Phase 6 changes what is *deducted* from a settlement, not how a set is
found; if the search reached into a ``Settlement`` for a field, Phase 6 would have to modify
the search itself and every number measured in Phase 5 would stop being reproducible. The
same separation let Phase 4's fee model arrive without re-keying the index.

**Count subsets; never return the first.** A subset-sum search that returns its first hit
looks identical to a correct one on data where the answer happens to be unique, and silently
commits to a coin-flip everywhere else. So this search counts, stops at two, and reports
which of four cases it is in. ``tier1.py`` banked this discipline in Phase 3 for this phase;
here it is cashed:

  * ``ExactlyOne``   -- one subset sums to the target. A Tier 2 match.
  * ``TwoOrMore``    -- at least two do. ``AMBIGUOUS_MULTI_SUBSET``: the data cannot say.
  * ``None``         -- none do. ``NO_CANDIDATE``, and **not** a cue to widen the pool.
  * ``PoolTooLarge`` -- the pool exceeded the cap. Refused before any work is done.

**Why the empty case must not widen the search.** A pool that yields no subset is either
missing a payment (the money is not all here) or carrying a deduction this matcher does not
model. Widening the date window until something adds up converts both into a match, which is
the failure mode Phase 4 measured for the nearest-date tie-break: coverage that barely moves
while wrong matches appear. The window is fixed by the posting lag (#15b).

**Enumeration, not dynamic programming.** A DP over the target is the textbook subset-sum
answer and it is the wrong shape here for two reasons. It is pseudo-polynomial in the
*target*, and these targets are paise -- a ten-thousand-rupee credit is a million-cell table
for a pool of twenty. More importantly it answers "is there a subset?" while the question
here is "is there exactly one?", and counting distinct subsets through a DP table means
reconstructing them anyway. So: depth-first over amount-sorted members, with two prunes that
make the bound real.

  * **Ascending sort with an early break.** Members are sorted by amount, so once a partial
    sum exceeds the target, every later member at that level exceeds it too and the whole
    remaining fan-out is cut rather than tested.
  * **A depth limit** (``MAX_SUBSET_SIZE``), which is what makes the worst case
    ``C(pool, k)`` rather than ``2 ** pool``.

**The depth limit is the one constant here that can cost correctness, and it must never be
lowered.** Measured at limit 3 against data whose true batches reach 4 (seed 1, n=1000,
``setl_0072``): the true four-member subset is invisible at depth 3, while a three-member
decoy sums to the same target. Being the *only* solution the search can see, its count is
exactly one and the search commits -- a confident wrong answer where the honest one was an
abstention. At limit 4 both are visible, the count is two, and it abstains. So a limit below
the true maximum batch size does not merely lose coverage; it manufactures wrong matches out
of honest abstentions. The trade-off is asymmetric: raising the limit costs only coverage and
time (measured at limit 5: 148 of 188 resolved against 170, ambiguous 40 against 18, wrong
matches still 0), while lowering it risks the one number this project refuses to trade.
``ASSUMPTIONS.md`` carries the limit as a stated assumption for exactly that reason, and the
self-check below reproduces the wrong match rather than describing it.

**The cap is a work bound, and refusing is always safe.** Unlike the depth limit, a pool cap
cannot produce a wrong answer: an over-cap pool is refused before the search starts, which is
an abstention. Measured across caps 32 to 128 at n up to 2000, wrong matches were 0 at every
one. What the cap buys is that the answer to "what happens at 10,000 records?" is *bounded
work and a stated refusal* rather than a throughput claim that holds on the seeds someone
happened to run. At the declared cap the worst case is C(64, 4) = 635,376 subsets; measured
pool maxima are 20 at n=200 and 63 at n=1000, so the cap costs nothing there, while at n=2000
it refuses the over-cap rows and keeps the pass at ~0.35s instead of ~20s.

**Nothing here reads ``settlement_items.csv``.** That is the file whose absence created the
problem, and where it *is* present Tier 1 has already used it; a search that peeked would be
scoring its own hint. ``tier1.py``'s self-check pins this by running the search with the items
map emptied and asserting the same set comes back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# The largest pool this search will enumerate. Above it, an abstention -- see the module
# docstring on why a refusal is the honest answer rather than a longer search. The bound it
# promises is C(MAX_POOL, MAX_SUBSET_SIZE) = 635,376 subsets.
MAX_POOL = 64

# The largest subset this search will consider. **Never lower this without re-reading the
# docstring:** a limit below the true maximum batch size converts abstentions into wrong
# matches, and the self-check below reproduces that.
MAX_SUBSET_SIZE = 4


@dataclass(frozen=True, slots=True)
class Member:
    """One candidate payment, reduced to the two things the search needs.

    Deliberately not a ``Payment``: the search must not be able to reach a method, a date or
    a status, because any of those would be a second signal smuggled into what is declared to
    be an amount-only decision.
    """

    payment_id: str
    net_paise: int


@dataclass(frozen=True, slots=True)
class ExactlyOne:
    """One subset summed to the target. The only outcome that becomes a match."""

    payment_ids: frozenset[str]
    subsets_examined: int


@dataclass(frozen=True, slots=True)
class TwoOrMore:
    """At least two subsets summed to the target, so the data cannot say which.

    Carries the two it found -- not to choose between them, but so the verdict note can name
    both. An ambiguity a human can see is one they can resolve from a source this matcher
    does not have.
    """

    first: frozenset[str]
    second: frozenset[str]
    subsets_examined: int


@dataclass(frozen=True, slots=True)
class PoolTooLarge:
    """The pool exceeded the cap; refused before any enumeration.

    Carries both numbers so the verdict note can name the bound rather than reporting that
    the search gave up.
    """

    pool_size: int
    cap: int


Result = ExactlyOne | TwoOrMore | PoolTooLarge | None


def resolve(
    members: Sequence[Member],
    target: int,
    cap: int = MAX_POOL,
    max_subset_size: int = MAX_SUBSET_SIZE,
) -> Result:
    """Find the unique subset of ``members`` summing to ``target``, or refuse to guess.

    ``members`` is the pool; ``target`` is an integer amount in paise. Both bounds are
    arguments so that the self-check can probe the refusals and a caller can *tighten* them
    -- not so a caller can widen them until a run looks better.
    """
    if target < 0:
        raise ValueError(f"target must be a non-negative paise amount, got {target}")
    if cap < 0:
        raise ValueError(f"cap must be non-negative, got {cap}")
    if max_subset_size < 1:
        raise ValueError(f"max_subset_size must be at least 1, got {max_subset_size}")

    ids = [m.payment_id for m in members]
    if len(set(ids)) != len(ids):
        raise ValueError("pool contains a duplicate payment_id")

    # A non-positive member would make the ascending break unsound to reason about and would
    # multiply subsets that differ only by a zero-amount member. Neither is a real payment,
    # so this is the caller's filter to apply, not something to absorb silently.
    for member in members:
        if member.net_paise <= 0:
            raise ValueError(
                f"{member.payment_id} has a non-positive net of {member.net_paise} paise; "
                f"the caller must exclude it from the pool"
            )

    if len(members) > cap:
        return PoolTooLarge(pool_size=len(members), cap=cap)

    # Ascending by amount, then by id so equal amounts have a fixed order and two runs over
    # the same pool enumerate in the same sequence. Nothing here iterates a set.
    pool = sorted((m.net_paise, m.payment_id) for m in members)

    found: list[frozenset[str]] = []
    examined = 0

    def walk(start: int, depth: int, acc: int, chosen: tuple[str, ...]) -> None:
        nonlocal examined
        for i in range(start, len(pool)):
            amount, payment_id = pool[i]
            examined += 1
            total = acc + amount
            if total > target:
                # Ascending: every later member at this level overshoots too.
                break
            if total == target:
                found.append(frozenset(chosen + (payment_id,)))
                if len(found) >= 2:
                    return
                # A longer subset extending this one cannot also sum to the target, since
                # every remaining amount is positive.
                continue
            if depth + 1 < max_subset_size:
                walk(i + 1, depth + 1, total, chosen + (payment_id,))
                if len(found) >= 2:
                    return

    walk(0, 0, 0, ())

    if len(found) >= 2:
        return TwoOrMore(first=found[0], second=found[1], subsets_examined=examined)
    if len(found) == 1:
        return ExactlyOne(payment_ids=found[0], subsets_examined=examined)
    return None


if __name__ == "__main__":
    import math
    import random

    def pool_of(*amounts: int) -> list[Member]:
        return [Member(f"pay_{i:04d}", a) for i, a in enumerate(amounts)]

    # -- the three ordinary outcomes ------------------------------------------------------
    unique = resolve(pool_of(100, 250, 375, 900), 350)
    assert isinstance(unique, ExactlyOne), unique
    assert unique.payment_ids == frozenset({"pay_0000", "pay_0001"}), unique

    single = resolve(pool_of(100, 250, 375), 375)
    assert isinstance(single, ExactlyOne) and single.payment_ids == frozenset({"pay_0002"})

    # 100 + 250 and 350 both reach 350, so the data cannot say which.
    ambiguous = resolve(pool_of(100, 250, 350), 350)
    assert isinstance(ambiguous, TwoOrMore), ambiguous
    assert ambiguous.first != ambiguous.second
    assert sum(len(s) for s in (ambiguous.first, ambiguous.second)) == 3

    assert resolve(pool_of(100, 250, 375), 999) is None, "no subset sums to 999"
    assert resolve((), 500) is None, "an empty pool resolves nothing"
    assert resolve((), 0) is None, "a zero target is not an excuse to match nothing"

    # -- the cap, which is a refusal and never a wrong answer -----------------------------
    over = resolve(pool_of(*range(1, 12)), 3, cap=10)
    assert isinstance(over, PoolTooLarge), over
    assert (over.pool_size, over.cap) == (11, 10), over
    # ... and the same pool one under the cap does resolve, so the cap is what refused it.
    assert isinstance(resolve(pool_of(*range(1, 11)), 3, cap=10), TwoOrMore)
    assert MAX_POOL == 64 and MAX_SUBSET_SIZE == 4
    assert math.comb(MAX_POOL, MAX_SUBSET_SIZE) == 635_376, "the declared bound moved"

    # -- the depth limit, and the wrong match a lower one manufactures --------------------
    # This reproduces setl_0072 (seed 1, n=1000) in miniature: a true four-member batch, and
    # a three-member decoy summing to the same target.
    #
    # The amounts are chosen so that those two are the *only* subsets that can reach the
    # target, by arithmetic rather than by luck -- a fixture where some third combination
    # also happens to hit would make depth 3 abstain and the probe would prove nothing. Every
    # true member is 1 mod 10 and every decoy is 8 mod 10, so a subset of `a` true and `b`
    # decoy members sums to (a + 8b) mod 10, and the target is 4 mod 10. That leaves a = 4,
    # b = 0 and a = 0, b = 3; the b = 1 and b = 2 cases need 6 and 8 true members and there
    # are only 4. Verified by exhaustive enumeration over all sizes when the fixture was
    # built.
    TRUE_TOTAL = 297_004
    truth_set = pool_of(45_001, 61_001, 88_001, 103_001)
    decoy = [Member("dec_0", 71_008), Member("dec_1", 94_008), Member("dec_2", 131_988)]
    assert sum(m.net_paise for m in truth_set) == TRUE_TOTAL
    assert sum(m.net_paise for m in decoy) == TRUE_TOTAL, "the decoy must hit the same target"
    mixed = truth_set + decoy
    want = frozenset(m.payment_id for m in truth_set)

    truncated = resolve(mixed, TRUE_TOTAL, max_subset_size=3)
    assert isinstance(truncated, ExactlyOne), (
        f"the depth-3 probe must find exactly one subset for the wrong match to happen at "
        f"all, got {truncated}"
    )
    assert truncated.payment_ids == frozenset({"dec_0", "dec_1", "dec_2"}), truncated
    assert truncated.payment_ids != want, (
        "the depth-3 probe is meant to reproduce a WRONG match: the true four-member subset "
        "is invisible at depth 3, so the decoy is the only solution the search can see and "
        "it commits to it. This is why MAX_SUBSET_SIZE must never be lowered"
    )
    honest = resolve(mixed, TRUE_TOTAL, max_subset_size=4)
    assert isinstance(honest, TwoOrMore), (
        f"at depth 4 both subsets are visible, so the only honest answer is an abstention, "
        f"got {honest}"
    )
    assert {honest.first, honest.second} == {want, frozenset({"dec_0", "dec_1", "dec_2"})}
    # With the decoy gone, depth 4 finds the true set and commits. The node count is pinned
    # because it is the prune working: 15 of the 2**4 - 1 = 15 non-empty subsets, walked
    # without the ascending break ever firing on a pool this small.
    assert resolve(truth_set, TRUE_TOTAL, max_subset_size=4) == ExactlyOne(
        payment_ids=want, subsets_examined=15
    )
    # A subset longer than the limit is simply not found -- coverage lost, nothing invented.
    assert resolve(pool_of(1, 2, 3, 4, 5), 15, max_subset_size=4) is None
    assert isinstance(resolve(pool_of(1, 2, 3, 4, 5), 15, max_subset_size=5), ExactlyOne)

    # -- determinism, and independence from input order -----------------------------------
    shuffled = list(mixed)
    random.Random("tier2-selfcheck").shuffle(shuffled)
    assert resolve(shuffled, TRUE_TOTAL, max_subset_size=3) == truncated, (
        "the result moved when the pool was shuffled -- the sort is not doing its job"
    )
    assert resolve(mixed, TRUE_TOTAL, max_subset_size=3) == truncated

    # Equal amounts are a real case: two payments of the same value are indistinguishable,
    # so the only honest answer is TwoOrMore rather than whichever the sort put first.
    twins = resolve([Member("a", 500), Member("b", 500), Member("c", 125)], 500)
    assert isinstance(twins, TwoOrMore), twins
    assert {tuple(twins.first), tuple(twins.second)} == {("a",), ("b",)}, twins

    # -- the work bound the cap promises --------------------------------------------------
    # A full-cap pool of distinct amounts against an unreachable target is the worst case:
    # no subset ever hits, so no ascending break fires and nothing short-circuits on a
    # second solution. That this returns at all is the bound being real rather than stated.
    # It cannot report its node count -- decision 10 fixes the empty case as bare ``None``
    # -- so the assertion is that the walk terminates and invents nothing.
    assert resolve(pool_of(*(1_000 + i for i in range(MAX_POOL))), 10**9) is None

    # -- the argument refusals -----------------------------------------------------------
    def must_raise(what: str, fn) -> None:
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"resolve() accepted {what}")

    must_raise("a negative target", lambda: resolve(pool_of(1, 2), -1))
    must_raise("a negative cap", lambda: resolve(pool_of(1, 2), 3, cap=-1))
    must_raise("a zero max_subset_size", lambda: resolve(pool_of(1, 2), 3, max_subset_size=0))
    must_raise(
        "a duplicate payment_id",
        lambda: resolve([Member("dup", 100), Member("dup", 200)], 300),
    )
    must_raise("a zero-net member", lambda: resolve([Member("z", 0), Member("y", 5)], 5))
    must_raise(
        "a negative-net member", lambda: resolve([Member("n", -5), Member("y", 5)], 5)
    )

    print(
        f"tier2.py self-check ok  (cap {MAX_POOL}, subset limit {MAX_SUBSET_SIZE}, "
        f"worst case {math.comb(MAX_POOL, MAX_SUBSET_SIZE):,} subsets)"
    )
