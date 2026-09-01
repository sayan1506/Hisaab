"""The identity, the confusion matrix, and the four rates.

The one module that joins matcher output against the answer key. Everything it knows
about being *right* comes from here; everything it knows about being *confident* comes
from the verdict file. Keeping those two words apart is the entire point of the phase:

    coverage    -- how often the matcher committed to an answer
    correctness -- how often the answer it committed to was right

A single accuracy number averages them and loses the distinction, so this module
returns a frozen dataclass with both and no combined field. There is nothing to
collapse.

Phase 4 adds a **third** axis on the same principle, rather than folding it into either:

    decomposition_agreement -- of the rows it linked correctly, how often it also
                               priced the money correctly, term by term

It is separate because correctness has meant set equality on ``payment_ids`` since
Phase 2 (decision 3), and quietly widening it to also require the arithmetic would change
what every number measured in Phases 2 and 3 meant. So a row can be a correct *linkage*
and a wrong *explanation*: ``tier1`` refuses to resolve unless its derived deduction
closes the gap exactly, which forces the **total** to agree while leaving the **split**
free -- a fee too high and a GST too low by the same amount closes the identical gap. This
rate is the only place that shows.

**Order matters.** The identity is asserted before any arithmetic runs. A matcher that
emits 58 verdicts for 60 bank rows is not scored on 58 -- ``verdict_io.reconcile``
refuses it. By the time a ``Metrics`` exists, "matched + exceptions = total, exactly"
is already true rather than hoped for.

**What this module deliberately does not expose.** Per-verdict detail is limited to the
cell a verdict landed in, its residual, and its reason. It never carries truth's
``payment_ids`` for a row the matcher got wrong. If the scorer printed the right answer
for every miss, Phase 3 would become an exercise in fitting the answer key, and the
match rate would be measuring the developer rather than the matcher. Tune against the
aggregate, not against the diff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..common.effort import DISMISSAL_MINUTES, MINUTES_PER_EXCEPTION
from ..common.reasons import Reason
from ..common.verdict import Decomposition, Outcome, Verdict, VerdictFile
from .truth_io import Truth, TruthCredit, TruthDecomposition

# Re-exported, not re-declared. The effort table moved to ``hisaab/common/effort.py`` in
# Phase 9 so that ``hisaab/triage`` -- which check 1 forbids from importing this package --
# reads the same numbers rather than a second copy of them. The names stay importable from
# here because gate 7 reads ``MINUTES_PER_EXCEPTION`` off this module to compare against
# ASSUMPTIONS.md row 34, and that check should not have to know where the table lives.
__all__ = [
    "DISMISSAL_MINUTES",
    "MINUTES_PER_EXCEPTION",
    "Cell",
    "Landing",
    "Metrics",
    "MetricsError",
    "expected_credit_ids",
    "minutes_for",
    "score",
]

# ``MINUTES_PER_EXCEPTION`` and ``DISMISSAL_MINUTES`` are imported above, not declared
# here. Phase 9 moved the table to ``hisaab/common/effort.py`` because ``hisaab/triage``
# cannot import this package (check_isolation check 1) and needed the same numbers; the
# exhaustiveness assertion moved with it.


def minutes_for(reason: Reason | None) -> int:
    """Minutes for one exception, by reason code. **The only way to price a row.**

    Phase 9 step 1, and it replaces two ``.get`` calls that disagreed. Until now this table
    was read at two sites with two different fallbacks: ``metrics.score`` defaulted an
    unpriced code to ``DEFAULT_MINUTES_PER_EXCEPTION = 10`` while ``report.exception_queue``
    defaulted the same code to ``0``. So an unpriced code cost 10 minutes in the reported
    total and displayed 0 against its own row, and the queue's visible minutes did not have
    to sum to the figure printed above them.

    Unreachable while ``effort.py``'s exhaustiveness assertion holds -- which is exactly why
    it was safe to leave wrong for seven phases, and why it stops being safe in step 6, where
    per-group estimates make "a code with no price" a live state for the first time. The
    fallbacks are deleted rather than reconciled: two defaults for one table is a silent
    disagreement wearing an assertion's protection, and picking either number would have kept
    the shape while making the value agree by luck.

    Raises rather than defaulting, for the reason ``reasons.py`` deleted
    ``CORRECT_ABSTENTION_CODES`` instead of wiring it up: a plausible fallback invites the
    caller to stop declaring prices, and a queue that quietly prices a new code at somebody's
    guess is worse than one that refuses to print until the price is written down.
    """
    if reason is None:
        raise MetricsError(
            "cannot price an exception with no reason code -- Verdict.__post_init__ "
            "refuses an EXCEPTION without one, so this is a scorer bug, not bad input"
        )
    try:
        return MINUTES_PER_EXCEPTION[reason]
    except KeyError:
        raise MetricsError(
            f"{reason} has no effort estimate in MINUTES_PER_EXCEPTION. Add one (and the "
            f"matching entry in ASSUMPTIONS.md row 34, which gate 7 compares against this "
            f"table) rather than letting it price at a default -- an unpriced code used to "
            f"cost 10 minutes in the total and show 0 in the queue."
        ) from None


class Cell(str, Enum):
    """Where one verdict landed. Every verdict lands in exactly one.

    Seven cells, not the five a confusion matrix would suggest, because two
    distinctions are worth keeping that a 2x2 would flatten:

      * ``WRONG_MATCH`` vs ``WRONG_MATCH_INVENTED`` -- asserting a wrong answer where
        one existed, versus asserting one where the inputs contain none. The second is
        the worst outcome in the system and averaging it away hides exactly the failure
        a finance team cares about.
      * ``WRONG_IGNORE`` vs ``WRONG_MATCH`` -- discarding a real credit as non-gateway
        income, versus mis-linking it. Both are wrong; only one puts a false match in
        the books, and they have different fixes.
    """

    CORRECT = "correct"
    WRONG_MATCH = "wrong_match"
    WRONG_MATCH_INVENTED = "wrong_match_invented"
    LUCKY_GUESS = "lucky_guess"
    MISSED = "missed"
    WRONG_IGNORE = "wrong_ignore"
    CORRECT_ABSTENTION = "correct_abstention"
    NOISE_CORRECTLY_IGNORED = "noise_correctly_ignored"
    NOISE_MISHANDLED = "noise_mishandled"

    def __str__(self) -> str:
        return self.value


#: Cells where the matcher committed to an answer -- the coverage numerator. An
#: abstention is not here, which is the whole reason abstaining is cheap.
COMMITTAL_CELLS: frozenset[Cell] = frozenset(
    {Cell.CORRECT, Cell.WRONG_MATCH, Cell.WRONG_MATCH_INVENTED, Cell.LUCKY_GUESS}
)

#: Cells that assert a false match into the books. ``LUCKY_GUESS`` is here on purpose --
#: see ``_classify``.
WRONG_MATCH_CELLS: frozenset[Cell] = frozenset(
    {Cell.WRONG_MATCH, Cell.WRONG_MATCH_INVENTED, Cell.LUCKY_GUESS}
)


#: The terms compared between the matcher's decomposition and truth's, in truth's order.
#: ``expected_credit_paise`` is included: it is derived on both sides, so a disagreement
#: there with all six terms agreeing would mean one of the two derivations is broken.
DECOMPOSITION_TERMS: tuple[str, ...] = (
    "gross_paise", "fee_paise", "gst_paise", "tds_paise", "refunds_paise",
    "reserve_paise", "expected_credit_paise",
)


def compare_decomposition(claimed: Decomposition, expected: TruthDecomposition) -> tuple[str, ...]:
    """Which terms disagree. Empty tuple means the arithmetic matches truth's exactly.

    **Term by term, never on the total**, and the reason is stronger than diligence. A
    resolved row's total deduction is already *forced* to agree: ``tier1`` refuses to
    resolve unless its derived deduction closes the gap between the members' gross and the
    credit to the paisa, and truth's deduction is the gap by construction. So whenever the
    gross agrees, the totals agree necessarily -- comparing them proves nothing at all.

    What is not forced is the **split**. A fee 307p too high with a GST 307p too low closes
    the identical gap, resolves the row, and reports the same total; only a per-term
    comparison can tell that the model priced the wrong thing. By Phase 6, with refunds,
    TDS and reserves as further candidates, that is the failure mode most likely to hide
    inside a coverage percentage.

    Returns names, deliberately, and never truth's values -- see the module docstring on
    what this module refuses to print. Naming ``fee_paise`` is triage: it says which rule
    to go and check. Printing truth's fee would let the rate table be *fitted* to the
    answer key instead of derived from a pricing page, and the rates are exactly what
    ASSUMPTIONS.md #5-#9 stake a claim on.
    """
    return tuple(
        term for term in DECOMPOSITION_TERMS
        if getattr(claimed, term) != getattr(expected, term)
    )


@dataclass(frozen=True, slots=True)
class Landing:
    """One verdict's cell, with only what is safe to print.

    No truth ``payment_ids``. See the module docstring -- this dataclass is the
    mechanism that makes "do not print the answer" structural rather than a habit.
    """

    credit_id: str
    cell: Cell
    outcome: Outcome
    reason: Reason | None
    residual_paise: int | None
    value_paise: int
    #: Which decomposition terms disagree with truth's. Genuinely three-state, so ``None``
    #: rather than an empty tuple for the third: ``None`` means the arithmetic was not
    #: scored on this row (an abstention, a noise row, or a wrong match, where the two
    #: decompositions describe different payment sets and any comparison would be
    #: meaningless), while ``()`` means it was scored and agreed. Collapsing those two
    #: would report an unchecked row as a passing one.
    decomposition_mismatch: tuple[str, ...] | None = None

    @property
    def is_wrong(self) -> bool:
        return self.cell in WRONG_MATCH_CELLS or self.cell in (
            Cell.WRONG_IGNORE, Cell.NOISE_MISHANDLED
        )

    @property
    def arithmetic_disagrees(self) -> bool:
        """True only when the arithmetic was scored *and* disagreed."""
        return bool(self.decomposition_mismatch)


@dataclass(frozen=True, slots=True)
class Metrics:
    """The scored result. Coverage and correctness are separate fields, deliberately.

    Every rate is a property returning ``float | None``; ``None`` means the denominator
    was zero and the honest rendering is ``n/a``. Clean mode has zero planted
    unresolvables and zero noise rows, so three of these are ``None`` on every run this
    phase -- which is why they are modelled rather than special-cased at print time.
    """

    seed: int
    month: str
    clean_mode: bool
    flags_enabled: tuple[str, ...]
    matcher: str
    wall_clock_seconds: float | None

    total_bank_rows: int
    gateway_credits: int
    non_gateway_credits: int
    planted_unresolvable: int

    #: Payments behind those bank rows. **Two counts, because ``--batching`` makes them two
    #: different numbers** -- 200 payments settle as ~120 bank rows at mean 1.60 per
    #: settlement -- and "record count" is the figure a judge checks against the track's
    #: 50-record floor. One number reported alone is ambiguous the moment batching is on, and
    #: which one it is changes whether the floor is cleared (.plan/phase5.md decision 3).
    #:
    #: **Derived from truth's credits, not read from its ``counts`` header.** The header is a
    #: number the generator states; this is one the answer key can be held to.
    total_payments: int

    cells: dict[Cell, int]
    landings: tuple[Landing, ...] = field(repr=False)

    exceptions: int = 0
    exception_value_paise: int = 0
    exception_minutes: int = 0
    ignores_total: int = 0

    #: Money already booked wrong -- the credit value behind every ``WRONG_MATCH_CELLS``
    #: landing. Distinct from ``exception_value_paise``: an exception is money *awaiting* a
    #: human, a wrong match is money a human was never told to look at because the matcher
    #: committed to an answer instead of raising one. A ₹2,00,000 wrong match and a ₹49 one
    #: both used to just increment ``wrong_matches`` by one -- this is the amendment's owed
    #: value-at-risk line, priced instead of merely counted.
    wrong_match_value_paise: int = 0

    #: Minutes charged for dismissals -- ``ignores_total x DISMISSAL_MINUTES``. **A separate
    #: counter rather than more minutes in ``exception_minutes``**, because an ``IGNORED`` row
    #: is not an exception: that field serialises as ``exceptions.estimated_minutes``, and
    #: folding a different population into it would make the JSON's own name wrong. The two
    #: are summed only where the sum is what is meant -- the by-hand comparison, where the
    #: question is what the whole queue costs an operator.
    dismissal_minutes: int = 0

    #: Rows whose decomposition was compared against truth's, and of those, how many
    #: disagreed on at least one term. Both are carried because the second is
    #: uninterpretable without the first -- ``0 mismatches`` reads as a clean bill of health
    #: whether 200 rows were checked or none were, which is the same reasoning that makes
    #: every rate here ``None`` rather than ``0`` on an empty denominator.
    decomposition_checked: int = 0
    decomposition_mismatches: int = 0

    # --- rates. None == 0/0, rendered as n/a, never as 0% -------------------

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float | None:
        """``numerator / denominator``, or ``None`` when the denominator is zero.

        The single division in this module. Clean mode makes the abstention denominator
        zero on every run, and ``0%`` there reads as total failure at something that was
        never attempted.
        """
        if denominator <= 0:
            return None
        return numerator / denominator

    @property
    def committed(self) -> int:
        return sum(self.cells[c] for c in COMMITTAL_CELLS)

    @property
    def wrong_matches(self) -> int:
        return sum(self.cells[c] for c in WRONG_MATCH_CELLS)

    @property
    def coverage(self) -> float | None:
        """How often the matcher committed, over gateway credits."""
        return self._rate(self.committed, self.gateway_credits)

    @property
    def correctness(self) -> float | None:
        """How often a commitment was right. ``n/a`` when it never committed."""
        return self._rate(self.cells[Cell.CORRECT], self.committed)

    @property
    def wrong_rate(self) -> float | None:
        """The number that matters. A wrong match silently corrupts books."""
        return self._rate(self.wrong_matches, self.gateway_credits)

    @property
    def abstention_rate(self) -> float | None:
        """Of the cases planted as unresolvable, how many did it correctly decline?

        ``n/a`` through Phase 7 -- clean mode plants none. Phase 8 makes it real.
        """
        return self._rate(self.cells[Cell.CORRECT_ABSTENTION], self.planted_unresolvable)

    @property
    def noise_recall(self) -> float | None:
        """Of the non-gateway rows, how many were correctly ignored?"""
        return self._rate(self.cells[Cell.NOISE_CORRECTLY_IGNORED], self.non_gateway_credits)

    @property
    def decomposition_agreement(self) -> float | None:
        """Of the rows whose arithmetic was checkable, how many priced it exactly right?

        A **third** axis, reported beside coverage and correctness rather than folded into
        either, and the reason is the same one that keeps those two apart in the first
        place. Correctness is set equality on ``payment_ids`` (decision 3) and has been
        since Phase 2; redefining it now to also require the arithmetic would silently
        change what every number measured in Phases 2 and 3 meant. So this counts on its
        own line, and gate 10 requires it to be 1.0.

        Which means a row *can* score CORRECT while pricing the money wrongly, and it is
        worth being explicit about the one way that happens rather than leaving it implicit:
        ``tier1`` only resolves when its derived deduction closes the gap exactly, so the
        total is forced to agree. The split is not. A fee too high and a GST too low by the
        same amount closes the identical gap. That row is a correct *linkage* and a wrong
        *explanation*, and this rate is the only place the difference shows.

        ``n/a`` when nothing was checkable -- a run where the matcher abstained on
        everything has not earned a 0% here, and has not earned a 100% either.
        """
        return self._rate(
            self.decomposition_checked - self.decomposition_mismatches,
            self.decomposition_checked,
        )

    @property
    def noise_precision(self) -> float | None:
        """Of the rows it called non-gateway, how many actually were?

        Reported apart from the headline (decision 2): counting correctly-ignored noise
        as coverage would inflate the rate with the easiest rows in the file.
        """
        return self._rate(self.cells[Cell.NOISE_CORRECTLY_IGNORED], self.ignores_total)

    def as_json(self) -> dict[str, object]:
        """Line 1 of stdout. Phase 11 parses this instead of scraping the text block.

        Everything non-deterministic is confined to ``timing``, exactly as
        ``emit.build_manifest`` does it, so two runs of the same matcher on the same
        seed differ only inside that one object. The metric block is quoted into a
        report subject to the reproducibility rule.
        """
        return {
            "schema_version": METRICS_SCHEMA_VERSION,
            "run": {
                "seed": self.seed,
                "month": self.month,
                "clean_mode": self.clean_mode,
                "flags": list(self.flags_enabled),
                "matcher": self.matcher,
            },
            "timing": {"wall_clock_seconds": self.wall_clock_seconds},
            "totals": {
                "bank_rows": self.total_bank_rows,
                # Both counts, always, even when they are equal: a reader must not have to
                # infer from one number whether batching was on.
                "payments": self.total_payments,
                "gateway_credits": self.gateway_credits,
                "non_gateway_credits": self.non_gateway_credits,
                "planted_unresolvable": self.planted_unresolvable,
            },
            "cells": {str(c): self.cells[c] for c in Cell},
            "rates": {
                "coverage": self.coverage,
                "correctness": self.correctness,
                "wrong_rate": self.wrong_rate,
                "abstention_rate": self.abstention_rate,
                "noise_precision": self.noise_precision,
                "noise_recall": self.noise_recall,
                "decomposition_agreement": self.decomposition_agreement,
            },
            "exceptions": {
                "count": self.exceptions,
                "value_paise": self.exception_value_paise,
                "estimated_minutes": self.exception_minutes,
            },
            # Dismissals are their own block and not more minutes under ``exceptions``: an
            # IGNORED row is not an exception, and the key above is named for the population
            # it counts. Phase 9 charges these for the first time.
            "dismissals": {
                "count": self.ignores_total,
                "estimated_minutes": self.dismissal_minutes,
            },
            "decomposition": {
                "checked": self.decomposition_checked,
                "mismatches": self.decomposition_mismatches,
            },
            # Its own block, not a key under ``exceptions``: a wrong match raises no
            # exception, so folding this into that block would make its name wrong for the
            # population it counts -- the same reasoning ``dismissals`` already got its own
            # block for.
            "risk": {
                "wrong_match_value_paise": self.wrong_match_value_paise,
            },
        }


#: Bumped on a breaking change to ``Metrics.as_json``. Phase 9 re-ranks this document
#: and Phase 11 renders it; a format that shifts silently costs both.
#:
#: v5 (Phase 12 step 1a): a new ``risk`` block, carrying ``wrong_match_value_paise`` -- the
#: amendment's owed value-at-risk line. Breaking for the same reason v4 was: a v4 reader
#: that defaults a missing ``risk`` key to ``{"wrong_match_value_paise": 0}`` cannot tell
#: "measured zero" from "never measured", and the whole point of this field is that a run
#: with a real wrong-match value must not read as indistinguishable from one that has none.
#:
#: v4 (Phase 9 step 1): a new ``dismissals`` block. Breaking by the same rule as v2 and v3
#: below -- a renderer reading ``dismissals.estimated_minutes`` would otherwise have to branch
#: on whether the key exists, and a v3 document is a run where dismissals were *priced and
#: never charged*, which is a different fact from a run where they cost nothing.
#:
#: What did **not** change is worth stating, because it is the trap this bump avoids:
#: ``exceptions.estimated_minutes`` still counts exceptions only. The new minutes went into a
#: new block rather than swelling a key named for a different population, so a v3 reader that
#: keeps reading the old key gets the number it always got instead of a larger one under the
#: same name.
#:
#: v3 (Phase 5 step 3): ``totals.payments``. Breaking by the same rule as v2 below, and the
#: reason is the phase itself: until ``--batching`` a bank row *was* a payment, so a v2
#: document's single count was unambiguous. Under batching it is not, and a renderer that
#: defaulted the missing key to ``bank_rows`` would silently restate 200 payments as 120 --
#: which is the number the track's 50-record floor is checked against.
#:
#: v2 (Phase 4 step 5): a new ``decomposition`` block and a seventh rate. Counted as
#: breaking even though it only *adds* keys, for the reason this codebase refuses
#: absent-versus-null everywhere else: a Phase 11 renderer that reads
#: ``rates.decomposition_agreement`` would have to branch on whether the key exists, and a
#: v1 document is a run where the arithmetic was never checked -- which is a different fact
#: from a run where it was checked and agreed. Refusing the old document says so; reading
#: it and defaulting the missing key would not.
METRICS_SCHEMA_VERSION = 5


class MetricsError(Exception):
    """The verdict file cannot be scored against this answer key."""


def expected_credit_ids(truth: Truth) -> tuple[str, ...]:
    """Every bank row that must carry a verdict, in truth's own order.

    The union of the credits in the answer key and the non-gateway IDs, because a noise
    row is a bank row the matcher sees and must rule on, whether or not Phase 7 chooses
    to give it a full entry under ``credits``. Taking the union means this function is
    correct either way rather than correct until Phase 7.
    """
    ids = [c.credit_id for c in truth.credits]
    known = set(ids)
    ids.extend(cid for cid in truth.non_gateway_credit_ids if cid not in known)
    return tuple(ids)


def _classify(verdict: Verdict, credit: TruthCredit, is_noise: bool) -> Cell:
    """The cell for one verdict. Exactly one, no partial credit anywhere.

    Correctness is **set equality** on ``payment_ids``. A match with one extra payment
    in it is wrong, not 80% right: partial credit would let a matcher that guesses
    broadly outscore one that abstains honestly, which is backwards for finance and
    backwards for this track.

    Non-gateway rows are scored on their own axis and never touch the headline.
    """
    if is_noise:
        return Cell.NOISE_CORRECTLY_IGNORED if verdict.outcome is Outcome.IGNORED \
            else Cell.NOISE_MISHANDLED

    if verdict.outcome is Outcome.RESOLVED:
        if verdict.payment_set != frozenset(credit.payment_ids):
            return Cell.WRONG_MATCH_INVENTED if credit.is_planted_unresolvable \
                else Cell.WRONG_MATCH
        # Sets are equal. On a resolvable row that is simply correct.
        if not credit.is_planted_unresolvable:
            return Cell.CORRECT
        # ...but on a planted unresolvable it is a *lucky guess*, and it is not
        # impossible: --dup-amounts plants two credits sharing a date and an amount, so
        # the inputs cannot separate them while truth still records which is which. A
        # matcher that commits there has even odds of being right.
        #
        # It counts as a wrong match, and the reasoning is the one the whole submission
        # rests on: the answer key states these rows are unresolvable *from the inputs*,
        # so a matcher resolving one either guessed or read something it should not
        # have. Crediting luck would reward guessing over abstaining, which is the
        # inversion this scorer exists to prevent. Tracked in its own cell because a
        # non-zero count is also the cheapest available leak detector.
        return Cell.LUCKY_GUESS

    if verdict.outcome is Outcome.EXCEPTION:
        return Cell.CORRECT_ABSTENTION if credit.is_planted_unresolvable else Cell.MISSED

    # IGNORED. Declining a planted unresolvable is still a correct abstention -- the row
    # does need a human -- but discarding a resolvable credit as non-gateway income
    # loses real money from the books.
    return Cell.CORRECT_ABSTENTION if credit.is_planted_unresolvable else Cell.WRONG_IGNORE


def score(run: VerdictFile, truth: Truth) -> Metrics:
    """Join ``run`` against ``truth`` and return the metric block.

    Assumes ``verdict_io.reconcile`` has already passed -- one verdict per bank row,
    right seed, right month. The identity is re-asserted here anyway, because this
    function is what everything downstream trusts and the check costs nothing.
    """
    by_id = {c.credit_id: c for c in truth.credits}
    noise_ids = set(truth.non_gateway_credit_ids)
    expected = expected_credit_ids(truth)

    # --- the identity, before any arithmetic --------------------------------
    counts = run.counts()
    stated = counts["RESOLVED"] + counts["EXCEPTION"] + counts["IGNORED"]
    if stated != len(expected):
        raise MetricsError(
            f"the identity does not hold: {counts['RESOLVED']} resolved + "
            f"{counts['EXCEPTION']} exceptions + {counts['IGNORED']} ignored = {stated}, "
            f"but this run has {len(expected)} bank rows. No record may be dropped."
        )
    if len(run.verdicts) != len(expected):
        raise MetricsError(
            f"{len(run.verdicts)} verdicts for {len(expected)} bank rows -- "
            f"run verdict_io.reconcile() before scoring to see which IDs are at fault"
        )

    cells: dict[Cell, int] = {c: 0 for c in Cell}
    landings: list[Landing] = []
    exception_value = 0
    exception_minutes = 0
    wrong_match_value = 0
    checked = 0
    mismatched = 0

    for verdict in run.verdicts:
        is_noise = verdict.credit_id in noise_ids
        credit = by_id.get(verdict.credit_id)
        if credit is None:
            if not is_noise:
                raise MetricsError(
                    f"{verdict.credit_id} is not in the answer key -- run "
                    f"verdict_io.reconcile() first"
                )
            # A Phase 7 noise row with no full truth entry: value unknown, and it is
            # not gateway income, so it contributes nothing to the headline anyway.
            value = 0
        else:
            value = credit.decomposition.expected_credit_paise

        cell = (
            _classify(verdict, credit, is_noise)
            if credit is not None
            else (Cell.NOISE_CORRECTLY_IGNORED if verdict.outcome is Outcome.IGNORED
                  else Cell.NOISE_MISHANDLED)
        )
        cells[cell] += 1

        # --- the arithmetic, against truth's own decomposition (step 5) ------
        # Scored only where the payment sets agree, and the gate is that condition rather
        # than the cell: two decompositions over *different* payment sets have different
        # grosses by construction, so every term would differ and the count would measure
        # the linkage failure a second time instead of measuring the arithmetic. That
        # deliberately includes a LUCKY_GUESS -- its set matches, so its arithmetic is
        # genuinely comparable even though the linkage is not credited.
        mismatch: tuple[str, ...] | None = None
        if (
            credit is not None
            and verdict.outcome is Outcome.RESOLVED
            and verdict.payment_set == frozenset(credit.payment_ids)
        ):
            if verdict.decomposition is None:
                # Unreachable through the dataclass, which requires it on RESOLVED. Raised
                # rather than skipped: silently not comparing would leave the agreement rate
                # at a clean 100% over a shrinking denominator, and a rate whose denominator
                # can quietly fall to zero is the failure mode this whole module is built to
                # avoid.
                raise MetricsError(
                    f"{verdict.credit_id}: RESOLVED with no decomposition -- the verdict "
                    f"contract forbids this, so the file bypassed verdict_io"
                )
            mismatch = compare_decomposition(verdict.decomposition, credit.decomposition)
            checked += 1
            if mismatch:
                mismatched += 1

        landings.append(
            Landing(
                credit_id=verdict.credit_id,
                cell=cell,
                outcome=verdict.outcome,
                reason=verdict.reason,
                residual_paise=verdict.residual_paise,
                value_paise=value,
                decomposition_mismatch=mismatch,
            )
        )

        if verdict.outcome is Outcome.EXCEPTION:
            exception_value += value
            exception_minutes += minutes_for(verdict.reason)
        if cell in WRONG_MATCH_CELLS:
            wrong_match_value += value

    gateway = [c for c in truth.credits if c.credit_id not in noise_ids]

    # The payment count, derived rather than quoted. Every payment either sits behind some
    # credit or is named as unsettled, so the union is the whole population -- and unlike
    # ``truth.counts["payments"]`` it is a number this answer key can be held to. Where the
    # header does state one, the two must agree: a truth file whose header disagrees with its
    # own credits is one where every rate below has an unclear denominator, and that is worth
    # refusing rather than averaging over.
    total_payments = len(
        {pid for c in truth.credits for pid in c.payment_ids}
        | set(truth.unsettled_payment_ids)
    )
    stated_payments = truth.counts.get("payments")
    if isinstance(stated_payments, int) and stated_payments != total_payments:
        raise MetricsError(
            f"truth states {stated_payments} payments but its credits and unsettled list "
            f"account for {total_payments}. The scorer will not guess which denominator "
            f"the coverage and arithmetic rates below are over."
        )

    metrics = Metrics(
        seed=truth.seed,
        month=truth.month,
        clean_mode=truth.clean_mode,
        flags_enabled=truth.flags_enabled,
        matcher=run.matcher,
        wall_clock_seconds=run.wall_clock_seconds,
        total_bank_rows=len(expected),
        gateway_credits=len(gateway),
        non_gateway_credits=len(noise_ids),
        planted_unresolvable=sum(1 for c in gateway if c.is_planted_unresolvable),
        total_payments=total_payments,
        cells=cells,
        landings=tuple(landings),
        exceptions=counts["EXCEPTION"],
        exception_value_paise=exception_value,
        exception_minutes=exception_minutes,
        wrong_match_value_paise=wrong_match_value,
        ignores_total=counts["IGNORED"],
        # Charged per row, by the act rather than the code -- see ``DISMISSAL_MINUTES``.
        dismissal_minutes=counts["IGNORED"] * DISMISSAL_MINUTES,
        decomposition_checked=checked,
        decomposition_mismatches=mismatched,
    )

    # Every verdict landed in exactly one cell, and the cells cover every row.
    if sum(cells.values()) != len(expected):
        raise MetricsError(
            f"internal: {sum(cells.values())} classified verdicts for "
            f"{len(expected)} bank rows"
        )
    return metrics


if __name__ == "__main__":
    from ..common.verdict import Verdict as V
    from .truth_io import TruthDecomposition

    #: The default synthetic row: gross equals credit, nothing withheld. The value is named
    #: rather than repeated because the matcher-side and truth-side helpers below have to
    #: agree on it or every row would look like an arithmetic mismatch.
    VALUE = 100_000

    def _credit(cid: str, pids: tuple[str, ...], *, resolvable: bool = True,
                value: int = VALUE, reason: Reason | None = None,
                dec: TruthDecomposition | None = None) -> TruthCredit:
        return TruthCredit(
            credit_id=cid,
            settlement_ids=(f"setl_{cid[1:]}",),
            payment_ids=pids,
            refunds_netted=(),
            reserve_held_paise=0,
            decomposition=dec or TruthDecomposition(value, 0, 0, 0, 0, 0, value),
            resolvable=resolvable,
            reason=None if reason is None else str(reason),
            note=None,
        )

    def _truth(credits: tuple[TruthCredit, ...], noise: tuple[str, ...] = ()) -> Truth:
        return Truth(
            schema_version=1, seed=42, month="2026-08", clean_mode=not noise,
            flags={}, counts={"credits": len(credits)}, credits=credits,
            unsettled_payment_ids=(), settlements_without_credit=(),
            non_gateway_credit_ids=noise,
        )

    def _resolved(cid: str, pids: tuple[str, ...], *, value: int = VALUE,
                  dec: Decomposition | None = None) -> V:
        return V(cid, Outcome.RESOLVED, (f"setl_{cid[1:]}",), pids, tier=1,
                 residual_paise=0, credit_amount_paise=value,
                 decomposition=dec or Decomposition(value))

    def _except(cid: str, reason: Reason = Reason.NO_CANDIDATE) -> V:
        return V(cid, Outcome.EXCEPTION, reason=reason)

    def _run(verdicts: tuple[V, ...], matcher: str = "selfcheck") -> VerdictFile:
        return VerdictFile(42, "2026-08", matcher, verdicts, wall_clock_seconds=0.01)

    three = tuple(_credit(f"C{i:04d}", (f"pay_{i:04d}",)) for i in (1, 2, 3))

    # --- the oracle shape: everything right --------------------------------
    m = score(_run(tuple(_resolved(c.credit_id, c.payment_ids) for c in three)), _truth(three))
    assert m.coverage == 1.0 and m.correctness == 1.0
    assert m.wrong_matches == 0 and m.cells[Cell.CORRECT] == 3
    assert m.abstention_rate is None, "clean mode plants none -- must be n/a, not 0%"
    assert m.noise_precision is None and m.noise_recall is None
    assert m.exceptions == 0 and m.exception_value_paise == 0 and m.exception_minutes == 0
    assert m.wrong_match_value_paise == 0, "nothing wrong, nothing at risk"
    # Step 5: the arithmetic was checked on all three, and agreed on all three.
    assert m.decomposition_checked == 3 and m.decomposition_mismatches == 0
    assert m.decomposition_agreement == 1.0
    # Checked-and-agreed is `()`, never None. The two states are different facts.
    assert all(l.decomposition_mismatch == () for l in m.landings)
    assert not any(l.arithmetic_disagrees for l in m.landings)

    # --- the split, which the total cannot catch (Phase 4 step 5) -----------
    # THE case this axis exists for. Truth withheld a 2,000p fee and 360p of GST on it. The
    # matcher claims the whole 2,360p was fee and none of it was GST. Both close to the same
    # expected credit, so:
    #   * the row resolves -- tier1's gate only requires the gap to close exactly;
    #   * it lands CORRECT -- correctness is set equality on payment_ids (decision 3);
    #   * its residual is 0 and its stated total agrees with truth's to the paisa.
    # Every number measured before this step says the row is perfect. It is not: the money
    # was priced against the wrong rule, and in a real ledger that is a misfiled tax
    # liability. Only the per-term comparison sees it.
    SPLIT_TRUTH = TruthDecomposition(VALUE, 2_000, 360, 0, 0, 0, VALUE - 2_360)
    swapped = (_credit("C0001", ("pay_0001",), dec=SPLIT_TRUTH),)
    m = score(
        _run((_resolved("C0001", ("pay_0001",), value=VALUE - 2_360,
                        dec=Decomposition(VALUE, fee_paise=2_360, gst_paise=0)),)),
        _truth(swapped),
    )
    assert m.cells[Cell.CORRECT] == 1, "the linkage is right, and stays right"
    assert m.correctness == 1.0, (
        "correctness must remain set equality on payment_ids -- widening it here would "
        "silently change what every number measured in Phases 2 and 3 meant"
    )
    assert m.decomposition_checked == 1 and m.decomposition_mismatches == 1
    assert m.decomposition_agreement == 0.0, "a wrong split is not a passing arithmetic"
    # It names the two terms that disagree, and only those -- the total agrees, which is
    # exactly why a check on the total would have passed this row.
    assert m.landings[0].decomposition_mismatch == ("fee_paise", "gst_paise"), (
        m.landings[0].decomposition_mismatch
    )
    assert m.landings[0].arithmetic_disagrees
    assert not m.landings[0].is_wrong, "a wrong explanation is not a wrong match"
    # ...and truth's numbers do not leak out with the names. A mismatch report is triage --
    # "go and check the GST rule" -- not a diff against the answer key, which is how the
    # rate table would get fitted to truth instead of derived from a pricing page.
    assert "2000" not in repr(m.landings[0]) and "360" not in repr(m.landings[0]), repr(
        m.landings[0]
    )
    # The document carries both the rate and its denominator: "0 mismatches" is
    # uninterpretable without knowing how many rows were looked at.
    assert m.as_json()["decomposition"] == {"checked": 1, "mismatches": 1}
    assert m.as_json()["rates"]["decomposition_agreement"] == 0.0

    # The same trap one level up: a gross a paisa high *and* a fee a paisa high. The two
    # errors cancel, so the expected credit agrees, the residual is 0, and the row resolves
    # -- yet the matcher priced a different quantity of money than actually moved. This is
    # why ``expected_credit_paise`` cannot be the comparison even though it is the term the
    # residual is built from: it is the one number all three cases above agree on.
    m = score(
        _run((_resolved("C0001", ("pay_0001",), value=VALUE - 2_360,
                        dec=Decomposition(VALUE + 1, fee_paise=2_001, gst_paise=360)),)),
        _truth(swapped),
    )
    assert m.landings[0].decomposition_mismatch == ("gross_paise", "fee_paise"), (
        m.landings[0].decomposition_mismatch
    )
    assert m.cells[Cell.CORRECT] == 1 and m.decomposition_mismatches == 1

    # --- the stub shape: 0% coverage and every n/a path ---------------------
    m = score(_run(tuple(_except(c.credit_id) for c in three)), _truth(three))
    assert m.coverage == 0.0, "0% coverage is a real number, not n/a"
    assert m.correctness is None, "correctness with no commitments must be n/a"
    assert m.wrong_rate == 0.0 and m.cells[Cell.MISSED] == 3
    assert m.exceptions == 3
    assert m.exception_value_paise == 300_000
    assert m.exception_minutes == 3 * MINUTES_PER_EXCEPTION[Reason.NO_CANDIDATE]
    # Nothing was committed, so nothing was checkable. ``n/a``, never 100% -- a matcher
    # that abstained on every row has not proven its arithmetic on anything, and an
    # empty-denominator rate rendered as perfect is exactly the inversion this scorer
    # exists to prevent.
    assert m.decomposition_checked == 0
    assert m.decomposition_agreement is None, "0/0 arithmetic must be n/a, not 1.0"
    assert all(l.decomposition_mismatch is None for l in m.landings), (
        "an unchecked row must be None, not () -- () means checked and agreed"
    )

    # --- set equality, no partial credit -----------------------------------
    batched = (_credit("C0001", ("pay_0001", "pay_0002")),)
    for guess, expect in (
        (("pay_0002", "pay_0001"), Cell.CORRECT),        # order must not matter
        (("pay_0001",), Cell.WRONG_MATCH),               # a subset is not 50% right
        (("pay_0001", "pay_0002", "pay_0003"), Cell.WRONG_MATCH),  # nor is a superset
    ):
        m = score(_run((_resolved("C0001", guess),)), _truth(batched))
        assert m.cells[expect] == 1, (guess, expect, m.cells)
        # Priced only when it's wrong: a correct match is not money at risk.
        assert m.wrong_match_value_paise == (VALUE if expect is Cell.WRONG_MATCH else 0), (
            expect, m.wrong_match_value_paise,
        )
        # The arithmetic is scored only where the payment sets agree. On a wrong match the
        # two decompositions describe *different* sets of money, so every term would differ
        # and the mismatch count would be measuring the linkage failure a second time --
        # inflating one number with another number's problem.
        if expect is Cell.CORRECT:
            assert m.decomposition_checked == 1, "a correct linkage must be priced-checked"
        else:
            assert m.decomposition_checked == 0, (
                "a wrong match's arithmetic was compared -- it describes a different "
                "payment set, so the comparison is meaningless"
            )
            assert m.decomposition_agreement is None
            assert m.landings[0].decomposition_mismatch is None
    assert m.correctness == 0.0 and m.coverage == 1.0, "a wrong match is still coverage"

    # --- planted unresolvable: four ways it can go -------------------------
    planted = (_credit("C0001", ("pay_0001",), resolvable=False,
                       reason=Reason.AMBIGUOUS_DUPLICATE_AMOUNT),)
    t = _truth(planted)
    assert t.planted_unresolvable and score(_run((_except("C0001"),)), t).planted_unresolvable == 1
    # abstaining is right
    m = score(_run((_except("C0001", Reason.AMBIGUOUS_DUPLICATE_AMOUNT),)), t)
    assert m.cells[Cell.CORRECT_ABSTENTION] == 1 and m.abstention_rate == 1.0
    assert m.coverage == 0.0 and m.correctness is None
    # inventing an answer is the worst cell
    m = score(_run((_resolved("C0001", ("pay_0009",)),)), t)
    assert m.cells[Cell.WRONG_MATCH_INVENTED] == 1 and m.wrong_matches == 1
    assert m.cells[Cell.WRONG_MATCH] == 0, "invented must not be pooled with wrong_match"
    assert m.wrong_match_value_paise == VALUE, "WRONG_MATCH_INVENTED is priced too -- both live in WRONG_MATCH_CELLS"
    # guessing right is luck, not correctness -- and is NOT impossible
    m = score(_run((_resolved("C0001", ("pay_0001",)),)), t)
    assert m.cells[Cell.LUCKY_GUESS] == 1 and m.cells[Cell.CORRECT] == 0
    assert m.correctness == 0.0, "credit for a lucky guess would reward guessing"
    assert m.wrong_matches == 1
    assert m.wrong_match_value_paise == VALUE, "a lucky guess is still money nobody was told to check"
    # ...and its arithmetic *is* scored, deliberately: the payment set matches, so the two
    # decompositions describe the same money and the comparison is meaningful. The gate is
    # set equality, not the cell -- the linkage is uncredited here for a separate reason
    # (the row was planted unresolvable, so committing to it was a guess), and folding that
    # judgement into the arithmetic axis would leave the guessed rows silently unpriced.
    assert m.decomposition_checked == 1 and m.decomposition_mismatches == 0
    assert m.decomposition_agreement == 1.0
    # ignoring it is also an acceptable abstention
    m = score(_run((V("C0001", Outcome.IGNORED, reason=Reason.NON_GATEWAY_CREDIT),)), t)
    assert m.cells[Cell.CORRECT_ABSTENTION] == 1

    # --- discarding a real credit is its own failure -----------------------
    m = score(_run((V("C0001", Outcome.IGNORED, reason=Reason.NON_GATEWAY_CREDIT),)),
              _truth((_credit("C0001", ("pay_0001",)),)))
    assert m.cells[Cell.WRONG_IGNORE] == 1
    assert m.wrong_matches == 0, "a wrong ignore is not a wrong match -- different fix"
    assert m.coverage == 0.0, "ignoring is not committing"
    assert m.wrong_match_value_paise == 0, "WRONG_IGNORE is not in WRONG_MATCH_CELLS"

    # --- noise stays out of the headline (Phase 7 shape) -------------------
    mixed = (_credit("C0001", ("pay_0001",)), _credit("C0002", ("pay_0002",)))
    t = _truth(mixed, noise=("C0003",))
    run = _run((
        _resolved("C0001", ("pay_0001",)),
        _resolved("C0002", ("pay_0002",)),
        V("C0003", Outcome.IGNORED, reason=Reason.NON_GATEWAY_CREDIT),
    ))
    m = score(run, t)
    assert m.total_bank_rows == 3 and m.gateway_credits == 2 and m.non_gateway_credits == 1
    assert m.coverage == 1.0, "2/2 gateway -- the noise row must not dilute it"
    assert m.noise_recall == 1.0 and m.noise_precision == 1.0
    assert m.cells[Cell.NOISE_CORRECTLY_IGNORED] == 1
    # a noise row with no full truth entry still scores, and still is not a match
    t2 = _truth(mixed, noise=("C0009",))
    assert expected_credit_ids(t2) == ("C0001", "C0002", "C0009")
    m = score(_run((
        _resolved("C0001", ("pay_0001",)), _resolved("C0002", ("pay_0002",)),
        _resolved("C0009", ("pay_0001",)),
    )), t2)
    assert m.cells[Cell.NOISE_MISHANDLED] == 1 and m.coverage == 1.0

    # --- the identity is asserted, not assumed -----------------------------
    try:
        score(_run((_resolved("C0001", ("pay_0001",)),)), _truth(three))
    except MetricsError as e:
        assert "3 bank rows" in str(e), e
    else:
        raise AssertionError("scored a run that dropped two of three rows")

    # --- the JSON document: n/a is null, timing is quarantined -------------
    m = score(_run(tuple(_except(c.credit_id) for c in three)), _truth(three))
    doc = m.as_json()
    assert doc["rates"]["correctness"] is None, "n/a must serialise as null, not 0"
    assert doc["rates"]["coverage"] == 0.0
    assert set(doc["cells"]) == {str(c) for c in Cell}
    assert doc["timing"] == {"wall_clock_seconds": 0.01}
    slower = score(_run(tuple(_except(c.credit_id) for c in three)), _truth(three))
    object.__setattr__(slower, "wall_clock_seconds", 9.99)
    assert {k: v for k, v in doc.items() if k != "timing"} == {
        k: v for k, v in slower.as_json().items() if k != "timing"
    }, "two runs must differ only inside timing"

    # --- the answer key must not leak through the landings ----------------
    m = score(_run((_resolved("C0001", ("pay_0009",)),)),
              _truth((_credit("C0001", ("pay_0001",)),)))
    landing = m.landings[0]
    assert landing.cell is Cell.WRONG_MATCH and landing.is_wrong
    assert not hasattr(landing, "expected_payment_ids")
    assert "pay_0001" not in repr(landing), (
        "a Landing must never carry truth's answer -- that is how Phase 3 starts "
        "fitting the answer key"
    )
    assert m.wrong_match_value_paise == VALUE
    assert m.as_json()["risk"] == {"wrong_match_value_paise": VALUE}

    # --- the payment count: derived, and cross-checked against the header -----
    # The positive path runs on every real file, since the generator always writes
    # ``counts.payments``; this is the negative one. A truth file whose header disagrees with
    # its own credits has an unclear denominator for every rate in the block, and an
    # unexercised refusal is decoration -- so the disagreement is staged here.
    from dataclasses import replace

    three_t = _truth(three)
    assert score(_run(tuple(_except(c.credit_id) for c in three)), three_t).total_payments == 3
    # The header is absent in these fixtures, so the check must stay quiet rather than read a
    # missing key as a zero and refuse every truth file that omits it.
    assert "payments" not in three_t.counts
    try:
        score(
            _run(tuple(_except(c.credit_id) for c in three)),
            replace(three_t, counts={"credits": 3, "payments": 99}),
        )
    except MetricsError as e:
        assert "99" in str(e) and "3" in str(e), e
    else:
        raise AssertionError(
            "a truth file stating 99 payments while its credits account for 3 must be "
            "refused, not scored against whichever denominator won"
        )
    # An unsettled payment belongs to the population even though no credit cites it, so the
    # derived count is the union rather than the credited set alone.
    assert score(
        _run(tuple(_except(c.credit_id) for c in three)),
        replace(three_t, unsettled_payment_ids=("pay_9001", "pay_9002")),
    ).total_payments == 5

    # --- Phase 9 step 1: one accessor, and both of its refusals fired ---------
    # ``minutes_for`` replaced two ``.get`` calls whose defaults disagreed by 10 minutes
    # (score defaulted to 10, the queue line to 0), so an unpriced code cost 10 in the total
    # and showed 0 in the queue. The fallbacks are gone; these are the branches that replaced
    # them, and an unexercised refusal is decoration.
    #
    # Positive control first, over the whole vocabulary: if ``minutes_for`` raised on
    # everything, the two refusals below would still "pass" and this accessor would be broken
    # in the direction no fixture would notice.
    assert all(minutes_for(r) == MINUTES_PER_EXCEPTION[r] for r in Reason)
    assert minutes_for(Reason.NO_CANDIDATE) == 10

    try:
        minutes_for(None)
    except MetricsError as e:
        assert "no reason code" in str(e), e
    else:
        raise AssertionError(
            "priced an exception carrying no reason code -- that is the state "
            "Verdict.__post_init__ refuses, so reaching a number here would be a scorer "
            "bug reported as an estimate"
        )

    # The unpriced-code branch, reached the only way the exhaustiveness assertion allows: by
    # removing a price. Restored in ``finally``, because a self-check that leaves the module's
    # own table mutated would corrupt every assertion after it.
    _victim = Reason.NO_CANDIDATE
    _price = MINUTES_PER_EXCEPTION.pop(_victim)
    try:
        assert _victim not in MINUTES_PER_EXCEPTION, (
            "the mutation did not take, so the branch below is not the one being tested"
        )
        try:
            minutes_for(_victim)
        except MetricsError as e:
            assert "no effort estimate" in str(e) and str(_victim) in str(e), e
            # It names the row a maintainer has to edit as well as the table, because gate 7
            # compares the two and a fix to one alone fails there instead of here.
            assert "ASSUMPTIONS.md" in str(e), e
        else:
            raise AssertionError(
                f"{_victim} priced without an entry in MINUTES_PER_EXCEPTION -- the default "
                f"this accessor exists to delete is still reachable"
            )
    finally:
        MINUTES_PER_EXCEPTION[_victim] = _price
    assert set(MINUTES_PER_EXCEPTION) == set(Reason), "the table was left mutated"

    print("metrics.py self-check ok")
