"""The identity, the confusion matrix, and the four rates.

The one module that joins matcher output against the answer key. Everything it knows
about being *right* comes from here; everything it knows about being *confident* comes
from the verdict file. Keeping those two words apart is the entire point of the phase:

    coverage    -- how often the matcher committed to an answer
    correctness -- how often the answer it committed to was right

A single accuracy number averages them and loses the distinction, so this module
returns a frozen dataclass with both and no combined field. There is nothing to
collapse.

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

from ..common.reasons import CORRECT_ABSTENTION_CODES, Reason
from ..common.verdict import Outcome, Verdict, VerdictFile
from .truth_io import Truth, TruthCredit

#: Minutes a human needs to clear one exception, by reason code. **Assumptions, not
#: measurements** -- stated in ASSUMPTIONS.md so a judge can challenge the number
#: rather than discover it was invented at demo time. Phase 9 refines these into
#: per-group estimates once exceptions are ranked; Phase 2 only needs the field to
#: exist and the numbers to be declared.
MINUTES_PER_EXCEPTION: dict[Reason, int] = {
    Reason.NO_CANDIDATE: 10,
    Reason.AMBIGUOUS_MULTI_SUBSET: 15,
    Reason.AMBIGUOUS_DUPLICATE_AMOUNT: 8,
    Reason.UNEXPLAINED_RESIDUAL: 12,
    Reason.PARTIAL_SETTLEMENT_PENDING: 5,
    Reason.REFUND_UNLINKED: 10,
    Reason.FX_RATE_GAP: 20,
    Reason.NON_GATEWAY_CREDIT: 3,
    Reason.CREDIT_MISSING: 15,
    Reason.SETTLEMENT_MISSING: 15,
    Reason.ROUNDING_DRIFT: 5,
}

#: Fallback for a reason code with no estimate. Should be unreachable -- the assertion
#: below keeps the table exhaustive -- but a scorer must not crash on an unpriced
#: exception when the alternative is reporting a slightly wrong number of minutes.
DEFAULT_MINUTES_PER_EXCEPTION = 10

assert set(MINUTES_PER_EXCEPTION) == set(Reason), (
    "every reason code needs an effort estimate, or the human-time figure silently "
    "under-reports: missing " + str(sorted(str(r) for r in set(Reason) - set(MINUTES_PER_EXCEPTION)))
)


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

    @property
    def is_wrong(self) -> bool:
        return self.cell in WRONG_MATCH_CELLS or self.cell in (
            Cell.WRONG_IGNORE, Cell.NOISE_MISHANDLED
        )


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

    cells: dict[Cell, int]
    landings: tuple[Landing, ...] = field(repr=False)

    exceptions: int = 0
    exception_value_paise: int = 0
    exception_minutes: int = 0
    ignores_total: int = 0

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
            },
            "exceptions": {
                "count": self.exceptions,
                "value_paise": self.exception_value_paise,
                "estimated_minutes": self.exception_minutes,
            },
        }


#: Bumped on a breaking change to ``Metrics.as_json``. Phase 9 re-ranks this document
#: and Phase 11 renders it; a format that shifts silently costs both.
METRICS_SCHEMA_VERSION = 1


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
        landings.append(
            Landing(
                credit_id=verdict.credit_id,
                cell=cell,
                outcome=verdict.outcome,
                reason=verdict.reason,
                residual_paise=verdict.residual_paise,
                value_paise=value,
            )
        )

        if verdict.outcome is Outcome.EXCEPTION:
            exception_value += value
            exception_minutes += MINUTES_PER_EXCEPTION.get(
                verdict.reason, DEFAULT_MINUTES_PER_EXCEPTION
            )

    gateway = [c for c in truth.credits if c.credit_id not in noise_ids]
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
        cells=cells,
        landings=tuple(landings),
        exceptions=counts["EXCEPTION"],
        exception_value_paise=exception_value,
        exception_minutes=exception_minutes,
        ignores_total=counts["IGNORED"],
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

    def _credit(cid: str, pids: tuple[str, ...], *, resolvable: bool = True,
                value: int = 100_000, reason: Reason | None = None) -> TruthCredit:
        return TruthCredit(
            credit_id=cid,
            settlement_ids=(f"setl_{cid[1:]}",),
            payment_ids=pids,
            refunds_netted=(),
            reserve_held_paise=0,
            decomposition=TruthDecomposition(value, 0, 0, 0, 0, 0, value),
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

    def _resolved(cid: str, pids: tuple[str, ...]) -> V:
        return V(cid, Outcome.RESOLVED, (f"setl_{cid[1:]}",), pids, tier=1, residual_paise=0)

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

    # --- the stub shape: 0% coverage and every n/a path ---------------------
    m = score(_run(tuple(_except(c.credit_id) for c in three)), _truth(three))
    assert m.coverage == 0.0, "0% coverage is a real number, not n/a"
    assert m.correctness is None, "correctness with no commitments must be n/a"
    assert m.wrong_rate == 0.0 and m.cells[Cell.MISSED] == 3
    assert m.exceptions == 3
    assert m.exception_value_paise == 300_000
    assert m.exception_minutes == 3 * MINUTES_PER_EXCEPTION[Reason.NO_CANDIDATE]

    # --- set equality, no partial credit -----------------------------------
    batched = (_credit("C0001", ("pay_0001", "pay_0002")),)
    for guess, expect in (
        (("pay_0002", "pay_0001"), Cell.CORRECT),        # order must not matter
        (("pay_0001",), Cell.WRONG_MATCH),               # a subset is not 50% right
        (("pay_0001", "pay_0002", "pay_0003"), Cell.WRONG_MATCH),  # nor is a superset
    ):
        m = score(_run((_resolved("C0001", guess),)), _truth(batched))
        assert m.cells[expect] == 1, (guess, expect, m.cells)
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
    # guessing right is luck, not correctness -- and is NOT impossible
    m = score(_run((_resolved("C0001", ("pay_0001",)),)), t)
    assert m.cells[Cell.LUCKY_GUESS] == 1 and m.cells[Cell.CORRECT] == 0
    assert m.correctness == 0.0, "credit for a lucky guess would reward guessing"
    assert m.wrong_matches == 1
    # ignoring it is also an acceptable abstention
    m = score(_run((V("C0001", Outcome.IGNORED, reason=Reason.NON_GATEWAY_CREDIT),)), t)
    assert m.cells[Cell.CORRECT_ABSTENTION] == 1

    # --- discarding a real credit is its own failure -----------------------
    m = score(_run((V("C0001", Outcome.IGNORED, reason=Reason.NON_GATEWAY_CREDIT),)),
              _truth((_credit("C0001", ("pay_0001",)),)))
    assert m.cells[Cell.WRONG_IGNORE] == 1
    assert m.wrong_matches == 0, "a wrong ignore is not a wrong match -- different fix"
    assert m.coverage == 0.0, "ignoring is not committing"

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

    print("metrics.py self-check ok")
