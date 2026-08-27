"""The matcher output contract — one verdict per bank row, no exceptions to that rule.

This is the interface between the two halves of the system, which is why it lives
in ``common/`` rather than in either one:

  * **Phase 3's matcher writes this format.** It may not import ``hisaab.scoring``
    (``tools/check_isolation.py`` fails the build if it tries), so the contract
    cannot live on the scoring side.
  * **Phase 2's scorer reads it.** So it cannot live under ``hisaab/matcher/`` either.

It touches no truth, so being importable from the matching path costs no isolation.
``tools/check_isolation.py`` must never list this module in ``TRUTH_READERS``.

**There is deliberately no reader here.** Loading and validating ``matches.json`` is
``hisaab/scoring/verdict_io.py``'s job, on the scoring side, because a matcher that
validates its own output can bless a file the scorer would refuse. Writing is safe to
share; blessing is not.

The dataclasses guard construction *in memory*; ``verdict_io`` guards the same rules
again on read-back from disk. That duplication is the pattern Phase 1 established for
the generator (``invariants.check_story`` before the write, ``tools/verify_output.py``
after it), and it earns its keep the same way: the write step itself can corrupt, and a
hand-written or hand-edited verdict file never passed through these constructors at all.

Three outcomes, and the third matters more than it looks:

  * ``RESOLVED``   -- the matcher commits to an answer. Scored for correctness.
  * ``EXCEPTION``  -- the matcher abstains, with a reason code. Cheap and honest.
  * ``IGNORED``    -- the matcher claims the row is not gateway income at all.

Nothing emits ``IGNORED`` until Phase 7 adds noise rows. It is in the contract from the
start because the alternative is changing the contract in Phase 7 and re-running every
number measured before it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .money import paise
from .reasons import Reason

#: Bumped only on a breaking change to the shape below. ``verdict_io`` refuses a
#: version it does not know rather than guessing at the difference -- the same rule
#: ``hisaab/scoring/truth_io.py`` applies to the answer key.
#:
#: v2 (Phase 4 step 5): a resolved verdict carries ``credit_amount_paise`` and a
#: ``decomposition``. Breaking, because both are required on every ``RESOLVED`` row, so a
#: v1 file cannot be read as v2 -- and being refused is the point. A v1 verdict asserted a
#: residual that nothing could check; scoring it under v2 rules would silently treat an
#: unproven match as a proven one.
VERDICT_SCHEMA_VERSION = 2

MATCHES_JSON = "matches.json"


class Outcome(str, Enum):
    """What the matcher decided about one bank row."""

    RESOLVED = "RESOLVED"
    EXCEPTION = "EXCEPTION"
    IGNORED = "IGNORED"

    def __str__(self) -> str:  # so f-strings and json.dump give the bare code
        return self.value

    @property
    def is_committal(self) -> bool:
        """True when the matcher asserted an answer that can be *wrong*.

        The distinction the whole submission rests on: an abstention costs a human a
        look, a wrong match silently corrupts the books.
        """
        return self is Outcome.RESOLVED


@dataclass(frozen=True, slots=True)
class Decomposition:
    """How a resolved row's credit is accounted for, to the paisa.

    Phase 4 step 5. Six components in the same shape and the same order as
    ``truth.json``'s own decomposition block, so the scorer can compare them term by term
    rather than only comparing a single total -- a fee that is right in sum because GST was
    wrong in the opposite direction is a real arithmetic error and lands on the same total.

    **A third implementation of this shape, deliberately.** ``generator/model.py`` has one
    and ``scoring/truth_io.py`` has another, and neither is imported here: the matching path
    may not import the generator (``tools/check_isolation.py`` check 6), and the scorer must
    not import the matcher. Same rule as ``load.py``'s CSV headers -- this is a *schema*, so
    it is duplicated in order that drift fails loudly, while the arithmetic that produces
    the numbers (``money.mul_bps``) is shared so drift there is impossible.

    ``tds_paise``, ``refunds_paise`` and ``reserve_paise`` are present at zero from the
    start. Phase 6 makes them non-zero, which is then a change of value rather than a
    change of shape -- decision #10's reasoning applied to the output contract.
    """

    gross_paise: int
    fee_paise: int = 0
    gst_paise: int = 0
    tds_paise: int = 0
    refunds_paise: int = 0
    reserve_paise: int = 0
    #: The rule that accounts for the deductions, e.g. "gateway fee + GST at declared
    #: rates". Named so a row says *why* it balanced and not only that it did: with several
    #: candidate rules in play by Phase 6, a row that balanced by coincidence is only
    #: visible if the output states what it credited.
    rule: str | None = None

    def __post_init__(self) -> None:
        for label, amount in self.components().items():
            paise(amount)
            if amount < 0:
                # A negative component would let two of them cancel into a decomposition
                # that balances while describing something impossible.
                raise ValueError(f"decomposition {label} is negative: {amount}")

    def components(self) -> dict[str, int]:
        """The six terms, in truth's order. Excludes ``rule``, which is prose."""
        return {
            "gross_paise": self.gross_paise,
            "fee_paise": self.fee_paise,
            "gst_paise": self.gst_paise,
            "tds_paise": self.tds_paise,
            "refunds_paise": self.refunds_paise,
            "reserve_paise": self.reserve_paise,
        }

    @property
    def deductions_paise(self) -> int:
        return (
            self.fee_paise + self.gst_paise + self.tds_paise
            + self.refunds_paise + self.reserve_paise
        )

    @property
    def expected_credit_paise(self) -> int:
        """What this decomposition says the bank should have credited."""
        return self.gross_paise - self.deductions_paise

    def as_json(self) -> dict[str, object]:
        return {
            **self.components(),
            "expected_credit_paise": self.expected_credit_paise,
            "rule": self.rule,
        }


@dataclass(frozen=True, slots=True)
class Verdict:
    """One bank row's outcome.

    Every field is present on every verdict, including the ones that are ``None`` --
    ``residual_paise`` on an abstention, ``reason`` on a match. Absent-versus-null is
    the branch ``hisaab/scoring/truth_io.py`` refuses to allow in the answer key, and
    for the same reason: a reader that has to distinguish "missing" from "null" grows a
    branch per field, and one of those branches is eventually wrong.
    """

    credit_id: str
    outcome: Outcome
    settlement_ids: tuple[str, ...] = ()
    payment_ids: tuple[str, ...] = ()
    tier: int | None = None
    confidence: float | None = None
    reason: Reason | None = None
    note: str | None = None
    residual_paise: int | None = None
    #: What the bank actually credited, as the matcher read it. Carried so the residual is
    #: **checkable** rather than merely stated: with the amount and the decomposition both
    #: present, ``residual == credit - expected`` is arithmetic anyone can re-run. A v1
    #: verdict asserted a residual that no reader could verify.
    credit_amount_paise: int | None = None
    #: How the credit is accounted for. Required on ``RESOLVED``: a match whose money
    #: nobody can price is not proven, and Phase 4's whole claim is that every point of
    #: coverage was earned by arithmetic.
    decomposition: Decomposition | None = None

    def __post_init__(self) -> None:
        if not self.credit_id:
            raise ValueError("a verdict needs a credit_id")
        if self.outcome is Outcome.RESOLVED:
            # A match with nothing in it is an abstention wearing a match's label, and
            # it would score as coverage.
            if not self.payment_ids:
                raise ValueError(f"{self.credit_id}: RESOLVED with no payment_ids")
            if not self.settlement_ids:
                raise ValueError(f"{self.credit_id}: RESOLVED with no settlement_ids")
            if self.reason is not None:
                raise ValueError(f"{self.credit_id}: RESOLVED carries a reason code")
            # Phase 4 forces this to mean something. Requiring it from Phase 3 -- where
            # an exact match trivially residuals to zero -- means "proven" and "found"
            # can never quietly become the same claim.
            if self.residual_paise is None:
                raise ValueError(
                    f"{self.credit_id}: RESOLVED must state a residual "
                    f"(0 for an exact match) -- a match nobody can price is not proven"
                )
            paise(self.residual_paise)
            # Phase 4 step 5. A residual alone is an unverifiable claim: it is one number
            # asserted about arithmetic nobody else can see. With the credit amount and the
            # decomposition both present, the claim becomes re-runnable by any reader.
            if self.credit_amount_paise is None:
                raise ValueError(
                    f"{self.credit_id}: RESOLVED must state credit_amount_paise -- "
                    f"without it the residual is a number no reader can check"
                )
            paise(self.credit_amount_paise)
            if self.decomposition is None:
                raise ValueError(
                    f"{self.credit_id}: RESOLVED must carry a decomposition -- every "
                    f"point of coverage is supposed to be earned by arithmetic, and a "
                    f"match that cannot say how the money adds up has not earned one"
                )
            if not isinstance(self.decomposition, Decomposition):
                raise TypeError(
                    f"{self.credit_id}: decomposition must be a Decomposition, got "
                    f"{type(self.decomposition).__name__}"
                )
            # **The balance assertion** (.plan/phase4.md step 5). The residual is redundant
            # with the other two fields on purpose -- it is a checksum, not a third
            # independent number, and this is where the redundancy earns its keep. A
            # decomposition that does not reconcile to the credit it describes is caught at
            # construction, so it can never reach a file, be scored, or be quoted into a
            # report.
            want = self.credit_amount_paise - self.decomposition.expected_credit_paise
            if self.residual_paise != want:
                raise ValueError(
                    f"{self.credit_id}: the decomposition does not balance. Credit "
                    f"{self.credit_amount_paise}p - expected "
                    f"{self.decomposition.expected_credit_paise}p = {want}p, but the "
                    f"verdict states a residual of {self.residual_paise}p"
                )
        else:
            if self.payment_ids or self.settlement_ids:
                raise ValueError(
                    f"{self.credit_id}: {self.outcome} must not name payments or "
                    f"settlements -- an abstention that carries an answer is a match"
                )
            if self.tier is not None:
                raise ValueError(f"{self.credit_id}: {self.outcome} carries a tier")
            if self.residual_paise is not None:
                raise ValueError(
                    f"{self.credit_id}: {self.outcome} carries a residual, but there is "
                    f"no claimed decomposition for it to be the remainder of"
                )
            if self.decomposition is not None:
                raise ValueError(
                    f"{self.credit_id}: {self.outcome} carries a decomposition, but it "
                    f"named no payments to decompose -- an abstention that prices an "
                    f"answer is a match"
                )
            # ``credit_amount_paise`` is deliberately NOT forbidden here. It is the
            # matcher's reading of a bank column, not a claim about a linkage, and an
            # abstention that quotes the amount it could not explain is more useful than
            # one that does not. Nothing scores it.
            if self.credit_amount_paise is not None:
                paise(self.credit_amount_paise)
        if self.outcome is Outcome.EXCEPTION and self.reason is None:
            # Same rule truth_io.py applies to an unresolvable credit: an unexplained
            # exception is a silent drop dressed up as a finding.
            raise ValueError(
                f"{self.credit_id}: EXCEPTION must carry a reason code -- an "
                f"unexplained exception is a silent drop"
            )
        if self.reason is not None and not isinstance(self.reason, Reason):
            raise TypeError(
                f"{self.credit_id}: reason must be a hisaab.common.reasons.Reason, "
                f"got {type(self.reason).__name__} ({self.reason!r})"
            )
        if len(set(self.payment_ids)) != len(self.payment_ids):
            raise ValueError(f"{self.credit_id}: duplicate payment_ids")
        if len(set(self.settlement_ids)) != len(self.settlement_ids):
            raise ValueError(f"{self.credit_id}: duplicate settlement_ids")

    @property
    def payment_set(self) -> frozenset[str]:
        """The comparison key for correctness.

        A ``frozenset`` because correctness is **set equality, no partial credit**
        (decision 3): order must not matter, and neither must any notion of overlap.
        Returning a set rather than a sorted tuple is what makes a Jaccard score
        awkward to write, which is the intent.
        """
        return frozenset(self.payment_ids)

    def as_json(self) -> dict[str, object]:
        return {
            "credit_id": self.credit_id,
            "outcome": str(self.outcome),
            "settlement_ids": list(self.settlement_ids),
            "payment_ids": list(self.payment_ids),
            "tier": self.tier,
            "confidence": self.confidence,
            "reason": None if self.reason is None else str(self.reason),
            "note": self.note,
            "residual_paise": self.residual_paise,
            "credit_amount_paise": self.credit_amount_paise,
            "decomposition": None if self.decomposition is None else self.decomposition.as_json(),
        }


@dataclass(frozen=True, slots=True)
class VerdictFile:
    """A complete matcher run: provenance, timing, and one verdict per bank row.

    ``seed`` and ``month`` are carried so the scorer can refuse to score run A's
    verdicts against run B's answer key. That mistake produces a plausible number
    rather than a crash, which makes it the most expensive kind of bug to have.
    """

    seed: int
    month: str
    matcher: str
    verdicts: tuple[Verdict, ...]
    wall_clock_seconds: float | None = None
    schema_version: int = VERDICT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.matcher:
            raise ValueError("a run must name its matcher, e.g. 'tier1@0.3.0'")
        seen: set[str] = set()
        for v in self.verdicts:
            if v.credit_id in seen:
                raise ValueError(f"duplicate verdict for {v.credit_id}")
            seen.add(v.credit_id)

    def counts(self) -> dict[str, int]:
        """Verdicts per outcome, with every outcome present even at zero.

        Present-at-zero so ``metrics`` can assert the identity by lookup instead of
        branching on key presence, and so a report never omits a row because the count
        happened to be nothing.
        """
        counts = {str(o): 0 for o in Outcome}
        for v in self.verdicts:
            counts[str(v.outcome)] += 1
        return counts

    def as_json(self) -> dict[str, object]:
        """The document, with everything non-deterministic confined to ``timing``.

        ``emit.build_manifest`` does the same, for the same reason: Phase 11 quotes the
        metric block into a report that is subject to the reproducibility rule, and a
        wall clock in the body would make a byte-comparison of two identical runs fail.
        Compare the document minus ``timing``; print the clock in the human block.
        """
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "month": self.month,
            "matcher": self.matcher,
            "timing": {"wall_clock_seconds": self.wall_clock_seconds},
            "verdicts": [v.as_json() for v in self.verdicts],
        }


def write_verdicts(path: Path | str, run: VerdictFile) -> Path:
    """Write ``matches.json``. Returns the path actually written.

    Byte-for-byte the same writer settings as ``hisaab/generator/emit.write_json``:
    ``indent=2``, no ASCII escaping, ``allow_nan=False`` so a stray ``inf`` in a
    confidence score fails here instead of producing JSON no other parser will read,
    explicit ``\\n`` so Windows does not silently make the file un-diffable, and a
    trailing newline.
    """
    out = Path(path)
    if out.is_dir():
        out = out / MATCHES_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(run.as_json(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return out


if __name__ == "__main__":
    import tempfile

    def refuses(build, label: str) -> None:
        try:
            build()
        except (ValueError, TypeError):
            return
        raise AssertionError(f"Verdict accepted {label}")

    # --- the decomposition, on its own ------------------------------------
    # The track spec's worked example: a Rs 1,111 card sale, 2% fee, 18% GST on the fee.
    d = Decomposition(gross_paise=111_100, fee_paise=2_222, gst_paise=400,
                      rule="gateway fee + GST at declared rates")
    assert d.deductions_paise == 2_622
    assert d.expected_credit_paise == 108_478
    # Six components, in truth's order, with the total and the rule beside them.
    assert list(d.components()) == [
        "gross_paise", "fee_paise", "gst_paise", "tds_paise", "refunds_paise", "reserve_paise",
    ]
    assert d.as_json()["expected_credit_paise"] == 108_478
    assert d.as_json()["rule"] == "gateway fee + GST at declared rates"
    # Phase 6's terms are present at zero, so they are a change of value not of shape.
    assert d.tds_paise == d.refunds_paise == d.reserve_paise == 0
    # A zero-deduction decomposition is legitimate: clean mode, and any zero-rated method.
    assert Decomposition(50_000).expected_credit_paise == 50_000
    # Negative components are refused -- two of them could cancel into a decomposition that
    # balances while describing something impossible.
    refuses(lambda: Decomposition(1_000, fee_paise=-100), "a negative fee")
    refuses(lambda: Decomposition(-1_000), "a negative gross")
    refuses(lambda: Decomposition(1.0), "a float gross")  # type: ignore[arg-type]

    resolved = Verdict(
        "C0001", Outcome.RESOLVED,
        settlement_ids=("setl_0005",), payment_ids=("pay_0001",),
        tier=1, residual_paise=0,
        credit_amount_paise=108_478, decomposition=d,
    )
    assert resolved.outcome.is_committal
    assert resolved.payment_set == frozenset({"pay_0001"})
    assert resolved.as_json()["reason"] is None
    # Every key present on every verdict, nullable ones included.
    assert set(resolved.as_json()) == {
        "credit_id", "outcome", "settlement_ids", "payment_ids",
        "tier", "confidence", "reason", "note", "residual_paise",
        "credit_amount_paise", "decomposition",
    }
    assert resolved.as_json()["decomposition"] == d.as_json()  # type: ignore[index]

    # --- the balance assertion (Phase 4 step 5) ---------------------------
    # The residual is a checksum over the other two fields, not a third independent
    # number, so a decomposition that does not reconcile to its own credit is refused at
    # construction and can never reach a file, a score, or a report.
    refuses(
        lambda: Verdict("C1", Outcome.RESOLVED, ("setl_0001",), ("pay_0001",),
                        residual_paise=0, credit_amount_paise=108_478,
                        decomposition=Decomposition(111_100, fee_paise=2_222, gst_paise=399)),
        "a decomposition one paisa out",
    )
    refuses(
        lambda: Verdict("C1", Outcome.RESOLVED, ("setl_0001",), ("pay_0001",),
                        residual_paise=0, credit_amount_paise=108_479, decomposition=d),
        "a credit amount that disagrees with the decomposition",
    )
    refuses(
        lambda: Verdict("C1", Outcome.RESOLVED, ("setl_0001",), ("pay_0001",),
                        residual_paise=0, decomposition=d),
        "RESOLVED with no credit amount",
    )
    refuses(
        lambda: Verdict("C1", Outcome.RESOLVED, ("setl_0001",), ("pay_0001",),
                        residual_paise=0, credit_amount_paise=108_478),
        "RESOLVED with no decomposition",
    )
    refuses(
        lambda: Verdict("C1", Outcome.RESOLVED, ("setl_0001",), ("pay_0001",),
                        residual_paise=0, credit_amount_paise=1,
                        decomposition={"gross_paise": 1}),  # type: ignore[arg-type]
        "a decomposition that is a bare dict",
    )
    # ...and a stated non-zero residual must be exactly the shortfall, not merely non-zero.
    ok = Verdict("C1", Outcome.RESOLVED, ("setl_0001",), ("pay_0001",),
                 residual_paise=-500, credit_amount_paise=107_978,
                 decomposition=Decomposition(108_478))
    assert ok.residual_paise == -500
    refuses(
        lambda: Verdict("C1", Outcome.RESOLVED, ("setl_0001",), ("pay_0001",),
                        residual_paise=-499, credit_amount_paise=107_978,
                        decomposition=Decomposition(108_478)),
        "a residual that is close but not the actual shortfall",
    )

    exception = Verdict(
        "C0014", Outcome.EXCEPTION,
        reason=Reason.AMBIGUOUS_DUPLICATE_AMOUNT,
        note="two candidates within ₹0 and 0 days",
    )
    assert not exception.outcome.is_committal
    assert exception.as_json()["reason"] == "AMBIGUOUS_DUPLICATE_AMOUNT"
    assert exception.as_json()["residual_paise"] is None
    assert exception.payment_set == frozenset()

    ignored = Verdict("C0060", Outcome.IGNORED, reason=Reason.NON_GATEWAY_CREDIT)
    assert not ignored.outcome.is_committal

    # Set equality ignores order; that is the point of payment_set.
    two = Decomposition(gross_paise=200_000)
    a = Verdict("C0002", Outcome.RESOLVED, ("setl_0001",), ("pay_0002", "pay_0003"),
                residual_paise=0, credit_amount_paise=200_000, decomposition=two)
    b = Verdict("C0002", Outcome.RESOLVED, ("setl_0001",), ("pay_0003", "pay_0002"),
                residual_paise=0, credit_amount_paise=200_000, decomposition=two)
    assert a.payment_set == b.payment_set
    assert a.payment_set != frozenset({"pay_0002"})  # subset is NOT a match

    # --- the guards must actually fire -------------------------------------
    # Every RESOLVED probe below carries a complete, balancing proof and then breaks
    # exactly one rule. Without ``PROVEN`` each would now fail on the missing
    # ``credit_amount_paise`` instead of on the thing it names -- ``refuses`` would still
    # pass, and the guard it was written for would be untested. A check that goes inert is
    # worse than one never written, because the file still reads as though it covers this.
    PROVEN = dict(residual_paise=0, credit_amount_paise=108_478, decomposition=d)

    refuses(lambda: Verdict("", Outcome.EXCEPTION, reason=Reason.NO_CANDIDATE),
            "an empty credit_id")
    refuses(lambda: Verdict("C1", Outcome.RESOLVED, ("setl_0001",), (), **PROVEN),
            "RESOLVED with no payments")
    refuses(lambda: Verdict("C1", Outcome.RESOLVED, (), ("pay_0001",), **PROVEN),
            "RESOLVED with no settlements")
    refuses(lambda: Verdict("C1", Outcome.RESOLVED, ("setl_0001",), ("pay_0001",),
                            credit_amount_paise=108_478, decomposition=d),
            "RESOLVED with no residual")
    refuses(lambda: Verdict("C1", Outcome.RESOLVED, ("setl_0001",), ("pay_0001",),
                            reason=Reason.NO_CANDIDATE, **PROVEN),
            "RESOLVED carrying a reason")
    refuses(lambda: Verdict("C1", Outcome.EXCEPTION), "EXCEPTION with no reason")
    refuses(lambda: Verdict("C1", Outcome.EXCEPTION, payment_ids=("pay_0001",),
                            reason=Reason.NO_CANDIDATE),
            "EXCEPTION naming a payment")
    refuses(lambda: Verdict("C1", Outcome.EXCEPTION, reason=Reason.NO_CANDIDATE, tier=1),
            "EXCEPTION carrying a tier")
    refuses(lambda: Verdict("C1", Outcome.EXCEPTION, reason=Reason.NO_CANDIDATE,
                            residual_paise=0),
            "EXCEPTION carrying a residual")
    refuses(lambda: Verdict("C1", Outcome.EXCEPTION, reason=Reason.NO_CANDIDATE,
                            decomposition=d),
            "EXCEPTION carrying a decomposition")
    refuses(lambda: Verdict("C1", Outcome.EXCEPTION, reason="NO_CANDIDATE"),  # type: ignore[arg-type]
            "a reason that is a bare string")
    refuses(lambda: Verdict("C1", Outcome.RESOLVED, ("setl_0001",),
                            ("pay_0001", "pay_0001"), **PROVEN),
            "duplicate payment_ids")
    refuses(lambda: Verdict("C1", Outcome.RESOLVED, ("setl_0001",), ("pay_0001",),
                            residual_paise=1.5,  # type: ignore[arg-type]
                            credit_amount_paise=108_478, decomposition=d),
            "a float residual")
    # ...and the positive control: with nothing broken, the same shape is accepted. Without
    # this, every probe above could be passing because ``PROVEN`` itself is invalid.
    assert Verdict("C1", Outcome.RESOLVED, ("setl_0001",), ("pay_0001",),
                   **PROVEN).residual_paise == 0

    # An abstention may still quote the amount it could not explain -- that is a reading of
    # a bank column, not a claim about a linkage, and nothing scores it.
    assert Verdict("C1", Outcome.EXCEPTION, reason=Reason.NO_CANDIDATE,
                   credit_amount_paise=108_478).credit_amount_paise == 108_478

    # A residual may be non-zero in **either** direction, and the sign is now a statement
    # about the world rather than a convention: ``residual = credit - expected``, so
    #   negative -> the bank credited *less* than the priced deductions account for, i.e.
    #               money is missing beyond anything the model can name;
    #   positive -> the bank credited *more* than the gross, an over-credit, which no
    #               deduction can ever explain (``fees.explain_gap`` refuses it outright).
    # Both are real Phase 4 findings. Before v2 the sign was unverifiable, and this pair
    # was written the wrong way round without anything noticing.
    short = Verdict("C3", Outcome.RESOLVED, ("setl_0001",), ("pay_0001",),
                    residual_paise=-100, credit_amount_paise=99_900,
                    decomposition=Decomposition(100_000))
    assert short.residual_paise == -100
    over = Verdict("C4", Outcome.RESOLVED, ("setl_0001",), ("pay_0001",),
                   residual_paise=100, credit_amount_paise=100_100,
                   decomposition=Decomposition(100_000))
    assert over.residual_paise == 100

    run = VerdictFile(seed=42, month="2026-08", matcher="fixture:selfcheck@1",
                      verdicts=(resolved, exception, ignored), wall_clock_seconds=0.02)
    # The literal is deliberate. A bump is a breaking change to a file other tools read,
    # so it should cost an edit here and a conscious one -- ``== VERDICT_SCHEMA_VERSION``
    # alone would agree with any value the constant happened to hold.
    assert run.schema_version == VERDICT_SCHEMA_VERSION == 2
    assert run.counts() == {"RESOLVED": 1, "EXCEPTION": 1, "IGNORED": 1}
    # Every outcome present at zero, so metrics never branches on key presence.
    assert VerdictFile(1, "2026-08", "m", ()).counts() == {
        "RESOLVED": 0, "EXCEPTION": 0, "IGNORED": 0
    }
    refuses(lambda: VerdictFile(42, "2026-08", "", ()), "a run with no matcher name")
    refuses(lambda: VerdictFile(42, "2026-08", "m", (resolved, resolved)),
            "two verdicts for one credit")

    doc = run.as_json()
    assert doc["timing"] == {"wall_clock_seconds": 0.02}
    # The reproducibility contract: identical runs differ only inside timing.
    slower = VerdictFile(42, "2026-08", "fixture:selfcheck@1",
                         (resolved, exception, ignored), wall_clock_seconds=9.99)
    assert {k: v for k, v in doc.items() if k != "timing"} == {
        k: v for k, v in slower.as_json().items() if k != "timing"
    }

    with tempfile.TemporaryDirectory(prefix="hisaab-verdict-") as tmp:
        written = write_verdicts(Path(tmp), run)
        assert written.name == MATCHES_JSON
        raw = written.read_bytes()
        assert raw.endswith(b"\n") and b"\r\n" not in raw, "line endings must be LF"
        assert json.loads(raw.decode("utf-8")) == doc
        # Directory or explicit filename, both work.
        assert write_verdicts(Path(tmp) / "sub" / "out.json", run).name == "out.json"

    print("verdict.py self-check ok")
