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
VERDICT_SCHEMA_VERSION = 1

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

    resolved = Verdict(
        "C0001", Outcome.RESOLVED,
        settlement_ids=("setl_0005",), payment_ids=("pay_0001",),
        tier=1, residual_paise=0,
    )
    assert resolved.outcome.is_committal
    assert resolved.payment_set == frozenset({"pay_0001"})
    assert resolved.as_json()["reason"] is None
    # Every key present on every verdict, nullable ones included.
    assert set(resolved.as_json()) == {
        "credit_id", "outcome", "settlement_ids", "payment_ids",
        "tier", "confidence", "reason", "note", "residual_paise",
    }

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
    a = Verdict("C0002", Outcome.RESOLVED, ("setl_0001",), ("pay_0002", "pay_0003"),
                residual_paise=0)
    b = Verdict("C0002", Outcome.RESOLVED, ("setl_0001",), ("pay_0003", "pay_0002"),
                residual_paise=0)
    assert a.payment_set == b.payment_set
    assert a.payment_set != frozenset({"pay_0002"})  # subset is NOT a match

    # --- the guards must actually fire -------------------------------------
    refuses(lambda: Verdict("", Outcome.EXCEPTION, reason=Reason.NO_CANDIDATE),
            "an empty credit_id")
    refuses(lambda: Verdict("C1", Outcome.RESOLVED, ("setl_0001",), (), residual_paise=0),
            "RESOLVED with no payments")
    refuses(lambda: Verdict("C1", Outcome.RESOLVED, (), ("pay_0001",), residual_paise=0),
            "RESOLVED with no settlements")
    refuses(lambda: Verdict("C1", Outcome.RESOLVED, ("setl_0001",), ("pay_0001",)),
            "RESOLVED with no residual")
    refuses(lambda: Verdict("C1", Outcome.RESOLVED, ("setl_0001",), ("pay_0001",),
                            residual_paise=0, reason=Reason.NO_CANDIDATE),
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
    refuses(lambda: Verdict("C1", Outcome.EXCEPTION, reason="NO_CANDIDATE"),  # type: ignore[arg-type]
            "a reason that is a bare string")
    refuses(lambda: Verdict("C1", Outcome.RESOLVED, ("setl_0001",),
                            ("pay_0001", "pay_0001"), residual_paise=0),
            "duplicate payment_ids")
    refuses(lambda: Verdict("C1", Outcome.RESOLVED, ("setl_0001",), ("pay_0001",),
                            residual_paise=1.5),  # type: ignore[arg-type]
            "a float residual")

    # A residual may be negative -- an over-credit is a real Phase 4 finding.
    assert Verdict("C3", Outcome.RESOLVED, ("setl_0001",), ("pay_0001",),
                   residual_paise=-100).residual_paise == -100

    run = VerdictFile(seed=42, month="2026-08", matcher="fixture:selfcheck@1",
                      verdicts=(resolved, exception, ignored), wall_clock_seconds=0.02)
    assert run.schema_version == VERDICT_SCHEMA_VERSION == 1
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
