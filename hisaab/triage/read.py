"""Reads the four fields of ``matches.json`` that triage needs, and refuses the rest.

**Why this is not ``scoring/verdict_io.load_verdicts``.** ``hisaab/triage`` is in
``MATCHER_PACKAGES`` (Phase 9 step 2), so ``tools/check_isolation.py`` check 1 forbids this
package from importing ``hisaab.scoring`` at all. That ban is the point of the package rather
than an obstacle to it: an operator ranking their own month has no answer key, so a queue that
could reach one would be a demo. The ban is *static* -- a prefix match on the import, not on
what the import happens to execute -- so it holds even though importing ``verdict_io`` today
runs no loader.

**Why a second reader is acceptable here, when duplication usually is not.** check 6's own
failure message draws the line: shared *logic* moves to ``hisaab/common/``, while a *schema*
gets duplicated with a comment saying why, "the way matcher/load.py does with the CSV headers,
so drift fails loudly instead of hiding behind a shared symbol". What ``verdict_io`` adds over
the contract is tamper detection for a file that is about to be **scored** -- the decomposition
checksum, the residual balance, every key present on every verdict -- and none of that is
triage's business. Triage reads four fields and ranks work.

So what is duplicated is the *act of subscripting four keys*, and the things that could
actually drift are imported rather than copied: ``Outcome`` and ``Reason`` are the same enums
the matcher wrote, ``VERDICT_SCHEMA_VERSION`` is the same constant, ``MATCHES_JSON`` the same
filename. A renamed key raises here; a new outcome or reason code cannot silently mean
something different.

The rejected alternative was moving ``load_verdicts`` down into ``hisaab/common/`` beside
``write_verdicts``, which is the tidier shape and is where the writer already lives. It is a
four-file change to infrastructure fifteen gates depend on, and Phase 9 does not need it.
Worth doing when a second package needs full validation; not worth doing to save the thirty
lines below.

**Subscript, never ``.get``.** Every read here is a subscript whose ``KeyError`` becomes a
refusal. A ``.get`` with a default is how "no exceptions" and "I read the wrong key" became the
same answer twice in Phase 8, and in a ranking tool a defaulted value sorts to the bottom of
the queue and is never looked at again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..common.reasons import Reason
from ..common.verdict import MATCHES_JSON, VERDICT_SCHEMA_VERSION, Outcome


class TriageError(Exception):
    """``matches.json`` is missing, malformed, or not the file triage was pointed at."""


@dataclass(frozen=True, slots=True)
class Ruling:
    """What the matcher decided about one bank row -- the whole of what triage reads.

    Deliberately **not** ``hisaab.common.verdict.Verdict``: that carries the payment sets,
    tiers and decompositions a matcher must justify, and triage has no use for any of it. A
    row's money comes from the bank statement (step 4), not from here, so a narrow record
    keeps "what did the matcher say" and "how much is at risk" from being read off the same
    object and confused.
    """

    credit_id: str
    outcome: Outcome
    #: ``None`` only where the contract allows it. ``EXCEPTION`` always carries a code
    #: (``verdict.py`` refuses one that does not), ``IGNORED`` need not, and ``RESOLVED``
    #: normally has none.
    reason: Reason | None
    #: The bank amount **as the matcher stated it** (``credit_amount_paise``), or ``None``
    #: where it did not -- ``RESOLVED`` must state it, ``IGNORED`` does in practice, and
    #: ``EXCEPTION`` serialises it as null.
    #:
    #: Named for what it is rather than copying the JSON key: this is a *claim*, and the queue's
    #: money comes from the bank statement instead (``value.amounts``). It is read for exactly
    #: one purpose -- comparing the two, so a verdict file and a data directory from different
    #: runs cannot produce a queue. **That comparison is the only thing that catches the case
    #: where both runs are the same size**, because then the credit ids coincide exactly and no
    #: id check can tell them apart. Measured across 12 cells (3 seeds x n=60/200 x clean and
    #: --all-mess): 1140 stated amounts, all equal to their own run's bank row. Swap two seeds
    #: and 35 of 37 disagree.
    stated_amount_paise: int | None = None

    @property
    def is_exception(self) -> bool:
        return self.outcome is Outcome.EXCEPTION

    @property
    def is_dismissal(self) -> bool:
        return self.outcome is Outcome.IGNORED


def load_rulings(path: Path | str) -> tuple[Ruling, ...]:
    """Parse ``matches.json`` (a file, or a directory holding one) into rulings.

    Order is the file's order, which is the bank statement's order. Triage sorts by value
    later; preserving the input order here means a group's members read in statement order
    when their values tie, rather than in whatever order a dict happened to yield.
    """
    p = Path(path)
    if p.is_dir():
        p = p / MATCHES_JSON
    if not p.exists():
        raise TriageError(
            f"matcher output not found: {p}. Triage reads what the matcher wrote -- run "
            f"the matcher first, or point --matches at the file it produced."
        )

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise TriageError(f"{p} is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise TriageError(f"{p}: expected a JSON object at the top level")

    # The version gate, before any field is read. A v1 file states a residual nothing can
    # verify; ranking it would present unproven numbers in the order of proven ones.
    try:
        version = raw["schema_version"]
    except KeyError:
        raise TriageError(
            f"{p}: no schema_version. Triage will not guess at the shape of a file that "
            f"does not say what it is."
        ) from None
    if version != VERDICT_SCHEMA_VERSION:
        raise TriageError(
            f"{p}: verdict schema v{version}, but triage reads v{VERDICT_SCHEMA_VERSION}. "
            f"Re-run the matcher -- do not guess at the difference."
        )

    try:
        entries = raw["verdicts"]
    except KeyError:
        raise TriageError(f"{p}: no verdicts key") from None
    if not isinstance(entries, list):
        raise TriageError(f"{p}: verdicts must be a list, got {type(entries).__name__}")

    rulings: list[Ruling] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        where = f"{p}: verdicts[{i}]"
        if not isinstance(entry, dict):
            raise TriageError(f"{where}: expected an object, got {type(entry).__name__}")

        try:
            credit_id = entry["credit_id"]
            raw_outcome = entry["outcome"]
            raw_reason = entry["reason"]
            raw_amount = entry["credit_amount_paise"]
        except KeyError as e:
            raise TriageError(
                f"{where}: missing key {e.args[0]!r}. Every verdict carries every key, "
                f"null where it does not apply -- a missing key is usually a typo, and a "
                f"typo'd reason would group as a code of its own."
            ) from None

        if not isinstance(credit_id, str) or not credit_id:
            raise TriageError(f"{where}: credit_id must be a non-empty string, got {credit_id!r}")
        if credit_id in seen:
            # One row, one ruling. A duplicate would be counted twice in its group's total
            # and would double the money the group claims to be worth.
            raise TriageError(
                f"{where}: {credit_id} already has a ruling in this file -- one bank row "
                f"cannot be two entries in the queue"
            )
        seen.add(credit_id)

        try:
            outcome = Outcome(raw_outcome)
        except ValueError:
            raise TriageError(
                f"{where} ({credit_id}): outcome {raw_outcome!r} is not one of "
                f"{[str(o) for o in Outcome]}"
            ) from None

        reason: Reason | None = None
        if raw_reason is not None:
            try:
                reason = Reason(raw_reason)
            except ValueError:
                raise TriageError(
                    f"{where} ({credit_id}): reason {raw_reason!r} is not a known code. "
                    f"Triage groups by this field, so an unknown code would become a group "
                    f"nobody declared and would carry no effort estimate. Known codes: "
                    f"{[str(r) for r in Reason]}"
                ) from None

        # ``bool`` first: it is an ``int`` subclass, so ``True`` would otherwise pass as 1
        # paise. The same guard ``money.paise`` applies at its own boundary, for the same
        # reason -- a JSON ``true`` here is a bug in whatever wrote the file.
        if raw_amount is not None and (
            isinstance(raw_amount, bool) or not isinstance(raw_amount, int)
        ):
            raise TriageError(
                f"{where} ({credit_id}): credit_amount_paise must be an integer number of "
                f"paise or null, got {raw_amount!r} -- money is integer paise everywhere in "
                f"this system, so a float or a string here is a real bug"
            )

        if outcome is Outcome.EXCEPTION and reason is None:
            # The contract already refuses this on write; triage refuses it on read because
            # an exception with no code is a row that cannot be grouped, and silently
            # dropping it would shrink the queue without saying so.
            raise TriageError(
                f"{where} ({credit_id}): EXCEPTION with no reason code -- there is no group "
                f"to put it in, and a queue that quietly omits a row is worse than one that "
                f"stops"
            )

        rulings.append(
            Ruling(
                credit_id=credit_id,
                outcome=outcome,
                reason=reason,
                stated_amount_paise=raw_amount,
            )
        )

    return tuple(rulings)


if __name__ == "__main__":
    import tempfile

    from ..common.verdict import Decomposition, Verdict, VerdictFile, write_verdicts

    def refuses(fn, label: str, expect_in: str) -> str:
        try:
            fn()
        except TriageError as e:
            msg = str(e)
            assert expect_in in msg, f"{label}: message lacks {expect_in!r}\n  got: {msg}"
            return msg
        raise AssertionError(f"accepted {label}")

    GROSS, FEE, GST, CREDIT = 111_100, 2_222, 400, 108_478

    good = VerdictFile(
        seed=42, month="2026-08", matcher="fixture:selfcheck@1",
        verdicts=(
            Verdict("C0001", Outcome.RESOLVED, ("setl_0001",), ("pay_0001",), tier=1,
                    residual_paise=0, credit_amount_paise=CREDIT,
                    decomposition=Decomposition(GROSS, fee_paise=FEE, gst_paise=GST)),
            Verdict("C0002", Outcome.EXCEPTION, reason=Reason.NO_CANDIDATE),
            Verdict("C0003", Outcome.IGNORED, reason=Reason.NON_GATEWAY_CREDIT),
        ),
        wall_clock_seconds=0.02,
    )

    with tempfile.TemporaryDirectory(prefix="hisaab-triage-read-") as tmp:
        root = Path(tmp)
        written = write_verdicts(root, good)

        # --- the positive control, first: what the matcher writes, triage reads ----------
        # Without this every refusal below is consistent with a reader that rejects
        # everything, which would pass each probe and ship a queue that never loads.
        rulings = load_rulings(written)
        assert len(rulings) == 3
        assert [r.credit_id for r in rulings] == ["C0001", "C0002", "C0003"], (
            "file order must survive -- it is the bank statement's order"
        )
        assert rulings[0].outcome is Outcome.RESOLVED and rulings[0].reason is None
        assert rulings[1].is_exception and rulings[1].reason is Reason.NO_CANDIDATE
        assert rulings[2].is_dismissal and rulings[2].reason is Reason.NON_GATEWAY_CREDIT
        assert not rulings[1].is_dismissal and not rulings[2].is_exception
        # A directory works as well as a file, so a caller may pass --matches out/.
        assert load_rulings(root) == rulings

        def write_raw(name: str, doc: object) -> Path:
            p = root / name
            p.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
            return p

        base = good.as_json()

        def mutated(**changes: object) -> dict[str, object]:
            return {**base, **changes}

        def one(**changes: object) -> dict[str, object]:
            """The whole file, reduced to the resolved verdict with fields altered."""
            return mutated(verdicts=[{**base["verdicts"][0], **changes}])  # type: ignore[index]

        # --- the file itself ------------------------------------------------------------
        refuses(lambda: load_rulings(root / "nope.json"), "a missing file", "not found")
        bad = root / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        refuses(lambda: load_rulings(bad), "malformed JSON", "not valid JSON")
        refuses(lambda: load_rulings(write_raw("list.json", [])), "a top-level list",
                "JSON object")
        refuses(lambda: load_rulings(write_raw("v9.json", mutated(schema_version=9))),
                "a future schema", "v9")
        refuses(lambda: load_rulings(write_raw("v1.json", mutated(schema_version=1))),
                "a v1 file, whose residual nothing verifies", "v1")
        no_ver = {k: v for k, v in base.items() if k != "schema_version"}
        refuses(lambda: load_rulings(write_raw("nover.json", no_ver)),
                "a file that does not say what it is", "no schema_version")
        no_v = {k: v for k, v in base.items() if k != "verdicts"}
        refuses(lambda: load_rulings(write_raw("nov.json", no_v)), "no verdicts key",
                "no verdicts")
        refuses(lambda: load_rulings(write_raw("vobj.json", mutated(verdicts={}))),
                "verdicts as an object", "must be a list")
        refuses(lambda: load_rulings(write_raw("vstr.json", mutated(verdicts=["C0001"]))),
                "a verdict that is a string", "expected an object")

        # --- the three fields, each absent in turn --------------------------------------
        # Named individually rather than looped, so a slip that stops one of the three being
        # required fails here instead of being covered by another key's probe.
        for key in ("credit_id", "outcome", "reason", "credit_amount_paise"):
            entry = {k: v for k, v in base["verdicts"][0].items() if k != key}  # type: ignore[index,union-attr]
            refuses(lambda e=entry, k=key: load_rulings(
                        write_raw(f"nk_{k}.json", mutated(verdicts=[e]))),
                    f"a verdict with no {key}", f"missing key '{key}'")

        refuses(lambda: load_rulings(write_raw("cid.json", one(credit_id=""))),
                "a blank credit_id", "non-empty string")
        refuses(lambda: load_rulings(write_raw("cidn.json", one(credit_id=None))),
                "a null credit_id", "non-empty string")
        refuses(lambda: load_rulings(write_raw("out.json", one(outcome="MAYBE"))),
                "an unknown outcome", "MAYBE")
        refuses(lambda: load_rulings(write_raw("rsn.json", one(reason="BECAUSE"))),
                "an unknown reason code", "BECAUSE")

        # --- the two rules triage owns beyond the contract ------------------------------
        dup = mutated(verdicts=[base["verdicts"][0], base["verdicts"][0]])  # type: ignore[index]
        refuses(lambda: load_rulings(write_raw("dup.json", dup)), "a duplicated row",
                "cannot be two entries")
        # An EXCEPTION with no code cannot be grouped. The contract refuses it on write, so
        # this file has to be built by hand -- which is exactly the file this guard is for.
        headless = mutated(verdicts=[{**base["verdicts"][1], "reason": None}])  # type: ignore[index]
        refuses(lambda: load_rulings(write_raw("noreason.json", headless)),
                "an exception with no reason", "no group to put it in")

        # --- tolerated: an IGNORED row with no code, and unknown extra keys -------------
        # The contract allows it (only EXCEPTION must carry one), so refusing it here would
        # make triage stricter than the file it reads and would reject a legal run.
        quiet = mutated(verdicts=[{**base["verdicts"][2], "reason": None}])  # type: ignore[index]
        only = load_rulings(write_raw("quiet.json", quiet))
        assert only[0].is_dismissal and only[0].reason is None
        assert load_rulings(write_raw("extra.json", mutated(future_field=1))) == rulings
        # An empty file parses to an empty queue. That is a real state -- a month where
        # everything resolved -- and step 7's first control depends on it not raising.
        assert load_rulings(write_raw("none.json", mutated(verdicts=[]))) == ()

    print("triage/read.py self-check ok")
