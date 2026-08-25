"""Reads ``matches.json`` and refuses to hand on anything malformed.

This is the gate that makes the match rate mean something. A matcher which emits 58
verdicts for 60 bank rows does not get scored on 58 -- it fails, loudly, naming the two
it dropped. Without that, a rate is not "how often it was right" but "how often it was
right among the rows it chose to report", and the difference is invisible in the output.

**This module does not read the answer key, by construction.** It lives in
``hisaab.scoring`` but imports no ``truth_io`` and calls no ``load_truth``; every
expectation it checks against -- the credit IDs that must be covered, the seed, the
month -- arrives as a plain value from the caller. ``cli.py`` is the module that opens
truth and passes them down.

That split is deliberate and slightly awkward on purpose. Validating the matcher's
output is the one scoring job that has no business seeing the answers, so the module
doing it is built so that it *cannot*, and ``tools/check_isolation.py`` keeps it off the
``TRUTH_READERS`` allowlist as evidence. A future change that hands this module a
``Truth`` object will fail that gate, which is the intended outcome.

Per-verdict rules are not re-implemented here. They live on the dataclasses in
``hisaab/common/verdict.py`` and are re-run by constructing through them, so the write
path and the read path cannot drift apart. What this module adds is everything a single
verdict cannot know about itself: strict key presence, the enums, provenance agreement,
and completeness across the whole file.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ..common.verdict import (
    MATCHES_JSON,
    VERDICT_SCHEMA_VERSION,
    Outcome,
    Verdict,
    VerdictFile,
)
from ..common.reasons import Reason

#: Longest list of offending IDs to print before eliding. Enough to see the pattern,
#: few enough that a 200-row disagreement is still readable.
MAX_IDS_SHOWN = 8

#: Keys that must be **present** on every verdict, ``null`` included where the field is
#: nullable. Absent-versus-null is the branch ``truth_io.py`` refuses to allow in the
#: answer key, for the same reason: one branch per field, and one of them ends up wrong.
#: It also turns a misspelled key into a loud failure -- ``"payments_ids"`` reads as an
#: absent ``payment_ids`` and would otherwise score as an empty match.
REQUIRED_VERDICT_KEYS: tuple[str, ...] = (
    "credit_id", "outcome", "settlement_ids", "payment_ids",
    "tier", "confidence", "reason", "note", "residual_paise",
)

REQUIRED_TOP_KEYS: tuple[str, ...] = ("schema_version", "seed", "month", "matcher", "verdicts")


class VerdictError(Exception):
    """``matches.json`` is missing, malformed, or does not describe the run it claims."""


def _fmt_ids(ids: Iterable[str]) -> str:
    """Sorted, elided, so a large disagreement stays readable."""
    listed = sorted(ids)
    head = ", ".join(listed[:MAX_IDS_SHOWN])
    if len(listed) > MAX_IDS_SHOWN:
        return f"{head}, ... (+{len(listed) - MAX_IDS_SHOWN} more)"
    return head


def _require_int(obj: dict[str, object], key: str, where: str) -> int:
    v = obj.get(key)
    if isinstance(v, bool) or not isinstance(v, int):
        raise VerdictError(f"{where}: {key} must be an int, got {type(v).__name__} ({v!r})")
    return v


def _require_str(obj: dict[str, object], key: str, where: str) -> str:
    v = obj.get(key)
    if not isinstance(v, str) or not v:
        raise VerdictError(f"{where}: {key} must be a non-empty string, got {v!r}")
    return v


def _require_str_list(obj: dict[str, object], key: str, where: str) -> tuple[str, ...]:
    v = obj.get(key)
    if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
        raise VerdictError(f"{where}: {key} must be a list of non-empty strings, got {v!r}")
    return tuple(v)


def _optional_int(obj: dict[str, object], key: str, where: str) -> int | None:
    v = obj.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        raise VerdictError(f"{where}: {key} must be an int or null, got {v!r}")
    return v


def _optional_str(obj: dict[str, object], key: str, where: str) -> str | None:
    v = obj.get(key)
    if v is None:
        return None
    if not isinstance(v, str):
        raise VerdictError(f"{where}: {key} must be a string or null, got {v!r}")
    return v


def _optional_float(obj: dict[str, object], key: str, where: str) -> float | None:
    v = obj.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise VerdictError(f"{where}: {key} must be a number or null, got {v!r}")
    return float(v)


def _parse_verdict(entry: object, where: str) -> Verdict:
    if not isinstance(entry, dict):
        raise VerdictError(f"{where}: expected an object, got {type(entry).__name__}")
    missing = [k for k in REQUIRED_VERDICT_KEYS if k not in entry]
    if missing:
        raise VerdictError(
            f"{where}: missing key(s) {missing} -- every key is required on every "
            f"verdict, null where it does not apply. A missing key is usually a typo, "
            f"and a typo'd payment list would score as an empty match."
        )

    credit_id = _require_str(entry, "credit_id", where)
    where = f"{where} ({credit_id})"

    raw_outcome = entry.get("outcome")
    try:
        outcome = Outcome(raw_outcome)
    except ValueError:
        raise VerdictError(
            f"{where}: outcome {raw_outcome!r} is not one of "
            f"{[str(o) for o in Outcome]}"
        ) from None

    raw_reason = _optional_str(entry, "reason", where)
    reason: Reason | None = None
    if raw_reason is not None:
        try:
            reason = Reason(raw_reason)
        except ValueError:
            raise VerdictError(
                f"{where}: reason {raw_reason!r} is not a known code. The generator and "
                f"the matcher share one vocabulary (hisaab/common/reasons.py) so that "
                f"'did it abstain for the reason we planted?' is a count rather than a "
                f"judgement call. Known codes: {[str(r) for r in Reason]}"
            ) from None

    try:
        return Verdict(
            credit_id=credit_id,
            outcome=outcome,
            settlement_ids=_require_str_list(entry, "settlement_ids", where),
            payment_ids=_require_str_list(entry, "payment_ids", where),
            tier=_optional_int(entry, "tier", where),
            confidence=_optional_float(entry, "confidence", where),
            reason=reason,
            note=_optional_str(entry, "note", where),
            residual_paise=_optional_int(entry, "residual_paise", where),
        )
    except (ValueError, TypeError) as e:
        # The per-verdict rules live on the dataclass so the writer and the reader
        # cannot drift. Re-raised with the file position the dataclass cannot know.
        raise VerdictError(f"{where}: {e}") from e


def load_verdicts(path: Path | str) -> VerdictFile:
    """Parse ``matches.json`` from ``path`` (a file, or a directory holding one).

    Validates shape only. Whether the file describes *this* run, and whether it covers
    every bank row, are ``reconcile()``'s job -- they need expectations this module is
    deliberately unable to obtain for itself.
    """
    p = Path(path)
    if p.is_dir():
        p = p / MATCHES_JSON
    if not p.exists():
        raise VerdictError(f"matcher output not found: {p}")

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise VerdictError(f"{p} is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise VerdictError(f"{p}: expected a JSON object at the top level")

    missing = [k for k in REQUIRED_TOP_KEYS if k not in raw]
    if missing:
        raise VerdictError(f"{p}: missing required key(s) {missing}")

    version = _require_int(raw, "schema_version", str(p))
    if version != VERDICT_SCHEMA_VERSION:
        raise VerdictError(
            f"{p}: verdict schema v{version}, but this scorer reads "
            f"v{VERDICT_SCHEMA_VERSION}. Re-run the matcher or update "
            f"hisaab/common/verdict.py -- do not guess at the difference."
        )

    verdicts_raw = raw["verdicts"]
    if not isinstance(verdicts_raw, list):
        raise VerdictError(f"{p}: verdicts must be a list, got {type(verdicts_raw).__name__}")

    verdicts = tuple(
        _parse_verdict(entry, f"{p}: verdicts[{i}]") for i, entry in enumerate(verdicts_raw)
    )

    timing = raw.get("timing")
    if timing is not None and not isinstance(timing, dict):
        raise VerdictError(f"{p}: timing must be an object or absent, got {timing!r}")
    wall_clock = None
    if isinstance(timing, dict):
        wall_clock = _optional_float(timing, "wall_clock_seconds", f"{p}: timing")

    try:
        return VerdictFile(
            seed=_require_int(raw, "seed", str(p)),
            month=_require_str(raw, "month", str(p)),
            matcher=_require_str(raw, "matcher", str(p)),
            verdicts=verdicts,
            wall_clock_seconds=wall_clock,
            schema_version=version,
        )
    except ValueError as e:
        # Duplicate credit_id lands here -- VerdictFile owns that rule.
        raise VerdictError(f"{p}: {e}") from e


def reconcile(
    run: VerdictFile,
    expected_credit_ids: Iterable[str],
    expected_seed: int,
    expected_month: str,
) -> None:
    """Refuse to score unless ``run`` describes exactly the run it is being scored against.

    Every argument is a plain value, never a ``Truth`` object -- see the module
    docstring. The caller reads the answer key; this function only compares.

    Two families of failure, both fatal:

    **Wrong run.** A verdict file scored against another run's answer key produces a
    plausible number rather than a crash, which makes it the most expensive available
    bug. Seed and month are cheap to carry and cheap to check.

    **Incomplete coverage.** Every expected credit gets exactly one verdict: none
    missing, none invented, none duplicated. This is the submission checklist's
    *"matched + exceptions = total, exactly. No record is dropped"* made mechanical
    instead of promised.
    """
    if run.seed != expected_seed or run.month != expected_month:
        raise VerdictError(
            f"this verdict file describes seed {run.seed}, {run.month}, but it is being "
            f"scored against seed {expected_seed}, {expected_month}. Scoring one run "
            f"against another's answer key yields a plausible number, not an error -- "
            f"re-run the matcher against the data you are scoring."
        )

    expected = set(expected_credit_ids)
    got = [v.credit_id for v in run.verdicts]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for cid in got:
        if cid in seen:
            duplicates.add(cid)
        seen.add(cid)

    problems: list[str] = []
    if duplicates:
        problems.append(f"{len(duplicates)} duplicated: {_fmt_ids(duplicates)}")
    if missing := expected - seen:
        problems.append(f"{len(missing)} with no verdict: {_fmt_ids(missing)}")
    if invented := seen - expected:
        problems.append(f"{len(invented)} not in this run at all: {_fmt_ids(invented)}")

    if problems:
        raise VerdictError(
            f"the verdict file does not cover the run: {'; '.join(problems)}. "
            f"Expected {len(expected)} bank rows, got {len(got)} verdicts. A matcher "
            f"that omits rows must not be able to score well by dropping the hard ones, "
            f"so this is refused rather than scored on what is present."
        )


def load_and_reconcile(
    path: Path | str,
    expected_credit_ids: Iterable[str],
    expected_seed: int,
    expected_month: str,
) -> VerdictFile:
    """``load_verdicts`` then ``reconcile``. The only entry point a caller needs."""
    run = load_verdicts(path)
    reconcile(run, expected_credit_ids, expected_seed, expected_month)
    return run


if __name__ == "__main__":
    import tempfile

    from ..common.verdict import write_verdicts

    def refuses(fn, label: str, expect_in: str = "") -> str:
        try:
            fn()
        except VerdictError as e:
            msg = str(e)
            assert expect_in in msg, f"{label}: message lacks {expect_in!r}\n  got: {msg}"
            return msg
        raise AssertionError(f"accepted {label}")

    def resolved(cid: str, pid: str) -> Verdict:
        return Verdict(cid, Outcome.RESOLVED, (f"setl_{pid[4:]}",), (pid,),
                       tier=1, residual_paise=0)

    good = VerdictFile(
        seed=42, month="2026-08", matcher="fixture:selfcheck@1",
        verdicts=(
            resolved("C0001", "pay_0001"),
            Verdict("C0002", Outcome.EXCEPTION, reason=Reason.NO_CANDIDATE),
            Verdict("C0003", Outcome.IGNORED, reason=Reason.NON_GATEWAY_CREDIT),
        ),
        wall_clock_seconds=0.02,
    )
    ids = ("C0001", "C0002", "C0003")

    with tempfile.TemporaryDirectory(prefix="hisaab-verdict-io-") as tmp:
        root = Path(tmp)
        path = write_verdicts(root, good)

        # Round-trip: what the writer wrote is what the reader reads.
        back = load_verdicts(path)
        assert back.as_json() == good.as_json()
        assert back.counts() == {"RESOLVED": 1, "EXCEPTION": 1, "IGNORED": 1}
        assert back.wall_clock_seconds == 0.02
        reconcile(back, ids, 42, "2026-08")
        assert load_and_reconcile(root, ids, 42, "2026-08").matcher == "fixture:selfcheck@1"

        def write_raw(name: str, doc: object) -> Path:
            p = root / name
            p.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
            return p

        base = good.as_json()

        def mutated(**changes: object) -> dict[str, object]:
            return {**base, **changes}

        # --- wrong run ------------------------------------------------------
        msg = refuses(lambda: reconcile(back, ids, 43, "2026-08"), "the wrong seed", "seed 43")
        assert "plausible number" in msg
        refuses(lambda: reconcile(back, ids, 42, "2026-09"), "the wrong month", "2026-09")

        # --- incomplete coverage, each naming the offending ID --------------
        refuses(lambda: reconcile(back, (*ids, "C0004"), 42, "2026-08"),
                "a credit with no verdict", "C0004")
        refuses(lambda: reconcile(back, ("C0001", "C0002"), 42, "2026-08"),
                "an invented credit", "C0003")
        dup = write_raw("dup.json", mutated(verdicts=[*base["verdicts"], base["verdicts"][0]]))
        refuses(lambda: load_verdicts(dup), "a duplicate verdict", "C0001")

        # --- shape ----------------------------------------------------------
        refuses(lambda: load_verdicts(root / "nope.json"), "a missing file", "not found")
        bad_json = root / "bad.json"
        bad_json.write_text("{not json", encoding="utf-8")
        refuses(lambda: load_verdicts(bad_json), "malformed JSON", "not valid JSON")
        refuses(lambda: load_verdicts(write_raw("list.json", [])), "a top-level list",
                "JSON object")
        refuses(lambda: load_verdicts(write_raw("v9.json", mutated(schema_version=9))),
                "an unknown schema version", "v9")
        no_matcher = write_raw("anon.json", mutated(matcher=""))
        refuses(lambda: load_verdicts(no_matcher), "an unnamed matcher", "matcher")
        for key in REQUIRED_TOP_KEYS:
            doc = {k: v for k, v in base.items() if k != key}
            refuses(lambda d=doc, k=key: load_verdicts(write_raw(f"no_{k}.json", d)),
                    f"a file with no {key}", key)

        # --- per-verdict rules, re-run through the dataclass ----------------
        def one(**changes: object) -> dict[str, object]:
            return mutated(verdicts=[{**base["verdicts"][0], **changes}])

        for key in REQUIRED_VERDICT_KEYS:
            entry = {k: v for k, v in base["verdicts"][0].items() if k != key}
            refuses(lambda e=entry, k=key: load_verdicts(write_raw(f"vk_{k}.json",
                                                                  mutated(verdicts=[e]))),
                    f"a verdict with no {key}", key)
        refuses(lambda: load_verdicts(write_raw("out.json", one(outcome="MAYBE"))),
                "an unknown outcome", "MAYBE")
        refuses(lambda: load_verdicts(write_raw("rsn.json",
                                                one(outcome="EXCEPTION", reason="BECAUSE",
                                                    settlement_ids=[], payment_ids=[],
                                                    tier=None, residual_paise=None))),
                "an unknown reason code", "BECAUSE")
        refuses(lambda: load_verdicts(write_raw("empty.json", one(payment_ids=[]))),
                "RESOLVED with no payments", "C0001")
        refuses(lambda: load_verdicts(write_raw("nores.json", one(residual_paise=None))),
                "RESOLVED with no residual", "residual")
        refuses(lambda: load_verdicts(write_raw("noreason.json",
                                                one(outcome="EXCEPTION", settlement_ids=[],
                                                    payment_ids=[], tier=None,
                                                    residual_paise=None))),
                "EXCEPTION with no reason", "reason")
        refuses(lambda: load_verdicts(write_raw("float.json", one(residual_paise=1.5))),
                "a float residual", "residual_paise")
        refuses(lambda: load_verdicts(write_raw("nested.json", one(payment_ids=[["pay_0001"]]))),
                "a nested payment list", "payment_ids")
        refuses(lambda: load_verdicts(write_raw("blank.json", one(payment_ids=[""]))),
                "a blank payment id", "payment_ids")
        refuses(lambda: load_verdicts(write_raw("vlist.json", mutated(verdicts={}))),
                "verdicts as an object", "must be a list")

        # --- tolerated: absent timing, and unknown extra keys --------------
        no_timing = {k: v for k, v in base.items() if k != "timing"}
        assert load_verdicts(write_raw("notiming.json", no_timing)).wall_clock_seconds is None
        assert load_verdicts(write_raw("extra.json", mutated(future_field=1))).seed == 42

        # An empty verdict list parses; it fails at reconcile, where completeness lives.
        empty_run = load_verdicts(write_raw("none.json", mutated(verdicts=[])))
        assert empty_run.counts() == {"RESOLVED": 0, "EXCEPTION": 0, "IGNORED": 0}
        refuses(lambda: reconcile(empty_run, ids, 42, "2026-08"), "an empty verdict file",
                "3 with no verdict")

        # Elision keeps a large disagreement readable.
        many = [f"C{i:04d}" for i in range(1, 41)]
        msg = refuses(lambda: reconcile(empty_run, many, 42, "2026-08"), "40 missing", "+32 more")

    print("verdict_io.py self-check ok")
