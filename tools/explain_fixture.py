"""Phase 10 step 2 -- the frozen prompt fixture, and the command that regenerates it.

    python tools/explain_fixture.py            # rebuild fixtures/explain/*.json
    python tools/explain_fixture.py --check    # rebuild and diff; drift fails loudly

The LLM layer needs input it can be tested against without a network call, and the
temptation is to hand-write a few plausible exception rows. That fixture would be a
fiction: hand-written rows drift from what the matcher actually emits, and every
assertion built on them would pass while testing nothing. So this fixture is a
**recording**. Every byte of it comes from running the real generator, the real matcher
and the real queue, and ``--check`` re-runs all three and refuses a mismatch.

**Why this lives in tools/ and not in hisaab/explain/.** Building a cell means running
the generator, and ``hisaab/explain`` is on ``check_isolation.MATCHER_PACKAGES``, so
check 6 forbids it from importing the generator at all. The constraint is doing its job
here rather than being worked around: the model layer must not be able to reach the code
that knows the fee rates and the narration templates, so the thing that *builds* its
test input lives outside it. ``tools/`` is not on the matching path.

**Two cells, not two seeds, and that is a measured correction.** The natural instinct is
one seed carrying every reason code. Measured across seeds 1/2/3 x n=200/1000 under
``--all-mess``: **every cell tops out at 7 of 13**, so a second seed buys nothing. The
eighth code needs a different *flag set* -- ``AMBIGUOUS_DUPLICATE_AMOUNT`` requires
``--dup-amounts``, which ``--all-mess`` excludes by construction (four separate
exclusivity rules, ``ASSUMPTIONS.md`` #24f). Hence two cells with different flags, and
**8 of 13** stated openly with the five absentees named in the artifact itself.

**What each row carries, and why it is a distilled record rather than a copy of
matches.json.** The fixture is exactly what the model is shown: the reason code, the
matcher's own note, and the bank amount. Freezing the whole verdict file instead would
mean the prompt builder re-derives its input at run time, and then what was reviewed is
not what was sent. Fields are assembled from an **allowlist**, so nothing can leak in by
accident -- notably nothing from ``truth.json``, which is generated into a temp directory
and discarded unread. The fixture is built from ``data/`` and ``matches.json`` only, which
is the same diet the product runs on.

**The note is the find that shaped this file.** The triage queue's JSON carries
``credit_id``, ``value_paise`` and ``reason`` and **drops the matcher's note** -- Phase 9
did that deliberately (``triage/hint.py``: the note is evidence, the hint is the next
action). But the note is the only populated field on an exception row: measured 295/295
non-empty, mean 569 chars, 98% naming a settlement id and 90% carrying paise figures. A
model handed the queue alone would get a code and an amount with none of the evidence, so
the fixture reads ``matches.json`` for notes and the queue for grouping and rank.

**A third section, ``resolved_sample``, for the Q&A half of step 7.** ``qa.py`` answers
questions about RESOLVED rows, which the triage queue never carries -- it exists to hold
exceptions. So this file also freezes a handful of RESOLVED verdicts from the ``all_mess``
cell, chosen for a real deduction (``fee_paise``, ``gst_paise`` and ``tds_paise`` all
non-zero) so the arithmetic check in ``qa.verify_answer`` has more than one term to close
over. Same allowlist discipline as the exception rows: five fields, not the whole verdict.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURE_DIR = ROOT / "fixtures" / "explain"

#: The posting lag. Every cell here runs a delay flag, and at ``--window 0`` a delayed run
#: scores 0% coverage -- see ASSUMPTIONS.md #15b.
WINDOW = 1

MONTH = "2026-08"


class FixtureError(Exception):
    """The fixture could not be built, or no longer matches what is committed."""


#: ``(label, seed, n, generator flags)``. Two cells because no single flag set reaches
#: more than 7 of the 13 codes; see the module docstring.
CELLS: tuple[tuple[str, int, int, tuple[str, ...]], ...] = (
    # 7 codes. --all-mess resolves to eleven composable flags (rounding_edge is
    # undeclared, dup_amounts is mutually exclusive with four others).
    ("all_mess", 1, 1000, ("--all-mess",)),
    # The eighth code, and the only cell that can carry it.
    ("dup_amounts", 1, 200, ("--fees", "--settlement-delay", "--dup-amounts")),
)

#: Why the remaining five codes appear in no cell. Stated in the artifact rather than
#: left as a silent gap, because a prompt suite built by iterating ``Reason`` would carry
#: five cases no run can produce and they would pass vacuously -- which is Phase 8's
#: "three codes producerless in three different senses" recurring for the fourth time.
#:
#: The five fail for five *different* reasons, and that is the point of writing them out:
#: "absent" reads like one problem and is not.
ABSENT_CODES: dict[str, str] = {
    "AMBIGUOUS_ADJUSTMENT": (
        "Unreachable at the declared rates, not structurally dead. It needs two rate "
        "schedules that close the same gap, which takes a --fee-bps override; "
        "matcher/tier1.py says so in its own comment and points at the measurement."
    ),
    "MEMBERSHIP_UNDECLARED": (
        "Needs a Tier 2 candidate pool above MAX_POOL (80). Did not appear at n=1000, so "
        "provoking it takes a size or density these cells do not reach -- and raising the "
        "cap to stop it binding makes gate 12 fail, so the bound is deliberately "
        "mid-range."
    ),
    "CREDIT_MISSING": (
        "Has no construction site at all. engine.py emits exactly one verdict per bank "
        "row and asserts it, so a settlement with no credit has no verdict slot to carry "
        "the code."
    ),
    "SETTLEMENT_MISSING": (
        "Constructed in tier1.py, but only on a branch load.py already refuses (a "
        "settlement citing an unknown payment). It is defence against that check being "
        "weakened rather than a code that fires."
    ),
    "ROUNDING_DRIFT": (
        "Producerless as a direct consequence of declining --rounding-edge, the one "
        "declared flag that was never implemented. Phase 8 was the last phase that adds "
        "flags, so it was the last that could have given this code a producer."
    ),
}


def _run(argv: list[str], label: str) -> str:
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr)[-1800:]
        raise FixtureError(f"{label} exited {proc.returncode}:\n{tail}")
    return proc.stdout


def _commands(seed: int, n: int, flags: tuple[str, ...]) -> list[str]:
    """The three commands that reproduce a cell, recorded verbatim in the artifact.

    Written with placeholder paths rather than the temp directory this run used: the
    point is that a reader can re-run them, and an absolute path from someone else's
    machine is noise that also makes the file machine-specific.
    """
    return [
        f"python -m hisaab.generator --seed {seed} --n {n} --month {MONTH} "
        f"--out data/ --truth truth/ {' '.join(flags)}",
        f"python -m hisaab.matcher --data data/ --out out/matches.json --window {WINDOW}",
        "python -m hisaab.triage --matches out/matches.json --data data/ --quiet",
    ]


def build_cell(label: str, seed: int, n: int, flags: tuple[str, ...]) -> dict[str, object]:
    """Generate, match and triage one cell, and distil the result into prompt input.

    Everything happens in a temp directory: the committed ``data/`` and ``out/`` belong to
    the seed-42 clean run that gates 4 and 8 audit, and a fixture build must not touch
    them. ``truth/`` is written there too and never read -- the fixture is built from the
    same inputs the product sees.
    """
    from hisaab.triage.value import amounts

    with tempfile.TemporaryDirectory() as td:
        scratch = Path(td)
        data, truth = scratch / "data", scratch / "truth"
        matches = scratch / "matches.json"

        _run(
            [sys.executable, "-m", "hisaab.generator", "--seed", str(seed), "--n", str(n),
             "--month", MONTH, "--out", str(data), "--truth", str(truth), "--quiet", *flags],
            f"generate {label}",
        )
        _run(
            [sys.executable, "-m", "hisaab.matcher", "--data", str(data),
             "--out", str(matches), "--window", str(WINDOW), "--quiet"],
            f"match {label}",
        )
        queue_text = _run(
            [sys.executable, "-m", "hisaab.triage", "--matches", str(matches),
             "--data", str(data), "--quiet"],
            f"triage {label}",
        )

        queue = json.loads(queue_text)
        verdicts = json.loads(matches.read_text(encoding="utf-8"))["verdicts"]
        bank = amounts(data)

    # The note, by credit id. Read from the verdict file because the queue drops it.
    notes = {v["credit_id"]: (v.get("note") or "") for v in verdicts}
    outcomes = {v["credit_id"]: v["outcome"] for v in verdicts}

    # RESOLVED rows carrying every deduction term, for qa.py's arithmetic check -- picked
    # deterministically (sorted by credit_id, capped) so the frozen sample does not depend
    # on dict order. Only ``all_mess`` is expected to have any: --dup-amounts alone need not
    # produce a fee+GST+TDS row, and an empty list here is a fact about that cell, not a bug.
    resolved_sample = sorted(
        (
            v for v in verdicts
            if v.get("outcome") == "RESOLVED"
            and v.get("decomposition")
            and all(v["decomposition"].get(f) for f in ("fee_paise", "gst_paise", "tds_paise"))
        ),
        key=lambda v: v["credit_id"],
    )[:3]

    groups: list[dict[str, object]] = []
    for group in queue["groups"]:
        rows: list[dict[str, object]] = []
        for credit in group["credits"]:
            cid = credit["credit_id"]
            if cid not in bank:
                raise FixtureError(
                    f"{label}: {cid} is in the queue but not in the bank statement. The "
                    f"amount a citation is checked against comes from the statement, so a "
                    f"row without one cannot be verified at all."
                )
            # Allowlist, deliberately: four fields, each named. Nothing is copied
            # wholesale, so no field from a verdict or a truth file can arrive by
            # accident in a later refactor.
            rows.append({
                "credit_id": cid,
                "reason": credit["reason"],
                "outcome": outcomes[cid],
                "bank_amount_paise": bank[cid],
                "note": notes[cid],
            })
        groups.append({
            "cause": group["cause"],
            "kind": group["kind"],
            "reason": group["reason"],
            "rows": group["rows"],
            "value_paise": group["value_paise"],
            "minutes_per_row": group["minutes_per_row"],
            "estimated_minutes": group["estimated_minutes"],
            "action": group["action"],
            "unblocks": group["unblocks"],
            "credits": rows,
        })

    codes: dict[str, int] = {}
    for group in groups:
        for row in group["credits"]:  # type: ignore[index]
            if row["reason"]:  # type: ignore[index]
                codes[row["reason"]] = codes.get(row["reason"], 0) + 1  # type: ignore[index]

    return {
        # No absolute paths anywhere: the queue's own ``inputs`` block carries the temp
        # directory this build used, and freezing that would make the fixture differ on
        # every machine while the data underneath was identical.
        "provenance": {
            "label": label,
            "seed": seed,
            "n": n,
            "flags": list(flags),
            "window_days": WINDOW,
            "month": MONTH,
            "commands": _commands(seed, n, flags),
        },
        "totals": queue["totals"],
        "codes": dict(sorted(codes.items())),
        "groups": groups,
        # Allowlist again, deliberately: qa.py's own docstring lists exactly these fields
        # (credit_id, outcome, credit_amount_paise, residual_paise, decomposition, tier,
        # payment_ids, settlement_ids) as what it reads. Nothing from truth.json is in scope
        # to leak here in the first place -- these come from matches.json alone.
        "resolved_sample": [
            {
                "credit_id": v["credit_id"],
                "outcome": v["outcome"],
                "credit_amount_paise": v["credit_amount_paise"],
                "residual_paise": v["residual_paise"],
                "tier": v["tier"],
                "payment_ids": v["payment_ids"],
                "settlement_ids": v["settlement_ids"],
                "decomposition": v["decomposition"],
            }
            for v in resolved_sample
        ],
    }


def build_all() -> dict[str, dict[str, object]]:
    return {label: build_cell(label, seed, n, flags) for label, seed, n, flags in CELLS}


def coverage(cells: dict[str, dict[str, object]]) -> dict[str, object]:
    """The 8-of-13 statement, computed rather than asserted, plus the five absentees.

    Computed from the cells themselves so it cannot drift into a claim: if a cell stops
    producing a code, this block changes and ``--check`` fails.
    """
    from hisaab.common.reasons import Reason

    declared = sorted(r.value for r in Reason)
    present: dict[str, list[str]] = {}
    for label, cell in cells.items():
        for code in cell["codes"]:  # type: ignore[union-attr]
            present.setdefault(code, []).append(label)

    absent = [c for c in declared if c not in present]
    unexplained = [c for c in absent if c not in ABSENT_CODES]
    if unexplained:
        raise FixtureError(
            f"these codes appear in no cell and no reason is recorded for them: "
            f"{unexplained}.\n"
            f"  Every absence needs a stated cause. A prompt suite built by iterating "
            f"Reason would carry them as cases no run can produce, and they would pass "
            f"vacuously -- which has now happened four times in this project, in four "
            f"different senses. Add an entry to ABSENT_CODES saying which sense this is."
        )
    stale = [c for c in ABSENT_CODES if c in present]
    if stale:
        raise FixtureError(
            f"ABSENT_CODES claims these are unreachable, but a cell produced them: "
            f"{stale}. The explanation is now wrong; delete the entry rather than "
            f"leaving prose that contradicts the artifact next to it."
        )

    return {
        "declared": len(declared),
        "covered": len(present),
        "statement": f"{len(present)} of {len(declared)} reason codes",
        "present": {c: sorted(present[c]) for c in sorted(present)},
        "absent": {c: ABSENT_CODES[c] for c in absent},
    }


def document(cells: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "what": (
            "Frozen prompt input for hisaab/explain: the exception queue plus the "
            "matcher's per-row note, recorded from real runs. Built from data/ and "
            "matches.json only -- no truth.json, no network."
        ),
        "regenerate": "python tools/explain_fixture.py",
        "verify": "python tools/explain_fixture.py --check",
        "coverage": coverage(cells),
        "cells": cells,
    }


def path_for(label: str) -> Path:
    return FIXTURE_DIR / f"{label}.json"


def _serialise(doc: dict[str, object]) -> str:
    # Sorted keys and a trailing newline: the file is diffed by --check and read by a
    # human in review, and both want a stable byte order.
    return json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def write(doc: dict[str, object]) -> Path:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    target = FIXTURE_DIR / "fixture.json"
    target.write_text(_serialise(doc), encoding="utf-8")
    return target


def load() -> dict[str, object]:
    target = FIXTURE_DIR / "fixture.json"
    if not target.exists():
        raise FixtureError(
            f"{target.relative_to(ROOT).as_posix()} is missing. Build it with "
            f"`python tools/explain_fixture.py`."
        )
    return json.loads(target.read_text(encoding="utf-8"))


def check() -> None:
    """Rebuild every cell and require the result to match what is committed.

    This is what makes the fixture a recording rather than a file someone once wrote.
    A drift here means one of two things, and the message says so: either the matcher's
    output changed (and the fixture must be rebuilt deliberately, in a diff a reviewer can
    read), or the build is no longer reproducible -- which would be a much larger problem,
    since this repo's reproducibility claim rests on the same machinery.
    """
    committed = load()
    fresh = document(build_all())
    if _serialise(fresh) == _serialise(committed):
        cov = fresh["coverage"]
        print(
            f"fixture: rebuilt byte-identical to what is committed "
            f"({cov['statement']})"  # type: ignore[index]
        )
        return

    # Name the first real difference rather than dumping two documents.
    diffs: list[str] = []
    for label in sorted(set(fresh["cells"]) | set(committed.get("cells", {}))):  # type: ignore[arg-type]
        a = committed.get("cells", {}).get(label)  # type: ignore[union-attr]
        b = fresh["cells"].get(label)  # type: ignore[union-attr]
        if a is None:
            diffs.append(f"cell {label!r} is new")
        elif b is None:
            diffs.append(f"cell {label!r} disappeared")
        elif a.get("totals") != b.get("totals"):
            diffs.append(f"cell {label!r} totals: committed {a['totals']} vs fresh {b['totals']}")
        elif a.get("codes") != b.get("codes"):
            diffs.append(f"cell {label!r} codes: committed {a['codes']} vs fresh {b['codes']}")
        elif a != b:
            diffs.append(f"cell {label!r} differs inside its rows (note or amount text)")
    if committed.get("coverage", {}).get("statement") != fresh["coverage"]["statement"]:  # type: ignore[union-attr,index]
        diffs.append(
            f"coverage: committed "
            f"{committed.get('coverage', {}).get('statement')!r} vs fresh "  # type: ignore[union-attr]
            f"{fresh['coverage']['statement']!r}"  # type: ignore[index]
        )
    raise FixtureError(
        "the rebuilt fixture does not match fixtures/explain/fixture.json:\n    "
        + "\n    ".join(diffs or ["the documents differ but no cell-level cause was found"])
        + "\n  Either the matcher's output changed -- rebuild with "
        "`python tools/explain_fixture.py` so the change lands in a reviewable diff -- "
        "or the build stopped being reproducible, which is the larger problem."
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build or verify the explain fixture.")
    p.add_argument(
        "--check", action="store_true",
        help="rebuild and compare against the committed fixture instead of writing it",
    )
    args = p.parse_args(argv)
    try:
        if args.check:
            check()
            return 0
        doc = document(build_all())
        target = write(doc)
        cov = doc["coverage"]
        size = target.stat().st_size
        print(f"wrote {target.relative_to(ROOT).as_posix()}  ({size:,} bytes)")
        print(f"  {cov['statement']}")  # type: ignore[index]
        for label, cell in doc["cells"].items():  # type: ignore[union-attr]
            t = cell["totals"]
            print(
                f"  cell {label:<12} {t['rows']:>4} rows in {t['groups']} group(s), "
                f"{len(cell['codes'])} code(s)"
            )
        print(f"  absent, with a stated reason each: {len(cov['absent'])}")  # type: ignore[index]
    except FixtureError as e:
        print(f"FIXTURE FAILED\n  {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
