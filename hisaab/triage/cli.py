"""Command-line entry point for the exception queue.

    python -m hisaab.triage --matches out/matches.json --data data/

This is the phase's deliverable: what a person actually opens after a run. It reads the
matcher's verdicts and the input files, and **nothing else** -- no truth, no manifest, no score.
``tools/check_isolation.py`` enforces that statically (``hisaab/triage`` is in
``MATCHER_PACKAGES``, so checks 1, 2, 6 and 7 all apply to it), which is what makes the queue
something an operator could run on their own month rather than a demo that quietly knows the
answers.

**Contract with the rest of the pipeline**, the same shape ``scoring/cli.py`` and
``generator/cli.py`` use, for the same reason -- a caller should parse a line, not scrape prose:

  * **Line 1 of stdout is the queue as JSON**, complete: every group, every row. The
    human-readable block follows after a blank line, and truncates long groups for reading.
    ``--quiet`` prints line 1 alone.
  * **Exit code is the verdict on the *inputs*, never on the queue.** 0 = a queue was produced,
    however much work is in it; 1 = the inputs could not be trusted; 2 = bad usage.

That second half is worth stating as plainly as the scorer states it: **a huge queue exits 0.**
A hundred unresolved rows is a true report about a hard month, not a failure of the tool. Exiting
non-zero on a long queue would make this unusable in exactly the situation it is for -- and would
put a person under pressure to make the number smaller rather than to clear the work. Exit 1 means
"this queue does not exist", not "this queue is bad news".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..common.money import fmt
from .group import Kind, group, total_minutes
from .hint import DISMISSAL_HINT, Hint, hint_for
from .read import TriageError, load_rulings
from .value import RankedGroup, amounts, check_total, rank, total_value

EXIT_OK = 0
EXIT_UNUSABLE_INPUT = 1
EXIT_USAGE = 2

#: Bumped when the JSON line's shape changes. Adding a key is breaking here for the same reason
#: it is in ``metrics.py``: a reader that subscripts a key it expects cannot tell "absent" from
#: "null", so a consumer pinned to v1 must fail loudly rather than silently read around a v2.
TRIAGE_SCHEMA_VERSION = 1

#: Rows listed per group in the **text** block before it truncates. The JSON line is never
#: truncated -- a machine reader wants all of it, and a person wants to see the top of each pile.
#: Same convention and same reasoning as ``report.MAX_EXCEPTIONS_LISTED``.
MAX_ROWS_LISTED = 8


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m hisaab.triage",
        description=(
            "Turn a matcher run into a work queue: what could not be reconciled, grouped by "
            "cause, ranked by money at risk, with an effort estimate and a next action for "
            "each group."
        ),
        epilog=(
            "Exit 0 means a queue was produced, however long it is. Exit 1 means the inputs "
            "could not be trusted -- a verdict for a row that is not in the statement, or a "
            "bank row with no verdict, which is what two different runs look like."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Both default to where the matcher puts things, so the common case is a bare invocation.
    # A stale default is not a silent risk here: ``check_total`` requires the join to be total
    # in both directions, so last month's matches.json against this month's data refuses.
    p.add_argument(
        "--matches", type=Path, default=Path("out/matches.json"), metavar="PATH",
        help="matches.json from the matcher, or a directory holding one "
             "(default: out/matches.json)",
    )
    p.add_argument(
        "--data", type=Path, default=Path("data"), metavar="DIR",
        help="the input files the matcher read, for the amounts (default: data/)",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="print only the JSON line, for a caller that parses rather than reads",
    )
    return p


def _utf8_stdout() -> None:
    """Make the rupee sign survive a pipe on Windows.

    A fourth copy of the same six lines that sit in the generator, matcher and scorer CLIs, and
    left as a copy deliberately: it is stdlib boilerplate with no quantity in it, so two copies
    cannot disagree about an answer the way two effort tables could. Moving it to
    ``hisaab/common/`` would touch three entry points that fifteen gates depend on to save four
    lines, which is a trade to make when something actually needs to change, not now.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass  # already wrapped, or not a real stream -- nothing to fix


def hint_of(rg: RankedGroup) -> Hint:
    """The hint for a group, chosen on ``Kind`` rather than by passing a possibly-``None`` code.

    ``hint_for(None)`` raises on purpose, so the dismissal group's advice has to be selected
    here explicitly. That keeps a code that arrived as ``None`` by accident from silently
    collecting "these were set aside as not gateway money" -- which would read as a considered
    judgement about a row nobody judged.
    """
    if rg.group.kind is Kind.DISMISSAL:
        return DISMISSAL_HINT
    return hint_for(rg.group.reason)


def as_json(ranked: tuple[RankedGroup, ...], matches: Path, data: Path) -> dict[str, object]:
    """The machine-readable queue. Complete: every group, every row, no truncation."""
    return {
        "schema_version": TRIAGE_SCHEMA_VERSION,
        "inputs": {"matches": str(matches), "data": str(data)},
        "totals": {
            "groups": len(ranked),
            "rows": sum(rg.count for rg in ranked),
            "value_paise": total_value(ranked),
            "estimated_minutes": sum(rg.total_minutes for rg in ranked),
        },
        "groups": [
            {
                "cause": rg.label,
                "kind": str(rg.group.kind),
                # The code, or null for the dismissal group -- which files rows under one
                # heading whatever code each carried, so there is no single code to name.
                "reason": str(rg.group.reason) if rg.group.reason is not None else None,
                "rows": rg.count,
                "value_paise": rg.value_paise,
                "minutes_per_row": rg.group.minutes_per_row,
                "estimated_minutes": rg.total_minutes,
                "action": hint_of(rg).action,
                "unblocks": hint_of(rg).unblocks,
                "credits": [
                    {
                        "credit_id": i.credit_id,
                        "value_paise": i.value_paise,
                        # Each row's own code, which survives the join even inside the
                        # dismissal group -- see ``value.Item.ruling``.
                        "reason": str(i.ruling.reason) if i.ruling.reason is not None else None,
                    }
                    for i in rg.items
                ],
            }
            for rg in ranked
        ],
    }


def text_report(ranked: tuple[RankedGroup, ...]) -> str:
    """The block a person reads. Heaviest group first, and its biggest row first inside it."""
    if not ranked:
        return (
            "Exception queue: empty -- every bank row was resolved or dismissed.\n"
            "Nothing needs a human."
        )

    rows = sum(rg.count for rg in ranked)
    minutes = sum(rg.total_minutes for rg in ranked)
    lines = [
        f"Exception queue: {rows} row(s) in {len(ranked)} group(s), "
        f"{fmt(total_value(ranked))} at risk, ~{minutes} min to clear",
        "",
    ]

    for n, rg in enumerate(ranked, 1):
        h = hint_of(rg)
        lines.append(
            f"{n}. {rg.label}  --  {fmt(rg.value_paise)} across {rg.count} row(s), "
            f"~{rg.total_minutes} min ({rg.group.minutes_per_row} min each)"
        )
        for i in rg.items[:MAX_ROWS_LISTED]:
            lines.append(f"     {i.credit_id:<8} {i.display:>14}")
        if rg.count > MAX_ROWS_LISTED:
            lines.append(f"     ... and {rg.count - MAX_ROWS_LISTED} more")
        lines += ["", f"   Do: {h.action}"]
        if h.unblocks is not None:
            lines.append(f"   Would stop it recurring: {h.unblocks}")
        lines.append("")

    return "\n".join(lines).rstrip()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _utf8_stdout()

    try:
        rulings = load_rulings(args.matches)
        by_id = amounts(args.data)
        # Both directions, before anything is ranked: a verdict for a row that is not in the
        # statement, and a bank row that no verdict examined. Either one means these two inputs
        # are from different runs, and every number below would be quietly wrong.
        check_total(rulings, by_id)
        ranked = rank(group(rulings), by_id, rulings)
    except TriageError as e:
        print(f"REFUSING TO BUILD A QUEUE\n  {e}", file=sys.stderr)
        return EXIT_UNUSABLE_INPUT

    # The two totals are computed by different routes on purpose: ``total_minutes`` sums the
    # ``Group``s, the JSON sums the ``RankedGroup``s. They must agree, and the join is the only
    # thing between them -- so a disagreement is the join having lost or duplicated a row.
    assert total_minutes(tuple(rg.group for rg in ranked)) == sum(
        rg.total_minutes for rg in ranked
    ), "the ranked queue and the grouped queue disagree about effort"

    print(json.dumps(as_json(ranked, args.matches, args.data), ensure_ascii=False,
                     allow_nan=False))
    if not args.quiet:
        print()
        print(text_report(ranked))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
