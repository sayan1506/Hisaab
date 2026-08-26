"""Command-line entry point for the matcher.

    python -m hisaab.matcher --data data/ --out out/matches.json

Contract with the rest of the pipeline, the same shape ``hisaab/generator/cli.py`` and
``hisaab/scoring/cli.py`` use, for the same reasons:

  * **Line 1 of stdout is the resolved config, as JSON.** Phase 11's report header
    consumes it, and gate 9 parses it to confirm the run was configured as claimed.
  * **Exit code is the verdict on the *run*, never on the score.** 0 = verdicts written,
    1 = the input could not be read or parsed, 2 = bad usage. This CLI does not know
    whether its answers are right -- that needs the answer key, which this package may
    not read -- so there is no exit code for "scored badly". Run
    ``python -m hisaab.scoring`` for that.

``--seed`` and ``--month`` exist because ``matches.json`` carries provenance and the
matcher cannot derive it: the seed appears nowhere in ``data/``, and
``run_manifest.json`` lives under ``truth/``. The month defaults to whatever the bank
statement's own ``value_date`` column says, so a September run cannot silently claim
August; the seed defaults to the generator's default and is otherwise a claim you make.
Get either wrong and the scorer refuses to grade rather than reporting a plausible
number -- which is the intended behaviour, not an obstacle.

``--window`` is exposed so Phase 4 can widen it from the command line and watch the
number move. That feedback loop is the whole reason Phase 2's scorer exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from ..common.money import fmt
from ..common.verdict import write_verdicts
from .blocking import DEFAULT_MAX_ADJUSTMENT_PAISE, DEFAULT_WINDOW_DAYS
from .engine import MATCHER_NAME, RunSummary, run
from .load import Dataset, LoadError, load

EXIT_OK = 0
EXIT_LOAD_FAILED = 1
EXIT_USAGE = 2

#: The generator's default. Restated rather than imported -- the matcher may not import
#: ``hisaab.generator``, and a default that silently tracked the generator's would hide
#: a seed mismatch that the scorer is supposed to catch loudly.
DEFAULT_SEED = 42


def _month(value: str) -> str:
    """Validate ``YYYY-MM`` and return it unchanged.

    Kept as a string because that is what ``truth.json`` stores and what the scorer
    compares; parsing it to a tuple here would only mean formatting it back.
    """
    try:
        year_s, month_s = value.split("-")
        year, month = int(year_s), int(month_s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM, got {value!r}") from None
    if not 1 <= month <= 12:
        raise argparse.ArgumentTypeError(f"month must be 01-12, got {value!r}")
    if not 2000 <= year <= 2100:
        raise argparse.ArgumentTypeError(f"year looks wrong: {value!r}")
    return f"{year:04d}-{month:02d}"


def infer_month(dataset: Dataset) -> str:
    """The month this bank statement is about, from its own ``value_date`` column.

    The **modal** month rather than the earliest: Phase 4's ``--settlement-delay`` can
    push a late-December settlement's credit into January, and one straddling row should
    not rename the run. Ties break toward the earlier month so the answer is
    deterministic rather than dependent on ``Counter`` insertion order.
    """
    if not dataset.credits:
        raise LoadError(
            "the bank statement has no rows, so the month cannot be inferred -- "
            "pass --month YYYY-MM explicitly"
        )
    tally = Counter(f"{c.value_date.year:04d}-{c.value_date.month:02d}" for c in dataset.credits)
    best = max(tally.values())
    return min(month for month, count in tally.items() if count == best)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m hisaab.matcher",
        description=(
            "Match bank credits against gateway settlements and emit one verdict per "
            "bank row. Phase 3 is Tier 1 only: an exact join on (value_date, net_paise) "
            "inside a +/-0 business-day window. Deterministic, integer paise, no LLM on "
            "the match path."
        ),
        epilog=(
            "Score the output:\n"
            "    python -m hisaab.scoring --matches out/matches.json --truth truth/\n\n"
            "Exit 0 means verdicts were written, whatever they say. This command cannot "
            "know whether they are right -- that needs the answer key, which the "
            "matching path is not allowed to read."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data", type=Path, default=Path("data"), metavar="DIR",
                   help="directory holding the five input CSVs (default: data/)")
    p.add_argument("--out", type=Path, default=Path("out/matches.json"), metavar="PATH",
                   help="where to write matches.json (default: out/matches.json)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"the seed this data was generated with (default: {DEFAULT_SEED}). "
                        f"Carried into matches.json so the scorer can refuse a mismatch")
    p.add_argument("--month", type=_month, default=None, metavar="YYYY-MM",
                   help="the month being matched (default: inferred from the bank statement)")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW_DAYS, metavar="DAYS",
                   help=f"date window in *business* days, either side "
                        f"(default: {DEFAULT_WINDOW_DAYS}). Phase 4 widens this")
    p.add_argument("--max-adjustment", type=int, default=DEFAULT_MAX_ADJUSTMENT_PAISE,
                   metavar="PAISE",
                   help=f"amount tolerance in paise (default: "
                        f"{DEFAULT_MAX_ADJUSTMENT_PAISE}, exact). Phase 4 widens this")
    p.add_argument("--quiet", action="store_true",
                   help="print only the resolved-config JSON line")
    return p


def _utf8_stdout() -> None:
    """Make the rupee sign survive a pipe on Windows.

    The summary prints ``money.fmt`` output, and a redirected stdout on Windows defaults
    to cp1252, where that raises ``UnicodeEncodeError``. ``hisaab/scoring/cli.py`` does
    the same for the same reason.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def resolved_config(args: argparse.Namespace, month: str) -> dict[str, object]:
    """Line 1 of stdout. Everything that changes the output, and nothing that does not."""
    return {
        "matcher": MATCHER_NAME,
        "seed": args.seed,
        "month": month,
        "data_dir": str(args.data),
        "out": str(args.out),
        "window_days": args.window,
        "max_adjustment_paise": args.max_adjustment,
        "tier": 1,
    }


def _print_summary(summary: RunSummary, dataset: Dataset, written: Path) -> None:
    counts = dataset.counts()
    print(
        f"\n{summary.bank_rows} bank rows matched in "
        f"{summary.wall_clock_seconds * 1000:.0f} ms "
        f"({counts['payments']} payments, {counts['settlements']} settlements, "
        f"{counts['settlement_items']} membership rows)"
    )
    print(
        f"  window +/-{summary.window_days} business days, "
        f"amount tolerance {summary.max_adjustment_paise}p (exact)"
        if not summary.max_adjustment_paise
        else f"  window +/-{summary.window_days} business days, "
             f"amount tolerance {fmt(summary.max_adjustment_paise)}"
    )
    print(
        f"  resolved {summary.resolved}, exceptions {summary.exceptions}, "
        f"ignored {summary.ignored}"
    )
    claimed = summary.coverage_claimed
    if claimed is not None:
        print(
            f"  committed on {claimed * 100:.1f}% of rows -- this is *not* a score. "
            f"Whether those commitments are right needs the answer key:"
        )
        print("      python -m hisaab.scoring --matches "
              f"{written} --truth truth/")
    if summary.residual_nonzero:
        print(
            f"  {summary.residual_nonzero} resolved rows carry a non-zero residual -- "
            f"expected once --fees is on, a finding before that"
        )
    # The measurement behind keying on the pair rather than the bare amount.
    print(
        f"  {summary.settlements_indexed} settlements indexed, "
        f"{summary.amount_collisions} sharing a bare net amount "
        f"({'the date is what separates them' if summary.amount_collisions else 'none at this size'})"
    )
    print(f"\n  {written}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _utf8_stdout()

    if args.window < 0:
        parser.error(f"--window must be >= 0, got {args.window}")
    if args.max_adjustment < 0:
        parser.error(f"--max-adjustment must be >= 0, got {args.max_adjustment}")

    # Load before echoing the config: --month defaults to what the data says, and a
    # config line naming a month the file contradicts would be worse than none.
    # A load failure therefore prints nothing to stdout and exits 1, which is correct --
    # there is no run to describe.
    try:
        dataset = load(args.data)
        month = args.month or infer_month(dataset)
    except LoadError as e:
        print(f"CANNOT READ INPUT\n  {e}", file=sys.stderr)
        return EXIT_LOAD_FAILED

    print(json.dumps(resolved_config(args, month), ensure_ascii=False))

    run_file, summary = run(
        dataset,
        seed=args.seed,
        month=month,
        window_days=args.window,
        max_adjustment_paise=args.max_adjustment,
    )
    written = write_verdicts(args.out, run_file)

    if not args.quiet:
        _print_summary(summary, dataset, written)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
