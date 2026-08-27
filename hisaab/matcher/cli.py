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
from .adjustments import AdjustmentReport, compare
from .blocking import DEFAULT_MAX_ADJUSTMENT_PAISE, DEFAULT_WINDOW_DAYS
from .engine import MATCHER_NAME, RunSummary, run
from .fees import DEFAULT_FEE_BPS, DEFAULT_GST_BPS, FeeSchedule
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
    # The rates are ASSUMPTIONS about a counterparty, not properties of the data
    # (ASSUMPTIONS.md #5-#9, still unverified against Razorpay's published pricing). They
    # are overridable from the command line so correcting one is a flag rather than a code
    # edit -- which is what makes "assumption" an honest label instead of a hedge on a
    # hardcoded number.
    rates = p.add_argument_group(
        "fee model",
        "Integer basis points, never float percents. 2% = 200. GST is charged on the "
        "fee, not on the gross. These rates are assumptions -- see ASSUMPTIONS.md.",
    )
    rates.add_argument(
        "--fee-bps", action="append", default=[], metavar="METHOD=BPS",
        help="override or add one method's fee rate, e.g. --fee-bps card=195. Repeatable. "
             "A method with no rate cannot be reconciled at all -- it is never assumed free",
    )
    rates.add_argument(
        "--gst-bps", type=int, default=None, metavar="BPS",
        help=f"GST on the fee (default: {DEFAULT_GST_BPS} = 18%%)",
    )
    p.add_argument("--quiet", action="store_true",
                   help="print only the resolved-config JSON line")
    return p


def schedule_from_args(args: argparse.Namespace) -> FeeSchedule:
    """Build the fee schedule, applying any ``--fee-bps`` overrides to the defaults.

    Overrides rather than replaces: passing one rate should not silently unprice the other
    three methods, which would turn a one-rate correction into a run where most rows cannot
    be explained at all. A method absent from the defaults is *added*, so a rate can be
    declared for something the table has never seen.

    Raises ``ValueError`` on malformed input; the CLI turns that into exit 2.
    """
    table = dict(DEFAULT_FEE_BPS)
    for spec in args.fee_bps:
        method, sep, bps_s = spec.partition("=")
        if not sep or not method.strip():
            raise ValueError(f"--fee-bps expects METHOD=BPS, got {spec!r}")
        try:
            bps = int(bps_s)
        except ValueError:
            raise ValueError(
                f"--fee-bps {spec!r}: {bps_s!r} is not an integer. Rates are basis "
                f"points, so 2% is 200 -- a float percent here is a real bug"
            ) from None
        if bps < 0:
            raise ValueError(f"--fee-bps {spec!r}: a negative fee rate would add money")
        table[method.strip()] = bps
    gst = DEFAULT_GST_BPS if args.gst_bps is None else args.gst_bps
    if gst < 0:
        raise ValueError(f"--gst-bps must be >= 0, got {gst}")
    if gst >= 10_000:
        raise ValueError(
            f"--gst-bps {gst} is at or above 100%, so GST would exceed the fee it is "
            f"charged on -- that is the two rates applied to the same base"
        )
    return FeeSchedule(table, gst)


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


def resolved_config(args: argparse.Namespace, month: str, schedule: FeeSchedule) -> dict[str, object]:
    """Line 1 of stdout. Everything that changes the output, and nothing that does not.

    The fee rates belong here for the same reason the window does: they change which rows
    resolve. They are also an *assumption* about a counterparty rather than a property of
    the data (ASSUMPTIONS.md #5-#9), so a run that does not state them leaves its
    coverage number uninterpretable -- a reader cannot tell a fee model that worked from
    rates that happened to match.
    """
    return {
        "matcher": MATCHER_NAME,
        "seed": args.seed,
        "month": month,
        "data_dir": str(args.data),
        "out": str(args.out),
        "window_days": args.window,
        "max_adjustment_paise": args.max_adjustment,
        "fee_bps_by_method": dict(sorted(schedule.fee_bps_by_method.items())),
        "gst_bps": schedule.gst_bps,
        "tier": 1,
    }


def _print_summary(
    summary: RunSummary,
    dataset: Dataset,
    written: Path,
    adjustments: AdjustmentReport,
) -> None:
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
    # The rates are stated next to the result, not buried in the code. A coverage number
    # produced by assumed rates is only interpretable if the assumption is on the page.
    print(f"  fee rates assumed: {summary.fee_rates}")
    if summary.unpriced:
        print(
            f"  NO RATE DECLARED for {', '.join(summary.unpriced)} -- every row using "
            f"one of those methods is unexplainable and will abstain. An unpriced method "
            f"is never assumed free"
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
    # Phase 6 step 2: the assumed rates, checked against what the settlement file itself
    # declares. It sits here rather than in the engine on purpose -- ``run()`` has already
    # committed every verdict above by the time this is computed, so the comparison cannot
    # reach a decision even by accident. ``check_isolation.py`` check 7 enforces that
    # direction; ``adjustments.py``'s docstring says why it is the whole design.
    #
    # Printed on every run, including the many where every term agrees, because the value of
    # this line is that a reader can tell "the rates were checked" from "the rates were
    # assumed" -- and a check that only speaks up when it fails is indistinguishable from one
    # that was never wired in. The same argument as stating the rates at all.
    for line in adjustments.lines():
        print(line)
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
    # Bad usage, so exit 2 and before anything is read: a malformed rate is not a property
    # of the data and there is no point loading five CSVs to reject it.
    try:
        schedule = schedule_from_args(args)
    except ValueError as e:
        parser.error(str(e))
        return EXIT_USAGE  # unreachable; parser.error exits 2

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

    print(json.dumps(resolved_config(args, month, schedule), ensure_ascii=False))

    run_file, summary = run(
        dataset,
        seed=args.seed,
        month=month,
        window_days=args.window,
        max_adjustment_paise=args.max_adjustment,
        schedule=schedule,
    )
    written = write_verdicts(args.out, run_file)

    # **Computed after the verdicts are written, and that order is the point.** This is the
    # only read of settlements.csv's declared fee/gst/tds columns in the whole matcher; every
    # verdict above was produced from independently declared rates instead. Running the
    # comparison here means there is no execution path by which a declared figure could reach
    # a resolution -- the answers are on disk before the columns are opened.
    adjustments = compare(dataset, schedule)

    if not args.quiet:
        _print_summary(summary, dataset, written, adjustments)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
