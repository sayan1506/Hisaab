"""Command-line entry point.

    python -m hisaab.generator --seed 42 --n 60 --month 2026-08 --out data/ --truth truth/

All thirteen mess flags are declared here, every one defaulting to off, so Phases
4 through 8 flip a boolean instead of refactoring the config. Clean mode is simply
*all of them off*.

Contract with the rest of the pipeline:

  * **Line 1 of stdout is the resolved config, as JSON.** Phase 11's report header
    consumes it. (``truth/run_manifest.json`` carries the same object under
    ``config``, which is the more robust source once the report exists.)
  * **Exit code is the verdict.** 0 = written and verified, 1 = an invariant
    failed and nothing was written, 2 = bad usage. A silent bad run is worse than
    a crash, so invariants run *before* the first byte hits disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path

from ..common.money import fmt
from .config import DEFAULT_N, DEFAULT_N_BATCHED, GenConfig, MessFlags
from .emit import DETERMINISTIC_FILES, emit
from .invariants import InvariantError, check_story
from .story import build

EXIT_OK = 0
EXIT_INVARIANT_FAILED = 1
EXIT_USAGE = 2

#: One line of --help per flag, in mess-dial order, so --help reads as a
#: difficulty ramp rather than an alphabet soup.
FLAG_HELP: dict[str, str] = {
    "fees": "gateway fee + GST, so the amount never matches exactly (Phase 4)",
    "settlement_delay": "T+n settlement with weekend skew (Phase 4)",
    "batching": "many payments settle as one bank credit (Phase 5)",
    "netted_refunds": "a refund for an earlier order reduces a payout (Phase 6)",
    "reserve": "part of a payout held back, arrives later (Phase 6)",
    "tds": "tax deducted at source (Phase 6)",
    "noise_rows": "bank rows unrelated to the gateway, which must be ignored (Phase 7)",
    "unsettled": "payments captured but never paid out (Phase 7)",
    "dup_amounts": "identical date, amount and UTR, genuinely unresolvable (Phase 4b)",
    "fx": "rate moves between capture and settlement (Phase 8)",
    "rounding_edge": "amounts where fee x GST lands on a half-paisa (Phase 8)",
    "settlement_report_late": "withhold settlement_items.csv, forcing subset-sum (Phase 8)",
    "utr_patchy": "UTR missing or truncated on some rows (Phase 8)",
}


def _month(value: str) -> tuple[int, int]:
    """``2026-08`` -> ``(2026, 8)``.

    A month is required rather than defaulted from ``date.today()`` (decision #7):
    wall-clock dependence would make today's run differ from yesterday's at the
    same seed, which quietly destroys the reproducibility claim.
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
    return year, month


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m hisaab.generator",
        description=(
            "Generate synthetic payments, settlements and a bank statement, plus a "
            "truth file the matcher must never read. Phase 1 is clean mode: every "
            "mess flag off, one payment -> one settlement -> one bank credit."
        ),
        epilog=(
            "Reproducibility: the same --seed and --month produce byte-identical "
            "files. Verify with tools/repro_check.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--seed", type=int, default=42, help="master seed (default: 42)")
    p.add_argument(
        # ``None`` rather than a number, so "the user asked for 60" and "nobody said"
        # stay distinguishable. --batching may only raise a *default*: an explicit
        # --n 60 is a request and is honoured under every flag.
        "--n", type=int, default=None, metavar="N",
        help=(
            f"payment count; the track's floor is 50 *bank rows* (default: {DEFAULT_N}, "
            f"or {DEFAULT_N_BATCHED} with --batching, which settles ~1.6 payments per row)"
        ),
    )
    p.add_argument(
        "--month", type=_month, default=(2026, 8), metavar="YYYY-MM",
        help="the month to generate, e.g. 2026-08 (default: 2026-08)",
    )
    p.add_argument("--out", type=Path, default=Path("data"),
                   help="directory for the CSVs the matcher reads (default: data/)")
    p.add_argument("--truth", type=Path, default=Path("truth"),
                   help="directory for truth.json, which it must not (default: truth/)")
    p.add_argument(
        "--narration-styles", type=int, default=4, metavar="N",
        help="bank narration templates in play, 1-4; 1 is sterile (default: 4)",
    )
    # The two delays, as two numbers. Only the second is visible to the matcher's date
    # window -- see the field comments in config.py. Both are inert without
    # --settlement-delay, and GenConfig refuses a magnitude set while the flag is off
    # rather than accepting a number nothing reads.
    delays = p.add_argument_group(
        "settlement timing",
        "Both take effect only with --settlement-delay, and both count *business* days.",
    )
    delays.add_argument(
        "--settlement-delay-days", type=int, default=2, metavar="N",
        help="business days from capture to settlement, T+n (default: 2). Invisible to "
             "the matcher: the join never reads captured_at",
    )
    delays.add_argument(
        "--posting-lag-days", type=int, default=1, metavar="N",
        help="business days from settlement to the bank credit landing (default: 1). "
             "This is the delay the matcher's --window has to cover",
    )
    # Same shape as the delays above, and for the same reason: a magnitude that only one
    # flag reads. GenConfig refuses it when --dup-amounts is off rather than accepting a
    # number nothing acts on and then describing the run with it in run_manifest.json.
    planted = p.add_argument_group(
        "planted unresolvables",
        "Takes effect only with --dup-amounts.",
    )
    planted.add_argument(
        "--dup-pairs", type=int, default=2, metavar="N",
        help="colliding (date, amount, utr) pairs to plant (default: 2). Each pair costs "
             "two payments and yields two credits whose only correct verdict is an "
             "abstention. Two rather than one so the correct_abstention denominator is "
             "never 1",
    )
    p.add_argument("--quiet", action="store_true",
                   help="print only the resolved-config JSON line")

    mess = p.add_argument_group(
        "mess flags",
        "All default to off. Clean mode is all of them off. Turn them on one at a "
        "time, in this order -- each one tells you which capability is missing next.",
    )
    for f in fields(MessFlags):
        mess.add_argument(
            f"--{f.name.replace('_', '-')}", dest=f.name, action="store_true",
            help=FLAG_HELP[f.name],
        )
    mess.add_argument("--all-mess", action="store_true",
                     help="turn on every mess flag at once (not Phase 1)")
    return p


def _utf8_stdout() -> None:
    """Make the rupee sign survive a pipe on Windows.

    The summary prints ``money.fmt`` output, and a redirected stdout on Windows defaults
    to cp1252, where ``₹`` raises ``UnicodeEncodeError``. ``hisaab/matcher/cli.py`` and
    ``hisaab/scoring/cli.py`` have carried this since Phase 3; the generator did not, so
    ``python -m hisaab.generator > log.txt`` wrote all seven files and *then* died in the
    summary print -- exiting 1, the code that means "an invariant failed and nothing was
    written". The most misleading possible exit for a run that fully succeeded.

    Acceptance never caught it because ``tools/acceptance.py`` sets ``PYTHONUTF8=1`` in
    the subprocess environment, so the gates exercise the one configuration where the bug
    is invisible.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def config_from_args(args: argparse.Namespace) -> GenConfig:
    year, month = args.month
    flag_names = MessFlags.names()
    flags = (
        MessFlags.all_on()
        if args.all_mess
        else MessFlags(**{n: getattr(args, n) for n in flag_names})
    )
    # Decision 3: --batching raises the default ``n``, because ``n`` counts payments while
    # the track's 50-record floor is counted in *bank rows*, and batching makes those two
    # different numbers. At mean 1.60 payments per settlement, n=60 yields ~37 bank rows --
    # under the floor -- and n=200 yields ~125. Measured on every candidate distribution
    # (.plan/phase5.md section 1(d)).
    #
    # Resolved here rather than inside GenConfig, which is the tempting place for it: a
    # config that rewrote an explicit ``n`` would change what every caller passing one was
    # asking for. Gate 12 scores n=60 under --batching deliberately, and story.py asserts
    # exact payment counts at a given ``n``; both would silently become runs of 200.
    n = args.n
    if n is None:
        n = DEFAULT_N_BATCHED if flags.batching else DEFAULT_N

    return GenConfig(
        seed=args.seed,
        n=n,
        year=year,
        month=month,
        out_dir=args.out,
        truth_dir=args.truth,
        narration_styles=args.narration_styles,
        flags=flags,
        settlement_delay_days=args.settlement_delay_days,
        posting_lag_days=args.posting_lag_days,
        dup_pairs=args.dup_pairs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _utf8_stdout()

    try:
        cfg = config_from_args(args)
    except ValueError as e:  # GenConfig validation
        parser.error(str(e))
        return EXIT_USAGE  # unreachable; parser.error exits 2

    # Line 1 of stdout: the resolved config, as JSON.
    print(json.dumps(cfg.resolved(), ensure_ascii=False))

    started = time.perf_counter()
    story = build(cfg)
    try:
        report = check_story(story, cfg)
    except InvariantError as e:
        print(f"\nINVARIANT FAILED -- nothing written.\n  {e}", file=sys.stderr)
        return EXIT_INVARIANT_FAILED
    elapsed = time.perf_counter() - started

    result = emit(story, cfg, report, elapsed_seconds=elapsed)
    total = time.perf_counter() - started

    if args.quiet:
        return EXIT_OK

    counts = story.counts()
    mode = "clean" if cfg.clean_mode else f"mess[{','.join(cfg.flags.enabled())}]"
    print(
        f"\n{counts['payments']} records, {mode}, seed {cfg.seed}, {cfg.month_label} "
        f"-- generated in {total * 1000:.0f} ms"
    )
    print(
        f"  gross {fmt(story.total_gross_paise())}  =  "
        f"net {fmt(story.total_net_paise())}  =  "
        f"credited {fmt(story.total_credited_paise())}"
    )
    print(
        f"  invariants pass: {report['date_blocks']} date blocks, "
        f"numbering fixed points {report['numbering_fixed_points']}, "
        f"within-block alignment {report['within_block_aligned']}"
    )
    print(f"\n  {result.data_dir}{Path().anchor}  (the matcher reads these)")
    for name in ("payments.csv", "settlements.csv", "settlement_items.csv",
                 "bank_statement.csv", "refunds.csv"):
        n_rows = result.rows_written[name]
        note = "  <- header only, on purpose" if n_rows == 0 else ""
        print(f"    {name:<22} {n_rows:>4} rows  {result.hashes[name][:12]}{note}")
    print(f"\n  {result.truth_dir}{Path().anchor}  (the matcher NEVER reads these)")
    for name in ("truth.json", "run_manifest.json"):
        print(f"    {name:<22} {result.rows_written[name]:>4} rows  {result.hashes[name][:12]}")
    print(
        f"\n  reproduce: python -m hisaab.generator --seed {cfg.seed} "
        f"--n {cfg.n} --month {cfg.month_label}"
    )
    if not args.quiet:
        assert set(DETERMINISTIC_FILES) <= set(result.hashes)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
