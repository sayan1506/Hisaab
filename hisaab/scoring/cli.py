"""Command-line entry point for the scorer.

    python -m hisaab.scoring --matches out/matches.json --truth truth/

This is the module that **opens the answer key**, and the only one in the scoring
package that does so on purpose rather than incidentally. It reads truth, then hands
plain values down: the credit IDs that must be covered, the seed, the month. Neither
``verdict_io`` (which validates the matcher's output) nor ``report`` (which formats the
result) ever receives a ``Truth`` object, so neither can reach the answers.
``tools/check_isolation.py`` keeps that arrangement honest by allowlisting this module
and not those two.

Contract with the rest of the pipeline, the same shape ``hisaab/generator/cli.py`` uses
for the same reasons:

  * **Line 1 of stdout is the metric block as JSON.** Phase 11's report parses it
    instead of scraping the text. The human-readable block follows on stdout after a
    blank line; ``--quiet`` prints line 1 alone.
  * **Exit code is the verdict on the *file*, never on the score.** 0 = scored,
    1 = the verdict file or answer key could not be trusted, 2 = bad usage.

That second half is decision 7 of ``.plan/phase2.md`` and it is worth stating plainly:
**a bad score exits 0.** The scorer reports, it does not judge. Phase 3's whole loop is
run the matcher, read the number, change something, run again -- and a scorer that
exited non-zero on a low score could not be used to measure a bad matcher, which is
precisely the matcher it will spend the most time measuring. Exit 1 means "this number
does not exist", not "this number is disappointing".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .metrics import MetricsError, expected_credit_ids, score
from .report import full_report
from .truth_io import TruthError, load_manifest, load_truth
from .verdict_io import VerdictError, load_and_reconcile

EXIT_OK = 0
EXIT_CONTRACT_VIOLATION = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m hisaab.scoring",
        description=(
            "Score a matcher's output against the answer key. Reports coverage and "
            "correctness as separate numbers, because they answer different questions: "
            "how often the matcher committed, and how often it was right."
        ),
        epilog=(
            "Exit 0 means the run was scored, however low the score. Exit 1 means the "
            "verdict file could not be trusted -- a dropped row, a wrong seed, a "
            "malformed record -- and no number was produced."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--matches", type=Path, required=True, metavar="PATH",
        help="matches.json from the matcher, or a directory holding one",
    )
    p.add_argument(
        "--truth", type=Path, default=Path("truth"), metavar="DIR",
        help="directory holding truth.json (default: truth/)",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="print only the JSON line, for a caller that parses rather than reads",
    )
    return p


def _utf8_stdout() -> None:
    """Make the rupee sign survive a pipe on Windows.

    ``report.py`` prints ``₹`` via ``money.fmt``, and a redirected stdout on Windows
    defaults to cp1252, where that raises ``UnicodeEncodeError``. The block exists to be
    piped into a report, so failing to encode it is a real failure mode rather than a
    theoretical one.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass  # already wrapped, or not a real stream -- nothing to fix


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _utf8_stdout()

    try:
        truth = load_truth(args.truth)
    except TruthError as e:
        print(f"ANSWER KEY UNREADABLE\n  {e}", file=sys.stderr)
        return EXIT_CONTRACT_VIOLATION

    try:
        # Plain values from here down: the IDs to cover, the seed, the month. The
        # validator never sees the answers.
        run = load_and_reconcile(
            args.matches, expected_credit_ids(truth), truth.seed, truth.month
        )
    except VerdictError as e:
        print(f"REFUSING TO SCORE\n  {e}", file=sys.stderr)
        return EXIT_CONTRACT_VIOLATION

    try:
        metrics = score(run, truth)
    except MetricsError as e:
        print(f"REFUSING TO SCORE\n  {e}", file=sys.stderr)
        return EXIT_CONTRACT_VIOLATION

    document = metrics.as_json()
    # Tie the number to the bytes it was measured on. Best-effort: the manifest is
    # provenance, not input, so a run without one still scores -- but the key is always
    # present so Phase 11 never has to branch on whether it exists.
    document["data"] = {"bank_statement_sha256": _bank_hash(args.truth)}

    print(json.dumps(document, ensure_ascii=False, allow_nan=False))
    if not args.quiet:
        print()
        print(full_report(metrics))
    return EXIT_OK


def _bank_hash(truth_dir: Path) -> str | None:
    """The committed hash of ``bank_statement.csv``, from ``run_manifest.json``.

    Which data a score was measured on is exactly the question a judge asks when two
    numbers disagree, and the manifest already records it. Absent manifest -> ``None``
    rather than an error: the score is still valid, it is just less traceable.
    """
    try:
        manifest = load_manifest(truth_dir)
    except TruthError:
        return None
    hashes = manifest.get("file_sha256")
    if not isinstance(hashes, dict):
        return None
    value = hashes.get("bank_statement.csv")
    return value if isinstance(value, str) else None


if __name__ == "__main__":
    raise SystemExit(main())
