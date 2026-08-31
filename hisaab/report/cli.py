"""Command-line entry point for the report.

    python -m hisaab.report --matches out/matches.json --metrics out/metrics.json \\
        --triage out/triage.json --out out/report.html

Assembles the four (or five) documents a run produces and renders them as one
self-contained HTML page. Reads no truth directly (``hisaab/report`` is not on
``TRUTH_READERS`` except for ``metric_block.py``'s narrow, documented import -- see that
module's docstring) and makes no network call: this is a renderer, not a decision.

**Contract with the rest of the pipeline**, the same shape every other CLI in this tree uses:

  * Exit code is the verdict on the *inputs*, never on what the report says. 0 = a page was
    written, however bad the numbers in it; 1 = a required document was missing, malformed,
    or from a different run; 2 = bad usage.
  * ``--explain`` and ``--qa`` are optional. A run with neither renders a complete page with
    a visible note in each section saying so, per plan correction (3).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .assemble import ReportError, assemble
from .html import render as render_html

EXIT_OK = 0
EXIT_UNUSABLE_INPUT = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m hisaab.report",
        description=(
            "Render a run's matches.json, scoring document, triage document and (optionally) "
            "explain/Q&A artifacts as one self-contained HTML page."
        ),
        epilog=(
            "Exit 0 means a page was written, however bad the numbers in it. Exit 1 means a "
            "required document was missing, malformed, or from a different run -- rendering "
            "would produce a plausible-looking page about a run that never happened."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--matches", type=Path, required=True, metavar="PATH",
                   help="matches.json from the matcher, or a directory holding one")
    p.add_argument("--metrics", type=Path, required=True, metavar="PATH",
                   help="the scoring --out document")
    p.add_argument("--triage", type=Path, required=True, metavar="PATH",
                   help="the triage --out document")
    p.add_argument("--explain", type=Path, default=None, metavar="PATH",
                   help="the explain artifact, if hisaab.explain was run (optional)")
    p.add_argument("--qa", type=Path, default=None, metavar="PATH",
                   help="the Q&A artifact from hisaab.explain --ask --out (optional)")
    p.add_argument("--out", type=Path, required=True, metavar="PATH",
                   help="write the rendered HTML page here")
    p.add_argument("--quiet", action="store_true",
                   help="print nothing but the written path")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        ri = assemble(args.matches, args.metrics, args.triage, args.explain, args.qa)
    except ReportError as e:
        print(f"REFUSING TO RENDER\n  {e}", file=sys.stderr)
        return EXIT_UNUSABLE_INPUT

    page = render_html(ri)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8", newline="\n")

    if not args.quiet:
        print(f"seed {ri.seed}, {ri.month}, matcher {ri.matcher}")
        print(f"explain artifact: {'present' if ri.explain is not None else 'absent'}")
        print(f"Q&A artifact:     {'present' if ri.qa is not None else 'absent'}")
    print(f"wrote {args.out.as_posix()}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
