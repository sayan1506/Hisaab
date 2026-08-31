"""Assembles the five sections into one self-contained HTML page. Stdlib only.

`.plan/phase11.md` §0 measured that no templating library is installed and none is needed:
five sections of mostly tabular data over small JSON documents is well within f-string
territory, and pulling one in would add the project's second dependency
(``pyproject.toml``'s ``dependencies = []``) for a job stdlib string formatting already does.
So this module is plain ``html.escape`` plus ``<pre>`` blocks around the text each section
module already renders -- no new markup vocabulary, no partial templates, one function.

**Reproducibility.** Two renders of the same input documents must be byte-identical outside
one identifiable line -- the generated-at timestamp, which is the only non-deterministic
value this module introduces (every section module it calls is already pure over its input).
That line is marked with an HTML comment (``<!-- generated-at -->``) so Step 9's gate can
locate and strip it before comparing two renders, rather than guessing at a line number.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any

from . import exceptions as exceptions_mod
from . import header as header_mod
from . import matched as matched_mod
from . import metric_block as metric_block_mod
from . import qa as qa_mod
from .assemble import ReportInput

#: The marker Step 9's gate greps for, to find and strip the one non-deterministic line
#: before comparing two renders byte-for-byte.
GENERATED_AT_MARKER = "generated-at"


def _pre(text: str) -> str:
    """One section's plain text, escaped and wrapped -- no syntax highlighting, no markup."""
    return f"<pre>{html.escape(text)}</pre>"


def _section(title: str, body: str) -> str:
    return f"<section>\n<h2>{html.escape(title)}</h2>\n{body}\n</section>"


def render(ri: ReportInput, *, now: datetime | None = None) -> str:
    """The full page. ``now`` is injectable for the self-check and for Step 9's gate."""
    ts = (now or datetime.now(timezone.utc)).isoformat()

    sections = [
        _section("Header", _pre(header_mod.render(ri.matches, ri.metrics, ri.triage))),
        _section("Metric block", _pre(metric_block_mod.render(ri.metrics))),
        _section("Exception queue", _pre(exceptions_mod.render(ri.triage, ri.explain))),
        _section("Matched records", _pre(matched_mod.render(ri.matches))),
        _section("Q&A", _pre(qa_mod.render(ri.qa))),
    ]

    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>Hisaab report -- seed {ri.seed}, {ri.month}</title>\n"
        "<style>\n"
        "body { font-family: monospace; margin: 2rem; max-width: 100ch; }\n"
        "pre { white-space: pre-wrap; }\n"
        "section { margin-bottom: 2rem; border-top: 1px solid #ccc; padding-top: 1rem; }\n"
        "footer { color: #888; font-size: 0.85em; }\n"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        f"<h1>Hisaab report -- seed {ri.seed}, {ri.month}, matcher {html.escape(ri.matcher)}</h1>\n"
        + "\n".join(sections)
        + f"\n<footer><!-- {GENERATED_AT_MARKER} -->rendered {html.escape(ts)}</footer>\n"
        "</body>\n"
        "</html>\n"
    )


def strip_generated_at(page: str) -> str:
    """The page with its one non-deterministic line blanked, for a byte comparison.

    Used by Step 9's gate. Splits on the marker's own comment rather than a fixed line
    number, so a future reflow of the footer cannot silently make the comparison too loose
    or too strict without also moving the thing it is supposed to find.
    """
    marker = f"<!-- {GENERATED_AT_MARKER} -->"
    if marker not in page:
        raise ValueError(f"page has no {marker!r} -- render() did not mark its timestamp")
    before, _, after_marker = page.partition(marker)
    # Blank out the rest of that one line only -- everything after the footer's closing tag
    # on the same line must still compare, and there is exactly one line here to touch.
    line_end = after_marker.find("\n")
    return before + marker + after_marker[line_end:]


if __name__ == "__main__":
    matches = {
        "schema_version": 2, "seed": 42, "month": "2026-08", "matcher": "tier1@0.3.0",
        "timing": {"wall_clock_seconds": 0.02}, "verdicts": [],
    }
    metrics = {
        "schema_version": 4,
        "run": {"seed": 42, "month": "2026-08", "clean_mode": True, "flags": [], "matcher": "tier1@0.3.0"},
        "timing": {"wall_clock_seconds": 0.01},
        "totals": {"bank_rows": 0, "payments": 0, "gateway_credits": 0, "non_gateway_credits": 0,
                   "planted_unresolvable": 0},
        "cells": {str(c): 0 for c in metric_block_mod.Cell},
        "rates": {}, "exceptions": {"count": 0, "value_paise": 0, "estimated_minutes": 0},
        "dismissals": {"count": 0, "estimated_minutes": 0},
        "decomposition": {"checked": 0, "mismatches": 0},
    }
    triage = {
        "schema_version": 1, "inputs": {"matches": "matches.json", "data": "data"},
        "totals": {"groups": 0, "rows": 0, "value_paise": 0, "estimated_minutes": 0}, "groups": [],
    }
    ri = ReportInput(matches=matches, metrics=metrics, triage=triage, explain=None, qa=None)

    fixed = datetime(2026, 8, 31, 12, 0, 0, tzinfo=timezone.utc)
    page = render(ri, now=fixed)

    assert page.startswith("<!doctype html>")
    assert "<title>Hisaab report -- seed 42, 2026-08</title>" in page
    assert "tier1@0.3.0" in page
    assert "Header" in page and "Metric block" in page and "Exception queue" in page
    assert "Matched records" in page and "Q&amp;A" in page
    assert "set equality" in page  # the match definition, quoted via header
    assert "Q&amp;A: none" in page  # escaped -- the section body goes through html.escape
    assert "matches.json carries no verdicts" in page
    assert "empty -- every bank row was resolved or dismissed" in page
    assert f"<!-- {GENERATED_AT_MARKER} -->" in page
    assert "2026-08-31T12:00:00" in page

    # --- truth-vocabulary scoping, measured rather than assumed ------------------------
    #
    # Two words the plan's §3 criterion names as banned -- "tier" and "resolvable" --
    # legitimately appear in a real report's own text, for two different reasons, and
    # Step 9's gate has to scope its grep around both rather than pass on this fixture and
    # fail on the first real run:
    #
    #   * ``tier`` is matcher-side vocabulary (see this module's own docstring): a real
    #     ``Verdict.as_json()`` carries it, and "which hypothesis resolved this row" leaks
    #     nothing about the answer key.
    #   * ``resolvable`` is different and was found by running this self-check, not
    #     anticipated by the plan: ``hisaab/scoring/report.py``'s ``metric_block()`` --
    #     shipped, audited Phase 9 code this module is required to quote verbatim -- prints
    #     a fixed caption on its "Missed" line, ``_line("Missed", ..., "resolvable, but it
    #     abstained")``. That caption is static prose explaining what the MISSED cell
    #     counts, not a per-row fact from ``truth.json`` -- it says the same words on every
    #     run regardless of which rows were actually resolvable -- but it does contain the
    #     bare word, so a literal "grep for resolvable" gate would fail on every real
    #     report, not just a leaking one. Rewording it is out of scope: this phase quotes
    #     ``metric_block()`` unmodified rather than re-deriving its text (step 4), and
    #     changing Phase 9's own shipped module is not this phase's job.
    #
    # Word-boundary matching (not substring) is still required regardless: "planted
    # unresolvable" and "planted-unresolvable rows" both contain "resolvable" as a
    # substring, and neither is even the caption above -- they are a third, separate
    # occurrence, also static and also safe.
    import re

    lowered = page.lower()
    assert re.search(r"\btruth\.json\b", lowered) is None
    assert re.search(r"\btrue_", lowered) is None
    # "resolvable" is not asserted absent here -- see the comment above. Step 9's actual
    # gate must scope its grep to exclude this exact caption, by string rather than by
    # word, so a future truth-side leak elsewhere still fails it.
    assert "unresolvable" in lowered, (
        "this fixture's own metric block should still say 'planted unresolvable in clean "
        "mode' -- if it does not, the vocabulary this test is distinguishing has changed"
    )
    assert "resolvable, but it abstained" in page, (
        "metric_block()'s Missed-line caption should still be present verbatim -- if the "
        "wording changed, the scoping comment above needs to be re-checked against it"
    )

    # --- reproducibility: two renders differ only in the generated-at line ------------
    later = datetime(2026, 8, 31, 13, 30, 0, tzinfo=timezone.utc)
    page2 = render(ri, now=later)
    assert page != page2, "a later timestamp must actually change something"
    assert strip_generated_at(page) == strip_generated_at(page2), (
        "two renders of the same input must be byte-identical once the timestamp is stripped"
    )

    # --- the marker-based strip is robust to content, not just position ---------------
    try:
        strip_generated_at("<html>no marker here</html>")
    except ValueError as e:
        assert "generated-at" in str(e)
    else:
        raise AssertionError("accepted a page with no marker")

    print("report/html.py self-check ok")
