"""The report's header: provenance line plus the one-sentence match definition.

Step 3. Quotes ``hisaab.common.verdict.MATCH_DEFINITION`` rather than composing a sentence
from the three prose sources `.plan/phase11.md` correction (2) found (``tier1.py``'s
docstring, ``ASSUMPTIONS.md`` #30 and #31) -- that constant already exists and is
self-checked (``hisaab/common/verdict.py``'s own ``__main__`` block asserts its wording), so
this module's only job is to read the three provenance-bearing documents and print one line
each, then the constant, unmodified.

Importing ``MATCH_DEFINITION`` costs nothing on ``tools/check_isolation.py``: that module's
own docstring is explicit that it "touches no truth" and "must never [be] list[ed] ...  in
``TRUTH_READERS``" -- it lives in ``hisaab/common/``, which every package including the
matcher already imports.
"""

from __future__ import annotations

from typing import Any

from ..common.verdict import MATCH_DEFINITION


def render(matches: dict[str, Any], metrics: dict[str, Any], triage: dict[str, Any]) -> str:
    """Plain text: seed, month, matcher, mode, then the match definition.

    Provenance is read from ``matches.json`` rather than the metrics document -- both carry
    seed and month, and ``assemble.assemble`` has already refused to reach this point if
    they disagreed, so either source is equally correct; ``matches.json`` is picked because
    it also carries the matcher's own name string with no reconstruction needed.
    """
    run = metrics["run"]
    mode = "clean mode" if run["clean_mode"] else f"mess[{','.join(run['flags'])}]"
    lines = [
        f"Seed {matches['seed']}, {matches['month']}, {mode}",
        f"Matcher: {matches['matcher']}",
        f"Queue: {triage['totals']['groups']} group(s), {triage['totals']['rows']} row(s)",
        "",
        MATCH_DEFINITION,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    matches = {"seed": 42, "month": "2026-08", "matcher": "tier1@0.3.0"}
    metrics = {"run": {"clean_mode": True, "flags": []}}
    triage = {"totals": {"groups": 0, "rows": 0}}

    text = render(matches, metrics, triage)
    assert "Seed 42, 2026-08, clean mode" in text
    assert "Matcher: tier1@0.3.0" in text
    assert "Queue: 0 group(s), 0 row(s)" in text
    assert MATCH_DEFINITION in text
    assert "set equality" in text and "no partial credit" in text

    messy_metrics = {"run": {"clean_mode": False, "flags": ["fx", "batching"]}}
    messy = render(matches, messy_metrics, triage)
    assert "mess[fx,batching]" in messy

    print("report/header.py self-check ok")
