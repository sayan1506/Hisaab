"""Renders the one Q&A section, from ``Answer.as_json()`` (``hisaab/explain/qa.py``).

Step 7's other half: ``ask_row()`` persists an ``Answer`` as JSON so this section reads a
recorded artifact rather than only a transcript printed to a terminal that closed. The
artifact carries no seed, month or matches path (see ``assemble.REQUIRED_QA_KEYS``'s
comment) -- it names one row by ``credit_id``, and that is what this module renders.

Optional-by-path-existence, the same as the explain artifact: a caller who never ran
``--ask`` produces a complete report with no Q&A section, not an incomplete one.
"""

from __future__ import annotations

from typing import Any

from ..common.money import fmt


def render(qa: dict[str, Any] | None) -> str:
    """The section text, or the absence note when no ``--ask`` was ever run."""
    if qa is None:
        return "Q&A: none -- no question was recorded for this run (hisaab.explain --ask)."

    lines = [
        f"Q&A -- {qa['credit_id']}",
        "",
        f"Q: {qa['question']}",
        f"A: {qa['answer']}",
    ]
    arithmetic = qa.get("arithmetic")
    if arithmetic is not None:
        terms = ", ".join(f"{t['label']}={fmt(t['paise'])}" for t in arithmetic["terms"])
        lines.append(f"   arithmetic: {terms} -> total {fmt(arithmetic['total_paise'])}")

    if qa["ok"]:
        lines.append("Verified: every citation checked out.")
    else:
        lines.append(f"Verified: {len(qa['findings'])} problem(s) found --")
        for f in qa["findings"]:
            lines.append(f"  - {f}")
    return "\n".join(lines)


if __name__ == "__main__":
    assert render(None) == "Q&A: none -- no question was recorded for this run (hisaab.explain --ask)."

    clean = {
        "credit_id": "C0001", "question": "why is this less than the gross?",
        "answer": "The gateway withheld its fee, GST on that fee and TDS.",
        "cited_row_ids": ["C0001"], "cited_amounts_paise": [21200, 424, 76, 21, 20679],
        "arithmetic": {
            "terms": [
                {"label": "gross", "paise": 21200}, {"label": "fee", "paise": -424},
                {"label": "GST", "paise": -76}, {"label": "TDS", "paise": -21},
            ],
            "total_paise": 20679,
        },
        "ok": True, "findings": [],
        "usage": {"input_tokens": 100, "output_tokens": 20,
                  "cache_read_input_tokens": 50, "cache_creation_input_tokens": 0},
    }
    text = render(clean)
    assert "Q&A -- C0001" in text
    assert "Q: why is this less than the gross?" in text
    assert "A: The gateway withheld its fee, GST on that fee and TDS." in text
    assert "gross=₹212.00" in text and "fee=-₹4.24" in text and "-> total ₹206.79" in text
    assert "Verified: every citation checked out." in text

    no_math = {**clean, "arithmetic": None}
    text2 = render(no_math)
    assert "arithmetic:" not in text2

    dirty = {**clean, "ok": False, "findings": ["id 'setl_9999' appears nowhere in this row"]}
    text3 = render(dirty)
    assert "Verified: 1 problem(s) found --" in text3
    assert "- id 'setl_9999' appears nowhere in this row" in text3

    print("report/qa.py self-check ok")
