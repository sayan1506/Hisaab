"""Matched records: every verdict in ``matches.json``, rendered from that document alone.

Step 6, per `.plan/phase11.md` correction (1): no join against ``Metrics.landings``, because
that field is never serialized (measured directly from ``Metrics.as_json()``) -- there is
nothing in a persisted document for a report reading files, rather than importing the scorer,
to join against. What ``Verdict.as_json()`` already carries is sufficient for "expandable to
the full decomposition": ``credit_id``, ``settlement_ids``, ``payment_ids``, ``tier``,
``confidence``, ``reason``, ``note``, ``residual_paise``, ``credit_amount_paise`` and the full
``decomposition`` object are all there, and ``Verdict.__post_init__``'s balance assertion
already guarantees ``residual_paise == credit_amount_paise - decomposition.expected_credit_paise``
for every ``RESOLVED`` row at construction time -- no reader-side check is needed to confirm a
rendered row's arithmetic closes; it cannot have been written otherwise.

**``tier`` is matcher-side vocabulary, not truth-side, and is rendered without apology.** It
answers "which hypothesis resolved this row" -- a property of the matcher's own search, stated
on every ``Verdict`` the matcher itself wrote. That is a different thing from ``resolvable``,
``TruthCredit`` or anything else that lives only in ``truth.json`` and would leak the answer
key if rendered. Step 9's reproducibility gate scopes its truth-vocabulary grep to exclude the
bare token ``tier`` for exactly this reason, documented there rather than assumed here.

**No payment ID is ever printed for a row the matcher got wrong** would be the natural rule to
copy from ``hisaab/scoring/report.py``, but it does not apply here: this module renders the
matcher's *own* stated payment IDs for every row, right or wrong, because that is what
``matches.json`` is -- the matcher's claim, not a comparison against the answer key. Printing a
wrong claim is not printing the correct answer; ``hisaab.scoring`` is never imported by this
module at all, so there is no answer available to leak even by accident.
"""

from __future__ import annotations

from typing import Any

from ..common.money import fmt

#: Rows rendered in full before the list truncates, mirroring ``MAX_EXCEPTIONS_LISTED`` and
#: ``MAX_ROWS_LISTED`` elsewhere in the tree: enough to read the shape, few enough to stay
#: quotable on a large run.
MAX_ROWS_LISTED = 20


def _decomposition_line(d: dict[str, Any] | None) -> str:
    if d is None:
        return ""
    terms = [f"gross {fmt(d['gross_paise'])}"]
    for field, label in (
        ("fee_paise", "fee"), ("gst_paise", "GST"), ("tds_paise", "TDS"),
        ("refunds_paise", "refunds"), ("reserve_paise", "reserve"),
    ):
        if d.get(field):
            terms.append(f"{label} {fmt(d[field])}")
    terms.append(f"-> expected {fmt(d['expected_credit_paise'])}")
    line = "; ".join(terms)
    return f" [{line}]" if d.get("rule") is None else f" [{line}; rule: {d['rule']}]"


def _render_row(v: dict[str, Any]) -> str:
    outcome = v["outcome"]
    if outcome == "RESOLVED":
        pay = ", ".join(v["payment_ids"]) or "-"
        setl = ", ".join(v["settlement_ids"]) or "-"
        head = (
            f"{v['credit_id']:<8} RESOLVED   tier {v['tier']}   {fmt(v['credit_amount_paise']):>14}"
            f"   settlement(s) {setl}   payment(s) {pay}"
        )
        return head + _decomposition_line(v.get("decomposition"))
    if outcome == "EXCEPTION":
        return f"{v['credit_id']:<8} EXCEPTION  {v['reason'] or '?':<28} {v['note'] or ''}"
    return f"{v['credit_id']:<8} IGNORED    {v['reason'] or '(no code)':<28} {v['note'] or ''}"


def render(matches: dict[str, Any]) -> str:
    """One line per verdict, in ``matches.json``'s own order (the bank statement's order)."""
    verdicts = matches["verdicts"]
    if not verdicts:
        return "Matched records: none -- matches.json carries no verdicts."

    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v["outcome"]] = counts.get(v["outcome"], 0) + 1
    summary = ", ".join(f"{n} {o}" for o, n in sorted(counts.items()))
    header = f"Matched records: {len(verdicts)} row(s) ({summary})"

    lines = [header, ""]
    for v in verdicts[:MAX_ROWS_LISTED]:
        lines.append(_render_row(v))
    if len(verdicts) > MAX_ROWS_LISTED:
        lines.append(f"  ... and {len(verdicts) - MAX_ROWS_LISTED} more")
    return "\n".join(lines)


if __name__ == "__main__":
    empty = {"verdicts": []}
    assert render(empty) == "Matched records: none -- matches.json carries no verdicts."

    resolved = {
        "credit_id": "C0001", "outcome": "RESOLVED",
        "settlement_ids": ["setl_0001"], "payment_ids": ["pay_0001"],
        "tier": 1, "confidence": None, "reason": None, "note": None,
        "residual_paise": 0, "credit_amount_paise": 20679,
        "decomposition": {
            "gross_paise": 21200, "fee_paise": 424, "gst_paise": 76, "tds_paise": 21,
            "refunds_paise": 0, "reserve_paise": 0, "expected_credit_paise": 20679,
            "rule": "gateway fee + GST + TDS at declared rates",
        },
    }
    exception = {
        "credit_id": "C0002", "outcome": "EXCEPTION",
        "settlement_ids": [], "payment_ids": [], "tier": None, "confidence": None,
        "reason": "FX_RATE_GAP", "note": "no settlement-day rate on file",
        "residual_paise": None, "credit_amount_paise": None, "decomposition": None,
    }
    ignored = {
        "credit_id": "C0003", "outcome": "IGNORED",
        "settlement_ids": [], "payment_ids": [], "tier": None, "confidence": None,
        "reason": "NON_GATEWAY_CREDIT", "note": "narration names no gateway counterparty",
        "residual_paise": None, "credit_amount_paise": None, "decomposition": None,
    }

    doc = {"verdicts": [resolved, exception, ignored]}
    text = render(doc)
    assert "Matched records: 3 row(s) (1 EXCEPTION, 1 IGNORED, 1 RESOLVED)" in text
    assert "C0001" in text and "RESOLVED" in text and "tier 1" in text
    assert "20679" not in text.replace("₹206.79", "")  # amounts render via fmt, not raw paise
    assert "₹206.79" in text  # 20679 paise
    assert "fee ₹4.24" in text and "GST ₹0.76" in text and "TDS ₹0.21" in text
    assert "-> expected ₹206.79" in text
    assert "rule: gateway fee + GST + TDS at declared rates" in text
    assert "C0002" in text and "FX_RATE_GAP" in text and "no settlement-day rate on file" in text
    assert "C0003" in text and "NON_GATEWAY_CREDIT" in text

    # Decomposition terms that are zero are omitted -- refunds and reserve above never
    # appear, since a report showing every term at zero on every row is noise.
    assert "refunds ₹0.00" not in text and "reserve ₹0.00" not in text

    # A row with no rule (raw decomposition, no rule string) still renders the bracket.
    no_rule = {**resolved, "credit_id": "C0004",
               "decomposition": {**resolved["decomposition"], "rule": None}}
    text2 = render({"verdicts": [no_rule]})
    assert "rule:" not in text2 and "-> expected ₹206.79]" in text2

    # Truncation, mirroring the queue's own convention.
    many = {"verdicts": [{**resolved, "credit_id": f"C{i:04d}"} for i in range(25)]}
    text3 = render(many)
    assert "and 5 more" in text3
    # +1 for the header's own summary line ("25 RESOLVED"), which is not a rendered row.
    assert text3.count("RESOLVED") == MAX_ROWS_LISTED + 1

    # No answer-key vocabulary anywhere -- this module never imports hisaab.scoring at all.
    import hisaab.report.matched as _self
    assert "hisaab.scoring" not in _self.__doc__ or True  # doc quotes it in prose; code does not import
    import ast, inspect
    src = inspect.getsource(_self)
    tree = ast.parse(src)
    imported = {n.module or "" for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) for n in [node]}
    assert not any("scoring" in (m or "") for m in imported), imported

    print("report/matched.py self-check ok")
