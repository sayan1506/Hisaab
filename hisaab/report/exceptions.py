"""The exception queue: one section per triage group, joined against the explain artifact.

Step 5. The join is on ``cause`` -- a plain string, confirmed identical between the two
documents for one run by `.plan/phase11.md` §0: ``hisaab/triage/cli.py``'s ``as_json()``
stores ``rg.label`` under ``"cause"`` per group, ``hisaab/explain/cli.py``'s
``groups_from_live`` builds its groups by calling ``triage.group.group_rulings`` and
``triage.value.rank`` directly rather than re-deriving the partition, and each explanation
carries that same value under its own ``"cause"`` key. So this module needs no fuzzy
matching -- an exact string match on ``cause`` is the join, or there is no explanation for
that group.

**No explain artifact is not a degraded report.** Per plan correction (3), a bare run (no
``hisaab.explain`` invocation at all) is a legitimate state, matching what Phase 10 itself
ships as an optional layer. Every group renders its triage-sourced action and unblocks either
way; an explanation only adds a second, model-generated paragraph beside them when one exists
for that cause.
"""

from __future__ import annotations

from typing import Any

from ..common.money import fmt


def _explanation_by_cause(explain: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if explain is None:
        return {}
    return {e["cause"]: e for e in explain["explanations"]}


def _render_group(group: dict[str, Any], explanation: dict[str, Any] | None) -> str:
    lines = [
        f"{group['cause']}  --  {fmt(group['value_paise'])} across {group['rows']} row(s), "
        f"~{group['estimated_minutes']} min ({group['minutes_per_row']} min each)",
        f"  Do: {group['action']}",
    ]
    if group["unblocks"] is not None:
        lines.append(f"  Would stop it recurring: {group['unblocks']}")

    if explanation is None:
        lines.append("  (no model explanation for this group -- template hint above only)")
    else:
        exp = explanation["explanation"]
        check = explanation["citation_check"]
        citation_state = "clean" if check["ok"] else f"{len(check['findings'])} finding(s)"
        lines += [
            "",
            f"  Explained: {exp['summary']}",
            f"  Why unresolved: {exp['why_unresolved']}",
            f"  Next step: {exp['next_step']}",
            f"  Citations: {citation_state} ({check['checked']} claim(s) checked)",
        ]
    return "\n".join(lines)


def render(triage: dict[str, Any], explain: dict[str, Any] | None) -> str:
    """The queue, one section per group, heaviest first (the triage document's own order)."""
    groups = triage["groups"]
    if not groups:
        return "Exception queue: empty -- every bank row was resolved or dismissed."

    by_cause = _explanation_by_cause(explain)
    totals = triage["totals"]
    header = (
        f"Exception queue: {totals['rows']} row(s) in {totals['groups']} group(s), "
        f"{fmt(totals['value_paise'])} at risk, ~{totals['estimated_minutes']} min to clear"
    )
    if explain is None:
        header += "\n(no explain artifact for this run -- every group shows its template hint only)"

    sections = [_render_group(g, by_cause.get(g["cause"])) for g in groups]
    return "\n\n".join([header, *sections])


if __name__ == "__main__":
    TRIAGE_EMPTY = {
        "schema_version": 1,
        "inputs": {"matches": "matches.json", "data": "data"},
        "totals": {"groups": 0, "rows": 0, "value_paise": 0, "estimated_minutes": 0},
        "groups": [],
    }
    assert render(TRIAGE_EMPTY, None) == "Exception queue: empty -- every bank row was resolved or dismissed."
    assert render(TRIAGE_EMPTY, {"explanations": []}) == render(TRIAGE_EMPTY, None), (
        "an empty queue renders the same whether or not an explain artifact was given"
    )

    fx_group = {
        "cause": "FX_RATE_GAP", "kind": "exception", "reason": "FX_RATE_GAP",
        "rows": 2, "value_paise": 500_000, "minutes_per_row": 20, "estimated_minutes": 40,
        "action": "Supply the settlement-day rate.",
        "unblocks": "the settlement-day conversion rate",
        "credits": [
            {"credit_id": "C0001", "value_paise": 300_000, "reason": "FX_RATE_GAP"},
            {"credit_id": "C0002", "value_paise": 200_000, "reason": "FX_RATE_GAP"},
        ],
    }
    dismissal_group = {
        "cause": "DISMISSED (not gateway money)", "kind": "dismissal", "reason": None,
        "rows": 1, "value_paise": 10_000, "minutes_per_row": 3, "estimated_minutes": 3,
        "action": "Scan the list to agree none of them belongs in the reconciliation.",
        "unblocks": None,
        "credits": [{"credit_id": "C0003", "value_paise": 10_000, "reason": "NON_GATEWAY_CREDIT"}],
    }
    triage_doc = {
        "schema_version": 1,
        "inputs": {"matches": "matches.json", "data": "data"},
        "totals": {"groups": 2, "rows": 3, "value_paise": 510_000, "estimated_minutes": 43},
        "groups": [fx_group, dismissal_group],
    }

    # --- no explain artifact: every group shows only its template hint -----------------
    text = render(triage_doc, None)
    assert "no explain artifact for this run" in text
    assert "FX_RATE_GAP" in text and "Supply the settlement-day rate" in text
    assert "no model explanation for this group" in text
    assert text.count("no model explanation for this group") == 2, "both groups must fall back"
    assert "Would stop it recurring: the settlement-day conversion rate" in text
    # The dismissal group's unblocks is None and must not print a line for it.
    assert "Would stop it recurring: None" not in text

    # --- an explain artifact covering only one of the two groups -----------------------
    explain_doc = {
        "schema_version": 1, "model": "m", "endpoint": "e", "groups": 1, "explained": 1,
        "citations_clean": 1,
        "usage_total": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0},
        "explanations": [
            {
                "group_reason": "FX_RATE_GAP", "cause": "FX_RATE_GAP", "rows": 2,
                "value_paise": 500_000, "cell": None,
                "explanation": {
                    "summary": "Two FX-captured payments have no settlement-day rate.",
                    "why_unresolved": "The gross was fixed at capture, the payout is not.",
                    "next_step": "Get the settlement-day rate for both.",
                    "cited_row_ids": ["C0001", "C0002"], "cited_amounts_paise": [300_000, 200_000],
                },
                "citation_check": {"ok": True, "checked": 4, "findings": []},
                "hint_comparison": {},
                "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                          "cache_creation_input_tokens": 0},
            },
        ],
    }
    text2 = render(triage_doc, explain_doc)
    assert "no explain artifact for this run" not in text2
    # The explained group shows the model's paragraph.
    assert "Explained: Two FX-captured payments have no settlement-day rate." in text2
    assert "Citations: clean (4 claim(s) checked)" in text2
    # The un-explained group still falls back, on its own -- not the whole report.
    assert "no model explanation for this group" in text2
    assert text2.count("no model explanation for this group") == 1, (
        "only the group with no matching cause should fall back once the artifact exists"
    )

    # --- a dirty citation check renders the finding count, not a bare 'ok' -------------
    dirty_explain = {**explain_doc, "explanations": [
        {**explain_doc["explanations"][0],
         "citation_check": {"ok": False, "checked": 4, "findings": ["bad id", "bad amount"]}},
    ]}
    text3 = render(triage_doc, dirty_explain)
    assert "Citations: 2 finding(s) (4 claim(s) checked)" in text3

    print("report/exceptions.py self-check ok")
