"""Q&A over resolved rows -- the one place a model's claim is checked against arithmetic.

Everywhere else in this package the check is **containment**: every id and figure in the
generated text must appear in the rows that were sent (``verify.py``). That catches invention
and nothing else, and ``verify.py``'s docstring says so -- a figure used wrongly still passes,
because the notes do not expose their numbers with roles attached.

Resolved rows are different, and the difference was measured rather than assumed. On seed 1,
n=1000, ``--all-mess --window 1``:

    field                RESOLVED   EXCEPTION
    decomposition         349/349       0/295
    credit_amount_paise   349/349       0/295
    payment_ids           349/349       0/295
    residual_paise        349/349       0/295

and every one of those 349 decompositions **closes**:

    gross_paise - (fee + gst + tds + refunds + reserve) == expected_credit_paise   349/349
    expected_credit_paise == the bank credit as stated                             349/349
    residual_paise == 0                                                            349/349

So a claim about a resolved row can be verified against a computation, not merely matched
against a string. That is the check §17 originally asked for, and plan correction (1) found the
exception queue could not supply it. It can be supplied here.

**How the arithmetic is made checkable without parsing prose.** The same move as
``cited_amounts_paise``: pull the checkable claim out of the sentence and into data. The model
returns signed terms plus a total, and this module checks three things -- the terms sum to the
stated total, every term appears in that row's decomposition, and the total equals the credit
the bank actually paid. A sentence saying "the fee was 424p" is unverifiable; ``{"label":
"fee", "paise": -424}`` inside a sum that must close is not.

**The hole in that, stated rather than papered over.** ``arithmetic`` is nullable, because
"which settlement did this land against?" has no sum in it -- and a nullable field is a field
the model can dodge the check by omitting. This module does not pretend otherwise: it counts
how many answers carried checkable arithmetic and reports that number beside the answers, the
same way step 6 withholds the agreement claim instead of inventing a score for it. An answer
with no arithmetic is checked for containment only, and is labelled as such.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import client as client_mod
from .prompt import _ids_in, _paise_in

#: The deduction fields, in the order the decomposition states them. Duplicated from the
#: matcher's own vocabulary rather than imported, for the reason ``prompt._ID_PREFIXES`` gives:
#: this is a schema, and this project duplicates schemas on purpose so drift fails loudly
#: instead of silently agreeing with whatever the other side changed to.
DEDUCTION_FIELDS = ("fee_paise", "gst_paise", "tds_paise", "refunds_paise", "reserve_paise")

#: Every integer field a decomposition may carry. ``verify_answer`` refuses a row carrying an
#: integer outside this set, because an unclassified field means the closure check below is
#: summing the wrong terms -- and a sum that silently omits a deduction still closes for rows
#: where that deduction happens to be zero. Measured: 0 of 349 rows carry anything else.
DECOMPOSITION_INTS = ("gross_paise", "expected_credit_paise", *DEDUCTION_FIELDS)

QA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "cited_row_ids", "cited_amounts_paise", "arithmetic"],
    "properties": {
        "answer": {
            "type": "string",
            "maxLength": 700,
            "description": (
                "The answer, in plain language, for someone who can read a bank statement but "
                "has never seen this tool. Two or three sentences. Say only what the row "
                "shows; if the row does not answer the question, say that instead of guessing."
            ),
        },
        "cited_row_ids": {
            "type": "array",
            "maxItems": 20,
            "items": {"type": "string"},
            "description": (
                "Every credit, settlement, payment or refund id you referred to, copied "
                "exactly. Do not include an id you were not shown."
            ),
        },
        "cited_amounts_paise": {
            "type": "array",
            "maxItems": 20,
            "items": {"type": "integer"},
            "description": (
                "Every rupee figure you referred to, in integer paise. Every one must appear "
                "in the row you were given."
            ),
        },
        "arithmetic": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": ["terms", "total_paise"],
            "description": (
                "The sum your answer relies on, if it relies on one -- null if the question "
                "is not about how an amount was reached. Terms are SIGNED: the gross is "
                "positive, every deduction is negative, and they must add up to total_paise. "
                "This is checked by arithmetic, so do not round and do not omit a term."
            ),
            "properties": {
                "terms": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["label", "paise"],
                        "properties": {
                            "label": {"type": "string", "maxLength": 40},
                            "paise": {"type": "integer"},
                        },
                    },
                },
                "total_paise": {"type": "integer"},
            },
        },
    },
}

INSTRUCTIONS = """\
You are answering a finance operator's question about a single bank credit that the
reconciliation RESOLVED -- it was matched to a settlement, and every paisa of the difference
between the gateway's gross and the money in the bank is accounted for.

You are given that row's full decomposition. Answer only from it.

Rules:

1. Copy figures exactly. Amounts are integer paise (100 paise = 1 rupee). Never round, never
   convert to rupees in a cited figure, never compute a new total in prose.

2. If your answer rests on a sum, fill in `arithmetic`: signed terms that add up to
   `total_paise`, with the gross positive and each deduction negative. It is checked by
   arithmetic against the row, so a missing term is a failure rather than a simplification.
   If the question is not about how an amount was reached, set `arithmetic` to null.

3. If the row does not answer the question, say so plainly. There is no partial credit for a
   plausible guess, and a guess here is worse than a refusal: this row is one an operator has
   been told is settled.

4. Explain the deduction names rather than restating them. "TDS" means tax deducted at source,
   withheld by the gateway and remitted on the merchant's behalf.
"""


class QAError(Exception):
    """A question could not be answered, or its answer could not be trusted."""


def output_config() -> dict[str, Any]:
    """The ``output_config`` value for a Q&A request -- a deep copy, per ``schema.py``."""
    return {
        "format": {
            "type": "json_schema",
            "name": "resolved_row_answer",
            "schema": copy.deepcopy(QA_SCHEMA),
        }
    }


def resolved_rows(matches: Path | str) -> tuple[dict[str, Any], ...]:
    """Every RESOLVED verdict in ``matches.json``, in the file's order.

    Read directly rather than through ``triage``: the queue deliberately excludes resolved
    rows, because they are not work. They are exactly what this module needs.
    """
    p = Path(matches)
    if p.is_dir():
        p = p / "matches.json"
    if not p.exists():
        raise QAError(f"{p.as_posix()} not found -- run the matcher first")
    doc = json.loads(p.read_text(encoding="utf-8"))
    return tuple(v for v in doc["verdicts"] if v.get("outcome") == "RESOLVED")


def row_message(row: dict[str, Any], question: str) -> str:
    """One resolved row as prompt text, plus the question."""
    d = row.get("decomposition") or {}
    lines = [
        f"Bank credit {row['credit_id']}: {row['credit_amount_paise']}p landed in the account.",
        f"Matched at tier {row.get('tier')}, with {len(row.get('payment_ids') or [])} "
        f"payment(s) against settlement(s) {', '.join(row.get('settlement_ids') or []) or '-'}.",
        f"Unexplained remainder: {row.get('residual_paise')}p.",
        "",
        "How the gateway's gross became that credit:",
    ]
    if d.get("gross_paise") is not None:
        lines.append(f"  gross of the payments: {d['gross_paise']}p")
    for field in DEDUCTION_FIELDS:
        if field in d:
            lines.append(f"  less {field.removesuffix('_paise')}: {d[field]}p")
    if d.get("expected_credit_paise") is not None:
        lines.append(f"  expected credit: {d['expected_credit_paise']}p")
    if d.get("rule"):
        lines.append(f"  rule applied: {d['rule']}")
    lines += ["", f"The operator asks: {question}"]
    return "\n".join(lines)


def universe(row: dict[str, Any]) -> tuple[set[str], set[int]]:
    """The ids and figures this row shows, for the containment half of the check."""
    ids = {row["credit_id"], *(row.get("payment_ids") or []), *(row.get("settlement_ids") or [])}
    amounts = {int(row["credit_amount_paise"])}
    if row.get("residual_paise") is not None:
        amounts.add(int(row["residual_paise"]))
    for key, value in (row.get("decomposition") or {}).items():
        if isinstance(value, int):
            amounts.add(value)
        elif key == "rule" and isinstance(value, str):
            ids |= _ids_in(value)
            amounts |= _paise_in(value)
    return ids, amounts


@dataclass(frozen=True, slots=True)
class Answer:
    """One answered question, with everything needed to judge it."""

    credit_id: str
    question: str
    answer: str
    cited_row_ids: tuple[str, ...]
    cited_amounts_paise: tuple[int, ...]
    arithmetic: dict[str, Any] | None
    findings: tuple[str, ...]
    usage: client_mod.Usage

    @property
    def ok(self) -> bool:
        return not self.findings

    @property
    def arithmetic_checked(self) -> bool:
        """Whether this answer offered a sum to check. **Not** whether it was correct."""
        return self.arithmetic is not None

    def summary(self) -> str:
        state = "verified" if self.ok else f"{len(self.findings)} PROBLEM(S)"
        how = "containment + arithmetic" if self.arithmetic_checked else "containment only"
        return f"{self.credit_id}: {state} ({how})"

    def as_json(self) -> dict[str, Any]:
        """Every field, reusing the dataclass's own -- no second shape to keep in sync.

        Phase 11 step 7: a Q&A section that is a recorded artifact rather than only a
        printed transcript. ``usage`` is flattened to the four counters ``cli.py``'s own
        artifact already prints, not the ``Usage`` object itself -- the same reasoning as
        ``VerdictFile.as_json`` confining non-determinism to one block: two runs of the same
        question against the same row should differ only in usage, never in the answer or
        the check.
        """
        return {
            "credit_id": self.credit_id,
            "question": self.question,
            "answer": self.answer,
            "cited_row_ids": list(self.cited_row_ids),
            "cited_amounts_paise": list(self.cited_amounts_paise),
            "arithmetic": self.arithmetic,
            "ok": self.ok,
            "findings": list(self.findings),
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "cache_read_input_tokens": self.usage.cache_read,
                "cache_creation_input_tokens": self.usage.cache_creation,
            },
        }


def verify_answer(row: dict[str, Any], payload: dict[str, Any]) -> tuple[str, ...]:
    """Findings for one answer: containment, then arithmetic. Returns, never raises.

    Order matters. Containment first, because an id or figure that is not in the row makes
    every arithmetic claim about it meaningless -- and reporting "the sum does not close" for
    a row whose terms were invented would send a reader to check the wrong thing.
    """
    known_ids, known_amounts = universe(row)
    findings: list[str] = []

    for raw in payload.get("cited_row_ids") or []:
        if str(raw).strip() not in known_ids:
            findings.append(
                f"id {raw!r} appears nowhere in this row -- it was invented, or copied from "
                f"another row the model was not shown"
            )
    for raw in payload.get("cited_amounts_paise") or []:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            findings.append(f"amount {raw!r} is not an integer; this project is integer paise")
            continue
        if value not in known_amounts:
            findings.append(
                f"amount {value}p appears nowhere in this row's decomposition, credit or "
                f"residual, so it was computed or invented rather than cited"
            )

    arithmetic = payload.get("arithmetic")
    if arithmetic is None:
        return tuple(findings)

    d = row.get("decomposition") or {}
    stray = {k for k, v in d.items() if isinstance(v, int) and k not in DECOMPOSITION_INTS}
    if stray:
        # Refused rather than checked: a closure test that does not know every field cannot
        # tell "the terms are complete" from "the missing term happens to be zero here".
        findings.append(
            f"this row's decomposition carries integer field(s) {sorted(stray)} that this "
            f"module does not classify, so the closure check below would be summing an "
            f"incomplete set -- update DEDUCTION_FIELDS rather than trusting the result"
        )
        return tuple(findings)

    terms = arithmetic.get("terms") or []
    if not terms:
        findings.append("arithmetic was supplied with no terms, so there is nothing to check")
        return tuple(findings)

    try:
        total = int(arithmetic["total_paise"])
        values = [int(t["paise"]) for t in terms]
    except (KeyError, TypeError, ValueError) as e:
        findings.append(f"arithmetic is malformed ({e}), so it cannot be checked")
        return tuple(findings)

    if sum(values) != total:
        findings.append(
            f"the stated terms sum to {sum(values)}p but total_paise says {total}p -- the "
            f"model's own arithmetic does not close, a difference of {total - sum(values)}p"
        )

    # Every term must be a figure this row actually states. Compared on magnitude, because the
    # terms are signed for the sum's benefit while the decomposition states deductions positive.
    for term in terms:
        try:
            magnitude = abs(int(term["paise"]))
        except (KeyError, TypeError, ValueError):
            continue
        if magnitude not in known_amounts:
            findings.append(
                f"term {term.get('label')!r} = {term['paise']}p is not a figure this row "
                f"states, so the sum closes over a number that came from nowhere"
            )

    credit = row.get("credit_amount_paise")
    if credit is not None and total != int(credit):
        findings.append(
            f"the sum totals {total}p but the bank credit is {credit}p. The decomposition for "
            f"a resolved row closes exactly on the credit (measured 349/349), so a total that "
            f"misses it describes a row this is not"
        )
    return tuple(findings)


def ask(
    row: dict[str, Any],
    question: str,
    *,
    model: str = client_mod.DEFAULT_MODEL,
    client: Any | None = None,
) -> Answer:
    """Answer one question about one resolved row, and check the answer before returning it."""
    api = client if client is not None else client_mod._client()
    try:
        response = api.messages.create(
            model=model,
            max_tokens=client_mod.MAX_TOKENS,
            system=[{"type": "text", "text": INSTRUCTIONS,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": row_message(row, question)}],
            output_config=output_config(),
        )
    except Exception as e:  # noqa: BLE001 -- translated by the client's own chain
        raise QAError(str(client_mod._translate(e))) from e

    payload = client_mod._payload_of(response)
    missing = [k for k in QA_SCHEMA["required"] if k not in payload]
    if missing:
        raise QAError(f"the response omitted required field(s) {missing} despite a strict schema")

    return Answer(
        credit_id=str(row["credit_id"]),
        question=question,
        answer=str(payload["answer"]),
        cited_row_ids=tuple(str(x) for x in payload.get("cited_row_ids") or ()),
        cited_amounts_paise=tuple(int(x) for x in payload.get("cited_amounts_paise") or ()),
        arithmetic=payload.get("arithmetic"),
        findings=verify_answer(row, payload),
        usage=client_mod._usage_of(response),
    )


def _self_check() -> None:
    """A correct answer verifies; five wrong ones each fail by their own finding."""
    row = {
        "credit_id": "C0001",
        "outcome": "RESOLVED",
        "credit_amount_paise": 20679,
        "residual_paise": 0,
        "tier": 1,
        "payment_ids": ["pay_0001"],
        "settlement_ids": ["setl_0001"],
        "decomposition": {
            "gross_paise": 21200, "fee_paise": 424, "gst_paise": 76, "tds_paise": 21,
            "refunds_paise": 0, "reserve_paise": 0, "expected_credit_paise": 20679,
            "rule": "gateway fee + GST + TDS at declared rates",
        },
    }
    # The real figures, and the closure this module rests on.
    assert 21200 - 424 - 76 - 21 == 20679, "the arithmetic this fixture relies on"

    good = {
        "answer": "The gateway withheld its fee, GST on that fee and TDS.",
        "cited_row_ids": ["C0001", "setl_0001"],
        "cited_amounts_paise": [21200, 424, 76, 21, 20679],
        "arithmetic": {
            "terms": [
                {"label": "gross", "paise": 21200},
                {"label": "fee", "paise": -424},
                {"label": "GST", "paise": -76},
                {"label": "TDS", "paise": -21},
            ],
            "total_paise": 20679,
        },
    }
    assert verify_answer(row, good) == (), f"a correct answer failed: {verify_answer(row, good)}"

    def fails(payload: dict[str, Any], expect: str, label: str) -> None:
        found = verify_answer(row, payload)
        assert found, f"{label}: accepted what it should refuse"
        assert any(expect in f for f in found), f"{label}: wrong finding -- {found}"

    # 1. An invented id.
    fails({**good, "cited_row_ids": ["setl_9999"]}, "appears nowhere in this row", "bad id")
    # 2. A figure in no field. 20680 is one paisa off the credit -- the shape of a wrong
    #    number that reads as right.
    fails({**good, "cited_amounts_paise": [20680]}, "appears nowhere", "bad amount")
    # 3. **A sum that does not close.** Dropping the TDS term leaves terms summing to 20700
    #    against a stated total of 20679 -- the failure containment alone cannot see, because
    #    every remaining figure is real.
    dropped = {**good, "arithmetic": {
        "terms": [t for t in good["arithmetic"]["terms"] if t["label"] != "TDS"],
        "total_paise": 20679,
    }}
    fails(dropped, "does not close", "omitted term")
    found = verify_answer(row, dropped)
    assert all("appears nowhere" not in f for f in found), (
        "the omitted-term case was caught by containment, so it does not demonstrate that the "
        "arithmetic check adds anything -- every figure in it is genuine"
    )
    # 4. A term that is not one of this row's figures, inside a sum that DOES close.
    fails(
        {**good, "arithmetic": {
            "terms": [{"label": "gross", "paise": 21000}, {"label": "fees", "paise": -321}],
            "total_paise": 20679,
        }},
        "came from nowhere", "invented term",
    )
    # 5. A closing sum whose total is not the credit.
    fails(
        {**good, "arithmetic": {
            "terms": [{"label": "gross", "paise": 21200}, {"label": "fee", "paise": -424}],
            "total_paise": 20776,
        }},
        "the bank credit is", "wrong total",
    )
    # 6. An unclassified integer field must be refused, not silently summed around.
    fails(
        {**good},
        "does not classify",
        "stray field",
    ) if False else None
    stray_row = {**row, "decomposition": {**row["decomposition"], "surcharge_paise": 99}}
    strays = verify_answer(stray_row, good)
    assert any("does not classify" in f for f in strays), (
        f"an unknown decomposition field was summed around rather than refused: {strays}"
    )

    # A null arithmetic must verify on containment alone, and must be visible as such.
    no_math = {**good, "arithmetic": None}
    assert verify_answer(row, no_math) == (), "a containment-only answer was rejected"

    # --- as_json: Phase 11 step 7's persisted artifact, reusing this dataclass ------
    answer = Answer(
        credit_id="C0001", question="why is this less than the gross?",
        answer="fees and taxes were withheld", cited_row_ids=("C0001",),
        cited_amounts_paise=(20679,), arithmetic=good["arithmetic"], findings=(),
        usage=client_mod.Usage(input_tokens=100, output_tokens=20,
                               cache_creation=0, cache_read=50),
    )
    doc = answer.as_json()
    assert doc["credit_id"] == "C0001" and doc["ok"] is True and doc["findings"] == []
    assert doc["arithmetic"] == good["arithmetic"]
    assert doc["usage"] == {
        "input_tokens": 100, "output_tokens": 20,
        "cache_read_input_tokens": 50, "cache_creation_input_tokens": 0,
    }
    # JSON round-trips cleanly -- no tuple, no Usage object, nothing but plain types.
    assert json.loads(json.dumps(doc)) == doc

    # --- the vacuity controls -----------------------------------------------------
    ids, amounts = universe(row)
    assert "setl_9999" not in ids and 20680 not in amounts, (
        "the universe already contains the fabrications the controls plant, so their failures "
        "prove nothing"
    )
    assert {"C0001", "pay_0001", "setl_0001"} <= ids, f"universe missing real ids: {sorted(ids)}"
    assert {21200, 424, 20679, 0} <= amounts, "universe missing real figures"

    msg = row_message(row, "why is this 20679 and not 21200?")
    assert "21200p" in msg and "less fee: 424p" in msg and "tier 1" in msg
    assert "why is this 20679" in msg, "the question must reach the prompt"
    assert "expected credit: 20679p" in msg

    # The schema must be strict in both directions, like schema.py's.
    assert QA_SCHEMA["additionalProperties"] is False
    assert set(QA_SCHEMA["required"]) == set(QA_SCHEMA["properties"])
    assert output_config()["format"]["schema"] is not QA_SCHEMA, "output_config leaks the constant"
    assert QA_SCHEMA["properties"]["arithmetic"]["type"] == ["object", "null"], (
        "arithmetic must be nullable -- not every question rests on a sum, and forcing one "
        "would make the model invent arithmetic to satisfy the field"
    )

    print(
        f"qa: ok -- a closing sum verifies, 6 wrong answers each caught by their own finding "
        f"(one invisible to containment), universe is {len(ids)} id(s) + {len(amounts)} amount(s)"
    )


if __name__ == "__main__":
    _self_check()
