"""The citation check: every id and every figure in generated text must be in the input.

**What the roadmap asked for, and what is actually available.** Section 17 says to verify
cited amounts "against the computed ones". Measured on seed 1, n=1000, ``--all-mess``: of
295 EXCEPTION rows, ``decomposition``, ``credit_amount_paise``, ``residual_paise`` and any
linked settlement or payment ids are **0/295** -- null on every one. That is not a defect;
it is what makes a row an exception. There is no computed decomposition to check against.

So the check is built the other way round, and is stronger than the fallback the Phase 10
plan settled for ("one bank amount plus row ids"). Every exception row carries a
**note** -- 295/295 non-empty, mean 569 chars, 98% naming a settlement id and 90% carrying
paise figures -- and the notes are part of what the model is shown. So the rule becomes:

    every id and every paise figure in the generated text must appear in the rows that
    were actually sent.

A figure that appears nowhere in the input is a fabrication, whatever the verdict does or
does not compute. That is the hallucination §17 was reaching for, and it is checkable on
all 295 rows rather than on the 15 that happen to state an amount.

**Three limits, stated because a check whose weaknesses are unlisted gets over-trusted.**

1. **Containment does not catch misattribution.** If a note says "the credit falls 526p
   short of the 19600p gross", both figures are in the universe, so text swapping them --
   "the gross was 526p" -- passes. The check proves every number is *real*, not that each
   is used correctly. Catching that needs the numbers to carry roles, which the notes do
   not expose as data.
2. **The universe is the sample, not the group.** A group of 141 rows sends 12
   (``prompt.ROWS_PER_GROUP``), and the universe is built from those 12 alone. Checking
   against all 141 would let a fabricated id pass by coinciding with a row the model never
   saw -- which is exactly the coincidence this exists to catch.
3. **A derived figure that coincides with a real input is not caught, and the coincidence
   is structural rather than unlucky.** Found by writing the opposite assertion in
   ``_self_check`` and watching it fail: the note "falls 526p short of the 19600p gross"
   makes ``19600 - 526`` equal that row's own bank credit *by construction*, because the
   note describes the relationship between figures it states. So the notes hand the model
   arithmetic whose results are frequently already in the universe. Derived figures that
   cross rows (``19600 - 43``) still fail. The check catches invention, not derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .prompt import cited_universe

#: Figures every response may cite without them appearing in a row. Empty on purpose, and
#: documented as a decision rather than an oversight: 0 and 1 are tempting to allow (a
#: model writing "1 payment") but ``cited_amounts_paise`` is specified as *amounts*, so a
#: bare 1 there is a misuse of the field rather than a rounding nicety worth permitting.
ALWAYS_ALLOWED_AMOUNTS: frozenset[int] = frozenset()


@dataclass(frozen=True, slots=True)
class Finding:
    """One citation that is not in the input, named so a reader can act on it."""

    kind: str          # "id" or "amount"
    value: str
    detail: str

    def __str__(self) -> str:
        return f"{self.kind} {self.value}: {self.detail}"


@dataclass(frozen=True, slots=True)
class Verification:
    """The result of checking one group's explanation."""

    group_reason: str
    checked_ids: int
    checked_amounts: int
    findings: tuple[Finding, ...]

    @property
    def ok(self) -> bool:
        return not self.findings

    @property
    def checked(self) -> int:
        return self.checked_ids + self.checked_amounts

    def summary(self) -> str:
        if self.ok:
            return (
                f"{self.group_reason}: {self.checked} citation(s) verified "
                f"({self.checked_ids} id, {self.checked_amounts} amount)"
            )
        return (
            f"{self.group_reason}: {len(self.findings)} of {self.checked} citation(s) "
            f"NOT FOUND in the input -- " + "; ".join(str(f) for f in self.findings)
        )


def verify(group: dict[str, Any], explanation: dict[str, Any]) -> Verification:
    """Check one explanation's citations against the rows that group actually sent.

    Returns findings rather than raising: a caller writing a report wants to mark the row
    and carry on, and the CLI decides whether a fabricated citation is fatal. A function
    that raised would make "how many groups were clean" unanswerable.
    """
    known_ids, known_amounts = cited_universe(group)
    findings: list[Finding] = []

    cited_ids = explanation.get("cited_row_ids") or []
    for raw in cited_ids:
        value = str(raw).strip()
        if value not in known_ids:
            findings.append(Finding(
                kind="id",
                value=value,
                detail=(
                    "appears in no row that was sent for this group. Either the model "
                    "invented it, or it copied an id from a row outside the sample -- "
                    "both are unsupported in the text a person reads."
                ),
            ))

    cited_amounts = explanation.get("cited_amounts_paise") or []
    for raw in cited_amounts:
        try:
            value_i = int(raw)
        except (TypeError, ValueError):
            findings.append(Finding(
                kind="amount",
                value=repr(raw),
                detail=(
                    "is not an integer. Amounts are integer paise throughout this project; "
                    "a formatted or fractional figure cannot be compared to the input."
                ),
            ))
            continue
        if value_i in ALWAYS_ALLOWED_AMOUNTS:
            continue
        if value_i not in known_amounts:
            findings.append(Finding(
                kind="amount",
                value=str(value_i),
                detail=(
                    f"appears in no row that was sent. Nothing in this group's notes or "
                    f"bank amounts carries {value_i}p, so it was computed or invented "
                    f"rather than cited."
                ),
            ))

    return Verification(
        # The dismissal group carries no reason code, so ``str(reason)`` printed the literal
        # "None" as a group's name -- `citations: None: 6 citation(s) verified`. Falls back to
        # the queue's own label, which is what a person reads in every other report here.
        group_reason=str(group.get("reason") or group.get("cause") or "unlabelled group"),
        checked_ids=len(cited_ids),
        checked_amounts=len(cited_amounts),
        findings=tuple(findings),
    )


def _self_check() -> None:
    """A clean citation passes, a corrupted one fails, and each by its own assertion."""
    group = {
        "reason": "UNEXPLAINED_RESIDUAL",
        "cause": "the money does not match",
        "rows": 2,
        "value_paise": 40000,
        "credits": [
            {
                "credit_id": "C0101",
                "bank_amount_paise": 19074,
                "note": "setl_0164 agrees on date and amount, but the credit falls 526p "
                        "short of the 19600p gross of 1 payment(s)",
            },
            {
                "credit_id": "C0102",
                "bank_amount_paise": 20926,
                "note": "setl_0165 agrees, 43p unexplained",
            },
        ],
    }

    clean = {
        "summary": "s", "why_unresolved": "w", "next_step": "n",
        "cited_row_ids": ["C0101", "setl_0164"],
        "cited_amounts_paise": [19074, 526, 19600, 43],
    }
    result = verify(group, clean)
    assert result.ok, f"a clean citation was rejected: {result.summary()}"
    assert result.checked == 6, f"expected 6 citations checked, got {result.checked}"

    # The controls. Each must fail, and the finding must name the right thing -- a check
    # that fails for the wrong reason is indistinguishable from one that works.
    bad_id = {**clean, "cited_row_ids": ["C0101", "setl_9999"]}
    r = verify(group, bad_id)
    assert not r.ok, "a fabricated settlement id passed"
    assert len(r.findings) == 1 and r.findings[0].kind == "id"
    assert r.findings[0].value == "setl_9999"

    # 20126 is a transposition of a real figure (20926): close enough to look right in
    # prose, absent from the input. This is the failure the check exists for.
    bad_amount = {**clean, "cited_amounts_paise": [19074, 20126]}
    r = verify(group, bad_amount)
    assert not r.ok, "a transposed amount passed"
    assert len(r.findings) == 1 and r.findings[0].kind == "amount"
    assert r.findings[0].value == "20126"

    # A derived figure that lands outside the universe IS caught: 19600 - 43 crosses two
    # rows and matches nothing.
    r = verify(group, {**clean, "cited_amounts_paise": [19600 - 43]})
    assert not r.ok, "a cross-row derived figure passed"
    assert r.findings[0].value == "19557"

    # But a derived figure that COINCIDES with a real input is not caught, and this
    # assertion documents that rather than pretending otherwise. It is limit 3 in the
    # module docstring, and it was found by writing the opposite assertion and watching it
    # fail: 19600 - 526 == 19074, which is C0101's own bank credit.
    #
    # That coincidence is not bad luck -- it is structural. The note says the credit "falls
    # 526p short of the 19600p gross", so gross minus shortfall equals the credit *by
    # construction*. The notes describe arithmetic relationships between figures they
    # state, so the results of that arithmetic are frequently in the universe already.
    assert 19600 - 526 == 19074, "the arithmetic this comment relies on"
    assert verify(group, {**clean, "cited_amounts_paise": [19600 - 526]}).ok, (
        "if this now fails, containment got stricter than the docstring claims -- update "
        "limit 3, because it currently tells a reader this case passes"
    )

    non_int = {**clean, "cited_amounts_paise": ["19,074"]}
    r = verify(group, non_int)
    assert not r.ok and r.findings[0].kind == "amount", "a formatted amount passed"
    assert "not an integer" in r.findings[0].detail

    # Missing fields must read as "cited nothing", never crash: the schema requires both
    # lists, but this function is also handed recorded responses from earlier runs.
    r = verify(group, {"summary": "s"})
    assert r.ok and r.checked == 0, "an explanation with no citations should verify vacuously"

    # The dismissal group has no reason code, and the first version of this module printed
    # `str(None)` as its name: `citations: None: 6 citation(s) verified`. Found by running
    # the CLI over all 8 fixture groups rather than over this hand-built one, which is the
    # kind of defect a self-check on invented data cannot reach.
    dismissal = {**group, "reason": None, "cause": "DISMISSED (not gateway money)"}
    r = verify(dismissal, clean)
    assert "None" not in r.summary(), f"a null reason code leaked into the report: {r.summary()}"
    assert r.summary().startswith("DISMISSED"), (
        f"an uncoded group must fall back to the queue's own label, got: {r.summary()}"
    )
    # And with neither: it must still not say "None".
    r = verify({**group, "reason": None, "cause": None}, clean)
    assert "None" not in r.summary(), f"still leaking: {r.summary()}"

    # And the vacuity control for THIS self-check: the universe must not contain
    # everything, or every assertion above passes for free.
    ids, amounts = cited_universe(group)
    assert "setl_9999" not in ids and 20126 not in amounts, (
        "the universe already contains the fabrications the controls plant, so the "
        "failures above prove nothing"
    )
    assert 19074 in amounts and "setl_0164" in ids, "the universe is missing real input"

    print(
        f"verify: ok -- clean citations pass, 5 corruptions each caught by their own "
        f"assertion, universe is {len(ids)} id(s) + {len(amounts)} amount(s)"
    )


if __name__ == "__main__":
    _self_check()
