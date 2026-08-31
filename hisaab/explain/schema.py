"""The shape the model must return, and why every field in it is checkable.

Phase 9's smoke probe found the model **fencing its JSON in a code block despite being
told not to**, and concluded Phase 10 would need a "tolerant extractor" that strips
fences before parsing. Structured output retires that: ``output_config.format`` with a
JSON schema constrains the response, so there is nothing to strip and no tolerance to
tune. A tolerant parser is a parser that cannot tell malformed output from output it
guessed at, which is the wrong instrument for a component whose failure mode is
confident fabrication.

**Every field here exists to be verified against the input, or to be counted.** A field
the model can fill with anything and nobody checks is a field that will eventually be
wrong in a way nothing detects -- so ``cited_row_ids`` and ``cited_amounts_paise`` are
extracted as *data* rather than left inside prose, precisely so ``verify.py`` can hold
them against the fixture. The prose itself is unverifiable by construction; the citations
carried beside it are not.
"""

from __future__ import annotations

import copy
from typing import Any

#: The per-group explanation schema, as ``output_config.format`` takes it.
#:
#: ``additionalProperties: false`` plus a complete ``required`` list is what makes this
#: strict rather than advisory: the model cannot add a field, and cannot omit one. Both
#: matter here -- an extra field would arrive unverified, and a missing citation list
#: would read as "cited nothing" rather than as a malformed response.
EXPLANATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "why_unresolved", "next_step", "cited_row_ids", "cited_amounts_paise"],
    "properties": {
        "summary": {
            "type": "string",
            "maxLength": 400,
            "description": (
                "One or two sentences an accountant could paste into a note to their "
                "manager, naming what this group of rows has in common. No jargon from "
                "the reason code itself -- explain it, do not restate it."
            ),
        },
        "why_unresolved": {
            "type": "string",
            "maxLength": 600,
            "description": (
                "Why the matcher could not resolve these rows, in terms of what the "
                "input files do and do not contain. This must describe missing or "
                "ambiguous EVIDENCE, never a fault in the matcher."
            ),
        },
        "next_step": {
            "type": "string",
            "maxLength": 400,
            "description": (
                "The single most useful thing a person can do next, phrased as an "
                "action with an owner. Prefer 'ask the gateway for X' over 'investigate'."
            ),
        },
        # The two verifiable fields. Named as lists of primitives rather than free text
        # so verify.py can compare them exactly, instead of regexing prose and arguing
        # about what counts as a number.
        "cited_row_ids": {
            "type": "array",
            "maxItems": 20,
            "items": {"type": "string"},
            "description": (
                "Every row, settlement, payment or refund id you referred to. Copy them "
                "exactly as given. Do not include an id you were not shown."
            ),
        },
        "cited_amounts_paise": {
            "type": "array",
            "maxItems": 20,
            "items": {"type": "integer"},
            "description": (
                "Every rupee figure you referred to, in PAISE, as an integer. Every one "
                "must appear in the rows you were given. Do not compute new totals -- if "
                "a figure is not in the input, do not cite it."
            ),
        },
    },
}


def output_config() -> dict[str, Any]:
    """The ``output_config`` value for a per-group explanation request.

    Wrapped in a function rather than exposed as a constant so the caller cannot mutate
    the module-level schema in place: ``client.py`` sends this on every request, and a
    dict quietly edited by one call site would change what every later call asks for.

    **The copy is not decoration, and the self-check below found that the hard way.** The
    first version of this function returned the module constant by reference, so the
    wrapper delivered none of the isolation this docstring claims -- and the assertion
    that says so failed on its first run. Deep, because the schema nests two levels and a
    shallow copy would still share ``properties``.
    """
    return {
        "format": {
            "type": "json_schema",
            "name": "exception_group_explanation",
            "schema": copy.deepcopy(EXPLANATION_SCHEMA),
        }
    }


def _self_check() -> None:
    """The schema must be internally consistent, and strict in both directions."""
    props = EXPLANATION_SCHEMA["properties"]
    required = EXPLANATION_SCHEMA["required"]

    assert EXPLANATION_SCHEMA["additionalProperties"] is False, (
        "additionalProperties must be false, or the model may return fields nothing "
        "verifies -- which is how an unchecked claim gets into the report."
    )
    assert set(required) == set(props), (
        f"required and properties disagree: required-only={set(required) - set(props)}, "
        f"properties-only={set(props) - set(required)}. A property that is not required "
        f"can be silently omitted, and a required name that is not a property is a 400."
    )
    for name, spec in props.items():
        assert "description" in spec, f"{name} has no description -- the model reads these"
        if spec["type"] == "string":
            assert "maxLength" in spec, f"{name} is an unbounded string"
        if spec["type"] == "array":
            assert "maxItems" in spec, f"{name} is an unbounded array"

    # The two verifiable fields must stay verifiable: ids as strings, amounts as
    # integers. Amounts as strings would let "1,234" and "1234" both arrive and compare
    # unequal to the same paise figure, and this project is integer-paise throughout.
    assert props["cited_amounts_paise"]["items"]["type"] == "integer", (
        "cited amounts must be integers in paise. A float or a formatted string would "
        "reintroduce the rounding ambiguity the whole codebase avoids by using paise."
    )
    assert props["cited_row_ids"]["items"]["type"] == "string"

    cfg = output_config()
    assert cfg["format"]["type"] == "json_schema"
    assert cfg["format"]["name"], "the format needs a name; the API rejects an empty one"

    # Identity, at both levels. The nested check is the one that matters: a shallow
    # copy passes the outer assertion and still shares ``properties``, which is the
    # dict a caller would actually reach into.
    #
    # The first draft of this block read `is not EXPLANATION_SCHEMA or True`, which
    # cannot fail for any input -- a check that reports its own silence. It is written
    # out here because a vacuous assertion beside three real ones is the hardest kind
    # to notice on review.
    assert cfg["format"]["schema"] is not EXPLANATION_SCHEMA, (
        "output_config() returned the module-level schema by reference"
    )
    assert cfg["format"]["schema"]["properties"] is not EXPLANATION_SCHEMA["properties"], (
        "output_config() copied only the top level, so callers still share `properties`"
    )

    # And the behavioural version of the same claim: mutating what we handed out must
    # not reach the constant.
    cfg["format"]["schema"]["properties"].pop("summary", None)
    assert "summary" in EXPLANATION_SCHEMA["properties"], (
        "output_config() handed out a reference to the module-level schema, so one "
        "caller's edit changes what every later request asks for. Return a copy."
    )
    # A second call must be unaffected by the first call's mutation.
    assert "summary" in output_config()["format"]["schema"]["properties"], (
        "the first caller's edit survived into the next request's schema"
    )
    print(
        f"schema: ok -- {len(props)} fields, strict both ways, every field bounded, "
        f"citations typed, config isolated at both levels"
    )


if __name__ == "__main__":
    _self_check()
