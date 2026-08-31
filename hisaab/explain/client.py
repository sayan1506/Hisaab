"""The only module here that touches a network, and the only one that imports the SDK.

Everything else in ``hisaab/explain`` is a pure function over data, which is what lets
gate 17 exercise the whole pipeline against a recorded fixture with nothing installed. The
untestable part of an LLM feature should be one seam, and this file is it.

**The import is lazy, and it must not use ``importlib``.** ``check_isolation.py`` check 8
bans ``importlib`` *everywhere* under ``hisaab/``, this package included, because
``importlib.import_module("anthropic")`` would defeat the AST scan that enforces the rest
of the ban. So the deferred import is a plain ``import anthropic`` inside the function --
the constraint shaping the code rather than being worked around, which is the point of
having written the check first.

Lazy because ``anthropic`` is an *optional* extra (``pip install -e ".[llm]"``). Importing
it at module scope would make ``python -m hisaab.explain --help`` die with a traceback on a
clean checkout, and would make this module unimportable in the gate that tests it offline.

**Three things measured in Phase 9 that this file is built around:**

  * The endpoint here is **not** the public API -- this shell sets
    ``ANTHROPIC_BASE_URL=http://localhost:9000``, and hardcoding ``api.anthropic.com``
    yields a 401 that reads as a bad key. So the base URL comes from the environment and
    is *reported*, never assumed.
  * The model **fenced its JSON despite being told not to**. Structured output
    (``output_config.format``) retires the tolerant extractor that would otherwise be
    needed -- there is nothing to strip.
  * **No token count measured here is the product's**, because the endpoint is not the
    public API. That is the whole of the caveat, and it needs no multiplier to stand.
    Phase 9 measured ``in=5201`` for a 578-char payload and read it as a large injected
    system prompt; Phase 10 sent a real request (prefix + rows, ~1,992 tokens estimated)
    and got ``in=1915`` -- about 1.0x. An injected prefix is *additive*, so ~5,000 extra
    tokens would have shown up here as ~7,000 rather than 1,915. So the 35x does not
    reproduce on this path and is not repeated (ASSUMPTIONS.md #41c).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from . import prompt as prompt_mod
from . import schema as schema_mod

#: The model, pinned. Not configurable by default and not inferred: a run whose model
#: silently changed is a run whose output is not comparable to the last one, and this
#: project's entire argument is that its numbers are reproducible. Overridable by explicit
#: CLI flag only, so the override appears in the recorded artifact.
DEFAULT_MODEL = "claude-opus-5"

#: Generous relative to the schema's own caps (~1,400 chars of prose, two short arrays),
#: because on this model thinking is on by default and its tokens count against this
#: ceiling. Hitting the cap truncates mid-structure and costs a retry.
MAX_TOKENS = 8192

#: The published rate for ``DEFAULT_MODEL``, USD per million tokens, for step 9's
#: arithmetic. Declared here rather than computed from a bill, and **flagged in
#: ASSUMPTIONS.md as unmeasurable in this environment**: the proxy's injected prefix makes
#: every local token count unrepresentative, so the arithmetic is shown with the rate
#: stated and the count marked as not-this-project's.
USD_PER_MTOK_INPUT = 5.00
USD_PER_MTOK_OUTPUT = 25.00


class ExplainError(Exception):
    """A model call could not be made, or its result could not be trusted."""


@dataclass(frozen=True, slots=True)
class Usage:
    """What one call consumed. ``cache_read`` is the number step 8 exists to check.

    **The cache fields are ``int | None``, and the distinction is the point.** ``None`` means
    the endpoint did not report cache telemetry at all; ``0`` means it reported none used.
    Measured live against this shell's proxy: both fields came back absent, and the first
    version of this dataclass typed them ``int`` and read them with ``or 0`` -- which turned
    "no telemetry" into "zero tokens cached" and let the caller conclude the prefix was below
    the minimum cacheable size. That conclusion does not follow from silence, and a figure
    that cannot distinguish "not measured" from "measured zero" is the shape of defect this
    project keeps finding.
    """

    input_tokens: int
    output_tokens: int
    cache_creation: int | None
    cache_read: int | None

    @property
    def reports_cache(self) -> bool:
        """Whether the endpoint said anything at all about caching."""
        return self.cache_creation is not None or self.cache_read is not None

    @property
    def usd(self) -> float:
        return (
            self.input_tokens / 1_000_000 * USD_PER_MTOK_INPUT
            + self.output_tokens / 1_000_000 * USD_PER_MTOK_OUTPUT
        )


@dataclass(frozen=True, slots=True)
class Explanation:
    """One group's explanation, exactly as the schema constrains it, plus provenance."""

    group_reason: str
    summary: str
    why_unresolved: str
    next_step: str
    cited_row_ids: tuple[str, ...]
    cited_amounts_paise: tuple[int, ...]
    model: str
    usage: Usage

    def as_dict(self) -> dict[str, Any]:
        """The four schema fields, for ``verify.verify`` and the recorded artifact."""
        return {
            "summary": self.summary,
            "why_unresolved": self.why_unresolved,
            "next_step": self.next_step,
            "cited_row_ids": list(self.cited_row_ids),
            "cited_amounts_paise": list(self.cited_amounts_paise),
        }


def base_url() -> str:
    """Where requests actually go, read from the environment and never assumed.

    Returned as a string for reporting rather than passed silently to the SDK: a run
    against a proxy and a run against the public API produce different token counts, so the
    artifact records which one it was.
    """
    return os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"


def _client() -> Any:
    """Construct the SDK client, or explain precisely what is missing.

    The plain in-function import is deliberate -- see the module docstring on why
    ``importlib`` is unavailable here by design.
    """
    try:
        import anthropic  # noqa: PLC0415 -- lazy on purpose; it is an optional extra
    except ModuleNotFoundError as e:
        raise ExplainError(
            "the `anthropic` package is not installed. It is an optional extra, because "
            "the rest of Hisaab -- generator, matcher, scorer, exception queue and every "
            "acceptance gate -- runs on the standard library alone:\n"
            "    pip install -e \".[llm]\"\n"
            "  Nothing outside hisaab/explain needs it, and check_isolation.py check 8 "
            "fails the build if anything outside this package imports it."
        ) from e

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise ExplainError(
            "no API credential in the environment (ANTHROPIC_API_KEY or "
            "ANTHROPIC_AUTH_TOKEN). This tool reads keys from the environment only and "
            "never from a file -- .env and *.pem are gitignored for that reason."
        )
    # base_url is left to the SDK's own env handling rather than passed explicitly, so
    # there is exactly one place that resolves it. base_url() above reports it.
    return anthropic.Anthropic()


def _cache_field(usage: Any, name: str) -> int | None:
    """A cache counter, preserving the difference between absent and zero.

    ``or 0`` would be wrong twice here: it turns ``None`` into ``0``, and it also turns a
    genuine ``0`` into ``0`` -- so the caller cannot tell "this endpoint reports no cache
    telemetry" from "caching was available and nothing hit". The first is a fact about the
    endpoint; the second is a fact about the prefix, and only the second says anything about
    whether the breakpoint works.
    """
    value = getattr(usage, name, None)
    return int(value) if value is not None else None


def _usage_of(response: Any) -> Usage:
    u = getattr(response, "usage", None)
    return Usage(
        input_tokens=getattr(u, "input_tokens", 0) or 0,
        output_tokens=getattr(u, "output_tokens", 0) or 0,
        cache_creation=_cache_field(u, "cache_creation_input_tokens"),
        cache_read=_cache_field(u, "cache_read_input_tokens"),
    )


def _payload_of(response: Any) -> dict[str, Any]:
    """The parsed object, having first refused anything that is not a clean answer.

    **A refusal arrives as HTTP 200.** ``stop_reason == "refusal"`` means safety
    classifiers declined the request; the call succeeded and ``content`` is not an answer.
    Reading content before checking this is how a refusal becomes an empty explanation that
    looks like a model with nothing to say.
    """
    stop = getattr(response, "stop_reason", None)
    if stop == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None)
        raise ExplainError(
            f"the model declined this request (stop_reason=refusal, category={category!r}). "
            f"The call itself succeeded -- this is not a transport error. Nothing was "
            f"explained, so the row keeps its templated hint rather than acquiring prose "
            f"nobody wrote."
        )
    if stop == "max_tokens":
        raise ExplainError(
            f"the response hit max_tokens ({MAX_TOKENS}) and is truncated, so the JSON is "
            f"incomplete. Raise MAX_TOKENS rather than parsing a fragment."
        )

    parsed = getattr(response, "parsed_output", None)
    if parsed is None:
        parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return parsed if isinstance(parsed, dict) else dict(parsed)

    # Fall back to the text block. Reached when the SDK returns an unparsed message --
    # still strict JSON, because output_config constrains the response; this is a shape
    # difference between SDK versions, not the fenced-JSON tolerance Phase 9 predicted.
    import json  # noqa: PLC0415 -- only needed on this path

    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            try:
                return json.loads(block.text)
            except json.JSONDecodeError as e:
                where = base_url()
                proxied = where != "https://api.anthropic.com"
                raise ExplainError(
                    f"the response was not the JSON the schema constrained it to: {e}.\n"
                    f"  First 120 chars: {block.text[:120]!r}\n"
                    + (
                        f"  **{where} is not the public API, and a proxy that does not "
                        f"implement structured output drops `output_config` silently** -- "
                        f"measured here: a deliberately invalid format value was ACCEPTED "
                        f"rather than rejected with a 400, and the same request with and "
                        f"without the schema returned the same markdown prose. So the "
                        f"likeliest reading is that the model never received the schema and "
                        f"formatted freely, not that it violated one.\n"
                        f"  Check the endpoint before changing any code here. Adding a "
                        f"tolerant parser would make this pass by accepting free-form output "
                        f"from the real API too, which is the one thing the schema exists to "
                        f"prevent.\n"
                        if proxied else
                        f"  output_config.format was sent to the public API, which enforces "
                        f"it, so this is a real protocol failure rather than the model "
                        f"formatting freely -- do not add a tolerant parser to paper over it.\n"
                    )
                ) from e
    raise ExplainError("the response carried no parsed output and no text block")


def explain_group(
    group: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
) -> Explanation:
    """Explain one exception group. The only function here that sends a request.

    ``client`` is injectable so a caller can drive this with a recorded double -- gate 17
    uses that to run the pipeline end to end with no network and nothing installed.
    """
    request = prompt_mod.build_request(group)
    api = client if client is not None else _client()

    try:
        response = api.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=request["system"],
            messages=request["messages"],
            output_config=schema_mod.output_config(),
        )
    except ExplainError:
        raise
    except Exception as e:  # noqa: BLE001 -- narrowed immediately below
        raise _translate(e) from e

    payload = _payload_of(response)
    missing = [k for k in schema_mod.EXPLANATION_SCHEMA["required"] if k not in payload]
    if missing:
        raise ExplainError(
            f"the response is missing required field(s) {missing} despite a strict schema "
            f"with additionalProperties=false. Do not fill them in with defaults -- an "
            f"invented field is exactly what the citation check exists to catch."
        )

    return Explanation(
        group_reason=str(group.get("reason")),
        summary=str(payload["summary"]),
        why_unresolved=str(payload["why_unresolved"]),
        next_step=str(payload["next_step"]),
        cited_row_ids=tuple(str(x) for x in payload["cited_row_ids"]),
        cited_amounts_paise=tuple(int(x) for x in payload["cited_amounts_paise"]),
        model=model,
        usage=_usage_of(response),
    )


def _translate(e: Exception) -> ExplainError:
    """Turn an SDK exception into one sentence a person can act on.

    Most-specific-first, because a single broad handler loses the distinction that matters:
    a 429 or a connection error is worth retrying, a 404 model id never is. The SDK is
    imported here rather than at module scope for the same reason as everywhere else in
    this file.
    """
    try:
        import anthropic  # noqa: PLC0415
    except ModuleNotFoundError:  # pragma: no cover -- unreachable past _client()
        return ExplainError(f"model call failed: {e}")

    where = base_url()
    if isinstance(e, anthropic.NotFoundError):
        return ExplainError(
            f"the endpoint or model was not found at {where} ({e}). If that base URL is a "
            f"proxy, it may not carry the pinned model -- Phase 9 measured that hardcoding "
            f"api.anthropic.com in this shell yields a 401 that reads as a bad key, so the "
            f"reverse confusion is live too."
        )
    if isinstance(e, anthropic.AuthenticationError):
        return ExplainError(
            f"authentication was rejected by {where}. When ANTHROPIC_BASE_URL points at a "
            f"proxy, the key in the environment must be that proxy's token, not an "
            f"api.anthropic.com key."
        )
    if isinstance(e, anthropic.RateLimitError):
        return ExplainError(f"rate limited by {where} ({e}). Retryable; the SDK already retried.")
    if isinstance(e, anthropic.APIConnectionError):
        return ExplainError(
            f"could not reach {where} ({e}). Nothing here falls back to a canned "
            f"explanation: a row with no explanation keeps its templated hint, which is "
            f"honest, whereas invented prose is not."
        )
    if isinstance(e, anthropic.APIStatusError):
        return ExplainError(f"{where} returned HTTP {e.status_code}: {e}")
    return ExplainError(f"model call failed against {where}: {e}")


def count_tokens(group: dict[str, Any], *, model: str = DEFAULT_MODEL, client: Any | None = None) -> int:
    """Token count for one group's request, for step 9's arithmetic.

    **Not this project's figure, because the endpoint is not the public API.** That is the
    caveat, and it stands without a multiplier: a count from a proxy is a count from a proxy.

    Phase 9 read ``in=5201`` for a 578-char payload as a large injected system prompt, and
    Phase 10 did not reproduce it -- a real request estimated at ~1,992 tokens reported
    ``in=1915``, about 1.0x, where an *additive* injection of ~5,000 tokens would have
    reported ~7,000. So the inflation is not asserted here (ASSUMPTIONS.md #41c). What is
    still true is that this number is reported with its base URL attached and is never
    recorded as this project's cost per row.
    """
    request = prompt_mod.build_request(group)
    api = client if client is not None else _client()
    try:
        result = api.messages.count_tokens(
            model=model, system=request["system"], messages=request["messages"]
        )
    except Exception as e:  # noqa: BLE001
        raise _translate(e) from e
    return int(getattr(result, "input_tokens", 0) or 0)


def _self_check() -> None:
    """Everything except the request itself, driven by a recorded double.

    A double rather than a mock framework: this asserts the shapes this module actually
    reads off a response, and a hand-built object makes those explicit.
    """

    class _Usage:
        input_tokens, output_tokens = 1700, 210
        cache_creation_input_tokens, cache_read_input_tokens = 0, 1650

    class _Block:
        type = "text"

        def __init__(self, text: str) -> None:
            self.text = text

    class _Response:
        def __init__(self, payload: str, stop: str = "end_turn") -> None:
            self.content = [_Block(payload)]
            self.stop_reason = stop
            self.usage = _Usage()
            self.parsed_output = None

    class _Messages:
        def __init__(self, response: Any) -> None:
            self._r = response
            self.seen: dict[str, Any] = {}

        def create(self, **kwargs: Any) -> Any:
            self.seen = kwargs
            if isinstance(self._r, Exception):
                raise self._r
            return self._r

        def count_tokens(self, **kwargs: Any) -> Any:
            self.seen = kwargs
            return type("T", (), {"input_tokens": 5201})()

    class _Api:
        def __init__(self, response: Any) -> None:
            self.messages = _Messages(response)

    group = {
        "reason": "UNEXPLAINED_RESIDUAL", "cause": "c", "rows": 1, "value_paise": 19074,
        "credits": [{"credit_id": "C0101", "bank_amount_paise": 19074, "note": "setl_0164 43p"}],
    }
    good = (
        '{"summary":"s","why_unresolved":"w","next_step":"n",'
        '"cited_row_ids":["C0101"],"cited_amounts_paise":[19074]}'
    )

    api = _Api(_Response(good))
    result = explain_group(group, client=api)
    assert result.summary == "s" and result.cited_amounts_paise == (19074,)
    assert result.model == DEFAULT_MODEL, "the pinned model must reach the request"
    assert result.usage.cache_read == 1650, "cache_read must survive; step 8 reads it"
    assert result.usage.reports_cache, "a response carrying both fields reports caching"

    # **The absent case, which is the one this shell's proxy actually produces.** Measured
    # live: `cache_creation_input_tokens` and `cache_read_input_tokens` came back missing
    # entirely, not zero. `_usage_of` read them with `or 0` until then, so "the endpoint said
    # nothing" and "nothing was cached" arrived at the caller as the same integer -- and the
    # step-8 report concluded the prefix was below the minimum cacheable size on the strength
    # of it. Three assertions, because collapsing any two of these states loses the finding.
    class _UsageNoCache:
        input_tokens, output_tokens = 1915, 658  # the live figures, for the record

    class _ResponseNoCache(_Response):  # type: ignore[misc,valid-type]
        def __init__(self, payload: str) -> None:
            super().__init__(payload)
            self.usage = _UsageNoCache()

    absent = explain_group(group, client=_Api(_ResponseNoCache(good))).usage
    assert absent.cache_read is None and absent.cache_creation is None, (
        f"an absent cache field must stay None, got read={absent.cache_read!r} "
        f"write={absent.cache_creation!r} -- `or 0` here is what let silence read as zero"
    )
    assert not absent.reports_cache, "a response with neither field reports no caching"
    assert absent.input_tokens == 1915, "the non-cache counters must still be read"

    # And zero must remain distinguishable from absent, or the fix only moved the problem.
    class _UsageZero:
        input_tokens, output_tokens = 1915, 658
        cache_creation_input_tokens, cache_read_input_tokens = 0, 0

    class _ResponseZero(_Response):  # type: ignore[misc,valid-type]
        def __init__(self, payload: str) -> None:
            super().__init__(payload)
            self.usage = _UsageZero()

    zero = explain_group(group, client=_Api(_ResponseZero(good))).usage
    assert zero.cache_read == 0 and zero.reports_cache, (
        "a reported zero must read as zero AND as reported -- it is a fact about the prefix, "
        "whereas an absent field is a fact about the endpoint"
    )

    # The request must carry the schema and the cache breakpoint -- the two things that
    # would silently degrade into free-form output and an uncached prefix.
    sent = api.messages.seen
    assert sent["output_config"]["format"]["type"] == "json_schema", (
        "the request went without a schema, so the model was free to format as it liked "
        "-- which Phase 9 measured it doing (fenced JSON)"
    )
    assert any("cache_control" in b for b in sent["system"]), "no cache breakpoint was sent"
    assert sent["model"] == DEFAULT_MODEL and sent["max_tokens"] == MAX_TOKENS

    def refuses(response: Any, expect: str, label: str) -> None:
        try:
            explain_group(group, client=_Api(response))
        except ExplainError as e:
            assert expect in str(e), f"{label}: wrong reason -- {e}"
            return
        raise AssertionError(f"{label}: accepted what it should refuse")

    # A refusal is HTTP 200 with a stop_reason. Reading content first would turn it into
    # an empty explanation that looks like a model with nothing to say.
    refuses(_Response(good, stop="refusal"), "declined this request", "refusal")
    refuses(_Response(good, stop="max_tokens"), "truncated", "truncation")
    refuses(_Response("not json at all"), "not the JSON the schema", "malformed")
    refuses(
        _Response('{"summary":"s","why_unresolved":"w","next_step":"n","cited_row_ids":[]}'),
        "missing required field", "short payload",
    )

    # A refusal must NOT be mistaken for success even though its payload parses.
    assert count_tokens(group, client=_Api(_Response(good))) == 5201

    u = Usage(input_tokens=1_000_000, output_tokens=0, cache_creation=0, cache_read=0)
    assert abs(u.usd - USD_PER_MTOK_INPUT) < 1e-9, "the cost arithmetic is wrong"

    print(
        f"client: ok -- pinned {DEFAULT_MODEL}, schema and cache breakpoint both sent, "
        f"4 bad responses refused by their own reason, cache telemetry absent/zero/present "
        f"all distinguished, base_url={base_url()}"
    )


if __name__ == "__main__":
    _self_check()
