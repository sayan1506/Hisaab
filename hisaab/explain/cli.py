"""The deliverable: explain the exception queue, and check every claim it makes.

    python -m hisaab.explain --fixture              # recorded input, one call per group
    python -m hisaab.explain --fixture --dry-run    # build prompts, send nothing
    python -m hisaab.explain --matches out/matches.json --data data/
    python -m hisaab.explain --fixture --ask 'C0001:why is the credit less than the gross?'

``--dry-run`` is the mode that makes this reviewable: it builds every request, prints the
token arithmetic and the cache layout, and sends nothing. A reader can see exactly what
would be transmitted before any money is spent, and gate 17 runs in it.

``--ask`` is step 7's other half, and it answers a different question over a different row
set. Explaining the queue is a batch pass over EXCEPTIONS; ``--ask`` is one question over
one RESOLVED row, checked against that row's own decomposition rather than by containment
alone -- see ``qa.py``'s docstring for why a resolved row is the only place that check can
run. The two never overlap: a row is in exactly one of the two sets.

**Three things this prints that are not the explanation itself**, each because a number
without its check is the failure this project keeps finding:

  * **The citation check** (step 5) -- every id and figure in the generated text, held
    against the rows that were sent. Reported per group, and a fabrication is fatal.
  * **The hint comparison** (step 6) -- the model's ``next_step`` beside the reconciler's
    own declared action for that code. Divergence is informative in both directions and is
    reported, never resolved automatically: the templated hint is the committed answer.
  * **Cache and cost** (steps 8, 9) -- ``cache_read_input_tokens`` measured rather than
    assumed, and the cost arithmetic shown with the base URL attached, because through a
    proxy no token count here is this product's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..common.reasons import Reason
from ..triage.hint import HINTS
from . import EXPLAIN_SCHEMA_VERSION
from . import client as client_mod
from . import prompt as prompt_mod
from . import qa as qa_mod
from . import verify as verify_mod

FIXTURE_PATH = Path("fixtures/explain/fixture.json")


class UsageError(Exception):
    """Bad invocation, or an input that cannot be explained."""


def _utf8_stdout() -> None:
    """Make the rupee sign survive a pipe on Windows -- a fifth copy of stdlib boilerplate.

    Left as a copy for the reason ``triage/cli.py`` states: there is no quantity in it, so
    two copies cannot disagree about an answer the way two effort tables could.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover -- already utf-8
            pass


def groups_from_fixture(path: Path) -> list[dict[str, Any]]:
    """Every group across every cell of the recorded fixture.

    Cells are kept distinct in the artifact but flattened here: a group is a group, and
    the cell only records which flag set produced it.
    """
    if not path.exists():
        raise UsageError(
            f"{path.as_posix()} is missing. Build it with `python tools/explain_fixture.py`."
        )
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for label, cell in sorted(doc["cells"].items()):
        for group in cell["groups"]:
            out.append({**group, "_cell": label})
    return out


def resolved_rows_from_fixture(path: Path) -> list[dict[str, Any]]:
    """Every RESOLVED row across every cell's ``resolved_sample``.

    The counterpart of ``groups_from_fixture`` for step 7's other half: the queue holds
    exceptions, so a RESOLVED row for the Q&A check has to come from a separate frozen
    sample rather than from ``groups``. See ``tools/explain_fixture.py``'s docstring on why
    that sample exists and what it is chosen for.
    """
    if not path.exists():
        raise UsageError(
            f"{path.as_posix()} is missing. Build it with `python tools/explain_fixture.py`."
        )
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for label, cell in sorted(doc["cells"].items()):
        for row in cell.get("resolved_sample", ()):
            out.append({**row, "_cell": label})
    return out


def groups_from_live(matches: Path, data: Path) -> list[dict[str, Any]]:
    """The same shape, built from a real run rather than the recording.

    Goes through ``triage`` for the grouping and ranking rather than regrouping here: the
    queue's partition is Phase 9's answer, asserted by its own gates, and a second
    implementation of it would be free to disagree.

    But it reads ``matches.json`` **directly** for the notes, because ``Ruling`` does not
    carry one -- ``triage/read.py`` never parses the field (deliberately, per its own
    docstring on keeping a narrow record). The note is the only populated field on an
    exception row, so without this second read there is nothing to explain.
    """
    from ..common.verdict import MATCHES_JSON
    from ..triage.group import group as group_rulings
    from ..triage.read import load_rulings
    from ..triage.value import amounts, rank

    rulings = load_rulings(matches)
    groups = group_rulings(rulings)
    by_id = amounts(data)
    ranked = rank(groups, by_id, rulings)

    # load_rulings accepts a directory or a file; this second read must resolve it the same
    # way, or pointing --matches at a directory would crash here having worked there.
    verdict_file = matches / MATCHES_JSON if matches.is_dir() else matches
    verdicts = json.loads(verdict_file.read_text(encoding="utf-8"))["verdicts"]
    notes = {v["credit_id"]: (v.get("note") or "") for v in verdicts}

    out: list[dict[str, Any]] = []
    for rg in ranked:
        out.append({
            "cause": rg.label,
            "kind": str(rg.group.kind),
            "reason": str(rg.group.reason) if rg.group.reason is not None else None,
            "rows": rg.count,
            "value_paise": rg.value_paise,
            "credits": [
                {
                    "credit_id": item.credit_id,
                    "bank_amount_paise": item.value_paise,
                    "note": notes.get(item.credit_id, ""),
                    "reason": str(item.ruling.reason) if item.ruling.reason else None,
                }
                for item in rg.items
            ],
            "_cell": "live",
        })
    return out


def compare_to_hint(group: dict[str, Any], next_step: str) -> dict[str, Any]:
    """Step 6: the model's next step beside the reconciler's declared one.

    **Reported, never resolved.** The templated hint is the answer this project committed to
    and cited to the line that raises the code; the model's is generated. Automatically
    preferring either would throw away the only signal here.

    **The number this returns is a copying indicator, not an agreement score, and the
    difference was measured rather than reasoned about.** Driving all 7 coded fixture groups
    through three kinds of ``next_step``:

        next_step                                     reproduced
        the hint's own text, verbatim                 1.00 on all 7
        a correct paraphrase ("ask the gateway to
          publish the missing figure")                0.00 - 0.08
        a flat contradiction ("nothing to do;
          ignore these rows")                         0.00 on all 7

    A correct paraphrase and a direct contradiction land in the same place. So the metric
    separates *copying* from *not copying* and nothing else, and printing it as "% agreement"
    would tell a reader that a right answer in different words was a disagreement. No lexical
    measure available here does better, and an LLM judging another LLM's output would add a
    second unverifiable claim to check the first -- so **the agreement claim is withheld**,
    the way ASSUMPTIONS.md #38's time-saving claim was withheld rather than re-rated, and
    both texts are printed for a person to read.

    **There is also a confound in the prompt itself, and it runs the other way from what I
    first wrote here.** ``prompt.py`` puts all 13 declared hints in the *cached prefix*, so
    the model has been shown this group's committed action before it answers. Its
    ``next_step`` is therefore not an independent opinion, which means agreement is expected
    and carries almost no information -- while divergence is the interesting case, because it
    means the model read the declared action and said something else anyway. That is worth a
    human's eye. Agreement is not.
    """
    reason = group.get("reason")
    hint = None
    if reason:
        try:
            hint = HINTS.get(Reason(reason))
        except ValueError:  # pragma: no cover -- reason comes from our own enum
            hint = None
    if hint is None:
        return {"has_hint": False, "reason": reason}

    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "it", "this",
        "that", "with", "from", "by", "at", "as", "be", "was", "which", "its", "their",
        "not", "no", "if", "so", "than", "then", "them", "has", "have", "had", "are",
    }
    def words(text: str) -> set[str]:
        return {
            w.strip(".,;:()'\"").lower()
            for w in text.split()
            if len(w.strip(".,;:()'\"")) > 3
        } - stop

    hint_words, model_words = words(hint.action), words(next_step)
    shared = hint_words & model_words
    return {
        "has_hint": True,
        "reason": reason,
        "hint_action": hint.action,
        "model_next_step": next_step,
        "shared_terms": sorted(shared),
        # Named for what it measures. It was called ``overlap`` until the table in this
        # function's docstring was measured, and that name invited exactly the reading the
        # measurement rules out: a correct paraphrase scores what a contradiction scores, so
        # this is the fraction of the hint's content words REPRODUCED, nothing more.
        "hint_terms_reproduced": round(len(shared) / len(hint_words), 3) if hint_words else 0.0,
        "agreement_scored": False,
        "why_not_scored": (
            "A correct paraphrase and a flat contradiction both score ~0 on any lexical "
            "measure available here, and the model is shown the declared hint in the cached "
            "prefix, so reproduction is not independent agreement. Both texts are printed "
            "for a person to compare."
        ),
    }


def _dry_run(groups: list[dict[str, Any]], model: str) -> int:
    """Build every request, send nothing, and report what would be sent.

    The prefix-length figure is the one to read: prompt caching has a model-dependent
    minimum cacheable prefix, so a prefix below it never caches however correctly the
    breakpoint is placed. Printing the number beside the claim keeps step 8 a measurement.
    """
    system = prompt_mod.system_blocks()
    prefix_chars = sum(len(b["text"]) for b in system)
    print(f"explain: DRY RUN -- {len(groups)} group(s), nothing will be sent")
    print(f"  model (pinned)      : {model}")
    print(f"  endpoint            : {client_mod.base_url()}")
    print(f"  cached prefix       : {prefix_chars:,} chars (~{prefix_chars // 4:,} tokens), "
          f"1 breakpoint")
    print(f"    a prefix below the model's minimum cacheable size never caches, however "
          f"the breakpoint is placed -- so this figure is printed, not assumed")
    print(f"  rows sent per group : at most {prompt_mod.ROWS_PER_GROUP} (largest by value)")

    total_chars = 0
    print(f"\n  {'group':<30} {'rows':>5} {'sent':>5} {'chars':>7}  universe")
    for group in groups:
        msg = prompt_mod.group_message(group)
        ids, amounts = prompt_mod.cited_universe(group)
        sent = min(group["rows"], prompt_mod.ROWS_PER_GROUP)
        total_chars += len(msg)
        label = str(group.get("reason") or group["cause"])[:29]
        print(f"  {label:<30} {group['rows']:>5} {sent:>5} {len(msg):>7}  "
              f"{len(ids)} id / {len(amounts)} amt")

    # The prefix is sent on EVERY request, so the uncached total carries it n times. The
    # first version of this block divided `prefix_chars + total_chars` once and then
    # reported the re-sent prefix as a component "of that" -- which printed 12,390 tokens as
    # part of a total of 11,536. A part larger than its whole, visible in the output, and it
    # got there the same way ASSUMPTIONS.md #38 did: two figures printed side by side with
    # nothing asserting the relationship between them.
    n = len(groups)
    uncached_tokens = (prefix_chars * n + total_chars) // 4
    cached_tokens = (prefix_chars + total_chars) // 4
    resent = uncached_tokens - cached_tokens
    assert 0 <= resent <= uncached_tokens, (
        f"the re-sent prefix ({resent}) is not a component of the total ({uncached_tokens})"
    )

    print(f"\n  volatile payload    : {total_chars:,} chars across {n} request(s)")
    print(f"  prefix, re-sent     : {prefix_chars:,} chars x {n} requests "
          f"= {prefix_chars * n:,} chars")
    print(f"  input if nothing caches: ~{uncached_tokens:,} tokens")
    # "if the prefix caches" rather than "if all N hit": the first request cannot read a
    # cache it is writing, so cached_tokens counts the prefix once -- for that first write --
    # and n-1 reads at zero. Saying "all N hit" would be wrong by one request.
    print(f"  input if it caches  : ~{cached_tokens:,} tokens "
          f"(~{resent:,} fewer -- {resent / uncached_tokens:.0%} of the input is prefix "
          f"re-sent on the {n - 1} request(s) after the first)")
    print(f"\n  at the published rate for {model} "
          f"(${client_mod.USD_PER_MTOK_INPUT:.2f}/${client_mod.USD_PER_MTOK_OUTPUT:.2f} "
          f"per MTok in/out):")
    print(f"    ~${uncached_tokens / 1_000_000 * client_mod.USD_PER_MTOK_INPUT:.4f} input "
          f"for the whole queue with no cache, before output")
    print(f"    an UPPER BOUND on the input line, deliberately: every token above is priced "
          f"at the full input rate. Cache writes and reads are billed at their own "
          f"multipliers, which this project has not verified for {model}, so they are not "
          f"modelled rather than guessed at.")
    print(f"    ~4 chars/token is an estimate, not a tokenisation. count_tokens() would give "
          f"a real one, but not from this endpoint: {client_mod.base_url()} is not the public "
          f"API, so its counts are not this project's either way (ASSUMPTIONS.md #41c).")
    return 0


def run(
    groups: list[dict[str, Any]],
    *,
    model: str,
    out: Path | None,
    strict: bool = True,
    client: Any | None = None,
) -> int:
    """Explain every group, check every claim, and report what it cost.

    ``client`` is injectable for the same reason ``explain_group``'s is, and it is what makes
    gate 17 possible: without it this function calls ``_client()`` unconditionally, so the
    only way to exercise the citation check, the artifact and the exit code would be to spend
    money on a live call -- which would also make the gate fail on a clean checkout with no
    key. A gate that cannot run offline is a gate that gets skipped.
    """
    explanations: list[dict[str, Any]] = []
    failures: list[str] = []
    total_in = total_out = total_cached = 0
    #: How many responses said anything at all about caching. Counted separately from
    #: ``total_cached`` so step 8 can tell "no cache hits" from "no cache telemetry" -- the
    #: proxy in this shell reports neither field, and reading that as zero hits would blame
    #: the prefix for the endpoint's silence.
    cache_reported = 0

    where = "a recorded double" if client is not None else client_mod.base_url()
    print(f"explain: {len(groups)} group(s) via {where}, model {model}")
    api = client if client is not None else client_mod._client()

    for i, group in enumerate(groups, 1):
        label = str(group.get("reason") or group["cause"])
        try:
            result = client_mod.explain_group(group, model=model, client=api)
        except client_mod.ExplainError as e:
            print(f"\n  [{i}/{len(groups)}] {label}: NOT EXPLAINED -- {e}")
            failures.append(f"{label}: {e}")
            continue

        checked = verify_mod.verify(group, result.as_dict())
        hint_cmp = compare_to_hint(group, result.next_step)
        u = result.usage
        total_in += u.input_tokens
        total_out += u.output_tokens
        # ``None`` means the endpoint reported no cache telemetry, which is not the same as
        # reporting zero -- see ``client.Usage``. Summing it as 0 would let the block below
        # conclude something about the prefix from the endpoint's silence.
        if u.cache_read is not None:
            total_cached += u.cache_read
        if u.reports_cache:
            cache_reported += 1

        print(f"\n  [{i}/{len(groups)}] {label} -- {group['rows']} row(s)")
        print(f"      {result.summary}")
        print(f"      why : {result.why_unresolved}")
        print(f"      next: {result.next_step}")
        print(f"      citations: {checked.summary()}")
        if hint_cmp["has_hint"]:
            # Both texts in full, because the number cannot carry the comparison -- the table
            # in compare_to_hint's docstring is the measurement that establishes that. The
            # duplication is the point: a person reading two next-steps can see whether they
            # agree, which no figure printed here can tell them.
            print(f"      declared hint: {hint_cmp['hint_action']}")
            print(f"      of the hint's terms, {hint_cmp['hint_terms_reproduced']:.0%} are "
                  f"reproduced above -- a COPYING indicator, not an agreement score, and the "
                  f"hint was in the prompt")
        cache_note = (
            f"cache_read={u.cache_read} cache_write={u.cache_creation}"
            if u.reports_cache else "cache: not reported by this endpoint"
        )
        print(f"      tokens: in={u.input_tokens} out={u.output_tokens} {cache_note}")

        if not checked.ok:
            failures.append(checked.summary())

        explanations.append({
            "group_reason": result.group_reason,
            "cause": group["cause"],
            "rows": group["rows"],
            "value_paise": group["value_paise"],
            "cell": group.get("_cell"),
            "explanation": result.as_dict(),
            "citation_check": {
                "ok": checked.ok,
                "checked": checked.checked,
                "findings": [str(f) for f in checked.findings],
            },
            "hint_comparison": hint_cmp,
            "usage": {
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cache_read_input_tokens": u.cache_read,
                "cache_creation_input_tokens": u.cache_creation,
            },
        })

    # --- step 8: caching, and three outcomes rather than two ------------------------
    #
    # The first version of this block had two branches, zero and non-zero, and would have
    # printed "ZERO cache reads ... this is a measurement about the prefix's SIZE" against an
    # endpoint that never mentioned caching at all. Measured live: this shell's proxy returns
    # `cache_creation_input_tokens` and `cache_read_input_tokens` **absent**, not zero. A
    # missing field says nothing about the prefix, and blaming the prefix for the endpoint's
    # silence is the same error as ASSUMPTIONS.md #38 -- a conclusion resting on a number that
    # cannot support it.
    if not cache_reported:
        print(f"\n  cache: NOT REPORTED by {where}. Not zero -- absent: no response carried "
              f"cache_creation_input_tokens or cache_read_input_tokens.")
        print(f"    So this run says nothing about whether the prefix caches. The breakpoint "
              f"is placed and prompt.py asserts the prefix is byte-stable, but 'it saves "
              f"tokens' is unmeasured here and is not claimed.")
    elif total_cached == 0:
        print(f"\n  cache: reported by {cache_reported}/{len(explanations)} response(s), and "
              f"0 of {total_in:,} input tokens were served from cache")
        print(f"    Zero reads WITH telemetry present, across {len(groups)} request(s) sharing "
              f"a byte-identical prefix -- prompt.py asserts that stability, so this is about "
              f"the prefix's SIZE: below this model's minimum cacheable prefix (512-4096 "
              f"tokens, model-dependent), the breakpoint is correct and inert.")
    else:
        print(f"\n  cache: {total_cached:,} of {total_in:,} input tokens served from cache "
              f"({cache_reported}/{len(explanations)} response(s) reported it)")
        print(f"    the stable prefix is being reused, so 'cacheable' is measured here rather "
              f"than asserted")

    # --- step 9: cost arithmetic, with the caveat that makes it honest ------------
    usd = (
        total_in / 1_000_000 * client_mod.USD_PER_MTOK_INPUT
        + total_out / 1_000_000 * client_mod.USD_PER_MTOK_OUTPUT
    )
    print(f"\n  cost: {total_in:,} in + {total_out:,} out at "
          f"${client_mod.USD_PER_MTOK_INPUT:.2f}/${client_mod.USD_PER_MTOK_OUTPUT:.2f} "
          f"per MTok = ${usd:.4f}")
    if client is not None:
        # A recorded double's token counts are whatever the recording says. Attaching the
        # proxy caveat here would be worse than saying nothing: it would imply these figures
        # came off a wire.
        print(f"    these token counts came from a recorded double, not from any endpoint. "
              f"They exercise the arithmetic; they measure nothing about this model.")
    elif client_mod.base_url() != "https://api.anthropic.com":
        print(f"    *** NOT THIS PROJECT'S COST FIGURE. *** These tokens were counted "
              f"through {client_mod.base_url()}, which is not the public API. The rate above "
              f"is the published one and the arithmetic is right; the counts belong to this "
              f"endpoint, so nothing derived from them is quoted as this project's.")
        print(f"    No inflation factor is claimed either way: Phase 9 read 5,201 tokens for "
              f"a 578-char payload as an injected prefix, and Phase 10 did not reproduce it "
              f"(~1,992 estimated, 1,915 reported -- about 1.0x, where an additive injection "
              f"would have shown ~7,000). See ASSUMPTIONS.md #41c.")

    if out is not None:
        doc = {
            "schema_version": EXPLAIN_SCHEMA_VERSION,
            "model": model,
            "endpoint": client_mod.base_url(),
            "groups": len(groups),
            "explained": len(explanations),
            "citations_clean": sum(1 for e in explanations if e["citation_check"]["ok"]),
            "usage_total": {
                "input_tokens": total_in,
                "output_tokens": total_out,
                "cache_read_input_tokens": total_cached,
            },
            "explanations": explanations,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\n  wrote {out.as_posix()} -- Phase 11 renders this rather than "
              f"importing this package (check 8b forbids the import)")

    if failures:
        print(f"\n{len(failures)} problem(s):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        # A fabricated citation is fatal by default. The whole argument for putting a model
        # anywhere near a reconciliation is that its claims are checked, so shipping text
        # that failed its check would retire the argument.
        return 1 if strict else 0
    return 0


def ask_row(
    row: dict[str, Any],
    question: str,
    *,
    model: str,
    client: Any | None = None,
) -> int:
    """The other half of step 7: one question, one RESOLVED row, checked by arithmetic.

    Printed rather than folded into ``run()``'s artifact -- this is a one-shot Q&A over a
    single row an operator points at, not a batch pass over the queue, and the two features
    read disjoint row sets for the reason ``qa.py``'s docstring measured: an exception row
    has no decomposition to check a claim against, and a resolved row has nothing else.
    """
    print(f"explain: asking about {row['credit_id']} via "
          f"{'a recorded double' if client is not None else client_mod.base_url()}, "
          f"model {model}")
    try:
        answer = qa_mod.ask(row, question, model=model, client=client)
    except qa_mod.QAError as e:
        raise client_mod.ExplainError(str(e)) from e

    print(f"\n  Q: {question}")
    print(f"  A: {answer.answer}")
    if answer.arithmetic_checked:
        terms = ", ".join(
            f"{t['label']}={t['paise']}p" for t in (answer.arithmetic or {}).get("terms", ())
        )
        print(f"     arithmetic: {terms} -> total {(answer.arithmetic or {}).get('total_paise')}p")
    print(f"  {answer.summary()}")
    if not answer.ok:
        print("\n  problem(s):", file=sys.stderr)
        for f in answer.findings:
            print(f"    {f}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    p = argparse.ArgumentParser(
        description="Explain the exception queue in plain language, and check every claim.",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--fixture", action="store_true",
        help=f"read the recorded queue at {FIXTURE_PATH.as_posix()} instead of a live run",
    )
    src.add_argument("--matches", type=Path, help="matches.json from the matcher")
    p.add_argument("--data", type=Path, default=Path("data"), help="the input files (default: data/)")
    p.add_argument(
        "--dry-run", action="store_true",
        help="build every request and print what would be sent, without sending it",
    )
    p.add_argument("--model", default=client_mod.DEFAULT_MODEL, help="override the pinned model")
    p.add_argument("--out", type=Path, help="write the explanation artifact here")
    p.add_argument(
        "--permissive", action="store_true",
        help="exit 0 even if a citation could not be verified (default: exit 1)",
    )
    p.add_argument(
        "--ask", metavar="CREDIT_ID:QUESTION",
        help="ask a question about one RESOLVED row instead of explaining the queue, e.g. "
             "--ask 'C0001:why is the credit less than the gross?' (with --fixture, reads "
             "the row from the frozen resolved_sample)",
    )
    args = p.parse_args(argv)

    try:
        if args.ask is not None:
            credit_id, _, question = args.ask.partition(":")
            if not question:
                raise UsageError("--ask needs CREDIT_ID:QUESTION, with the colon present")
            # Same default as the explain path below: --fixture, or neither source given.
            if args.fixture or args.matches is None:
                rows = resolved_rows_from_fixture(FIXTURE_PATH)
            else:
                rows = list(qa_mod.resolved_rows(args.matches))
            row = next((r for r in rows if r["credit_id"] == credit_id), None)
            if row is None:
                raise UsageError(
                    f"{credit_id!r} is not a RESOLVED row here -- {len(rows)} RESOLVED row(s) "
                    f"available. Only a resolved row has a decomposition to answer questions "
                    f"against; an exception row has none, by definition."
                )
            return ask_row(row, question, model=args.model)

        if args.fixture or args.matches is None:
            groups = groups_from_fixture(FIXTURE_PATH)
        else:
            groups = groups_from_live(args.matches, args.data)
        if not groups:
            print("explain: the queue is empty -- nothing needs explaining")
            return 0
        if args.dry_run:
            return _dry_run(groups, args.model)
        return run(groups, model=args.model, out=args.out, strict=not args.permissive)
    except (UsageError, client_mod.ExplainError) as e:
        print(f"EXPLAIN FAILED\n  {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
