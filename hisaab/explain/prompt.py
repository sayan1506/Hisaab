"""Building the request, and keeping the cacheable half genuinely stable.

Three decisions here, each measured rather than assumed.

**1. Explanations are per GROUP, not per row.** The queue at seed 1, n=1000 has 295
exception rows whose notes total 168KB -- about 45,000 tokens of payload before any
instructions. One request per row would be 295 requests to say seven things, since rows
inside a group differ only in their ids and amounts: the group *is* the unit of
explanation, which is why Phase 9 grouped by cause in the first place.

**2. A group sends a bounded SAMPLE of its rows, and says so in the prompt.** The largest
group measured is ``FX_RATE_GAP`` at 141 rows -- roughly 20,000 tokens of notes for one
request. Sending all of them buys nothing: the 141st note restates the 3rd. So a group
sends at most ``ROWS_PER_GROUP`` rows, chosen by value descending (the queue's own ranking
-- the biggest money is what a person acts on first), and the prompt states the true row
count alongside the sample. **The model is told it is seeing a sample**, because a model
shown 12 of 141 rows and told nothing will write "these 12 rows", and that sentence would
be wrong in the report.

**3. The stable prefix is stable by construction, not by intention.** Prompt caching
matches on an exact byte prefix, so anything volatile before the last breakpoint silently
costs the whole saving. The system block here is built from ``triage.hint.HINTS`` and
frozen module text -- no timestamps, no row data, no dict iteration whose order could
shift. ``_self_check`` asserts two builds are byte-identical, and that the per-group half
does not appear in the cached half.
"""

from __future__ import annotations

from typing import Any

from ..common.reasons import Reason
from ..triage.hint import HINTS
from . import cluster as cluster_mod

#: Rows sent per group. 12 is a decision, not a limit imposed by anything:
#:
#:   * every group in the frozen fixture has <= 141 rows and >= 4, so 12 sends the whole
#:     group for the small ones and a sample for the two large ones;
#:   * 12 * ~570 chars of note is ~1,700 tokens of row payload, which keeps a group
#:     request an order of magnitude under the smallest cacheable prefix rather than
#:     competing with it;
#:   * and it is enough rows that a reader can see the pattern the summary claims,
#:     which is the point of sending more than one.
ROWS_PER_GROUP = 12

#: The instruction block. Frozen text: this is the first half of the cache prefix, so a
#: single interpolated value in here would invalidate every request's cache.
#:
#: The prohibitions are the load-bearing part, and each one names a failure this project
#: has already measured somewhere else:
#:
#:   * "do not invent figures" -- the citation check exists because a plausible number is
#:     the failure mode of a fluent model, and a wrong match is invisible to a queue-minutes
#:     comparison (ASSUMPTIONS.md #38).
#:   * "never say the matcher is broken" -- an abstention here is the *correct* answer on
#:     evidence, not a defect. 6.59% of credits abstain as AMBIGUOUS_MULTI_SUBSET and truth
#:     calls all of them resolvable; the inputs genuinely cannot separate them (#33).
#:   * "do not guess the resolution" -- the reserve is deliberately unmodelled precisely
#:     because a rule that fitted it would close every gap by construction (#22b).
INSTRUCTIONS = """\
You are helping a finance operator work an exception queue from a payment-gateway
reconciliation. Each group below holds bank credits that are still somebody's job, all
sharing one cause.

Most groups hold rows the reconciler could NOT resolve. One kind of group instead holds
rows it deliberately DECLINED, having decided they are not gateway money at all. Each group
states which kind it is, and the difference is not cosmetic: an unresolved row is missing
evidence somebody can go and get, whereas a declined row is a decision already made.

Your job is to explain each group in plain language an accountant can act on, and to do it
without ever exceeding what the evidence says.

Rules, in order of importance:

1. Cite only figures and ids that appear in the rows you are given. Never compute a new
   total, never round, never convert paise to rupees in a cited figure. If you want to
   refer to an amount, copy it exactly. Every figure and id you use goes into
   `cited_amounts_paise` and `cited_row_ids`, and both are checked against the input.

2. An unresolved row is not a bug. The reconciler refused it because the input files do
   not contain enough evidence to prove a match -- that refusal is the correct answer, and
   it is what makes this tool trustworthy. Never say or imply the matching is broken,
   incomplete, or in need of fixing.

2b. A DECLINED group is different, and must not be described as missing evidence. Those
   rows were read and judged not to be gateway money -- a salary credit, a transfer, a
   refund from somewhere else. Nothing is missing and there is nobody to ask. The useful
   next step for a declined group is confirmation by someone who knows the account, not
   an enquiry to the gateway.

3. Do not guess what the resolution will turn out to be. Say what evidence is missing and
   who would have it. "The gateway has not published the settlement-day rate" is useful;
   "this is probably a rate difference of about 2%" is a fabrication with a number in it.

4. Amounts are integer paise (100 paise = 1 rupee). Never treat a paise figure as rupees.

5. Write for someone who has never heard the reason code. Explain the situation; do not
   restate the code's name back to them.
"""


def _hint_block() -> str:
    """The 13 templated hints, as stable reference text for the cached prefix.

    Included so the model's ``next_step`` is grounded in the action this project already
    committed to for each code, rather than invented in parallel. Step 6 compares the two:
    the hint is the declared answer, the model's is the generated one, and a divergence is
    worth reading in either direction.

    Iterated over ``Reason`` -- a declaration order, fixed in source -- rather than over
    ``HINTS.keys()``, so the block cannot reorder if that dict is ever rebuilt. Byte
    stability is what makes the prefix cacheable.
    """
    lines = ["Reference: the reconciler's own declared next step for each reason code.", ""]
    for reason in Reason:
        hint = HINTS.get(reason)
        if hint is None:  # pragma: no cover -- HINTS asserts exhaustiveness itself
            continue
        lines.append(f"- {reason.value}: {hint.action}")
        if hint.unblocks:
            lines.append(f"    unblocked by: {hint.unblocks}")
    return "\n".join(lines)


def system_blocks() -> list[dict[str, Any]]:
    """The cacheable prefix: instructions plus the hint table, one cache breakpoint.

    One breakpoint at the end of the last block, not one per block. The blocks never vary
    independently -- they are both frozen text -- so a second breakpoint would consume one
    of the four available slots to cache a boundary nothing ever splits.
    """
    return [
        {"type": "text", "text": INSTRUCTIONS},
        {
            "type": "text",
            "text": _hint_block(),
            # Everything up to here is identical on every request in a run, so it is read
            # from cache after the first. Verified by cache_read_input_tokens, not assumed
            # -- see client.py.
            "cache_control": {"type": "ephemeral"},
        },
    ]


def _row_line(row: dict[str, Any]) -> str:
    note = " ".join((row.get("note") or "").split())
    return (
        f"- {row['credit_id']}: bank credit {row['bank_amount_paise']}p\n"
        f"    reconciler's note: {note}"
    )


def sampled_rows(group: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """The rows one group actually sends. **The single source of truth for that set.**

    ``group_message`` and ``cited_universe`` both need it, and they must never disagree: the
    universe is what a citation is checked against, so a row in the message but not the
    universe makes a *valid* citation look fabricated, and a row in the universe but not the
    message lets a fabrication pass by matching something the model never saw. Both functions
    computed the slice independently until clustering arrived, which made the divergence a
    live hazard rather than a latent one -- so there is now one function and both call it.

    Delegates to ``cluster.sample``: value-ranked selection can omit an entire sub-cause (the
    measured case is ``FX_RATE_GAP``'s 15-row minority, none of whose rows are among the
    group's 12 largest), and a model shown only the majority will describe the group as though
    the minority were not in it.
    """
    return cluster_mod.sample(group, ROWS_PER_GROUP)


def group_message(group: dict[str, Any]) -> str:
    """The volatile half: one group's rows, after the cache breakpoint.

    States the true row count next to the sample size whenever they differ, so the model
    cannot describe a sample as the whole group -- and, when the group holds more than one
    sub-cause, says how many rows each holds and how many are shown.
    """
    shown = list(sampled_rows(group))
    total = group["rows"]

    # The dismissal group carries no reason code -- ``triage/group.py`` files every declined
    # row under one heading regardless of the code it held, because the contract lets an
    # IGNORED row carry none at all. Printing "Reason code: None" would be false in the one
    # direction that matters: it reads as a missing value, when in fact this group's cause is
    # a decision rather than a code. So the header names the kind instead.
    declined = group.get("kind") == "dismissal" or group.get("reason") is None
    header = [
        f"Group kind: {'DECLINED -- judged not to be gateway money' if declined else 'UNRESOLVED -- the reconciler could not prove a match'}",
    ]
    if not declined:
        header.append(f"Reason code: {group['reason']}")
    header += [
        f"Cause as the queue labels it: {group['cause']}",
        f"Rows in this group: {total}",
        f"Total money at risk: {group['value_paise']}p",
    ]
    if len(shown) < total:
        # **Not "the N largest", which stopped being true when sampling became cluster-aware.**
        # `cluster.sample` takes each sub-cause's biggest row first and only then fills by
        # value, precisely so a small sub-cause cannot be dropped -- so the sample is ordered
        # by value but is not the top N. Saying "largest" would be a false statement in the
        # prompt about the prompt's own contents, and the model would have no way to know.
        header.append(
            f"Below are {len(shown)} of those {total} rows, ordered by amount, chosen to cover "
            f"every sub-cause in the group rather than only the biggest rows. The other "
            f"{total - len(shown)} share this reason code and are not shown -- describe the "
            f"group of {total}, and do not imply you were shown all of them."
        )
    else:
        header.append(f"All {total} rows in this group are shown below.")

    census = cluster_mod.describe(group, tuple(shown))
    if census:
        header.append(census)

    return "\n".join(header) + "\n\n" + "\n".join(_row_line(r) for r in shown)


def build_request(group: dict[str, Any]) -> dict[str, Any]:
    """Everything about one group's request except the model, tokens and transport.

    Split from ``client.py`` so the prompt is testable with nothing installed: this returns
    plain data, and gate 17 asserts against it without a network or an SDK.
    """
    return {
        "system": system_blocks(),
        "messages": [{"role": "user", "content": group_message(group)}],
    }


def cited_universe(group: dict[str, Any]) -> tuple[set[str], set[int]]:
    """The ids and paise figures the model was actually shown, for ``verify.py``.

    **Scoped to the sample, not the group.** A model shown 12 rows cannot legitimately
    cite the 13th row's id, so the universe a citation is checked against has to be what
    was sent -- checking against all 141 rows would pass a fabricated id that happened to
    match an unsent row, which is precisely the coincidence the check exists to catch.
    """
    # **The same rows the message sent, via the same function.** Independently re-sorting here
    # was correct only while both slices were "the 12 largest by value"; the moment sampling
    # became cluster-aware the two would have diverged, and the divergence is worse in both
    # directions than a missing feature: a row in the message but not the universe makes a
    # *legitimate* citation read as a fabrication, and a row in the universe but not the
    # message lets a real fabrication pass by matching something the model never saw.
    rows = sampled_rows(group)

    ids: set[str] = set()
    amounts: set[int] = set()
    for row in rows:
        ids.add(row["credit_id"])
        amounts.add(int(row["bank_amount_paise"]))
        note = row.get("note") or ""
        ids.update(_ids_in(note))
        amounts.update(_paise_in(note))
    amounts.add(int(group["value_paise"]))
    return ids, amounts


#: Id shapes the generator emits. Duplicated here rather than imported from the generator
#: -- check 6 forbids that import, and this is a schema, which this project duplicates on
#: purpose so drift fails loudly (the same contract as ``matcher/load.py``'s CSV headers).
_ID_PREFIXES = ("setl_", "pay_", "rfnd_", "C")


def _ids_in(text: str) -> set[str]:
    import re

    found = set(re.findall(r"\b(?:setl_|pay_|rfnd_)\d+\b", text))
    found.update(re.findall(r"\bC\d{4,}\b", text))
    return found


def _paise_in(text: str) -> set[int]:
    """Integers written as a paise figure in a note, e.g. ``39771p`` or ``+/-0p``.

    Deliberately narrow: only digits carrying the ``p`` suffix the matcher's notes use.
    A looser pattern would sweep up ``2bd``, ``6.93%`` and ``1 payment(s)`` and hand the
    verifier a universe so wide that no fabricated figure could fall outside it -- a check
    that cannot fail. Notes write day counts as ``2bd`` and ``+1bd``, never as digits
    alone, which is what makes the narrow pattern sufficient here.
    """
    import re

    return {int(m) for m in re.findall(r"(\d+)\s*p\b", text)}


def _self_check() -> None:
    """The cached half must be byte-stable, and must not contain the volatile half."""
    a, b = system_blocks(), system_blocks()
    assert a == b, "system_blocks() is not deterministic; the cache prefix would never hit"
    text_a = "".join(block["text"] for block in a)
    text_b = "".join(block["text"] for block in b)
    assert text_a == text_b, "the cached prefix differs between two builds of it"

    breakpoints = [blk for blk in a if "cache_control" in blk]
    assert len(breakpoints) == 1, f"expected exactly 1 cache breakpoint, got {len(breakpoints)}"
    assert breakpoints[0] is a[-1], "the breakpoint must be on the last stable block"

    # Every code must appear in the reference block, or the model gets no declared action
    # for some group it will be asked about.
    block = _hint_block()
    missing = [r.value for r in Reason if r in HINTS and r.value not in block]
    assert not missing, f"reason codes absent from the hint block: {missing}"
    assert len(HINTS) == len(list(Reason)), (
        f"HINTS covers {len(HINTS)} of {len(list(Reason))} codes; the reference block "
        f"would be silently short"
    )

    # A group whose rows exceed the cap must be described by its true size, and must say
    # it is a sample. This is the assertion that stops "these 12 rows" reaching a report.
    big = {
        "reason": "FX_RATE_GAP",
        "cause": "foreign currency",
        "rows": 141,
        "value_paise": 999,
        "credits": [
            {"credit_id": f"C{i:04d}", "bank_amount_paise": 1000 + i, "note": f"setl_{i:04d} gap {50 + i}p"}
            for i in range(141)
        ],
    }
    msg = group_message(big)
    assert "Rows in this group: 141" in msg
    # The wording is checked, not just the presence of a sample line, because the claim the
    # prompt makes about itself has to stay true: it said "the 12 largest of those 141" until
    # sampling became cluster-aware, and that sentence is now false -- `cluster.sample` takes
    # each sub-cause's biggest row before filling by value, so the sample is value-ORDERED but
    # is not the top 12. A prompt that misdescribes its own contents is a defect the model
    # cannot detect.
    assert "12 of those 141 rows" in msg, "a truncated group must say it is a sample"
    assert "largest" not in msg, (
        "the message still claims to hold the largest rows, which cluster-aware sampling makes "
        "untrue"
    )
    assert msg.count("bank credit") == ROWS_PER_GROUP, (
        f"expected {ROWS_PER_GROUP} rows in the message, found {msg.count('bank credit')}"
    )
    # These 141 rows all share one note template, so there is one sub-cause and the sample is
    # the 12 largest after all -- which is what makes this a control: cluster-aware sampling
    # must not perturb a group that has nothing to cluster.
    assert "C0140" in msg and "C0000" not in msg, (
        "a single-sub-cause group must still send its largest rows -- clustering changed the "
        "selection for a group that has only one cluster"
    )

    small = {**big, "rows": 4, "credits": big["credits"][:4]}
    msg_small = group_message(small)
    assert "All 4 rows" in msg_small, "an untruncated group must not claim to be a sample"
    assert "of those" not in msg_small, "an untruncated group must not describe a sample"

    # --- the invariant the sampled_rows() refactor exists to hold ---------------------
    # A group with a large majority and a small minority whose rows are all smaller. The
    # message and the universe must contain THE SAME ROWS: a row shown but absent from the
    # universe turns a legitimate citation into a reported fabrication, and a row in the
    # universe but not shown lets a fabrication pass by matching something never sent. The two
    # sliced their rows independently until this refactor, so the invariant is asserted rather
    # than trusted to two functions agreeing by construction.
    split = {
        "reason": "FX_RATE_GAP", "cause": "fx", "rows": 23, "value_paise": 7,
        "credits": [
            *[{"credit_id": f"C{i:04d}", "bank_amount_paise": 90000 + i,
               "note": f"matched setl_{i:04d} on date, one subset of {i} payment(s) sums"}
              for i in range(20)],
            *[{"credit_id": f"C9{i:03d}", "bank_amount_paise": 100 + i,
               "note": f"matched setl_9{i:03d} on date, no subset of {i} payment(s) sums"}
              for i in range(3)],
        ],
    }
    sampled = sampled_rows(split)
    assert len(sampled) == ROWS_PER_GROUP
    msg_split = group_message(split)
    u_ids, _ = cited_universe(split)
    for row in sampled:
        assert row["credit_id"] in msg_split, f"{row['credit_id']} sampled but not in the message"
        assert row["credit_id"] in u_ids, (
            f"{row['credit_id']} was SENT to the model but is absent from the universe its "
            f"citations are checked against, so citing it would be reported as a fabrication"
        )
    unsent = {r["credit_id"] for r in split["credits"]} - {r["credit_id"] for r in sampled}
    assert not (unsent & u_ids), (
        f"the universe contains row(s) never sent ({sorted(unsent & u_ids)[:3]}), so a "
        f"fabricated id could pass by coinciding with one of them"
    )
    # And the minority must be in there at all -- the reason for the whole change.
    assert any(r["credit_id"].startswith("C9") for r in sampled), (
        "the 3-row sub-cause was dropped from the sample, so the model would describe a "
        "23-row group having seen only one of its two situations"
    )
    assert "2 distinct sub-causes" in msg_split, "a multi-sub-cause group must say so"

    # The dismissal group. It is in the frozen fixture (15 rows, ``reason`` null), and the
    # first version of this function printed "Reason code: None" for it while the
    # instructions asserted every group held rows that could not be resolved. Both were
    # wrong in the same direction: a declined row is a decision, not missing evidence, and
    # prose built on the other reading would tell an operator to chase the gateway about a
    # salary credit.
    dismissal = {
        "reason": None, "kind": "dismissal", "cause": "DISMISSED (not gateway money)",
        "rows": 2, "value_paise": 5000,
        "credits": [
            {"credit_id": "C0001", "bank_amount_paise": 3000, "note": "payroll, not gateway"},
            {"credit_id": "C0002", "bank_amount_paise": 2000, "note": "own transfer"},
        ],
    }
    msg_d = group_message(dismissal)
    assert "DECLINED" in msg_d, "the dismissal group must be marked as declined"
    assert "Reason code: None" not in msg_d and "None" not in msg_d, (
        f"a null reason code leaked into the prompt as text: {msg_d.splitlines()[:2]}"
    )
    # And the control: an exception group must NOT be labelled declined, or the marker
    # above is unconditional and proves nothing.
    assert "DECLINED" not in msg_small and "UNRESOLVED" in msg_small, (
        "every group is being labelled the same way, so the label carries no information"
    )
    assert "Reason code: FX_RATE_GAP" in msg_small, "an exception group must state its code"
    # A dismissal detected by kind alone, with a reason present, must still read as declined:
    # the two signals must not disagree silently.
    assert "DECLINED" in group_message({**dismissal, "reason": "NO_CANDIDATE"})

    # The volatile half must not have leaked into the cached half. Asserted with the same
    # extractors verify.py uses -- no row ids, no paise figures -- rather than by searching
    # for a phrase.
    #
    # The first version of this block searched for the substring "bank credit" and failed
    # on its own prompt: INSTRUCTIONS legitimately says "bank credits the reconciler could
    # NOT resolve". A phrase that appears in both frozen prose and row data cannot
    # distinguish them, and the check was reporting on its own wording rather than on the
    # property. What actually matters is that nothing row-specific is in there.
    for label, block_text in (("INSTRUCTIONS", INSTRUCTIONS), ("hint block", _hint_block())):
        leaked_ids = _ids_in(block_text)
        leaked_amounts = _paise_in(block_text)
        assert not leaked_ids, (
            f"the cached prefix ({label}) names row ids {sorted(leaked_ids)[:5]}. Anything "
            f"row-specific before the breakpoint invalidates the cache on every group."
        )
        assert not leaked_amounts, (
            f"the cached prefix ({label}) carries paise figures "
            f"{sorted(leaked_amounts)[:5]}, so it varies per group and never caches."
        )

    # The control: the assertion above must be able to fail. Without this, a prefix with
    # row data in it and a broken extractor look identical from here.
    assert _ids_in(INSTRUCTIONS + " C0140 setl_0007"), (
        "the leak check cannot detect row ids even when they are present, so its pass "
        "above proves nothing about the prefix"
    )
    assert _paise_in(INSTRUCTIONS + " 39771p"), (
        "the leak check cannot detect paise figures even when present"
    )

    ids, amounts = cited_universe(big)
    assert "C0140" in ids and "setl_0140" in ids, f"universe missing sampled ids: {sorted(ids)[:5]}"
    assert "C0000" not in ids, (
        "the universe includes a row that was never sent, so a fabricated id could pass "
        "by coinciding with an unsent row"
    )
    assert 1140 in amounts, "the sampled row's bank amount is not in the universe"
    assert 190 in amounts, "a paise figure from the note is not in the universe"

    # The narrow paise pattern must not sweep up day counts or percentages -- a universe
    # that contains every integer cannot fail a fabrication.
    assert _paise_in("within 2bd sums to 55313p at 6.93% over 1 payment(s)") == {55313}, (
        f"the paise pattern is too wide: {_paise_in('within 2bd sums to 55313p at 6.93% over 1 payment(s)')}"
    )

    req = build_request(small)
    assert set(req) == {"system", "messages"}
    assert req["messages"][0]["role"] == "user"
    print(
        f"prompt: ok -- prefix byte-stable ({len(text_a)} chars, 1 breakpoint), "
        f"{len(HINTS)} hints, sample capped at {ROWS_PER_GROUP} and declared as one"
    )


if __name__ == "__main__":
    _self_check()
