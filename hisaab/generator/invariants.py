"""Invariants — asserted before anything touches disk, and again on read-back.

Each check here is a bug you would otherwise meet in Phase 5 with no idea where
it came from. ``check_story`` runs on the in-memory story; ``tools/verify_output.py``
re-reads the written CSVs and re-runs the row-level checks against the files,
because the write step itself can corrupt (column order, quoting, encoding) and
because it doubles as the first smoke test of the Phase 2/3 loader.

I7 (no gateway identifier in a bank narration) and I10 (truth isolation) are the
two standing between us and an invalidated submission. They are real checks, not
habits.

A note on I8, because the naive version of it is wrong
-----------------------------------------------------
The temptation is to assert "the payment -> credit index permutation has few fixed
points, because a random permutation of any size has ~1 expected fixed point."
That reasoning holds for payment -> settlement, which is a genuine shuffle. It does
**not** hold for payment -> credit: in clean mode ``value_date`` equals the capture
date by construction, so payments and credits are blocked into identical date
groups and the permutation is block-diagonal. A block of size k contributes ~1
fixed point, so the total is ~(number of blocks), not ~1. Measured: 19-24 at
n=60 across seeds, which the naive ceiling of 5 would fail on every honest run.

Within a date block, payment order is by capture time and credit order is by
amount. Those are independent, so positional coincidence at the ~1/k rate is the
birthday-problem baseline, not a leak -- and it cannot be driven to zero without
*anti*-correlating the files, which would itself be a signal. So I8 splits:

  I8a  the ID numbering carries no information        (a true shuffle, ceiling 5)
  I8b  within-block position carries no information   (a rate well under identity)

I8b's ceiling is set to catch the regressions that matter -- someone sorting the
bank rows by capture time instead of amount, or dropping the re-sort altogether --
both of which drive the rate to 100%.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date

from ..common import ids
from .config import GenConfig
from .model import (
    BANK_HEADER,
    PAYMENTS_HEADER,
    REFUNDS_HEADER,
    SETTLEMENT_ITEMS_HEADER,
    SETTLEMENTS_HEADER,
    Decomposition,
    Story,
)
from .money import RUPEE

#: I8a: expected fixed points of a true random permutation is 1, independent of n.
#: Five is a generous ceiling that still fails loudly on identity ordering.
MAX_NUMBERING_FIXED_POINTS = 5

#: I8b: identity ordering scores 100%. Observed honest rates are 29-41% at n=60,
#: 8-13% at n=200, 3-5% at n=500 -- the rate falls as blocks grow. A 60% ceiling
#: leaves large headroom above honest noise while catching any real regression.
MAX_WITHIN_BLOCK_ALIGNMENT_RATE = 0.60

#: Below this many block-resident records the rate is statistically meaningless
#: (at n=12 most dates hold a single record, and 2/2 = 100% is pure chance).
MIN_BLOCK_POPULATION_FOR_RATE = 20


class InvariantError(AssertionError):
    """A generated story or written file violated a structural guarantee."""


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise InvariantError(f"{code}: {message}")


# ---------------------------------------------------------------------------
# Row-level checks. Callable on the in-memory story AND on rows read back from
# disk, so the two passes cannot drift apart.
# ---------------------------------------------------------------------------

def check_unique_ids(code: str, label: str, id_values: list[str]) -> None:
    """I1 — every ID is unique within its file."""
    seen: set[str] = set()
    dupes: set[str] = set()
    for v in id_values:
        if v in seen:
            dupes.add(v)
        else:
            seen.add(v)
    _require(not dupes, code, f"duplicate {label} ids: {sorted(dupes)[:5]}")


def check_headers(actual: dict[str, tuple[str, ...]]) -> None:
    """I9 — frozen headers, in order.

    Guards trap 8: every field added to the bank statement is difficulty deleted
    from our own submission, so the header is pinned rather than merely reviewed.
    """
    expected = {
        "payments.csv": PAYMENTS_HEADER,
        "settlements.csv": SETTLEMENTS_HEADER,
        "settlement_items.csv": SETTLEMENT_ITEMS_HEADER,
        "bank_statement.csv": BANK_HEADER,
        "refunds.csv": REFUNDS_HEADER,
    }
    for name, want in expected.items():
        _require(name in actual, "I9", f"{name} is missing")
        _require(
            tuple(actual[name]) == want,
            "I9",
            f"{name} header drifted\n  expected: {want}\n  actual:   {tuple(actual[name])}",
        )


def check_no_leak(bank_rows: list[tuple[str, int, str]]) -> None:
    """I7 — no gateway identifier and no self-amount in any bank narration.

    ``bank_rows`` is (row_id, amount_paise, narration).

    The literal reading of "no narration contains pay_, setl_ or C" both fires on
    ``IMPS-JOHNDOE-DIRECTTRANSFER`` (a 'C') and misses what matters, so this
    matches ID *patterns* on word boundaries (trap 6). The amount check compares
    whole digit runs, not substrings, so a UTR tail of 1004 is not read as
    containing the rupee figure 100.
    """
    for row_id, amount_paise, narration in bank_rows:
        leaked = ids.leaked_identifiers(narration)
        _require(
            not leaked,
            "I7",
            f"{row_id} narration leaks {leaked}: {narration!r} -- "
            f"a gateway identifier in the bank statement generates the answer "
            f"into the input and invalidates the submission",
        )
        runs = set(ids.digit_runs(narration))
        whole, frac = divmod(amount_paise, RUPEE)
        echoes = runs & {str(amount_paise), str(whole)} if frac == 0 else runs & {str(amount_paise)}
        _require(
            not echoes,
            "I7",
            f"{row_id} narration echoes its own amount {echoes}: {narration!r}",
        )


def check_totals(
    gross_total: int, net_total: int, credit_total: int, fee_cells: list[int]
) -> None:
    """I4 — the three totals agree, and every fee/gst/tds cell is zero.

    Clean mode has *zero* fees, not "fees applied consistently" (trap 7). That is
    what makes the first amount mismatch in Phase 3 a one-suspect debug instead of
    a two-suspect one.
    """
    _require(
        gross_total == net_total == credit_total,
        "I4",
        f"totals disagree: gross={gross_total} net={net_total} credited={credit_total}",
    )
    nonzero = [c for c in fee_cells if c != 0]
    _require(
        not nonzero,
        "I4",
        f"{len(nonzero)} non-zero fee/gst/tds cells while --fees is off: {nonzero[:5]}",
    )


def check_int_money(values: dict[str, object]) -> None:
    """I5 — every monetary value is an ``int`` (trap 9).

    Re-run after the CSV round-trip, where a float would arrive as a string like
    ``'1000000.0'`` and fail ``int()`` loudly rather than silently.
    """
    for label, v in values.items():
        _require(
            isinstance(v, int) and not isinstance(v, bool),
            "I5",
            f"{label} is {type(v).__name__} ({v!r}), not int paise",
        )


def check_within_block_alignment(
    payment_ids_by_date: dict[date, list[str]],
    credit_payment_ids_by_date: dict[date, list[str]],
) -> tuple[int, int]:
    """I8b — within a date block, row position carries no information.

    Returns (aligned, population) for reporting. See the module docstring for why
    this is a rate check against identity rather than a fixed-point ceiling.
    """
    aligned = population = 0
    for d, pl in payment_ids_by_date.items():
        cl = credit_payment_ids_by_date.get(d, [])
        if len(pl) < 2:
            continue  # a single-record date is trivially "aligned" and means nothing
        population += len(pl)
        aligned += sum(1 for i, pid in enumerate(pl) if i < len(cl) and cl[i] == pid)
    if population >= MIN_BLOCK_POPULATION_FOR_RATE:
        rate = aligned / population
        _require(
            rate <= MAX_WITHIN_BLOCK_ALIGNMENT_RATE,
            "I8b",
            f"within-date-block positional alignment is {rate:.0%} "
            f"({aligned}/{population}), above the {MAX_WITHIN_BLOCK_ALIGNMENT_RATE:.0%} "
            f"ceiling -- bank rows appear to be ordered by capture time rather than "
            f"amount, which lets a zip matcher score without doing any work",
        )
    return aligned, population


# ---------------------------------------------------------------------------
# Story-level checks
# ---------------------------------------------------------------------------

def check_story(story: Story, cfg: GenConfig) -> dict[str, object]:
    """Run every invariant on the in-memory story. Raises ``InvariantError``.

    Returns a small report dict for the CLI to echo, so a passing run still shows
    the numbers behind the pass rather than only the word "ok".
    """
    payments, settlements, credits = story.payments, story.settlements, story.credits

    # I1 — unique ids within each file
    check_unique_ids("I1", "payment", [p.payment_id for p in payments])
    check_unique_ids("I1", "order", [p.order_id for p in payments])
    check_unique_ids("I1", "settlement", [s.settlement_id for s in settlements])
    check_unique_ids("I1", "credit", [c.credit_id for c in credits])
    check_unique_ids("I1", "refund", [r.refund_id for r in story.refunds])

    # I3 — clean mode has no noise and no orphans
    if cfg.clean_mode:
        _require(
            len(payments) == len(settlements) == len(credits) == cfg.n,
            "I3",
            f"clean mode must be 1:1:1 at n={cfg.n}, got "
            f"{len(payments)}/{len(settlements)}/{len(credits)}",
        )
        _require(not story.refunds, "I3", "clean mode emits no refunds")
        _require(
            not (story.unsettled_payment_ids or story.settlements_without_credit
                 or story.non_gateway_credit_ids),
            "I3",
            "clean mode has no orphans and no noise rows",
        )

    # I2 — every payment in exactly one settlement; every settlement in one credit
    settled_count: dict[str, int] = defaultdict(int)
    for s in settlements:
        for pid in s.payment_ids:
            settled_count[pid] += 1
    payment_ids = {p.payment_id for p in payments}
    unknown = sorted(set(settled_count) - payment_ids)
    _require(not unknown, "I2", f"settlements reference unknown payments: {unknown[:5]}")
    if cfg.clean_mode:
        missing = sorted(payment_ids - set(settled_count))
        _require(not missing, "I2", f"payments never settled: {missing[:5]}")
    multi = sorted(pid for pid, k in settled_count.items() if k > 1)
    _require(not multi, "I2", f"payments in more than one settlement: {multi[:5]}")

    credited_count: dict[str, int] = defaultdict(int)
    for c in credits:
        for sid in c.settlement_ids:
            credited_count[sid] += 1
    settlement_ids = {s.settlement_id for s in settlements}
    unknown_s = sorted(set(credited_count) - settlement_ids)
    _require(not unknown_s, "I2", f"credits reference unknown settlements: {unknown_s[:5]}")
    if cfg.clean_mode:
        uncredited = sorted(settlement_ids - set(credited_count))
        _require(not uncredited, "I2", f"settlements never credited: {uncredited[:5]}")
    multi_s = sorted(sid for sid, k in credited_count.items() if k > 1)
    _require(not multi_s, "I2", f"settlements in more than one credit: {multi_s[:5]}")

    # I4 — totals agree, fee columns are zero
    check_totals(
        story.total_gross_paise(),
        story.total_net_paise(),
        story.total_credited_paise(),
        [x for s in settlements for x in (s.fee_paise, s.gst_paise, s.tds_paise)],
    )

    # I5 — every monetary value is an int
    money: dict[str, object] = {}
    for p in payments:
        money[f"{p.payment_id}.gross_paise"] = p.gross_paise
    for s in settlements:
        money[f"{s.settlement_id}.net_paise"] = s.net_paise
    for c in credits:
        money[f"{c.credit_id}.amount_paise"] = c.amount_paise
        money[f"{c.credit_id}.expected"] = c.decomposition.expected_credit_paise
    check_int_money(money)

    # Clean-mode arithmetic: each credit's decomposition must close to its amount.
    for c in credits:
        _require(
            c.decomposition.expected_credit_paise == c.amount_paise,
            "I4",
            f"{c.credit_id} decomposition expects "
            f"{c.decomposition.expected_credit_paise} but the credit is {c.amount_paise}",
        )

    # I7 — the leak check
    check_no_leak([(c.credit_id, c.amount_paise, c.narration) for c in credits])

    # I6 — truth references resolve in both directions
    for c in credits:
        for sid in c.settlement_ids:
            _require(sid in settlement_ids, "I6", f"{c.credit_id} cites unknown {sid}")
        for pid in c.payment_ids:
            _require(pid in payment_ids, "I6", f"{c.credit_id} cites unknown {pid}")
    if cfg.clean_mode:
        cited = {pid for c in credits for pid in c.payment_ids}
        _require(
            cited == payment_ids,
            "I6",
            f"{len(payment_ids - cited)} payments are in no credit's truth entry",
        )

    # Dates line up, because --settlement-delay is off.
    by_pay = {p.payment_id: p for p in payments}
    by_setl = {s.settlement_id: s for s in settlements}
    for s in settlements:
        for pid in s.payment_ids:
            _require(
                s.settled_on == by_pay[pid].business_date,
                "I4",
                f"{s.settlement_id} settled {s.settled_on} but {pid} captured "
                f"{by_pay[pid].business_date} while --settlement-delay is off",
            )
    for c in credits:
        for sid in c.settlement_ids:
            _require(
                c.value_date == by_setl[sid].settled_on,
                "I4",
                f"{c.credit_id} dated {c.value_date} but {sid} settled "
                f"{by_setl[sid].settled_on} while --settlement-delay is off",
            )

    # Clean mode must be resolvable at 100% -- row 1 of the mess dial. A duplicate
    # (date, amount) would be genuinely indistinguishable from the two legitimate
    # signals, so the honest verdict would be an abstention. That case is real and
    # --dup-amounts plants it deliberately in Phase 8; it must not appear here by
    # accident.
    if cfg.clean_mode:
        key_counts = Counter((c.value_date, c.amount_paise) for c in credits)
        dupes = {k for k, count in key_counts.items() if count > 1}
        _require(
            not dupes,
            "I3",
            f"{len(dupes)} duplicate (date, amount) credits make clean mode "
            f"unresolvable: {sorted(dupes)[:3]}",
        )

    # I8a — the ID numbering carries no information
    pay_index = {p.payment_id: i for i, p in enumerate(payments)}
    setl_index = {s.settlement_id: i for i, s in enumerate(settlements)}
    numbering_fixed = sum(
        1 for s in settlements if pay_index[s.payment_ids[0]] == setl_index[s.settlement_id]
    )
    _require(
        numbering_fixed <= MAX_NUMBERING_FIXED_POINTS,
        "I8a",
        f"{numbering_fixed} settlements share an index with their payment "
        f"(ceiling {MAX_NUMBERING_FIXED_POINTS}) -- settlement IDs appear to be "
        f"assigned in payment order, which makes the numbering itself the answer key",
    )

    # I8b — within-block position carries no information
    p_by_date: dict[date, list[str]] = defaultdict(list)
    for p in payments:
        p_by_date[p.business_date].append(p.payment_id)
    c_by_date: dict[date, list[str]] = defaultdict(list)
    for c in credits:
        c_by_date[c.value_date].append(c.payment_ids[0])
    aligned, population = check_within_block_alignment(p_by_date, c_by_date)

    return {
        "records": len(payments),
        "date_blocks": len(p_by_date),
        "numbering_fixed_points": numbering_fixed,
        "within_block_aligned": f"{aligned}/{population}",
        "gross_paise_total": story.total_gross_paise(),
    }


if __name__ == "__main__":
    import dataclasses

    from .story import build

    cfg = GenConfig(seed=42, n=60)
    story = build(cfg)
    report = check_story(story, cfg)
    assert report["records"] == 60
    print(f"invariants: {report}")

    # Every seed and size in the acceptance matrix must pass.
    for n in (1, 12, 60, 200):
        for seed in (1, 2, 3, 42, 43):
            c = GenConfig(seed=seed, n=n)
            check_story(build(c), c)
    print("invariants.py: 20 story configurations pass")

    # --- the checks must FAIL on broken stories, or they are decoration --------
    def must_fail(code: str, broken: Story, cfg_: GenConfig = cfg) -> None:
        try:
            check_story(broken, cfg_)
        except InvariantError as e:
            assert str(e).startswith(code), f"expected {code}, got: {e}"
        else:
            raise AssertionError(f"{code} did not fire")

    # I7: a leaked settlement id in a narration
    bad = dataclasses.replace(
        story.credits[0], narration=f"NEFT-{story.credits[0].settlement_ids[0]}-XXXX4471"
    )
    must_fail("I7", dataclasses.replace(story, credits=[bad, *story.credits[1:]]))

    # I7: a narration echoing its own amount
    c0 = story.credits[0]
    bad = dataclasses.replace(c0, narration=f"NEFT-RAZORPAYSOFT-{c0.amount_paise}")
    must_fail("I7", dataclasses.replace(story, credits=[bad, *story.credits[1:]]))

    # I4: a non-zero fee cell while --fees is off
    s0 = story.settlements[0]
    must_fail(
        "I4",
        dataclasses.replace(
            story, settlements=[dataclasses.replace(s0, fee_paise=1), *story.settlements[1:]]
        ),
    )

    # I3: a duplicate (date, amount) pair.
    # Cloning only the credit's amount would change the credited total and trip I4
    # first, so the whole chain behind it is rewritten -- payment gross, settlement
    # net, credit amount, decomposition. That is also exactly what --dup-amounts
    # will construct in Phase 8, so this negative case doubles as a rehearsal.
    a, b = story.credits[0], story.credits[1]
    a_pid, b_pid = a.payment_ids[0], b.payment_ids[0]
    a_sid, b_sid = a.settlement_ids[0], b.settlement_ids[0]
    a_pay = next(p for p in story.payments if p.payment_id == a_pid)
    must_fail(
        "I3",
        dataclasses.replace(
            story,
            payments=[
                dataclasses.replace(
                    p,
                    gross_paise=a.amount_paise,
                    captured_at=a_pay.captured_at.replace(minute=0, second=0),
                )
                if p.payment_id == b_pid
                else p
                for p in story.payments
            ],
            settlements=[
                dataclasses.replace(s, net_paise=a.amount_paise, settled_on=a.value_date)
                if s.settlement_id == b_sid
                else s
                for s in story.settlements
            ],
            credits=[
                dataclasses.replace(
                    c,
                    value_date=a.value_date,
                    amount_paise=a.amount_paise,
                    narration=f"IMPS-RZRPAY-{a_sid[-4:]}X",
                    decomposition=Decomposition(gross_paise=a.amount_paise),
                )
                if c.credit_id == b.credit_id
                else c
                for c in story.credits
            ],
        ),
    )

    # I8a: settlements ordered and numbered in payment order (what skipping the
    # shuffle in step 7 would produce). Renaming must propagate to the credits, or
    # the dangling references trip I2/I4 first and I8a is never reached -- which
    # would make this a test of the wrong thing.
    pay_ix = {p.payment_id: i for i, p in enumerate(story.payments)}
    in_payment_order = sorted(story.settlements, key=lambda s: pay_ix[s.payment_ids[0]])
    rename = {s.settlement_id: ids.settlement_id(i)
              for i, s in enumerate(in_payment_order, start=1)}
    must_fail(
        "I8a",
        dataclasses.replace(
            story,
            settlements=[dataclasses.replace(s, settlement_id=rename[s.settlement_id])
                         for s in in_payment_order],
            credits=[dataclasses.replace(c, settlement_ids=[rename[sid]
                                                            for sid in c.settlement_ids])
                     for c in story.credits],
        ),
    )

    # I8b: bank rows ordered by capture time instead of amount
    by_pay_ix = {p.payment_id: i for i, p in enumerate(story.payments)}
    time_ordered = [
        dataclasses.replace(c, credit_id=ids.credit_id(i))
        for i, c in enumerate(
            sorted(story.credits, key=lambda c: (c.value_date, by_pay_ix[c.payment_ids[0]])),
            start=1,
        )
    ]
    must_fail("I8b", dataclasses.replace(story, credits=time_ordered))

    print("invariants.py self-check ok  (6 negative cases fire correctly)")
