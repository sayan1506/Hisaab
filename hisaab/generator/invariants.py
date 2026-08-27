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
from ..common.bizdays import BusinessCalendar
from ..common.reasons import Reason
from .config import GenConfig, MessFlags
from .model import (
    BANK_HEADER,
    PAYMENTS_HEADER,
    REFUNDS_HEADER,
    SETTLEMENT_ITEMS_HEADER,
    SETTLEMENTS_HEADER,
    Decomposition,
    Story,
)
from ..common.money import RUPEE

#: Which mess flags legitimately invalidate each conditional invariant.
#:
#: A check runs unless one of *its own* flags is on, so a flag suspends exactly the
#: checks it actually breaks. This replaces a single ``cfg.clean_mode`` gate, which was
#: a footgun rather than a policy: ``clean_mode`` is "all thirteen flags off", so
#: ``--fees`` -- which changes no cardinality and no date -- switched off the cardinality,
#: membership and uniqueness checks along with everything else. The mislabelled run then
#: lost the very checks that would have noticed it was mislabelled.
#:
#: Keys are ``<invariant id>.<what it asserts>``; a typo raises ``KeyError`` at the call
#: site rather than silently never suspending. Every flag named here is validated against
#: ``MessFlags`` in the self-check, for the same reason.
#:
#: Note what is deliberately **absent**: ``fees`` and ``settlement_delay``. Phase 4 measured
#: whether either can force a natural ``(value_date, amount_paise)`` collision, and the answer
#: decided that both stay off this list. See ``ASSUMPTIONS.md`` #24a/#24b. Measured twice --
#: once in step 6, then again in step 7 after the rate table was corrected, because the first
#: measurement's *stated mechanism* named a rate that no longer exists. Re-running it changed
#: the explanation while leaving the decision intact, which is the useful kind of surprise:
#:
#:   * ``--settlement-delay`` cannot collide **by construction**, and this is the one claim
#:     here that is structural rather than empirical. The capture-to-value date map is
#:     injective, so the delay only *relabels* days: every within-day amount set carries over
#:     intact, and delay-alone reproduces clean mode's collision count exactly at every size
#:     tested. Note that injectivity, not equality, is the property that matters -- an earlier
#:     probe wrongly concluded the delay was unguaranteed from the fact that
#:     ``capture_date != value_date`` on every row, which tests the wrong thing.
#:   * ``--fees`` does both, and the balance shifted when the rates did. It *disperses* when
#:     equal-gross payments sit on different rates (they net apart) and *concentrates* when a
#:     priced row's net lands on some other row's amount. Under the old table, with UPI at 0
#:     bps against card at 200, dispersal dominated: 60-75% of equal-gross groups broke apart.
#:     With four methods now sharing 200 bps that falls to 20-30%, so fees is a weaker
#:     disperser -- and every collision that survives at n>=4000 is one **fees created**,
#:     since clean and delay stay at 0 throughout.
#:
#: The concentration channel has a specific shape worth recording, because it says which rail
#: to watch rather than just that a number exists. ``_unique_amount`` guarantees distinct
#: *gross* within a capture date; a zero-rated row settles **at** its gross, so its credit
#: inherits that protection, while a priced row's net is a derived value no invariant compares
#: to anything. So the free rail acts as a **magnet**: 47-58% of collisions at n>=8000 involve
#: a ``pos_upi`` row, against 11% if method were irrelevant, and the priced member's gross sits
#: 2.42%-2.60% above the collision value -- which is exactly the net-to-gross ratio at 200 and
#: 215 bps, not a fitted range. Phase 5's batching and Phase 6's TDS both add derived nets, so
#: this is the channel that will widen.
#:
#: **336 runs** (12 seeds x 4 flag settings x 7 sizes to the UTR ceiling) put the first
#: collision at n=4000, with 0 at every size at or below 2000. The largest size this project
#: runs is n=1000, so the margin is 4x and clean mode holds everywhere. That margin is why the
#: check stays **strict** rather than being relaxed to D6's "every colliding pair is marked
#: unresolvable": at n<=1000 a collision would be a generator bug or a changed amount
#: distribution, not an honest indistinguishable pair, and a check that silently absorbed it
#: would remove the tripwire exactly where it is load-bearing. The genuinely indistinguishable
#: case has its own home -- ``--dup-amounts`` plants it deliberately and does suspend this
#: check. If it ever does fire unexpectedly, the failure message below carries D6's instruction
#: rather than leaving a reader to invent the wrong fix.
SUSPENDED_BY: dict[str, tuple[str, ...]] = {
    "I3.cardinality":                ("batching", "noise_rows", "unsettled"),
    "I3.no_refunds":                 ("netted_refunds",),
    "I3.no_orphans":                 ("unsettled", "reserve", "noise_rows"),
    "I2.every_payment_settled":      ("unsettled",),
    "I2.every_settlement_credited":  ("reserve", "unsettled"),
    "I6.all_payments_cited":         ("unsettled", "noise_rows"),
    "I3.unique_date_amount":         ("dup_amounts",),
}


def _suspended(cfg: GenConfig, check: str, record: dict[str, list[str]]) -> bool:
    """True if ``check`` must be skipped because a flag it cannot survive is on.

    Records *why* into ``record``, so the run can announce the skip instead of quietly
    performing fewer checks than it appears to. Silence is what made the old gate
    dangerous: the checks vanished and the output looked identical.
    """
    on = [f for f in SUSPENDED_BY[check] if getattr(cfg.flags, f)]
    if on:
        record[check] = on
    return bool(on)


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
    gross_total: int,
    net_total: int,
    credit_total: int,
    fee_cells: list[int],
    *,
    fees_on: bool = False,
) -> None:
    """I4 — the money adds up in aggregate, and deductions exist only when asked for.

    Three assertions, and only the last is conditional:

      * ``net_total == credit_total`` -- every settled paisa reached the bank.
      * ``gross_total - (every fee/gst/tds cell) == net_total`` -- the wedge between
        gross and net is *exactly* the declared deductions and nothing else.
      * with ``--fees`` off, every deduction cell is zero.

    The second one is Phase 4's strengthening. Until now this asserted
    ``gross == net == credited``, which ``--fees`` necessarily violates, so the tempting
    move was to suspend it under the flag. That is precisely the footgun the
    ``SUSPENDED_BY`` docstring describes: the check would stand down at the moment the
    arithmetic it guards started moving. Subtracting the cells instead makes it
    *stronger*, and at zero deductions it is character-for-character the old assertion --
    so clean mode is still checked exactly the way Phase 1 checked it.

    Clean mode has *zero* fees, not "fees applied consistently" (trap 7): that is what
    makes the first amount mismatch in Phase 3 a one-suspect debug rather than a
    two-suspect one, and it is why ``fees_on`` gates the third assertion instead of
    ``cfg.clean_mode``. Any flag at all makes ``clean_mode`` false; only ``--fees`` is
    allowed to put a number in these columns.

    ``fee_cells`` is every fee, GST and TDS cell of every settlement, so their sum is the
    whole wedge. Splitting them per column would be no stronger here and would only give
    the two callers a chance to disagree about the order of three ints.

    Precondition, and Phase 7 is where it breaks: every payment sits in exactly one
    settlement, so summing gross over payments and net over settlements counts the same
    money. ``--unsettled`` and ``--reserve`` void that and belong in ``SUSPENDED_BY``
    when they land; neither is implemented yet, so neither is listed there today.
    """
    deducted = sum(fee_cells)
    _require(
        net_total == credit_total,
        "I4",
        f"settled and credited disagree: net={net_total} credited={credit_total} "
        f"({net_total - credit_total:+d} paise never reached the bank)",
    )
    _require(
        gross_total - deducted == net_total,
        "I4",
        f"the gross/net wedge is not the declared deductions: gross={gross_total} "
        f"- deductions={deducted} = {gross_total - deducted}, but net={net_total} "
        f"({gross_total - deducted - net_total:+d} paise unaccounted for)",
    )
    if not fees_on:
        nonzero = [c for c in fee_cells if c != 0]
        _require(
            not nonzero,
            "I4",
            f"{len(nonzero)} non-zero fee/gst/tds cells while --fees is off: {nonzero[:5]}",
        )


def check_settlement_arithmetic(rows: list[tuple[str, int, int, int, int, int]]) -> None:
    """I4 — per settlement: ``net == gross - fee - gst - tds``, and GST sits on the fee.

    ``rows`` is ``(settlement_id, gross_of_members, net, fee, gst, tds)``.

    The per-row companion to ``check_totals``, and it earns its keep for a reason the
    aggregate cannot cover: totals that agree in sum can still be wrong row by row, and
    two compensating errors in opposite directions cancel exactly. Under ``--fees`` every
    settlement carries its own rounding, so this is where a half-up slip in one direction
    would show.

    Both callers previously had this as an inline "net equals the member gross **while
    ``--fees`` is off**" check. Same treatment as I11 and ``check_totals``: strengthened
    to the full subtraction rather than suspended, which at zero deductions is the old
    equality unchanged.

    The second assertion is not arithmetic bookkeeping but a check on the *composition*:
    GST is a share of the fee, never of the gross. Since the GST rate is well under 100%,
    ``gst <= fee`` holds for any fee -- and it fails immediately if the two rates are ever
    applied to the same base, which inflates GST by roughly fifty times and is the single
    easiest error available in this model (see ``story._deductions``). Deliberately
    rate-free: re-deriving the fee from ``cfg.fees`` here would only assert that the
    generator agrees with itself. The independent re-derivation is the *matcher's* job,
    and the residual closing to zero is what proves the two sides agree.
    """
    for sid, gross, net, fee, gst, tds in rows:
        _require(
            net == gross - fee - gst - tds,
            "I4",
            f"{sid}: net {net} != gross {gross} - fee {fee} - gst {gst} - tds {tds} "
            f"= {gross - fee - gst - tds}",
        )
        _require(
            gst <= fee,
            "I4",
            f"{sid}: gst {gst} exceeds fee {fee} -- GST is charged on the fee, not on "
            f"the gross, so this is the two rates applied to the same base",
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

    **Both dicts must be keyed by the same date and hold the same payments per key.**
    The caller keys both on the credit's ``value_date``, which is not a detail: keying the
    payment side on ``business_date`` instead was correct only while the two dates were
    equal. Under Phase 4's posting lag the keys offset by n business days, so each block
    compared a credit list against the payments of a *different* day -- the rate collapsed
    from 19/58 to 0/58 and the check could no longer fail at all. An invariant that goes
    quietly inert the moment the thing it guards starts moving is worse than one that was
    never written, because the report still prints a number for it.
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

def check_story(
    story: Story, cfg: GenConfig, calendar: BusinessCalendar | None = None
) -> dict[str, object]:
    """Run every invariant on the in-memory story. Raises ``InvariantError``.

    Returns a small report dict for the CLI to echo, so a passing run still shows
    the numbers behind the pass rather than only the word "ok".

    ``calendar`` is injectable for the same reason ``story.build`` takes one: I11 recomputes
    the two delays and must do it over the *same* calendar the generator used. Pass the
    same object to both, or a holiday set on one side and not the other turns a correct
    story into an invariant failure -- and a shared default would hide that a caller
    forgot to pass it.
    """
    cal = calendar or BusinessCalendar()
    payments, settlements, credits = story.payments, story.settlements, story.credits

    # I1 — unique ids within each file
    check_unique_ids("I1", "payment", [p.payment_id for p in payments])
    check_unique_ids("I1", "order", [p.order_id for p in payments])
    check_unique_ids("I1", "settlement", [s.settlement_id for s in settlements])
    check_unique_ids("I1", "credit", [c.credit_id for c in credits])
    check_unique_ids("I1", "refund", [r.refund_id for r in story.refunds])

    #: Conditional checks that did not run, and the flag that excused each. Echoed by
    #: the CLI and carried into the report: a skipped check must be *visible*, because
    #: an invisible skip is exactly what made the old ``clean_mode`` gate dangerous.
    skipped: dict[str, list[str]] = {}

    # I3 — cardinality, refunds and orphans. Three checks, three different flag sets:
    # --fees breaks none of them, which is the whole point of splitting the old gate.
    if not _suspended(cfg, "I3.cardinality", skipped):
        _require(
            len(payments) == len(settlements) == len(credits) == cfg.n,
            "I3",
            f"expected 1:1:1 cardinality at n={cfg.n}, got "
            f"{len(payments)}/{len(settlements)}/{len(credits)}",
        )
    if not _suspended(cfg, "I3.no_refunds", skipped):
        _require(not story.refunds, "I3", "refunds emitted without --netted-refunds")
    if not _suspended(cfg, "I3.no_orphans", skipped):
        _require(
            not (story.unsettled_payment_ids or story.settlements_without_credit
                 or story.non_gateway_credit_ids),
            "I3",
            "orphans or noise rows present while every flag that creates them is off",
        )

    # I2 — every payment in exactly one settlement; every settlement in one credit
    settled_count: dict[str, int] = defaultdict(int)
    for s in settlements:
        for pid in s.payment_ids:
            settled_count[pid] += 1
    payment_ids = {p.payment_id for p in payments}
    unknown = sorted(set(settled_count) - payment_ids)
    _require(not unknown, "I2", f"settlements reference unknown payments: {unknown[:5]}")
    if not _suspended(cfg, "I2.every_payment_settled", skipped):
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
    if not _suspended(cfg, "I2.every_settlement_credited", skipped):
        uncredited = sorted(settlement_ids - set(credited_count))
        _require(not uncredited, "I2", f"settlements never credited: {uncredited[:5]}")
    multi_s = sorted(sid for sid, k in credited_count.items() if k > 1)
    _require(not multi_s, "I2", f"settlements in more than one credit: {multi_s[:5]}")

    # I4 — the money adds up, in aggregate and then per settlement
    gross_of = {p.payment_id: p.gross_paise for p in payments}
    check_totals(
        story.total_gross_paise(),
        story.total_net_paise(),
        story.total_credited_paise(),
        [x for s in settlements for x in (s.fee_paise, s.gst_paise, s.tds_paise)],
        fees_on=cfg.flags.fees,
    )
    check_settlement_arithmetic(
        [
            (
                s.settlement_id,
                sum(gross_of[pid] for pid in s.payment_ids),
                s.net_paise,
                s.fee_paise,
                s.gst_paise,
                s.tds_paise,
            )
            for s in settlements
        ]
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
    if not _suspended(cfg, "I6.all_payments_cited", skipped):
        cited = {pid for c in credits for pid in c.payment_ids}
        _require(
            cited == payment_ids,
            "I6",
            f"{len(payment_ids - cited)} payments are in no credit's truth entry",
        )

    # I11 — both dates are exactly where the declared delays put them.
    #
    # These two checks used to assert plain equality "while --settlement-delay is off".
    # Phase 4 makes them *stronger* rather than suspending them, for the reason the
    # SUSPENDED_BY docstring gives: a check that stands down exactly when the thing it
    # guards starts moving is the footgun, not the policy. At delay 0 and lag 0 the
    # assertion below is character-for-character the old one; under the flag it is a real
    # test of the delay model, and it is the only thing that would catch a silent
    # off-by-one in either direction.
    #
    # Note which of the two the matcher can see. The capture->settlement delay is
    # invisible to it (the join never reads captured_at); the settlement->credit lag is
    # the one the date window is measured against. Getting the first wrong is a
    # generator-only bug that nothing downstream would ever reveal, which is precisely
    # why it is asserted here.
    by_pay = {p.payment_id: p for p in payments}
    by_setl = {s.settlement_id: s for s in settlements}
    for s in settlements:
        for pid in s.payment_ids:
            want = cal.add_business_days(by_pay[pid].business_date, cfg.delay_days)
            _require(
                s.settled_on == want,
                "I11",
                f"{s.settlement_id} settled {s.settled_on} but {pid} captured "
                f"{by_pay[pid].business_date}, which is {want} at a "
                f"{cfg.delay_days}-business-day settlement delay",
            )
    for c in credits:
        for sid in c.settlement_ids:
            want = cal.add_business_days(by_setl[sid].settled_on, cfg.lag_days)
            _require(
                c.value_date == want,
                "I11",
                f"{c.credit_id} dated {c.value_date} but {sid} settled "
                f"{by_setl[sid].settled_on}, which is {want} at a "
                f"{cfg.lag_days}-business-day bank posting lag",
            )

    # The data must be resolvable from (date, amount) -- row 1 of the mess dial. A
    # duplicate pair is genuinely indistinguishable from the two legitimate signals, so
    # the honest verdict would be an abstention. That case is real and --dup-amounts
    # plants it deliberately in Phase 4b; it must not appear here by accident.
    #
    # Suspended only by --dup-amounts. Step 6 measured whether --fees or
    # --settlement-delay can force a collision, because the plan predicted both would and
    # a suspension was on the table if they did. Neither does: the delay's date map is
    # injective so it only relabels days, and fees *disperse* amounts (the per-method
    # rates push equal-gross payments onto different nets) rather than compressing them.
    # Zero collisions in 48 runs, first appearing at n=4000 -- see SUSPENDED_BY's docstring
    # and ASSUMPTIONS.md #24a/#24b for the numbers.
    #
    # That margin is why this check stays strict. At the sizes this project runs, a
    # collision is a generator bug or a changed amount distribution, not an honest
    # indistinguishable pair -- so the failure message below sorts the two cases by size
    # rather than leaving a reader to guess which one they have.
    if not _suspended(cfg, "I3.unique_date_amount", skipped):
        key_counts = Counter((c.value_date, c.amount_paise) for c in credits)
        dupes = {k for k, count in key_counts.items() if count > 1}
        # Named so the two halves of the guidance below cannot drift apart from the
        # measurement that justifies them.
        measured_clean_to = 2000
        _require(
            not dupes,
            "I3",
            f"{len(dupes)} duplicate (date, amount) credits are unresolvable from the "
            f"inputs: {sorted(dupes)[:3]}\n"
            f"  Do NOT resample it away: that would delete the most honest row in the "
            f"file to protect an invariant.\n"
            f"  This run is n={cfg.n}. Measured (ASSUMPTIONS.md #24a): the pair is "
            f"collision-free to n={measured_clean_to} on seeds 1/2/3/42 under any "
            f"combination of --fees and --settlement-delay, with the first natural "
            f"collision at n=4000.\n"
            + (
                f"  At n={cfg.n} a collision is therefore NOT expected pressure -- suspect "
                f"a generator change (the amount distribution, the fee rates, or "
                f"_unique_amount's key) before concluding the data is honestly ambiguous."
                if cfg.n <= measured_clean_to
                else f"  At n={cfg.n} this is past the measured range and may be a genuine "
                f"indistinguishable pair. Then decision D6 applies: mark it "
                f"resolvable=false in truth (ASSUMPTIONS.md #24) so the matcher is scored "
                f"on abstaining rather than on guessing."
            ),
        )

    # I12 — the successor to the check --dup-amounts suspends.
    #
    # ``I3.unique_date_amount`` stands down under this flag because the collision is the
    # *intent*. A suspension that is merely announced still leaves the run less checked than
    # it looks: with I3 off, a planting bug that collided three rows, or zero, or that
    # collided the amounts while leaving the UTRs distinct, would pass everything. So the
    # flag brings its own invariant, and it is stricter than the one it replaces -- it
    # asserts the collision is exactly what was ordered.
    #
    # The UTR clause is the load-bearing one, and it is why this check exists rather than a
    # comment. Measured before the flag was built, re-run by gate 11 (tools/acceptance.py): a
    # tail-only strategy reading no date and no amount resolves 60/60, 200/200 and 1000/1000
    # credits *correctly* on every dev seed, because tails are drawn without replacement. A
    # pair colliding on (date, amount) with distinct tails is therefore still separable by
    # exhaustive narration matching -- ``resolvable=False`` would be a false statement about
    # the data, and the flag would not test the capability its name claims. That is finrecon's
    # recorded failure: its showcase tier fell to narration enumeration, 200/200.
    if cfg.flags.dup_amounts:
        key_counts = Counter((c.value_date, c.amount_paise) for c in credits)
        groups = sorted(k for k, count in key_counts.items() if count > 1)
        _require(
            len(groups) == cfg.dup_pairs,
            "I12",
            f"--dup-amounts asked for {cfg.dup_pairs} planted pair(s) but the data holds "
            f"{len(groups)} colliding (date, amount) group(s). The planted count is the "
            f"denominator of the correct_abstention rate, so a wrong count silently "
            f"rescales this project's central claim.",
        )
        utr_of = {s.settlement_id: s.utr for s in settlements}
        unresolvable = [c for c in credits if not c.resolvable]
        for key in groups:
            members = [c for c in credits if (c.value_date, c.amount_paise) == key]
            _require(
                len(members) == 2,
                "I12",
                f"planted group {key} has {len(members)} members, not 2. Three credits "
                f"sharing a (date, amount) is a different and harder case than the pair "
                f"this flag documents, and it would be scored as though it were the pair.",
            )
            utrs = {utr_of[sid] for c in members for sid in c.settlement_ids}
            _require(
                len(utrs) == 1,
                "I12",
                f"planted group {key} spans {len(utrs)} distinct UTRs ({sorted(utrs)}). "
                f"The UTR tail reaches the bank narration and resolves 100% of rows on its "
                f"own, so a pair with distinct tails is still separable by exhaustive "
                f"narration matching -- it is NOT unresolvable, and marking it "
                f"resolvable=false would be a false statement about the data.\n"
                f"  Do not fix this by removing the UTR from the narration: that is "
                f"--utr-patchy's job (Phase 8) and it would make absence-of-tail a tell "
                f"unique to planted rows.",
            )
            for c in members:
                _require(
                    not c.resolvable
                    and c.reason == str(Reason.AMBIGUOUS_DUPLICATE_AMOUNT),
                    "I12",
                    f"{c.credit_id} shares a (date, amount, utr) with another credit but "
                    f"truth marks it resolvable={c.resolvable} reason={c.reason!r}. An "
                    f"indistinguishable row recorded as resolvable would score an honest "
                    f"abstention as a MISS.",
                )
        _require(
            len(unresolvable) == 2 * cfg.dup_pairs,
            "I12",
            f"{len(unresolvable)} credit(s) are marked unresolvable but "
            f"{2 * cfg.dup_pairs} were planted. A row marked unresolvable outside a planted "
            f"group inflates the correct_abstention denominator with a row the matcher "
            f"could in fact have resolved.",
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

    # I8b — within-block position carries no information.
    #
    # Both sides are keyed on the **credit's** value_date, so a block holds one day of
    # bank rows and exactly the payments those rows paid out. Keying the payment side on
    # ``business_date`` was equivalent only while the two dates agreed; under a posting
    # lag it compared each credit block against a different day's payments and the check
    # went inert (see check_within_block_alignment's docstring).
    #
    # Payments keep capture order within a block because ``payments`` is capture-sorted
    # and this loop appends in that order -- which is the ordering the check is *about*:
    # bank rows are sorted by amount, and if position still tracked capture time a zip
    # matcher would score without doing any work.
    value_date_of: dict[str, date] = {
        pid: c.value_date for c in credits for pid in c.payment_ids
    }
    p_by_date: dict[date, list[str]] = defaultdict(list)
    for p in payments:
        # A payment in no credit is Phase 7's --unsettled; it belongs to no bank block.
        if (when := value_date_of.get(p.payment_id)) is not None:
            p_by_date[when].append(p.payment_id)
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
        # Which conditional checks did not run, and the flag that excused each. Empty in
        # clean mode. This travels into run_manifest.json beside "status": "pass", which
        # is the honest pairing: everything that ran, passed -- and here is what did not
        # run. A pass count that quietly shrinks with each new flag is how a generator
        # ends up certifying data nothing checked.
        "checks_skipped": {k: list(v) for k, v in sorted(skipped.items())},
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
    fired: list[str] = []

    def must_fail(code: str, broken: Story, cfg_: GenConfig = cfg) -> None:
        try:
            check_story(broken, cfg_)
        except InvariantError as e:
            assert str(e).startswith(code), f"expected {code}, got: {e}"
            fired.append(code)
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

    # --- step 4: the fee arithmetic ------------------------------------------
    # ``fees`` is in MessFlags.IMPLEMENTED as of step 4, so this config needs no patching
    # seam -- the flag now genuinely changes the data.
    fees_cfg = GenConfig(seed=42, n=60, flags=MessFlags(fees=True))
    fees_story = build(fees_cfg)
    assert check_story(fees_story, fees_cfg)["checks_skipped"] == {}, (
        "--fees must suspend nothing at all"
    )

    # I4, the per-settlement check, on the case the aggregate provably cannot see: move a
    # paisa of fee from one settlement to another. Both totals still agree to the paisa --
    # the deduction sum is unchanged and no net or credit moved -- so every aggregate
    # assertion passes and only the per-row subtraction fails. Two compensating errors
    # cancelling is not a hypothetical; it is what a sign slip in a batching loop looks
    # like, and it is the whole reason check_settlement_arithmetic exists.
    # Chosen by carrying a fee, not by index: ``pos_upi`` is zero-rated, so a settlement in
    # shuffled order may have nothing to move. That was near-certain when the zero-rated
    # share was ~36% of rows and is merely possible now that it is ~6% -- which is exactly
    # why the selection stays written this way. A probe that passes because the sample got
    # lucky is a probe that fails on a seed nobody ran. And the
    # replacement is positional -- reordering the list would change what I8a measures, so
    # the probe would risk failing for the wrong reason.
    a_s, b_s = [s for s in fees_story.settlements if s.fee_paise][:2]
    nudged = {a_s.settlement_id: 1, b_s.settlement_id: -1}
    shifted = dataclasses.replace(
        fees_story,
        settlements=[
            dataclasses.replace(s, fee_paise=s.fee_paise + nudged[s.settlement_id])
            if s.settlement_id in nudged
            else s
            for s in fees_story.settlements
        ],
    )
    _cells = [x for s in shifted.settlements for x in (s.fee_paise, s.gst_paise, s.tds_paise)]
    assert sum(_cells) == sum(
        x for s in fees_story.settlements for x in (s.fee_paise, s.gst_paise, s.tds_paise)
    ), "the probe must leave the aggregate untouched, or it tests the wrong assertion"
    must_fail("I4", shifted, fees_cfg)

    # The two arithmetic checks, probed directly. Some failures cannot be reached through
    # a whole story without tripping an earlier assertion first -- a GST error large
    # enough to exceed its fee also breaks the subtraction and the totals -- so the unit
    # probe is the only way to know the specific assertion fires rather than a neighbour.
    def must_raise(what: str, fn) -> None:
        try:
            fn()
        except InvariantError as e:
            assert str(e).startswith("I4"), f"{what}: expected I4, got: {e}"
            fired.append("I4")
        else:
            raise AssertionError(f"{what} did not fire")

    # GST charged on the gross instead of on the fee: 18% of a ₹10,000 gross is nine times
    # a 2% fee, so it lands above the fee it is supposed to sit on.
    must_raise(
        "gst on the gross",
        lambda: check_settlement_arithmetic(
            [("setl_x", 1_000_000, 1_000_000 - 20_000 - 180_000, 20_000, 180_000, 0)]
        ),
    )
    must_raise(
        "net off by a paisa",
        lambda: check_settlement_arithmetic(
            [("setl_x", 1_000_000, 976_401, 20_000, 3_600, 0)]
        ),
    )
    # ... and the correct row passes, or the probe above proves nothing.
    check_settlement_arithmetic([("setl_x", 1_000_000, 976_400, 20_000, 3_600, 0)])
    # Zero deductions: the assertion is the pre-Phase-4 equality, unchanged.
    check_settlement_arithmetic([("setl_x", 1_000_000, 1_000_000, 0, 0, 0)])

    must_raise(
        "wedge is not the deductions",
        lambda: check_totals(1_000_000, 976_400, 976_400, [20_000, 3_500, 0], fees_on=True),
    )
    must_raise(
        "settled but never credited",
        lambda: check_totals(1_000_000, 976_400, 976_399, [20_000, 3_600, 0], fees_on=True),
    )
    check_totals(1_000_000, 976_400, 976_400, [20_000, 3_600, 0], fees_on=True)
    check_totals(1_000_000, 1_000_000, 1_000_000, [0, 0, 0])

    # I3: a duplicate (date, amount) pair.
    # Cloning only the credit's amount would change the credited total and trip I4
    # first, so the whole chain behind it is rewritten -- payment gross, settlement
    # net, credit amount, decomposition.
    #
    # Phase 4b made the real thing available, and this hand-built story is now more useful
    # as the *wrong* plant than as a rehearsal for the right one: it collides one pair while
    # the config asks for two, it leaves the two settlements holding **distinct UTRs**, and
    # it leaves both credits marked ``resolvable=True``. All three are what I12 exists to
    # reject, so it serves below as I12's negative probe while remaining I3's.
    a, b = story.credits[0], story.credits[1]
    a_pid, b_pid = a.payment_ids[0], b.payment_ids[0]
    a_sid, b_sid = a.settlement_ids[0], b.settlement_ids[0]
    a_pay = next(p for p in story.payments if p.payment_id == a_pid)
    dup_story = dataclasses.replace(
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
    )
    must_fail("I3", dup_story)

    # --- step 1: suspension works, and only the right flag does it ------------
    # A *genuinely* built --dup-amounts story: the duplicate is the intent, so
    # I3.unique_date_amount is suspended -- and the report says so, rather than quietly
    # running one check fewer. I12 must pass here, on data the generator really produces.
    dup_cfg = GenConfig(seed=42, n=60, flags=MessFlags(dup_amounts=True))
    real_dup = build(dup_cfg)
    rep = check_story(real_dup, dup_cfg)
    assert rep["checks_skipped"] == {"I3.unique_date_amount": ["dup_amounts"]}, rep

    # --- I12: the successor check must reject every wrong plant ----------------
    # Three ways to plant badly, and I12 exists because a suspension that is only
    # *announced* leaves all three passing. Each is probed on its own, because a check that
    # fires for the wrong reason is indistinguishable from one that works.
    #
    # (i) the wrong *count*. dup_story collides one pair; the config asks for two.
    must_fail("I12", dup_story, dup_cfg)

    # (ii) distinct UTRs -- the load-bearing clause. Same fixture, but with the count
    # reconciled so the first clause passes and this one is what fires. dup_story leaves the
    # two settlements holding their original, distinct tails, which is precisely the plant
    # that measured as *still separable* by a tail-only strategy (60/60, 200/200, 1000/1000
    # correct) before this flag was built.
    one_pair_cfg = GenConfig(seed=42, n=60, flags=MessFlags(dup_amounts=True), dup_pairs=1)
    must_fail("I12", dup_story, one_pair_cfg)

    # (iii) a planted row recorded as resolvable. Truth would then score an honest
    # abstention as a MISS, and the correct_abstention denominator would be wrong in the
    # direction that flatters the matcher.
    _planted_ids = [c.credit_id for c in real_dup.credits if not c.resolvable]
    assert len(_planted_ids) == 2 * dup_cfg.dup_pairs, _planted_ids
    mismarked = dataclasses.replace(
        real_dup,
        credits=[
            dataclasses.replace(c, resolvable=True, reason=None, note=None)
            if c.credit_id == _planted_ids[0]
            else c
            for c in real_dup.credits
        ],
    )
    must_fail("I12", mismarked, dup_cfg)

    # --fees must NOT excuse a collision. This is the exact conflation the old clean_mode
    # gate made: fees change amounts, so they raise collision pressure, but a collision is a
    # finding to record (ASSUMPTIONS.md #24) and not a check to switch off. Under the old
    # gate this story passed silently.
    fees_cfg = GenConfig(seed=42, n=60, flags=MessFlags(fees=True))
    must_fail("I3", dup_story, fees_cfg)
    assert check_story(story, fees_cfg)["checks_skipped"] == {}, (
        "--fees must suspend nothing at all"
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

    # I11: either date off by a day, in either direction. Four cases, because the two
    # delays fail independently and only one of them is visible to the matcher -- a
    # capture->settlement error would never surface downstream, so if it is not caught
    # here it is not caught at all.
    from datetime import timedelta

    for label, mutate in (
        ("settled_on late", lambda st: dataclasses.replace(
            st, settlements=[dataclasses.replace(st.settlements[0],
                                                 settled_on=st.settlements[0].settled_on
                                                 + timedelta(days=7)),
                             *st.settlements[1:]])),
        ("settled_on early", lambda st: dataclasses.replace(
            st, settlements=[dataclasses.replace(st.settlements[0],
                                                 settled_on=st.settlements[0].settled_on
                                                 - timedelta(days=7)),
                             *st.settlements[1:]])),
        ("value_date late", lambda st: dataclasses.replace(
            st, credits=[dataclasses.replace(st.credits[0],
                                             value_date=st.credits[0].value_date
                                             + timedelta(days=7)),
                         *st.credits[1:]])),
        ("value_date early", lambda st: dataclasses.replace(
            st, credits=[dataclasses.replace(st.credits[0],
                                             value_date=st.credits[0].value_date
                                             - timedelta(days=7)),
                         *st.credits[1:]])),
    ):
        # Seven days, not one: a one-day nudge off a Friday lands on a weekend, and
        # add_business_days rolls a weekend forward to the same Monday -- so the story
        # would still satisfy I11 and the negative case would silently not fire.
        must_fail("I11", mutate(story))

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

    # Counted, not written down: the literal "6" here went stale the moment Phase 4
    # added a seventh case, and a self-check that misreports its own coverage is a
    # small version of exactly the problem this phase is fixing.
    print(
        f"invariants.py self-check ok  ({len(fired)} negative cases fire correctly: "
        f"{', '.join(sorted(set(fired)))})"
    )
