"""Re-read the written files from disk and re-check every invariant.

    python tools/verify_output.py [--data data] [--truth truth]

Why a second pass, when ``invariants.check_story`` already ran in-memory: the
write step itself can corrupt. Column order, quoting, encoding, a stray newline,
an int that became ``'1000000.0'`` -- none of those are visible to a check that
runs on dataclasses. This tool sees only what a judge would see.

**The reader here is deliberately independent of ``emit.py``.** It parses the CSVs
with its own code rather than round-tripping the generator's writers, so a bug in
the writer cannot hide behind a symmetric bug in the reader. It also doubles as the
first exercise of the Phase 2/3 loader path.

Section 2 is the **leak audit**, which replaces acceptance gate 4 from the plan.

The plan's wording was "confirm you cannot tell, by eye, which payment any credit
came from". That is not achievable and should not be: clean mode pins
``value_date == capture date`` and ``amount == gross``, so date+amount *does*
identify the payment -- it has to, because row 1 of the mess dial says clean mode
must be 100% resolvable. If a human genuinely could not tell, the matcher could
not either, and the phase would be unbuildable.

What the gate is really protecting is that the answer must not be readable from
**structure**. So the audit measures four resolution strategies and asserts a gap:

  * by row position        -- must be far below 100%
  * by ID numbering        -- must be at chance
  * by narration           -- must be exactly 0 (nothing to read)
  * by date + amount       -- must be exactly 100% (the legitimate signal)

The gap between the last line and the first three is the proof that resolving this
dataset requires the arithmetic join. That is a measurement a judge can rerun,
which is strictly better than "I looked at it and could not tell."
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hisaab.common import ids  # noqa: E402
from hisaab.generator.invariants import (  # noqa: E402
    InvariantError,
    check_batch_adjacency,
    check_headers,
    check_int_money,
    check_no_leak,
    check_settlement_arithmetic,
    check_totals,
    check_unique_ids,
    check_within_block_alignment,
)
from hisaab.generator.model import (  # noqa: E402
    BANK_HEADER,
    PAYMENTS_HEADER,
    REFUNDS_HEADER,
    SETTLEMENT_ITEMS_HEADER,
    SETTLEMENTS_HEADER,
)
from hisaab.scoring.truth_io import load_truth  # noqa: E402

CSV_FILES = {
    "payments.csv": PAYMENTS_HEADER,
    "settlements.csv": SETTLEMENTS_HEADER,
    "settlement_items.csv": SETTLEMENT_ITEMS_HEADER,
    "bank_statement.csv": BANK_HEADER,
    "refunds.csv": REFUNDS_HEADER,
}


def read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """Read one CSV with an independent parser. Returns (header, rows)."""
    if not path.exists():
        raise InvariantError(f"I9: {path.name} was not written")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = tuple(next(reader))
        except StopIteration:
            raise InvariantError(f"I9: {path.name} is completely empty (no header)") from None
        rows = [dict(zip(header, row)) for row in reader if row]
    return header, rows


def parse_int(value: str, where: str) -> int:
    """Strict int parse. A float that leaked through arrives as '1000000.0' here."""
    try:
        return int(value)
    except ValueError:
        raise InvariantError(
            f"I5: {where} is not an integer: {value!r} -- a float reached the CSV"
        ) from None


def parse_date(value: str, where: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise InvariantError(f"{where}: not an ISO date: {value!r}") from None


def parse_captured_at(value: str, where: str) -> datetime:
    """``2026-08-10T05:34:22Z`` -> aware datetime. The trailing Z is mandatory."""
    if not value.endswith("Z"):
        raise InvariantError(
            f"{where}: timestamp must be UTC with a trailing Z, got {value!r}"
        )
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise InvariantError(f"{where}: not an ISO timestamp: {value!r}") from None


#: Which mess flags legitimately invalidate each check in *this* pass, mirroring
#: ``invariants.SUSPENDED_BY``. Same reasoning, and not a stylistic echo: this file used a
#: binary ``truth.clean_mode`` gate, and ``clean_mode`` means "all thirteen flags off", so any
#: flag at all switched off the partition check along with the cardinality one.
#:
#: Phase 5 is where that stopped being theoretical. ``--batching`` changes the payment:
#: settlement ratio and nothing else -- measured on real batched output at n=200: every
#: payment is still cited exactly once in ``settlement_items.csv``, ``truth.credits`` still
#: equals the bank row count, and settlements still equal credits. Only 1:1:1 genuinely
#: breaks. Under the old gate all four stood down together, so the *on-disk* pass -- the one
#: that sees only what a judge sees -- lost partition coverage precisely where a new grouping
#: loop could break it. ``.plan/phase5.md``'s trap 8 worried about this and located it in
#: ``invariants.py``, where measurement showed I2's unconditional clauses already catch it
#: (a dropped batch member raises "payments never settled"); here the gap was real.
#:
#: Keys are ``<invariant>.<what it asserts>``, so a typo raises ``KeyError`` at the call site
#: rather than silently never suspending.
DISK_SUSPENDED_BY: dict[str, tuple[str, ...]] = {
    # Genuinely broken by batching: n payments become ~n/1.6 settlements and that many credits.
    # ``noise_rows`` was removed in Phase 7 step 4: like ``unsettled`` before it, it **subtracts**
    # a set truth names rather than invalidating the equality -- orphans come off the payment
    # side, noise rows off the bank side. See the check itself.
    "I3.one_to_one_to_one":     ("batching",),
    "I3.no_refunds":            ("netted_refunds",),
    # **``unsettled`` was removed from two entries in Phase 7 step 1, and the removals are the
    # correction rather than an omission.** ``.plan/phase7.md`` correction (d) requires *both*
    # suspension lists to be re-derived rather than trusted, and this list is the one a reader
    # misses -- it has a different name from ``invariants.SUSPENDED_BY`` and different keys, and
    # it is the pass that sees only what a judge sees.
    #
    # ``I2.every_payment_settled`` had its own entry here and no longer needs one: truth.json
    # **names** the payments that never settled, so the check became an equality against that
    # list instead of standing down (see the I2 block below). ``I3.one_to_one_to_one`` subtracts
    # the orphan count for the same reason. Both are strictly stronger than a suspension, and
    # they matter most on exactly the runs the suspension covered: a payment that goes missing
    # for some *other* reason while ``--unsettled`` is on is still caught.
    #
    # What ``--unsettled`` genuinely needs from this file is a new **term**, not a new
    # suspension: I4's wedge gains the never-settled gross, sourced from the payments truth
    # names because no settlement or credit cell mentions an orphan at all.
    # **Emptied in Phase 7 step 4, and the key is kept deliberately** -- ``_disk_suspended``
    # indexes this dict, so deleting the entry would raise ``KeyError`` at the call site rather
    # than un-suspend the check. An empty tuple is the honest statement: nothing suspends this.
    #
    # The old comment here read "a noise row is a bank row with no truth entry, so the two counts
    # diverge", and the first clause is simply **false** -- a noise row *is* in truth, named under
    # ``orphans.non_gateway_credit_ids``. What it lacks is a full ``credits`` entry, which is a
    # different thing, and the difference is exactly what lets this check be strengthened into an
    # equality on identities instead of stood down. A prediction that was wrong about the data,
    # kept visible rather than quietly deleted.
    "I6.truth_covers_bank":     (),
    # The legitimate signal must resolve the dataset. This list was **corrected by
    # measurement** rather than reasoned, and it was wrong in both directions first time:
    #
    #   clean          200/200      fees            200/200      batching       120/120
    #   fees+batching  120/120      delay             0/200      delay+batching   0/120
    #   fees+delay       0/200      dup_amounts     196/200
    #
    # ``settlement_delay`` breaks it **completely**, and was omitted from the first draft of
    # this list. Strategy D joins on an *exact* date, and the posting lag puts every credit one
    # business day after its settlement, so the join finds nothing at all -- 0/200, not a
    # shortfall. That is the same fact the matcher's ``--window`` exists for
    # (ASSUMPTIONS.md #15b: window 0 scores 0% coverage under the lag, window 1 scores 100%),
    # seen from the audit side.
    #
    # ``settlement_report_late`` was in the first draft and does **not** belong: it withholds
    # ``settlement_items.csv``, which declares payment-to-settlement membership, while this
    # strategy joins credits to *settlements* on (date, amount) and never reads membership.
    # Phase 5 step 5 will withhold that file and this check must keep passing.
    #
    # ``dup_amounts`` falls short by exactly the planted rows (196/200 = 2 pairs) and that is
    # the flag working. ``fees`` and ``batching`` are deliberately absent -- both still resolve
    # 100%, so both are now *asserted* where the old ``clean_mode`` gate skipped them, which is
    # a strengthening and the same one Phase 4 made in ``invariants.py``.
    "I3.date_amount_resolves":  ("settlement_delay", "dup_amounts"),
}


def _disk_suspended(flags: dict[str, bool], check: str) -> list[str]:
    """Flags this run turns on that legitimately invalidate ``check``."""
    return [f for f in DISK_SUSPENDED_BY[check] if flags.get(f)]


def verify(data_dir: Path, truth_dir: Path, verbose: bool = True) -> dict[str, object]:
    """Re-run every file-level invariant plus the leak audit. Raises on failure."""
    files = {name: read_csv(data_dir / name) for name in CSV_FILES}

    # --- I9: frozen headers -------------------------------------------------
    check_headers({name: header for name, (header, _) in files.items()})

    payments = files["payments.csv"][1]
    settlements = files["settlements.csv"][1]
    items = files["settlement_items.csv"][1]
    bank = files["bank_statement.csv"][1]
    refunds = files["refunds.csv"][1]

    # --- Trap 5: no CR survived the write ----------------------------------
    for name in CSV_FILES:
        raw = (data_dir / name).read_bytes()
        if b"\r" in raw:
            raise InvariantError(
                f"I9: {name} contains a CR byte -- csv.writer's default line "
                f"terminator or a missing newline='' got through, so the files are "
                f"no longer byte-identical across platforms"
            )
        if raw and not raw.endswith(b"\n"):
            raise InvariantError(f"I9: {name} does not end with a newline")

    # --- I1: unique ids within each file -----------------------------------
    check_unique_ids("I1", "payment", [r["payment_id"] for r in payments])
    check_unique_ids("I1", "order", [r["order_id"] for r in payments])
    check_unique_ids("I1", "settlement", [r["settlement_id"] for r in settlements])
    check_unique_ids("I1", "credit", [r["row_id"] for r in bank])
    check_unique_ids("I1", "refund", [r["refund_id"] for r in refunds])

    # --- I5: money survived the round trip as ints -------------------------
    money: dict[str, object] = {}
    gross_by_payment: dict[str, int] = {}
    for r in payments:
        pid = r["payment_id"]
        gross_by_payment[pid] = parse_int(r["gross_paise"], f"{pid}.gross_paise")
        money[f"{pid}.gross_paise"] = gross_by_payment[pid]
    net_by_settlement: dict[str, int] = {}
    # Kept per settlement, not just flattened: the aggregate check sums them, and
    # check_settlement_arithmetic needs each row's own three deductions to catch two
    # compensating errors that cancel in the total.
    deductions: dict[str, tuple[int, int, int]] = {}
    # Split per column for I4's zero clause, while ``deductions`` keeps all three per row for
    # ``check_settlement_arithmetic``. Two different granularities of the same numbers: the
    # aggregate needs to know *which* flag owns each column, the per-row check needs the row.
    fee_cells: list[int] = []
    tds_cells: list[int] = []
    for r in settlements:
        sid = r["settlement_id"]
        net_by_settlement[sid] = parse_int(r["net_paise"], f"{sid}.net_paise")
        money[f"{sid}.net_paise"] = net_by_settlement[sid]
        cells = tuple(parse_int(r[col], f"{sid}.{col}")
                      for col in ("fee_paise", "gst_paise", "tds_paise"))
        deductions[sid] = cells  # type: ignore[assignment]
        fee_cells.extend(cells[:2])
        tds_cells.append(cells[2])
        for col, value in zip(("fee_paise", "gst_paise", "tds_paise"), cells):
            money[f"{sid}.{col}"] = value
    amount_by_credit: dict[str, int] = {}
    for r in bank:
        cid = r["row_id"]
        amount_by_credit[cid] = parse_int(r["amount_paise"], f"{cid}.amount_paise")
        money[f"{cid}.amount_paise"] = amount_by_credit[cid]
    check_int_money(money)

    # Truth is loaded here rather than at I3 because I4 now needs to know whether
    # ``--fees`` was on: the "every deduction cell is zero" assertion is gated on that one
    # flag, not on ``clean_mode``. Taking it from the answer key's own flag block keeps
    # this pass independent of the generator's config object -- it sees only what was
    # written, which is the whole point of the second pass.
    truth = load_truth(truth_dir)
    flags: dict[str, bool] = {k: bool(v) for k, v in truth.flags.items()}
    fees_on = flags.get("fees", False)
    # Phase 6: read per column, because I4's zero clause is now per column. One flag gating
    # three columns both refused a legal ``--tds`` run and hid a stray TDS cell on a
    # ``--fees`` run -- see ``invariants.check_totals``.
    tds_on = flags.get("tds", False)
    refunds_on = flags.get("netted_refunds", False)
    reserve_on = flags.get("reserve", False)
    late_on = flags.get("settlement_report_late", False)
    unsettled_on = flags.get("unsettled", False)
    noise_on = flags.get("noise_rows", False)
    #: settlement -> payments, as the *answer key* declares it. Used only to stand in for the
    #: rows ``--settlement-report-late`` withholds from settlement_items.csv, so that the
    #: partition and per-settlement arithmetic checks keep running at full coverage instead of
    #: being suspended. Built only from credits citing exactly one settlement; a credit paid by
    #: several cannot attribute its payments to one of them, and inventing an attribution here
    #: would be worse than declining to check that row.
    truth_members: dict[str, tuple[str, ...]] = {}
    for c in truth.credits:
        if len(c.settlement_ids) == 1:
            truth_members[c.settlement_ids[0]] = tuple(c.payment_ids)
    truth_members_all = {pid for pids in truth_members.values() for pid in pids}
    #: Checks this pass skipped, and the flag that excused each. Reported rather than silent:
    #: an invisible skip is what made the old clean_mode gate dangerous.
    skipped: dict[str, list[str]] = {}

    # --- I4: the money adds up, in aggregate --------------------------------
    # Phase 6 step 6: the refund cells come from **truth's decomposition**, not from a
    # settlement column, because ``settlements.csv``'s header is frozen by I9 -- a netted
    # refund is declared in ``refunds.csv`` and attributed by the answer key. That makes this
    # pass read the same number a scorer will hold the matcher to. The per-settlement map below
    # is built the same way, and ``check_refunds`` on the in-memory side independently checks
    # these terms against the refund rows each credit cites, so a wrong attribution cannot
    # quietly balance here.
    refund_cells = [c.decomposition.refunds_paise for c in truth.credits]
    refunds_of_settlement: dict[str, int] = {sid: 0 for sid in net_by_settlement}
    for c in truth.credits:
        for sid in c.settlement_ids:
            if sid in refunds_of_settlement:
                refunds_of_settlement[sid] += c.decomposition.refunds_paise
    # Phase 6 step 7. Same sourcing as the refunds and one step further: a refund at least has
    # ``refunds.csv``, while the reserve is declared in **no input file at all**, so truth's
    # decomposition is its only record anywhere. Note what this term is *not* fed into --
    # ``check_settlement_arithmetic`` below is left alone, because the reserve sits between the
    # net and the credit rather than between the gross and the net. That is design B, and
    # adding it there would assert the settlement declared a smaller payout than it did.
    reserve_cells = [c.decomposition.reserve_paise for c in truth.credits]
    # Phase 7 step 1, and the sourcing is one step further out again. A refund has
    # ``refunds.csv`` and a reserve has truth's decomposition; a never-settled payment has
    # **no cell anywhere** -- no settlement claims it and no credit mentions it, which is what
    # being an orphan means. So the term is the gross of the payments truth *names* as
    # unsettled, read against ``payments.csv``'s own amounts.
    #
    # A name truth carries that ``payments.csv`` does not is a defect rather than a term, and
    # it is refused here instead of raising ``KeyError`` from the comprehension: I13 pins the
    # payment count unconditionally, so an orphan missing from the file means the answer key
    # and the data disagree about which payments exist.
    unknown_orphans = sorted(set(truth.unsettled_payment_ids) - set(gross_by_payment))
    if unknown_orphans:
        raise InvariantError(
            f"I4: truth names {len(unknown_orphans)} unsettled payment(s) that are not in "
            f"payments.csv: {unknown_orphans[:5]} -- the answer key and the data disagree "
            f"about which payments exist, so no gross total means what it claims"
        )
    unsettled_cells = [gross_by_payment[pid] for pid in truth.unsettled_payment_ids]

    # Phase 7 step 4, and the sourcing is the mirror image of the orphan term above. An orphan
    # is a payment **no** bank row mentions; a noise row is a bank row **no** settlement
    # mentions. So this term is read from ``bank_statement.csv``'s own amounts, keyed on the ids
    # truth names as non-gateway -- the same "read the file, keyed by the answer key" shape.
    #
    # **This term is why ``--noise-rows`` needed no suspension on this side**, and it was found
    # by measurement rather than predicted: forcing the two ``noise_rows`` entries in
    # ``DISK_SUSPENDED_BY`` to run (`.plan/probe_phase7_noise_suspensions.py`) failed on **I4**
    # rather than on either of those checks -- I4 runs first and aborts the pass, so the noise
    # total showed up as "settled and credited disagree" by exactly the noise sum. A suspension
    # of the two cardinality checks would not have fixed that and would have hidden it.
    #
    # A truth-named noise id absent from the bank file is a defect rather than a term, refused
    # here for the same reason ``unknown_orphans`` is: it would otherwise raise ``KeyError`` from
    # the comprehension, from a line that names neither the answer key nor the file.
    unknown_noise = sorted(set(truth.non_gateway_credit_ids) - set(amount_by_credit))
    if unknown_noise:
        raise InvariantError(
            f"I4: truth names {len(unknown_noise)} non-gateway row(s) that are not in "
            f"bank_statement.csv: {unknown_noise[:5]} -- the answer key names a bank row the "
            f"statement does not carry, so no credited total means what it claims"
        )
    noise_cells = [amount_by_credit[cid] for cid in truth.non_gateway_credit_ids]
    check_totals(
        sum(gross_by_payment.values()),
        sum(net_by_settlement.values()),
        sum(amount_by_credit.values()),
        fee_cells,
        tds_cells,
        refund_cells,
        reserve_cells,
        unsettled_cells,
        noise_cells=noise_cells,
        fees_on=fees_on,
        tds_on=tds_on,
        refunds_on=refunds_on,
        reserve_on=reserve_on,
        unsettled_on=unsettled_on,
        noise_on=noise_on,
    )

    # --- I3: cardinality, per flag rather than per clean_mode ----------------
    if on := _disk_suspended(flags, "I3.one_to_one_to_one"):
        skipped["I3.one_to_one_to_one"] = on
    # ``--unsettled`` subtracts rather than suspending, matching the in-memory ``I3.cardinality``
    # (Phase 7 step 1): every payment is still in the file and each orphan removes exactly the
    # one-member settlement it was in, so the settlement and bank counts fall by the orphan count
    # and nothing else. At zero orphans this is the old three-way equality unchanged.
    # ``--noise-rows`` subtracts on the **bank** side, the way ``--unsettled`` subtracts on the
    # payments side, and for the same reason: truth names the rows, so the check can account for
    # them instead of standing down. Phase 7 step 4 removed its suspension here.
    elif not (
        len(payments) - len(truth.unsettled_payment_ids)
        == len(settlements)
        == len(bank) - len(truth.non_gateway_credit_ids)
    ):
        raise InvariantError(
            f"I3: 1:1:1 must hold while every flag that changes cardinality is off -- "
            f"allowing for {len(truth.unsettled_payment_ids)} payment(s) truth names as never "
            f"settled and {len(truth.non_gateway_credit_ids)} bank row(s) truth names as "
            f"non-gateway -- got {len(payments)}/{len(settlements)}/{len(bank)}"
        )
    if on := _disk_suspended(flags, "I3.no_refunds"):
        skipped["I3.no_refunds"] = on
    elif refunds:
        raise InvariantError(
            f"I3: {len(refunds)} refund rows written while --netted-refunds is off"
        )

    # --- I2: membership, via settlement_items.csv ---------------------------
    members: dict[str, list[str]] = defaultdict(list)
    for r in items:
        members[r["settlement_id"]].append(r["payment_id"])
    unknown_s = sorted(set(members) - set(net_by_settlement))
    if unknown_s:
        raise InvariantError(f"I2: settlement_items cites unknown settlements: {unknown_s[:5]}")
    unknown_p = sorted({p for pl in members.values() for p in pl} - set(gross_by_payment))
    if unknown_p:
        raise InvariantError(f"I2: settlement_items cites unknown payments: {unknown_p[:5]}")
    counts = Counter(r["payment_id"] for r in items)
    multi = sorted(p for p, k in counts.items() if k > 1)
    if multi:
        raise InvariantError(f"I2: payments in more than one settlement: {multi[:5]}")
    # **The check Phase 5 restored.** Gated on ``clean_mode`` this stood down under any flag,
    # so a batching loop that dropped a payment from a member list passed this pass entirely.
    # Only ``--unsettled`` legitimately breaks it: batching regroups payments, it never loses
    # one, and the partition claim is what makes every per-settlement gross sum meaningful.
    # **Runs unconditionally as of Phase 7 step 1, having lost its ``unsettled`` suspension.**
    # Correction (d) of `.plan/phase7.md` requires both suspension lists to be re-derived, and
    # this is the entry that changes here: truth.json *names* the payments that never settled,
    # so the check becomes "the payments in no settlement are exactly the ones truth names"
    # rather than standing down. That is strictly stronger than the old form plus a suspension,
    # and it matters most on precisely the runs the suspension covered -- a batching loop or an
    # orphan draw that loses a payment truth did **not** name is still caught here.
    #
    # ``--settlement-report-late`` withholds some settlements' rows, so the *disk* file is
    # deliberately not a partition. The partition claim does not weaken -- it moves to
    # truth.json, which still declares every member. Sourcing it from there keeps this
    # check at full coverage instead of standing it down, which is the difference between
    # a relaxation with a successor and a hole.
    cited = set(counts) | (truth_members_all if late_on else set())
    missing = sorted(set(gross_by_payment) - cited)
    orphans = sorted(truth.unsettled_payment_ids)
    if missing != orphans:
        where = (
            "settlement_items.csv (plus truth.json for the withheld settlements)"
            if late_on else "settlement_items.csv"
        )
        unexpected = sorted(set(missing) - set(orphans))
        claimed_but_settled = sorted(set(orphans) - set(missing))
        raise InvariantError(
            f"I2: the payments in no settlement are not the ones truth names as unsettled. "
            f"{len(missing)} in no settlement ({missing[:5]}) against {len(orphans)} named "
            f"in truth ({orphans[:5]}); unsettled but unclaimed by truth: "
            f"{unexpected[:5]}; named by truth but settled anyway: "
            f"{claimed_but_settled[:5]} -- {where} plus truth's orphan list must partition "
            f"payments.csv, or no gross sum over a settlement means what it claims"
        )

    # --- I17: an orphan is indistinguishable *on disk*, not just in memory ---
    # `.plan/phase7.md` decision 3, re-checked after the round trip. ``check_orphans`` asserts
    # this on the in-memory story, where the CSV writer cannot yet have touched it; this pass
    # reads the file a matcher reads, which is the copy that would actually leak. **The columns
    # were already parsed here and never consulted** -- ``read_csv`` builds every row against the
    # frozen header -- so the gap was one dict lookup wide, in the assertion the whole flag's
    # honesty rests on. Same reasoning as I5 re-validating int money after the write: this file
    # exists because in-memory agreement is not evidence about the artefact.
    #
    # Compared against the settled population rather than a hard-coded ``"captured"``/``"INR"``,
    # matching the in-memory version: the claim is indistinguishability, so it must keep holding
    # when ``--fx`` legitimately starts moving ``currency``.
    #
    # Measured in `.plan/probe_phase7_i17d.py`, which mutates a written dataset one cell at a
    # time. Both leaks fire here; the third case is the one that had to be run: a **settled**
    # payment given a distinct status *passes*. The subset direction is deliberate -- what
    # identifies an orphan is a value only orphans carry, so widening the settled set is not a
    # leak, and a check that fired on it would be a column-uniformity assertion wearing I17's
    # name and would fail the day ``--fx`` mixes currencies legitimately.
    #
    # One interaction a later phase should expect: if ``--fx`` ever gives a non-INR currency to
    # *only* orphans, this fires -- and correctly, because in that dataset the currency column
    # really does identify the payments that never settled. It is a constraint on how the two
    # flags may be drawn together, not a defect in this check.
    #
    # Vacuous when nothing is orphaned, which is correct rather than lazy: with an empty orphan
    # list there is no leak to find, and a run naming orphans while ``--unsettled`` is off is
    # already refused by I4's cell-zero gate on ``unsettled_cells``.
    orphan_set = set(orphans)
    for column in ("status", "currency"):
        orphan_values = {r[column] for r in payments if r["payment_id"] in orphan_set}
        settled_values = {r[column] for r in payments if r["payment_id"] not in orphan_set}
        leaked = sorted(orphan_values - settled_values)
        if leaked:
            raise InvariantError(
                f"I17: on disk, {column} distinguishes a never-settled payment: {leaked} "
                f"against {sorted(settled_values)} on the settled rows -- decision 3: a value "
                f"carried by exactly the payments that never settled publishes the answer key "
                f"in a column, and a matcher filtering on it would score perfect coverage "
                f"while demonstrating nothing"
            )

    # --- I4: the money adds up, per settlement ------------------------------
    # This used to be an inline "net equals the member gross, while --fees is off" check
    # nested under clean mode. Phase 4 strengthens it to the full subtraction and runs it
    # unconditionally, which at zero deductions is the old equality unchanged.
    #
    # Membership normally comes from settlement_items.csv rather than truth.json, because this
    # pass must reach its verdict from the files a judge gets. ``--settlement-report-late`` is
    # the one flag that makes a missing member row legitimate, and **this is the deliberate
    # relaxation that comment predicted** -- written as a relaxation with a successor rather
    # than a suspension: ``effective_members`` falls back to truth's declaration for exactly
    # the withheld settlements, so the per-settlement arithmetic still runs on **every**
    # settlement. Nothing is skipped; one input is sourced from a different file, and the run
    # says which.
    orphan_settlements = sorted(set(net_by_settlement) - set(members))
    effective_members = dict(members)
    if late_on:
        recovered = {sid: truth_members.get(sid, ()) for sid in orphan_settlements}
        unrecoverable = sorted(sid for sid, pids in recovered.items() if not pids)
        if unrecoverable:
            raise InvariantError(
                f"I2: {len(unrecoverable)} settlement(s) have no members in "
                f"settlement_items.csv *and* none in truth.json: {unrecoverable[:5]}. "
                f"--settlement-report-late withholds a declaration from the matcher's files; "
                f"it must never remove the membership from the answer key, or a searched "
                f"payment set could not be graded against anything."
            )
        # Partial, never total (decision 4). All-withheld would make the tier distribution a
        # swap rather than a mix, and gate 12 would then pass on a run where Tier 1 is dead.
        if not orphan_settlements:
            raise InvariantError(
                "I2: --settlement-report-late is on but every settlement declares its "
                "members, so nothing was withheld and the search is unexercised"
            )
        if len(orphan_settlements) == len(net_by_settlement):
            raise InvariantError(
                f"I2: every one of the {len(net_by_settlement)} settlements had its "
                f"membership withheld. Withholding must be partial (decision 4): a total "
                f"withholding turns the tier mix into a swap and lets a Tier 1 regression "
                f"hide behind a Tier 2 success."
            )
        effective_members.update(recovered)
        skipped["I2.membership_on_disk"] = ["settlement_report_late"]
    elif orphan_settlements:
        raise InvariantError(
            f"I2: settlements with no rows in settlement_items.csv: "
            f"{orphan_settlements[:5]} -- their net cannot be checked against any gross"
        )
    check_settlement_arithmetic(
        [
            (sid, sum(gross_by_payment[p] for p in effective_members[sid]),
             net_by_settlement[sid], *deductions[sid], refunds_of_settlement[sid])
            for sid in sorted(net_by_settlement)
        ]
    )

    # --- I7: the leak check, on the bytes actually written ------------------
    check_no_leak([(r["row_id"], amount_by_credit[r["row_id"]], r["narration"]) for r in bank])

    # --- I6: every truth reference resolves --------------------------------
    for c in truth.credits:
        if c.credit_id not in amount_by_credit:
            raise InvariantError(f"I6: truth cites credit {c.credit_id}, absent from the bank file")
        for sid in c.settlement_ids:
            if sid not in net_by_settlement:
                raise InvariantError(f"I6: truth {c.credit_id} cites unknown {sid}")
        for pid in c.payment_ids:
            if pid not in gross_by_payment:
                raise InvariantError(f"I6: truth {c.credit_id} cites unknown {pid}")
        if c.decomposition.expected_credit_paise != amount_by_credit[c.credit_id]:
            raise InvariantError(
                f"I4: truth expects {c.decomposition.expected_credit_paise} for "
                f"{c.credit_id} but the bank file says {amount_by_credit[c.credit_id]}"
            )
    # **Strengthened from a count into an equality on identities, Phase 7 step 4** -- the same
    # treatment ``--reserve`` got for Strategy D in Phase 6, and preferred to the suspension this
    # entry used to carry. Every bank row must be accounted for by the answer key as *either*
    # gateway income or a named non-gateway row, and every credit truth declares must be in the
    # file. A count equality would let a missing credit and an unnamed noise row cancel out;
    # identities cannot cancel.
    #
    # This is why ``--noise-rows`` needed no suspension on this side: the flag adds rows the
    # answer key **names**, so the check gains a term rather than losing coverage. Left as a
    # suspension it would have stood down on every noisy run -- at exactly the moment a bank row
    # can go missing from truth -- which is the footgun ``DISK_SUSPENDED_BY``'s docstring warns
    # about and which the orphan terms above already declined.
    if on := _disk_suspended(flags, "I6.truth_covers_bank"):
        skipped["I6.truth_covers_bank"] = on
    else:
        bank_ids = {r["row_id"] for r in bank}
        accounted = {c.credit_id for c in truth.credits} | set(truth.non_gateway_credit_ids)
        unexplained = sorted(bank_ids - accounted)
        missing = sorted(accounted - bank_ids)
        if unexplained or missing:
            raise InvariantError(
                f"I6: the answer key and the bank file disagree about which rows exist -- "
                f"{len(unexplained)} bank row(s) have no truth entry and are not named "
                f"non-gateway ({unexplained[:5]}), and {len(missing)} row(s) truth accounts "
                f"for are absent from the file ({missing[:5]}). Every bank row needs an "
                f"answer-key entry: a credit, or a named non-gateway row."
            )

    # --- Section 2: the leak audit (the honest gate 4) ---------------------
    audit = leak_audit(payments, settlements, bank, items, truth)

    # --- I8b: within-block positional alignment ----------------------------
    captured = {r["payment_id"]: parse_captured_at(r["captured_at"], r["payment_id"])
                for r in payments}
    truth_payment_of = {c.credit_id: c.payment_ids[0] for c in truth.credits
                        if len(c.payment_ids) == 1}
    p_by_date: dict[date, list[str]] = defaultdict(list)
    for r in sorted(payments, key=lambda r: r["payment_id"]):
        p_by_date[captured[r["payment_id"]].date()].append(r["payment_id"])
    c_by_date: dict[date, list[str]] = defaultdict(list)
    for r in bank:
        pid = truth_payment_of.get(r["row_id"])
        if pid:
            c_by_date[parse_date(r["value_date"], r["row_id"])].append(pid)
    aligned, population = check_within_block_alignment(p_by_date, c_by_date)

    # --- I14: batch membership carries no id-adjacency information -----------
    # Re-run here against ``settlement_items.csv`` rather than trusted from the in-memory
    # pass: this file is the membership declaration a judge reads, and the shuffle in
    # ``story._group_into_batches`` is what stops consecutive-id enumeration from standing in
    # for a subset search once Phase 5 step 5 withholds it.
    batch_numbers = [
        sorted(int(pid.removeprefix("pay_")) for pid in pids)
        for pids in members.values()
        if len(pids) > 1
    ]
    consecutive, batch_population = check_batch_adjacency(batch_numbers)

    report: dict[str, object] = {
        "files_verified": len(CSV_FILES),
        "consecutive_id_batches": f"{consecutive}/{batch_population}",
        # Every check this pass skipped and the flag that excused it. Collected above and
        # reported here -- a skip that is recorded and never surfaced is the same silence that
        # made the ``clean_mode`` gate dangerous in the first place.
        "checks_skipped": {k: list(v) for k, v in sorted(skipped.items())},
        "payments": len(payments),
        "settlements": len(settlements),
        "settlement_items": len(items),
        "bank_rows": len(bank),
        "refund_rows": len(refunds),
        "gross_paise_total": sum(gross_by_payment.values()),
        "within_block_aligned": f"{aligned}/{population}",
        **audit,
    }
    if verbose:
        _print_report(report, truth)
    return report


def leak_audit(
    payments: list[dict[str, str]],
    settlements: list[dict[str, str]],
    bank: list[dict[str, str]],
    items: list[dict[str, str]],
    truth: object,
) -> dict[str, object]:
    """Measure four resolution strategies and assert the structural ones fail.

    Reads only ``data/`` for the strategies themselves; truth is used solely to
    score them, which is exactly the split the real scorer uses.
    """
    truth_pid = {c.credit_id: c.payment_ids[0] for c in truth.credits  # type: ignore[attr-defined]
                 if len(c.payment_ids) == 1}
    truth_sid = {c.credit_id: c.settlement_ids[0] for c in truth.credits  # type: ignore[attr-defined]
                 if len(c.settlement_ids) == 1}
    n = len(bank)
    if not n:
        return {}

    #: Phase 7 step 4. Rows that are not gateway money at all (``--noise-rows``), named by truth.
    #:
    #: **Every threshold below is against ``gateway_n``, not ``n``, and that is a correction
    #: rather than a refinement.** A noise row has no ``credits`` entry, so ``truth_pid`` and
    #: ``truth_sid`` both return ``None`` for it and *no* strategy can ever score it. Left
    #: comparing against ``n``, the three structural refusals -- "position must not resolve the
    #: dataset", "numbering must not resolve the dataset" -- become **unreachable** the moment a
    #: single noise row exists: ``zip_hits == n`` cannot hold when some rows are unscoreable, so
    #: the assertions stand down invisibly on every noisy run. That is the footgun this file's
    #: own docstring warns about, and it was not in either suspension list because it is not a
    #: suspension -- it is a denominator quietly becoming the wrong number.
    #:
    #: Found by measurement (`.plan/probe_phase7_noise_suspensions.py`), not by prediction: the
    #: probe forced the two predicted checks to run and a *third* check failed first.
    noise_ids = set(truth.non_gateway_credit_ids)  # type: ignore[attr-defined]
    gateway_n = n - len(noise_ids)
    if not gateway_n:
        raise InvariantError(
            f"I6: all {n} bank rows are named non-gateway -- a run with no gateway income at "
            f"all has nothing to reconcile, so no resolution strategy means anything"
        )

    # Strategy A -- by row position: payments[i] <-> bank[i].
    pay_order = [r["payment_id"] for r in payments]
    zip_hits = sum(
        1 for i, r in enumerate(bank)
        if i < len(pay_order) and truth_pid.get(r["row_id"]) == pay_order[i]
    )

    # Strategy B -- by ID numbering: pay_NNNN -> C_NNNN, and pay_NNNN -> setl_NNNN.
    number_hits = sum(
        1 for r in bank
        if truth_pid.get(r["row_id"]) == ids.payment_id(int(r["row_id"].lstrip("C")))
    )
    setl_number_hits = sum(
        1 for r in items
        if r["settlement_id"] == ids.settlement_id(int(r["payment_id"].removeprefix("pay_")))
    )

    # Strategy C -- by narration: does the bank row name its own source?
    narration_hits = sum(1 for r in bank if ids.leaked_identifiers(r["narration"]))

    # Strategy D -- by date + amount, the legitimate signal. Unique key -> match.
    key_of_settlement: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in settlements:
        key_of_settlement[(r["settled_on"], r["net_paise"])].append(r["settlement_id"])
    arithmetic_hits = 0
    #: Which rows the legitimate join failed to resolve, not merely how many. Phase 6 step 7
    #: needs the identities: ``--reserve`` makes this strategy fail on a known, nameable set of
    #: rows, and asserting *which* ones is what turns a suspension into a strengthening.
    arithmetic_missed: list[str] = []
    for r in bank:
        # A noise row is not gateway income, so no strategy can resolve it and its failure to
        # resolve says nothing about the signal. Skipped rather than counted as a miss, because
        # ``arithmetic_missed`` carries **identities** into the reserved-row equality below --
        # counted, every noisy run would report the noise rows as unexplained misses and that
        # equality could never hold again.
        if r["row_id"] in noise_ids:
            continue
        candidates = key_of_settlement.get((r["value_date"], r["amount_paise"]), [])
        if len(candidates) == 1 and truth_sid.get(r["row_id"]) == candidates[0]:
            arithmetic_hits += 1
        else:
            arithmetic_missed.append(r["row_id"])

    # The three structural strategies must not resolve the dataset.
    if narration_hits:
        raise InvariantError(
            f"I7: {narration_hits} bank narrations name their own source -- the "
            f"answer was generated into the input"
        )
    # ``gateway_n`` rather than ``n`` in all three refusals below -- see its definition. An
    # unscoreable row in the denominator makes an ``== n`` test unsatisfiable, which turns an
    # anti-cheat assertion into a no-op instead of a failure.
    if zip_hits == gateway_n:
        raise InvariantError(
            "I8b: row position alone resolves every credit -- a matcher that zips "
            "the files together would score 100% without doing any work"
        )
    # ``setl_number_hits`` is counted over ``items`` rather than over bank rows, so its
    # denominator is ``len(items)`` and never ``n``. That was pre-existing looseness this step
    # had to resolve rather than inherit: comparing it to a bank-row count is right only while
    # the two happen to be equal (they are not under ``--batching``), and switching it to
    # ``gateway_n`` would have made a *false* refusal reachable. Fixed to the population the
    # strategy actually enumerates.
    if number_hits == gateway_n or (items and setl_number_hits == len(items)):
        raise InvariantError(
            f"I8a: ID numbering alone resolves the dataset "
            f"(pay->credit {number_hits}/{gateway_n}, "
            f"pay->settlement {setl_number_hits}/{len(items)}) "
            f"-- the numbering is the answer key"
        )
    # ... and the legitimate one must resolve it completely, or row 1 of the mess dial is not
    # the 100% baseline the whole ramp is measured against. Gated per flag rather than on
    # ``clean_mode``, so ``--fees`` and ``--batching`` are held to it too.
    audit_flags = {k: bool(v) for k, v in truth.flags.items()}  # type: ignore[attr-defined]
    #: Phase 6 step 7. ``--reserve`` genuinely breaks this strategy -- a reserved credit is
    #: short of its settlement's net, so the exact ``(date, amount)`` key finds **zero**
    #: candidates for it -- and that break is the mess itself rather than a defect. Measured on
    #: the first ``--reserve`` run: 55/60 at seed 42, n=60, where 60 x RESERVE_SHARE = 5.
    #:
    #: **So this is the third check the flag touches, and none of the three were the two
    #: ``SUSPENDED_BY`` predicted.** The cheap response is a fourth ``DISK_SUSPENDED_BY`` entry.
    #: It is refused here for the reason that file's own docstring gives: a check that stands
    #: down exactly when the thing it guards starts moving is the footgun. And suspension is
    #: strictly worse than available here, because the shortfall is *predictable* -- the rows
    #: this strategy loses are exactly the rows truth records a reserve against. So the check
    #: is **strengthened into an equality on identities** rather than relaxed into a count: it
    #: now catches a reserved row that resolves anyway (which would mean
    #: ``story._separate_reserved_amounts`` failed to clear some net, the silent-wrong-match
    #: hazard I16 also guards) *and* an unreserved row that stopped resolving, which a
    #: suspension would have hidden completely.
    reserved_ids = {
        c.credit_id for c in truth.credits  # type: ignore[attr-defined]
        if c.decomposition.reserve_paise
    }
    if suspended_by := _disk_suspended(audit_flags, "I3.date_amount_resolves"):
        pass  # named in the returned report, never silently dropped
    elif reserved_ids:
        if set(arithmetic_missed) != reserved_ids:
            only_missed = sorted(set(arithmetic_missed) - reserved_ids)
            still_resolving = sorted(reserved_ids - set(arithmetic_missed))
            raise InvariantError(
                f"I3: date+amount resolves {arithmetic_hits}/{n} credits, but the rows it "
                f"fails on are not exactly the {len(reserved_ids)} reserved one(s). "
                f"Unreserved rows that stopped resolving: {only_missed[:5] or 'none'}. "
                f"Reserved rows that resolved anyway: {still_resolving[:5] or 'none'} -- a "
                f"reserved credit whose short amount still hits a settlement's net gives the "
                f"matcher exactly one candidate, the WRONG one, with arithmetic that closes "
                f"perfectly; see story._separate_reserved_amounts and I16"
            )
    elif arithmetic_hits != gateway_n:
        raise InvariantError(
            f"I3: date+amount resolves only {arithmetic_hits}/{gateway_n} gateway credits "
            f"while every flag that legitimately breaks that join is off -- the mess dial "
            f"requires 100% here, so the Phase 3 matcher cannot hit its target on this data"
        )
    return {
        "leak_audit": {
            # Denominators match the populations the assertions above test, so a reader
            # comparing a reported figure against a threshold is comparing like with like.
            # Identical to the old ``/{n}`` on every run without ``--noise-rows``, since
            # ``gateway_n == n`` there -- the figures recorded in ``DISK_SUSPENDED_BY``'s
            # comment block (clean 200/200, batching 120/120) are unchanged.
            "by_row_position": f"{zip_hits}/{gateway_n}",
            "by_id_numbering": f"{number_hits}/{gateway_n}",
            "by_settlement_numbering": f"{setl_number_hits}/{len(items)}",
            "by_narration": f"{narration_hits}/{n}",
            "by_date_and_amount": f"{arithmetic_hits}/{gateway_n}",
            # Present so a reader can tell "resolves everything" from "was not required to".
            "date_amount_not_required_by": suspended_by,
        }
    }


def _print_report(report: dict[str, object], truth: object) -> None:
    print(f"verified {report['files_verified']} files re-read from disk")
    print(
        f"  rows: {report['payments']} payments, {report['settlements']} settlements, "
        f"{report['settlement_items']} items, {report['bank_rows']} bank, "
        f"{report['refund_rows']} refunds"
    )
    audit = report.get("leak_audit", {})
    if audit:
        print("\n  leak audit -- how much of the answer is readable from data/ alone:")
        labels = {
            "by_row_position": "row position",
            "by_id_numbering": "ID numbering (pay->credit)",
            "by_settlement_numbering": "ID numbering (pay->settlement)",
            "by_narration": "bank narration",
            "by_date_and_amount": "date + amount  <- the legitimate signal",
        }
        for key, label in labels.items():
            print(f"    {label:<34} {audit[key]}")  # type: ignore[index]
        print(
            "\n  The gap between the last line and the rest is the proof: resolving\n"
            "  this dataset requires the arithmetic join, not a structural shortcut."
        )
    print(f"\n  within-date-block positional alignment: {report['within_block_aligned']}")
    print("\nall invariants pass on the written files")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Re-read the generated files and re-check every invariant."
    )
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--truth", type=Path, default=Path("truth"))
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    try:
        verify(args.data, args.truth, verbose=not args.quiet)
    except InvariantError as e:
        print(f"VERIFICATION FAILED\n  {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
