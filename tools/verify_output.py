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
    "I3.one_to_one_to_one":     ("batching", "noise_rows", "unsettled"),
    "I3.no_refunds":            ("netted_refunds",),
    # A payment in no settlement is --unsettled's business and nothing else's.
    "I2.every_payment_settled": ("unsettled",),
    # A noise row is a bank row with no truth entry, so the two counts diverge.
    "I6.truth_covers_bank":     ("noise_rows",),
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
    late_on = flags.get("settlement_report_late", False)
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
    check_totals(
        sum(gross_by_payment.values()),
        sum(net_by_settlement.values()),
        sum(amount_by_credit.values()),
        fee_cells,
        tds_cells,
        fees_on=fees_on,
        tds_on=tds_on,
    )

    # --- I3: cardinality, per flag rather than per clean_mode ----------------
    if on := _disk_suspended(flags, "I3.one_to_one_to_one"):
        skipped["I3.one_to_one_to_one"] = on
    elif not (len(payments) == len(settlements) == len(bank)):
        raise InvariantError(
            f"I3: 1:1:1 must hold while every flag that changes cardinality is off, got "
            f"{len(payments)}/{len(settlements)}/{len(bank)}"
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
    if on := _disk_suspended(flags, "I2.every_payment_settled"):
        skipped["I2.every_payment_settled"] = on
    else:
        # ``--settlement-report-late`` withholds some settlements' rows, so the *disk* file is
        # deliberately not a partition. The partition claim does not weaken -- it moves to
        # truth.json, which still declares every member. Sourcing it from there keeps this
        # check at full coverage instead of standing it down, which is the difference between
        # a relaxation with a successor and a hole.
        cited = set(counts) | (truth_members_all if late_on else set())
        missing = sorted(set(gross_by_payment) - cited)
        if missing:
            where = (
                "settlement_items.csv (plus truth.json for the withheld settlements)"
                if late_on else "settlement_items.csv"
            )
            raise InvariantError(
                f"I2: {len(missing)} payment(s) are in no settlement: {missing[:5]} -- "
                f"{where} is not a partition of payments.csv, so no gross sum "
                f"over a settlement means what it claims"
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
             net_by_settlement[sid], *deductions[sid])
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
    if on := _disk_suspended(flags, "I6.truth_covers_bank"):
        skipped["I6.truth_covers_bank"] = on
    elif len(truth.credits) != len(bank):
        raise InvariantError(
            f"I6: truth has {len(truth.credits)} credits, bank file has {len(bank)} rows -- "
            f"every bank row needs an answer-key entry unless --noise-rows is on"
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
    for r in bank:
        candidates = key_of_settlement.get((r["value_date"], r["amount_paise"]), [])
        if len(candidates) == 1 and truth_sid.get(r["row_id"]) == candidates[0]:
            arithmetic_hits += 1

    # The three structural strategies must not resolve the dataset.
    if narration_hits:
        raise InvariantError(
            f"I7: {narration_hits} bank narrations name their own source -- the "
            f"answer was generated into the input"
        )
    if zip_hits == n:
        raise InvariantError(
            "I8b: row position alone resolves every credit -- a matcher that zips "
            "the files together would score 100% without doing any work"
        )
    if number_hits == n or setl_number_hits == n:
        raise InvariantError(
            f"I8a: ID numbering alone resolves the dataset "
            f"(pay->credit {number_hits}/{n}, pay->settlement {setl_number_hits}/{n}) "
            f"-- the numbering is the answer key"
        )
    # ... and the legitimate one must resolve it completely, or row 1 of the mess dial is not
    # the 100% baseline the whole ramp is measured against. Gated per flag rather than on
    # ``clean_mode``, so ``--fees`` and ``--batching`` are held to it too.
    audit_flags = {k: bool(v) for k, v in truth.flags.items()}  # type: ignore[attr-defined]
    if suspended_by := _disk_suspended(audit_flags, "I3.date_amount_resolves"):
        pass  # named in the returned report, never silently dropped
    elif arithmetic_hits != n:
        raise InvariantError(
            f"I3: date+amount resolves only {arithmetic_hits}/{n} credits while every flag "
            f"that legitimately breaks that join is off -- the mess dial requires 100% here, "
            f"so the Phase 3 matcher cannot hit its target on this data"
        )
    return {
        "leak_audit": {
            "by_row_position": f"{zip_hits}/{n}",
            "by_id_numbering": f"{number_hits}/{n}",
            "by_settlement_numbering": f"{setl_number_hits}/{n}",
            "by_narration": f"{narration_hits}/{n}",
            "by_date_and_amount": f"{arithmetic_hits}/{n}",
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
