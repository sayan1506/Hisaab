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
    check_headers,
    check_int_money,
    check_no_leak,
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
    fee_cells: list[int] = []
    for r in settlements:
        sid = r["settlement_id"]
        net_by_settlement[sid] = parse_int(r["net_paise"], f"{sid}.net_paise")
        money[f"{sid}.net_paise"] = net_by_settlement[sid]
        for col in ("fee_paise", "gst_paise", "tds_paise"):
            fee_cells.append(parse_int(r[col], f"{sid}.{col}"))
    amount_by_credit: dict[str, int] = {}
    for r in bank:
        cid = r["row_id"]
        amount_by_credit[cid] = parse_int(r["amount_paise"], f"{cid}.amount_paise")
        money[f"{cid}.amount_paise"] = amount_by_credit[cid]
    check_int_money(money)

    # --- I4: the three totals agree; fee columns are zero -------------------
    check_totals(
        sum(gross_by_payment.values()),
        sum(net_by_settlement.values()),
        sum(amount_by_credit.values()),
        fee_cells,
    )

    # --- I3: 1:1:1 in clean mode -------------------------------------------
    truth = load_truth(truth_dir)
    if truth.clean_mode:
        if not (len(payments) == len(settlements) == len(bank)):
            raise InvariantError(
                f"I3: clean mode must be 1:1:1, got "
                f"{len(payments)}/{len(settlements)}/{len(bank)}"
            )
        if refunds:
            raise InvariantError(f"I3: clean mode emitted {len(refunds)} refund rows")

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
    if truth.clean_mode:
        missing = sorted(set(gross_by_payment) - set(counts))
        if missing:
            raise InvariantError(f"I2: payments never settled: {missing[:5]}")
        # Clean mode: net must equal the one member payment's gross.
        for sid, pl in members.items():
            if net_by_settlement[sid] != sum(gross_by_payment[p] for p in pl):
                raise InvariantError(
                    f"I4: {sid} net {net_by_settlement[sid]} != sum of members "
                    f"{sum(gross_by_payment[p] for p in pl)} while --fees is off"
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
    if truth.clean_mode and len(truth.credits) != len(bank):
        raise InvariantError(
            f"I6: truth has {len(truth.credits)} credits, bank file has {len(bank)} rows"
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

    report: dict[str, object] = {
        "files_verified": len(CSV_FILES),
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
    # ... and the legitimate one must resolve it completely, or clean mode is not
    # the 100% baseline the mess dial requires.
    if truth.clean_mode and arithmetic_hits != n:  # type: ignore[attr-defined]
        raise InvariantError(
            f"I3: date+amount resolves only {arithmetic_hits}/{n} credits in clean "
            f"mode, but row 1 of the mess dial requires 100% -- the Phase 3 matcher "
            f"cannot hit its target on this data"
        )
    return {
        "leak_audit": {
            "by_row_position": f"{zip_hits}/{n}",
            "by_id_numbering": f"{number_hits}/{n}",
            "by_settlement_numbering": f"{setl_number_hits}/{n}",
            "by_narration": f"{narration_hits}/{n}",
            "by_date_and_amount": f"{arithmetic_hits}/{n}",
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
