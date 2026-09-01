"""The one undesigned mess type the generator refuses to produce: a cross-date batch.

    python tools/splice_cross_date_batch.py --data out/holdout-seed99/data \\
        --out out/holdout-seed99/data-spliced --matches out/holdout-seed99/out/matches.json

Phase 12's amendment names this failure mode deliberately: a settlement whose declared
membership spans two different ``settled_on`` dates. The generator will not build one --
``story._group_into_batches`` partitions candidate payments by ``settled_on`` before
choosing a batch, and invariant **I11** (``hisaab/generator/invariants.py:1785-1821``)
independently re-derives every settlement's and every credit's expected date from the
delay/lag model, so a batch that straddled two dates fails before it reaches disk (see
``story.py``'s own comment on ``_group_into_batches``). Note: this is I11, not I14
(``check_batch_adjacency``) -- I14 checks whether a batch's payment ids form a consecutive
run, which is a different property with nothing to do with dates; the Phase 12 explainer and
plan mislabelled this guard and this module cites the corrected name.

This script produces the shape the generator will not, by hand-editing three already-valid
CSVs a normal ``--batching`` run already wrote to disk. **It never imports
``hisaab.generator.story`` or ``hisaab.generator.invariants.check_story``**, and it never
reads or writes anything under a ``truth/`` directory -- both are hard constraints, not
defaults that happen to hold. Every check it applies before writing (referential integrity,
uniqueness) is the same set ``hisaab/matcher/load.py`` applies at load time, run here first so
a mistake in this script surfaces with a clear message rather than three steps later as a
``LoadError``.

**Why this needs a ``matches.json`` and cannot infer the settlement-to-credit link itself.**
``bank_statement.csv`` carries no ``settlement_id`` column -- recovering which bank row
belongs to which settlement *is* the task the rest of this project measures. This script does
not attempt a shortcut version of that join: it only merges two settlements that a real
matcher run already resolved, each to exactly one settlement (never an already-batched
credit, and never anything the matcher could not commit to). A "resolvable by inspection"
version of this script -- inferring the link by exact (date, amount) equality against
``settlements.csv``'s declared net -- would silently break on the same flags this script is
meant to run against: ``--reserve`` and ``--fx`` both make the credit disagree with the
declared net by design (ASSUMPTIONS.md #22b, #22h), and ``--tds``/``--netted-refunds`` net
further deductions in before the money reaches the bank. So ``--matches`` is required, not a
convenience.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hisaab.common.bizdays import BusinessCalendar  # noqa: E402
from hisaab.generator.model import (  # noqa: E402
    BANK_HEADER,
    PAYMENTS_HEADER,
    REFUNDS_HEADER,
    SETTLEMENT_ITEMS_HEADER,
    SETTLEMENTS_HEADER,
)

#: The five files a run writes to ``--out``. Payments and refunds pass through unchanged --
#: nothing about a payment or a refund moves when two settlements are merged after the fact.
DATA_FILES: tuple[str, ...] = (
    "payments.csv", "settlements.csv", "settlement_items.csv",
    "bank_statement.csv", "refunds.csv",
)


class SpliceError(Exception):
    """No valid pair could be found, or the merged output failed self-verification."""


def _read_csv(path: Path, expected_header: tuple[str, ...]) -> list[dict[str, str]]:
    """Read one CSV with the stdlib ``csv`` module directly -- no round-trip through
    ``hisaab.generator.emit`` or ``hisaab.matcher.load``, so a bug in either cannot hide
    behind a symmetric bug here."""
    if not path.exists():
        raise SpliceError(f"{path} not found -- point --data at a run's own data/ directory")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = tuple(next(reader))
        except StopIteration:
            raise SpliceError(f"{path.name} is completely empty") from None
        if header != expected_header:
            raise SpliceError(
                f"{path.name} header drift.\n  expected {expected_header}\n  found    {header}"
            )
        return [dict(zip(header, row)) for row in reader if row]


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    """``newline=""`` plus an explicit ``lineterminator="\\n"`` -- emit.py's trap 5, repeated
    here because this script writes CSVs of its own rather than calling emit.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def _load_resolved_single_settlement_credits(matches_path: Path) -> dict[str, str]:
    """settlement_id -> its one resolved bank row_id, for verdicts naming exactly one
    settlement.

    Restricted to a single named settlement per verdict so both halves of the pair this
    script merges are themselves simple, unbatched credits -- an already-batched settlement
    is skipped rather than folded into a bigger one, which would make "two settlements, one
    credit each" a false description of what actually got merged.
    """
    if not matches_path.exists():
        raise SpliceError(f"{matches_path} not found -- run the matcher against --data first")
    doc = json.loads(matches_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for v in doc.get("verdicts", []):
        if v.get("outcome") != "RESOLVED":
            continue
        settlement_ids = v.get("settlement_ids") or []
        if len(settlement_ids) != 1:
            continue
        out[settlement_ids[0]] = v["credit_id"]
    return out


def _pick_pair(
    settlements: list[dict[str, str]], settlement_to_credit: dict[str, str]
) -> tuple[str, str]:
    """The earliest (settled_on, then settlement_id) pair exactly one business day apart.

    Both members must already carry a single resolved credit (via ``settlement_to_credit``)
    -- a settlement the matcher never simply resolved is skipped rather than guessed into
    the pair.
    """
    cal = BusinessCalendar()
    by_date: dict[date, list[str]] = {}
    for row in settlements:
        sid = row["settlement_id"]
        if sid not in settlement_to_credit:
            continue
        by_date.setdefault(date.fromisoformat(row["settled_on"]), []).append(sid)

    dates = sorted(by_date)
    for i, d in enumerate(dates):
        for d2 in dates[i + 1:]:
            if cal.business_days_between(d, d2) != 1:
                continue
            return sorted(by_date[d])[0], sorted(by_date[d2])[0]
    raise SpliceError(
        "no pair of resolved, unbatched settlements sits exactly one business day apart in "
        "this run -- cannot construct a cross-date batch from it"
    )


def _self_verify(
    *,
    payments: list[dict[str, str]],
    new_settlements_rows: list[tuple[str, ...]],
    new_items_rows: list[tuple[str, str]],
    new_bank_rows: list[tuple[str, ...]],
) -> None:
    """The same referential checks ``hisaab/matcher/load.py:308-326`` applies at load time,
    run here first so a mistake in this script is caught by name rather than surfaced three
    steps later as a ``LoadError``."""
    known_payments = {r["payment_id"] for r in payments}
    known_settlements = [row[0] for row in new_settlements_rows]
    known_row_ids = [row[0] for row in new_bank_rows]

    if len(set(known_settlements)) != len(known_settlements):
        raise SpliceError("duplicate settlement_id in the rewritten settlements.csv")
    if len(set(known_row_ids)) != len(known_row_ids):
        raise SpliceError("duplicate row_id in the rewritten bank_statement.csv")

    known_settlements_set = set(known_settlements)
    seen_pairs: set[tuple[str, str]] = set()
    for sid, pid in new_items_rows:
        if sid not in known_settlements_set:
            raise SpliceError(f"settlement_items.csv cites unknown settlement {sid}")
        if pid not in known_payments:
            raise SpliceError(f"settlement_items.csv cites unknown payment {pid}")
        if (sid, pid) in seen_pairs:
            raise SpliceError(f"settlement_items.csv repeats the pair ({sid}, {pid})")
        seen_pairs.add((sid, pid))


def splice(
    data_dir: Path, out_dir: Path, matches_path: Path, *, quiet: bool = False
) -> dict[str, object]:
    data_dir, out_dir, matches_path = Path(data_dir), Path(out_dir), Path(matches_path)

    if out_dir.resolve() == data_dir.resolve():
        raise SpliceError("--out must be a different directory from --data")
    for label, p in (("--data", data_dir), ("--out", out_dir)):
        if "truth" in p.resolve().parts:
            raise SpliceError(
                f"{label} ({p}) sits under a 'truth' directory -- this script must not read "
                f"or write truth of any kind"
            )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SpliceError(f"{out_dir} already exists and is not empty -- refusing to overwrite")

    payments = _read_csv(data_dir / "payments.csv", PAYMENTS_HEADER)
    settlements = _read_csv(data_dir / "settlements.csv", SETTLEMENTS_HEADER)
    items = _read_csv(data_dir / "settlement_items.csv", SETTLEMENT_ITEMS_HEADER)
    bank = _read_csv(data_dir / "bank_statement.csv", BANK_HEADER)
    refunds = _read_csv(data_dir / "refunds.csv", REFUNDS_HEADER)

    settlement_to_credit = _load_resolved_single_settlement_credits(matches_path)
    a_id, b_id = _pick_pair(settlements, settlement_to_credit)
    a_credit_id, b_credit_id = settlement_to_credit[a_id], settlement_to_credit[b_id]

    settlements_by_id = {r["settlement_id"]: r for r in settlements}
    bank_by_id = {r["row_id"]: r for r in bank}
    a_settle, b_settle = settlements_by_id[a_id], settlements_by_id[b_id]
    a_bank, b_bank = bank_by_id[a_credit_id], bank_by_id[b_credit_id]

    a_payments = [r["payment_id"] for r in items if r["settlement_id"] == a_id]
    b_payments = [r["payment_id"] for r in items if r["settlement_id"] == b_id]
    if set(a_payments) & set(b_payments):
        raise SpliceError(f"{a_id} and {b_id} already share a payment -- not a valid pair")

    merged_net = int(a_settle["net_paise"]) + int(b_settle["net_paise"])
    merged_amount = int(a_bank["amount_paise"]) + int(b_bank["amount_paise"])
    merged_payments = sorted(set(a_payments) | set(b_payments))

    # --- settlements.csv: drop b's row, replace a's net with the summed one -------------
    new_settlements: list[tuple[str, ...]] = []
    for r in settlements:
        if r["settlement_id"] == b_id:
            continue
        if r["settlement_id"] == a_id:
            r = {**r, "net_paise": str(merged_net)}
        new_settlements.append(tuple(r[col] for col in SETTLEMENTS_HEADER))

    # --- settlement_items.csv: drop both settlements' rows, add merged rows under a_id --
    new_items: list[tuple[str, str]] = [
        (r["settlement_id"], r["payment_id"])
        for r in items
        if r["settlement_id"] not in (a_id, b_id)
    ]
    new_items.extend((a_id, pid) for pid in merged_payments)
    new_items.sort()

    # --- bank_statement.csv: drop b's credit row, replace a's amount with the sum -------
    new_bank: list[tuple[str, ...]] = []
    for r in bank:
        if r["row_id"] == b_credit_id:
            continue
        if r["row_id"] == a_credit_id:
            r = {**r, "amount_paise": str(merged_amount)}
        new_bank.append(tuple(r[col] for col in BANK_HEADER))

    _self_verify(
        payments=payments, new_settlements_rows=new_settlements,
        new_items_rows=new_items, new_bank_rows=new_bank,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "payments.csv", PAYMENTS_HEADER,
               [tuple(r[col] for col in PAYMENTS_HEADER) for r in payments])
    _write_csv(out_dir / "settlements.csv", SETTLEMENTS_HEADER, new_settlements)
    _write_csv(out_dir / "settlement_items.csv", SETTLEMENT_ITEMS_HEADER, new_items)
    _write_csv(out_dir / "bank_statement.csv", BANK_HEADER, new_bank)
    _write_csv(out_dir / "refunds.csv", REFUNDS_HEADER,
               [tuple(r[col] for col in REFUNDS_HEADER) for r in refunds])

    report = {
        "merged_settlement_id": a_id,
        "dropped_settlement_id": b_id,
        "settled_on": [a_settle["settled_on"], b_settle["settled_on"]],
        "merged_credit_id": a_credit_id,
        "dropped_credit_id": b_credit_id,
        "merged_amount_paise": merged_amount,
        "merged_payment_count": len(merged_payments),
        "out_dir": str(out_dir),
    }
    if not quiet:
        print(
            f"merged {a_id} (settled {a_settle['settled_on']}) and {b_id} "
            f"(settled {b_settle['settled_on']}) into {a_id}: credit {a_credit_id} now "
            f"{merged_amount}p over {len(merged_payments)} payment(s), {b_credit_id} dropped "
            f"-> {out_dir}"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Merge two adjacent-date settlements into one cross-date batch -- the "
                     "one undesigned mess type the generator refuses to produce.",
        epilog="Never touches truth/. See the module docstring.",
    )
    p.add_argument("--data", type=Path, required=True,
                    help="a --batching run's data/ directory")
    p.add_argument("--out", type=Path, required=True,
                    help="new sibling directory for the spliced files (must not exist, or "
                         "must be empty)")
    p.add_argument("--matches", type=Path, required=True,
                    help="that same run's matches.json, used to name which bank row belongs "
                         "to which settlement -- see the module docstring for why this is "
                         "required rather than inferred")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    try:
        splice(args.data, args.out, args.matches, quiet=args.quiet)
    except SpliceError as e:
        print(f"SPLICE FAILED\n  {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
