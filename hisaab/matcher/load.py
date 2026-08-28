"""Read ``data/``, refuse surprises.

Five CSVs in, five typed row lists out. Every parse is strict and every failure is
loud, because the alternative -- a silently coerced value -- becomes a wrong match
downstream, and a wrong match is the one outcome this whole submission is built to
avoid.

**The header tuples below are deliberately duplicated from
``hisaab/generator/model.py`` rather than imported.** Two reasons, and the second is
the load-bearing one:

  * The matching path may not import ``hisaab.generator`` at all -- that package
    knows the fee rates, the T+n settlement cycle and the narration templates, so
    importing it is reading the answer with extra steps.
    ``tools/check_isolation.py`` check 6 enforces this.
  * A schema drift should surface as a **loud mismatch**, not be papered over by a
    shared symbol. ``hisaab/scoring/truth_io.py`` makes the same choice for
    ``SUPPORTED_SCHEMA_VERSION``, for the same reason.

Note which kind of thing gets duplicated: a *schema*, where drift must be caught.
Not *logic* -- the business-day calendar is shared from ``hisaab/common/bizdays.py``
precisely because two calendars that disagree by one day would produce a plausible
wrong answer that nothing detects.

The parsing discipline follows ``tools/verify_output.py``: strict ``int`` (a float
that leaked through arrives as ``'1000000.0'`` and must fail rather than truncate), a
mandatory trailing ``Z`` on timestamps, an ISO date or nothing.

``refunds.csv`` is header-only in clean mode and parses to an empty list. That is why
Phase 1 wrote the file rather than skipping it: "the file is absent" and "the file is
empty" are different states, and only one of them is clean mode.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

# --------------------------------------------------------------------------
# Frozen file schemas -- duplicated on purpose. See the module docstring.
# Keep these in the same order as hisaab/generator/model.py; a mismatch in
# *order* is as much a drift as a mismatch in names.
# --------------------------------------------------------------------------

PAYMENTS_HEADER = (
    "payment_id", "order_id", "captured_at", "gross_paise", "method", "currency", "status",
)
SETTLEMENTS_HEADER = (
    "settlement_id", "settled_on", "net_paise", "fee_paise", "gst_paise", "tds_paise", "utr",
)
SETTLEMENT_ITEMS_HEADER = ("settlement_id", "payment_id")
BANK_HEADER = ("row_id", "value_date", "amount_paise", "narration")
REFUNDS_HEADER = ("refund_id", "payment_id", "created_at", "amount_paise")

#: Every file the matcher reads, with the header it must carry.
EXPECTED_HEADERS: dict[str, tuple[str, ...]] = {
    "payments.csv": PAYMENTS_HEADER,
    "settlements.csv": SETTLEMENTS_HEADER,
    "settlement_items.csv": SETTLEMENT_ITEMS_HEADER,
    "bank_statement.csv": BANK_HEADER,
    "refunds.csv": REFUNDS_HEADER,
}


class LoadError(Exception):
    """``data/`` is missing a file, or one of them is not what it claims to be."""


@dataclass(frozen=True, slots=True)
class Payment:
    payment_id: str
    order_id: str
    captured_at: datetime
    gross_paise: int
    method: str
    currency: str
    status: str


@dataclass(frozen=True, slots=True)
class Settlement:
    settlement_id: str
    settled_on: date
    net_paise: int
    fee_paise: int
    gst_paise: int
    tds_paise: int
    utr: str


@dataclass(frozen=True, slots=True)
class Credit:
    """One bank row. Four fields, and no linkage among them -- that is the task."""

    credit_id: str
    value_date: date
    amount_paise: int
    narration: str


@dataclass(frozen=True, slots=True)
class Refund:
    refund_id: str
    payment_id: str
    created_at: datetime
    amount_paise: int


@dataclass(frozen=True, slots=True)
class Dataset:
    """Everything the matcher may see. Deliberately no truth of any kind.

    ``credits`` keeps **bank-file order**, because the verdict file is emitted in that
    order and byte-identical output across two runs is a Phase 3 acceptance item.
    """

    payments: tuple[Payment, ...]
    settlements: tuple[Settlement, ...]
    credits: tuple[Credit, ...]
    refunds: tuple[Refund, ...]
    #: settlement_id -> its payment_ids, in file order. The membership declaration
    #: that ``--settlement-report-late`` withholds in Phase 8, forcing Phase 5's
    #: subset-sum to *find* what is simply read here.
    items: dict[str, tuple[str, ...]]

    def gross_by_payment(self) -> dict[str, int]:
        return {p.payment_id: p.gross_paise for p in self.payments}

    def payments_by_id(self) -> dict[str, Payment]:
        """Every payment by id -- gross **and** method.

        Phase 4's fee model needs both: the rate depends on the method, so a residual
        cannot be explained from the amount alone. ``gross_by_payment`` stays because the
        callers that only need money should not have to know that.
        """
        return {p.payment_id: p for p in self.payments}

    def refunds_by_payment(self) -> dict[str, int]:
        """Payment id -> the refunds citing it, summed in paise. Phase 6 step 6.

        **A lookup, never a search** (decision 9). ``refunds.csv`` states which payment each
        refund belongs to, so the link is *given*; searching for it would multiply the subset
        search's hypothesis count by the refund power set to recover information the file
        already carries.

        Summed rather than returned per refund because a payment may be refunded more than
        once and the arithmetic only cares about the total. The ids stay available through
        ``refunds`` for a caller that needs to name them.

        Only payments that appear in ``payments.csv`` are keyed here; a refund citing anything
        else is an orphan and belongs to ``orphan_refunds`` below, because summing the two
        together is precisely the mistake that would let unattributable money close a gap.
        """
        known = {p.payment_id for p in self.payments}
        totals: dict[str, int] = {}
        for r in self.refunds:
            if r.payment_id in known:
                totals[r.payment_id] = totals.get(r.payment_id, 0) + r.amount_paise
        return totals

    def orphan_refunds(self) -> tuple[Refund, ...]:
        """Refunds citing a payment that is not in this month's ``payments.csv``.

        Real and expected rather than a corruption: a refund for a sale from an earlier month
        is netted off a payout in *this* month, and a single-month export cannot contain the
        payment it reverses. The money genuinely left a settlement, and nothing in these three
        files says which -- so these are the rows that make a residual unexplainable, and
        naming them is what turns ``UNEXPLAINED_RESIDUAL`` into ``REFUND_UNLINKED``.

        ``load`` deliberately does **not** refuse them, unlike an unknown payment in
        ``settlement_items.csv``: membership is a claim about this month's data and a dangling
        reference there is a broken file, while a refund's ``payment_id`` legitimately points
        outside the window.
        """
        known = {p.payment_id for p in self.payments}
        return tuple(r for r in self.refunds if r.payment_id not in known)

    def counts(self) -> dict[str, int]:
        return {
            "payments": len(self.payments),
            "settlements": len(self.settlements),
            "settlement_items": sum(len(v) for v in self.items.values()),
            "bank_rows": len(self.credits),
            "refunds": len(self.refunds),
        }


def _read_csv(path: Path, expected: tuple[str, ...]) -> list[dict[str, str]]:
    """Read one CSV, asserting its header. Returns rows in file order."""
    if not path.exists():
        raise LoadError(f"{path.name} not found at {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = tuple(next(reader))
        except StopIteration:
            raise LoadError(f"{path.name} is completely empty -- not even a header") from None
        if header != expected:
            raise LoadError(
                f"{path.name} header drift.\n"
                f"  expected {expected}\n"
                f"  found    {header}\n"
                f"  The matcher duplicates these tuples rather than importing the "
                f"generator's, so drift shows up here as a loud mismatch. Fix whichever "
                f"side is wrong -- do not widen this check."
            )
        return [dict(zip(header, row)) for row in reader if row]


def _int(value: str, where: str) -> int:
    """Strict int. A float that leaked through arrives as ``'1000000.0'`` and fails."""
    try:
        return int(value)
    except ValueError:
        raise LoadError(
            f"{where}: not an integer: {value!r} -- money is integer paise everywhere, "
            f"so a float here is a real bug rather than a formatting quirk"
        ) from None


def _date(value: str, where: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise LoadError(f"{where}: not an ISO date: {value!r}") from None


def _timestamp(value: str, where: str) -> datetime:
    """``2026-08-10T05:34:22Z`` -> aware datetime. The trailing ``Z`` is mandatory.

    An unlabelled local time written as if it were UTC is a silent one-day error in
    the date window, which is exactly the class of bug the window cannot self-diagnose.
    """
    if not value.endswith("Z"):
        raise LoadError(
            f"{where}: timestamp must be UTC with a trailing Z, got {value!r}"
        )
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise LoadError(f"{where}: not an ISO timestamp: {value!r}") from None


def _unique(label: str, values: list[str]) -> None:
    seen: set[str] = set()
    for v in values:
        if v in seen:
            raise LoadError(f"duplicate {label} in the input data: {v}")
        seen.add(v)


def load(data_dir: Path | str) -> Dataset:
    """Read every file under ``data_dir``. Raises ``LoadError`` on anything unexpected."""
    root = Path(data_dir)
    if not root.is_dir():
        raise LoadError(
            f"{root} is not a directory. Generate a run first:\n"
            f"    python -m hisaab.generator --seed 42 --n 60"
        )

    rows = {name: _read_csv(root / name, header) for name, header in EXPECTED_HEADERS.items()}

    payments = tuple(
        Payment(
            payment_id=r["payment_id"],
            order_id=r["order_id"],
            captured_at=_timestamp(r["captured_at"], f"{r['payment_id']}.captured_at"),
            gross_paise=_int(r["gross_paise"], f"{r['payment_id']}.gross_paise"),
            method=r["method"],
            currency=r["currency"],
            status=r["status"],
        )
        for r in rows["payments.csv"]
    )
    settlements = tuple(
        Settlement(
            settlement_id=r["settlement_id"],
            settled_on=_date(r["settled_on"], f"{r['settlement_id']}.settled_on"),
            net_paise=_int(r["net_paise"], f"{r['settlement_id']}.net_paise"),
            fee_paise=_int(r["fee_paise"], f"{r['settlement_id']}.fee_paise"),
            gst_paise=_int(r["gst_paise"], f"{r['settlement_id']}.gst_paise"),
            tds_paise=_int(r["tds_paise"], f"{r['settlement_id']}.tds_paise"),
            utr=r["utr"],
        )
        for r in rows["settlements.csv"]
    )
    credits = tuple(
        Credit(
            credit_id=r["row_id"],
            value_date=_date(r["value_date"], f"{r['row_id']}.value_date"),
            amount_paise=_int(r["amount_paise"], f"{r['row_id']}.amount_paise"),
            narration=r["narration"],
        )
        for r in rows["bank_statement.csv"]
    )
    refunds = tuple(
        Refund(
            refund_id=r["refund_id"],
            payment_id=r["payment_id"],
            created_at=_timestamp(r["created_at"], f"{r['refund_id']}.created_at"),
            amount_paise=_int(r["amount_paise"], f"{r['refund_id']}.amount_paise"),
        )
        for r in rows["refunds.csv"]
    )

    _unique("payment_id", [p.payment_id for p in payments])
    _unique("settlement_id", [s.settlement_id for s in settlements])
    _unique("credit_id (bank row_id)", [c.credit_id for c in credits])
    _unique("refund_id", [r.refund_id for r in refunds])

    # settlement_items: settlement -> payments, in file order.
    items: dict[str, list[str]] = {}
    known_settlements = {s.settlement_id for s in settlements}
    known_payments = {p.payment_id for p in payments}
    for r in rows["settlement_items.csv"]:
        sid, pid = r["settlement_id"], r["payment_id"]
        if sid not in known_settlements:
            raise LoadError(f"settlement_items.csv cites unknown settlement {sid}")
        if pid not in known_payments:
            raise LoadError(f"settlement_items.csv cites unknown payment {pid}")
        bucket = items.setdefault(sid, [])
        if pid in bucket:
            raise LoadError(f"settlement_items.csv lists {pid} twice under {sid}")
        bucket.append(pid)

    return Dataset(
        payments=payments,
        settlements=settlements,
        credits=credits,
        refunds=refunds,
        items={sid: tuple(pids) for sid, pids in items.items()},
    )


if __name__ == "__main__":
    import tempfile

    def refuses(fn, label: str, expect_in: str = "") -> None:
        try:
            fn()
        except LoadError as e:
            assert expect_in in str(e), f"{label}: message lacks {expect_in!r}\n  got: {e}"
            return
        raise AssertionError(f"load() accepted {label}")

    GOOD: dict[str, list[tuple[str, ...]]] = {
        "payments.csv": [
            ("pay_0001", "ord_0001", "2026-08-03T05:34:22Z", "85358", "card", "INR", "captured"),
            ("pay_0002", "ord_0002", "2026-08-03T06:10:00Z", "197600", "upi", "INR", "captured"),
        ],
        "settlements.csv": [
            ("setl_0005", "2026-08-03", "85358", "0", "0", "0", "XXXX8104"),
            ("setl_0009", "2026-08-03", "197600", "0", "0", "0", "XXXX4451"),
        ],
        "settlement_items.csv": [("setl_0005", "pay_0001"), ("setl_0009", "pay_0002")],
        "bank_statement.csv": [
            ("C0001", "2026-08-03", "85358", "NEFT-RZRPAY-8104"),
            ("C0002", "2026-08-03", "197600", "IMPS CR/RAZORPAY SOFTWARE/4451"),
        ],
        "refunds.csv": [],
    }

    def write(root: Path, files: dict[str, list[tuple[str, ...]]]) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        for name, body in files.items():
            with (root / name).open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, lineterminator="\n")
                w.writerow(EXPECTED_HEADERS[name])
                w.writerows(body)
        return root

    with tempfile.TemporaryDirectory(prefix="hisaab-load-") as tmp:
        base = Path(tmp)
        ds = load(write(base / "good", GOOD))

        assert ds.counts() == {
            "payments": 2, "settlements": 2, "settlement_items": 2,
            "bank_rows": 2, "refunds": 0,
        }
        # refunds.csv is header-only in clean mode and must parse to empty, not absent.
        assert ds.refunds == ()
        assert ds.credits[0].credit_id == "C0001"
        assert ds.credits[0].value_date == date(2026, 8, 3)
        assert ds.credits[0].amount_paise == 85358
        assert ds.settlements[0].settled_on == date(2026, 8, 3)
        assert ds.items == {"setl_0005": ("pay_0001",), "setl_0009": ("pay_0002",)}
        assert ds.gross_by_payment() == {"pay_0001": 85358, "pay_0002": 197600}
        # Bank-file order is preserved -- byte-identical output depends on it.
        assert [c.credit_id for c in ds.credits] == ["C0001", "C0002"]
        # Timestamps arrive aware, so no naive/aware comparison can blow up later.
        assert ds.payments[0].captured_at.tzinfo is not None

        # --- the guards must actually fire ---------------------------------
        refuses(lambda: load(base / "nope"), "a missing directory", "not a directory")

        # A drifted header, in name and in order.
        drifted = base / "drift"
        write(drifted, GOOD)
        with (drifted / "bank_statement.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(("row_id", "value_date", "narration", "amount_paise"))  # reordered
            w.writerow(("C0001", "2026-08-03", "NEFT-RZRPAY-8104", "85358"))
        refuses(lambda: load(drifted), "a reordered header", "header drift")

        # A float that leaked into a money column.
        refuses(
            lambda: load(write(base / "float", {**GOOD, "bank_statement.csv": [
                ("C0001", "2026-08-03", "85358.0", "NEFT-RZRPAY-8104")]})),
            "a float amount", "not an integer",
        )
        # A timestamp with no timezone marker.
        refuses(
            lambda: load(write(base / "naive", {**GOOD, "payments.csv": [
                ("pay_0001", "ord_0001", "2026-08-03T05:34:22", "85358", "card", "INR", "captured"),
            ]})),
            "a timestamp with no Z", "trailing Z",
        )
        refuses(
            lambda: load(write(base / "baddate", {**GOOD, "bank_statement.csv": [
                ("C0001", "03-08-2026", "85358", "NEFT-RZRPAY-8104")]})),
            "a non-ISO date", "not an ISO date",
        )
        # Referential integrity in settlement_items.
        refuses(
            lambda: load(write(base / "ghost_s", {**GOOD, "settlement_items.csv": [
                ("setl_9999", "pay_0001")]})),
            "an unknown settlement", "unknown settlement",
        )
        refuses(
            lambda: load(write(base / "ghost_p", {**GOOD, "settlement_items.csv": [
                ("setl_0005", "pay_9999")]})),
            "an unknown payment", "unknown payment",
        )
        # Duplicate IDs, which would make a verdict per bank row ill-defined.
        refuses(
            lambda: load(write(base / "dup_c", {**GOOD, "bank_statement.csv": [
                ("C0001", "2026-08-03", "85358", "NEFT-RZRPAY-8104"),
                ("C0001", "2026-08-04", "1", "NEFT-RZRPAY-8105")]})),
            "a duplicate credit id", "duplicate credit_id",
        )
        # A missing file, one at a time.
        for name in EXPECTED_HEADERS:
            partial = base / f"missing_{name}"
            write(partial, GOOD)
            (partial / name).unlink()
            refuses(lambda p=partial: load(p), f"a run with no {name}", "not found")

    # --- the committed run, if one exists ---------------------------------
    committed = Path(__file__).resolve().parent.parent.parent / "data"
    if (committed / "bank_statement.csv").exists():
        ds = load(committed)
        counts = ds.counts()
        assert counts["bank_rows"] == counts["payments"] == counts["settlements"]
        assert counts["refunds"] == 0, "clean mode emits no refunds"
        print(f"load.py self-check ok  (committed run: {counts['bank_rows']} bank rows)")
    else:
        print("load.py self-check ok  (no committed data/ to cross-read)")
