"""Joins the queue to the bank statement, so groups can be ranked by money at risk.

**Effort ranks the wrong thing.** ``group.py`` orders by minutes because that is all it can see,
and a queue ordered that way puts forty three-minute dismissals above one ₹4-lakh unresolved
credit. A finance team clears the money first. So the money has to come from somewhere, and the
only place it exists is ``bank_statement.csv``.

**This reuses ``matcher.load.load`` rather than parsing the CSV again.** ``hisaab/matcher`` is
itself inside ``MATCHER_PACKAGES``, so ``tools/check_isolation.py`` permits the import -- the
allowlist is what the *matching path* may read, and both packages are on it. The reuse is not
merely permitted but preferred: ``load`` already enforces the header tuple, parses money through
strict ``_int`` (a float that leaked through fails rather than truncating), and refuses a
duplicate ``row_id``. An amount is a **declared quantity**, and two parsers that disagree about
one produce a queue ranked in a plausible wrong order that nothing detects -- the reasoning that
keeps the calendar shared in ``bizdays.py``. Contrast ``read.py``, which duplicates the act of
subscripting three JSON keys precisely because a *schema* drift must fail loudly.

**The two inputs are not provably from the same run, so the join proves it instead.** Nothing in
``data/`` carries a run id -- the manifest that would is a truth file, unreadable here by design
-- so ``matches.json`` from one run and ``data/`` from another is a live mistake, easy to make
with two terminals open. A ranking built on that mismatch would be silently wrong about every
number. So the join is required to be **total in both directions**: every ruling names a bank
row, and every bank row has a ruling. Either gap refuses, naming the ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..common.money import fmt
from ..matcher.load import LoadError, load
from .group import Group
from .read import Ruling, TriageError


@dataclass(frozen=True, slots=True)
class Item:
    """One row in the queue: what the matcher said, and what it is worth."""

    credit_id: str
    value_paise: int
    #: The ruling as ``matches.json`` stated it, carried through rather than rebuilt from the
    #: group. A dismissal's own code survives here even though ``group.py`` deliberately files
    #: every dismissal under one heading: the grouping decision is about causes of *work*, and
    #: throwing away a code the file actually stated would make the queue unable to answer
    #: "why was this one set aside" for a row a person queries.
    ruling: Ruling

    @property
    def display(self) -> str:
        return fmt(self.value_paise)


@dataclass(frozen=True, slots=True)
class RankedGroup:
    """A ``Group`` with its money attached, and its members ordered by value."""

    group: Group
    items: tuple[Item, ...]

    @property
    def label(self) -> str:
        return self.group.label

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def total_minutes(self) -> int:
        return self.group.total_minutes

    @property
    def value_paise(self) -> int:
        """Money at risk in this group. Summed from members, so it cannot disagree with them."""
        return sum(i.value_paise for i in self.items)


def amounts(data_dir: Path | str) -> dict[str, int]:
    """``credit_id -> amount_paise`` for every bank row, via the matcher's own loader.

    ``LoadError`` is translated rather than propagated: a person who ran ``hisaab.triage`` should
    read a triage failure, not a matcher one, and the two packages are separately runnable.
    """
    try:
        dataset = load(data_dir)
    except LoadError as e:
        raise TriageError(f"cannot read the bank statement: {e}") from e

    by_id: dict[str, int] = {}
    for credit in dataset.credits:
        # ``load`` has already refused duplicates; asserted rather than trusted, because this
        # dict is what every value in the queue is read from.
        assert credit.credit_id not in by_id, f"load() returned {credit.credit_id} twice"
        if credit.amount_paise < 0:
            # Never observed: no invariant declares bank rows non-negative, and no generated
            # run has produced one. Refused rather than ranked, because a debit row needs a
            # deliberate answer to "what is at risk here" -- and ranking by a signed amount
            # would sort it below every ₹0 row, which is where things go to be ignored.
            raise TriageError(
                f"{credit.credit_id}: bank amount is negative ({credit.amount_paise}p). The "
                f"queue ranks by money at risk and has no rule for a debit -- decide what one "
                f"is worth before ranking it, rather than letting it sort to the bottom."
            )
        by_id[credit.credit_id] = credit.amount_paise
    return by_id


def rank(
    groups: tuple[Group, ...],
    by_id: dict[str, int],
    rulings: tuple[Ruling, ...],
) -> tuple[RankedGroup, ...]:
    """Attach money to each group and order the queue by it, heaviest first.

    Takes the rulings as well as the groups because a ``Group`` keeps ids, not rulings, and an
    ``Item`` must carry the ruling **as the file stated it**. Rebuilding one from its group's
    ``reason`` was the first shape of this function and it was wrong twice over: exact for
    exception groups but lossy for the dismissal group, which files rows under one heading
    regardless of the code each carried, so every dismissal came back with ``reason=None`` and
    the file's own answer to "why was this set aside" was destroyed in the join.

    Ordered by value descending, then by minutes descending, then by label -- a total order that
    does not depend on dict iteration, so two runs of the same data print the same queue. Value
    leads because that is what a finance team clears by; effort breaks a tie because between two
    groups worth the same, the expensive one is the one to start on.

    Members are ordered by value descending too, then by ``credit_id``. So the first line of the
    first group is the single most valuable unresolved row in the month, which is the one number
    a person reads if they read nothing else.
    """
    by_credit = {r.credit_id: r for r in rulings}

    ranked: list[RankedGroup] = []
    for g in groups:
        items: list[Item] = []
        for cid in g.credit_ids:
            try:
                value = by_id[cid]
            except KeyError:
                raise TriageError(
                    f"{cid} has a verdict but is not in the bank statement. matches.json and "
                    f"the data directory are from different runs -- re-run the matcher against "
                    f"this data rather than ranking a row that does not exist."
                ) from None
            # An id in a group but not in the rulings means the groups were built from a
            # different file than the one passed here. Asserted rather than defaulted: the
            # alternative is an Item carrying a made-up ruling, which is the defect this
            # parameter exists to remove.
            assert cid in by_credit, (
                f"{cid} is in a group but not in the rulings -- rank() was given groups built "
                f"from a different verdict file"
            )
            items.append(Item(credit_id=cid, value_paise=value, ruling=by_credit[cid]))
        items.sort(key=lambda i: (-i.value_paise, i.credit_id))
        ranked.append(RankedGroup(group=g, items=tuple(items)))

    ranked.sort(key=lambda r: (-r.value_paise, -r.total_minutes, r.label))
    return tuple(ranked)


def check_total(rulings: tuple[Ruling, ...], by_id: dict[str, int]) -> None:
    """Refuse unless the verdict file and the bank statement describe the same rows.

    The other direction of the check in ``rank``. A bank row with no verdict is not a smaller
    queue, it is an **unexamined row**: the matcher never reached it, and a queue that omits it
    silently reports less work than exists. ``engine.py:123`` already refuses to emit such a
    file, so this catches the mismatched-runs case rather than a matcher bug.
    """
    judged = {r.credit_id for r in rulings}
    missing = sorted(set(by_id) - judged)
    if missing:
        shown = ", ".join(missing[:5]) + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else "")
        raise TriageError(
            f"{len(missing)} bank row(s) have no verdict: {shown}. The matcher emits one verdict "
            f"per bank row, so this is matches.json and data/ from different runs -- a queue "
            f"built on it would under-report the work, which is the one error a queue must not "
            f"make."
        )

    # Extra ids, the other direction. ``rank`` also refuses this, but only for rows that reach a
    # group -- a stray verdict on a RESOLVED row never does, so without this it slips through.
    extra = sorted(judged - set(by_id))
    if extra:
        shown = ", ".join(extra[:5]) + (f" (+{len(extra) - 5} more)" if len(extra) > 5 else "")
        raise TriageError(
            f"{len(extra)} verdict(s) name rows that are not in the bank statement: {shown}. "
            f"matches.json and data/ are from different runs."
        )

    # --- and the check that catches two runs of the *same size* -------------------------
    # Neither test above can: when both runs have the same number of bank rows the credit ids
    # coincide exactly, so every id is accounted for in both directions and the queue comes out
    # plausible and wrong -- seed 1's amounts attached to seed 2's verdicts. Gate 16 found this
    # by producing exactly that queue at exit 0.
    #
    # The matcher's own stated amount is the discriminator. Measured across 12 cells (3 seeds x
    # n=60/200 x clean and --all-mess): 1140 verdicts state an amount and all 1140 equal their
    # own run's bank row, so a disagreement is never normal. Swapping two seeds makes 35 of 37
    # disagree, so one mismatched pair is enough to refuse and the first is named.
    #
    # Only where the matcher stated one: ``EXCEPTION`` serialises it as null, and those rows are
    # checked by id alone. That is a real limit rather than a hidden one -- a file of nothing but
    # exceptions has no amounts to compare -- and it is the reason this sits alongside the two id
    # checks instead of replacing them.
    for r in rulings:
        if r.stated_amount_paise is None:
            continue
        actual = by_id[r.credit_id]
        if r.stated_amount_paise != actual:
            raise TriageError(
                f"{r.credit_id}: the verdict says this bank row is {r.stated_amount_paise}p but "
                f"the statement says {actual}p. matches.json and data/ describe different runs "
                f"-- re-run the matcher against this data. Every number in a queue built on "
                f"these two files would be about the wrong month, which is worse than no queue: "
                f"it would look right."
            )


def total_value(ranked: tuple[RankedGroup, ...]) -> int:
    return sum(r.value_paise for r in ranked)


if __name__ == "__main__":
    import csv
    import tempfile

    from ..common.reasons import Reason
    from ..common.verdict import Outcome
    from .group import group

    E, I, R = Outcome.EXCEPTION, Outcome.IGNORED, Outcome.RESOLVED

    def _r(cid: str, outcome: Outcome, reason: Reason | None = None) -> Ruling:
        return Ruling(credit_id=cid, outcome=outcome, reason=reason)

    def refuses(fn, label: str, expect_in: str) -> None:
        try:
            fn()
        except TriageError as e:
            assert expect_in in str(e), f"{label}: message lacks {expect_in!r}\n  got: {e}"
            return
        raise AssertionError(f"accepted {label}")

    # A minimal but *valid* dataset: load() enforces headers and referential integrity, so
    # these files have to be real ones rather than a bank statement on its own.
    BANK = [
        ("C0001", "2026-08-03", "85358", "NEFT-RZRPAY-8104"),
        ("C0002", "2026-08-03", "500000", "IMPS CR/RAZORPAY/4451"),
        ("C0003", "2026-08-04", "12000", "NEFT SOMEONE ELSE"),
        ("C0004", "2026-08-04", "300000", "NEFT-RZRPAY-9001"),
        ("C0005", "2026-08-05", "700", "UPI MISC"),
    ]
    FILES = {
        "payments.csv": (
            ("payment_id", "order_id", "captured_at", "gross_paise", "method", "currency",
             "status"),
            [("pay_0001", "ord_0001", "2026-08-03T05:34:22Z", "85358", "card", "INR",
              "captured")],
        ),
        "settlements.csv": (
            ("settlement_id", "settled_on", "net_paise", "fee_paise", "gst_paise", "tds_paise",
             "utr"),
            [("setl_0001", "2026-08-03", "85358", "0", "0", "0", "XXXX8104")],
        ),
        "settlement_items.csv": (
            ("settlement_id", "payment_id"), [("setl_0001", "pay_0001")],
        ),
        "bank_statement.csv": (("row_id", "value_date", "amount_paise", "narration"), BANK),
        "refunds.csv": (("refund_id", "payment_id", "created_at", "amount_paise"), []),
    }

    def write_data(root: Path, bank: list[tuple[str, ...]] | None = None) -> Path:
        d = root
        d.mkdir(parents=True, exist_ok=True)
        for name, (header, rows) in FILES.items():
            body = bank if (name == "bank_statement.csv" and bank is not None) else rows
            with (d / name).open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(body)
        return d

    with tempfile.TemporaryDirectory(prefix="hisaab-triage-value-") as tmp:
        root = Path(tmp)
        data = write_data(root / "data")

        # --- the positive control: the loader's numbers, unchanged ------------------------
        by_id = amounts(data)
        assert by_id == {"C0001": 85358, "C0002": 500000, "C0003": 12000,
                         "C0004": 300000, "C0005": 700}, by_id

        rulings = (
            _r("C0001", R),                                 # resolved: not in the queue
            _r("C0002", E, Reason.NO_CANDIDATE),            # ₹5,000.00, 10 min
            _r("C0003", I, Reason.NON_GATEWAY_CREDIT),      # ₹120.00, 3 min
            _r("C0004", E, Reason.NO_CANDIDATE),            # ₹3,000.00, 10 min
            _r("C0005", I, None),                           # ₹7.00, 3 min
        )
        check_total(rulings, by_id)
        ranked = rank(group(rulings), by_id, rulings)

        # Two groups: one code that occurred, plus the dismissals.
        assert [r.label for r in ranked] == ["NO_CANDIDATE", "DISMISSED (not gateway money)"], (
            [r.label for r in ranked]
        )
        # Money leads, and it inverts nothing here only because effort agrees; the losing
        # branch is exercised below.
        assert ranked[0].value_paise == 800_000 and ranked[0].total_minutes == 20
        assert ranked[1].value_paise == 12_700 and ranked[1].total_minutes == 6
        assert total_value(ranked) == 812_700
        # The resolved row's ₹853.58 is in no group, so it is not "at risk".
        assert 85_358 not in [i.value_paise for r in ranked for i in r.items]

        # Members ordered by value: the biggest unresolved row is the queue's first line.
        assert [i.credit_id for i in ranked[0].items] == ["C0002", "C0004"]
        assert ranked[0].items[0].display == "₹5,000.00", ranked[0].items[0].display
        # Every item's value is its own bank row's, not its group's or its neighbour's.
        for rg in ranked:
            for i in rg.items:
                assert i.value_paise == by_id[i.credit_id], i
        # Each item carries the outcome its group is made of.
        assert all(i.ruling.is_exception for i in ranked[0].items)
        # Every dismissal keeps the code the file stated, even though they share one heading.
        # **This is the assertion that fails if rank() ever rebuilds a ruling from its group
        # again**, which the first version of it did: C0003 came off disk as
        # NON_GATEWAY_CREDIT and must still say so, while C0005 legitimately carried none.
        # Under the reconstruction both read None and nothing noticed.
        assert all(i.ruling.is_dismissal for i in ranked[1].items)
        assert [(i.credit_id, i.ruling.reason) for i in ranked[1].items] == [
            ("C0003", Reason.NON_GATEWAY_CREDIT),
            ("C0005", None),
        ], [(i.credit_id, i.ruling.reason) for i in ranked[1].items]
        # ...while the group heading still drops the code, which is group.py's decision and
        # remains true: the two facts live at different levels rather than contradicting.
        assert ranked[1].group.reason is None and ranked[1].label.startswith("DISMISSED")

        # Every queued row appears exactly once across the whole queue.
        placed = [i.credit_id for r in ranked for i in r.items]
        assert sorted(placed) == ["C0002", "C0003", "C0004", "C0005"]
        assert len(placed) == len(set(placed)), "a row appears in two groups"

        # --- value must be able to *beat* effort, or the sort key is decoration ----------
        # One 3-minute dismissal worth ₹9,000 outranks two 10-minute exceptions worth ₹127
        # between them. Effort-descending would print these the other way round.
        flipped = (
            _r("C0001", R),
            _r("C0002", I, Reason.NON_GATEWAY_CREDIT),      # ₹5,000.00, 3 min
            _r("C0003", E, Reason.NO_CANDIDATE),            # ₹120.00, 10 min
            _r("C0005", E, Reason.NO_CANDIDATE),            # ₹7.00, 10 min
            _r("C0004", R),
        )
        fr = rank(group(flipped), amounts(data), flipped)
        assert [r.label for r in fr] == ["DISMISSED (not gateway money)", "NO_CANDIDATE"], (
            [(r.label, r.value_paise, r.total_minutes) for r in fr]
        )
        assert fr[0].total_minutes == 3 < fr[1].total_minutes == 20, "effort did not lose"

        # --- the empty queue, again: a month where everything resolved -------------------
        allgood = tuple(_r(cid, R) for cid in by_id)
        assert rank(group(allgood), by_id, allgood) == ()
        assert total_value(()) == 0
        check_total(allgood, by_id)  # total in both directions, with nothing queued

        # --- mismatched runs, all three ways -------------------------------------------
        refuses(lambda: rank(group((_r("C9999", E, Reason.NO_CANDIDATE),)), by_id,
                            (_r("C9999", E, Reason.NO_CANDIDATE),)),
                "a verdict for a row not in the statement", "different runs")
        refuses(lambda: check_total((_r("C0001", R),), by_id),
                "a statement with unexamined rows", "no verdict")
        # ...and the message names them, since a count alone is not actionable.
        try:
            check_total((_r("C0001", R),), by_id)
        except TriageError as e:
            assert "C0002" in str(e) and "4 bank row(s)" in str(e), e

        # The extra-ids direction, which ``rank`` cannot cover: a stray verdict on a RESOLVED
        # row reaches no group, so only ``check_total`` ever sees it.
        refuses(lambda: check_total(rulings + (_r("C9999", R),), by_id),
                "a verdict for a row the statement does not have", "not in the bank statement")

        # --- two runs of the same size, which no id check can catch ---------------------
        # The case gate 16 found: same row count means the ids coincide exactly, both id
        # checks pass, and the queue comes out plausible and wrong. The stated amount is the
        # only discriminator left.
        def stating(cid: str, outcome: Outcome, amount: int | None,
                    reason: Reason | None = None) -> Ruling:
            return Ruling(credit_id=cid, outcome=outcome, reason=reason,
                          stated_amount_paise=amount)

        # Positive control first: agreeing amounts must pass, or the refusal below proves
        # nothing -- a check that rejects every file would satisfy it too.
        agreeing = tuple(stating(cid, R, amt) for cid, amt in by_id.items())
        check_total(agreeing, by_id)
        # One row off by a single paisa, with every id present on both sides.
        swapped = tuple(
            stating(cid, R, amt + 1 if cid == "C0003" else amt) for cid, amt in by_id.items()
        )
        assert {r.credit_id for r in swapped} == set(by_id), "the id sets must still match"
        refuses(lambda: check_total(swapped, by_id), "a same-size run swap", "wrong month")
        try:
            check_total(swapped, by_id)
        except TriageError as e:
            assert "C0003" in str(e) and "12001p" in str(e) and "12000p" in str(e), e
        # A verdict that states no amount is tolerated, which is the check's stated limit:
        # EXCEPTION rows serialise it as null and are covered by id alone.
        check_total(tuple(stating(cid, R, None) for cid in by_id), by_id)

        # --- a negative bank amount is refused, not ranked last -------------------------
        neg = write_data(root / "neg", bank=[("C0001", "2026-08-03", "85358", "NEFT-RZRPAY-8104"),
                                             ("C0002", "2026-08-03", "-500", "REVERSAL")])
        refuses(lambda: amounts(neg), "a negative bank amount", "no rule for a debit")

        # --- a broken data directory reads as a triage failure, not a matcher one -------
        refuses(lambda: amounts(root / "nowhere"), "a missing data directory", "cannot read")
        bad = write_data(root / "bad")
        (bad / "bank_statement.csv").write_text("row_id,value_date,amount\nC1,x,1\n",
                                                encoding="utf-8")
        refuses(lambda: amounts(bad), "a drifted bank header", "header drift")

    print("triage/value.py self-check ok")
