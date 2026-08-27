"""The metric block, as text. Quoted verbatim into Phase 11's report header.

Two rules govern this module, and both are about what it is *not* allowed to know.

**It never reads the answer key directly.** Every public function here takes a computed
``Metrics`` and nothing else; the module opens no file and holds no ``Truth``.

It does appear in ``tools/check_isolation.py``'s ``TRUTH_READERS`` allowlist, which is
worth explaining rather than glossing: that check treats importing *anything* under
``hisaab.scoring`` as reaching truth, and this module must import ``Metrics`` for the
type. So the allowlist cannot express "formats but never reads" -- the entry is
transitive, not a licence. The property that actually holds is enforced below instead,
by the self-check asserting that no payment ID reaches the rendered page.

**It never prints the correct answer for a row the matcher got wrong.** A verdict is
reported as the cell it landed in, its reason code, its residual and its value. Never
the payment IDs it should have found. If the scorer printed those, tuning the Phase 3
matcher would become an exercise in fitting the answer key, and the match rate would be
measuring the developer instead of the matcher. ``Landing`` carries no answer to print,
so this rule is structural; the self-check asserts it anyway, because it is the one that
quietly stops being true.

**Wall clock lives here and not in the JSON body.** ``Metrics.as_json`` confines it to
``timing`` so two runs of one matcher on one seed differ only inside that object -- the
metric block is quoted into a report subject to the same reproducibility rule as
everything else. The human block prints it freely; nothing compares the human block
byte-for-byte.
"""

from __future__ import annotations

from ..common.money import fmt
from .metrics import MINUTES_PER_EXCEPTION, Cell, Metrics

#: Minutes a person needs per bank row doing this reconciliation by hand: open the
#: statement, find the settlement, tick it off. **An assumption, not a measurement** --
#: stated in ASSUMPTIONS.md so a judge can challenge the figure instead of discovering
#: it was invented at demo time. 60 rows x 2 min ~ 2 hours, which is the "vs ~2 h by
#: hand" comparison in the block.
BY_HAND_MINUTES_PER_ROW = 2

#: Exceptions listed individually before the queue is summarised by reason. Enough to
#: read the shape of the failure, few enough that a 200-row run stays quotable.
MAX_EXCEPTIONS_LISTED = 10

#: Column width for the label gutter, so every value lines up in a fixed-width report.
LABEL = 26


def pct(rate: float | None, places: int = 1) -> str:
    """A rate as a percentage, or ``n/a`` when the denominator was zero.

    ``n/a`` rather than ``0%`` is load-bearing, not politeness. Clean mode plants zero
    unresolvable cases, so the correct-abstention denominator is 0/0 on every run this
    phase, and ``0%`` there reads as total failure at something that was never
    attempted.
    """
    if rate is None:
        return "n/a"
    return f"{rate * 100:.{places}f}%"


def ratio(numerator: int, denominator: int) -> str:
    """``60/60``, or ``0/0`` -- the raw counts behind a rate, always shown.

    A percentage without its counts hides its own sample size: ``100.0%`` reads the same
    at 3 rows as at 300.
    """
    return f"{numerator}/{denominator}"


def duration(minutes: int) -> str:
    """Minutes as ``0 min`` / ``45 min`` / ``2 h 05 min``, for a human skimming."""
    if minutes < 60:
        return f"{minutes} min"
    hours, mins = divmod(minutes, 60)
    return f"{hours} h {mins:02d} min"


def _line(label: str, value: str, note: str = "") -> str:
    body = f"{label:<{LABEL}} {value}"
    return f"{body}   {note}".rstrip() if note else body


def run_line(m: Metrics) -> str:
    """The provenance line. A score without its mess level means nothing from Phase 4 on.

    Reporting a rate without saying which flags produced it is how a clean-mode 100%
    ends up quoted as though it were measured on messy data.
    """
    mode = "clean mode" if m.clean_mode else f"mess[{','.join(m.flags_enabled)}]"
    flags = ",".join(m.flags_enabled) if m.flags_enabled else "none"
    return f"seed {m.seed}, {m.month}, {mode}, flags: {flags}"


def metric_block(m: Metrics) -> str:
    """The block. One function, text out, stable line order.

    Every counter appears even at zero. A row omitted because its count was nothing is
    a row a reader cannot distinguish from a counter that does not exist.
    """
    rows = m.total_bank_rows
    invented = m.cells[Cell.WRONG_MATCH_INVENTED] + m.cells[Cell.LUCKY_GUESS]
    by_hand = rows * BY_HAND_MINUTES_PER_ROW

    lines = [
        _line(
            "Records processed",
            f"{rows} bank rows "
            f"({m.gateway_credits} gateway, {m.non_gateway_credits} non-gateway)",
        ),
        _line("Run", run_line(m)),
        _line("Matcher", m.matcher),
    ]
    if m.wall_clock_seconds is not None:
        lines.append(_line("Wall clock", f"{m.wall_clock_seconds:.2f}s, unattended"))

    lines += [
        "",
        _line("Coverage", f"{ratio(m.committed, m.gateway_credits):<9}({pct(m.coverage)})",
              "how often it committed"),
        _line("Correctness",
              f"{ratio(m.cells[Cell.CORRECT], m.committed):<9}({pct(m.correctness)})",
              "how often it was right"),
        # The third axis, on its own line and with its denominator visible. Printed even at
        # ``n/a`` for the reason the whole block prints zeros: a rate omitted because it had
        # no denominator is indistinguishable from a rate nobody computes. The denominator
        # matters more here than anywhere else in the block -- 100% over 54 rows and 100%
        # over nothing render identically without it, and only one of them is a result.
        _line("Arithmetic proved",
              f"{ratio(m.decomposition_checked - m.decomposition_mismatches, m.decomposition_checked):<9}"
              f"({pct(m.decomposition_agreement)})",
              "linked right and priced right"),
        _line("Wrong matches", str(m.wrong_matches),
              f"({invented} of them on planted-unresolvable rows)"),
        _line("Wrong ignores", str(m.cells[Cell.WRONG_IGNORE]),
              "real credits discarded as non-gateway" if m.cells[Cell.WRONG_IGNORE] else ""),
        _line(
            "Correct abstentions",
            f"{ratio(m.cells[Cell.CORRECT_ABSTENTION], m.planted_unresolvable):<9}"
            f"({pct(m.abstention_rate)})"
            if m.planted_unresolvable
            else "n/a",
            "" if m.planted_unresolvable
            else f"(0 planted unresolvable in {'clean mode' if m.clean_mode else 'this run'})",
        ),
        _line("Missed", str(m.cells[Cell.MISSED]), "resolvable, but it abstained"),
    ]

    if m.non_gateway_credits or m.ignores_total:
        # Reported apart from the headline, always. Counting correctly-ignored noise as
        # coverage would inflate the rate with the easiest rows in the file.
        lines += [
            "",
            _line("Non-gateway precision",
                  f"{ratio(m.cells[Cell.NOISE_CORRECTLY_IGNORED], m.ignores_total):<9}"
                  f"({pct(m.noise_precision)})"),
            _line("Non-gateway recall",
                  f"{ratio(m.cells[Cell.NOISE_CORRECTLY_IGNORED], m.non_gateway_credits):<9}"
                  f"({pct(m.noise_recall)})",
                  "excluded from the headline, on purpose"),
        ]

    lines += [
        "",
        _line("Exceptions", str(m.exceptions)),
        _line("Value in exceptions", fmt(m.exception_value_paise)),
        _line("Est. human time to clear", duration(m.exception_minutes),
              f"(vs ~{duration(by_hand)} for the batch by hand)"),
    ]
    return "\n".join(lines)


def exception_queue(m: Metrics) -> str:
    """The exception list: what a human would actually work through.

    Ranked by value, because that is the order a finance team clears them in. Phase 9
    replaces this with real grouping and per-group effort; Phase 2 owes it a queue that
    already carries reason, value and estimate so Phase 9 refines a number instead of
    inventing a field.
    """
    queued = sorted(
        (land for land in m.landings if land.outcome.name == "EXCEPTION"),
        key=lambda land: (-land.value_paise, land.credit_id),
    )
    if not queued:
        return "Exception queue                 empty -- every row was resolved or ignored"

    lines = [f"Exception queue ({len(queued)}, by value)", ""]
    for land in queued[:MAX_EXCEPTIONS_LISTED]:
        reason = str(land.reason) if land.reason else "?"
        minutes = MINUTES_PER_EXCEPTION.get(land.reason, 0) if land.reason else 0
        lines.append(
            f"  {land.credit_id:<8} {fmt(land.value_paise):>14}  {reason:<28} ~{minutes} min"
        )
    if len(queued) > MAX_EXCEPTIONS_LISTED:
        lines.append(f"  ... and {len(queued) - MAX_EXCEPTIONS_LISTED} more")

    by_reason: dict[str, list[int]] = {}
    for land in queued:
        by_reason.setdefault(str(land.reason) if land.reason else "?", []).append(
            land.value_paise
        )
    lines += ["", "  by reason:"]
    for reason, values in sorted(by_reason.items(), key=lambda kv: (-sum(kv[1]), kv[0])):
        lines.append(f"    {reason:<30} {len(values):>4}   {fmt(sum(values)):>14}")
    return "\n".join(lines)


def full_report(m: Metrics) -> str:
    """Metric block plus exception queue, which is what ``--text`` prints."""
    return f"{metric_block(m)}\n\n{exception_queue(m)}"


if __name__ == "__main__":
    from ..common.reasons import Reason
    from ..common.verdict import Decomposition, Outcome, Verdict, VerdictFile
    from .metrics import score
    from .truth_io import Truth, TruthCredit, TruthDecomposition

    def _credit(cid: str, pids: tuple[str, ...], *, resolvable: bool = True,
                value: int = 100_000, reason: Reason | None = None) -> TruthCredit:
        return TruthCredit(
            credit_id=cid, settlement_ids=(f"setl_{cid[1:]}",), payment_ids=pids,
            refunds_netted=(), reserve_held_paise=0,
            decomposition=TruthDecomposition(value, 0, 0, 0, 0, 0, value),
            resolvable=resolvable, reason=None if reason is None else str(reason), note=None,
        )

    def _truth(credits: tuple[TruthCredit, ...], noise: tuple[str, ...] = ()) -> Truth:
        return Truth(
            schema_version=1, seed=42, month="2026-08", clean_mode=not noise, flags={},
            counts={"credits": len(credits)}, credits=credits, unsettled_payment_ids=(),
            settlements_without_credit=(), non_gateway_credit_ids=noise,
        )

    def _run(verdicts: tuple[Verdict, ...]) -> VerdictFile:
        return VerdictFile(42, "2026-08", "fixture:selfcheck@1", verdicts,
                           wall_clock_seconds=0.02)

    def _resolved(cid: str, pids: tuple[str, ...], *, gross: int = 100_000,
                  fee: int = 0, gst: int = 0, residual: int = 0) -> Verdict:
        """A resolved verdict whose proof balances, so the contract accepts it.

        ``residual`` shifts the credit amount rather than being stated independently: the
        verdict contract requires ``residual == credit - expected``, so a caller asking for
        a 500p residual gets a credit 500p above what the decomposition accounts for. That
        is the only way to build one now, and it is the point of v2 -- a residual is a
        checksum over published arithmetic, not a number a fixture can simply assert.
        """
        return Verdict(
            cid, Outcome.RESOLVED, (f"setl_{cid[1:]}",), pids, tier=1,
            residual_paise=residual,
            credit_amount_paise=gross - fee - gst + residual,
            decomposition=Decomposition(gross, fee_paise=fee, gst_paise=gst),
        )

    # --- formatters ---------------------------------------------------------
    assert pct(None) == "n/a" and pct(1.0) == "100.0%" and pct(0.0) == "0.0%"
    assert pct(0.9) == "90.0%" and pct(21 / 60) == "35.0%"
    assert ratio(60, 60) == "60/60" and ratio(0, 0) == "0/0"
    assert duration(0) == "0 min" and duration(59) == "59 min"
    assert duration(60) == "1 h 00 min" and duration(125) == "2 h 05 min"

    three = tuple(_credit(f"C{i:04d}", (f"pay_{i:04d}",)) for i in (1, 2, 3))

    # --- the oracle shape: 100% and every n/a rendered as n/a --------------
    oracle = score(
        _run(tuple(_resolved(c.credit_id, c.payment_ids) for c in three)),
        _truth(three),
    )
    block = metric_block(oracle)
    assert "Coverage" in block and "3/3" in block and "(100.0%)" in block
    # The third axis, with its denominator on the page. A rate without it reads the same
    # over 3 rows as over none, and only one of those is a result.
    assert "Arithmetic proved          3/3      (100.0%)" in block, block
    assert "linked right and priced right" in block
    assert "Correct abstentions        n/a" in block, block
    assert "0 planted unresolvable in clean mode" in block
    assert "seed 42, 2026-08, clean mode, flags: none" in block
    assert "Wall clock" in block and "0.02s" in block
    assert "Value in exceptions        ₹0.00" in block, block
    assert "vs ~6 min for the batch by hand" in block  # 3 rows x 2 min
    assert "empty" in exception_queue(oracle)
    # Non-gateway lines stay out entirely when there is no noise and nothing ignored.
    assert "Non-gateway" not in block

    # --- the stub shape: 0% coverage, n/a correctness -----------------------
    stub = score(_run(tuple(
        Verdict(c.credit_id, Outcome.EXCEPTION, reason=Reason.NO_CANDIDATE) for c in three
    )), _truth(three))
    block = metric_block(stub)
    assert "Coverage                   0/3      (0.0%)" in block, block
    assert "Correctness                0/0      (n/a)" in block, block
    # Nothing committed means nothing to price-check. The line still appears, and it shows
    # ``0/0 (n/a)`` rather than a flattering ``100%`` over an empty denominator -- a matcher
    # that abstained on everything has proved no arithmetic, not all of it.
    assert "Arithmetic proved          0/0      (n/a)" in block, block
    assert "Missed                     3" in block
    assert "Est. human time to clear   30 min" in block, block  # 3 x 10 min
    queue = exception_queue(stub)
    assert "Exception queue (3, by value)" in queue
    assert "NO_CANDIDATE" in queue and "~10 min" in queue
    assert "by reason:" in queue

    # --- the answer key must not reach the page ----------------------------
    wrong = score(
        _run((_resolved("C0001", ("pay_0009",), residual=500),)),
        _truth((_credit("C0001", ("pay_0001",)),)),
    )
    text = full_report(wrong)
    assert "Wrong matches              1" in text
    assert "pay_0001" not in text, (
        "the block printed the correct answer for a wrong match -- that is how Phase 3 "
        "starts fitting the answer key instead of matching"
    )
    assert "pay_" not in text, "no payment id belongs in the report at all"
    # A wrong match's arithmetic is never compared -- it describes a different payment set --
    # so the denominator here is 0 while coverage is 1/1. Asserted on the rendered page as
    # well as in metrics, because this is the line a reader would misread as "the arithmetic
    # checked out" if it silently showed 100%.
    assert "Arithmetic proved          0/0      (n/a)" in text, text

    # --- planted unresolvables make the abstention line real ---------------
    planted = score(
        _run((Verdict("C0001", Outcome.EXCEPTION,
                      reason=Reason.AMBIGUOUS_DUPLICATE_AMOUNT),)),
        _truth((_credit("C0001", ("pay_0001",), resolvable=False,
                        reason=Reason.AMBIGUOUS_DUPLICATE_AMOUNT),)),
    )
    block = metric_block(planted)
    assert "Correct abstentions        1/1      (100.0%)" in block, block
    assert "planted unresolvable in" not in block

    # --- noise gets its own pair, apart from the headline ------------------
    mixed = score(
        _run((
            _resolved("C0001", ("pay_0001",)),
            Verdict("C0002", Outcome.IGNORED, reason=Reason.NON_GATEWAY_CREDIT),
        )),
        _truth((_credit("C0001", ("pay_0001",)),), noise=("C0002",)),
    )
    block = metric_block(mixed)
    assert "1 bank rows" not in block and "2 bank rows (1 gateway, 1 non-gateway)" in block
    assert "Coverage                   1/1      (100.0%)" in block, block
    assert "Non-gateway precision      1/1      (100.0%)" in block, block
    assert "excluded from the headline" in block

    # Every line fits a terminal and the gutter is aligned.
    for line in full_report(mixed).splitlines():
        assert len(line) <= 100, f"{len(line)} chars: {line}"

    print("report.py self-check ok")
