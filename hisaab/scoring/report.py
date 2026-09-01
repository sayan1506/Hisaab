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

from typing import NamedTuple

from ..common.money import fmt
from ..common.verdict import Outcome
from .metrics import Cell, Metrics, minutes_for

#: Minutes a person needs for a bank row that reconciles on sight: open the statement, find
#: the settlement, tick it off. **An assumption, not a measurement** -- stated in
#: ASSUMPTIONS.md so a judge can challenge the figure instead of discovering it was invented
#: at demo time.
#:
#: This was ``BY_HAND_MINUTES_PER_ROW``, a single flat rate applied to every row in the file,
#: and Phase 9 split it because one rate made the whole ROI claim read backwards. A flat 2
#: minutes says a row nothing matches costs the same as a row that ticks off, so the by-hand
#: total came out *below* the tool's own estimate on all six measured cells -- the report
#: printed both figures and never subtracted them, so nothing caught it for eight phases.
BY_HAND_MINUTES_EASY = 2

#: Minutes for a row a person cannot reconcile on sight -- the ones the matcher raises as
#: exceptions. Chase the settlement report, call the bank, reconstruct a payment set.
#:
#: **15, and the number it has to beat is measured rather than argued.**
#: `.plan/probe_phase9_roi_breakeven.py` computes the break-even rate per cell -- the rate at
#: which doing the batch by hand stops costing more than working the tool's queue. Across
#: three seeds at two sizes the binding cell needs **13.17 min** (seed 1, n=200; the others
#: run 8.67-12.93), and charging dismissals moves that to **13.34**. So 15 clears every
#: measured cell with 1.66 minutes to spare, and the measured saving runs 9.1-29.8% with the
#: floor on that same binding cell.
#:
#: The margin is thin on purpose rather than by luck: 12 minutes -- a defensible reading of
#: "a person just calls the bank" -- inverts every cell. That is why the block prints this
#: run's break-even beside the saving instead of a bare percentage: a reader can see how much
#: room the claim has rather than trusting the point estimate.
BY_HAND_MINUTES_HARD = 15

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


class Roi(NamedTuple):
    """The two totals, side by side, and the rate at which they cross.

    **The whole point is that something subtracts them.** Before Phase 9 the block printed
    ``exception_minutes`` and ``rows x 2`` on one line and never compared them; on all six
    measured cells the second was *smaller*, so the shipped report was quietly claiming a tool
    that cost an operator 2-3x more time than ignoring it. Eight phases and fifteen gates
    missed it because no assertion anywhere put the two numbers on opposite sides of a ``>``.
    """

    #: Rows a person clears on sight, at ``BY_HAND_MINUTES_EASY``. Resolved rows, dismissed
    #: rows, and exceptions raised on money that is not gateway money -- an operator glances
    #: at those and moves on, whatever the matcher said about them.
    easy_rows: int
    #: Rows a person has to chase, at ``BY_HAND_MINUTES_HARD``: exceptions on real gateway
    #: credits. This is the population the break-even was measured against.
    hard_rows: int
    by_hand_minutes: int
    #: What the tool leaves a person to do: exception minutes plus charged dismissals.
    tool_minutes: int

    @property
    def saved_minutes(self) -> int:
        return self.by_hand_minutes - self.tool_minutes

    @property
    def saved_fraction(self) -> float | None:
        """``None`` when there is no by-hand baseline to be a fraction of."""
        if not self.by_hand_minutes:
            return None
        return self.saved_minutes / self.by_hand_minutes

    @property
    def break_even_hard_minutes(self) -> float | None:
        """The by-hand hard rate at which the two totals cross.

        Below it, doing the batch by hand is cheaper and the claim inverts. ``None`` when
        there are no hard rows -- with nothing to chase there is no rate to be sensitive to,
        and a number here would be an artefact of dividing by zero rows.
        """
        if not self.hard_rows:
            return None
        return (self.tool_minutes - self.easy_rows * BY_HAND_MINUTES_EASY) / self.hard_rows

    @property
    def tool_wins(self) -> bool:
        return self.by_hand_minutes > self.tool_minutes


def roi(m: Metrics) -> Roi:
    """Split the file into rows a person clears on sight and rows they chase.

    Counted **row by row from the landings**, not derived by subtracting counters. The
    arithmetic shortcut is available -- ``.plan/probe_phase9_hard_denominator.py`` measured
    ``hard == exceptions - cells[NOISE_MISHANDLED]`` and it is exact on all six cells and in
    clean mode -- and it is not used, because it is exact only while no noise row is ever
    RESOLVED. That is a ``WRONG_*``-flavoured event: rare rather than impossible, so it would
    pass every cell anyone tested and then quietly compare a by-hand total against a
    population the break-even was never measured on. Counting the buckets needs no invariant
    to hold.
    """
    easy = hard = 0
    for land in m.landings:
        if land.outcome is Outcome.EXCEPTION and land.cell is not Cell.NOISE_MISHANDLED:
            hard += 1
        else:
            easy += 1

    # The partition, asserted rather than assumed: every bank row is on exactly one side of
    # this comparison, or the two totals are over different populations.
    assert easy + hard == m.total_bank_rows, (
        f"the ROI split lost rows: {easy} easy + {hard} hard != {m.total_bank_rows} bank rows"
    )
    return Roi(
        easy_rows=easy,
        hard_rows=hard,
        by_hand_minutes=easy * BY_HAND_MINUTES_EASY + hard * BY_HAND_MINUTES_HARD,
        tool_minutes=m.exception_minutes + m.dismissal_minutes,
    )


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
    r = roi(m)

    lines = [
        _line(
            "Records processed",
            f"{rows} bank rows "
            f"({m.gateway_credits} gateway, {m.non_gateway_credits} non-gateway)"
            # Stated only when the two differ, which is exactly when --batching is on. On a
            # 1:1 run "from 60 payments" after "60 bank rows" is noise; under batching its
            # absence would let 200 payments read as 120 records, and the track's floor is
            # checked against one of those two numbers (.plan/phase5.md decision 3).
            + (
                f" from {m.total_payments} payments"
                if m.total_payments and m.total_payments != rows
                else ""
            ),
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
        # The amendment's owed line: money already booked wrong, not money awaiting a human.
        # A wrong match raises no exception, so it is invisible to the line above -- this is
        # the only place a ₹2,00,000 wrong match and a ₹49 one stop reading identically.
        _line("Value at risk", fmt(m.wrong_match_value_paise),
              "wrong matches, already booked wrong"),
        # Four lines where there was one, and the fourth is the only one that is a *claim*.
        # The old single line printed the tool's minutes with the by-hand total as a note and
        # never compared them; these print both totals, then subtract them out loud.
        _line("Est. human time to clear", duration(r.tool_minutes),
              f"({duration(m.exception_minutes)} exceptions "
              f"+ {duration(m.dismissal_minutes)} dismissals)"),
        _line("Same batch by hand", duration(r.by_hand_minutes),
              f"({r.easy_rows} on sight x {BY_HAND_MINUTES_EASY} min "
              f"+ {r.hard_rows} chased x {BY_HAND_MINUTES_HARD} min)"),
    ]

    # The branch. A time-saved figure is a claim that can be false, so the false case gets
    # its own wording rather than a negative percentage in the same sentence -- that is
    # exactly how the inverted claim survived eight phases: a reader skimming for a number
    # found one, and it never said which way round the comparison had come out.
    if m.wrong_matches:
        # **A wrong match is invisible to this comparison, so the comparison must not be made.**
        # Found by regenerating README's fixture output: ``zip``, which matches by row position,
        # scores 35% correctness with 39 wrong matches and printed "Time saved 100.0%" -- the
        # best possible figure, earned by committing wrongly on every row. ``saboteur`` did the
        # same with 6. The arithmetic was right and the claim was absurd: a wrong match raises
        # no exception, so it costs the tool side nothing, and the queue a person clears says
        # nothing about the rows they were never told to look at.
        #
        # This is the third rendering of this figure that read as informative and could not be
        # checked, after the un-subtracted totals and the negative break-even. It also inverts
        # this project's own thesis -- that zero wrong matches beats higher coverage -- so the
        # figure is withheld rather than qualified: a percentage printed beside a caveat is
        # still the number a reader takes away.
        #
        # No remediation rate is invented to charge them with. Nobody was timed for a routine
        # row, let alone for finding a mis-booked one months later, and a made-up figure here
        # would be the same defect wearing a correction's clothes.
        note = (
            f"(a wrong match leaves no queue row, so the {duration(r.tool_minutes)} above is "
            f"not what this run left a person to do)"
        )
        if not r.tool_wins:
            note = note[:-1] + f", and it already costs {duration(-r.saved_minutes)} more)"
        lines.append(_line(
            "Time saved", f"not claimable -- {m.wrong_matches} wrong match(es)", note,
        ))
    elif r.tool_wins:
        lines.append(_line("Time saved", pct(r.saved_fraction),
                           f"({duration(r.saved_minutes)} less than by hand)"))
    else:
        lines.append(_line(
            "Time saved", "none -- COSTS MORE",
            f"(clearing this queue takes {duration(-r.saved_minutes)} longer than by hand)",
        ))

    # Printed beside the claim on purpose: the saving is sensitive to one assumed rate, and a
    # reader who can see where it crosses can judge how much room it has. On the measured
    # binding cell that is 13.34 against an assumed 15 -- a 1.66-minute margin, which is a
    # fact about the claim that no percentage on its own conveys.
    be = r.break_even_hard_minutes
    if be is None:
        lines.append(_line("Break-even chased rate", "n/a",
                           "(nothing chased, so no rate to cross)"))
    elif be <= 0:
        # **A negative crossing point is not a rate, so it must not be printed as one.** The
        # arithmetic is right -- it is where the two lines meet -- but a by-hand rate below
        # zero cannot happen, so "below this, by hand is cheaper" would be an unfalsifiable
        # claim about an impossible number. That is the same defect as the inverted line this
        # block replaced: a figure that reads as informative and cannot be checked.
        #
        # What it actually means is stronger than any rate: the easy rows alone already cost
        # more by hand than the whole queue costs with the tool, so no assumed chased rate --
        # not even zero -- inverts the comparison. Found by regenerating README's second
        # report block, which printed "-390.00 min" on a run with 199 easy rows and 1 chased.
        lines.append(_line(
            "Break-even chased rate", "none -- saving is unconditional",
            f"(the {r.easy_rows} rows cleared on sight cost more by hand than this whole "
            f"queue, so no chased rate inverts it)",
        ))
    else:
        lines.append(_line(
            "Break-even chased rate", f"{be:.2f} min",
            f"(below this, doing all {rows} rows by hand is cheaper)",
        ))
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
        # ``minutes_for``, not a second ``.get`` with a second default. This line used to
        # default an unpriced code to 0 while the total above defaulted it to 10.
        minutes = minutes_for(land.reason)
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
    # Phase 12: nothing wrong on the oracle run, so nothing at risk -- printed anyway, per
    # this block's own rule that every counter appears even at zero.
    assert "Value at risk              ₹0.00" in block, block
    # Phase 9: the comparison, with both sides shown. Nothing to chase here, so the by-hand
    # total is 3 rows on sight and the break-even is n/a rather than a division by no rows.
    assert "Est. human time to clear   0 min" in block, block
    assert "(0 min exceptions + 0 min dismissals)" in block, block
    assert "Same batch by hand         6 min" in block, block  # 3 x 2 + 0 x 15
    assert "(3 on sight x 2 min + 0 chased x 15 min)" in block, block
    assert "Time saved                 100.0%" in block, block
    assert "Break-even chased rate     n/a" in block, block
    assert "nothing chased, so no rate to cross" in block, block
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

    # --- the branch this phase exists for: the tool LOSING ------------------
    # Every other fixture in this file has the tool winning, which is how the inverted claim
    # survived: the losing wording had no test because nothing produced a losing run. Both
    # cases below are constructed to lose, because a branch that cannot be reached in a test
    # is a branch nobody has read.
    #
    # (a) One expensive exception. FX_RATE_GAP is priced at 20 min against a chased row's 15,
    #     so a file of nothing but FX gaps costs more to triage than to reconcile by hand.
    losing = score(
        _run((Verdict("C0001", Outcome.EXCEPTION, reason=Reason.FX_RATE_GAP),)),
        _truth((_credit("C0001", ("pay_0001",)),)),
    )
    block = metric_block(losing)
    assert "Est. human time to clear   20 min" in block, block
    assert "Same batch by hand         15 min" in block, block  # 0 x 2 + 1 x 15
    assert "Time saved                 none -- COSTS MORE" in block, block
    assert "takes 5 min longer than by hand" in block, block
    assert "100.0%" not in block, (
        "a losing run must not print a percentage anywhere near the claim -- a reader "
        "skimming for a number is exactly who the old single line misled"
    )
    # The break-even reads as the rate at which this run would break even: at 20 min a chased
    # row it is a wash, and the assumed 15 is below it, which is why this run loses.
    assert "Break-even chased rate     20.00 min" in block, block

    # (b) The dismissal case, and the one the Phase 9 decision made reachable: a file that is
    #     almost all noise. Four rows a person clears on sight cost 8 minutes by hand, while
    #     the tool charges 3 min for each of three dismissals. Nothing is chased, so the
    #     saving is negative with no hard rows at all -- which the old code could not express,
    #     since dismissals cost zero and the comparison never happened.
    noisy = tuple(_credit(f"C{i:04d}", (f"pay_{i:04d}",)) for i in (1,))
    all_noise = score(
        _run((
            _resolved("C0001", ("pay_0001",)),
            Verdict("C0007", Outcome.IGNORED, reason=Reason.NON_GATEWAY_CREDIT),
            Verdict("C0008", Outcome.IGNORED, reason=Reason.NON_GATEWAY_CREDIT),
            Verdict("C0009", Outcome.IGNORED, reason=Reason.NON_GATEWAY_CREDIT),
        )),
        _truth(noisy, noise=("C0007", "C0008", "C0009")),
    )
    block = metric_block(all_noise)
    assert "Est. human time to clear   9 min" in block, block  # 0 exceptions + 3 x 3
    assert "(0 min exceptions + 9 min dismissals)" in block, block
    assert "Same batch by hand         8 min" in block, block  # 4 x 2 + 0 x 15
    assert "Time saved                 none -- COSTS MORE" in block, block
    assert "Break-even chased rate     n/a" in block, block
    # And the pair that makes the point: dismissals are charged now, so this run has a cost
    # even though it raised no exceptions at all. Before Phase 9 it read as free.
    assert all_noise.dismissal_minutes == 9 and all_noise.exception_minutes == 0

    # (c) The **negative crossing point**, which is a winning run rather than a losing one.
    #     Ten rows resolved and one cheap exception: the tool charges 5 min while the ten easy
    #     rows alone cost 20 by hand, so the two lines cross at a chased rate of -15 min/row.
    #     That is arithmetically correct and meaningless as a rate, and it used to print as
    #     "-390.00 min   (below this, doing all 200 rows by hand is cheaper)" -- a claim about
    #     an impossible number, found by regenerating README's second report block rather than
    #     by any assertion here. The saving on such a run is unconditional, which is a stronger
    #     statement than any threshold, so the block now says that instead.
    easy_ids = tuple(f"C{i:04d}" for i in range(1, 11))
    unconditional = score(
        _run(
            tuple(_resolved(cid, (f"pay_{cid[1:]}",)) for cid in easy_ids)
            + (Verdict("C0011", Outcome.EXCEPTION, reason=Reason.PARTIAL_SETTLEMENT_PENDING),)
        ),
        _truth(
            tuple(_credit(cid, (f"pay_{cid[1:]}",)) for cid in easy_ids)
            + (_credit("C0011", ("pay_0011",)),)
        ),
    )
    block = metric_block(unconditional)
    r_uncond = roi(unconditional)
    assert (r_uncond.easy_rows, r_uncond.hard_rows) == (10, 1), r_uncond
    assert r_uncond.tool_minutes == 5 and r_uncond.by_hand_minutes == 35, r_uncond
    # The property that makes the branch necessary: a real, negative crossing point.
    assert r_uncond.break_even_hard_minutes == -15.0, r_uncond.break_even_hard_minutes
    assert "Break-even chased rate     none -- saving is unconditional" in block, block
    assert "10 rows cleared on sight cost more by hand" in block, block
    # No negative minute figure anywhere near the claim -- that is the whole point.
    assert "-15" not in block and "-390" not in block, block
    # And this is a *winning* run, so the saving still prints as a percentage: the branch is
    # about the threshold being meaningless, not about the comparison having gone the wrong way.
    assert "Time saved                 85.7%" in block, block
    assert "COSTS MORE" not in block, block

    # (d) **A wrong match withholds the claim entirely.** Two rows: one resolved to the wrong
    #     payment, one resolved correctly. The queue is empty, so the tool side costs 0 min and
    #     every earlier branch here would print "Time saved 100.0%" -- which is what ``zip``
    #     (35% correctness, 39 wrong matches) and ``saboteur`` (6) actually printed until this
    #     branch existed. A wrong match raises no exception, so it is invisible to a comparison
    #     built on queue minutes, and the best possible figure was being earned by committing
    #     wrongly on every row.
    wrong = score(
        _run((_resolved("C0001", ("pay_0002",)), _resolved("C0002", ("pay_0002",)))),
        _truth((_credit("C0001", ("pay_0001",)), _credit("C0002", ("pay_0002",)))),
    )
    assert wrong.wrong_matches == 1, wrong.cells
    # The value-at-risk line must read the real fixture value here, not ₹0 -- this is the
    # run the line exists for: money already booked wrong, priced.
    assert wrong.wrong_match_value_paise == 100_000, wrong.wrong_match_value_paise
    block = metric_block(wrong)
    r_wrong = roi(wrong)
    # The trap: by the minutes alone this run looks perfect.
    assert r_wrong.tool_minutes == 0 and r_wrong.hard_rows == 0, r_wrong
    assert r_wrong.tool_wins and r_wrong.saved_fraction == 1.0, r_wrong
    # ...and no percentage is printed anywhere near the claim, for the reason (a) gives: a
    # reader skimming for a number must not find one here.
    assert "Time saved                 not claimable -- 1 wrong match(es)" in block, block
    assert "100.0%   (" not in block.split("Time saved")[1], block
    assert "not what this run left a person to do" in block, block
    assert "Value at risk              ₹1,000.00" in block, block
    # The withholding is keyed on wrong matches, not on the queue being empty: the empty-queue
    # runs above still print their percentage.
    assert "Time saved                 100.0%" in metric_block(unconditional_zero_check := score(
        _run((_resolved("C0001", ("pay_0001",)),)),
        _truth((_credit("C0001", ("pay_0001",)),)),
    )), "a clean run with an empty queue must still be able to claim its saving"
    assert unconditional_zero_check.wrong_matches == 0

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
