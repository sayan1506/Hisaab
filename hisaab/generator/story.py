"""The forward build: sales -> settlements -> bank credits.

Steps 4 through 7 of .plan/phase1.md. Everything here happens in memory; nothing
touches disk until ``emit.py``, and ``invariants.check_story`` runs in between.

Clean mode (Phase 1): one payment -> one settlement -> one bank credit, with
``gross == net == credit`` and identical dates. Zero fees, not "fees applied
consistently" -- so the first time an amount fails to match in Phase 3, there is
exactly one possible cause instead of two.

Two ordering properties this module is responsible for, and they are the whole
reason step 7 exists:

  * **ID numbering carries no information.** Settlements are shuffled and credits
    are re-sorted *before* their IDs are assigned, so ``pay_0005`` does not map to
    ``setl_0005`` or ``C0005``. If the numbering is the answer key, Phase 3 is
    invalid and you will not notice until batching collapses.
  * **Row position carries no information within a date.** Bank rows are sorted
    by (value_date, amount, random tiebreak). The tiebreak matters: Python's sort
    is stable, so without it, same-date same-amount rows keep generation order --
    which *is* payment order (trap 2).

Cross-date correspondence is a different matter and is deliberately preserved:
in clean mode ``value_date == captured_at`` date by construction, and that is the
intended Tier 1 signal, not a leak. See ``invariants.py`` I8 for how the two are
told apart.
"""

from __future__ import annotations

import dataclasses
import random
from datetime import date, datetime

from ..common import ids
from ..common.bizdays import BusinessCalendar
from ..common.reasons import Reason
from .config import (
    AMOUNT_BANDS,
    BANK_CHANNELS,
    CAPTURE_HOUR_MAX,
    CAPTURE_HOUR_MIN,
    COUNTERPARTY,
    COUNTERPARTY_SHORT,
    COUNTERPARTY_SPACED,
    IST,
    NARRATION_TEMPLATES,
    PAYMENT_METHODS,
    WHOLE_RUPEE_PERCENT,
    GenConfig,
)
from ..common.money import RUPEE, mul_bps, rupees
from .model import Credit, Decomposition, Payment, Settlement, Story
from .rng import substream, weighted_choice

#: UTR tails are 4 digits, matching the ``XXXX4471`` shape in Appendix A. Drawn
#: without replacement in Phase 1 so the tail is a genuinely unique link; making
#: it patchy or colliding is ``--utr-patchy``'s job in Phase 8.
TAIL_MIN, TAIL_MAX = 1000, 9999

#: How many distinct tails exist at all: 9,000. This is a **hard ceiling on ``n``**, because
#: the tails are drawn without replacement. Named as a constant because it was found the
#: unpleasant way -- ``n=16000`` used to raise ``IndexError: list index out of range`` from
#: the ``utr=f"XXXX{tails[seq - 1]}"`` line, a message that names neither ``n`` nor the tail
#: space, from a loop that is not the thing at fault. ``_draw_tails`` capped its pool at
#: 9,000 and the settlement loop kept indexing to ``n``.
#:
#: Not fixed by widening the tails: ``XXXX4471`` is the shape Appendix A specifies, so the
#: width is a spec constraint and not a tuning knob. Fixed by refusing the config, which is
#: the honest answer -- this generator cannot produce more than 9,000 records with unique
#: UTRs, and that is a property worth stating rather than discovering at row 9,001.
TAIL_SPACE = TAIL_MAX - TAIL_MIN + 1

#: Spare tails the credit fixup in ``build`` draws from. The ceiling has to leave room for
#: these too, or a large ``n`` trades the ``IndexError`` for a ``StopIteration`` -- equally
#: obscure, and arriving from a different line.
TAIL_SPARES = 32


def _pick_band(rng: random.Random) -> tuple[int, int]:
    """One draw -> (min_rupees, max_rupees). Weights sum to 100 (asserted in config)."""
    ticket = rng.randrange(100)
    upto = 0
    for lo, hi, weight in AMOUNT_BANDS:
        upto += weight
        if ticket < upto:
            return lo, hi
    raise AssertionError("unreachable: AMOUNT_BANDS weights did not cover the ticket")


def _draw_gross_paise(rng: random.Random) -> int:
    """A long-tailed gross amount, in integer paise. Exactly four draws.

    Fixed draw count per record (rng.py rule 2): the paise part is drawn
    unconditionally and then used conditionally, so the whole-rupee decision does
    not shift the stream for subsequent records.
    """
    lo, hi = _pick_band(rng)
    whole_rupees = rng.randint(lo, hi)
    is_whole = rng.randrange(100) < WHOLE_RUPEE_PERCENT
    paise_part = rng.randrange(RUPEE)
    return rupees(whole_rupees) + (0 if is_whole else paise_part)


def _draw_capture_time(rng: random.Random, business_days: list[date]) -> datetime:
    """A plausible capture moment: a business day, business hours, IST. Four draws.

    Hours are clamped to 09:00-21:00 IST (decision #6 / trap 3) so the UTC
    calendar date always equals the IST calendar date. Without the clamp, a
    late-evening capture serialises to the previous UTC day and Phase 4 blames its
    own delay model for a gap the timezone created.
    """
    day = business_days[rng.randrange(len(business_days))]
    hour = rng.randint(CAPTURE_HOUR_MIN, CAPTURE_HOUR_MAX)
    minute = rng.randrange(60)
    second = rng.randrange(60)
    return datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=IST)


def _narration(rng: random.Random, tail: int, styles: int) -> str:
    """Assemble a bank narration from parts. Two draws.

    Deliberately *not* a single format: real statements vary by bank and channel,
    which is what gives Phase 3's parser and Phase 10's LLM real work. Style
    variance changes no amount and no date, so clean mode stays clean where it
    counts. ``--narration-styles 1`` for a sterile file while debugging.
    """
    template = NARRATION_TEMPLATES[rng.randrange(styles)]
    channel = weighted_choice(rng, BANK_CHANNELS)
    return template.format(
        channel=channel,
        counterparty=COUNTERPARTY,
        counterparty_spaced=COUNTERPARTY_SPACED,
        counterparty_short=COUNTERPARTY_SHORT,
        tail=tail,
    )


def _deductions(cfg: GenConfig, gross_paise: int, method: str) -> tuple[int, int]:
    """``(fee_paise, gst_paise)`` for one payment. ``(0, 0)`` unless ``--fees`` is on.

    Two rates, applied in one order and never the other: the fee is a share of the
    **gross**, and GST is a share of the **fee** -- not of the gross. Getting that
    backwards inflates GST by more than fifty times and is the single easiest arithmetic
    error available here, which is why the composition lives in one function that both
    the invariant and the truth decomposition read.

    Half-up at the paisa via ``mul_bps``, and Phase 4 is that function's first caller: it
    was written and tested in Phase 1 precisely so the rounding rule existed before two
    components could disagree about it. The rule is *declared* (ASSUMPTIONS.md), which is
    what lets the matcher re-derive these numbers instead of reading them off
    ``settlements.csv``.

    ``pos_upi`` is 0 bps, so a POS payment settles at its gross and its residual is zero
    even under ``--fees``. That is a property of the rate table, not a bug -- and it is why
    "the residual moved" has to be measured per method rather than in aggregate. Note that
    the free rail is POS specifically: standard-PG UPI is priced at 200 bps like every other
    domestic instrument, because zero *MDR* does not mean zero *fee* (see ``FeeConfig``).
    The zero-rated share is now ~6% of rows rather than the ~36% an earlier table produced.

    Nothing below names a method. The zero branch is reached by rate, so correcting a rate
    moves which rows take it without editing this function -- which is the only reason the
    correction above was a one-line change here.
    """
    if not cfg.flags.fees:
        return 0, 0
    fee = mul_bps(gross_paise, cfg.fees.fee_bps(method))
    return fee, mul_bps(fee, cfg.fees.gst_bps)


def _unique_amount(taken: set[tuple[date, int]], day: date, amount: int) -> int:
    """Nudge ``amount`` up by whole paise until ``(day, amount)`` is unused.

    Clean mode must be resolvable at 100% -- that is row 1 of the mess dial. Two
    payments sharing a date *and* an amount would be genuinely indistinguishable
    from date+amount alone, so the honest verdict would be an abstention and clean
    mode could not reach 100%. That case is real and it is planted deliberately by
    ``--dup-amounts`` in Phase 4b; it must not appear here by accident.

    A nudge rather than a redraw, so the draw count per record stays fixed.

    **This function is also what guarantees a planted pair is exactly a pair.** Because
    every drafted ``(date, gross)`` is unique when ``_plant_dup_pairs`` overwrites one
    member, no third payment can be holding the value it copies -- so the collision it
    creates has cardinality 2, and the invariant that counts planted pairs can assert it.
    """
    while (day, amount) in taken:
        amount += 1
    taken.add((day, amount))
    return amount


def _plant_dup_pairs(
    rng: random.Random, drafts: list[tuple[datetime, int, str]], pairs: int
) -> set[tuple[date, int]]:
    """``--dup-amounts``: force ``pairs`` disjoint draft pairs to collide. Mutates ``drafts``.

    Returns the ``(capture_date, gross_paise)`` keys it made collide, so ``build`` can find
    the planted rows **by value** rather than by tracking indices through the settlement
    shuffle. Every returned key has exactly two holders, per ``_unique_amount``.

    Each pair's second member copies the first's **capture date, gross and method**. Those
    three carry the collision all the way down: the same capture date gives both settlements
    the same ``settled_on``, and the same gross with the same method derives the same fee, so
    both settlements land on one ``net_paise`` and both credits on one
    ``(value_date, amount_paise)``.

    **The time of day is deliberately not copied.** What the matcher sees is the credit's
    date, which comes from the IST calendar date; copying the whole timestamp would make the
    two payments byte-identical but for their ids, which is a stronger and less realistic
    claim than this flag needs. Two customers buying the same item on the same afternoon is
    ordinary; two doing it in the same second is a coincidence a reader would query.

    Drawn from the ``dup`` substream, so planting perturbs none of the Phase 1 streams and
    the clean-vs-dup diff at one seed stays readable -- the property that made
    ASSUMPTIONS.md #24b's cross-run comparison valid.

    Sharing the UTR is the other half of the job and cannot happen here, because tails are
    assigned to settlements. ``build`` does it, and ``config.dup_pairs`` records why it is
    the load-bearing half.
    """
    chosen = rng.sample(range(len(drafts)), 2 * pairs)
    planted: set[tuple[date, int]] = set()
    for i in range(pairs):
        source, target = chosen[2 * i], chosen[2 * i + 1]
        captured_at, gross, method = drafts[source]
        keep = drafts[target][0]
        drafts[target] = (
            datetime(
                captured_at.year, captured_at.month, captured_at.day,
                keep.hour, keep.minute, keep.second, tzinfo=captured_at.tzinfo,
            ),
            gross,
            method,
        )
        planted.add((captured_at.date(), gross))
    return planted


def _draw_tails(rng: random.Random, n: int) -> list[int]:
    """``n`` distinct 4-digit UTR tails, plus spares for the fixup in ``build``.

    Refuses an ``n`` the tail space cannot cover, rather than returning a short list for
    ``build`` to run off the end of. See ``TAIL_SPACE`` for why the fix is a refusal and not
    a wider tail.
    """
    if n + TAIL_SPARES > TAIL_SPACE:
        raise ValueError(
            f"n={n} exceeds what unique 4-digit UTR tails can "
            f"cover: {TAIL_SPACE} tails exist, and {TAIL_SPARES} are held back as spares for "
            f"the credit fixup, so n must be <= {TAIL_SPACE - TAIL_SPARES}. The tail is a "
            f"unique link in clean mode (--utr-patchy is Phase 8), and Appendix A fixes the "
            f"XXXX9999 shape, so widening it is not available."
        )
    pool_size = min(TAIL_SPACE, max(n * 2, n + TAIL_SPARES))
    return rng.sample(range(TAIL_MIN, TAIL_MAX + 1), pool_size)


def build(cfg: GenConfig, calendar: BusinessCalendar | None = None) -> Story:
    """Generate one month's story. Pure: same config in, same story out."""
    cal = calendar or BusinessCalendar()
    business_days = cal.business_days_in_month(cfg.year, cfg.month)
    assert business_days, f"no business days in {cfg.month_label}"

    rng_amounts = substream(cfg.seed, "amounts")
    rng_times = substream(cfg.seed, "timestamps")
    rng_methods = substream(cfg.seed, "methods")
    rng_utr = substream(cfg.seed, "utr")
    rng_narration = substream(cfg.seed, "narration")
    rng_settlements = substream(cfg.seed, "settlement_order")
    rng_bank = substream(cfg.seed, "bank_order")
    rng_dup = substream(cfg.seed, "dup")

    # --- Step 4: invent the payments -------------------------------------
    # Drawn first, then sorted by capture time, then numbered -- a real gateway
    # export is ordered by capture time, so pay_0001 being the earliest payment
    # is realistic and reveals nothing about settlement or bank ordering.
    drafts: list[tuple[datetime, int, str]] = []
    taken: set[tuple[date, int]] = set()
    for _ in range(cfg.n):
        captured_at = _draw_capture_time(rng_times, business_days)
        gross = _draw_gross_paise(rng_amounts)
        method = weighted_choice(rng_methods, PAYMENT_METHODS)
        gross = _unique_amount(taken, captured_at.date(), gross)
        drafts.append((captured_at, gross, method))

    # --- Step 4b: --dup-amounts plants the collisions ---------------------
    # After every draw, so planting consumes no Phase 1 randomness and the clean-vs-dup
    # diff at one seed stays readable. Before the sort, so the two members are ordered by
    # their own capture times like any other pair -- they keep their own times of day, so
    # they do not land adjacent in payments.csv and their ids are not consecutive.
    planted_keys: set[tuple[date, int]] = set()
    if cfg.flags.dup_amounts:
        planted_keys = _plant_dup_pairs(rng_dup, drafts, cfg.dup_pairs)

    drafts.sort(key=lambda d: (d[0], d[1], d[2]))
    payments = [
        Payment(
            payment_id=ids.payment_id(i),
            order_id=ids.order_id(i),
            captured_at=captured_at,
            gross_paise=gross,
            method=method,
        )
        for i, (captured_at, gross, method) in enumerate(drafts, start=1)
    ]

    # --- Step 5: derive settlements, 1:1 ---------------------------------
    # net == gross and every deduction is zero: clean mode has no fees at all.
    # settled_on is ``cfg.delay_days`` business days after the IST capture date --
    # which is the capture date itself in clean mode, since the delay is 0 there and
    # every capture lands on a business day. payment_ids is a one-element *list*
    # (decision #10) so Phase 5's batching is a change of contents, not a change of type.
    tails = _draw_tails(rng_utr, cfg.n)
    order = list(range(cfg.n))
    rng_settlements.shuffle(order)  # step 7: numbering must carry no information

    settlements: list[Settlement] = []
    for seq, payment_index in enumerate(order, start=1):
        p = payments[payment_index]
        # net = gross - fee - GST. Both are 0 without --fees, so this is ``net == gross``
        # in clean mode and the expression stops being trivial exactly when the flag
        # turns on. tds_paise stays 0: TDS is dial row 7, Phase 6 (decision D7) -- two
        # gross/net wedges at once means two hypotheses per failing row.
        fee, gst = _deductions(cfg, p.gross_paise, p.method)
        settlements.append(
            Settlement(
                settlement_id=ids.settlement_id(seq),
                settled_on=cal.add_business_days(p.business_date, cfg.delay_days),
                payment_ids=[p.payment_id],
                net_paise=p.gross_paise - fee - gst,
                fee_paise=fee,
                gst_paise=gst,
                tds_paise=0,
                utr=f"XXXX{tails[seq - 1]}",
            )
        )

    # --- Step 5b: --dup-amounts shares one UTR across each planted pair ---
    # The load-bearing half of the flag, and the reason it is not "generator-side only" in
    # the trivial sense. Measured before it was written, and re-run on every acceptance
    # run by gate 11 of ``tools/acceptance.py``: a tail-only strategy reading no date and no
    # amount resolves 60/60, 200/200 and 1000/1000 credits *correctly* on every dev seed,
    # because ``_draw_tails`` samples without replacement. So a pair colliding on
    # ``(date, amount)`` with distinct tails stays separable by exhaustive narration
    # matching -- the flag would not test the capability its name claims, and
    # ``resolvable=False`` would be a false statement about the data. This is finrecon's
    # recorded failure exactly: its "AI showcase" tier fell to narration enumeration.
    #
    # Iterating ``settlements`` in list order rather than iterating ``planted_keys``:
    # ``date.__hash__`` goes through ``bytes``, which PYTHONHASHSEED randomises, so a set
    # of ``(date, int)`` has no stable iteration order and ``tools/repro_check.py`` would
    # catch it as a cross-process byte difference. The set is only ever *looked up*.
    planted_settlement_ids: set[str] = set()
    if planted_keys:
        first_utr: dict[tuple[date, int], str] = {}
        for i, s in enumerate(settlements):
            p = payments[order[i]]
            key = (p.business_date, p.gross_paise)
            if key not in planted_keys:
                continue
            if key in first_utr:
                settlements[i] = dataclasses.replace(s, utr=first_utr[key])
            else:
                first_utr[key] = s.utr
            planted_settlement_ids.add(settlements[i].settlement_id)

    # --- Step 6: derive bank credits, 1:1 --------------------------------
    # Same date, same amount, and a narration assembled from parts. Four fields
    # reach the CSV; the linkage below exists only in memory and in truth.json.
    by_payment = {p.payment_id: p for p in payments}
    spare_tails = iter(tails[cfg.n:])
    # The echo fixup has to be *memoised*, or step 5b's work is undone here: a planted pair
    # shares a tail and a net, so if that tail echoes the amount both members enter the loop
    # below and each would draw its own spare -- handing the pair two different narration
    # tails and separating it again. Keyed on ``(tail, net)`` because that pair is what the
    # decision depends on. Clean mode is byte-identical: every tail is distinct there, so
    # every lookup misses and the loop runs exactly as before.
    fixed_tails: dict[tuple[int, int], int] = {}
    drafted_credits: list[tuple[date, int, float, Settlement, str]] = []
    for s in settlements:
        original = int(s.utr.removeprefix("XXXX"))
        cached = fixed_tails.get((original, s.net_paise))
        if cached is None:
            tail = original
            # A tail that happens to equal its own credit's rupee figure would put the
            # amount into the narration, handing the matcher a free join. Swap it for
            # an unused tail rather than let invariant I7 fail on a rare seed.
            while _tail_echoes_amount(tail, s.net_paise):
                tail = next(spare_tails)
            fixed_tails[(original, s.net_paise)] = tail
        else:
            tail = cached
        # The narration template and channel are still drawn per credit, so a planted pair's
        # two rows agree on their parsed ``ref_tail`` and **may or may not** be byte-identical
        # overall: measured across seeds 1/2/3/42 x n=60/200, the two rows come out identical
        # in roughly half of runs and differ by template or channel in the rest. Both
        # outcomes are correct and neither is tuned for. What makes the pair unresolvable is
        # that the two *settlements* agree on every field that could link a credit to one of
        # them -- date, amount and UTR -- so a byte-identical pair is merely the same
        # ambiguity with less surface texture, not a stronger or weaker plant.
        narration = _narration(rng_narration, tail, cfg.narration_styles)
        # ``value_date`` is sourced from ``settled_on``, **not** from the payment's
        # capture date. Until Phase 4 this read ``p.business_date``, which was an
        # artifact rather than a model: a real bank credit follows the settlement that
        # paid it, and nothing about the capture date reaches the bank. The old sourcing
        # was invisible in clean mode (the two dates are equal there) and would have
        # broken the date window for the wrong reason the moment a settlement delay
        # arrived -- opening a gap the matcher would have had to widen its window to
        # absorb, with no real-world delay behind it. See .plan/phase4.md (a).
        #
        # Trap 2: the tiebreak is what stops a stable sort from preserving
        # generation order (== payment order) among same-date same-amount rows.
        value_date = cal.add_business_days(s.settled_on, cfg.lag_days)
        drafted_credits.append((value_date, s.net_paise, rng_bank.random(), s, narration))

    drafted_credits.sort(key=lambda c: (c[0], c[1], c[2]))

    credits: list[Credit] = []
    for seq, (value_date, amount, _tiebreak, s, narration) in enumerate(drafted_credits, start=1):
        # The decomposition is the *answer key's* arithmetic, and it must close to the
        # credit exactly: gross - fee - gst == amount. Phase 1 could write
        # ``Decomposition(gross_paise=amount)`` because gross, net and credit were one
        # number; under --fees they are three, so the gross is summed from the payments
        # (a sum over ``payment_ids``, which is already the Phase 5 batching shape) and
        # the deductions come from the settlement that declared them.
        #
        # ``Decomposition.expected_credit_paise`` recomputes the subtraction, and
        # invariant I4 asserts it equals ``amount_paise`` -- so an arithmetic slip here
        # fails before anything reaches disk rather than becoming a truth file that
        # disagrees with its own CSVs.
        gross_total = sum(by_payment[pid].gross_paise for pid in s.payment_ids)
        # A planted row is unresolvable *from the inputs*, and the answer key says so. The
        # reason comes from ``common.reasons.Reason`` rather than a hand-typed literal: the
        # generator's intent and the matcher's verdict have to be drawn from one vocabulary,
        # or "correct abstention" becomes a judgement call instead of a count
        # (``reasons.py``'s opening docstring).
        #
        # Truth still records *which* settlement really paid this credit. That is not a
        # contradiction: the answer key may know things the inputs do not contain, and it is
        # what lets the scorer tell an honest abstention from a lucky guess -- a matcher that
        # commits to one member has even odds of naming the right set, and
        # ``metrics._classify`` grades that as LUCKY_GUESS rather than CORRECT precisely
        # because the inputs could not have justified it.
        planted = s.settlement_id in planted_settlement_ids
        credits.append(
            Credit(
                credit_id=ids.credit_id(seq),
                value_date=value_date,
                amount_paise=amount,
                narration=narration,
                settlement_ids=[s.settlement_id],
                payment_ids=list(s.payment_ids),
                decomposition=Decomposition(
                    gross_paise=gross_total,
                    fee_paise=s.fee_paise,
                    gst_paise=s.gst_paise,
                ),
                refunds_netted=[],
                reserve_held_paise=0,
                resolvable=not planted,
                reason=str(Reason.AMBIGUOUS_DUPLICATE_AMOUNT) if planted else None,
                note=(
                    f"planted unresolvable: another credit shares this value_date "
                    f"({value_date.isoformat()}), this amount ({amount}p) and this UTR "
                    f"({s.utr}), so no field in the three input files separates them. "
                    f"Resolving it requires information outside these files; the only "
                    f"correct verdict is an abstention."
                    if planted
                    else None
                ),
            )
        )

    # Phase 1 has no orphans, no noise and no refunds -- the empty lists are the
    # point: Phase 7 fills them and the scorer already knows how to read them.
    return Story(payments=payments, settlements=settlements, credits=credits, refunds=[])


def _tail_echoes_amount(tail: int, amount_paise: int) -> bool:
    """Would this tail put the credit's own amount into the narration?"""
    tail_str = str(tail)
    whole, frac = divmod(amount_paise, RUPEE)
    return tail_str in {str(amount_paise), str(whole)} or (
        frac == 0 and tail_str == str(whole)
    )


if __name__ == "__main__":
    cfg = GenConfig(seed=42, n=60)
    story = build(cfg)

    assert story.counts() == {"payments": 60, "settlements": 60, "credits": 60,
                              "refunds": 0, "noise_rows": 0}
    # Clean mode: the three totals are the same number.
    assert story.total_gross_paise() == story.total_net_paise() == story.total_credited_paise()
    # Determinism.
    assert [c.csv_row() for c in build(cfg).credits] == [c.csv_row() for c in story.credits]
    # A different seed gives different data of the same shape.
    other = build(GenConfig(seed=43, n=60))
    assert other.counts() == story.counts()
    assert [c.csv_row() for c in other.credits] != [c.csv_row() for c in story.credits]
    # Payments are capture-ordered; credits are date-then-amount ordered.
    assert story.payments == sorted(story.payments, key=lambda p: p.captured_at)
    assert [(c.value_date, c.amount_paise) for c in story.credits] == sorted(
        (c.value_date, c.amount_paise) for c in story.credits
    )
    # Clean mode must be resolvable at 100%: (date, amount) uniquely identifies a credit.
    keys = [(c.value_date, c.amount_paise) for c in story.credits]
    assert len(set(keys)) == len(keys), "duplicate (date, amount) makes clean mode unresolvable"
    # Every deduction is zero and net == gross, per settlement.
    by_pay = {p.payment_id: p for p in story.payments}
    for s in story.settlements:
        assert (s.fee_paise, s.gst_paise, s.tds_paise) == (0, 0, 0)
        assert s.net_paise == by_pay[s.payment_ids[0]].gross_paise
        assert s.settled_on == by_pay[s.payment_ids[0]].business_date

    # --- step 4: the fee model ------------------------------------------------
    # What is asserted here is the *composition* -- which base each rate applies to, and
    # that the flag gates the whole thing. The rates themselves are assumptions
    # (ASSUMPTIONS.md #5-#9); re-deriving them from cfg.fees and comparing would only
    # assert that the generator agrees with itself.
    from .config import MessFlags

    fee_cfg = GenConfig(seed=42, n=200, flags=MessFlags(fees=True))
    fee_story = build(fee_cfg)
    assert fee_story.counts()["payments"] == 200
    # The three totals are now three different numbers, and the wedge is exactly the
    # deductions. This is the assertion that replaced ``gross == net == credited``.
    deducted = sum(s.fee_paise + s.gst_paise + s.tds_paise for s in fee_story.settlements)
    assert deducted > 0, "--fees deducted nothing at all"
    assert fee_story.total_gross_paise() - deducted == fee_story.total_net_paise()
    assert fee_story.total_net_paise() == fee_story.total_credited_paise()
    assert fee_story.total_net_paise() < fee_story.total_gross_paise()

    fee_by_pay = {p.payment_id: p for p in fee_story.payments}
    zero_rated = {m for m, bps in fee_cfg.fees.fee_bps_by_method.items() if bps == 0}
    charged = 0
    for s in fee_story.settlements:
        p = fee_by_pay[s.payment_ids[0]]
        assert s.net_paise == p.gross_paise - s.fee_paise - s.gst_paise
        assert s.tds_paise == 0, "TDS is Phase 6 (D7): one gross/net wedge at a time"
        # GST sits on the fee, not on the gross. At 18% on the fee it cannot approach the
        # fee itself, let alone exceed it -- whereas 18% of the *gross* would dwarf a
        # 2% fee by roughly nine times. That is the check, and it is the reason the
        # composition lives in one function.
        assert s.gst_paise <= s.fee_paise
        if p.method in zero_rated:
            assert (s.fee_paise, s.gst_paise) == (0, 0), f"{p.method} is zero-rated"
            assert s.net_paise == p.gross_paise, "a zero-rated method settles at its gross"
        else:
            assert s.fee_paise > 0, f"{p.method} at {fee_cfg.fees.fee_bps(p.method)}bps took nothing"
            assert s.gst_paise > 0
            assert s.gst_paise * 5 < s.fee_paise * 10, (
                f"gst {s.gst_paise} vs fee {s.fee_paise}: GST looks like a share of the "
                f"gross rather than of the fee"
            )
            charged += 1
    assert charged, "no settlement carried a fee -- the rate table may be all zeros"
    # A zero-rated method must actually appear in a 200-record sample, or the branch above
    # is asserting nothing and the per-method residual reasoning is untested.
    assert any(p.method in zero_rated for p in fee_story.payments), (
        "no zero-rated payment in 200 records -- the zero-fee branch went untested"
    )
    # Every credit's decomposition closes to its own amount, per credit and not just in
    # total: two compensating errors cancel in a sum.
    for c in fee_story.credits:
        assert c.decomposition.expected_credit_paise == c.amount_paise, c.credit_id
        assert c.decomposition.gross_paise == sum(
            fee_by_pay[pid].gross_paise for pid in c.payment_ids
        )
    # Determinism holds under the flag, and --fees alone moves no date.
    assert [c.csv_row() for c in build(fee_cfg).credits] == [
        c.csv_row() for c in fee_story.credits
    ]
    for s in fee_story.settlements:
        assert s.settled_on == fee_by_pay[s.payment_ids[0]].business_date
    # Fees change amounts and nothing else: same payments, same dates, same methods as the
    # clean run at the same seed. If this ever fails, --fees has reached into the draw
    # order and clean mode is no longer the regression check it is supposed to be.
    clean_200 = build(GenConfig(seed=42, n=200))
    assert [p.csv_row() for p in clean_200.payments] == [
        p.csv_row() for p in fee_story.payments
    ], "--fees perturbed the payment stream"
    assert [c.value_date for c in clean_200.credits] == [
        c.value_date for c in fee_story.credits
    ], "--fees moved a value_date"
    # The long tail is real, not a uniform spread.
    grosses = sorted(p.gross_paise for p in story.payments)
    assert grosses[-1] > 10 * grosses[len(grosses) // 2], "distribution is not long-tailed"
    assert any(g % RUPEE for g in grosses), "no amount carries paise"
    # n=1 and a single-narration-style run must both work.
    assert len(build(GenConfig(seed=7, n=1)).credits) == 1
    assert len(build(GenConfig(seed=7, n=12, narration_styles=1)).credits) == 12

    # --- the UTR tail ceiling ---------------------------------------------
    # Tails are drawn without replacement from a 9,000-value space, so n has a hard upper
    # bound. Asserted at the boundary and one past it, because the bug this replaces was an
    # off-by-nothing -- the pool was silently capped while the settlement loop kept indexing
    # to n, so it failed with an IndexError from a line about UTRs.
    MAX_N = TAIL_SPACE - TAIL_SPARES
    try:
        _draw_tails(random.Random(0), MAX_N + 1)
    except ValueError as exc:
        assert str(MAX_N) in str(exc), "the refusal must name the limit, not just refuse"
        assert "Phase 8" in str(exc), "and point at the flag that relaxes it"
    else:
        raise AssertionError(f"n={MAX_N + 1} exceeds the tail space and must be refused")
    # The boundary itself must actually *work*, and through the full build rather than
    # through _draw_tails alone: the guard's arithmetic is a claim about what build consumes
    # (n for settlements, then spares for the credit fixup), and only build tests that.
    edge = build(GenConfig(seed=7, n=MAX_N))
    assert len(edge.settlements) == MAX_N
    assert len({s.utr for s in edge.settlements}) == MAX_N, "tails must stay unique at the edge"
    print(f"story.py self-check ok  ({cfg.n} records, seed {cfg.seed})")
