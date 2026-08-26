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

import random
from datetime import date, datetime

from ..common import ids
from ..common.bizdays import BusinessCalendar
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
from ..common.money import RUPEE, rupees
from .model import Credit, Decomposition, Payment, Settlement, Story
from .rng import substream, weighted_choice

#: UTR tails are 4 digits, matching the ``XXXX4471`` shape in Appendix A. Drawn
#: without replacement in Phase 1 so the tail is a genuinely unique link; making
#: it patchy or colliding is ``--utr-patchy``'s job in Phase 8.
TAIL_MIN, TAIL_MAX = 1000, 9999


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


def _unique_amount(taken: set[tuple[date, int]], day: date, amount: int) -> int:
    """Nudge ``amount`` up by whole paise until ``(day, amount)`` is unused.

    Clean mode must be resolvable at 100% -- that is row 1 of the mess dial. Two
    payments sharing a date *and* an amount would be genuinely indistinguishable
    from date+amount alone, so the honest verdict would be an abstention and clean
    mode could not reach 100%. That case is real and it is planted deliberately by
    ``--dup-amounts`` in Phase 8; it must not appear here by accident.

    A nudge rather than a redraw, so the draw count per record stays fixed.
    """
    while (day, amount) in taken:
        amount += 1
    taken.add((day, amount))
    return amount


def _draw_tails(rng: random.Random, n: int) -> list[int]:
    """``n`` distinct 4-digit UTR tails, plus spares for the fixup in ``build``."""
    pool_size = min(TAIL_MAX - TAIL_MIN + 1, max(n * 2, n + 32))
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
    # settled_on is the IST capture date -- same day, because --settlement-delay
    # is off. payment_ids is a one-element *list* (decision #10) so Phase 5's
    # batching is a change of contents, not a change of type.
    tails = _draw_tails(rng_utr, cfg.n)
    order = list(range(cfg.n))
    rng_settlements.shuffle(order)  # step 7: numbering must carry no information

    settlements: list[Settlement] = []
    for seq, payment_index in enumerate(order, start=1):
        p = payments[payment_index]
        settlements.append(
            Settlement(
                settlement_id=ids.settlement_id(seq),
                settled_on=p.business_date,
                payment_ids=[p.payment_id],
                net_paise=p.gross_paise,
                fee_paise=0,
                gst_paise=0,
                tds_paise=0,
                utr=f"XXXX{tails[seq - 1]}",
            )
        )

    # --- Step 6: derive bank credits, 1:1 --------------------------------
    # Same date, same amount, and a narration assembled from parts. Four fields
    # reach the CSV; the linkage below exists only in memory and in truth.json.
    by_payment = {p.payment_id: p for p in payments}
    spare_tails = iter(tails[cfg.n:])
    drafted_credits: list[tuple[date, int, float, Settlement, str]] = []
    for s in settlements:
        p = by_payment[s.payment_ids[0]]
        tail = int(s.utr.removeprefix("XXXX"))
        # A tail that happens to equal its own credit's rupee figure would put the
        # amount into the narration, handing the matcher a free join. Swap it for
        # an unused tail rather than let invariant I7 fail on a rare seed.
        while _tail_echoes_amount(tail, s.net_paise):
            tail = next(spare_tails)
        narration = _narration(rng_narration, tail, cfg.narration_styles)
        # Trap 2: the tiebreak is what stops a stable sort from preserving
        # generation order (== payment order) among same-date same-amount rows.
        drafted_credits.append((p.business_date, s.net_paise, rng_bank.random(), s, narration))

    drafted_credits.sort(key=lambda c: (c[0], c[1], c[2]))

    credits: list[Credit] = []
    for seq, (value_date, amount, _tiebreak, s, narration) in enumerate(drafted_credits, start=1):
        credits.append(
            Credit(
                credit_id=ids.credit_id(seq),
                value_date=value_date,
                amount_paise=amount,
                narration=narration,
                settlement_ids=[s.settlement_id],
                payment_ids=list(s.payment_ids),
                decomposition=Decomposition(gross_paise=amount),
                refunds_netted=[],
                reserve_held_paise=0,
                resolvable=True,
                reason=None,
                note=None,
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
    # The long tail is real, not a uniform spread.
    grosses = sorted(p.gross_paise for p in story.payments)
    assert grosses[-1] > 10 * grosses[len(grosses) // 2], "distribution is not long-tailed"
    assert any(g % RUPEE for g in grosses), "no amount carries paise"
    # n=1 and a single-narration-style run must both work.
    assert len(build(GenConfig(seed=7, n=1)).credits) == 1
    assert len(build(GenConfig(seed=7, n=12, narration_styles=1)).credits) == 12
    print(f"story.py self-check ok  ({cfg.n} records, seed {cfg.seed})")
