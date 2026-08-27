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
from collections import defaultdict
from datetime import date, datetime

from ..common import ids
from ..common.bizdays import BusinessCalendar
from ..common.reasons import Reason
from .config import (
    AMOUNT_BANDS,
    BANK_CHANNELS,
    BATCH_SIZE_WEIGHTS,
    LATE_REPORT_SHARE,
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


def _batch_net(
    cfg: GenConfig, grosses: list[int], methods: list[str], members: list[int]
) -> int:
    """A batch's ``net_paise``: per member, per method, each rounded at the paisa, summed.

    The one place this arithmetic is spelled, so the uniqueness nudge below and the
    ``Settlement`` built in ``build`` cannot disagree about what a batch nets. Decision 7 --
    and see ``matcher/fees.derive``, which does the same sum independently and is what makes
    the residual a test rather than a tautology.
    """
    total = 0
    for i in members:
        fee, gst = _deductions(cfg, grosses[i], methods[i])
        total += grosses[i] - fee - gst
    return total


def _make_nets_unique(
    cfg: GenConfig,
    grosses: list[int],
    methods: list[str],
    capture_dates: list[date],
    settled_on_of: list[date],
    groups: list[list[int]],
    taken_gross: set[tuple[date, int]],
) -> int:
    """Nudge grosses until no two settlements share a ``(settled_on, net)``. Returns nudges.

    **``_unique_amount``'s guarantee does not survive summation, and this restores it.** That
    function keeps every ``(capture_date, gross)`` distinct, which is what makes a 1:1 credit
    resolvable from date and amount at all. A batch's net is a *sum* of member nets, and no
    invariant compared that derived value to anything -- so two settlements on one date can
    land on one net. ASSUMPTIONS.md #24c named batching as the channel that would widen this
    and it does.

    **Measured before this existed, seeds 1-5 plus 42 at n = 60/200/1000:** zero colliding
    pairs at n=60 and n=200, and **one pair on 3 of 6 seeds at n=1000** (seed 3 collided at
    454,300p on 2026-08-21: one payment against a two-payment batch summing to the same net).
    Step 4 of ``.plan/phase5.md`` pre-committed the response before that number was visible,
    and this is it. The alternative on the table -- relaxing ``I3.unique_date_amount`` to D6's
    "mark every colliding pair unresolvable" -- was refused in advance and stays refused: at
    n <= 1000 a collision is a generator artifact rather than honest ambiguity, and absorbing
    it would remove the tripwire exactly where it is load-bearing. The genuinely
    indistinguishable case has its own flag, ``--dup-amounts``, which plants it deliberately.

    **Called only under ``--batching``.** The 1:1 path is left byte-for-byte alone, so every
    number ASSUMPTIONS.md #24a/#24b records (collision-free to n=2000, first natural collision
    at n=4000) still describes the code that produced it. A nudge running on every path would
    have silently changed what those measurements mean.

    Consumes **no randomness**, for ``_unique_amount``'s reason: a redraw would make the draw
    count depend on whether a collision happened, which is trap 4 inside a stream.

    Both uniqueness sets are maintained together. Bumping a gross to dodge a net collision can
    walk onto another payment's ``(capture_date, gross)`` on the same day -- trading a batched
    collision for the 1:1 collision ``_unique_amount`` exists to prevent -- so every bump skips
    values that key is already holding.

    Iterates ``groups`` in the order given, which ``_group_into_batches`` builds by sorted
    date: no set or dict iteration order reaches the choice of which gross moves.
    """
    taken_net: set[tuple[date, int]] = set()
    nudges = 0
    for members in groups:
        when = settled_on_of[members[0]]
        # The lowest member index, so the victim is a function of the data alone.
        victim = members[0]
        while (when, _batch_net(cfg, grosses, methods, members)) in taken_net:
            nudges += 1
            assert nudges < 100 * max(len(groups), 1), (
                f"runaway nudging around {when} -- the amount space is saturated, which at "
                f"this project's sizes means a generator change rather than pressure"
            )
            day = capture_dates[victim]
            taken_gross.discard((day, grosses[victim]))
            # One paisa at a time, skipping any value that would collide on
            # (capture_date, gross). Under --fees a 1p bump need not move the *net* at all
            # (the fee absorbs it at some magnitudes), which is why the loop above re-tests
            # the net rather than assuming one bump was enough.
            while True:
                grosses[victim] += 1
                if (day, grosses[victim]) not in taken_gross:
                    break
            taken_gross.add((day, grosses[victim]))
        taken_net.add((when, _batch_net(cfg, grosses, methods, members)))
    return nudges


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


def _draw_batch_size(rng: random.Random, remaining: int) -> int:
    """One draw -> how many of a date's remaining payments settle together.

    Exactly one draw, made unconditionally and *then* clamped, so the values this stream
    consumes per batch do not depend on how many payments are left (``rng.py`` rule 2).
    ``remaining`` is a ceiling rather than a rejection: a date holding two payments cannot
    form a batch of four, and redrawing until the size fitted would make the draw count a
    function of the date's population -- which is trap 4 reintroduced inside one stream.

    The consequence is worth stating because it shows up in the record counts: the
    *effective* mean batch size is below ``BATCH_SIZE_WEIGHTS``' nominal mean at small ``n``,
    since a thinly populated date clamps the tail of the distribution away.
    """
    total = sum(w for _, w in BATCH_SIZE_WEIGHTS)
    ticket = rng.randrange(total)
    upto = 0
    for size, weight in BATCH_SIZE_WEIGHTS:
        upto += weight
        if ticket < upto:
            return min(size, remaining)
    raise AssertionError("unreachable: BATCH_SIZE_WEIGHTS weights did not cover the ticket")


def _group_into_batches(
    rng: random.Random, settled_on_of: list[date]
) -> list[list[int]]:
    """``--batching``: partition payment indices into settlements. Returns member lists.

    **Never across a settlement date.** Cross-date batching is section 19's undesigned mess
    type, added at Phase 12 and reported unprompted -- it is the cheapest credibility in the
    submission and only cheap while genuinely undesigned. Grouping by ``settled_on`` rather
    than by the capture date is the same partition today (every capture lands on a business
    day and ``add_business_days`` is strictly monotone there, so the map only relabels days),
    but it is the one that stays *correct* if the delay model ever stops being injective.
    Invariant I11 independently re-derives every member's settlement date, so a batch that
    straddled two dates fails before anything reaches disk.

    **The date's pool is shuffled before it is sliced, and that is the load-bearing line.**
    Payments are capture-sorted and numbered in that order, so one date's payments are a
    contiguous run of payment ids. Slicing that run in place would make every batch a set of
    *consecutive* ids -- and once ``--settlement-report-late`` withholds membership, a
    searcher could enumerate contiguous runs (O(k^2)) instead of subsets (O(2^k)) and resolve
    the file without ever performing a subset search. That is the plan amendment's enduring
    test applied to this flag ("could an unbounded, model-free brute-force strategy solve
    this?") and it is finrecon's recorded failure exactly: its showcase tier fell to
    exhaustive enumeration because the intended difficulty was not information-theoretic.
    Shuffling first makes a batch an arbitrary subset of its date, so consecutive ids carry
    no information about membership.

    Members are returned **sorted** even so. The order of a member list reaches
    ``truth.json`` and is not a signal the matcher can read -- ``settlement_items.csv`` is
    written sorted by ``emit.py``, and a gross sum does not care -- so the randomness belongs
    in *which* payments group, never in how the group is spelled.

    Dates are iterated in sorted order. Not a style choice: ``date.__hash__`` goes through
    ``bytes``, which ``PYTHONHASHSEED`` randomises per process, so any iteration that depended
    on the hash of a date would make the run irreproducible across processes and
    ``tools/repro_check.py`` would catch it as a cross-process byte difference.
    """
    by_date: dict[date, list[int]] = defaultdict(list)
    for i, when in enumerate(settled_on_of):
        by_date[when].append(i)

    groups: list[list[int]] = []
    for when in sorted(by_date):
        pool = by_date[when]
        rng.shuffle(pool)
        cut = 0
        while cut < len(pool):
            size = _draw_batch_size(rng, len(pool) - cut)
            groups.append(sorted(pool[cut : cut + size]))
            cut += size
    return groups


def _withhold_membership(
    rng: random.Random, settlement_ids: list[str], share: float
) -> list[str]:
    """Pick the settlements whose rows ``emit`` leaves out of ``settlement_items.csv``.

    ``--settlement-report-late``. Returns the ids, **sorted**, so the list written into
    ``run_manifest.json`` is stable across processes -- ``rng.sample`` returns selection order,
    which is deterministic for a given seed but reads as arbitrary in a report.

    **Partial by construction** (decision 4): the share is a fraction, and the count is
    clamped so that at least one settlement keeps its membership and at least one loses it
    wherever both are possible. Withholding everything would make the tier distribution a
    *swap* rather than a mix -- all Tier 2, no Tier 1 -- and a Tier 1 regression would then
    hide behind a Tier 2 success, which is the failure gate 12 exists to catch. Withholding
    nothing would leave the search unexercised, which is the other half of the same gate.

    One ``sample`` call rather than a draw per settlement: its consumption is a function of
    the population and the count alone, both fixed by ``cfg``, so the stream stays auditable
    in the sense rule 2 asks for. It is the last thing ``build`` does, so no other stream can
    shift when the flag turns on.
    """
    population = len(settlement_ids)
    if population == 0:
        return []
    k = round(population * share)
    # A single settlement cannot be both withheld and kept, so the floor gives way to the
    # population rather than the reverse: at n=1 the honest answer is to withhold nothing.
    if population > 1:
        k = max(1, min(k, population - 1))
    else:
        k = 0
    return sorted(rng.sample(settlement_ids, k))


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
    rng_batching = substream(cfg.seed, "batching")

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

    # --- Step 4c: group into settlements, before the payments are frozen -------
    # Batching is decided here rather than after ``payments`` is built because the batched-net
    # uniqueness nudge below has to be able to *move a gross*, and ``Payment`` is a frozen
    # dataclass. The three parallel lists are the mutable view; the frozen objects are built
    # from them once every amount is final.
    capture_dates = [d[0].date() for d in drafts]
    grosses = [d[1] for d in drafts]
    methods = [d[2] for d in drafts]
    settled_on_of = [cal.add_business_days(day, cfg.delay_days) for day in capture_dates]

    if cfg.flags.batching:
        groups = _group_into_batches(rng_batching, settled_on_of)
        # ``taken`` still holds every (capture_date, gross) the draw loop reserved, so the
        # nudge can dodge both collision channels at once. Gated on the flag: the 1:1 path
        # must stay byte-identical to Phase 1, since clean mode is the regression check and
        # ASSUMPTIONS.md #24a's numbers describe that exact code.
        _make_nets_unique(
            cfg, grosses, methods, capture_dates, settled_on_of, groups, taken
        )
    else:
        groups = [[i] for i in range(cfg.n)]

    payments = [
        Payment(
            payment_id=ids.payment_id(i),
            order_id=ids.order_id(i),
            captured_at=captured_at,
            gross_paise=grosses[i - 1],
            method=method,
        )
        for i, (captured_at, _drafted_gross, method) in enumerate(drafts, start=1)
    ]

    # --- Step 5: derive settlements --------------------------------------
    # net == gross and every deduction is zero: clean mode has no fees at all.
    # settled_on is ``cfg.delay_days`` business days after the IST capture date --
    # which is the capture date itself in clean mode, since the delay is 0 there and
    # every capture lands on a business day. payment_ids has been a *list* since
    # Phase 1 (decision #10), so ``--batching`` is a change of contents here and not a
    # change of type -- which is the whole reason that decision was taken this early.
    #
    # ``_draw_tails`` still draws ``cfg.n`` tails and not one per settlement (Phase 5
    # decision 11). Drawing for the settlement count would shift every UTR in the file the
    # moment batching turned on, making the clean-vs-batching diff at one seed unreadable --
    # the same property that kept ASSUMPTIONS.md #24b's cross-run comparison valid. Batching
    # produces *fewer* settlements than ``n``, so the pool is a superset and the spares the
    # credit fixup consumes still start at ``tails[cfg.n:]``.
    tails = _draw_tails(rng_utr, cfg.n)

    # ``groups`` was built above, before the payments were frozen. Shuffled here so the
    # numbering carries no information (step 7). Without ``--batching`` it is one payment per
    # settlement in payment order, and the shuffle draws as a function of list *length*
    # alone -- which is ``cfg.n`` either way, so clean mode stays byte-identical to Phase 1.
    rng_settlements.shuffle(groups)

    settlements: list[Settlement] = []
    for seq, members in enumerate(groups, start=1):
        batch = [payments[i] for i in members]
        # **Per member, per method, each rounded at the paisa -- never the rate on the batch
        # total at one blended rate.** Phase 5 decision 7, and the two failure modes it
        # guards differ by four orders of magnitude. Holding the rate constant, batch-total
        # rounding differs from the per-member sum by <=5 paise, but on 27-71% of batches.
        # Pricing a mixed-method batch at any single blended rate differs by up to **11,596
        # rupees on one row** -- that is the 0-300 bps spread across the method table, not
        # rounding, and the amount bands make a mixed batch the common case rather than the
        # corner.
        #
        # Those two figures are **upper bounds from the plan's synthetic sweep** (20,000
        # batches at k up to 8, .plan/phase5.md section 1(b)), not descriptions of this
        # file's output: ``BATCH_SIZE_WEIGHTS`` caps a batch at 4, so k=8's spread is out of
        # reach here. Measured on what this generator actually emits, across seeds 1-5 and 42
        # at n=200 and n=1000, same-method rounding maxes at **1p** and the blended-rate gap
        # reaches **Rs 7,310.15** on one settlement. Both bounds are cited rather than
        # replaced because a later phase raising the batch ceiling moves this data toward
        # them, and the discipline has to be justified by the failure mode's size rather than
        # by today's sample.
        #
        # ``matcher/fees.derive`` prices the same way, independently, which is what makes the
        # residual a test instead of a tautology.
        #
        # This side fails safe: getting it wrong here puts every batched row into
        # UNEXPLAINED_RESIDUAL, which is loud. The matcher side would fail quietly.
        deductions = [_deductions(cfg, p.gross_paise, p.method) for p in batch]
        fee = sum(f for f, _ in deductions)
        gst = sum(g for _, g in deductions)
        gross_total = sum(p.gross_paise for p in batch)
        # Every member of a batch shares a settlement date by construction
        # (``_group_into_batches`` partitions within one date), so the settlement's date is
        # any member's. Asserted rather than assumed: I11 re-derives it per member, and a
        # batch that straddled two dates would otherwise emit a date that is right for some
        # of its payments and wrong for the rest.
        when = settled_on_of[members[0]]
        assert all(settled_on_of[i] == when for i in members), (
            f"setl_{seq:04d} spans more than one settlement date -- cross-date batching is "
            f"section 19's undesigned mess type (Phase 12), not this flag"
        )
        settlements.append(
            Settlement(
                settlement_id=ids.settlement_id(seq),
                settled_on=when,
                payment_ids=[p.payment_id for p in batch],
                # net = gross - fee - GST, summed over the members. Both deductions are 0
                # without --fees, so this is ``net == sum of grosses`` in clean mode and the
                # expression stops being trivial exactly when the flag turns on. tds_paise
                # stays 0: TDS is dial row 7, Phase 6 (decision D7) -- two gross/net wedges
                # at once means two hypotheses per failing row.
                net_paise=gross_total - fee - gst,
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
    # Indexed through ``groups`` rather than the old flat ``order``: after the shuffle above
    # ``groups[i]`` is the member list of ``settlements[i]``, which is the same relationship
    # ``order[i]`` carried when every settlement held exactly one payment.
    planted_settlement_ids: set[str] = set()
    if planted_keys:
        first_utr: dict[tuple[date, int], str] = {}
        for i, s in enumerate(settlements):
            members = groups[i]
            p = payments[members[0]]
            key = (p.business_date, p.gross_paise)
            if key not in planted_keys:
                continue
            # Guaranteed single-member: ``GenConfig`` refuses ``--dup-amounts`` together with
            # ``--batching``, because a planted payment sharing a settlement with others turns
            # that settlement's net into a sum, diverging it from its pair's and quietly
            # un-planting the collision. Asserted rather than trusted -- the config check is
            # one edit away from being relaxed, and this loop is where the damage would be
            # silent (a wrong planted count, not a crash).
            assert len(members) == 1, (
                f"{s.settlement_id} holds {len(members)} payments and carries a planted "
                f"collision -- see GenConfig's --dup-amounts/--batching refusal"
            )
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
    # --- Step 5c: --settlement-report-late withholds membership declarations ---
    # Last, and gated, for two separate reasons. Gated so that every run without the flag
    # consumes the ``late`` stream not at all and stays byte-identical to Phase 1 through
    # Phase 5 step 1. Last so that this selection cannot move any other draw: it reads the
    # finished settlement list and returns ids, touching no amount, date, method or grouping.
    #
    # **What is withheld is the declaration, not the membership.** Every ``Settlement`` below
    # still lists its payments, and ``truth.json`` still publishes them -- so the answer key
    # stays complete, every in-memory invariant still runs, and the scorer can still grade a
    # searched payment set against the real one. ``emit`` is the only place the withholding
    # becomes visible, by leaving those rows out of ``settlement_items.csv``.
    withheld: list[str] = []
    if cfg.flags.settlement_report_late:
        withheld = _withhold_membership(
            substream(cfg.seed, "late"),
            [s.settlement_id for s in settlements],
            LATE_REPORT_SHARE,
        )
        # The backstop GenConfig cannot provide. It refuses ``n < 2``, which is certain from
        # the config; how many settlements a *batched* run produces is a draw, and a small n
        # can land on a single settlement (measured: n=2 with --batching on 1 of 60 seeds).
        # A refusal rather than an assert, and rather than withholding nothing: a run labelled
        # with a mess flag must either have that mess or say why it cannot.
        if not withheld:
            raise ValueError(
                f"--settlement-report-late cannot withhold partially: this run produced "
                f"{len(settlements)} settlement(s) from n={cfg.n}, and withholding the only "
                f"one would be total (decision 4). Raise --n."
            )
        assert len(withheld) < len(settlements), (
            "--settlement-report-late withheld every settlement: the tier distribution "
            "would be a swap rather than a mix (decision 4)"
        )

    return Story(
        payments=payments,
        settlements=settlements,
        credits=credits,
        refunds=[],
        membership_withheld=withheld,
    )


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

    # --- Phase 5 step 1: --batching -------------------------------------------
    # What is asserted here is the *partition* and the *containment*. Which payments group
    # together is a draw and is not asserted; that they group into exactly one settlement
    # each, never across a settlement date, and that the deductions sum per member rather
    # than being priced on the batch total, are structural.
    bat_cfg = GenConfig(seed=42, n=200, flags=MessFlags(batching=True))
    bat_story = build(bat_cfg)
    assert len(bat_story.payments) == 200, "--batching must not change how many payments exist"
    assert len(bat_story.settlements) == len(bat_story.credits), "one settlement, one credit"
    assert len(bat_story.settlements) < len(bat_story.payments), "--batching batched nothing"
    # A partition: every payment in exactly one settlement, every member a real payment.
    # Measured at seed 42, n=200: 200 payments into 120 settlements, sizes 1 to 4.
    members = [pid for s in bat_story.settlements for pid in s.payment_ids]
    assert len(members) == len(set(members)) == 200, (
        f"member lists are not a partition of payments.csv: {len(members)} members, "
        f"{len(set(members))} distinct, 200 payments"
    )
    assert set(members) == {p.payment_id for p in bat_story.payments}
    # Decision 2: size 1 must survive. If every settlement became a batch this would be a
    # *swap* rather than a mix, and a Tier 1 regression could hide behind a Tier 2 success.
    bat_sizes = [len(s.payment_ids) for s in bat_story.settlements]
    assert min(bat_sizes) == 1, "size-1 batches must remain, or Tier 1 has no rows left"
    assert max(bat_sizes) > 1, "some settlement must hold more than one payment"
    # No batch spans a settlement date. Cross-date batching is the undesigned mess type held
    # for Phase 12, and it must stay undesigned. I11 re-derives every member date
    # independently; naming the property here says it is intended rather than incidental.
    bat_by_pay = {p.payment_id: p for p in bat_story.payments}
    for s in bat_story.settlements:
        days = {bat_by_pay[pid].business_date for pid in s.payment_ids}
        assert len(days) == 1, f"{s.settlement_id} spans capture dates {sorted(days)}"
        assert s.settled_on == next(iter(days)), "delay 0: settled_on is the capture date"
        # No --fees, so net is the plain sum of member grosses and every deduction is zero.
        assert s.net_paise == sum(bat_by_pay[pid].gross_paise for pid in s.payment_ids)
        assert (s.fee_paise, s.gst_paise, s.tds_paise) == (0, 0, 0)
    # Each credit still closes to its own amount, now over a *set* of payments rather than
    # one. Per credit, not in total: two compensating errors cancel in a sum.
    for c in bat_story.credits:
        assert c.decomposition.expected_credit_paise == c.amount_paise, c.credit_id
        assert c.decomposition.gross_paise == sum(
            bat_by_pay[pid].gross_paise for pid in c.payment_ids
        )
    assert [c.csv_row() for c in build(bat_cfg).credits] == [
        c.csv_row() for c in bat_story.credits
    ], "--batching is not deterministic"

    # --batching changes the settlement grouping and nothing about the payments. At n=200 the
    # payment stream is byte-identical to the clean run at the same seed, which is what keeps
    # a clean-versus-batching diff readable. Built fresh rather than reusing the fee block
    # clean_200, so that editing one config cannot silently redefine the other assertion.
    bat_clean = build(GenConfig(seed=42, n=200))
    assert [p.csv_row() for p in bat_clean.payments] == [
        p.csv_row() for p in bat_story.payments
    ], "--batching perturbed the payment stream at n=200, where no nudge should fire"

    # The batched-net nudge, at the size where it genuinely fires. Batching creates a new way
    # for two settlements to collide: a sum can equal a single net. Measured at seed 3,
    # n=1000: a collision on 2026-08-21 at 454,300p, one 1-member settlement against a
    # 2-member batch. Step 4 of the plan pre-committed the response -- extend the nudge,
    # never relax I3.unique_date_amount -- because a relaxed check would let honestly
    # ambiguous data through as if it were designed that way.
    nudged = build(GenConfig(seed=3, n=1000, flags=MessFlags(batching=True)))
    nudged_nets = [(s.settled_on, s.net_paise) for s in nudged.settlements]
    assert len(set(nudged_nets)) == len(nudged_nets), "two settlements share a (settled_on, net)"
    nudged_keys = [(c.value_date, c.amount_paise) for c in nudged.credits]
    assert len(set(nudged_keys)) == len(nudged_keys), "the nudge left a colliding credit pair"
    # It is surgical, not a resample: exactly one payment moves, by exactly one paisa.
    # Measured: pay_0661 at seed 3, and no movement at all at n=60 or n=200 on any seed.
    clean_1k = {p.payment_id: p.gross_paise for p in build(GenConfig(seed=3, n=1000)).payments}
    nudge_moved = {
        p.payment_id: p.gross_paise - clean_1k[p.payment_id]
        for p in nudged.payments
        if p.gross_paise != clean_1k[p.payment_id]
    }
    assert nudge_moved == {pid: 1 for pid in nudge_moved}, (
        f"the nudge moved a gross by more than a paisa: {nudge_moved}"
    )
    assert len(nudge_moved) == 1, f"expected one nudged payment, got {sorted(nudge_moved)}"

    # --batching with --fees: deductions sum **per member at that member's own method rate**,
    # never the rate on the batch total (decision 7). Two distinct errors are excluded, and
    # they are not the same size. Pricing a same-method batch on its total is a rounding
    # difference of at most 1p. Pricing a *mixed*-method batch at one blended rate is the
    # 0-to-300 bps spread across the method table, and measured across seeds 1-5 and 42 it
    # reaches Rs 7,310.15 on a single settlement (setl_0491, seed 42, n=1000) -- this
    # generator's k<=4 ceiling, against the plan's synthetic Rs 11,596 at k=8. The cheap
    # error and the expensive one share a shape, so the per-member sum is asserted directly.
    bf_cfg = GenConfig(seed=42, n=200, flags=MessFlags(batching=True, fees=True))
    bf_story = build(bf_cfg)
    bf_by_pay = {p.payment_id: p for p in bf_story.payments}
    bf_mixed = 0
    for s in bf_story.settlements:
        mem = [bf_by_pay[pid] for pid in s.payment_ids]
        want = [_deductions(bf_cfg, p.gross_paise, p.method) for p in mem]
        assert s.fee_paise == sum(f for f, _ in want), (
            f"{s.settlement_id}: fee {s.fee_paise} is not the per-member sum "
            f"{sum(f for f, _ in want)} -- a batch priced at one blended rate"
        )
        assert s.gst_paise == sum(g for _, g in want)
        assert s.net_paise == sum(p.gross_paise for p in mem) - s.fee_paise - s.gst_paise
        if len({p.method for p in mem}) > 1:
            bf_mixed += 1
    # A mixed-method batch must actually occur, or the per-member assertion above is
    # asserting nothing and the expensive error goes untested. Measured: 46 of 49 multi-member
    # batches at seed 42, n=200 -- mixed is the common case, not the corner.
    assert bf_mixed, "no mixed-method batch in 200 records -- decision 7 went untested"

    # Edge sizes. One payment cannot be batched with anything; a handful may each land on
    # their own date and batch nothing, which is a legal outcome and not asserted away.
    assert len(build(GenConfig(seed=7, n=1, flags=MessFlags(batching=True))).settlements) == 1
    small_bat = build(GenConfig(seed=7, n=4, flags=MessFlags(batching=True)))
    assert sum(len(s.payment_ids) for s in small_bat.settlements) == 4

    # --dup-amounts and --batching are refused at config time. Batching a planted payment
    # makes the net of its settlement a sum, so the pair stops sharing one net and stops
    # being unresolvable -- truth would then claim resolvable=false about separable data.
    try:
        GenConfig(seed=42, n=200, flags=MessFlags(dup_amounts=True, batching=True))
    except ValueError as exc:
        assert "dup-amounts" in str(exc) and "batching" in str(exc), exc
    else:
        raise AssertionError("--dup-amounts with --batching must be refused")

    # --- Phase 5 step 5: --settlement-report-late ------------------------------
    # What is asserted is that the withholding is **partial**, that it is confined to the
    # membership *declaration*, and that it moves nothing else. Which settlements are picked
    # is a draw and is not asserted.
    late_cfg = GenConfig(seed=42, n=200, flags=MessFlags(settlement_report_late=True))
    late = build(late_cfg)
    late_ids = {s.settlement_id for s in late.settlements}
    assert late.membership_withheld, "--settlement-report-late withheld nothing"
    # Partial, never total (decision 4). Total withholding makes the tier distribution a
    # *swap* rather than a mix, so a Tier 1 regression could hide behind a Tier 2 success --
    # the failure gate 12 exists to catch, from the other side.
    assert len(late.membership_withheld) < len(late.settlements), (
        f"every one of {len(late.settlements)} settlements was withheld: that is a swap, "
        f"not a mix"
    )
    assert set(late.membership_withheld) <= late_ids, "withheld ids must be real settlements"
    assert len(set(late.membership_withheld)) == len(late.membership_withheld)
    # Sorted, so the list written into run_manifest.json is stable to read across runs.
    # ``rng.sample`` returns selection order, which is deterministic but reads as arbitrary.
    assert late.membership_withheld == sorted(late.membership_withheld)
    # Measured: 30% at every size where the share is reachable (60/200/1000 -> 18/60/300).
    assert len(late.membership_withheld) == 60, len(late.membership_withheld)

    # **The membership is not removed, only its declaration.** Every settlement still lists
    # its payments here and truth.json still publishes them, which is what keeps the answer
    # key complete: a payment set the matcher *searched* has to be gradeable against the real
    # one. ``emit`` is the only place the withholding exists.
    assert all(s.payment_ids for s in late.settlements), (
        "--settlement-report-late removed a membership from the story rather than from the "
        "file the matcher reads -- then a searched payment set could not be graded"
    )
    # It moves nothing else: same payments, settlements and credits as the clean run at the
    # same seed. The ``late`` stream is drawn last and only under the flag, so a run without
    # it consumes no randomness here at all.
    late_clean = build(GenConfig(seed=42, n=200))
    assert not late_clean.membership_withheld, "a clean run must withhold nothing"
    assert [p.csv_row() for p in late_clean.payments] == [p.csv_row() for p in late.payments]
    assert [s.csv_row() for s in late_clean.settlements] == [
        s.csv_row() for s in late.settlements
    ], "--settlement-report-late changed a settlement rather than only its declaration"
    assert [c.csv_row() for c in late_clean.credits] == [c.csv_row() for c in late.credits]
    assert build(late_cfg).membership_withheld == late.membership_withheld

    # Combined with --batching, which is the combination step 6 needs: a withheld
    # multi-member batch is a row whose payment set can only be found by searching. Measured
    # at seed 42, n=200: 120 settlements, 36 withheld, 12 of them multi-member.
    late_bat = build(
        GenConfig(seed=42, n=200, flags=MessFlags(batching=True, settlement_report_late=True))
    )
    late_bat_withheld = set(late_bat.membership_withheld)
    searchable = [
        s for s in late_bat.settlements
        if s.settlement_id in late_bat_withheld and len(s.payment_ids) > 1
    ]
    assert searchable, (
        "no withheld settlement holds more than one payment, so nothing in this run requires "
        "a subset search and Tier 2 would be unexercised"
    )
    # Tier 1 must still have rows: settlements that kept their declaration. Both halves
    # non-empty is the mix gate 12 asserts end to end.
    assert len(late_bat_withheld) < len(late_bat.settlements)

    # Both refusals. The flag needs two settlements to have a choice, so a run that cannot
    # withhold partially is refused rather than quietly withholding nothing -- a run labelled
    # with a mess flag must either have that mess or say why it cannot.
    try:
        GenConfig(seed=42, n=1, flags=MessFlags(settlement_report_late=True))
    except ValueError as exc:
        assert "at least 2 settlements" in str(exc), exc
    else:
        raise AssertionError("--settlement-report-late at n=1 must be refused at config time")
    # The build-time backstop, for what no config-time check can see: how many settlements a
    # *batched* run produces is a draw. Measured -- n=2 lands on a single settlement on seed
    # 10, where GenConfig sees only a legal n=2.
    assert len(build(GenConfig(seed=10, n=2, flags=MessFlags(batching=True))).settlements) == 1
    try:
        build(GenConfig(seed=10, n=2,
                        flags=MessFlags(batching=True, settlement_report_late=True)))
    except ValueError as exc:
        assert "partially" in str(exc), exc
    else:
        raise AssertionError(
            "a batched run that produced one settlement must refuse to withhold it: "
            "withholding the only settlement is total, not partial"
        )

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
