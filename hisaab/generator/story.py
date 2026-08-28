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
from datetime import date, datetime, timedelta

from ..common import ids
from ..common.bizdays import BusinessCalendar
from ..common.reasons import Reason
from .config import (
    AMOUNT_BANDS,
    BANK_CHANNELS,
    BATCH_SIZE_WEIGHTS,
    LATE_REPORT_SHARE,
    REFUND_BPS_BAND,
    REFUND_SHARE,
    RESERVE_BPS_BAND,
    RESERVE_SHARE,
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
from .model import Credit, Decomposition, Payment, Refund, Settlement, Story
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


def _tds(cfg: GenConfig, gross_paise: int) -> int:
    """TDS withheld on one payment's **gross**. ``0`` unless ``--tds`` is on.

    Three properties, each a deliberate difference from ``_deductions`` above.

    **The base is the gross, not the fee.** §194-O withholds on the gross amount of the
    sale, so this is not "another rate on the fee" like GST -- it is a second rate on the
    same base the fee uses. ASSUMPTIONS.md #9/#9a carry the rate (10 bps) and the scope
    caveat; the rate is verified and the claim that an aggregator withholds it is not.

    **No method argument, and that is the whole character of the term.** Every rate in
    ``_deductions`` depends on the rail; this one does not, because a tax rate is not a
    price. The consequence is worth stating because it removes a property the codebase has
    relied on since Phase 4: ``pos_upi`` is zero-*rated*, so a POS settlement pays out at
    its gross under ``--fees`` -- but it still has TDS withheld. So ``--tds`` is the flag
    that ends the zero-deduction row. Under ``--fees --tds`` **no** settlement settles at
    its gross, which is measured in the self-check below rather than assumed.

    **This function draws no randomness, and no stream exists for it to draw from.**
    ``rng.py`` reserves a ``tds`` stream and ``.plan/phase6.md`` step 2 says to withhold on
    it; both are wrong, and the plan's own reasoning is what refutes them. A withholding at
    a declared rate on a declared gross is *fully derived* -- there is nothing to draw. A
    draw would have to mean "TDS applies to a random subset of settlements", which is a
    modelling claim nobody made and which no invariant could then check, since truth would
    be the only record of which rows were chosen. So ``--tds`` is the first mess flag that
    adds a deduction **without touching the RNG**, and it therefore cannot shift any other
    stream: a clean-vs-tds diff at one seed moves the money columns and nothing else. That
    is a stronger reproducibility property than any other flag has, and spending a draw to
    look consistent with the others would destroy it.
    """
    if not cfg.flags.tds:
        return 0
    return mul_bps(gross_paise, cfg.fees.tds_bps)


def _draw_refunds(
    cfg: GenConfig, rng: random.Random, grosses: list[int], groups: list[list[int]]
) -> tuple[dict[int, int], dict[int, int]]:
    """``(linked, unlinked)``: payment index -> refund in **absolute paise**.

    Both are empty unless ``--netted-refunds`` is on. ``linked`` refunds cite a payment that
    is in ``payments.csv``; ``unlinked`` ones cite a payment that is **not**, which is the
    planted mess (see ``build``'s step 6b and ``Reason.REFUND_UNLINKED``). They are the same
    arithmetic against a settlement's net and differ only in whether the matcher can attribute
    them, which is why the two are merged to compute a net and kept apart for truth.

    **The magnitude is drawn as basis points and immediately frozen into paise, and that
    conversion happening *here* rather than at the point of use is the whole design.**
    ``_make_nets_unique`` moves a gross by a paisa at a time to keep two settlements off one
    net; if a refund were stored as a rate and re-derived from the gross later, every nudge
    would move the refund, which moves the net, which is the value the nudge is trying to
    place -- a loop whose fixed point is not guaranteed to exist. Frozen paise make the nudge
    monotone again: the refund is a constant the net is computed *from*, never a function of
    it. The cost is that a refund is not exactly ``x%`` of the final gross on a nudged row,
    which is invisible in the data and is the right trade.

    Drawn on the ``refunds`` stream, which ``rng.py`` has reserved since Phase 1, and drawn
    with a consumption that is a function of ``cfg`` alone (rule 2): one ``sample`` for the
    population and then exactly one ``randint`` per chosen payment. A per-payment
    "should this be refunded?" coin flip would consume ``n`` draws instead of ``k``, which is
    the same count but makes the *chosen set* depend on the amount distribution -- and the
    refunded rows would then move whenever an amount band changed.

    **Partial by construction.** ``REFUND_SHARE`` is a fraction, so a run always has both
    refunded and unrefunded settlements. That is the same requirement
    ``--settlement-report-late`` has and for the same reason: a flag that refunded every
    payment would make "the refund term" and "every row" the same set, and no comparison
    downstream could tell a refund-aware matcher from one that had absorbed the term into its
    fee model.
    """
    if not cfg.flags.netted_refunds:
        return {}, {}
    lo, hi = REFUND_BPS_BAND
    population = len(grosses)
    k = round(population * REFUND_SHARE)
    # At least one, or the run is labelled with a mess it does not have. Never so many that
    # nothing is left over: one payment must stay unrefunded so the term is partial, and one
    # more must be free to carry the planted unlinked refund below. ``GenConfig`` refuses the
    # ``n`` that cannot satisfy both, so this clamp is the backstop rather than the policy.
    k = max(1, min(k, population - 2))

    def _amount(gross: int) -> int:
        # ``randint`` is inclusive at both ends, so the band reads as written. A refund of
        # nothing is not a refund -- ``Refund.__post_init__`` asserts it is positive and a zero
        # row would be a declared refund the arithmetic cannot see -- so the floor is applied
        # rather than trusted, since it is reachable at the smallest drawable gross.
        return max(1, mul_bps(gross, rng.randint(lo, hi)))

    chosen = sorted(rng.sample(range(population), k))
    linked = {i: _amount(grosses[i]) for i in chosen}

    # --- the planted one ---------------------------------------------------
    # **Drawn from the payments no linked refund touched**, so one settlement never carries
    # both provenances at once. Not cosmetic: truth publishes a single ``refunds_paise`` term
    # per credit, and a settlement mixing an attributable refund with an unattributable one
    # would make that term impossible to reason about -- the matcher could explain part of it,
    # and "partly explained" is not a state this project's arithmetic has a name for.
    #
    # **Disjoint at the settlement level, not the payment level, and that distinction was a
    # live bug rather than a hypothetical.** Excluding only the refunded *payments* is what
    # this function used to do, and under ``--batching`` a settlement holding two payments can
    # take a linked refund off one and the planted one off the other -- so its single
    # ``refunds_paise`` term becomes partly attributable, which is the state ``build``'s
    # assertion below says the arithmetic has no name for. It fired on seed 3 and seed 7 at
    # n=200 the first time any gate ran ``--netted-refunds --batching`` together (gate 13),
    # having predicted the failure in its own message.
    #
    # So the exclusion is by settlement. Without ``--batching`` every group is a singleton and
    # this is exactly the old set, which is what keeps the 1:1 path byte-identical.
    tainted = {i for group in groups for i in group if any(m in linked for m in group)}
    remaining = [i for i in range(population) if i not in tainted]
    if not remaining:
        # Reachable only when every settlement in the run carries a linked refund, which needs
        # a batching draw that gathers all of them -- possible at the smallest legal ``n`` and
        # vanishingly unlikely above it. Dropping the plant is the honest response: a run
        # without an unattributable refund is a weaker run, but a run whose refund term is
        # half-attributable is an *incoherent* one, and truth would have to publish a number it
        # could not describe. ``REFUND_UNLINKED`` simply has no case here.
        return linked, {}
    # ``sample`` of one rather than ``choice``: it consumes a draw count that is a function of
    # the population alone, which is the property ``rng.py`` rule 2 asks for.
    planted_index = rng.sample(remaining, 1)[0]
    unlinked = {planted_index: _amount(grosses[planted_index])}
    return linked, unlinked


def _draw_reserves(
    cfg: GenConfig, rng: random.Random, settlements: list[Settlement]
) -> dict[str, int]:
    """``settlement_id -> paise held back``. Empty unless ``--reserve`` is on.

    **Design B: the reserve stays outside ``net_paise``.** The settlement declares its full
    net and the bank credit arrives *short* by this amount, which appears in no input file at
    all -- not as a column, not as a row, nowhere. That asymmetry is the entire mess, and the
    alternative collapses it: if the reserve were folded into ``net_paise`` the credit would
    equal the declared net again, every invariant would pass untouched, and nothing anywhere
    would say money had been held. There would be no mess to find. So a reserve *necessarily*
    makes ``credit < net``, and that is a consequence of modelling a reserve at all rather
    than of choosing design B (``.plan/phase6.md`` correction (c) reaches the same conclusion
    from the matcher's side).

    **Drawn after the settlements are built, which is the opposite of ``_draw_refunds`` and
    for a reason worth stating.** A refund is a term ``net_paise`` is computed *from*, so it
    has to exist before ``_make_nets_unique`` places a net -- and it has to be frozen into
    paise, or the nudge would chase a target that moves with it. A reserve is subtracted from
    a net that is already final, so it touches the nudge loop not at all and can be drawn from
    the finished settlements. Nothing about the reserve can perturb the uniqueness work.

    Drawn on the ``reserve`` stream, which ``rng.py`` has reserved since Phase 1. Consumption
    is one ``sample`` plus exactly one ``randint`` per chosen settlement -- a function of the
    settlement count, which is itself a function of ``cfg`` (rule 2). Phase 5 decision 8 is the
    reason that gets checked rather than assumed: drawing on a count that was not a function of
    ``cfg`` shifted every UTR in the file and made a cross-run diff unreadable. Here the stream
    is otherwise unused and drawn last, so no other stream can be displaced by it.

    Partial by construction (``RESERVE_SHARE`` is a fraction), for the reason every share in
    this generator is partial: a run where every settlement were reserved could not distinguish
    a reserve-aware matcher from one that had simply widened its tolerance until everything fit.
    """
    if not cfg.flags.reserve:
        return {}
    lo, hi = RESERVE_BPS_BAND
    population = len(settlements)
    k = round(population * RESERVE_SHARE)
    # At least one, or the run is labelled with a mess it does not have; never all of them, or
    # the term stops being partial. ``GenConfig`` refuses the ``n`` that cannot satisfy both,
    # so this clamp is the backstop rather than the policy.
    k = max(1, min(k, population - 1))
    chosen = sorted(rng.sample(range(population), k))
    held: dict[str, int] = {}
    for i in chosen:
        s = settlements[i]
        # ``randint`` is inclusive at both ends, so the band reads as written. A reserve of
        # nothing is not a reserve -- it would be a row truth labels as reserved while the
        # credit equals the net exactly, which is a claim the data contradicts -- so the floor
        # is applied rather than trusted.
        held[s.settlement_id] = max(1, mul_bps(s.net_paise, rng.randint(lo, hi)))
    return held


def _separate_reserved_amounts(
    settlements: list[Settlement], held: dict[str, int]
) -> None:
    """Nudge each held amount until the short credit matches **no settlement's net**.

    Mutates ``held`` in place. Without this, ``--reserve`` can plant a silent wrong match,
    and it is the sharpest hazard in this flag -- sharper than anything ``--netted-refunds``
    could produce, for a reason worth stating precisely.

    **A reserved credit is the first row in this project whose true settlement is not in its
    own candidate pool.** Every other credit equals some settlement's net, so blocking finds
    the right settlement and the only question is whether anything *else* is in the pool too.
    A reserved credit equals its own settlement's net *minus* the held amount, so its true
    settlement is invisible to an exact-band lookup. That inverts the failure mode:

      * For every pre-Phase-6 row, an amount clash puts **two** settlements in the pool and
        ``resolve_credit`` abstains with ``AMBIGUOUS_DUPLICATE_AMOUNT``. Honest, and safe.
      * For a reserved row, a clash puts **exactly one** settlement in the pool -- the
        *wrong* one -- and its arithmetic closes perfectly, because the credit really does
        equal that settlement's net. The matcher resolves it, names the wrong payment set,
        and has no way whatsoever to know. That is a ``WRONG_MATCH``, on the one line this
        project says never bends.

    So the guard is not "keep ``(value_date, amount)`` unique" -- that is I3's per-date
    property and it is **not sufficient here**. The clash that hurts is with any settlement
    inside the *date window*, and the window is a matcher-side parameter this generator does
    not know. The guard is therefore made window-independent and date-independent by
    excluding the amount from **every** net in the run, which is strictly stronger than
    anything a window could require. It also keeps the reserved credits distinct from each
    other, so two reserved rows cannot collide into a pool of two either.

    **The nudge is free, and that is what makes this legitimate rather than a fudge.** The
    held amount appears in no input file -- no column, no row -- so moving it by a few paise
    is unobservable to any matcher and changes no declared quantity. Compare
    ``_unique_amount``, which nudges a *gross* and therefore does perturb ``payments.csv``.
    Here the only visible consequence is the credit amount, which is already whatever the
    reserve made it.
    """
    if not held:
        return
    all_nets = {s.net_paise for s in settlements}
    taken_amounts: set[int] = set()
    for s in settlements:
        amount = held.get(s.settlement_id)
        if amount is None:
            continue
        # Nudge the *reserve* up, which walks the credit down a paisa at a time. Up rather
        # than down so the reserve can never round to zero and turn a row truth calls
        # reserved into one whose credit equals its net.
        while (s.net_paise - amount) in all_nets or (s.net_paise - amount) in taken_amounts:
            amount += 1
        # ``Credit.__post_init__`` asserts a positive amount. The band tops out at 20% of
        # net and the nudge moves single paise, so this has enormous headroom -- checked
        # rather than trusted, because a future band change is what would consume it.
        assert amount < s.net_paise, (
            f"{s.settlement_id}: a reserve of {amount}p is not less than its {s.net_paise}p "
            f"net, so the credit would be zero or negative. Lower RESERVE_BPS_BAND."
        )
        held[s.settlement_id] = amount
        taken_amounts.add(s.net_paise - amount)


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
    cfg: GenConfig,
    grosses: list[int],
    methods: list[str],
    members: list[int],
    refunds: dict[int, int] | None = None,
) -> int:
    """A batch's ``net_paise``: per member, per method, each rounded at the paisa, summed.

    The one place this arithmetic is spelled, so the uniqueness nudge below and the
    ``Settlement`` built in ``build`` cannot disagree about what a batch nets. Decision 7 --
    and see ``matcher/fees.derive``, which does the same sum independently and is what makes
    the residual a test rather than a tautology.

    ``refunds`` maps a payment index to the refund netted off it, in **absolute paise** (see
    ``_draw_refunds`` for why it is frozen rather than a rate). Defaulted to ``None`` rather
    than required: every caller that predates ``--netted-refunds`` means "no refunds", and the
    two are the same arithmetic.
    """
    netted = refunds or {}
    total = 0
    for i in members:
        fee, gst = _deductions(cfg, grosses[i], methods[i])
        # TDS is inside the net (I4, ``invariants.py:352``), so it has to be inside the value
        # this function protects. Leaving it out would nudge for uniqueness on a number the
        # emitted data never carries -- I3 would then check a key nothing guarded, and the
        # margin measured in ASSUMPTIONS #24a would be describing the wrong quantity.
        #
        # The refund is inside for the identical reason, and the reason is worth restating
        # because a refund is the first term here that is **not derivable from a rate**: it is
        # declared in ``refunds.csv``, so a matcher must look it up rather than compute it.
        # That changes who can verify the number, not which side of the net it falls on.
        total += grosses[i] - fee - gst - _tds(cfg, grosses[i]) - netted.get(i, 0)
    return total


def _make_nets_unique(
    cfg: GenConfig,
    grosses: list[int],
    methods: list[str],
    capture_dates: list[date],
    settled_on_of: list[date],
    groups: list[list[int]],
    taken_gross: set[tuple[date, int]],
    refunds: dict[int, int] | None = None,
) -> int:
    """Nudge grosses until no two settlements share a ``(settled_on, net)``. Returns nudges.

    ``refunds`` is the frozen per-payment refund in paise, threaded through to ``_batch_net``
    because a refund is one of the terms the net is computed from. Threading it rather than
    re-deriving it is what keeps this loop monotone: a bump moves the gross, the refund stays
    put, so the net moves by exactly the bump. A refund re-derived as a share of the gross
    would move with every bump and the loop would be chasing a target that runs away from it.

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
        while (when, _batch_net(cfg, grosses, methods, members, refunds)) in taken_net:
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
        taken_net.add((when, _batch_net(cfg, grosses, methods, members, refunds)))
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
    rng_refunds = substream(cfg.seed, "refunds")
    rng_reserve = substream(cfg.seed, "reserve")

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

    # --- Step 6a: --netted-refunds draws the refunds ----------------------
    # Drawn **before** the uniqueness nudge and after the amounts are sorted, which is the
    # only order that works. The nudge places a settlement's net, and a refund is one of the
    # terms that net is computed from -- so the refund has to exist first, as a frozen number
    # (see ``_draw_refunds``). Drawn after the sort so the chosen indices refer to the same
    # payment numbering that reaches ``payments.csv``.
    #
    # ``refunded`` maps payment index -> refund in paise, netted off the settlement holding
    # that payment. ``unlinked`` is the planted mess and maps the same way, but its refund row
    # cites a payment that is **not in this month's file** -- so the money comes off a real
    # settlement while nothing in the inputs says which settlement it came off. Both dicts
    # are the same arithmetic and differ only in provenance, which is why they are merged for
    # the net and kept apart for truth.
    # The batches are decided **before** the refunds are drawn, because the planted refund has
    # to know which settlement each payment lands in -- see ``_draw_refunds`` on why that
    # exclusion is by settlement rather than by payment. Reordering these two is free of
    # reproducibility cost because they draw on *different* substreams (``batching`` and
    # ``refunds``), so neither one's sequence of values changes; that independence is the
    # property ``rng.py``'s named streams exist to give, and it is what makes an ordering fix
    # like this one a local edit rather than a reseeding of the whole run.
    groups = (
        _group_into_batches(rng_batching, settled_on_of)
        if cfg.flags.batching
        else [[i] for i in range(cfg.n)]
    )

    refunded, unlinked = _draw_refunds(cfg, rng_refunds, grosses, groups)
    netted_by_index = {
        i: refunded.get(i, 0) + unlinked.get(i, 0)
        for i in set(refunded) | set(unlinked)
    }

    if cfg.flags.batching:
        # ``taken`` still holds every (capture_date, gross) the draw loop reserved, so the
        # nudge can dodge both collision channels at once. Gated on the flag: the 1:1 path
        # must stay byte-identical to Phase 1, since clean mode is the regression check and
        # ASSUMPTIONS.md #24a's numbers describe that exact code.
        _make_nets_unique(
            cfg, grosses, methods, capture_dates, settled_on_of, groups, taken,
            netted_by_index,
        )

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
        # Per member, then summed -- the same discipline as the fee, and at 10 bps it is not
        # a smaller version of the same trap. Measured over 20,000 synthetic batches drawn
        # from the modal band: per-member and batch-total rounding differ on **32.7%** of
        # batches by at most 2 paise. (.plan/phase6.md correction (a) predicted 0p per member
        # against 1-2p on the total -- "one of them is zero" -- which followed from reading
        # AMOUNT_BANDS as paise. They are rupees, so the smallest gross is 10,000p and its
        # TDS is 10p. See .plan/phase6.md section 9(a).)
        tds = sum(_tds(cfg, p.gross_paise) for p in batch)
        # The refund term, and it is the first one here that is **not computed from a rate**.
        # ``netted_by_index`` was frozen before the uniqueness nudge ran, so this is a lookup
        # of a number that already exists rather than a derivation -- which is exactly the
        # position the matcher is in (``refunds.csv`` declares it; decision 9 says look it up
        # and never search for it). Summed over the batch's members for the same reason the
        # fee is: a batch's terms are the sum of its members' terms, never a rate on the total.
        refunds_total = sum(netted_by_index.get(i, 0) for i in members)
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
                # net = gross - fee - GST - TDS, summed over the members. All three are 0
                # without their flags, so this is ``net == sum of grosses`` in clean mode and
                # the expression stops being trivial exactly when a flag turns on.
                #
                # **TDS is inside the net, and that is not this phase's choice to make.** I4
                # (``invariants.py:352``) has asserted ``net == gross - fee - gst - tds``
                # since Phase 4, and ``settlements.csv`` has carried the column since Phase 1
                # -- so the schema settled which side of the net TDS falls on long before
                # ``--tds`` existed. The consequence is the one worth stating: the credit
                # still equals the net, so ``--tds`` moves **no join at all**. The exact
                # amount band survives, blocking is untouched, and the whole flag lands as a
                # change of *value* in a column the matcher already reads. Refunds get the
                # same treatment in step 6; the reserve cannot, which is why it is last.
                # Phase 6 step 6 puts the refund inside the net too, and ``settlements.csv``
                # gains **no column for it** -- I9 freezes that header, and Appendix A is the
                # authority on it. So the refund is visible to the matcher only through
                # ``refunds.csv``, which cites a ``payment_id``: the matcher joins that to the
                # settlement's membership and sums. The consequence is the one worth stating:
                # like ``--tds``, this flag moves **no join at all** (the credit still equals
                # the net), and unlike ``--tds`` the term is not derivable from any rate, so
                # what it tests is a *lookup* rather than an arithmetic model. The reserve is
                # the flag that cannot be treated this way, which is why it is last.
                net_paise=gross_total - fee - gst - tds - refunds_total,
                fee_paise=fee,
                gst_paise=gst,
                tds_paise=tds,
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

    # --- Step 6b: --netted-refunds materialises the refund rows -----------
    # The amounts were drawn before the nudge; this turns them into ``refunds.csv`` rows and
    # records, per settlement, what truth has to publish. Nothing here draws randomness: the
    # magnitudes are already frozen and the timestamps are *derived*, which is deliberate --
    # a drawn refund time would consume a stream to produce a column no invariant reads and
    # no matcher joins on.
    #
    # **The planted refund cites ``pay_0000``, which this generator never emits** (numbering
    # starts at 1), and its ``created_at`` predates the month. That is the mess as the plan
    # specifies it: the money genuinely left a real settlement, the refund row is genuinely in
    # the file, and nothing in the three input files says *which* settlement it came off. The
    # window is not widened to admit it -- the window is fixed by the posting lag (#15b), and
    # a refund arriving from outside it is the mess rather than a bound to relax.
    refunds: list[Refund] = []
    #: settlement_id -> (the refund ids netted off it, their total in paise)
    refunds_of: dict[str, tuple[list[str], int]] = {}
    #: settlement_id -> the part of that total whose payment is not in this month's file
    unlinked_of: dict[str, int] = {}
    if netted_by_index:
        # One business day after capture, so a refund never predates the sale it reverses.
        # Derived rather than drawn -- see above.
        for seq, (i, amount) in enumerate(sorted(netted_by_index.items()), start=1):
            planted = i in unlinked
            refunds.append(
                Refund(
                    refund_id=ids.refund_id(seq),
                    # The out-of-scope citation. ``ids.payment_id(0)`` is well-formed and
                    # unissued, so the row is syntactically ordinary and referentially orphaned
                    # -- which is exactly the shape a real cross-month refund has in a
                    # single-month export.
                    payment_id=ids.payment_id(0) if planted else payments[i].payment_id,
                    created_at=(
                        datetime(cfg.year, cfg.month, 1, 10, 0, tzinfo=IST)
                        - timedelta(days=5)
                        if planted
                        else payments[i].captured_at + timedelta(days=1)
                    ),
                    amount_paise=amount,
                )
            )
        for idx, s in enumerate(settlements):
            members = groups[idx]
            ids_here = [
                r.refund_id
                for r, i in zip(refunds, sorted(netted_by_index))
                if i in members
            ]
            total = sum(netted_by_index.get(i, 0) for i in members)
            if total:
                refunds_of[s.settlement_id] = (ids_here, total)
            planted_here = sum(unlinked.get(i, 0) for i in members)
            if planted_here:
                unlinked_of[s.settlement_id] = planted_here
                # Truth publishes one ``refunds_paise`` per credit, so a settlement netting
                # both an attributable and an unattributable refund would have a term that is
                # *partly* attributable -- not a state this project's arithmetic has a name
                # for. ``_draw_refunds`` prevents it by excluding whole settlements from the
                # planted draw; this re-derives that property here, from the finished groups,
                # rather than trusting the draw to have got it right.
                #
                # Worth keeping precisely because it has already paid: when the exclusion was
                # by *payment* instead of by settlement, this fired on seed 3 and seed 7 at
                # n=200 the first time a gate ran ``--netted-refunds --batching`` together --
                # a loud crash before a byte was written, which is the failure mode the
                # generator's exit-code contract is built around. An assumption instead of an
                # assertion would have shipped an incoherent refund term quietly.
                assert planted_here == total, (
                    f"{s.settlement_id} nets both an attributable and an unattributable "
                    f"refund ({total - planted_here}p linked, {planted_here}p planted) -- "
                    f"see _draw_refunds on why the planted draw excludes whole settlements"
                )

    # --- Step 7a: --reserve holds part of a payout back --------------------
    # Drawn here, after the settlements are final and before the credits are drafted, which
    # is the only window that works. After the settlements because the held amount is a
    # fraction of a *finished* net (and because a reserve outside the net cannot perturb the
    # uniqueness nudge -- see ``_draw_reserves``). Before the credits because the credit
    # amount **is** ``net - held``, and everything below this line that reads an amount has
    # to read the short one, including the narration echo fixup.
    reserve_held = _draw_reserves(cfg, rng_reserve, settlements)
    # And separated before use, or the flag can plant a silent wrong match rather than a
    # detectable one. This is the single sharpest hazard in Phase 6 and the reasoning is in
    # ``_separate_reserved_amounts``: a reserved credit is the first row here whose true
    # settlement is *absent* from its own candidate pool, so an amount clash yields one
    # confident wrong answer instead of two candidates and an honest abstention.
    _separate_reserved_amounts(settlements, reserve_held)

    # --- Step 6: derive bank credits, 1:1 --------------------------------
    # Same date, same amount, and a narration assembled from parts. Four fields
    # reach the CSV; the linkage below exists only in memory and in truth.json.
    by_payment = {p.payment_id: p for p in payments}
    spare_tails = iter(tails[cfg.n:])
    # The echo fixup has to be *memoised*, or step 5b's work is undone here: a planted pair
    # shares a tail and a net, so if that tail echoes the amount both members enter the loop
    # below and each would draw its own spare -- handing the pair two different narration
    # tails and separating it again. Keyed on ``(tail, credit amount)`` because that pair is
    # what the decision depends on. Clean mode is byte-identical: every tail is distinct
    # there, so every lookup misses and the loop runs exactly as before.
    fixed_tails: dict[tuple[int, int], int] = {}
    drafted_credits: list[tuple[date, int, float, Settlement, str]] = []
    for s in settlements:
        original = int(s.utr.removeprefix("XXXX"))
        # **The amount the bank actually receives, which is the net only when nothing is
        # held.** Every use below -- the echo fixup, the sort key, the emitted row -- reads
        # this rather than ``s.net_paise``, and getting that wrong would be invisible until a
        # rare seed: I7 compares the narration against the *credit's* amount, so a tail
        # cleared against the net could still echo ``net - held`` and fail on a run nobody
        # was looking at. With ``--reserve`` off this is ``s.net_paise`` exactly and every
        # expression below is unchanged.
        credit_amount = s.net_paise - reserve_held.get(s.settlement_id, 0)
        cached = fixed_tails.get((original, credit_amount))
        if cached is None:
            tail = original
            # A tail that happens to equal its own credit's rupee figure would put the
            # amount into the narration, handing the matcher a free join. Swap it for
            # an unused tail rather than let invariant I7 fail on a rare seed.
            while _tail_echoes_amount(tail, credit_amount):
                tail = next(spare_tails)
            fixed_tails[(original, credit_amount)] = tail
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
        # ``credit_amount``, not ``s.net_paise``: the bank statement is ordered by what the
        # bank actually received, and under ``--reserve`` those differ. Sorting on the net
        # while emitting the short amount would order the file by a number that appears
        # nowhere in it -- and since the sort decides credit *numbering*, C0001..C000n would
        # be assigned in an order no reader of the CSV could reconstruct.
        drafted_credits.append((value_date, credit_amount, rng_bank.random(), s, narration))

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
        refund_ids, refunds_here = refunds_of.get(s.settlement_id, ([], 0))
        unlinked_here = unlinked_of.get(s.settlement_id, 0)
        # **An unattributable refund leaves the row ``resolvable: true``, and that is Phase
        # 4b's standard applied rather than the plan's wording followed.** ``.plan/phase6.md``
        # step 6 calls the out-of-window refund "the planted mess", which reads as
        # ``resolvable: false``. It is not, and the test is the one Phase 4b established: could
        # an unbounded, model-free strategy separate this row? It could, and cheaply. The
        # refund's amount *is* in ``refunds.csv``; a matcher that derives fee, GST and TDS from
        # the rates is left with a residual equal to exactly that amount, so the orphan refund
        # is identifiable by matching its declared amount against the unexplained remainder.
        # Claiming unresolvability here would be truth asserting something an exhaustive
        # matcher refutes -- the one outcome the plan's own step 4 rules out.
        #
        # So this matcher's ``REFUND_UNLINKED`` abstention is scored as a **miss**, not as a
        # correct abstention: coverage falls and correctness holds. That is the same shape as
        # Phase 5's ``AMBIGUOUS_MULTI_SUBSET`` rows at n=1000 -- an honest refusal on a row
        # that is resolvable in principle, costing coverage and never correctness -- and the
        # capability that would close it (attribute an orphan refund by its residual) is
        # available to a later phase, declared here rather than quietly built.
        held_here = reserve_held.get(s.settlement_id, 0)
        # **A reserved row stays ``resolvable: true``, and this is the plan's one clause that
        # measurement overturned rather than refined.** ``.plan/phase6.md`` step 7 says to mark
        # these ``resolvable: false`` with ``PARTIAL_SETTLEMENT_PENDING``. That would be false
        # about this data, by the standard Phase 4b established and gate 11 enforces.
        #
        # The test is whether an unbounded, model-free strategy could separate the row. It
        # can, and the channel is the one gate 11 already measured: a reserve moves the
        # credit's *amount* and leaves the settlement's **UTR** alone, so the narration tail
        # still points at exactly one settlement. Measured directly before this was written
        # (``.plan/probe_tail_vs_amount.py``, seed 42, n=200 and n=1000, every implemented flag
        # combination): the tail-only join -- reading no date and no amount -- hits a unique
        # settlement for **100% of gateway credits**, and declared membership then yields the
        # correct payment set for 100% of those. An amount-side wedge does not degrade that
        # channel at all. So the payment set of a reserved credit is identifiable; what is
        # *not* identifiable is the arithmetic, because the held amount appears in no input
        # file. Those are two different claims and ``resolvable`` is about the first.
        #
        # Marking it false would be actively harmful rather than merely imprecise.
        # ``truth_io.is_planted_unresolvable`` is literally ``not resolvable``, and
        # ``metrics._classify`` reads it twice: an abstention there becomes
        # ``CORRECT_ABSTENTION`` instead of ``MISSED``, and a **correct** resolution becomes
        # ``LUCKY_GUESS`` -- the cell ``metrics.py`` calls the cheapest available leak
        # detector. So the wrong flag would inflate the honest-abstention count with separable
        # rows *and* make the leak detector fire on honest work.
        #
        # ``--dup-amounts`` is the precedent in the other direction: to earn
        # ``resolvable: false`` it had to destroy the tail channel too, by forcing both
        # members to share one UTR (step 5b, and gate 11 asserts it). A reserve destroys
        # nothing, so it does not earn the flag. The consequence is the same shape as the
        # orphan refund above: the matcher's ``PARTIAL_SETTLEMENT_PENDING`` abstention scores
        # as a **miss**, coverage falls, correctness holds, and the capability that would
        # close it is declared rather than quietly taken.
        note_parts: list[str] = []
        if planted:
            note_parts.append(
                f"planted unresolvable: another credit shares this value_date "
                f"({value_date.isoformat()}), this amount ({amount}p) and this UTR "
                f"({s.utr}), so no field in the three input files separates them. "
                f"Resolving it requires information outside these files; the only "
                f"correct verdict is an abstention."
            )
        if unlinked_here:
            # A note **without** a reason, which the ``Credit`` contract permits only on a
            # resolvable row -- and that asymmetry is the point. This row is resolvable (the
            # refund's amount is declared, so the residual identifies it), so it may not carry
            # a reason code; but a reader of the answer key still needs to know the money left
            # via an orphan refund. See the block above on why ``resolvable`` stays true here.
            note_parts.append(
                f"{unlinked_here}p of this credit's shortfall is a refund whose "
                f"payment is not in this month's payments.csv (it cites "
                f"{ids.payment_id(0)}, and its created_at predates the window). The "
                f"refund row is present in refunds.csv, so the amount is declared and "
                f"the row is resolvable in principle -- a matcher that derives the "
                f"rate-based terms is left with a residual equal to exactly this "
                f"refund. Abstaining here is honest but scores as a miss, not as a "
                f"correct abstention."
            )
        if held_here:
            note_parts.append(
                f"{held_here}p of this settlement's {s.net_paise}p net was held back as a "
                f"rolling reserve, so the bank credit is short by that amount. The held "
                f"figure appears in **no input file** -- not a column, not a row -- so the "
                f"arithmetic cannot be closed from the inputs and the only honest verdict is "
                f"an abstention (PARTIAL_SETTLEMENT_PENDING). The row is nonetheless "
                f"resolvable: this settlement's UTR is unchanged, so the narration tail still "
                f"identifies it uniquely and its payment set is recoverable. Abstaining is "
                f"correct behaviour and scores as a miss, not as a correct abstention."
            )
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
                    # Phase 6 step 2. Omitting this while ``build`` subtracted TDS from the
                    # net was caught by I4 on the first ``--tds`` run, before any file was
                    # written: ``expected_credit_paise`` came out 85p above the credit on
                    # C0001, which is exactly that settlement's TDS. Worth recording as the
                    # decomposition doing its job -- truth's arithmetic is re-derived from
                    # the terms rather than copied from the amount, so a term the answer key
                    # forgets cannot silently agree with the CSVs.
                    tds_paise=s.tds_paise,
                    # Phase 6 step 6, and the same lesson as ``tds_paise`` above: a term the
                    # answer key forgets while ``build`` subtracts it from the net cannot
                    # silently agree with the CSVs, because I4 re-derives the subtraction from
                    # these six terms and compares it to the emitted amount.
                    refunds_paise=refunds_here,
                    # Phase 6 step 7, and the term that makes the answer key's arithmetic
                    # close on a reserved row. ``expected_credit_paise`` subtracts all six
                    # terms, so with the reserve recorded here I4's per-credit check
                    # (``expected == amount_paise``) holds unchanged -- the credit really is
                    # ``gross - fee - gst - tds - refunds - reserve``.
                    #
                    # **This is the one term in the decomposition that no input file
                    # declares**, and that asymmetry is the mess. The other five are derivable
                    # (a rate on a declared gross) or declared outright (``refunds.csv``); this
                    # one exists only in the answer key. So truth can state the arithmetic
                    # while the matcher provably cannot reproduce it, which is precisely why
                    # the matcher must abstain rather than fit a number to the gap.
                    reserve_paise=held_here,
                ),
                # The refund ids, so truth says *which* refunds composed the term rather than
                # only how much they came to. That is what lets I15 compare term by term
                # instead of in total -- two refunds that sum correctly while being attributed
                # to the wrong settlements are a real error that a total cannot see.
                refunds_netted=list(refund_ids),
                reserve_held_paise=held_here,
                resolvable=not planted,
                reason=str(Reason.AMBIGUOUS_DUPLICATE_AMOUNT) if planted else None,
                # Assembled from parts rather than nested conditionals, because Phase 6 makes
                # two of these conditions **co-occurrable**: one settlement can carry an
                # orphan refund and a reserve at once (the two draws are independent, and
                # nothing forbids the overlap -- unlike the linked/unlinked refund split,
                # which ``_draw_refunds`` keeps disjoint on purpose). The old nested
                # expression could express only one note at a time, so the second condition
                # would have silently overwritten the first in truth.
                note=" ".join(note_parts) if note_parts else None,
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
        refunds=refunds,
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
