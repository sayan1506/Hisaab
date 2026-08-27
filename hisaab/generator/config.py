"""Configuration surface — all thirteen mess flags, every one defaulting to off.

Clean mode (Phase 1) is simply *all of them off*. Declaring the whole surface now
means Phases 4 through 8 flip a boolean instead of refactoring the config.

Nothing here reads the wall clock: the month comes from ``--month`` (decision #7),
because ``date.today()`` would make byte-identity between yesterday's run and
today's run impossible, which quietly destroys the reproducibility claim.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field, fields
from datetime import timedelta, timezone
from pathlib import Path
from typing import ClassVar

#: Fixed UTC+05:30. Deliberately not ``zoneinfo.ZoneInfo("Asia/Kolkata")``: bare
#: Windows Python ships no tzdata and would raise ZoneInfoNotFoundError. IST has
#: never observed DST, so a fixed offset is exactly right, not an approximation.
IST = timezone(timedelta(hours=5, minutes=30), name="IST")

#: Decision #6 / trap 3: clamping capture hours to business hours guarantees the
#: UTC calendar date equals the IST calendar date, so ``captured_at`` and
#: ``settled_on`` cannot disagree by a day for timezone reasons alone. Phase 4
#: relaxes this deliberately if it wants overnight captures.
CAPTURE_HOUR_MIN = 9
CAPTURE_HOUR_MAX = 21

#: Payment methods and their relative weights. The mix exists so the **fee model is
#: genuinely method-dependent**: if every method carried the same rate, ``fee_bps()`` would
#: be a constant function and every per-method argument in this codebase would be untested
#: prose. Phase 4 step 7 checked the rates against Razorpay's published pricing and found
#: they are *mostly* flat (2% on all standard domestic instruments), so the diversity that
#: remains comes from the two rails that genuinely differ -- POS at 0.00% and corporate
#: cards at 2.15% -- rather than from numbers chosen to look varied.
#:
#: ``international_card`` is about the *card's origin*, not its currency: an international
#: card on a domestic merchant settles in INR and simply costs more (up to 3%). Currency
#: conversion is a separate concern and belongs to the declared ``--fx`` flag in Phase 6, so
#: nothing here should be read as pre-empting it.
PAYMENT_METHODS: dict[str, int] = {
    "card": 38,
    "upi": 33,
    "netbanking": 8,
    "wallet": 5,
    "corporate_card": 8,
    "pos_upi": 6,
    "international_card": 2,
}

#: Gross amount bands as (min_rupees, max_rupees, weight). Long-tailed on purpose:
#: a uniform Rs 100-50,000 spread is a tell that the data is fake, and the tail is
#: what makes Phase 9's value-ranked exception queue interesting.
AMOUNT_BANDS: tuple[tuple[int, int, int], ...] = (
    (100, 999, 45),
    (1_000, 4_999, 30),
    (5_000, 19_999, 15),
    (20_000, 99_999, 8),
    (1_00_000, 3_00_000, 2),
)

#: ``--batching``: how many payments settle together, as (size, weight). Phase 5 decision 2
#: -- **size 1 must stay in the distribution.** A distribution that eliminated it would make
#: the tier gate read a *swap* (all Tier 2, no Tier 1) rather than a mix, which is exactly the
#: failure step 2's gate exists to catch: a Tier 1 regression would hide behind a Tier 2
#: success. 60% singletons keeps Tier 1 the majority tier while still batching most of the
#: money, because the tail carries the large amounts.
#:
#: Mean 1.60 payments per settlement, so ``n`` payments produce ~``n/1.6`` bank rows. That
#: divisor is why ``--batching`` raises the default ``n``: measured, every candidate
#: distribution breaches the track's 50-record floor at ``--n 60`` (~37 rows here) and every
#: one clears it at ``--n 200`` (~125 rows here). See ``.plan/phase5.md`` section 1(d).
#:
#: A size is a *ceiling*, not a promise: batches never span a settlement date, so the draw is
#: capped by how many payments remain on that date. At small ``n`` a date holds ~3 payments and
#: the larger sizes are simply unreachable, which is why the effective mean rises with ``n``.
#: The CLI's default record count, and the raised default ``--batching`` resolves to.
#: ``n`` counts **payments**, and batching makes payments and bank rows different numbers:
#: at mean 1.60 payments per settlement, ``--n 60`` yields ~37 bank rows and breaches the
#: track's 50-record floor, while ``--n 200`` yields ~125 and clears it. Measured on every
#: candidate distribution -- see ``.plan/phase5.md`` section 1(d) and decision 3.
#:
#: **Resolved in the CLI, never in ``GenConfig``.** A config that silently rewrote
#: ``n=60`` into 200 whenever ``batching`` was set would change what every caller passing an
#: explicit ``n`` was asking for -- gate 12 scores n=60 deliberately, and ``story.py``'s
#: self-checks assert exact payment counts at a given ``n``. A default belongs to the
#: argument layer; here it would be a value that lies about what was requested.
#: ``--settlement-report-late``: the share of settlements whose membership is withheld from
#: ``settlement_items.csv``. **Partial, never total** -- Phase 5 decision 4. Withholding every
#: settlement would turn the tier distribution into a *swap* rather than a mix, and a Tier 1
#: regression could then hide behind a Tier 2 success; gate 12 refuses both one-sided shapes
#: for that reason. A third is enough to make the search carry real rows while leaving Tier 1
#: the majority tier.
#:
#: The file is still written, with its header and the rows that were not withheld: ``load.py``
#: raises ``LoadError`` on a missing file, so omitting it would fail the run for the wrong
#: reason and read as a loader bug rather than as withheld data (#22 settled the same question
#: for an empty ``refunds.csv``).
LATE_REPORT_SHARE = 0.30

DEFAULT_N = 60
DEFAULT_N_BATCHED = 200

BATCH_SIZE_WEIGHTS: tuple[tuple[int, int], ...] = (
    (1, 60),
    (2, 25),
    (3, 10),
    (4, 5),
)

#: Percent of amounts landing on a whole rupee. The remainder carry paise, which
#: is the variety ``--rounding-edge`` needs in Phase 12.
WHOLE_RUPEE_PERCENT = 70

BANK_CHANNELS: dict[str, int] = {"NEFT": 60, "IMPS": 30, "RTGS": 10}

#: Narration templates (deviation (c) in .plan/phase1.md). Style variance changes
#: no amount and no date, so clean mode stays clean by the definition that
#: matters, while giving Phase 3's parser and Phase 10's LLM real work. Use
#: ``--narration-styles 1`` for a maximally sterile file while debugging.
NARRATION_TEMPLATES: tuple[str, ...] = (
    "{channel}-{counterparty}-XXXX{tail}",
    "{channel} CR/{counterparty_spaced}/{tail}",
    "{channel}/{counterparty}/XXXX{tail}/SETTLEMENT",
    "{channel}-{counterparty_short}-{tail}",
)

COUNTERPARTY = "RAZORPAYSOFT"
COUNTERPARTY_SPACED = "RAZORPAY SOFTWARE"
COUNTERPARTY_SHORT = "RZRPAY"


@dataclass(frozen=True)
class MessFlags:
    """The mess dial. Every flag defaults off; clean mode is this dataclass bare.

    Order matches the build order in section 11 of the track spec, so the CLI
    ``--help`` reads as a difficulty ramp rather than an alphabet soup.
    """

    fees: bool = False                     # Phase 4  amount never matches exactly
    settlement_delay: bool = False         # Phase 4  T+n, weekend skew
    batching: bool = False                 # Phase 5  N payments -> 1 credit
    netted_refunds: bool = False           # Phase 6  credit short, earlier order
    reserve: bool = False                  # Phase 6  credit short, rest arrives later
    tds: bool = False                      # Phase 6  another gross/net wedge
    noise_rows: bool = False               # Phase 7  bank rows to be ignored
    unsettled: bool = False                # Phase 7  payments never paid out
    dup_amounts: bool = False              # Phase 4b planted unresolvable
    fx: bool = False                       # Phase 8  rate moves between capture/settle
    rounding_edge: bool = False            # Phase 8  fee x GST on a half-paisa
    settlement_report_late: bool = False   # Phase 8  withhold settlement_items.csv
    utr_patchy: bool = False               # Phase 8  UTR missing/truncated on some rows

    #: Flags whose data generation is actually implemented in ``story.py``.
    #:
    #: Every other flag is *declared* -- it has a CLI switch, a help line and a slot
    #: in ``truth.json`` -- but ``story.py`` does not read it, so turning it on would
    #: return unchanged data **labelled as having that mess**. That is worse than a
    #: no-op in a specific way: any flag makes ``clean_mode`` false, which used to
    #: switch off the clean-mode invariants, so the mislabelled run also lost the
    #: checks that would have noticed. ``GenConfig`` refuses such a flag outright.
    #:
    #: A phase that implements a flag adds it here, and that edit is the phase
    #: admitting the flag now does something. Phase 4 step 3 added
    #: ``settlement_delay``: ``story.py`` reads ``cfg.delay_days`` and ``cfg.lag_days``,
    #: both of which are gated on it. Step 4 added ``fees``: ``story._deductions`` reads
    #: ``cfg.fees`` and returns ``(0, 0)`` without the flag.
    #:
    #: The order of that edit is the whole discipline. Adding a name here *before*
    #: ``story.py`` reads the flag makes ``GenConfig`` accept a run it should refuse, and
    #: the run then emits unchanged data labelled as having that mess -- which is the
    #: failure this set exists to prevent. Generation first, declaration last.
    #: ``ClassVar`` is load-bearing, not decoration: a bare annotation inside a
    #: dataclass becomes a **field**, so ``IMPLEMENTED: frozenset[str]`` would make
    #: this a 14th mess flag -- breaking ``names()``, ``all_on()``, the CLI's
    #: generated switches and the flag block in ``truth.json``. The self-check below
    #: asserts the count is still 13 for exactly that reason.
    #: Phase 4b added ``dup_amounts``: ``story._plant_dup_pairs`` forces ``dup_pairs``
    #: payment pairs onto one ``(capture_date, gross, method)``, and ``build`` then forces
    #: each pair's two settlements onto one UTR. Both halves are required, and the second is
    #: the load-bearing one -- see ``dup_pairs`` below for the measurement that says why.
    #: Phase 5 step 1 added ``batching``: ``story._group_into_batches`` partitions each
    #: settlement date's payments into settlements of 1-4 members, and ``build`` sums the
    #: deductions per member. Added *after* that code existed, per the paragraph above.
    #: Phase 5 step 5 added ``settlement_report_late``, pulled forward from Phase 8:
    #: ``story._withhold_membership`` picks the settlements whose rows ``emit`` leaves out of
    #: ``settlement_items.csv``. Truth keeps the full membership -- the withholding is a
    #: property of the *files the matcher reads*, not of what happened.
    #: Phase 6 step 2 added ``tds``: ``story._tds`` withholds ``cfg.fees.tds_bps`` on each
    #: member's gross, ``build`` sums it per member into ``tds_paise`` and subtracts it from
    #: ``net_paise``, and ``_batch_net`` counts it so the net-uniqueness nudge protects the
    #: value the file actually carries. Added *after* that code existed, per the paragraph
    #: above. It is the first implemented flag that **draws no randomness at all** -- a
    #: withholding at a declared rate on a declared gross is fully derived, so there is
    #: nothing to draw and the reserved ``tds`` stream stays unused (see ``story._tds``).
    IMPLEMENTED: ClassVar[frozenset[str]] = frozenset(
        {"settlement_delay", "fees", "dup_amounts", "batching", "settlement_report_late",
         "tds"}
    )

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

    @classmethod
    def unimplemented(cls) -> tuple[str, ...]:
        return tuple(n for n in cls.names() if n not in cls.IMPLEMENTED)

    def declared_but_inert(self) -> list[str]:
        """Flags this config turns on that ``story.py`` would silently ignore."""
        return [n for n in self.enabled() if n not in self.IMPLEMENTED]

    @classmethod
    def all_on(cls) -> MessFlags:
        return cls(**{n: True for n in cls.names()})

    def any_on(self) -> bool:
        return any(getattr(self, n) for n in self.names())

    def as_dict(self) -> dict[str, bool]:
        """Every flag, in declaration order. Written verbatim into truth.json."""
        return {n: getattr(self, n) for n in self.names()}

    def enabled(self) -> list[str]:
        return [n for n in self.names() if getattr(self, n)]


@dataclass(frozen=True)
class FeeConfig:
    """Gateway fee model. Rates are integer basis points, never float percents.

    **Checked against Razorpay's published pricing in Phase 4 step 7**, and two of the
    five original assumptions were wrong. Recording what changed, because the status key in
    ``ASSUMPTIONS.md`` exists precisely to make this moment cheap rather than embarrassing:

      * ``netbanking`` was 190 bps on the guess that netbanking is cheaper than cards. The
        page prices all standard domestic instruments at one flat 2%, netbanking included.
      * ``upi`` was **0 bps**, which conflated two different things. UPI carries **zero
        MDR** by mandate, but Razorpay's own 2% *platform fee* still applies on the standard
        payment-gateway rail -- so a UPI sale is not free to the merchant. This was the
        expensive error of the two: it is not a number a reader would question, and 36% of
        rows on a ``--fees`` run were settling at their gross because of it.

    The zero-rated rail survives, but it is now a *verified* one instead of an invented one:
    POS terminals price UPI and RuPay debit at **0.00%**. That matters beyond bookkeeping --
    a zero-deduction path has to stay exercised (``story.py`` and ``matcher/fees.py`` both
    branch on it, and "the residual moved" has to be measured per method or the free rows
    quietly carry the result), and it should be exercised by something true.

    Rates verified 2026-08-26 against https://razorpay.com/pricing/. Two caveats kept
    deliberately visible rather than smoothed away: these are **list prices** and real rates
    are negotiated at volume, and Indian MDR structures move with NPCI/RBI policy. So the
    right reading is "sourced and dated", not "permanently true" -- which is why every rate
    is overridable from the matcher's command line (``--fee-bps METHOD=BPS``).
    """

    fee_bps_by_method: dict[str, int] = field(
        default_factory=lambda: {
            # 2% on all standard domestic instruments -- one flat rate, verified.
            "card": 200,
            "upi": 200,          # zero MDR, but the 2% platform fee still applies
            "netbanking": 200,   # explicitly "not priced differently from cards"
            "wallet": 200,
            # The two rails that genuinely differ, which is where the method-dependence of
            # this model stops being decorative.
            "corporate_card": 215,     # business/corporate credit cards
            "international_card": 300, # up to 3% by card origin, settled in INR
            "pos_upi": 0,              # POS terminals: UPI and RuPay debit at 0.00%
        }
    )
    gst_bps: int = 1800   # 18% GST, charged on the fee, not on the gross -- verified
    #: TDS withheld under **§194-O**, at **10 bps (0.1%)**. Verified 2026-08-27, and the
    #: value it replaces was a full order of magnitude too large: the section ran at 1% from
    #: its introduction on 2020-10-01 (0.75% under the FY 2020-21 relief), and the **Finance
    #: (No. 2) Act 2024 cut it to 0.1% effective 2024-10-01**. The 100 here was simply the
    #: pre-amendment position, carried since Phase 1 because nothing read it until now.
    #:
    #: Worth recording *where* the stale number came from, since this codebase cites
    #: Razorpay's own pages for every fee rate: Razorpay's §194-O explainer still states 1%
    #: and does not mention the amendment. Same publisher, one page verified and one stale --
    #: which is the argument for **dating** a citation rather than merely naming a source.
    #:
    #: **This is a rate, not a scope claim.** Whether a payment aggregator withholds §194-O
    #: on a merchant settlement at all is a separate question from what the rate is, and it
    #: stays a declared modelling assumption rather than a verified fact -- ASSUMPTIONS.md
    #: #9a, which is the entry a reader should be pointed at before this number is defended.
    #:
    #: At 10 bps this is the **smallest** of the deductions (against 36 bps effective for
    #: GST-on-fee and 200 bps for the fee), where at 100 bps it outweighed GST. It does not
    #: vanish: ``AMOUNT_BANDS`` is denominated in **rupees**, so the smallest gross drawable
    #: is ₹100 = 10,000 paise and its TDS is 10 paise. No settlement carries a zero term.
    tds_bps: int = 10

    def fee_bps(self, method: str) -> int:
        return self.fee_bps_by_method[method]


@dataclass(frozen=True)
class GenConfig:
    """Everything one generator run needs. Frozen: a run cannot reconfigure itself."""

    seed: int = 42
    n: int = 60                    # above the 50-record floor, so a default run is submittable-shaped
    year: int = 2026
    month: int = 8
    out_dir: Path = Path("data")
    truth_dir: Path = Path("truth")
    narration_styles: int = len(NARRATION_TEMPLATES)
    flags: MessFlags = field(default_factory=MessFlags)
    fees: FeeConfig = field(default_factory=FeeConfig)

    #: **Two delays, not one**, and conflating them is the trap Phase 4's plan (a)
    #: caught. ``ASSUMPTIONS.md`` #15/#16 describe only the first:
    #:
    #:   settlement cycle   ``captured_at`` -> ``settled_on``   invisible to the matcher
    #:   bank posting lag   ``settled_on``  -> ``value_date``   the ONLY one the window sees
    #:
    #: Tier 1 joins ``credit.value_date`` against ``settlement.settled_on`` and never
    #: reads ``captured_at``, so a pure settlement-cycle delay changes nothing the matcher
    #: can see -- widening ``--window`` for it would prove nothing. The posting lag is what
    #: makes the date window load-bearing, which is why it exists as its own number.
    #:
    #: Both are in **business days** and both take effect only while
    #: ``--settlement-delay`` is on (see ``delay_days``/``lag_days``). That gating is what
    #: keeps clean mode byte-identical to the Phase 1 run: with the flag off these are 0,
    #: ``settled_on`` is the capture date and ``value_date`` equals ``settled_on``, exactly
    #: as before. Clean mode is row 1 of the mess dial and stays the regression check.
    settlement_delay_days: int = 2   # T+2, ASSUMPTIONS.md's stated common default
    posting_lag_days: int = 1        # the credit lands the business day after settlement

    #: How many **planted unresolvable pairs** ``--dup-amounts`` creates (Phase 4b).
    #:
    #: A pair is two payments forced to share a capture date, a gross and a method, so
    #: their settlements derive the same net on the same day and their bank credits collide
    #: on ``(value_date, amount_paise)``. Both credits are marked ``resolvable=False`` in
    #: truth, and the only correct verdict on either is an abstention.
    #:
    #: **The pair also shares one UTR, and that is the load-bearing half.** Measured before
    #: this flag was built (gate 11, ``tools/acceptance.py``): a tail-only strategy that
    #: reads no date and no amount resolves 60/60, 200/200 and 1000/1000 credits *correctly*
    #: on every dev seed, clean and under --fees --settlement-delay alike, because
    #: ``_draw_tails`` samples without replacement. So a pair that collided on
    #: ``(date, amount)`` while keeping distinct tails would still be separable by
    #: exhaustive narration matching -- the flag would not test the capability its name
    #: claims, and ``resolvable=False`` would be a false statement about the data. Sharing
    #: the UTR is what makes every available strategy see a tie.
    #:
    #: Two rather than one, so the ``correct_abstention`` denominator is never 1 -- a rate
    #: of 1/1 is indistinguishable from a coincidence.
    dup_pairs: int = 2

    def __post_init__(self) -> None:
        # Refuse a flag whose data generation does not exist yet. This lives here
        # rather than in the CLI on purpose: constructing GenConfig directly -- from a
        # test, a tool, or a future harness -- must not be a way around it.
        if inert := self.flags.declared_but_inert():
            raise ValueError(
                f"these mess flags are declared but not implemented yet: "
                f"{', '.join('--' + n.replace('_', '-') for n in inert)}\n"
                f"  story.py does not read them, so the run would emit *unchanged* "
                f"data labelled as having that mess -- and because any flag makes "
                f"clean_mode false, the run would also skip the clean-mode invariants "
                f"that would have caught it. A wrong answer that looks like a real "
                f"result is the failure mode this project is built to prevent.\n"
                f"  Implemented today: "
                f"{', '.join(sorted(MessFlags.IMPLEMENTED)) or '(none -- clean mode only)'}. "
                f"A phase that implements a flag adds it to MessFlags.IMPLEMENTED."
            )
        if self.n < 1:
            raise ValueError(f"--n must be at least 1, got {self.n}")
        if not 1 <= self.month <= 12:
            raise ValueError(f"--month has an invalid month: {self.month}")
        # Forward-only, matching ``bizdays.add_business_days``. A negative delay would
        # settle money before it was captured, and the calendar asserts rather than
        # guessing at what backwards T+n means.
        if self.settlement_delay_days < 0:
            raise ValueError(
                f"--settlement-delay-days must be >= 0, got {self.settlement_delay_days}"
            )
        if self.posting_lag_days < 0:
            raise ValueError(
                f"--posting-lag-days must be >= 0, got {self.posting_lag_days}"
            )
        # A magnitude moved off its default while the flag that reads it is off. This is
        # the same class of mistake as a declared-but-inert mess flag above -- the run
        # would accept the number, ignore it, and describe the output in
        # ``run_manifest.json`` using a delay that never happened.
        if not self.flags.settlement_delay:
            declared = {f.name: f.default for f in fields(self)}
            ignored = [
                name
                for name in ("settlement_delay_days", "posting_lag_days")
                if getattr(self, name) != declared[name]
            ]
            if ignored:
                raise ValueError(
                    f"{', '.join('--' + n.replace('_', '-') for n in ignored)} was set, "
                    f"but --settlement-delay is off, so nothing reads it: settled_on would "
                    f"stay the capture date and value_date would stay equal to settled_on. "
                    f"Pass --settlement-delay to make the magnitude take effect."
                )
        # The same class of check as the delay magnitudes above: a number that is read only
        # under a flag, moved while that flag is off, would be accepted, ignored, and then
        # described in ``run_manifest.json`` as though it had taken effect.
        if not self.flags.dup_amounts:
            if self.dup_pairs != next(
                f.default for f in fields(self) if f.name == "dup_pairs"
            ):
                raise ValueError(
                    f"--dup-pairs was set to {self.dup_pairs}, but --dup-amounts is off, so "
                    f"nothing reads it: no pair would be planted and every credit would "
                    f"stay resolvable. Pass --dup-amounts to make the count take effect."
                )
        else:
            if self.dup_pairs < 1:
                raise ValueError(
                    f"--dup-pairs must be at least 1 when --dup-amounts is on, got "
                    f"{self.dup_pairs} -- a run that plants nothing while claiming to plant "
                    f"is the mislabelled-data failure MessFlags.IMPLEMENTED exists to prevent."
                )
            # Each pair consumes two payments, and a pair is only a pair if both members
            # exist. Refused here rather than raising an opaque ``sample larger than
            # population`` from ``random.sample`` deep inside the build.
            if 2 * self.dup_pairs > self.n:
                raise ValueError(
                    f"--dup-pairs {self.dup_pairs} needs {2 * self.dup_pairs} payments to "
                    f"plant {self.dup_pairs} colliding pair(s), but --n is {self.n}. "
                    f"Raise --n or lower --dup-pairs."
                )
        # ``--dup-amounts`` and ``--batching`` do not compose, so the combination is refused
        # rather than emitting a plant that is not a plant.
        #
        # A planted pair is unresolvable because two settlements share one net, one date and
        # one UTR. Batching either member with other payments makes its net a *sum*, the two
        # nets diverge, and the pair becomes separable -- at which point ``resolvable=False``
        # is a false statement about the data, which is the exact failure Phase 4b exists to
        # close. Invariant I12 does catch it downstream (the planted count comes out at 0),
        # but it reports a symptom; this names the cause.
        #
        # Deliberately **not** fixed by forcing planted payments into singleton settlements:
        # membership would then correlate with plantedness, handing a matcher "the settlements
        # holding exactly one payment are the ambiguous ones" as a structural tell. That is a
        # worse leak than the one being avoided and a self-inflicted one. The honest options
        # are to design the combination properly or to refuse it, and a combined run buys
        # nothing Phase 5 needs.
        if self.flags.dup_amounts and self.flags.batching:
            raise ValueError(
                "--dup-amounts and --batching cannot be combined: batching a planted "
                "payment with others makes its settlement's net a sum, so the pair no "
                "longer shares one net and stops being unresolvable -- truth would then "
                "claim resolvable=false about data that is separable. Run them separately; "
                "gate 11 covers the planted rows and gate 12 covers batching."
            )
        # Withholding must be **partial** (decision 4), so the flag needs at least two
        # settlements to have a choice: withholding the only one is total by definition, and a
        # total withholding turns the tier distribution into a swap rather than a mix. Refused
        # here rather than silently withholding nothing, which is the mess-flag footgun this
        # codebase refuses by name -- a run labelled as having a mess it does not have.
        #
        # ``n < 2`` is the part knowable from the config alone: 1:1 gives exactly ``n``
        # settlements and batching can only reduce that count, so ``n == 1`` is always one
        # settlement. Small ``n`` *with* batching can also land on one settlement as a matter
        # of the draw (measured: n=2 on 1 of 60 seeds), which no config-time check can see --
        # ``story.build`` carries the backstop for that.
        if self.flags.settlement_report_late and self.n < 2:
            raise ValueError(
                f"--settlement-report-late needs at least 2 settlements so that the "
                f"withholding can be partial, but --n {self.n} produces one. Raise --n."
            )
        if not 1 <= self.narration_styles <= len(NARRATION_TEMPLATES):
            raise ValueError(
                f"--narration-styles must be 1..{len(NARRATION_TEMPLATES)}, "
                f"got {self.narration_styles}"
            )

    @property
    def month_label(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    @property
    def clean_mode(self) -> bool:
        return not self.flags.any_on()

    @property
    def delay_days(self) -> int:
        """Business days from capture to settlement -- **0 unless the flag is on**.

        The gate lives here rather than at the call site so ``story.py`` can route
        through the calendar unconditionally: ``add_business_days(d, 0)`` returns ``d``
        for a business day, and every capture date is one. So clean mode gets the delay
        code path exercised at zero magnitude while staying byte-identical to Phase 1 --
        which matters, because clean mode is row 1 of the mess dial and a regression
        there is the one failure that invalidates every later number.
        """
        return self.settlement_delay_days if self.flags.settlement_delay else 0

    @property
    def lag_days(self) -> int:
        """Business days from settlement to the bank credit landing. 0 unless the flag is on.

        Separate from ``delay_days`` because this is the only one the matcher's date
        window can see -- see the field comments above.
        """
        return self.posting_lag_days if self.flags.settlement_delay else 0

    @property
    def planted_pairs(self) -> int:
        """Colliding pairs actually planted -- **0 unless ``--dup-amounts`` is on**.

        Same gate-at-the-property shape as ``delay_days``, and for a sharper reason: this
        number is the *denominator* of the ``correct_abstention`` rate, which is the cell
        carrying this project's central claim. Reporting the declared 2 on a run that
        planted nothing would describe an answer key that does not exist.
        """
        return self.dup_pairs if self.flags.dup_amounts else 0

    def resolved(self) -> dict[str, object]:
        """The resolved-config echo.

        The CLI prints this as one JSON line on stdout and it goes into
        ``truth/run_manifest.json``; Phase 11's report header consumes it verbatim.
        """
        return {
            "seed": self.seed,
            "n": self.n,
            "month": self.month_label,
            "out_dir": str(self.out_dir),
            "truth_dir": str(self.truth_dir),
            "narration_styles": self.narration_styles,
            "clean_mode": self.clean_mode,
            # The *effective* delays, not the declared fields: both are 0 while
            # --settlement-delay is off. Reporting the declared 2 and 1 on a clean run
            # would describe a delay that did not happen, and this object is what
            # run_manifest.json and Phase 11's report header quote verbatim.
            "settlement_delay_days": self.delay_days,
            "posting_lag_days": self.lag_days,
            # Effective, for the same reason as the delays: 0 on any run that planted
            # nothing. This is the correct_abstention denominator, so a reader comparing
            # two runs' exception lists needs it stated rather than inferred.
            "planted_pairs": self.planted_pairs,
            "flags_enabled": self.flags.enabled(),
            "flags": self.flags.as_dict(),
        }


if __name__ == "__main__":
    cfg = GenConfig()
    assert cfg.clean_mode, "the default config must be clean mode"
    assert cfg.n >= 50, "default --n must clear the 50-record floor"
    assert cfg.month_label == "2026-08"
    assert len(MessFlags.names()) == 13, MessFlags.names()
    assert MessFlags().as_dict() == dict.fromkeys(MessFlags.names(), False)
    assert MessFlags().enabled() == []
    assert MessFlags.all_on().any_on()
    assert len(MessFlags.all_on().enabled()) == 13

    # IMPLEMENTED must stay a ClassVar, never a field. Spelled as a bare annotation it
    # would become a 14th mess flag: names() would return 14, the CLI would generate an
    # --implemented switch, and truth.json's flag block would gain a non-boolean entry.
    assert "IMPLEMENTED" not in MessFlags.names(), "IMPLEMENTED leaked into the fields"
    assert isinstance(MessFlags.IMPLEMENTED, frozenset)
    assert MessFlags.IMPLEMENTED <= set(MessFlags.names()), "IMPLEMENTED names a flag that does not exist"

    # --- step 1: a declared-but-inert flag is refused -------------------------
    # Phase 4 step 3 moved ``settlement_delay`` into IMPLEMENTED, so this is no longer
    # "every flag is refused" -- it is "every flag story.py does not read is refused",
    # which is the assertion that keeps meaning something as the phases land. Driving it
    # off ``unimplemented()`` rather than a hand-written list is what makes it inverting
    # automatically instead of going stale.
    assert MessFlags().declared_but_inert() == []
    assert MessFlags(netted_refunds=True).declared_but_inert() == ["netted_refunds"]
    for _landed in ("settlement_delay", "fees", "dup_amounts", "batching"):
        assert _landed not in MessFlags.unimplemented()
        assert MessFlags(**{_landed: True}).declared_but_inert() == [], (
            f"{_landed} is implemented, so it must not be reported inert"
        )
    for flag in MessFlags.unimplemented():
        try:
            GenConfig(flags=MessFlags(**{flag: True}))
        except ValueError as e:
            assert flag.replace("_", "-") in str(e), f"{flag}: refusal must name the flag"
        else:
            raise AssertionError(
                f"--{flag.replace('_', '-')} was accepted but story.py does not "
                f"implement it, so the run would be mislabelled"
            )

    # --- step 3: the two delays -----------------------------------------------
    # An implemented flag is now reachable through the public constructor, so clean_mode
    # =False needs no patching seam to exercise. The seam is still used below for a flag
    # that is *not* implemented yet, because that path still has to work.
    _delayed = GenConfig(flags=MessFlags(settlement_delay=True))
    assert not _delayed.clean_mode, "a flag that is on must leave clean mode"
    assert _delayed.resolved()["flags_enabled"] == ["settlement_delay"]
    # The magnitudes are live under the flag and zero without it. Both halves matter: the
    # first is the delay model working, the second is what keeps clean mode byte-identical
    # to the Phase 1 run.
    assert (_delayed.delay_days, _delayed.lag_days) == (2, 1)
    assert (GenConfig().delay_days, GenConfig().lag_days) == (0, 0)
    # resolved() must echo the *effective* delays, not the declared fields -- otherwise a
    # clean run's manifest describes a T+2 settlement that never happened.
    assert GenConfig().resolved()["settlement_delay_days"] == 0
    assert GenConfig().resolved()["posting_lag_days"] == 0
    assert _delayed.resolved()["settlement_delay_days"] == 2
    assert _delayed.resolved()["posting_lag_days"] == 1
    # A magnitude set while the flag is off is refused rather than silently ignored --
    # the same rule as a declared-but-inert flag, and the refusal must name the switch.
    for kwargs in ({"settlement_delay_days": 5}, {"posting_lag_days": 3}):
        try:
            GenConfig(**kwargs)  # type: ignore[arg-type]
        except ValueError as e:
            key = next(iter(kwargs))
            assert key.replace("_", "-") in str(e), f"{key}: refusal must name the switch"
        else:
            raise AssertionError(f"GenConfig accepted {kwargs} with --settlement-delay off")
    # Forward-only, matching the calendar's own assertion.
    for bad in ({"settlement_delay_days": -1}, {"posting_lag_days": -1}):
        try:
            GenConfig(flags=MessFlags(settlement_delay=True), **bad)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError(f"GenConfig accepted {bad}")
    # A zero lag under the flag is legitimate: same-day posting, T+n settlement only.
    assert GenConfig(flags=MessFlags(settlement_delay=True), posting_lag_days=0).lag_days == 0

    # --- step 4: the fee model ------------------------------------------------
    # The rates are ASSUMPTIONS (ASSUMPTIONS.md #5-#9), so what is asserted here is their
    # *shape*, never their values -- pinning 200 bps would turn a documented assumption
    # into a test that has to be edited when the assumption is corrected.
    _fees = FeeConfig()
    assert set(_fees.fee_bps_by_method) == set(PAYMENT_METHODS), (
        "every payment method needs a fee rate, or story._deductions raises KeyError "
        "on the first payment of that method"
    )
    assert all(bps >= 0 for bps in _fees.fee_bps_by_method.values())
    # GST is charged on the fee, so a rate at or above 100% would make GST exceed the fee
    # it sits on -- the shape invariant behind invariants.check_settlement_arithmetic.
    assert 0 <= _fees.gst_bps < 10_000, "GST is a share of the fee, not a multiple of it"
    assert 0 <= _fees.tds_bps < 10_000
    # A zero-rate method is legitimate and load-bearing: pos_upi settles at its gross, so
    # its residual is zero even under --fees. "The residual moved" has to be measured per
    # method rather than in aggregate, and this is the assertion that says why.
    #
    # Asserted as "some method", not "pos_upi specifically", on purpose. The property that
    # matters is that the zero-deduction branch stays reachable; *which* rail is free is a
    # rate, and rates get corrected. This assertion did survive that correction: it was
    # written when upi was the free method, and it still holds now that upi is priced at 200
    # and POS carries the zero. Naming the method would have made it a test to edit.
    assert any(bps == 0 for bps in _fees.fee_bps_by_method.values()), (
        "at least one method is expected to be zero-rated; if that changes, the per-method "
        "residual reasoning in story._deductions needs revisiting"
    )
    _feed = GenConfig(flags=MessFlags(fees=True))
    assert not _feed.clean_mode
    assert _feed.resolved()["flags_enabled"] == ["fees"]
    # --fees and --settlement-delay are independent: the amount wedge and the date wedge
    # must be switchable one at a time, which is the whole point of the mess dial.
    assert (_feed.delay_days, _feed.lag_days) == (0, 0), (
        "--fees alone must not move a single date"
    )

    # The IMPLEMENTED seam still works for a flag no phase has landed yet. This probe was
    # written against ``batching``; Phase 5 landed it, so the probe moved to
    # ``netted_refunds`` (Phase 6) rather than being deleted -- the seam has to keep working
    # for as long as *any* flag is still declared and inert, and the moment it tests a landed
    # flag it tests nothing. Move it again, do not delete it, when Phase 6 lands.
    _original = MessFlags.IMPLEMENTED
    try:
        MessFlags.IMPLEMENTED = _original | {"netted_refunds"}
        assert MessFlags(netted_refunds=True).declared_but_inert() == []
        assert GenConfig(flags=MessFlags(netted_refunds=True)).resolved()["flags_enabled"] == [
            "netted_refunds"
        ]
    finally:
        MessFlags.IMPLEMENTED = _original
    assert MessFlags.IMPLEMENTED == _original, "the probe must not leak"
    assert "netted_refunds" in MessFlags.unimplemented(), "netted_refunds lands in Phase 6"
    assert "batching" in MessFlags.IMPLEMENTED, "Phase 5 step 1 implements batching"
    assert MessFlags.unimplemented(), (
        "no flag is unimplemented any more -- the seam above is testing nothing, and the "
        "declared-but-inert refusal in GenConfig has no case left to catch"
    )
    # --- Phase 5 step 1: the batch size distribution --------------------------
    # Shape, never the exact weights: the weights are a tuning choice and pinning them would
    # make this a test to edit. What must hold is that size 1 stays in play (decision 2 --
    # otherwise the tier gate reads a swap rather than a mix) and that some size above 1 does
    # too, or --batching is an expensive no-op.
    assert any(size == 1 for size, w in BATCH_SIZE_WEIGHTS if w), (
        "size 1 must stay in the distribution, or Tier 1 has no rows left and the "
        "tier-distribution gate reads a tier *swap* instead of a mix -- which would hide a "
        "Tier 1 regression behind a Tier 2 success (Phase 5 decision 2)"
    )
    assert any(size > 1 for size, w in BATCH_SIZE_WEIGHTS if w), "--batching must batch something"
    assert all(size >= 1 and w >= 0 for size, w in BATCH_SIZE_WEIGHTS)
    _batched = GenConfig(flags=MessFlags(batching=True))
    assert not _batched.clean_mode
    assert _batched.resolved()["flags_enabled"] == ["batching"]
    # --batching alone moves no date and no amount: it changes cardinality only. Same
    # one-variable-at-a-time discipline that put --settlement-delay before --fees.
    assert (_batched.delay_days, _batched.lag_days) == (0, 0), "--batching must move no date"
    # --dup-amounts + --batching is refused: batching a planted payment makes its net a sum,
    # so the pair stops colliding and truth's resolvable=false becomes a false statement.
    try:
        GenConfig(flags=MessFlags(dup_amounts=True, batching=True))
    except ValueError as e:
        assert "dup-amounts" in str(e) and "batching" in str(e), (
            f"the refusal must name both flags: {e}"
        )
    else:
        raise AssertionError("GenConfig accepted --dup-amounts with --batching")
    # Each flag alone stays legal, or the refusal above is too broad.
    assert GenConfig(flags=MessFlags(dup_amounts=True)).planted_pairs == 2
    assert GenConfig(flags=MessFlags(batching=True)).planted_pairs == 0

    assert sum(w for *_, w in AMOUNT_BANDS) == 100, "amount band weights must sum to 100"
    assert len(NARRATION_TEMPLATES) == 4
    # Bands must be ordered and non-overlapping, or "long tail" is a lie.
    for (lo, hi, _), (nlo, _, _) in zip(AMOUNT_BANDS, AMOUNT_BANDS[1:]):
        assert lo <= hi < nlo, f"bands overlap or are unordered near {lo}-{hi}"
    # Validation must reject, not silently clamp.
    for kwargs in ({"n": 0}, {"month": 13}, {"narration_styles": 0}, {"narration_styles": 9}):
        try:
            GenConfig(**kwargs)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError(f"GenConfig accepted {kwargs}")
    # IST is a fixed offset with no tzdata dependency.
    assert IST.utcoffset(None) == timedelta(hours=5, minutes=30)
    # The capture-hour clamp must keep the UTC date equal to the IST date.
    from datetime import datetime
    for hour in (CAPTURE_HOUR_MIN, CAPTURE_HOUR_MAX):
        ist = datetime(2026, 8, 10, hour, 0, tzinfo=IST)
        assert ist.astimezone(timezone.utc).date() == ist.date(), hour
    assert calendar.monthrange(2026, 8)[1] == 31
    print("config.py self-check ok")
