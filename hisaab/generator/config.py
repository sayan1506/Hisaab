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

#: ``--netted-refunds``: the share of payments that carry a refund netted off the settlement
#: their payment belongs to. Phase 6 step 6.
#:
#: **A refund reduces the settlement holding its own payment**, which is the modelling
#: simplification worth stating rather than hiding: a real gateway nets a refund against
#: whatever payout is open when the refund clears, which is usually a *later* one. The plan
#: settles this by extending I4 to six terms **per settlement**
#: (``net == gross - fee - gst - tds - refunds``), and that arithmetic only closes if the
#: refund sits on the settlement whose gross it is deducted from. Cross-settlement netting is
#: a different mess -- it would need truth to carry a refund-to-settlement map the CSVs do not
#: have -- and it belongs to a later phase if it is ever wanted.
REFUND_SHARE = 0.10

#: Refund magnitude, as basis points of the refunded payment's gross, drawn uniformly in
#: ``[lo, hi]``. **Partial, never full**, and the bound is load-bearing rather than tidy:
#: ``Settlement.__post_init__`` asserts ``net_paise > 0``, so a refund at or near 100% of a
#: singleton settlement's gross would drive its net to zero or below and fail before anything
#: reached disk. Capped well under 10,000 bps so a singleton batch always keeps a positive net
#: after the fee, GST and TDS have already been taken out.
REFUND_BPS_BAND: tuple[int, int] = (2_000, 6_000)

#: What fraction of settlements have part of their payout **held back** (``--reserve``).
#:
#: A rolling reserve is a real gateway practice: a percentage of each payout is retained
#: against future chargebacks and released after a fixed period. Modelled here as design B --
#: the reserve stays **outside** ``net_paise``, so the settlement declares its full net and the
#: bank credit arrives *short*. See ``story._draw_reserves`` for why that is the only shape
#: that makes the mess exist at all, and ``.plan/phase6.md`` decision 4.
#:
#: Partial for the reason every other share in this file is partial: a run must contain both
#: reserved and unreserved settlements, or "the reserved rows" and "every row" are the same set
#: and nothing downstream can tell a reserve-aware matcher from one that widened its tolerance
#: until everything fit.
RESERVE_SHARE = 0.08

#: Reserve magnitude, as basis points of the settlement's **net**, drawn uniformly in
#: ``[lo, hi]``. 5%-20%, which brackets the rolling-reserve rates gateways actually publish.
#:
#: The lower bound is the load-bearing one, and it is not about realism. The diagnostic in
#: ``matcher.tier1`` distinguishes "money was held back" from "the fee rates assumed here are
#: wrong" by the *size* of the shortfall, so a reserve has to be unmistakably larger than any
#: rounding or rate discrepancy. The largest rate-model error this data can produce is the
#: fee's own rounding divergence -- a few paise on a batch -- so 500 bps clears it by orders of
#: magnitude. A reserve of a few paise would be indistinguishable from a rounding bug, and
#: truth would be asserting a mess the inputs cannot support.
#:
#: The upper bound keeps the credit positive: ``Credit.__post_init__`` asserts
#: ``amount_paise > 0``, and at 20% of net the remaining 80% is comfortably positive.
RESERVE_BPS_BAND: tuple[int, int] = (500, 2_000)

#: Share of payments that are captured but **never paid out** (``--unsettled``, Phase 7).
#:
#: Partial for the reason every other share in this file is partial, and here the argument is
#: sharper than usual: if most payments never settled, "the unsettled ones" and "every payment"
#: would be the same set, and nothing downstream could tell a matcher that handles orphans from
#: one that has no concept of them.
#:
#: **2% rather than anything larger, because this share is paid for in Tier 2 enumeration.** An
#: orphan is claimed by no settlement, so ``tier1._tier2_pool``'s partition filter never removes
#: it and it sits in every pool whose window covers its capture date. Seed 1 at n=1000 had one
#: payment of headroom under the old cap of 64 (pool max 63); at this share its measured pool is
#: **65**, and eleven of its withheld settlements present a pool above 64 -- which is what forced
#: ``MAX_POOL`` to 80 (see ``matcher/tier2.py`` and ASSUMPTIONS.md row 23b).
#:
#: **The share and that bound are coupled tightly, and the curve is not smooth.** Measured with
#: this constant patched (`.plan/probe_phase7_growth_real.py`): a 5% share already reaches 79 on
#: seed 3, and a 15% share breaches the cap at 84. Growth is also non-monotonic -- seed 3 runs
#: 60 -> 79 -> 71 -> 63 across 2/5/10/15% -- because orphaning a settlement's only member deletes
#: that settlement and moves which date holds the maximum. So raising this share means
#: **re-measuring** the pool rather than interpolating, and an over-cap refusal carries
#: ``MEMBERSHIP_UNDECLARED``, which fails the acceptance gates rather than costing coverage.
UNSETTLED_SHARE = 0.02

#: Payments captured in a foreign currency (``--fx``, Phase 8), as a share of ``n``.
#:
#: 8% rather than a smaller fringe, and the reason is I17 rather than realism. ``--fx`` and
#: ``--unsettled`` draw independently (decision 7 and ``story._draw_fx``), so an orphan can
#: land on a foreign payment by coincidence -- and I17 asserts
#: ``orphan_currencies <= settled_currencies``, which fails the moment *every* holder of some
#: currency is an orphan. At this share n=60 draws ~5 foreign payments against ~1 orphan, so
#: the failure needs all five to be orphaned at once. The margin is what makes the interaction
#: safe; it is **not** safe by construction, which is why step 4 measures it rather than
#: assuming it (`.plan/phase8.md` trap 2, third bullet, predicts this from the other side).
FX_SHARE = 0.08

#: The foreign currencies drawn from -- **one**, and the single element is a decision.
#:
#: A wider tuple multiplies I17's coincidence rather than adding realism: with four currencies
#: over ~5 foreign payments most currencies have exactly one holder, and orphaning that holder
#: fires I17 legitimately -- roughly 8% of ``--fx --unsettled`` runs at n=60, by the arithmetic
#: above. One currency needs only one *settled* holder to satisfy the subset, which the share
#: above makes near-certain.
#:
#: It also keeps the flag honest about what it is. Decision 8: the mess is the undeclared
#: **movement** of a rate between capture and settlement, not a currency portfolio -- a rate
#: table is a rate the matcher can subtract, which closes the gap by construction. A single
#: foreign currency carries the movement with nothing extra to declare.
#:
#: Widening this re-opens the I17 coincidence and needs re-measuring, not interpolating --
#: the same coupling ``UNSETTLED_SHARE`` documents against ``MAX_POOL``.
FX_CURRENCIES: tuple[str, ...] = ("USD",)

#: How far the rate moves between capture and settlement, in basis points of the recorded gross,
#: as an unsigned magnitude. The **sign is drawn separately**, because a rate moves both ways and
#: a one-directional flag would let a matcher close the gap by always guessing the same way.
#:
#: 10-200 bps over a T+2 window is the realistic span for USD/INR, and the bound that matters is
#: the lower one: the move has to be unmistakably larger than the arithmetic it must not be
#: confused with. The largest rounding divergence this generator produces is **1-2 paise** on a
#: settlement (measured, see ``story._batch_net``), and the smallest gross in ``AMOUNT_BANDS`` is
#: 10,000p -- so 10 bps on the smallest row is 10p, an order of magnitude above the rounding
#: floor. Below that the FX gap would be indistinguishable from a fee-model rounding slip, and
#: truth would be asserting a mess the inputs cannot support. Same standard as
#: ``RESERVE_BPS_BAND``'s 100 bps floor, at a tenth the size because this term is not competing
#: with a diagnostic band.
#:
#: The ceiling is bounded by what keeps a settlement's net positive after every deduction: at
#: 200 bps down against ~236 bps of fee plus GST, a settlement still pays out ~95% of its gross.
FX_BPS_BAND: tuple[int, int] = (10, 200)

#: Bank rows that are **not gateway money at all** (``--noise-rows``, Phase 7), as a share of
#: the gateway credits a run produces. A real statement carries the business's whole banking
#: life; the reconciler's first job is deciding which rows are even in scope.
#:
#: 6% of credits, so the count tracks the bank file rather than ``n`` -- under ``--batching``
#: those differ by ~1.6x, and a share of ``n`` would make noise a *third* of a batched
#: statement instead of a fringe. Partial for the reason every share here is partial.
NOISE_SHARE = 0.06

#: Noise amounts, in whole rupees, drawn uniformly. Overlaps the gateway amount bands on
#: purpose: a noise row separable by *size* would be separable without reading the narration
#: at all, and the whole point of the flag is that scope is a narration question.
NOISE_AMOUNT_BAND_RUPEES: tuple[int, int] = (500, 50_000)

#: The three strata, and the split is **allocated by count rather than drawn**, which is the
#: load-bearing choice (`.plan/phase7.md` decision 2).
#:
#: ``noise_recall``'s floor is the plainly-foreign share and nothing more, and gate 14 asserts
#: it. A *randomly* drawn split makes that floor wobble seed to seed, so the gate would have to
#: assert a number loose enough to survive the wobble -- which is a weaker gate for no benefit.
#: Allocated by count, the share is exact, truth publishes the realised counts, and the floor is
#: fixed before any matcher runs.
#:
#: Why these three and not one:
#:
#:   * **plainly_foreign** -- vendor, salary or interest text, **no gateway counterparty and no
#:     settlement-hitting tail**. The only stratum the ``IGNORED`` rule can catch, so the only
#:     one the floor may count.
#:   * **gateway_plausible** -- a gateway counterparty spelling with **no resolvable tail** (the
#:     masked ``XXXX`` form, no digit run at all).
#:   * **look_alike** -- a gateway counterparty **and** a 4-digit tail that hits no settlement's
#:     UTR.
#:
#: The last two are why the flag means anything, and they are **deliberately not separable**.
#: Decision 4 ignores a row only when it carries neither a settlement-hitting tail nor a gateway
#: counterparty, and both of these carry the counterparty -- so both fall through to the reserve
#: diagnostic and score as ``NOISE_MISHANDLED``. That is not a gap to be closed by relaxing the
#: rule: under Phase 8's ``--utr-patchy`` a *genuine* gateway credit loses its tail and looks
#: exactly like ``gateway_plausible``, so a rule where "no tail" alone sufficed would convert
#: this phase's ``noise_recall`` into next phase's ``WRONG_IGNORE`` -- money dropped from the
#: books instead of merely left unexplained. The honest consequence is a recall near the
#: plainly-foreign share, reported rather than engineered upward.
#:
#: A recall of 100% is therefore a **finding that the strata are too easy**, never a win.
NOISE_STRATA_SPLIT: dict[str, float] = {
    "plainly_foreign": 0.40,
    "gateway_plausible": 0.30,
    "look_alike": 0.30,
}

#: Counterparty text for ``plainly_foreign`` rows. Authored rather than generated, so I7's leak
#: audit is a real exposure here: no ``pay_``/``setl_``/``C``-prefixed identifier shape, and no
#: digit run that could equal a row's own amount (the tail is the only digits these carry, and
#: ``story._draw_noise_rows`` re-draws it if it echoes).
#:
#: None of these contain ``RAZORPAY`` or ``RZRPAY`` in any spelling -- that is what makes the
#: stratum plainly foreign, and ``invariants.check_noise`` asserts it rather than trusting it.
NOISE_FOREIGN_COUNTERPARTIES: tuple[str, ...] = (
    "ACME SUPPLIES LTD",
    "KIRANA WHOLESALE",
    "MONTHLY SALARY",
    "OFFICE RENT",
    "GST REFUND",
    "TERM DEPOSIT INT",
    "ELECTRICITY BOARD",
    "COURIER SERVICES",
)

#: Templates for the three strata. ``{tail}`` is a 4-digit reference; ``gateway_plausible``
#: takes none, which is what makes its parsed ``ref_tail`` ``None``.
NOISE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "plainly_foreign": (
        "{channel}-{counterparty}-XXXX{tail}",
        "{channel} CR/{counterparty}/{tail}",
        "{channel}/{counterparty}/REF{tail}",
    ),
    # No digit run anywhere: ``normalize.parse`` reads the last run as the tail, so a masked
    # reference is how a row carries the gateway's name while offering nothing to join on.
    "gateway_plausible": (
        "{channel}-{counterparty}-XXXX",
        "{channel} CR/{counterparty}/REF",
        "{channel}/{counterparty}/SETTLEMENT",
    ),
    "look_alike": (
        "{channel}-{counterparty}-XXXX{tail}",
        "{channel} CR/{counterparty}/{tail}",
        "{channel}/{counterparty}/XXXX{tail}/SETTLEMENT",
    ),
}

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

#: Which counterparty spelling each ``NARRATION_TEMPLATES`` entry carries, by index. Declared
#: rather than parsed back out of the rendered string: ``--utr-patchy`` has to re-render a
#: narration in a *different* template while keeping this row's spelling, and recovering the
#: spelling from the output would mean matching "RAZORPAYSOFT" inside "RAZORPAY SOFTWARE" --
#: a prefix relation that ``normalize.py`` already documents as an ordering trap. The
#: self-check below asserts this against what the templates actually render, so the two cannot
#: drift.
NARRATION_SPELLINGS: tuple[str, ...] = (
    COUNTERPARTY,         # {channel}-{counterparty}-XXXX{tail}
    COUNTERPARTY_SPACED,  # {channel} CR/{counterparty_spaced}/{tail}
    COUNTERPARTY,         # {channel}/{counterparty}/XXXX{tail}/SETTLEMENT
    COUNTERPARTY_SHORT,   # {channel}-{counterparty_short}-{tail}
)

#: Share of **gateway** credits whose bank narration loses its reference tail. ``--utr-patchy``,
#: Phase 8 decision 8: the *narration's* tail, never ``settlements.csv``'s ``utr`` column, whose
#: survival is what makes truth's ``resolvable: true`` claim honest for an FX or reserved row.
#:
#: 0.15 rather than something smaller because this flag's entire purpose is to attack
#: ``WRONG_IGNORE``, and the attack has to land on enough rows to be a test: at the ``n=60``
#: default that is ~9 credits, and at ``n=200`` batched (~125 credits) about 19. Partial for the
#: reason every share here is partial -- a run where *every* genuine credit lost its tail would
#: not distinguish a matcher that reads the counterparty from one that had simply stopped
#: ignoring anything.
UTR_PATCHY_SHARE = 0.15

#: Genuine template index -> the ``gateway_plausible`` noise template that masks it.
#:
#: **The mapping exists because deleting the digits is not safe, and that is measured.** A naive
#: strip of the 4-digit tail leaves ``NEFT CR/RAZORPAYSOFT/``, ``NEFT-RZRPAY-`` and
#: ``NEFT/RAZORPAYSOFT/XXXX/SETTLEMENT`` -- and **3 of those 4 shapes are not producible by any
#: noise row**, because ``gateway_plausible``'s templates end in ``XXXX``, ``REF`` or
#: ``SETTLEMENT`` and never in a dangling separator. A masked genuine credit would then be
#: identifiable *by shape alone*, so ``WRONG_IGNORE == 0`` would pass because the attack was
#: visible rather than because decision 4's conjunction holds. That is I12's warning about
#: absence-of-tail becoming "a tell unique to planted rows", arriving on a different flag.
#:
#: So a masked row is **re-rendered** into the noise vocabulary, keeping its own channel and its
#: own spelling. Deterministic in the template index, so masking consumes no extra draw and
#: ``rng.py`` rule 2 is unaffected. The self-check asserts every masked form is byte-producible
#: as ``gateway_plausible`` noise, over every (template, spelling, channel) combination.
UTR_PATCHY_MASK: tuple[int, ...] = (0, 1, 2, 0)


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
    utr_patchy: bool = False               # Phase 8  UTR gone from bank narration

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
    #: Phase 6 step 6 added ``netted_refunds``: ``story._draw_refunds`` draws the refunded
    #: payments and their frozen magnitudes, ``build`` materialises ``refunds.csv`` and nets
    #: each refund off the settlement holding its payment, and ``_batch_net`` counts the term
    #: so the net-uniqueness nudge protects the value the file actually carries. Added *after*
    #: that code existed, per the paragraph above.
    #:
    #: **It nets inside ``net_paise``, so it moves no join** -- the same property ``--tds`` has
    #: and for the same reason: the credit still equals the net, so blocking and the amount band
    #: are untouched. What makes it a different test from ``--tds`` is that the term is **not
    #: derivable from any rate**: it is declared in ``refunds.csv`` and has to be *looked up*
    #: through the payment it cites. ``settlements.csv`` gains no column for it, because I9
    #: freezes that header.
    #: Phase 7 step 1 added ``unsettled``: ``story._draw_unsettled`` picks the payments that no
    #: settlement will claim and ``build`` removes them from ``groups`` before any settlement is
    #: derived, so a group that loses its only member never becomes a settlement at all. Added
    #: *after* that code existed, per the paragraph above. The two Phase 7 flags are declared in
    #: separate steps so a regression in one cannot be attributed to the other.
    #: Phase 7 step 4 added ``noise_rows``: ``story._draw_noise_rows`` draws bank rows that are
    #: not gateway money across three declared strata, and ``build`` merges them into the
    #: credits' sort so both kinds are numbered from one counter. Added *after* that code
    #: existed, per the paragraph above.
    #:
    #: **Declared in step 4 rather than step 8, which is where `.plan/phase7.md` put it.** The
    #: plan's ordering was unsafe for a reason the plan could not see: this refusal is what makes
    #: ``--noise-rows`` unusable from the *command line* while the flag is undeclared, so every
    #: in-process probe has to patch this ClassVar to build a noisy story. Step 7's gate 14
    #: invokes the generator as a **subprocess** and can patch nothing, so the declaration has to
    #: precede it regardless -- and the rule step 1 already recorded is the same one: a flag
    #: enters this set in the step that lands its generator code. Step 4 landed it.
    #: **Phase 8 step 8 added ``fx`` and ``utr_patchy``, and the order was the discipline
    #: above.** Both were read by ``story.py`` and checked by an invariant on both passes
    #: before their names appeared here: ``fx`` in step 2b (the currency column, the signed
    #: rate shift on its own substream, ``Decomposition.fx_paise``, I4 threaded through all
    #: four call sites) and ``utr_patchy`` in step 6 (``_draw_utr_patchy``, the re-render into
    #: the noise vocabulary, I19 in ``invariants.py`` *and* ``verify_output.py``). Until this
    #: edit every probe that needed either flag patched this ClassVar in-process under
    #: ``try/finally``, which is the visible cost of keeping declaration last -- and the right
    #: cost, because the alternative is a run labelled with a mess it does not have.
    #:
    #: ``rounding_edge`` is the only name left out, and it is left out **permanently**: Phase 8
    #: declines it (`.plan/phase8.md` §1(d)). That is what keeps the declared-but-inert refusal
    #: and the seam probe below it honest -- both need a flag that is genuinely inert, and a
    #: seam whose last subject lands is a seam that tests nothing. The self-check does not rely
    #: on that decision holding, though: it manufactures the condition from a *landed* flag as
    #: well, so revisiting the decline cannot quietly empty the check.
    IMPLEMENTED: ClassVar[frozenset[str]] = frozenset(
        {"settlement_delay", "fees", "dup_amounts", "batching", "settlement_report_late",
         "tds", "netted_refunds", "reserve", "unsettled", "noise_rows",
         "fx", "utr_patchy"}
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

    #: The flag pairs ``GenConfig.__post_init__`` refuses, as data.
    #:
    #: Declared because ``--all-mess`` has to *compute* a set the config will accept, and the
    #: only alternative is a hardcoded "drop ``dup_amounts``" that goes stale the moment a
    #: fifth rule lands -- silently, since the switch would still run and still look maximal.
    #:
    #: **This is an index of the four refusal sites below, never a second source of truth for
    #: them.** Each refusal's message and its argument stay at the ``if`` site; this table
    #: carries only the pair. The self-check asserts the two agree in *both* directions, which
    #: is the whole reason the split is safe: a table entry with no matching refusal fails, and
    #: a refusal with no matching entry fails too, because the reduced set stops constructing.
    EXCLUSIVE_PAIRS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("dup_amounts", "batching"),
        ("dup_amounts", "netted_refunds"),
        ("dup_amounts", "reserve"),
        ("dup_amounts", "fx"),
    )

    @classmethod
    def composable(cls) -> tuple[MessFlags, list[tuple[str, str]]]:
        """The largest flag set this config accepts, plus what it dropped and why.

        Backs ``--all-mess``. Two kinds of drop, and the caller prints both: a flag that is
        declared but inert (``story.py`` does not read it, so the run would be labelled with
        a mess it does not have), and a flag that cannot compose with others.

        **Which side of a conflict gets dropped is chosen by degree, not by name.** All four
        pairs today share ``dup_amounts``, so dropping that one flag keeps eleven where
        dropping the other side of each pair would keep eight. Spelling that as "drop
        ``dup_amounts``" would have been shorter and would have been a hardcode; the loop
        picks the flag appearing in the most unresolved pairs, breaking ties alphabetically so
        the switch's meaning is deterministic.

        Greedy max-degree is exact on a star-shaped table like today's, and only a heuristic
        in general -- a future table could make it keep fewer flags than the optimum. It
        cannot make it keep an *invalid* set, which is the property that matters here, and the
        self-check asserts that by constructing the result.
        """
        keep = set(cls.IMPLEMENTED)
        dropped = [
            (n, "declared but not implemented -- story.py does not read it, so the run would "
                "be labelled with a mess it does not have")
            for n in cls.unimplemented()
        ]
        while True:
            live = [(a, b) for a, b in cls.EXCLUSIVE_PAIRS if a in keep and b in keep]
            if not live:
                break
            degree: dict[str, int] = {}
            for pair in live:
                for name in pair:
                    degree[name] = degree.get(name, 0) + 1
            victim = min(degree, key=lambda n: (-degree[n], n))
            keep.discard(victim)
            partners = sorted({a if b == victim else b for a, b in live if victim in (a, b)})
            dropped.append((
                victim,
                "cannot be combined with "
                + ", ".join(f"--{p.replace('_', '-')}" for p in partners),
            ))
        return cls(**{n: True for n in sorted(keep)}), dropped

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
        # ``--netted-refunds`` needs three payments to say what it claims: one carrying an
        # attributable refund, one carrying the planted unattributable one, and one carrying
        # none at all so the term stays *partial*. At n=2 the clamp in ``_draw_refunds`` would
        # refund every payment, and "the refund term" and "every row" would be the same set --
        # at which point nothing downstream can tell a refund-aware matcher from one that
        # absorbed the term into its fee model. Refused here rather than silently dropping the
        # plant, which is the mislabelled-data failure ``MessFlags.IMPLEMENTED`` prevents.
        if self.flags.netted_refunds and self.n < 3:
            raise ValueError(
                f"--netted-refunds needs at least 3 payments -- one refunded, one carrying "
                f"the planted unlinked refund, and one left alone so the term is partial -- "
                f"but --n is {self.n}. Raise --n."
            )
        # Refused for the reason ``--dup-amounts`` and ``--batching`` are refused together,
        # and the mechanism is identical. A planted pair is unresolvable because two
        # settlements share one net, one date and one UTR. Netting a refund off one member
        # moves that member's net and the two diverge -- the pair becomes separable and
        # ``resolvable=false`` turns into a false statement about the data.
        #
        # Deliberately **not** fixed by excluding planted payments from the refund draw: the
        # refunded set would then correlate with plantedness, handing a matcher "the settlements
        # with no refund are the ambiguous ones" as a structural tell. That is the same
        # self-inflicted leak the batching refusal rejects, and a combined run buys nothing --
        # gate 11 covers the planted rows and gate 13 covers the refunds.
        if self.flags.dup_amounts and self.flags.netted_refunds:
            raise ValueError(
                "--dup-amounts and --netted-refunds cannot be combined: netting a refund off "
                "one member of a planted pair moves its net away from its partner's, so the "
                "pair stops sharing an amount and stops being unresolvable -- truth would then "
                "claim resolvable=false about data that is separable. Run them separately; "
                "gate 11 covers the planted rows and gate 13 covers the refunds."
            )
        # ``--reserve`` needs two settlements to say what it claims: one whose payout is held
        # back and one left whole, so the term stays *partial*. At n=1 the clamp in
        # ``_draw_reserves`` would reserve the only settlement, and "the reserved rows" and
        # "every row" would be the same set -- at which point nothing downstream can tell a
        # reserve-aware matcher from one that widened its tolerance until everything fit. I16
        # asserts the partiality; this refuses the ``n`` that cannot deliver it.
        if self.flags.reserve and self.n < 2:
            raise ValueError(
                f"--reserve needs at least 2 payments -- one settlement reserved and one left "
                f"whole so the term is partial -- but --n is {self.n}. Raise --n."
            )
        # Refused for the reason ``--dup-amounts`` is refused with ``--batching`` and with
        # ``--netted-refunds``, and the mechanism is the same one a third time. A planted pair
        # is unresolvable because two settlements share one net, one date and one UTR, so the
        # two *credits* share an amount. Holding a reserve back from one member moves that
        # member's credit away from its partner's, the pair becomes separable on amount alone,
        # and ``resolvable=false`` turns into a false statement about the data.
        #
        # Deliberately **not** fixed by excluding planted settlements from the reserve draw:
        # the reserved set would then correlate with plantedness, handing a matcher "the
        # settlements with no reserve are the ambiguous ones" as a structural tell. Same
        # self-inflicted leak the other two refusals reject, and a combined run buys nothing --
        # gate 11 covers the planted rows and gate 13 covers the reserve.
        if self.flags.dup_amounts and self.flags.reserve:
            raise ValueError(
                "--dup-amounts and --reserve cannot be combined: holding a reserve back from "
                "one member of a planted pair moves its credit away from its partner's, so the "
                "pair stops sharing an amount and stops being unresolvable -- truth would then "
                "claim resolvable=false about data that is separable. Run them separately; "
                "gate 11 covers the planted rows and gate 13 covers the reserve."
            )
        # The **fourth** instance of one hazard, and it was found by measurement rather than
        # predicted (`.plan/probe_phase8_fx_suspensions.py`, Phase 8 step 2). Batching makes a
        # planted member's net a sum, a netted refund moves it, a reserve moves its credit -- and
        # a rate movement moves it too. Every one of them ends with the pair no longer sharing an
        # amount, so ``resolvable=false`` becomes a false statement about separable data.
        #
        # **Refused rather than suspended, for two measured reasons.** I12's own message says the
        # planted count is the denominator of the ``correct_abstention`` rate, so standing it down
        # would silently rescale this project's central claim -- the one thing the whole
        # suspension discipline exists to prevent. And the failure is *probabilistic*: 15 of 30
        # seeds fail at n=200 and 12 of 30 at n=60, with seed 42 passing at n=60 and failing at
        # n=200. A combination that works on half the seeds is worse than one that never works,
        # because a reviewer sees a pass and a judge's seed sees the truth.
        #
        # Note which direction was measured and found empty: across every *passing* seed, no
        # surviving pair held a foreign member. So ``--fx`` never manufactures a spurious
        # collision -- it only destroys planted ones, and the seeds that pass are those where the
        # draw happened to miss both pairs. That is why this is an exclusivity rather than a
        # widened I12.
        #
        # Consequence for `.plan/phase8.md` §1(e): the maximal legal set is unchanged at eleven
        # flags, because ``dup_amounts`` was already outside it. This makes its exclusion rest on
        # four reasons instead of three, which is a strengthening rather than a new restriction.
        if self.flags.dup_amounts and self.flags.fx:
            raise ValueError(
                "--dup-amounts and --fx cannot be combined: a rate movement on one member of a "
                "planted pair moves its net away from its partner's, so the pair stops sharing "
                "an amount and stops being unresolvable -- truth would then claim "
                "resolvable=false about data that is separable. Measured at 15/30 seeds failing "
                "I12 at n=200 (12/30 at n=60), which makes it worse than a clean refusal: it "
                "passes often enough to look fine. Run them separately; gate 11 covers the "
                "planted rows and gate 15 covers the FX rows."
            )
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
    # Phase 6 step 6 landed ``netted_refunds``, so the inert probe moved to a Phase 7 flag; Phase
    # 7 step 4 landed *that* one (``noise_rows``), so it moved to ``fx`` -- and Phase 8 step 8
    # has now landed both ``fx`` and ``utr_patchy``, so it moves a fourth time. Moved rather than
    # deleted, for the reason the seam probe below carries: the moment this names a flag that
    # *is* implemented, it asserts nothing at all.
    #
    # ``rounding_edge`` is where it stops moving, because Phase 8 **declines** that flag
    # (`.plan/phase8.md` §1(d)) rather than deferring it -- so unlike its four predecessors this
    # subject is not scheduled to land. The synthetic guard further down no longer depends on
    # that: it manufactures an inert flag out of a *landed* one, so this check keeps a real
    # subject even if the decline is revisited.
    assert MessFlags(rounding_edge=True).declared_but_inert() == ["rounding_edge"]
    for _landed in ("settlement_delay", "fees", "dup_amounts", "batching",
                    "settlement_report_late", "tds", "netted_refunds", "reserve",
                    "unsettled", "noise_rows", "fx", "utr_patchy"):
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
    # ``netted_refunds`` (Phase 6), then to ``noise_rows`` (Phase 7 step 4), then to ``fx`` --
    # and Phase 8 step 8 landed both ``fx`` and ``utr_patchy``, so it moves a fourth time, to
    # ``rounding_edge``. Never deleted: the seam has to keep working for as long as *any* flag
    # is declared and inert, and the moment it tests a landed flag it tests nothing.
    _original = MessFlags.IMPLEMENTED
    try:
        MessFlags.IMPLEMENTED = _original | {"rounding_edge"}
        assert MessFlags(rounding_edge=True).declared_but_inert() == []
        assert GenConfig(
            flags=MessFlags(rounding_edge=True)
        ).resolved()["flags_enabled"] == ["rounding_edge"]
    finally:
        MessFlags.IMPLEMENTED = _original
    assert MessFlags.IMPLEMENTED == _original, "the probe must not leak"
    assert "rounding_edge" in MessFlags.unimplemented(), (
        "Phase 8 declines --rounding-edge (.plan/phase8.md 1(d)), so it stays declared and "
        "inert -- if it ever lands, the seam probe above needs a new subject and the "
        "synthetic guard below becomes the only thing testing the refusal"
    )
    assert "fx" in MessFlags.IMPLEMENTED, "Phase 8 step 2b implements fx"
    assert "utr_patchy" in MessFlags.IMPLEMENTED, "Phase 8 step 6 implements utr_patchy"
    assert "noise_rows" in MessFlags.IMPLEMENTED, "Phase 7 step 4 implements noise_rows"
    assert "netted_refunds" in MessFlags.IMPLEMENTED, "Phase 6 step 6 implements netted_refunds"
    assert "batching" in MessFlags.IMPLEMENTED, "Phase 5 step 1 implements batching"

    # --- Phase 8 step 8: trap 1's synthetic guard ------------------------------
    #
    # **Everything above this point depends on some flag still being unimplemented, and that
    # is a shrinking resource.** With ``fx`` and ``utr_patchy`` landed, exactly one name is
    # left, and it is left only because Phase 8 *declined* it. Trap 1 in the phase-8 explainer
    # is precisely this: when the last flag lands, ``unimplemented()`` empties, the refusal
    # loop above iterates **zero times**, and the inert assertion has no subject -- the whole
    # mechanism goes green while testing nothing, which is the vacuous-pass class Phase 7's
    # commit named in its own subject line.
    #
    # So the refusal is tested against a flag manufactured inert by *removing a landed one*
    # from the set. That inverts the dependency: instead of needing a flag nobody has built,
    # the guard needs a flag somebody **has** built, and there will only ever be more of those.
    # ``fees`` is the subject because it is the oldest landed flag and the least likely to be
    # renamed, and it is restored in ``finally`` so a failure here cannot leave the seam open
    # for the checks below.
    #
    # This is what makes the ``rounding_edge`` assertions above insurance rather than
    # load-bearing: revisiting the decline would cost the seam its subject and cost nothing
    # else, because the refusal mechanism is verified here without reference to it.
    _synthetic = MessFlags.IMPLEMENTED
    try:
        MessFlags.IMPLEMENTED = frozenset(_synthetic - {"fees"})
        assert MessFlags(fees=True).declared_but_inert() == ["fees"], (
            "a flag absent from IMPLEMENTED must report as inert regardless of whether "
            "story.py can in fact read it -- the set is the declaration, not the code"
        )
        try:
            GenConfig(flags=MessFlags(fees=True))
        except ValueError as _e:
            assert "--fees" in str(_e), f"the refusal must name the switch: {_e}"
        else:
            raise AssertionError(
                "GenConfig accepted a flag missing from IMPLEMENTED, so the declared-but-inert "
                "refusal does not actually gate on that set -- every 'flag is refused' "
                "assertion above would then be passing for an unrelated reason"
            )
    finally:
        MessFlags.IMPLEMENTED = _synthetic
    assert MessFlags.IMPLEMENTED == _synthetic, "the synthetic guard must not leak"
    assert MessFlags(fees=True).declared_but_inert() == [], (
        "fees must be back in IMPLEMENTED after the guard -- a leaked patch here would "
        "refuse every --fees run in this process"
    )

    # Kept, and now with a successor that survives it emptying: the assertions above this line
    # need a genuinely inert flag, the guard below it does not.
    assert MessFlags.unimplemented(), (
        "no flag is unimplemented any more -- the seam probe above is testing nothing, and "
        "the declared-but-inert refusal loop iterates zero times. The synthetic guard above "
        "still covers the refusal itself, so this is a signal to move the seam probe's "
        "subject rather than a broken invariant"
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

    # --- Phase 6 step 7: the reserve ------------------------------------------
    # Shape, not the exact values, for the same reason as BATCH_SIZE_WEIGHTS above: these are
    # tuning choices and pinning them would make this a test to edit. What must hold is that
    # the share is a genuine fraction (partiality is what I16 asserts and what stops "the
    # reserved rows" and "every row" being one set) and that the band is ordered, positive and
    # well under 100% of a net -- a reserve at or above the net would drive a credit to zero
    # or below, which ``Credit.__post_init__`` refuses.
    assert 0 < RESERVE_SHARE < 1, "the reserve share must be a genuine fraction"

    # --- Phase 8 step 2a: the FX share --------------------------------------
    # Imported **locally**, not at module scope. ``config.py`` has no package-level imports at
    # all -- it is the bottom of this package's dependency order, and ``model.py`` reaches *up*
    # to it (for ``IST``, inside its own self-check). A module-level ``from .model import ...``
    # here would invert that for the sake of one assertion and put the two files one edit away
    # from a genuine cycle. The local import keeps the layering and still lets the constants be
    # compared, which is the whole point: ``HOME_CURRENCY`` and ``FX_CURRENCIES`` live in
    # different files precisely because one is a fact about the entities and the other is a
    # tuning choice, and nothing else checks that they disagree.
    from .model import HOME_CURRENCY
    # **This is where partiality is actually guarded, and finding that out took a mutation
    # run.** ``story.py``'s self-check asserts both "the count is the clamp" and "the share is
    # partial", and the second is decoration *there*: both re-derive from the same clamp, so any
    # mutation of ``k`` trips the count assertion first and the partiality assertion can never
    # fire (measured, `.plan/probe_phase8_fx_column_mutants.py`). The property survives only if
    # the constant itself is constrained, which is what this line does -- the same job
    # ``RESERVE_SHARE``'s assertion above has always done for the reserve.
    assert 0 < FX_SHARE < 1, "the FX share must be a genuine fraction"
    # A run where every payment were foreign could not distinguish an FX-aware matcher from one
    # that had widened its tolerance until everything fit, and at n=60 a share above ~0.5 starts
    # making "the foreign rows" and "most rows" the same set. Loose ceiling, not a tuning pin.
    assert FX_SHARE < 0.5, "the foreign share must stay a minority of the file"
    assert FX_CURRENCIES, "--fx needs at least one foreign currency to assign"
    assert HOME_CURRENCY not in FX_CURRENCIES, (
        "the home currency is not a foreign currency: a payment 'moved' to INR would be "
        "labelled as carrying an FX gap while its rate never moved"
    )
    assert len(set(FX_CURRENCIES)) == len(FX_CURRENCIES), FX_CURRENCIES
    # Widening this re-opens I17's coincidence -- with one holder per currency, orphaning that
    # holder fires ``orphan_currencies <= settled_currencies`` legitimately. Not refused, since
    # a later phase may want the breadth; asserted as a **single** element so widening it is a
    # deliberate edit here rather than a silent one, and this comment is what it will read.
    assert len(FX_CURRENCIES) == 1, (
        "FX_CURRENCIES is deliberately one element -- see its docstring on the I17 "
        "coincidence, and re-measure --fx --unsettled before widening it"
    )
    _rlo, _rhi = RESERVE_BPS_BAND
    assert 0 < _rlo <= _rhi < 10_000, RESERVE_BPS_BAND
    # The lower bound is load-bearing rather than cosmetic, and this assertion is what says
    # so. ``matcher.tier1`` tells "money was held back" from "the fee rates assumed here are
    # wrong" by the *size* of the shortfall, so a reserve must be unmistakably larger than any
    # rounding or rate discrepancy this data can produce -- the largest of which is the fee's
    # own rounding divergence, a few paise on a batch. A reserve of a few paise would be
    # indistinguishable from a rounding bug, and truth would be asserting a mess the inputs
    # cannot support. 100 bps is a floor on the floor, well below the declared 500.
    assert _rlo >= 100, (
        "a reserve below 1% of net risks being indistinguishable from fee rounding drift, "
        "which is what tier1's PARTIAL_SETTLEMENT_PENDING diagnostic separates it from"
    )
    _reserved = GenConfig(flags=MessFlags(reserve=True))
    assert not _reserved.clean_mode
    assert _reserved.resolved()["flags_enabled"] == ["reserve"]
    # --reserve moves no date: it changes one amount, the credit's. Same
    # one-variable-at-a-time discipline that put --settlement-delay before --fees.
    assert (_reserved.delay_days, _reserved.lag_days) == (0, 0), "--reserve must move no date"
    assert _reserved.planted_pairs == 0, "--reserve plants no unresolvable pair"
    # n < 2 is refused rather than silently reserving the only settlement, which would make
    # the term total instead of partial.
    try:
        GenConfig(n=1, flags=MessFlags(reserve=True))
    except ValueError as e:
        assert "--reserve" in str(e), f"the refusal must name the flag: {e}"
    else:
        raise AssertionError("GenConfig accepted --reserve at n=1")
    # --dup-amounts + --reserve is refused: holding a reserve back from one member of a
    # planted pair moves its credit away from its partner's, so the pair stops sharing an
    # amount and truth's resolvable=false becomes a false statement about the data.
    try:
        GenConfig(flags=MessFlags(dup_amounts=True, reserve=True))
    except ValueError as e:
        assert "dup-amounts" in str(e) and "reserve" in str(e), (
            f"the refusal must name both flags: {e}"
        )
    else:
        raise AssertionError("GenConfig accepted --dup-amounts with --reserve")
    # --dup-amounts + --fx is refused for the same reason as the three above, found by
    # measurement in Phase 8 step 2 (`.plan/probe_phase8_fx_suspensions.py`): a rate movement on
    # one member moves its net away from its partner's.
    #
    # **Phase 8 step 8 turned this pair of checks inside out, and the reason is worth keeping.**
    # Until ``fx`` landed, ``MessFlags(dup_amounts=True, fx=True)`` was refused for ``fx`` being
    # *inert*, not for the exclusivity -- the unimplemented-flag check in ``__post_init__`` runs
    # before every exclusivity check. So the exclusivity assertion needed the seam patched to be
    # reachable at all, and a control above it proved the ordering was real rather than assumed.
    #
    # Now that ``fx`` is implemented the exclusivity refusal is reachable directly, and it is the
    # *control* that lost its subject. Both halves are kept, with the patch moved from one to the
    # other: the exclusivity is tested plainly, and the ordering is tested by manufacturing
    # inertness -- removing ``fx`` from the set rather than waiting for a flag nobody has built.
    # Same inversion as the synthetic guard further up, and for the same reason: a check whose
    # premise is "some flag is still unimplemented" is a check with an expiry date.
    try:
        GenConfig(flags=MessFlags(dup_amounts=True, fx=True))
    except ValueError as e:
        assert "dup-amounts" in str(e) and "--fx" in str(e), (
            f"the refusal must name both flags: {e}"
        )
    else:
        raise AssertionError("GenConfig accepted --dup-amounts with --fx")
    # Each flag alone stays legal, or the refusal is over-broad -- the failure mode a pairwise
    # check invites is refusing one of its operands outright.
    assert GenConfig(flags=MessFlags(fx=True)).flags.enabled() == ["fx"]
    assert GenConfig(flags=MessFlags(dup_amounts=True)).flags.enabled() == ["dup_amounts"]
    # The ordering control, now synthetic. With ``fx`` withheld from IMPLEMENTED the refusal must
    # cite inertness and must **not** name ``dup-amounts``: that is what proves the exclusivity
    # assertion above is testing the exclusivity rule rather than riding on a refusal that would
    # have happened anyway.
    _before_dup_fx = MessFlags.IMPLEMENTED
    try:
        MessFlags.IMPLEMENTED = frozenset(_before_dup_fx - {"fx"})
        try:
            GenConfig(flags=MessFlags(dup_amounts=True, fx=True))
        except ValueError as e:
            assert "not implemented" in str(e) and "dup-amounts" not in str(e), (
                f"the unimplemented check no longer runs first, so the exclusivity assertion "
                f"above may be passing on the inertness refusal instead: {e}"
            )
        else:
            raise AssertionError("GenConfig accepted a flag missing from IMPLEMENTED")
    finally:
        MessFlags.IMPLEMENTED = _before_dup_fx
    assert MessFlags.IMPLEMENTED == _before_dup_fx, "the dup/fx probe must not leak"
    # And the combination the flag is *designed* to be run in stays legal, or the refusals
    # above have quietly made the deliverable command unrunnable.
    assert GenConfig(
        n=200,
        flags=MessFlags(
            fees=True, settlement_delay=True, batching=True,
            settlement_report_late=True, netted_refunds=True, tds=True, reserve=True,
        ),
    # Compared as a *set*: ``enabled()`` returns field-declaration order, which is incidental
    # to what this asserts (all seven land, and the combination is legal). Pinning the order
    # would make this a test to edit the next time a flag is declared.
    ).flags.enabled() == sorted(
        [
            "settlement_delay", "fees", "batching", "settlement_report_late",
            "netted_refunds", "tds", "reserve",
        ],
        key=MessFlags.names().index,
    )

    # --- ``--all-mess``: EXCLUSIVE_PAIRS must index the refusals above (Phase 8 step 9) -------
    # **Forward -- every table entry names a refusal that exists.** A stale entry would make
    # ``--all-mess`` drop a flag citing a rule that had been lifted. The refusal must name *both*
    # switches, or the entry could be passing on a different rule (inertness, an ``n`` floor)
    # while claiming the exclusivity.
    for _a, _b in MessFlags.EXCLUSIVE_PAIRS:
        try:
            GenConfig(n=200, flags=MessFlags(**{_a: True, _b: True}))
        except ValueError as _e:
            for _switch in (f"--{_a.replace('_', '-')}", f"--{_b.replace('_', '-')}"):
                assert _switch in str(_e), (
                    f"EXCLUSIVE_PAIRS lists ({_a}, {_b}) but the refusal never names {_switch}, "
                    f"so the entry may be riding on a different rule: {_e}"
                )
        else:
            raise AssertionError(
                f"EXCLUSIVE_PAIRS lists ({_a}, {_b}) but GenConfig accepts the combination -- "
                f"the table claims a rule that does not exist, and --all-mess drops a flag for it"
            )
    # **Backward -- no refusal is missing from the table.** Asserted by *construction*, the only
    # form that catches a rule the table has never heard of: a fifth exclusivity landing without
    # an entry leaves its pair in ``composable()``'s result, and this line stops being able to
    # build it. Nothing else in this file would notice, and ``--all-mess`` would exit 2 on a set
    # it had announced one line earlier as the composable one.
    _keep, _dropped = MessFlags.composable()
    GenConfig(n=200, flags=_keep)
    # Every flag is either kept or dropped **with a stated reason**. A flag missing from both
    # sides is a silent drop, which is the mislabelled-run failure this mechanism exists to stop.
    assert {_n for _n, _ in _dropped} == set(MessFlags.names()) - set(_keep.enabled()), (
        f"--all-mess must account for every flag: kept {sorted(_keep.enabled())}, "
        f"dropped {sorted(_n for _n, _ in _dropped)}"
    )
    assert all(_r.strip() for _, _r in _dropped), "a drop with no reason is a silent drop"
    assert set(MessFlags.unimplemented()) <= {_n for _n, _ in _dropped}, (
        "an inert flag survived into --all-mess's set, so the run would be labelled with a mess "
        "story.py does not produce"
    )
    # The control that could embarrass the construction check above. With the table emptied,
    # ``composable()`` must return a set GenConfig *refuses* -- otherwise "it constructs" is
    # equally consistent with there being no exclusivity rules in this file at all.
    _before_pairs = MessFlags.EXCLUSIVE_PAIRS
    try:
        MessFlags.EXCLUSIVE_PAIRS = ()
        _naive, _ = MessFlags.composable()
        try:
            GenConfig(n=200, flags=_naive)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "with EXCLUSIVE_PAIRS emptied the whole implemented set constructed, so the "
                "construction check above proves nothing -- it would pass on an empty table"
            )
    finally:
        MessFlags.EXCLUSIVE_PAIRS = _before_pairs
    assert MessFlags.EXCLUSIVE_PAIRS == _before_pairs, "the EXCLUSIVE_PAIRS control must not leak"

    assert sum(w for *_, w in AMOUNT_BANDS) == 100, "amount band weights must sum to 100"
    assert len(NARRATION_TEMPLATES) == 4

    # --- ``--utr-patchy``'s two lookup tables (Phase 8 step 6) ----------------
    # Both mirror something the templates already state, so what is worth asserting is that
    # they cannot drift from it -- and the subset property below is checked over every
    # combination rather than argued, because **3 of the 4 naive digit-strips fail it**.
    assert len(NARRATION_SPELLINGS) == len(NARRATION_TEMPLATES)
    assert len(UTR_PATCHY_MASK) == len(NARRATION_TEMPLATES)
    assert 0 < UTR_PATCHY_SHARE < 1, (
        "a share that masks every genuine credit would not distinguish a matcher that reads "
        "the counterparty from one that had simply stopped ignoring anything"
    )
    _gp = NOISE_TEMPLATES["gateway_plausible"]
    _spellings = (COUNTERPARTY, COUNTERPARTY_SPACED, COUNTERPARTY_SHORT)
    # Every narration a ``gateway_plausible`` NOISE row can emit, over every spelling and
    # channel. The masked forms must be a subset of this set.
    _noise_shapes = {
        _t.format(channel=_c, counterparty=_s, tail="")
        for _t in _gp
        for _s in _spellings
        for _c in BANK_CHANNELS
    }
    for _i, _template in enumerate(NARRATION_TEMPLATES):
        _spelling = NARRATION_SPELLINGS[_i]
        _rendered = _template.format(
            channel="NEFT",
            counterparty=COUNTERPARTY,
            counterparty_spaced=COUNTERPARTY_SPACED,
            counterparty_short=COUNTERPARTY_SHORT,
            tail=4471,
        )
        assert _spelling in _rendered, (
            f"NARRATION_SPELLINGS[{_i}] says {_spelling!r} but template {_i} renders "
            f"{_rendered!r} -- masking would re-render the row under a spelling it never "
            f"carried, which changes the counterparty a matcher reads"
        )
        # And not a *longer* spelling as well: if the declared one is a substring of the real
        # one, masking would silently shorten the counterparty. ``normalize.py`` documents the
        # same prefix trap from the parsing side.
        assert not [
            _s for _s in _spellings if _s in _rendered and len(_s) > len(_spelling)
        ], f"template {_i} carries a longer spelling than NARRATION_SPELLINGS[{_i}] declares"
        assert 0 <= UTR_PATCHY_MASK[_i] < len(_gp), UTR_PATCHY_MASK[_i]
        for _channel in BANK_CHANNELS:
            _masked = _gp[UTR_PATCHY_MASK[_i]].format(
                channel=_channel, counterparty=_spelling, tail=""
            )
            assert _masked in _noise_shapes, (
                f"masking template {_i} on {_channel} yields {_masked!r}, which no "
                f"gateway_plausible noise row can produce -- a masked genuine credit would be "
                f"identifiable by shape alone, so WRONG_IGNORE would stay at zero because the "
                f"attack is visible rather than because the IGNORED conjunction holds"
            )
            assert not any(_ch.isdigit() for _ch in _masked), (
                f"{_masked!r} carries a digit run, so normalize.parse would report a ref_tail "
                f"and the row would still offer a join -- masking must remove the tail, not "
                f"relocate it"
            )
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
