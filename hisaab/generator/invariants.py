"""Invariants — asserted before anything touches disk, and again on read-back.

Each check here is a bug you would otherwise meet in Phase 5 with no idea where
it came from. ``check_story`` runs on the in-memory story; ``tools/verify_output.py``
re-reads the written CSVs and re-runs the row-level checks against the files,
because the write step itself can corrupt (column order, quoting, encoding) and
because it doubles as the first smoke test of the Phase 2/3 loader.

I7 (no gateway identifier in a bank narration) and I10 (truth isolation) are the
two standing between us and an invalidated submission. They are real checks, not
habits.

A note on I8, because the naive version of it is wrong
-----------------------------------------------------
The temptation is to assert "the payment -> credit index permutation has few fixed
points, because a random permutation of any size has ~1 expected fixed point."
That reasoning holds for payment -> settlement, which is a genuine shuffle. It does
**not** hold for payment -> credit: in clean mode ``value_date`` equals the capture
date by construction, so payments and credits are blocked into identical date
groups and the permutation is block-diagonal. A block of size k contributes ~1
fixed point, so the total is ~(number of blocks), not ~1. Measured: 19-24 at
n=60 across seeds, which the naive ceiling of 5 would fail on every honest run.

Within a date block, payment order is by capture time and credit order is by
amount. Those are independent, so positional coincidence at the ~1/k rate is the
birthday-problem baseline, not a leak -- and it cannot be driven to zero without
*anti*-correlating the files, which would itself be a signal. So I8 splits:

  I8a  the ID numbering carries no information        (a true shuffle, ceiling 5)
  I8b  within-block position carries no information   (a rate well under identity)

I8b's ceiling is set to catch the regressions that matter -- someone sorting the
bank rows by capture time instead of amount, or dropping the re-sort altogether --
both of which drive the rate to 100%.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date

from ..common import ids
from ..common.bizdays import BusinessCalendar
from ..common.reasons import Reason
from .config import GenConfig, MessFlags
from .model import (
    BANK_HEADER,
    PAYMENTS_HEADER,
    REFUNDS_HEADER,
    SETTLEMENT_ITEMS_HEADER,
    SETTLEMENTS_HEADER,
    Decomposition,
    Story,
)
from ..common.money import RUPEE

#: Which mess flags legitimately invalidate each conditional invariant.
#:
#: A check runs unless one of *its own* flags is on, so a flag suspends exactly the
#: checks it actually breaks. This replaces a single ``cfg.clean_mode`` gate, which was
#: a footgun rather than a policy: ``clean_mode`` is "all thirteen flags off", so
#: ``--fees`` -- which changes no cardinality and no date -- switched off the cardinality,
#: membership and uniqueness checks along with everything else. The mislabelled run then
#: lost the very checks that would have noticed it was mislabelled.
#:
#: Keys are ``<invariant id>.<what it asserts>``; a typo raises ``KeyError`` at the call
#: site rather than silently never suspending. Every flag named here is validated against
#: ``MessFlags`` in the self-check, for the same reason.
#:
#: Note what is deliberately **absent**: ``fees`` and ``settlement_delay``. Phase 4 measured
#: whether either can force a natural ``(value_date, amount_paise)`` collision, and the answer
#: decided that both stay off this list. See ``ASSUMPTIONS.md`` #24a/#24b. Measured twice --
#: once in step 6, then again in step 7 after the rate table was corrected, because the first
#: measurement's *stated mechanism* named a rate that no longer exists. Re-running it changed
#: the explanation while leaving the decision intact, which is the useful kind of surprise:
#:
#:   * ``--settlement-delay`` cannot collide **by construction**, and this is the one claim
#:     here that is structural rather than empirical. The capture-to-value date map is
#:     injective, so the delay only *relabels* days: every within-day amount set carries over
#:     intact, and delay-alone reproduces clean mode's collision count exactly at every size
#:     tested. Note that injectivity, not equality, is the property that matters -- an earlier
#:     probe wrongly concluded the delay was unguaranteed from the fact that
#:     ``capture_date != value_date`` on every row, which tests the wrong thing.
#:   * ``--fees`` does both, and the balance shifted when the rates did. It *disperses* when
#:     equal-gross payments sit on different rates (they net apart) and *concentrates* when a
#:     priced row's net lands on some other row's amount. Under the old table, with UPI at 0
#:     bps against card at 200, dispersal dominated: 60-75% of equal-gross groups broke apart.
#:     With four methods now sharing 200 bps that falls to 20-30%, so fees is a weaker
#:     disperser -- and every collision that survives at n>=4000 is one **fees created**,
#:     since clean and delay stay at 0 throughout.
#:
#: The concentration channel has a specific shape worth recording, because it says which rail
#: to watch rather than just that a number exists. ``_unique_amount`` guarantees distinct
#: *gross* within a capture date; a zero-rated row settles **at** its gross, so its credit
#: inherits that protection, while a priced row's net is a derived value no invariant compares
#: to anything. So the free rail acts as a **magnet**: 47-58% of collisions at n>=8000 involve
#: a ``pos_upi`` row, against 11% if method were irrelevant, and the priced member's gross sits
#: 2.42%-2.60% above the collision value -- which is exactly the net-to-gross ratio at 200 and
#: 215 bps, not a fitted range. Phase 5's batching and Phase 6's TDS both add derived nets, so
#: this is the channel that will widen.
#:
#: **336 runs** (12 seeds x 4 flag settings x 7 sizes to the UTR ceiling) put the first
#: collision at n=4000, with 0 at every size at or below 2000. The largest size this project
#: runs is n=1000, so the margin is 4x and clean mode holds everywhere. That margin is why the
#: check stays **strict** rather than being relaxed to D6's "every colliding pair is marked
#: unresolvable": at n<=1000 a collision would be a generator bug or a changed amount
#: distribution, not an honest indistinguishable pair, and a check that silently absorbed it
#: would remove the tripwire exactly where it is load-bearing. The genuinely indistinguishable
#: case has its own home -- ``--dup-amounts`` plants it deliberately and does suspend this
#: check. If it ever does fire unexpectedly, the failure message below carries D6's instruction
#: rather than leaving a reader to invent the wrong fix.
SUSPENDED_BY: dict[str, tuple[str, ...]] = {
    "I3.cardinality":                ("batching", "noise_rows", "unsettled"),
    "I3.no_refunds":                 ("netted_refunds",),
    # **``reserve`` was removed from the next two entries in Phase 6 step 7, and the removal is
    # the correction rather than an omission.** Both listed it as a *prediction*, written phases
    # before any reserve code existed, and building the flag falsified both. Under design B the
    # reserve is held outside ``net_paise``: the settlement declares its full net and the credit
    # arrives short, so every settlement still **has** its credit (merely a short one) and no
    # settlement, payment or bank row becomes an orphan. Both checks pass on a reserved run
    # untouched, and ``invariants.py``'s self-check now asserts ``checks_skipped == {}`` there --
    # which is strictly stronger than the suspensions were, because the checks actually run.
    #
    # Left in place they would have been the footgun this docstring warns about: two checks
    # standing down on every reserved run, announced but unnecessary, and the one that catches a
    # generator dropping a credit outright is exactly the check ``--reserve`` most needs kept.
    # (I4's precondition comment carried the same wrong prediction and is corrected in place.)
    # What ``--reserve`` genuinely does break is Strategy D in ``tools/verify_output.py``, which
    # was *strengthened* into an equality on identities rather than added to that file's
    # suspension list. Three checks touched, none of them the two predicted here.
    "I3.no_orphans":                 ("unsettled", "noise_rows"),
    "I2.every_payment_settled":      ("unsettled",),
    "I2.every_settlement_credited":  ("unsettled",),
    "I6.all_payments_cited":         ("unsettled", "noise_rows"),
    "I3.unique_date_amount":         ("dup_amounts",),
}


def _suspended(cfg: GenConfig, check: str, record: dict[str, list[str]]) -> bool:
    """True if ``check`` must be skipped because a flag it cannot survive is on.

    Records *why* into ``record``, so the run can announce the skip instead of quietly
    performing fewer checks than it appears to. Silence is what made the old gate
    dangerous: the checks vanished and the output looked identical.
    """
    on = [f for f in SUSPENDED_BY[check] if getattr(cfg.flags, f)]
    if on:
        record[check] = on
    return bool(on)


#: I8a: expected fixed points of a true random permutation is 1, independent of n.
#: Five is a generous ceiling that still fails loudly on identity ordering.
MAX_NUMBERING_FIXED_POINTS = 5

#: I8b: identity ordering scores 100%. Observed honest rates are 29-41% at n=60,
#: 8-13% at n=200, 3-5% at n=500 -- the rate falls as blocks grow. A 60% ceiling
#: leaves large headroom above honest noise while catching any real regression.
MAX_WITHIN_BLOCK_ALIGNMENT_RATE = 0.60

#: Below this many block-resident records the rate is statistically meaningless
#: (at n=12 most dates hold a single record, and 2/2 = 100% is pure chance).
MIN_BLOCK_POPULATION_FOR_RATE = 20

#: I14: what fraction of multi-member batches may be a *consecutive* run of payment ids.
#:
#: Same shape as I8b above, and for the same reason -- a rate against an identity baseline,
#: with a population floor below which the rate means nothing. Payments are capture-sorted and
#: numbered in that order, so one settlement date's payments are a contiguous id run. Slicing
#: that run in place would make every batch a set of consecutive ids, and once
#: ``--settlement-report-late`` withholds membership a searcher could enumerate contiguous runs
#: (O(k^2)) instead of subsets (O(2^k)) and resolve the file without ever performing a subset
#: search. ``story._group_into_batches`` shuffles each date's pool before slicing to prevent
#: exactly that, and this is the check that says the shuffle is still there.
#:
#: **Measured, seeds 1-5 plus 42, multi-member batches only** (a singleton is trivially a run
#: of one and carries no information, so it is excluded from the denominator):
#:
#:   n=60    10-15 batches   42.9%-80.0%   <- below the floor, deliberately not asserted
#:   n=200   45-51 batches    8.9%-19.6%
#:   n=1000 234-255 batches    1.6%- 4.5%
#:
#: Dropping the shuffle scores **100%**. The rate falls as dates grow more populated, which is
#: the birthday-problem baseline: a 2-batch drawn from a date holding three payments is
#: consecutive by chance a third of the time, and driving that to zero would require
#: *anti*-correlating membership against id order, which is itself a signal.
#:
#: So n=60 is left unasserted rather than given a loose ceiling. A 60% ceiling would pass
#: seed 2's honest 80% and would also pass a genuine regression at the same size -- a check
#: that cannot distinguish those is decoration, and the ceiling that would fit n=60 is above
#: the identity baseline this check exists to catch. n=200 is the default under ``--batching``
#: (decision 3), so the check is live on the configuration a judge runs.
MAX_CONSECUTIVE_ID_BATCH_RATE = 0.60

#: Below this many multi-member batches the rate above is not asserted. Set to clear n=60's
#: 10-15 and admit n=200's 45-51.
MIN_MULTI_BATCH_POPULATION_FOR_RATE = 40


class InvariantError(AssertionError):
    """A generated story or written file violated a structural guarantee."""


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise InvariantError(f"{code}: {message}")


# ---------------------------------------------------------------------------
# Row-level checks. Callable on the in-memory story AND on rows read back from
# disk, so the two passes cannot drift apart.
# ---------------------------------------------------------------------------

def check_unique_ids(code: str, label: str, id_values: list[str]) -> None:
    """I1 — every ID is unique within its file."""
    seen: set[str] = set()
    dupes: set[str] = set()
    for v in id_values:
        if v in seen:
            dupes.add(v)
        else:
            seen.add(v)
    _require(not dupes, code, f"duplicate {label} ids: {sorted(dupes)[:5]}")


def check_headers(actual: dict[str, tuple[str, ...]]) -> None:
    """I9 — frozen headers, in order.

    Guards trap 8: every field added to the bank statement is difficulty deleted
    from our own submission, so the header is pinned rather than merely reviewed.
    """
    expected = {
        "payments.csv": PAYMENTS_HEADER,
        "settlements.csv": SETTLEMENTS_HEADER,
        "settlement_items.csv": SETTLEMENT_ITEMS_HEADER,
        "bank_statement.csv": BANK_HEADER,
        "refunds.csv": REFUNDS_HEADER,
    }
    for name, want in expected.items():
        _require(name in actual, "I9", f"{name} is missing")
        _require(
            tuple(actual[name]) == want,
            "I9",
            f"{name} header drifted\n  expected: {want}\n  actual:   {tuple(actual[name])}",
        )


def check_no_leak(bank_rows: list[tuple[str, int, str]]) -> None:
    """I7 — no gateway identifier and no self-amount in any bank narration.

    ``bank_rows`` is (row_id, amount_paise, narration).

    The literal reading of "no narration contains pay_, setl_ or C" both fires on
    ``IMPS-JOHNDOE-DIRECTTRANSFER`` (a 'C') and misses what matters, so this
    matches ID *patterns* on word boundaries (trap 6). The amount check compares
    whole digit runs, not substrings, so a UTR tail of 1004 is not read as
    containing the rupee figure 100.
    """
    for row_id, amount_paise, narration in bank_rows:
        leaked = ids.leaked_identifiers(narration)
        _require(
            not leaked,
            "I7",
            f"{row_id} narration leaks {leaked}: {narration!r} -- "
            f"a gateway identifier in the bank statement generates the answer "
            f"into the input and invalidates the submission",
        )
        runs = set(ids.digit_runs(narration))
        whole, frac = divmod(amount_paise, RUPEE)
        echoes = runs & {str(amount_paise), str(whole)} if frac == 0 else runs & {str(amount_paise)}
        _require(
            not echoes,
            "I7",
            f"{row_id} narration echoes its own amount {echoes}: {narration!r}",
        )


def check_totals(
    gross_total: int,
    net_total: int,
    credit_total: int,
    fee_cells: list[int],
    tds_cells: list[int],
    refund_cells: list[int],
    reserve_cells: list[int],
    *,
    fees_on: bool = False,
    tds_on: bool = False,
    refunds_on: bool = False,
    reserve_on: bool = False,
) -> None:
    """I4 — the money adds up in aggregate, and deductions exist only when asked for.

    Six assertions, and only the last four are conditional:

      * ``net_total - reserve == credit_total`` -- every settled paisa reached the bank
        except what was deliberately held back.
      * ``gross_total - (every fee/gst/tds/refund cell) == net_total`` -- the wedge between
        gross and net is *exactly* the declared deductions and nothing else.
      * with ``--fees`` off, every fee and GST cell is zero.
      * with ``--tds`` off, every TDS cell is zero.
      * with ``--netted-refunds`` off, every refund cell is zero.
      * with ``--reserve`` off, every reserve cell is zero.

    Note which assertion ``reserve_cells`` appears in and which it does not: it is subtracted
    in the *first* and absent from the *second*. See the ``--reserve`` paragraph at the end of
    this docstring -- that asymmetry is design B stated as arithmetic, and swapping it would
    make the reserve invisible.

    **``refund_cells`` is Phase 6 step 6, and extending this function is the treatment the
    plan pre-committed to rather than the one that was tempting.** A refund inside
    ``net_paise`` makes the four-term wedge false, and the cheap fix -- listing I4 in
    ``SUSPENDED_BY`` under ``netted_refunds`` -- is exactly the footgun the ``SUSPENDED_BY``
    docstring describes: the check would stand down at the moment the arithmetic it guards
    started moving. Measured before this parameter existed: at seed 42, n=60 the flag failed
    I4 with "+716363 paise unaccounted for", which *is* the run's refund total. So the
    invariant caught the new term on the first run, which is the behaviour that makes
    strengthening the right move -- a suspension would have written the file silently.

    **The last two were one assertion until Phase 6, and splitting them fixed a false
    refusal and a hole at the same time.** The old form gated all three columns on
    ``fees_on``, which was right while ``--fees`` was the only flag that could put a number
    in any of them. Phase 6 adds a second, and one flag gating three columns then fails in
    both directions -- measured, not reasoned:

      * ``--tds`` **without** ``--fees`` was **refused**: the TDS cells are non-zero,
        ``fees_on`` is false, and the check reported "1 non-zero fee/gst/tds cells while
        --fees is off". A legal run, rejected.
      * ``--fees`` **with** a stray TDS cell **passed silently**: with ``fees_on`` true the
        assertion stood down for all three columns at once, so a TDS number appearing on a
        run that never asked for one was invisible. That is the more dangerous half, and it
        is the mess-flag footgun this file refuses everywhere else -- a check standing down
        at the moment the arithmetic it guards starts moving.

    This docstring previously argued the opposite, and the argument is worth keeping rather
    than quietly deleting: *"Splitting them per column would be no stronger here and would
    only give the two callers a chance to disagree about the order of three ints."* True
    while one flag owned all three columns; false the moment a second flag arrives. The
    caller-disagreement risk it worried about also goes **down**, not up, because each list
    is now homogeneous -- there is no order of three ints left to disagree about. A stated
    reason that expires is why the reason gets stated.

    ``tds_cells`` is required rather than defaulted, deliberately. A default would let a
    caller omit TDS from the wedge and still pass the *first* two assertions on clean data,
    which is a silent under-count of the very quantity this function exists to bound.

    The second one is Phase 4's strengthening. Until now this asserted
    ``gross == net == credited``, which ``--fees`` necessarily violates, so the tempting
    move was to suspend it under the flag. That is precisely the footgun the
    ``SUSPENDED_BY`` docstring describes: the check would stand down at the moment the
    arithmetic it guards started moving. Subtracting the cells instead makes it
    *stronger*, and at zero deductions it is character-for-character the old assertion --
    so clean mode is still checked exactly the way Phase 1 checked it.

    Clean mode has *zero* fees, not "fees applied consistently" (trap 7): that is what
    makes the first amount mismatch in Phase 3 a one-suspect debug rather than a
    two-suspect one, and it is why ``fees_on`` gates the third assertion instead of
    ``cfg.clean_mode``. Any flag at all makes ``clean_mode`` false; only ``--fees`` is
    allowed to put a number in these columns.

    ``fee_cells`` is every fee and GST cell and ``tds_cells`` every TDS cell, so the two
    sums together are the whole wedge.

    Precondition: every payment sits in exactly one settlement, so summing gross over
    payments and net over settlements counts the same money. ``--unsettled`` voids that and
    belongs in ``SUSPENDED_BY`` when it lands (Phase 7).

    **This docstring previously named ``--reserve`` alongside it, and building the flag showed
    that prediction was wrong -- worth keeping rather than quietly deleting, since it is the
    second wrong guess in this cluster.** A reserve does *not* void the precondition and does
    not belong in ``SUSPENDED_BY``, because it is held back **outside** ``net_paise``: the
    settlement still declares its full net, every payment still sits in exactly one
    settlement, and the gross/net wedge below is untouched. What it breaks is only the *first*
    assertion, ``net_total == credit_total`` -- money genuinely was settled and did not reach
    the bank, which is the entire mess. So the fix is ``reserve_cells``, subtracted from the
    net in that one assertion and **deliberately absent from ``deducted``** in the next.
    That asymmetry between the two assertions *is* design B, expressed as arithmetic: the
    reserve sits between net and credit, never between gross and net.
    """
    deducted = sum(fee_cells) + sum(tds_cells) + sum(refund_cells)
    held = sum(reserve_cells)
    _require(
        net_total - held == credit_total,
        "I4",
        f"settled and credited disagree: net={net_total} - reserve={held} = "
        f"{net_total - held} credited={credit_total} "
        f"({net_total - held - credit_total:+d} paise never reached the bank)",
    )
    _require(
        gross_total - deducted == net_total,
        "I4",
        f"the gross/net wedge is not the declared deductions: gross={gross_total} "
        f"- deductions={deducted} = {gross_total - deducted}, but net={net_total} "
        f"({gross_total - deducted - net_total:+d} paise unaccounted for)",
    )
    for label, cells, on in (
        ("fee/gst", fee_cells, fees_on),
        ("tds", tds_cells, tds_on),
        ("refund", refund_cells, refunds_on),
        ("reserve", reserve_cells, reserve_on),
    ):
        if on:
            continue
        nonzero = [c for c in cells if c != 0]
        _require(
            not nonzero,
            "I4",
            f"{len(nonzero)} non-zero {label} cells while the flag that fills them is "
            f"off: {nonzero[:5]}",
        )


def check_settlement_arithmetic(
    rows: list[tuple[str, int, int, int, int, int]] | list[tuple[str, int, int, int, int, int, int]],
) -> None:
    """I4 — per settlement: ``net == gross - fee - gst - tds - refunds``, GST sits on the fee.

    ``rows`` is ``(settlement_id, gross_of_members, net, fee, gst, tds)`` with an optional
    seventh element for the refunds netted off that settlement. **Optional rather than
    required**, which is the opposite of the choice ``check_totals`` makes one function up, and
    the asymmetry is deliberate: that function takes whole columns and a caller omitting one
    would under-count the wedge silently, while here a missing seventh element is a row that
    says "no refunds" -- and at zero refunds the assertion is character-for-character the
    five-term one Phase 6 step 2 left behind. Every pre-Phase-6 fixture in this file therefore
    keeps testing exactly what it tested before.

    The per-row companion to ``check_totals``, and it earns its keep for a reason the
    aggregate cannot cover: totals that agree in sum can still be wrong row by row, and
    two compensating errors in opposite directions cancel exactly. Under ``--fees`` every
    settlement carries its own rounding, so this is where a half-up slip in one direction
    would show.

    Both callers previously had this as an inline "net equals the member gross **while
    ``--fees`` is off**" check. Same treatment as I11 and ``check_totals``: strengthened
    to the full subtraction rather than suspended, which at zero deductions is the old
    equality unchanged.

    The second assertion is not arithmetic bookkeeping but a check on the *composition*:
    GST is a share of the fee, never of the gross. Since the GST rate is well under 100%,
    ``gst <= fee`` holds for any fee -- and it fails immediately if the two rates are ever
    applied to the same base, which inflates GST by roughly fifty times and is the single
    easiest error available in this model (see ``story._deductions``). Deliberately
    rate-free: re-deriving the fee from ``cfg.fees`` here would only assert that the
    generator agrees with itself. The independent re-derivation is the *matcher's* job,
    and the residual closing to zero is what proves the two sides agree.
    """
    for row in rows:
        sid, gross, net, fee, gst, tds = row[:6]
        refunds = row[6] if len(row) > 6 else 0
        _require(
            net == gross - fee - gst - tds - refunds,
            "I4",
            f"{sid}: net {net} != gross {gross} - fee {fee} - gst {gst} - tds {tds} "
            f"- refunds {refunds} = {gross - fee - gst - tds - refunds}",
        )
        _require(
            gst <= fee,
            "I4",
            f"{sid}: gst {gst} exceeds fee {fee} -- GST is charged on the fee, not on "
            f"the gross, so this is the two rates applied to the same base",
        )


def check_refunds(story: Story, cfg: GenConfig) -> None:
    """I15 — every refund is netted exactly once, against the settlement truth says.

    **The successor to ``I3.no_refunds``**, which asserted only that no refund existed. That
    check is suspended under ``--netted-refunds`` (``SUSPENDED_BY``), and a suspension with no
    successor is how a flag switches off the checks that would have caught its own bugs. Four
    assertions, and the third is the one the plan singled out:

      * every refund cites a payment in ``payments.csv``, **or** truth records it as
        out-of-scope -- the planted unlinked refund is the second case, and it is legitimate
        precisely because truth says so rather than because the check is lenient.
      * every refund in the file is netted off exactly one credit, and every id a credit cites
        exists. An emitted refund that reduces nobody's net is money that vanished from the
        books; a cited id that is not in the file is an answer key describing data it does not
        have.
      * **term by term, never in total.** Each credit's ``refunds_paise`` must equal the sum of
        the refunds it actually cites. A run-wide total would pass while two refunds were
        attributed to each other's settlements -- the aggregate is identical and every
        settlement's arithmetic is wrong, which is exactly the failure ``.plan/phase6.md`` step
        6 names. (Same reasoning as ASSUMPTIONS.md #25 grading a decomposition per term, and as
        ``adjustments.py`` comparing fee and GST separately.)
      * no refund is netted twice. Double-counting is the one error that *strengthens* the
        arithmetic's appearance -- I4's wedge still closes if a refund is subtracted twice and
        another term absorbs it -- so it needs naming on its own.

    ``cfg`` is read only to decide whether the flag is on, so a clean-mode story reaches this
    function and leaves it having asserted the honest thing: there are no refunds and nothing
    claims any.
    """
    refunds_by_id = {r.refund_id: r for r in story.refunds}
    payment_ids = {p.payment_id for p in story.payments}

    if not cfg.flags.netted_refunds:
        # Not merely skipped. With the flag off the file has no refunds and no credit may claim
        # one, which is the same statement I3.no_refunds makes -- asserted here too so that
        # this function is a total check rather than one that only runs on the messy path.
        _require(not story.refunds, "I15", "refunds emitted without --netted-refunds")
        claimed = [c.credit_id for c in story.credits if c.refunds_netted]
        _require(
            not claimed,
            "I15",
            f"credits claim netted refunds without --netted-refunds: {claimed[:5]}",
        )
        return

    _require(bool(story.refunds), "I15", "--netted-refunds emitted no refunds at all")

    # Assertion 1: every refund is either linked or declared out-of-scope by truth.
    #
    # "Truth records it as out-of-scope" is read off the credit that nets it: the note names
    # the orphan, and the payment it cites is absent by construction. What is checked here is
    # the *pairing* -- an unlinked refund must be netted off a credit whose truth entry says so
    # -- because an orphan refund with no such note would be an answer key that cannot explain
    # its own data.
    netted_at: dict[str, str] = {}
    for c in story.credits:
        for rid in c.refunds_netted:
            _require(
                rid in refunds_by_id,
                "I15",
                f"{c.credit_id} cites refund {rid}, which is not in refunds.csv",
            )
            # Assertion 4, and it is checked here rather than by counting at the end so the
            # failure names both claimants instead of only a total.
            _require(
                rid not in netted_at,
                "I15",
                f"{rid} is netted off both {netted_at.get(rid)} and {c.credit_id} -- a refund "
                f"subtracted twice can still leave I4's wedge closed, so it is named here",
            )
            netted_at[rid] = c.credit_id

    # Assertion 2: nothing in the file is netted off nobody.
    orphaned = sorted(set(refunds_by_id) - set(netted_at))
    _require(
        not orphaned,
        "I15",
        f"{len(orphaned)} refund(s) in refunds.csv reduce no credit's net: {orphaned[:5]}. "
        f"An emitted refund that nets against nothing is money missing from the books.",
    )

    by_credit_id = {c.credit_id: c for c in story.credits}
    for r in story.refunds:
        if r.payment_id in payment_ids:
            continue
        # Assertion 1's second branch. Every refund is netted somewhere (assertion 2, above),
        # so this lookup cannot miss -- and the note must **name the payment it cites**, not
        # merely be non-empty. A note that exists but describes something else would satisfy a
        # bare "is not None" while leaving the orphan unexplained, and the whole reason an
        # out-of-scope refund is admissible is that truth accounts for it.
        holder = by_credit_id[netted_at[r.refund_id]]
        _require(
            holder.note is not None and r.payment_id in holder.note,
            "I15",
            f"{r.refund_id} cites {r.payment_id}, which is not in payments.csv, and "
            f"{holder.credit_id}'s truth note does not name it -- an orphan refund the answer "
            f"key cannot explain is a generator bug, not a mess",
        )
        # And it stays **resolvable**, which is the Phase 4b standard rather than a preference:
        # the refund's amount is declared in ``refunds.csv``, so the residual identifies it and
        # an unbounded matcher could attribute it. Truth may not claim otherwise. Pinned here
        # because it is the assertion that would fail if a later phase "fixed" these rows by
        # marking them unresolvable -- which would move a real miss into the correct-abstention
        # cell and inflate the headline.
        _require(
            holder.resolvable,
            "I15",
            f"{holder.credit_id} nets an out-of-scope refund and is marked unresolvable -- "
            f"the refund's amount is declared, so the row is resolvable in principle and an "
            f"exhaustive matcher would refute the claim (Phase 4b's standard)",
        )

    # Assertion 3: term by term.
    for c in story.credits:
        cited = sum(refunds_by_id[rid].amount_paise for rid in c.refunds_netted)
        _require(
            c.decomposition.refunds_paise == cited,
            "I15",
            f"{c.credit_id} declares refunds_paise={c.decomposition.refunds_paise} but the "
            f"{len(c.refunds_netted)} refund(s) it cites sum to {cited} -- compared per credit "
            f"rather than in total, because a run-wide total is identical when two refunds are "
            f"attributed to each other's settlements",
        )


def check_reserves(story: Story, cfg: GenConfig) -> None:
    """I16 — the held-back reserve, the one deduction no input file declares.

    **A strengthening of ``I2.every_settlement_credited``, not a successor to it, and that
    correction is the point of reading a check before standing it down.**
    ``.plan/phase6.md`` step 7 called for a successor because ``SUSPENDED_BY`` listed
    ``I2.every_settlement_credited`` under ``reserve`` -- an entry written in an earlier phase
    as a *prediction*, before any reserve code existed. The prediction is wrong. That check
    asserts every settlement is **cited by some credit** (``invariants.py``'s I2 block), and
    under design B a reserved settlement still has its credit; the credit is merely *short*.
    So it passes untouched, and ``I3.no_orphans`` -- which gates on
    ``settlements_without_credit`` -- passes for the same reason.

    **Both entries are therefore deleted rather than left in place, and the self-check asserts
    ``checks_skipped == {}`` on a reserved run.** That assertion is what caught the first draft
    of this docstring, which claimed the two checks "run for real" while the entries were still
    listed -- they stood down and were recorded as skipped. Deleting them is strictly stronger
    than keeping them: a suspension that is unnecessary is two checks standing down on every
    reserved run, and the one that catches a generator dropping a credit outright is precisely
    the check ``--reserve`` most needs kept.

    The plan's **third case** goes with it: "a truth record saying the settlement is uncredited
    in this window" exists only to justify that suspension, and honouring it would have handed
    ``--reserve`` a settlement with no credit at all -- which is ``--unsettled``'s mess in
    Phase 7, arriving early and under the wrong flag's name. Two named cases, below.

    Six assertions:

      * **Case 1, unreserved:** the credit equals its settlement's net exactly.
      * **Case 2, reserved:** the credit is short of its settlement's net by *exactly* the
        amount truth records as held. Not "approximately", and not "by some amount" -- the
        held figure is the only record of this money anywhere, so if it disagreed with the
        shortfall then nothing in the universe would say what was withheld.
      * ``Credit.reserve_held_paise`` and ``Decomposition.reserve_paise`` agree. Two fields
        carry this one fact and a reader may consult either.
      * **The reserved credit's amount equals no settlement's net at all.** This is the
        wrong-match guard, re-derived here rather than trusted from
        ``story._separate_reserved_amounts``, and it is the assertion this invariant most
        earns its keep for. A reserved credit is the only row whose true settlement is absent
        from an exact-band candidate pool, so an amount clash gives the matcher **one**
        confident wrong answer instead of two candidates and an honest abstention -- a
        ``WRONG_MATCH`` on the line this project says never bends. Checked against every net
        in the run rather than same-date nets only, because the date window is a matcher-side
        parameter this generator does not know.
      * **Partial:** at least one settlement reserved and at least one not, whenever the flag
        is on. Otherwise "the reserved rows" and "every row" are the same set, and no
        downstream comparison could tell a reserve-aware matcher from one that widened its
        tolerance until everything fit.
      * **Reserved credits stay ``resolvable: true``.** The mirror of I15's pin, and the
        assertion that would fail if a later phase "fixed" these rows by marking them
        unresolvable. It would look like a fix and it would be a false statement about the
        data: a reserve leaves the settlement's UTR alone, so the narration tail still
        identifies it uniquely (measured: a tail-only join resolves 100% of gateway credits at
        n=200 and n=1000 under every implemented flag combination). Marking them unresolvable
        would inflate ``correct_abstention`` with separable rows and make ``LUCKY_GUESS`` --
        the leak detector -- fire on a matcher that got the answer right.
    """
    held_by_settlement = {
        sid: c.decomposition.reserve_paise
        for c in story.credits
        for sid in c.settlement_ids
    }
    if not cfg.flags.reserve:
        # The zero case is asserted by ``check_totals``' cell gate; this is the per-credit
        # companion, which catches a reserve attributed to a *credit* the aggregate would
        # net out against a missing one elsewhere.
        for c in story.credits:
            _require(
                c.decomposition.reserve_paise == 0 and c.reserve_held_paise == 0,
                "I16",
                f"{c.credit_id} carries a reserve "
                f"({c.decomposition.reserve_paise}p/{c.reserve_held_paise}p) while "
                f"--reserve is off",
            )
        return

    net_of = {s.settlement_id: s.net_paise for s in story.settlements}
    all_nets = {s.net_paise for s in story.settlements}
    reserved = 0

    for c in story.credits:
        held = c.decomposition.reserve_paise
        _require(
            c.reserve_held_paise == held,
            "I16",
            f"{c.credit_id} records reserve_held_paise={c.reserve_held_paise} but its "
            f"decomposition says {held} -- two fields carry this fact and a reader may "
            f"consult either, so they may not disagree",
        )
        # A credit cites one settlement in every phase up to here; summing keeps this honest
        # if a later phase makes a credit span several.
        net_total = sum(net_of[sid] for sid in c.settlement_ids)
        if held == 0:
            _require(
                c.amount_paise == net_total,
                "I16",
                f"{c.credit_id} holds no reserve but its {c.amount_paise}p does not equal "
                f"the {net_total}p net of {', '.join(c.settlement_ids)} -- case 1 of I16",
            )
            continue

        reserved += 1
        _require(
            c.amount_paise == net_total - held,
            "I16",
            f"{c.credit_id} is short of its {net_total}p net by "
            f"{net_total - c.amount_paise}p, but truth records {held}p as held -- case 2 of "
            f"I16. The held figure is the only record of this money in existence; if it "
            f"disagrees with the shortfall then nothing anywhere says what was withheld",
        )
        _require(
            0 < held < net_total,
            "I16",
            f"{c.credit_id} holds {held}p of a {net_total}p net -- a reserve of nothing is a "
            f"row truth calls reserved whose credit equals its net, and one at or above the "
            f"net makes the credit zero or negative",
        )
        _require(
            c.amount_paise not in all_nets,
            "I16",
            f"{c.credit_id}'s short amount {c.amount_paise}p equals some settlement's net. "
            f"Its own settlement is invisible to an exact-amount lookup, so the matcher finds "
            f"exactly one candidate -- the WRONG one -- with arithmetic that closes perfectly, "
            f"and resolves it confidently. That is a silent wrong match, not an abstention. "
            f"story._separate_reserved_amounts exists to nudge the held amount clear of every "
            f"net in the run; this is the independent re-derivation of that work",
        )
        _require(
            c.resolvable,
            "I16",
            f"{c.credit_id} is reserved and marked unresolvable -- the reserve leaves this "
            f"settlement's UTR untouched, so the narration tail still identifies it uniquely "
            f"and an unbounded matcher recovers the payment set (measured: a tail-only join "
            f"resolves 100% of gateway credits). resolvable=false would be a false statement "
            f"about the data (Phase 4b's standard), would inflate correct_abstention with "
            f"separable rows, and would make LUCKY_GUESS fire on a correct answer",
        )

    _require(
        reserved > 0,
        "I16",
        "--reserve is on but no credit is short -- the run is labelled with a mess it does "
        "not have",
    )
    _require(
        reserved < len(story.credits),
        "I16",
        f"every one of {len(story.credits)} credits is reserved, so the term is not partial "
        f"-- 'the reserved rows' and 'every row' become the same set, and nothing downstream "
        f"can tell a reserve-aware matcher from one that widened its tolerance until "
        f"everything fit",
    )


def check_int_money(values: dict[str, object]) -> None:
    """I5 — every monetary value is an ``int`` (trap 9).

    Re-run after the CSV round-trip, where a float would arrive as a string like
    ``'1000000.0'`` and fail ``int()`` loudly rather than silently.
    """
    for label, v in values.items():
        _require(
            isinstance(v, int) and not isinstance(v, bool),
            "I5",
            f"{label} is {type(v).__name__} ({v!r}), not int paise",
        )


def check_within_block_alignment(
    payment_ids_by_date: dict[date, list[str]],
    credit_payment_ids_by_date: dict[date, list[str]],
) -> tuple[int, int]:
    """I8b — within a date block, row position carries no information.

    Returns (aligned, population) for reporting. See the module docstring for why
    this is a rate check against identity rather than a fixed-point ceiling.

    **Both dicts must be keyed by the same date and hold the same payments per key.**
    The caller keys both on the credit's ``value_date``, which is not a detail: keying the
    payment side on ``business_date`` instead was correct only while the two dates were
    equal. Under Phase 4's posting lag the keys offset by n business days, so each block
    compared a credit list against the payments of a *different* day -- the rate collapsed
    from 19/58 to 0/58 and the check could no longer fail at all. An invariant that goes
    quietly inert the moment the thing it guards starts moving is worse than one that was
    never written, because the report still prints a number for it.
    """
    aligned = population = 0
    for d, pl in payment_ids_by_date.items():
        cl = credit_payment_ids_by_date.get(d, [])
        if len(pl) < 2:
            continue  # a single-record date is trivially "aligned" and means nothing
        population += len(pl)
        aligned += sum(1 for i, pid in enumerate(pl) if i < len(cl) and cl[i] == pid)
    if population >= MIN_BLOCK_POPULATION_FOR_RATE:
        rate = aligned / population
        _require(
            rate <= MAX_WITHIN_BLOCK_ALIGNMENT_RATE,
            "I8b",
            f"within-date-block positional alignment is {rate:.0%} "
            f"({aligned}/{population}), above the {MAX_WITHIN_BLOCK_ALIGNMENT_RATE:.0%} "
            f"ceiling -- bank rows appear to be ordered by capture time rather than "
            f"amount, which lets a zip matcher score without doing any work",
        )
    return aligned, population


def check_batch_adjacency(multi_batches: list[list[int]]) -> tuple[int, int]:
    """I14 — batch membership carries no payment-id adjacency information.

    ``multi_batches`` is one sorted list of payment *numbers* per settlement holding more than
    one payment. Singletons are excluded by the caller: a batch of one is trivially a run of
    one and says nothing. Returns ``(consecutive, population)`` for reporting.

    Factored out like ``check_within_block_alignment`` and for the same two reasons: the
    on-disk pass in ``tools/verify_output.py`` re-runs it against ``settlement_items.csv``, and
    a check that can only be reached by constructing a whole broken story is a check nobody
    can prove fires. See ``MAX_CONSECUTIVE_ID_BATCH_RATE`` for the measurement.
    """
    if len(multi_batches) < MIN_MULTI_BATCH_POPULATION_FOR_RATE:
        return (
            sum(1 for b in multi_batches if b == list(range(b[0], b[0] + len(b)))),
            len(multi_batches),
        )
    consecutive = sum(1 for b in multi_batches if b == list(range(b[0], b[0] + len(b))))
    rate = consecutive / len(multi_batches)
    _require(
        rate <= MAX_CONSECUTIVE_ID_BATCH_RATE,
        "I14",
        f"{rate:.0%} of multi-member batches ({consecutive}/{len(multi_batches)}) are a "
        f"consecutive run of payment ids, above the {MAX_CONSECUTIVE_ID_BATCH_RATE:.0%} "
        f"ceiling -- batches appear to be sliced out of capture order without shuffling the "
        f"date's pool first. Payments are numbered in capture order, so consecutive-id "
        f"batches let a searcher enumerate contiguous runs (O(k^2)) instead of subsets "
        f"(O(2^k)) and resolve withheld membership without ever performing a subset search. "
        f"See story._group_into_batches.",
    )
    return consecutive, len(multi_batches)


# ---------------------------------------------------------------------------
# Story-level checks
# ---------------------------------------------------------------------------

def check_story(
    story: Story, cfg: GenConfig, calendar: BusinessCalendar | None = None
) -> dict[str, object]:
    """Run every invariant on the in-memory story. Raises ``InvariantError``.

    Returns a small report dict for the CLI to echo, so a passing run still shows
    the numbers behind the pass rather than only the word "ok".

    ``calendar`` is injectable for the same reason ``story.build`` takes one: I11 recomputes
    the two delays and must do it over the *same* calendar the generator used. Pass the
    same object to both, or a holiday set on one side and not the other turns a correct
    story into an invariant failure -- and a shared default would hide that a caller
    forgot to pass it.
    """
    cal = calendar or BusinessCalendar()
    payments, settlements, credits = story.payments, story.settlements, story.credits

    # I1 — unique ids within each file
    check_unique_ids("I1", "payment", [p.payment_id for p in payments])
    check_unique_ids("I1", "order", [p.order_id for p in payments])
    check_unique_ids("I1", "settlement", [s.settlement_id for s in settlements])
    check_unique_ids("I1", "credit", [c.credit_id for c in credits])
    check_unique_ids("I1", "refund", [r.refund_id for r in story.refunds])

    #: Conditional checks that did not run, and the flag that excused each. Echoed by
    #: the CLI and carried into the report: a skipped check must be *visible*, because
    #: an invisible skip is exactly what made the old ``clean_mode`` gate dangerous.
    skipped: dict[str, list[str]] = {}

    # I3 — cardinality, refunds and orphans. Three checks, three different flag sets:
    # --fees breaks none of them, which is the whole point of splitting the old gate.
    if not _suspended(cfg, "I3.cardinality", skipped):
        _require(
            len(payments) == len(settlements) == len(credits) == cfg.n,
            "I3",
            f"expected 1:1:1 cardinality at n={cfg.n}, got "
            f"{len(payments)}/{len(settlements)}/{len(credits)}",
        )
    if not _suspended(cfg, "I3.no_refunds", skipped):
        _require(not story.refunds, "I3", "refunds emitted without --netted-refunds")
    if not _suspended(cfg, "I3.no_orphans", skipped):
        _require(
            not (story.unsettled_payment_ids or story.settlements_without_credit
                 or story.non_gateway_credit_ids),
            "I3",
            "orphans or noise rows present while every flag that creates them is off",
        )

    # I2 — every payment in exactly one settlement; every settlement in one credit
    settled_count: dict[str, int] = defaultdict(int)
    for s in settlements:
        for pid in s.payment_ids:
            settled_count[pid] += 1
    payment_ids = {p.payment_id for p in payments}
    unknown = sorted(set(settled_count) - payment_ids)
    _require(not unknown, "I2", f"settlements reference unknown payments: {unknown[:5]}")
    if not _suspended(cfg, "I2.every_payment_settled", skipped):
        missing = sorted(payment_ids - set(settled_count))
        _require(not missing, "I2", f"payments never settled: {missing[:5]}")
    multi = sorted(pid for pid, k in settled_count.items() if k > 1)
    _require(not multi, "I2", f"payments in more than one settlement: {multi[:5]}")

    credited_count: dict[str, int] = defaultdict(int)
    for c in credits:
        for sid in c.settlement_ids:
            credited_count[sid] += 1
    settlement_ids = {s.settlement_id for s in settlements}
    unknown_s = sorted(set(credited_count) - settlement_ids)
    _require(not unknown_s, "I2", f"credits reference unknown settlements: {unknown_s[:5]}")
    if not _suspended(cfg, "I2.every_settlement_credited", skipped):
        uncredited = sorted(settlement_ids - set(credited_count))
        _require(not uncredited, "I2", f"settlements never credited: {uncredited[:5]}")
    multi_s = sorted(sid for sid, k in credited_count.items() if k > 1)
    _require(not multi_s, "I2", f"settlements in more than one credit: {multi_s[:5]}")

    # I4 — the money adds up, in aggregate and then per settlement
    gross_of = {p.payment_id: p.gross_paise for p in payments}
    # The refund term comes from **truth's decomposition**, not from a settlement column:
    # ``settlements.csv``'s header is frozen by I9, so a netted refund is visible only through
    # ``refunds.csv``. Summing the credits' ``refunds_paise`` is therefore the only aggregate
    # available -- and it is the right one, because it is the number truth publishes and
    # therefore the number a scorer will hold the matcher to.
    refund_cells = [c.decomposition.refunds_paise for c in credits]
    # The reserve term, from the same source and for the same reason as the refunds: no
    # settlement column declares it (I9 freezes that header, and unlike a refund it has no
    # ``refunds.csv`` either -- it is declared *nowhere* in the inputs), so truth's
    # decomposition is the only place it exists. I16 independently checks these terms against
    # the per-credit shortfall, so this sum cannot quietly agree with a wrong attribution.
    reserve_cells = [c.decomposition.reserve_paise for c in credits]
    check_totals(
        story.total_gross_paise(),
        story.total_net_paise(),
        story.total_credited_paise(),
        [x for s in settlements for x in (s.fee_paise, s.gst_paise)],
        [s.tds_paise for s in settlements],
        refund_cells,
        reserve_cells,
        fees_on=cfg.flags.fees,
        tds_on=cfg.flags.tds,
        refunds_on=cfg.flags.netted_refunds,
        reserve_on=cfg.flags.reserve,
    )
    # I15 — the refund term is not merely *a* number that balances: every refund in the file
    # is accounted for exactly once, and each settlement's term equals the refunds truly
    # netted off it. The successor to ``I3.no_refunds``, which only asserted refunds were
    # absent; see ``check_refunds``.
    check_refunds(story, cfg)
    # I16 — the reserve term, which is the only deduction in this model that **no input file
    # declares**. See ``check_reserves``: it is a strengthening of
    # ``I2.every_settlement_credited`` rather than a successor to it, because that check turns
    # out to survive ``--reserve`` untouched.
    check_reserves(story, cfg)
    # Per settlement, the refund term truth attributes to it. Sourced from the credits rather
    # than from a settlement column because ``settlements.csv``'s header is frozen (I9): a
    # netted refund is declared in ``refunds.csv`` and attributed by truth, which is exactly
    # the position the matcher is in. I15 independently checks that these terms equal the
    # refunds each credit cites, so this sum cannot quietly agree with a wrong attribution.
    refunds_of_settlement: dict[str, int] = defaultdict(int)
    for c in credits:
        for sid in c.settlement_ids:
            refunds_of_settlement[sid] += c.decomposition.refunds_paise
    check_settlement_arithmetic(
        [
            (
                s.settlement_id,
                sum(gross_of[pid] for pid in s.payment_ids),
                s.net_paise,
                s.fee_paise,
                s.gst_paise,
                s.tds_paise,
                refunds_of_settlement[s.settlement_id],
            )
            for s in settlements
        ]
    )

    # I5 — every monetary value is an int
    money: dict[str, object] = {}
    for p in payments:
        money[f"{p.payment_id}.gross_paise"] = p.gross_paise
    for s in settlements:
        money[f"{s.settlement_id}.net_paise"] = s.net_paise
    for c in credits:
        money[f"{c.credit_id}.amount_paise"] = c.amount_paise
        money[f"{c.credit_id}.expected"] = c.decomposition.expected_credit_paise
    check_int_money(money)

    # Clean-mode arithmetic: each credit's decomposition must close to its amount.
    for c in credits:
        _require(
            c.decomposition.expected_credit_paise == c.amount_paise,
            "I4",
            f"{c.credit_id} decomposition expects "
            f"{c.decomposition.expected_credit_paise} but the credit is {c.amount_paise}",
        )

    # I7 — the leak check
    check_no_leak([(c.credit_id, c.amount_paise, c.narration) for c in credits])

    # I6 — truth references resolve in both directions
    for c in credits:
        for sid in c.settlement_ids:
            _require(sid in settlement_ids, "I6", f"{c.credit_id} cites unknown {sid}")
        for pid in c.payment_ids:
            _require(pid in payment_ids, "I6", f"{c.credit_id} cites unknown {pid}")
    if not _suspended(cfg, "I6.all_payments_cited", skipped):
        cited = {pid for c in credits for pid in c.payment_ids}
        _require(
            cited == payment_ids,
            "I6",
            f"{len(payment_ids - cited)} payments are in no credit's truth entry",
        )

    # I11 — both dates are exactly where the declared delays put them.
    #
    # These two checks used to assert plain equality "while --settlement-delay is off".
    # Phase 4 makes them *stronger* rather than suspending them, for the reason the
    # SUSPENDED_BY docstring gives: a check that stands down exactly when the thing it
    # guards starts moving is the footgun, not the policy. At delay 0 and lag 0 the
    # assertion below is character-for-character the old one; under the flag it is a real
    # test of the delay model, and it is the only thing that would catch a silent
    # off-by-one in either direction.
    #
    # Note which of the two the matcher can see. The capture->settlement delay is
    # invisible to it (the join never reads captured_at); the settlement->credit lag is
    # the one the date window is measured against. Getting the first wrong is a
    # generator-only bug that nothing downstream would ever reveal, which is precisely
    # why it is asserted here.
    by_pay = {p.payment_id: p for p in payments}
    by_setl = {s.settlement_id: s for s in settlements}
    for s in settlements:
        for pid in s.payment_ids:
            want = cal.add_business_days(by_pay[pid].business_date, cfg.delay_days)
            _require(
                s.settled_on == want,
                "I11",
                f"{s.settlement_id} settled {s.settled_on} but {pid} captured "
                f"{by_pay[pid].business_date}, which is {want} at a "
                f"{cfg.delay_days}-business-day settlement delay",
            )
    for c in credits:
        for sid in c.settlement_ids:
            want = cal.add_business_days(by_setl[sid].settled_on, cfg.lag_days)
            _require(
                c.value_date == want,
                "I11",
                f"{c.credit_id} dated {c.value_date} but {sid} settled "
                f"{by_setl[sid].settled_on}, which is {want} at a "
                f"{cfg.lag_days}-business-day bank posting lag",
            )

    # The data must be resolvable from (date, amount) -- row 1 of the mess dial. A
    # duplicate pair is genuinely indistinguishable from the two legitimate signals, so
    # the honest verdict would be an abstention. That case is real and --dup-amounts
    # plants it deliberately in Phase 4b; it must not appear here by accident.
    #
    # Suspended only by --dup-amounts. Step 6 measured whether --fees or
    # --settlement-delay can force a collision, because the plan predicted both would and
    # a suspension was on the table if they did. Neither does: the delay's date map is
    # injective so it only relabels days, and fees *disperse* amounts (the per-method
    # rates push equal-gross payments onto different nets) rather than compressing them.
    # Zero collisions in 48 runs, first appearing at n=4000 -- see SUSPENDED_BY's docstring
    # and ASSUMPTIONS.md #24a/#24b for the numbers.
    #
    # That margin is why this check stays strict. At the sizes this project runs, a
    # collision is a generator bug or a changed amount distribution, not an honest
    # indistinguishable pair -- so the failure message below sorts the two cases by size
    # rather than leaving a reader to guess which one they have.
    if not _suspended(cfg, "I3.unique_date_amount", skipped):
        key_counts = Counter((c.value_date, c.amount_paise) for c in credits)
        dupes = {k for k, count in key_counts.items() if count > 1}
        # Named so the two halves of the guidance below cannot drift apart from the
        # measurement that justifies them.
        measured_clean_to = 2000
        _require(
            not dupes,
            "I3",
            f"{len(dupes)} duplicate (date, amount) credits are unresolvable from the "
            f"inputs: {sorted(dupes)[:3]}\n"
            f"  Do NOT resample it away: that would delete the most honest row in the "
            f"file to protect an invariant.\n"
            f"  This run is n={cfg.n}. Measured (ASSUMPTIONS.md #24a): the pair is "
            f"collision-free to n={measured_clean_to} on seeds 1/2/3/42 under any "
            f"combination of --fees and --settlement-delay, with the first natural "
            f"collision at n=4000.\n"
            + (
                f"  At n={cfg.n} a collision is therefore NOT expected pressure -- suspect "
                f"a generator change (the amount distribution, the fee rates, or "
                f"_unique_amount's key) before concluding the data is honestly ambiguous."
                if cfg.n <= measured_clean_to
                else f"  At n={cfg.n} this is past the measured range and may be a genuine "
                f"indistinguishable pair. Then decision D6 applies: mark it "
                f"resolvable=false in truth (ASSUMPTIONS.md #24) so the matcher is scored "
                f"on abstaining rather than on guessing."
            ),
        )

    # I12 — the successor to the check --dup-amounts suspends.
    #
    # ``I3.unique_date_amount`` stands down under this flag because the collision is the
    # *intent*. A suspension that is merely announced still leaves the run less checked than
    # it looks: with I3 off, a planting bug that collided three rows, or zero, or that
    # collided the amounts while leaving the UTRs distinct, would pass everything. So the
    # flag brings its own invariant, and it is stricter than the one it replaces -- it
    # asserts the collision is exactly what was ordered.
    #
    # The UTR clause is the load-bearing one, and it is why this check exists rather than a
    # comment. Measured before the flag was built, re-run by gate 11 (tools/acceptance.py): a
    # tail-only strategy reading no date and no amount resolves 60/60, 200/200 and 1000/1000
    # credits *correctly* on every dev seed, because tails are drawn without replacement. A
    # pair colliding on (date, amount) with distinct tails is therefore still separable by
    # exhaustive narration matching -- ``resolvable=False`` would be a false statement about
    # the data, and the flag would not test the capability its name claims. That is finrecon's
    # recorded failure: its showcase tier fell to narration enumeration, 200/200.
    if cfg.flags.dup_amounts:
        key_counts = Counter((c.value_date, c.amount_paise) for c in credits)
        groups = sorted(k for k, count in key_counts.items() if count > 1)
        _require(
            len(groups) == cfg.dup_pairs,
            "I12",
            f"--dup-amounts asked for {cfg.dup_pairs} planted pair(s) but the data holds "
            f"{len(groups)} colliding (date, amount) group(s). The planted count is the "
            f"denominator of the correct_abstention rate, so a wrong count silently "
            f"rescales this project's central claim.",
        )
        utr_of = {s.settlement_id: s.utr for s in settlements}
        unresolvable = [c for c in credits if not c.resolvable]
        for key in groups:
            members = [c for c in credits if (c.value_date, c.amount_paise) == key]
            _require(
                len(members) == 2,
                "I12",
                f"planted group {key} has {len(members)} members, not 2. Three credits "
                f"sharing a (date, amount) is a different and harder case than the pair "
                f"this flag documents, and it would be scored as though it were the pair.",
            )
            utrs = {utr_of[sid] for c in members for sid in c.settlement_ids}
            _require(
                len(utrs) == 1,
                "I12",
                f"planted group {key} spans {len(utrs)} distinct UTRs ({sorted(utrs)}). "
                f"The UTR tail reaches the bank narration and resolves 100% of rows on its "
                f"own, so a pair with distinct tails is still separable by exhaustive "
                f"narration matching -- it is NOT unresolvable, and marking it "
                f"resolvable=false would be a false statement about the data.\n"
                f"  Do not fix this by removing the UTR from the narration: that is "
                f"--utr-patchy's job (Phase 8) and it would make absence-of-tail a tell "
                f"unique to planted rows.",
            )
            for c in members:
                _require(
                    not c.resolvable
                    and c.reason == str(Reason.AMBIGUOUS_DUPLICATE_AMOUNT),
                    "I12",
                    f"{c.credit_id} shares a (date, amount, utr) with another credit but "
                    f"truth marks it resolvable={c.resolvable} reason={c.reason!r}. An "
                    f"indistinguishable row recorded as resolvable would score an honest "
                    f"abstention as a MISS.",
                )
        _require(
            len(unresolvable) == 2 * cfg.dup_pairs,
            "I12",
            f"{len(unresolvable)} credit(s) are marked unresolvable but "
            f"{2 * cfg.dup_pairs} were planted. A row marked unresolvable outside a planted "
            f"group inflates the correct_abstention denominator with a row the matcher "
            f"could in fact have resolved.",
        )

    # I13 — the successor to the cardinality coverage --batching takes away.
    #
    # ``I3.cardinality`` asserts ``payments == settlements == credits == cfg.n`` and
    # ``batching`` genuinely breaks it: n payments become ~n/1.6 settlements and that many
    # credits. It is listed in ``SUSPENDED_BY`` and the run announces the skip -- but this is
    # the first suspension to remove *load-bearing* coverage, so the announcement is a bill
    # rather than a permission.
    #
    # **What the bill actually comes to was measured, not assumed, and it is smaller here than
    # .plan/phase5.md predicted.** Trap 8 says "with it off, a grouping loop that drops a
    # payment passes everything." In this module it does not: probed by deliberate violation,
    # dropping one member from a batch raises I2 ``payments never settled``, a credit dropped
    # raises I2 ``settlements never credited``, and a credit citing two settlements raises I2
    # ``settlements in more than one credit``. Those clauses are unconditional and dominate
    # every partition error reachable through a whole story -- ``Settlement.__post_init__``
    # forbids a repeated member and ``Credit.__post_init__`` forbids citing none, so by
    # pigeonhole a member-count or settlement/credit mismatch always surfaces as an uncited or
    # twice-cited id first. Writing those as I13 clauses too would have shipped three checks
    # that can never fire, which this module's own self-check calls decoration.
    #
    # What is genuinely lost is exactly one thing: the comparison against ``cfg.n``. Nothing
    # else in this module relates the record count to the config, so ``--batching --n 200``
    # emitting 199 payments would pass everything. Unconditional, under every flag --
    # ``--unsettled`` changes which payments get *settled*, never how many were captured.
    #
    # The rest of the bill is real but falls due in a different file. ``tools/verify_output.py``
    # gates its "payments never settled" check on ``truth.clean_mode``, which is false under
    # *any* flag -- so the on-disk pass, the one that sees only what a judge sees, does lose
    # the partition check under ``--batching``. That is trap 8's worry, correctly located, and
    # it is fixed there rather than here.
    _require(
        len(payments) == cfg.n,
        "I13",
        f"--n asked for {cfg.n} payments but the story holds {len(payments)}. Batching "
        f"changes how payments are grouped, never how many exist -- so this is a draw-loop "
        f"or grouping bug, and every per-record rate computed from it would be scaled wrong.",
    )

    # Payment index by id, used by I14 below and I8a after it. Position in ``payments`` is
    # capture order, which is also id order -- that identity is what I14 is about.
    pay_number = {p.payment_id: i for i, p in enumerate(payments)}

    # I14 — batch membership carries no id-adjacency information.
    #
    # The successor to nothing: this guards a property ``--batching`` *creates*. See
    # ``MAX_CONSECUTIVE_ID_BATCH_RATE`` for the measurement and for why n=60 is left
    # unasserted rather than given a ceiling loose enough to pass an honest 80%.
    #
    # Runs unconditionally, not only under the flag. Without ``--batching`` every settlement
    # holds one payment, the population is 0 and the check is satisfied -- which is the
    # correct answer, and it means a future flag that starts grouping payments without
    # announcing it does not escape this.
    multi_batches = [
        sorted(pay_number[pid] for pid in s.payment_ids)
        for s in settlements
        if len(s.payment_ids) > 1
    ]
    consecutive, batch_population = check_batch_adjacency(multi_batches)

    # I8a — the ID numbering carries no information.
    #
    # ``payment_ids[0]`` is the lowest-numbered member, which is the whole member list when
    # nothing is batched. Under ``--batching`` there are fewer settlements than payments so the
    # two index spaces only overlap on the low range, but the property under test is unchanged:
    # ``rng_settlements.shuffle(groups)`` is a genuine shuffle either way, so a fixed point is
    # still a coincidence and identity ordering still scores ~n.
    setl_index = {s.settlement_id: i for i, s in enumerate(settlements)}
    numbering_fixed = sum(
        1 for s in settlements if pay_number[s.payment_ids[0]] == setl_index[s.settlement_id]
    )
    _require(
        numbering_fixed <= MAX_NUMBERING_FIXED_POINTS,
        "I8a",
        f"{numbering_fixed} settlements share an index with their payment "
        f"(ceiling {MAX_NUMBERING_FIXED_POINTS}) -- settlement IDs appear to be "
        f"assigned in payment order, which makes the numbering itself the answer key",
    )

    # I8b — within-block position carries no information.
    #
    # Both sides are keyed on the **credit's** value_date, so a block holds one day of
    # bank rows and exactly the payments those rows paid out. Keying the payment side on
    # ``business_date`` was equivalent only while the two dates agreed; under a posting
    # lag it compared each credit block against a different day's payments and the check
    # went inert (see check_within_block_alignment's docstring).
    #
    # Payments keep capture order within a block because ``payments`` is capture-sorted
    # and this loop appends in that order -- which is the ordering the check is *about*:
    # bank rows are sorted by amount, and if position still tracked capture time a zip
    # matcher would score without doing any work.
    value_date_of: dict[str, date] = {
        pid: c.value_date for c in credits for pid in c.payment_ids
    }
    p_by_date: dict[date, list[str]] = defaultdict(list)
    for p in payments:
        # A payment in no credit is Phase 7's --unsettled; it belongs to no bank block.
        if (when := value_date_of.get(p.payment_id)) is not None:
            p_by_date[when].append(p.payment_id)
    c_by_date: dict[date, list[str]] = defaultdict(list)
    for c in credits:
        c_by_date[c.value_date].append(c.payment_ids[0])
    aligned, population = check_within_block_alignment(p_by_date, c_by_date)

    return {
        "records": len(payments),
        "date_blocks": len(p_by_date),
        "numbering_fixed_points": numbering_fixed,
        "within_block_aligned": f"{aligned}/{population}",
        # Reported even when the population is below the floor that makes the rate
        # assertable: the numbers travel into run_manifest.json, and "0/0" is how a reader
        # sees that a run did no batching at all.
        "consecutive_id_batches": f"{consecutive}/{batch_population}",
        "gross_paise_total": story.total_gross_paise(),
        # Which conditional checks did not run, and the flag that excused each. Empty in
        # clean mode. This travels into run_manifest.json beside "status": "pass", which
        # is the honest pairing: everything that ran, passed -- and here is what did not
        # run. A pass count that quietly shrinks with each new flag is how a generator
        # ends up certifying data nothing checked.
        "checks_skipped": {k: list(v) for k, v in sorted(skipped.items())},
    }


if __name__ == "__main__":
    import dataclasses

    from .story import build

    cfg = GenConfig(seed=42, n=60)
    story = build(cfg)
    report = check_story(story, cfg)
    assert report["records"] == 60
    print(f"invariants: {report}")

    # Every seed and size in the acceptance matrix must pass.
    for n in (1, 12, 60, 200):
        for seed in (1, 2, 3, 42, 43):
            c = GenConfig(seed=seed, n=n)
            check_story(build(c), c)
    print("invariants.py: 20 story configurations pass")

    # --- the checks must FAIL on broken stories, or they are decoration --------
    fired: list[str] = []

    def must_fail(code: str, broken: Story, cfg_: GenConfig = cfg) -> None:
        try:
            check_story(broken, cfg_)
        except InvariantError as e:
            assert str(e).startswith(code), f"expected {code}, got: {e}"
            fired.append(code)
        else:
            raise AssertionError(f"{code} did not fire")

    # I7: a leaked settlement id in a narration
    bad = dataclasses.replace(
        story.credits[0], narration=f"NEFT-{story.credits[0].settlement_ids[0]}-XXXX4471"
    )
    must_fail("I7", dataclasses.replace(story, credits=[bad, *story.credits[1:]]))

    # I7: a narration echoing its own amount
    c0 = story.credits[0]
    bad = dataclasses.replace(c0, narration=f"NEFT-RAZORPAYSOFT-{c0.amount_paise}")
    must_fail("I7", dataclasses.replace(story, credits=[bad, *story.credits[1:]]))

    # I4: a non-zero fee cell while --fees is off
    s0 = story.settlements[0]
    must_fail(
        "I4",
        dataclasses.replace(
            story, settlements=[dataclasses.replace(s0, fee_paise=1), *story.settlements[1:]]
        ),
    )

    # --- step 4: the fee arithmetic ------------------------------------------
    # ``fees`` is in MessFlags.IMPLEMENTED as of step 4, so this config needs no patching
    # seam -- the flag now genuinely changes the data.
    fees_cfg = GenConfig(seed=42, n=60, flags=MessFlags(fees=True))
    fees_story = build(fees_cfg)
    assert check_story(fees_story, fees_cfg)["checks_skipped"] == {}, (
        "--fees must suspend nothing at all"
    )

    # I4, the per-settlement check, on the case the aggregate provably cannot see: move a
    # paisa of fee from one settlement to another. Both totals still agree to the paisa --
    # the deduction sum is unchanged and no net or credit moved -- so every aggregate
    # assertion passes and only the per-row subtraction fails. Two compensating errors
    # cancelling is not a hypothetical; it is what a sign slip in a batching loop looks
    # like, and it is the whole reason check_settlement_arithmetic exists.
    # Chosen by carrying a fee, not by index: ``pos_upi`` is zero-rated, so a settlement in
    # shuffled order may have nothing to move. That was near-certain when the zero-rated
    # share was ~36% of rows and is merely possible now that it is ~6% -- which is exactly
    # why the selection stays written this way. A probe that passes because the sample got
    # lucky is a probe that fails on a seed nobody ran. And the
    # replacement is positional -- reordering the list would change what I8a measures, so
    # the probe would risk failing for the wrong reason.
    a_s, b_s = [s for s in fees_story.settlements if s.fee_paise][:2]
    nudged = {a_s.settlement_id: 1, b_s.settlement_id: -1}
    shifted = dataclasses.replace(
        fees_story,
        settlements=[
            dataclasses.replace(s, fee_paise=s.fee_paise + nudged[s.settlement_id])
            if s.settlement_id in nudged
            else s
            for s in fees_story.settlements
        ],
    )
    _cells = [x for s in shifted.settlements for x in (s.fee_paise, s.gst_paise, s.tds_paise)]
    assert sum(_cells) == sum(
        x for s in fees_story.settlements for x in (s.fee_paise, s.gst_paise, s.tds_paise)
    ), "the probe must leave the aggregate untouched, or it tests the wrong assertion"
    must_fail("I4", shifted, fees_cfg)

    # The two arithmetic checks, probed directly. Some failures cannot be reached through
    # a whole story without tripping an earlier assertion first -- a GST error large
    # enough to exceed its fee also breaks the subtraction and the totals -- so the unit
    # probe is the only way to know the specific assertion fires rather than a neighbour.
    def must_raise(code: str, what: str, fn) -> None:
        """Probe one check function directly, asserting *which* invariant fires.

        ``code`` is a parameter rather than a hard-coded ``"I4"`` because Phase 5's I14 uses
        this helper too, and a probe that accepts any InvariantError cannot tell the check it
        is testing from a neighbour firing first -- which is the whole reason these unit
        probes exist alongside the whole-story ones.
        """
        try:
            fn()
        except InvariantError as e:
            assert str(e).startswith(code), f"{what}: expected {code}, got: {e}"
            fired.append(code)
        else:
            raise AssertionError(f"{what} did not fire")

    # GST charged on the gross instead of on the fee: 18% of a ₹10,000 gross is nine times
    # a 2% fee, so it lands above the fee it is supposed to sit on.
    must_raise(
        "I4",
        "gst on the gross",
        lambda: check_settlement_arithmetic(
            [("setl_x", 1_000_000, 1_000_000 - 20_000 - 180_000, 20_000, 180_000, 0)]
        ),
    )
    must_raise(
        "I4",
        "net off by a paisa",
        lambda: check_settlement_arithmetic(
            [("setl_x", 1_000_000, 976_401, 20_000, 3_600, 0)]
        ),
    )
    # ... and the correct row passes, or the probe above proves nothing.
    check_settlement_arithmetic([("setl_x", 1_000_000, 976_400, 20_000, 3_600, 0)])
    # Zero deductions: the assertion is the pre-Phase-4 equality, unchanged.
    check_settlement_arithmetic([("setl_x", 1_000_000, 1_000_000, 0, 0, 0)])

    must_raise(
        "I4",
        "wedge is not the deductions",
        lambda: check_totals(
            1_000_000, 976_400, 976_400, [20_000, 3_500], [0], [0], [0], fees_on=True
        ),
    )
    must_raise(
        "I4",
        "settled but never credited",
        lambda: check_totals(
            1_000_000, 976_400, 976_399, [20_000, 3_600], [0], [0], [0], fees_on=True
        ),
    )
    check_totals(1_000_000, 976_400, 976_400, [20_000, 3_600], [0], [0], [0], fees_on=True)
    check_totals(1_000_000, 1_000_000, 1_000_000, [0, 0], [0], [0], [0])

    # Phase 6: the per-column zero gate, probed in **both** directions. Until this phase all
    # three columns were gated on ``fees_on``, and both of these were measured against the old
    # code before it was changed -- the first was refused and the second passed silently.
    #
    # A legal ``--tds`` run without ``--fees``: TDS cells are non-zero, fee and GST are not.
    # The old single gate reported "1 non-zero fee/gst/tds cells while --fees is off" and
    # refused a run this generator is required to support.
    check_totals(10_000, 9_990, 9_990, [0, 0], [10], [0], [0], fees_on=False, tds_on=True)
    # ...and the hole, which is the half that mattered. A TDS cell on a run that never asked
    # for one: with ``fees_on`` true the old gate stood down for all three columns at once, so
    # this passed. A deduction appearing from nowhere is exactly what I4's zero clause exists
    # to catch, and for one phase it could only catch two thirds of it.
    must_raise(
        "I4",
        "a TDS cell while --tds is off",
        lambda: check_totals(
            10_000, 9_990, 9_990, [0, 0], [10], [0], [0], fees_on=True, tds_on=False
        ),
    )
    # The converse, for symmetry: a fee cell while --fees is off, with --tds legitimately on.
    # Without this the pair above would still pass if the two labels were wired backwards.
    must_raise(
        "I4",
        "a fee cell while --fees is off",
        lambda: check_totals(
            10_000, 9_800, 9_800, [200, 0], [0], [0], [0], fees_on=False, tds_on=True
        ),
    )

    # --- Phase 6 step 6: the refund column, probed in both directions ---------
    # Same shape as the TDS pair above, because the same reasoning applies to a third column:
    # a legal run must not be refused, and a term appearing on a run that never asked for one
    # must not pass. Written as a pair rather than one assertion for the reason the TDS pair
    # records -- a single gate covering N columns fails in both directions at once.
    #
    # A legal ``--netted-refunds`` run: a 1,000p refund inside the net, no fee and no tax.
    check_totals(10_000, 9_000, 9_000, [0, 0], [0], [1_000], [0], refunds_on=True)
    # And the hole: the same refund on a run with the flag off. Before ``refund_cells`` existed
    # this was not merely unchecked, it was *unrepresentable* -- the wedge simply did not
    # include the term, so a refund had nowhere to be declared and I4 reported it as
    # unaccounted-for money (measured: "+716363 paise unaccounted for" at seed 42, n=60).
    must_raise(
        "I4",
        "a refund cell while --netted-refunds is off",
        lambda: check_totals(
            10_000, 9_000, 9_000, [0, 0], [0], [1_000], [0], refunds_on=False
        ),
    )
    # The full six-term wedge, every flag on: fee 200p, GST 36p, TDS 10p, refund 1,000p.
    # Present so that the terms are proved to compose rather than merely to exist one at a
    # time -- four columns that each pass alone can still be summed wrongly.
    check_totals(
        10_000, 8_754, 8_754, [200, 36], [10], [1_000], [0],
        fees_on=True, tds_on=True, refunds_on=True,
    )

    # --- Phase 6 step 7: the reserve column, and the asymmetry that defines it ------
    # A legal ``--reserve`` run with no other deduction at all: the whole 10,000p is settled,
    # 1,000p is held back, and 9,000p reaches the bank. Note what this fixture asserts that
    # none of the four above can: ``net_total != credit_total`` and the run is still valid.
    # Every earlier fixture has those two equal, so this is the first case where the first
    # assertion does real work.
    check_totals(10_000, 10_000, 9_000, [0, 0], [0], [0], [1_000], reserve_on=True)
    # **The asymmetry probe, and it is the one fixture here worth reading twice.** The reserve
    # is subtracted in the *first* assertion and deliberately absent from the *second*, because
    # it sits between net and credit rather than between gross and net -- that is design B
    # expressed as arithmetic. The fixture above would still pass if someone "tidied" the
    # implementation by folding ``reserve_cells`` into ``deducted``... no it would not, and this
    # comment is the proof: with the reserve in ``deducted`` the wedge reads
    # 10,000 - 1,000 = 9,000 against a declared net of 10,000 and raises. So the fixture above
    # *is* the regression test for the mis-wiring, and this is the case that says why it is one.
    # Kept as a comment rather than a second call because inverting the check would mean
    # asserting that a correct implementation fails.
    #
    # And the hole, the same shape as the TDS and refund pairs above: a held amount on a run
    # that never asked for one. Without the cell gate a reserve could appear from nowhere and
    # the aggregate would absorb it silently, since the first assertion would simply balance.
    must_raise(
        "I4",
        "a reserve cell while --reserve is off",
        lambda: check_totals(
            10_000, 10_000, 9_000, [0, 0], [0], [0], [1_000], reserve_on=False
        ),
    )
    # A shortfall the held amount does not account for: 1,000p held but 1,500p missing. This is
    # the reserved-run form of "settled but never credited", and it is the assertion that stops
    # ``--reserve`` becoming a licence for money to vanish -- which is exactly what suspending
    # I4 under the flag would have done.
    must_raise(
        "I4",
        "settled but never credited",
        lambda: check_totals(
            10_000, 10_000, 8_500, [0, 0], [0], [0], [1_000], reserve_on=True
        ),
    )
    # The full seven-term case, every flag on: fee 200p, GST 36p, TDS 10p, refund 1,000p inside
    # the net, then 500p held back outside it. Present for the reason the six-term case above
    # is: terms that each pass alone can still be composed wrongly, and this is the only
    # fixture where both sides of the net carry a deduction at once.
    check_totals(
        10_000, 8_754, 8_254, [200, 36], [10], [1_000], [500],
        fees_on=True, tds_on=True, refunds_on=True, reserve_on=True,
    )

    # I4 per settlement, with the refund term. The seventh element is optional, so the two
    # calls above this block still describe five-term rows -- what is checked here is that the
    # term is *subtracted* when present, and that a row omitting it is unchanged.
    check_settlement_arithmetic([("setl_x", 1_000_000, 976_400, 20_000, 3_600, 0, 0)])
    check_settlement_arithmetic([("setl_x", 1_000_000, 876_400, 20_000, 3_600, 0, 100_000)])
    must_raise(
        "I4",
        "a refund the net does not reflect",
        lambda: check_settlement_arithmetic(
            [("setl_x", 1_000_000, 976_400, 20_000, 3_600, 0, 100_000)]
        ),
    )

    # I3: a duplicate (date, amount) pair.
    # Cloning only the credit's amount would change the credited total and trip I4
    # first, so the whole chain behind it is rewritten -- payment gross, settlement
    # net, credit amount, decomposition.
    #
    # Phase 4b made the real thing available, and this hand-built story is now more useful
    # as the *wrong* plant than as a rehearsal for the right one: it collides one pair while
    # the config asks for two, it leaves the two settlements holding **distinct UTRs**, and
    # it leaves both credits marked ``resolvable=True``. All three are what I12 exists to
    # reject, so it serves below as I12's negative probe while remaining I3's.
    a, b = story.credits[0], story.credits[1]
    a_pid, b_pid = a.payment_ids[0], b.payment_ids[0]
    a_sid, b_sid = a.settlement_ids[0], b.settlement_ids[0]
    a_pay = next(p for p in story.payments if p.payment_id == a_pid)
    dup_story = dataclasses.replace(
        story,
        payments=[
            dataclasses.replace(
                p,
                gross_paise=a.amount_paise,
                captured_at=a_pay.captured_at.replace(minute=0, second=0),
            )
            if p.payment_id == b_pid
            else p
            for p in story.payments
        ],
        settlements=[
            dataclasses.replace(s, net_paise=a.amount_paise, settled_on=a.value_date)
            if s.settlement_id == b_sid
            else s
            for s in story.settlements
        ],
        credits=[
            dataclasses.replace(
                c,
                value_date=a.value_date,
                amount_paise=a.amount_paise,
                narration=f"IMPS-RZRPAY-{a_sid[-4:]}X",
                decomposition=Decomposition(gross_paise=a.amount_paise),
            )
            if c.credit_id == b.credit_id
            else c
            for c in story.credits
        ],
    )
    must_fail("I3", dup_story)

    # --- step 1: suspension works, and only the right flag does it ------------
    # A *genuinely* built --dup-amounts story: the duplicate is the intent, so
    # I3.unique_date_amount is suspended -- and the report says so, rather than quietly
    # running one check fewer. I12 must pass here, on data the generator really produces.
    dup_cfg = GenConfig(seed=42, n=60, flags=MessFlags(dup_amounts=True))
    real_dup = build(dup_cfg)
    rep = check_story(real_dup, dup_cfg)
    assert rep["checks_skipped"] == {"I3.unique_date_amount": ["dup_amounts"]}, rep

    # --- I12: the successor check must reject every wrong plant ----------------
    # Three ways to plant badly, and I12 exists because a suspension that is only
    # *announced* leaves all three passing. Each is probed on its own, because a check that
    # fires for the wrong reason is indistinguishable from one that works.
    #
    # (i) the wrong *count*. dup_story collides one pair; the config asks for two.
    must_fail("I12", dup_story, dup_cfg)

    # (ii) distinct UTRs -- the load-bearing clause. Same fixture, but with the count
    # reconciled so the first clause passes and this one is what fires. dup_story leaves the
    # two settlements holding their original, distinct tails, which is precisely the plant
    # that measured as *still separable* by a tail-only strategy (60/60, 200/200, 1000/1000
    # correct) before this flag was built.
    one_pair_cfg = GenConfig(seed=42, n=60, flags=MessFlags(dup_amounts=True), dup_pairs=1)
    must_fail("I12", dup_story, one_pair_cfg)

    # (iii) a planted row recorded as resolvable. Truth would then score an honest
    # abstention as a MISS, and the correct_abstention denominator would be wrong in the
    # direction that flatters the matcher.
    _planted_ids = [c.credit_id for c in real_dup.credits if not c.resolvable]
    assert len(_planted_ids) == 2 * dup_cfg.dup_pairs, _planted_ids
    mismarked = dataclasses.replace(
        real_dup,
        credits=[
            dataclasses.replace(c, resolvable=True, reason=None, note=None)
            if c.credit_id == _planted_ids[0]
            else c
            for c in real_dup.credits
        ],
    )
    must_fail("I12", mismarked, dup_cfg)

    # --fees must NOT excuse a collision. This is the exact conflation the old clean_mode
    # gate made: fees change amounts, so they raise collision pressure, but a collision is a
    # finding to record (ASSUMPTIONS.md #24) and not a check to switch off. Under the old
    # gate this story passed silently.
    fees_cfg = GenConfig(seed=42, n=60, flags=MessFlags(fees=True))
    must_fail("I3", dup_story, fees_cfg)
    assert check_story(story, fees_cfg)["checks_skipped"] == {}, (
        "--fees must suspend nothing at all"
    )

    # I8a: settlements ordered and numbered in payment order (what skipping the
    # shuffle in step 7 would produce). Renaming must propagate to the credits, or
    # the dangling references trip I2/I4 first and I8a is never reached -- which
    # would make this a test of the wrong thing.
    pay_ix = {p.payment_id: i for i, p in enumerate(story.payments)}
    in_payment_order = sorted(story.settlements, key=lambda s: pay_ix[s.payment_ids[0]])
    rename = {s.settlement_id: ids.settlement_id(i)
              for i, s in enumerate(in_payment_order, start=1)}
    must_fail(
        "I8a",
        dataclasses.replace(
            story,
            settlements=[dataclasses.replace(s, settlement_id=rename[s.settlement_id])
                         for s in in_payment_order],
            credits=[dataclasses.replace(c, settlement_ids=[rename[sid]
                                                            for sid in c.settlement_ids])
                     for c in story.credits],
        ),
    )

    # I11: either date off by a day, in either direction. Four cases, because the two
    # delays fail independently and only one of them is visible to the matcher -- a
    # capture->settlement error would never surface downstream, so if it is not caught
    # here it is not caught at all.
    from datetime import timedelta

    for label, mutate in (
        ("settled_on late", lambda st: dataclasses.replace(
            st, settlements=[dataclasses.replace(st.settlements[0],
                                                 settled_on=st.settlements[0].settled_on
                                                 + timedelta(days=7)),
                             *st.settlements[1:]])),
        ("settled_on early", lambda st: dataclasses.replace(
            st, settlements=[dataclasses.replace(st.settlements[0],
                                                 settled_on=st.settlements[0].settled_on
                                                 - timedelta(days=7)),
                             *st.settlements[1:]])),
        ("value_date late", lambda st: dataclasses.replace(
            st, credits=[dataclasses.replace(st.credits[0],
                                             value_date=st.credits[0].value_date
                                             + timedelta(days=7)),
                         *st.credits[1:]])),
        ("value_date early", lambda st: dataclasses.replace(
            st, credits=[dataclasses.replace(st.credits[0],
                                             value_date=st.credits[0].value_date
                                             - timedelta(days=7)),
                         *st.credits[1:]])),
    ):
        # Seven days, not one: a one-day nudge off a Friday lands on a weekend, and
        # add_business_days rolls a weekend forward to the same Monday -- so the story
        # would still satisfy I11 and the negative case would silently not fire.
        must_fail("I11", mutate(story))

    # I8b: bank rows ordered by capture time instead of amount
    by_pay_ix = {p.payment_id: i for i, p in enumerate(story.payments)}
    time_ordered = [
        dataclasses.replace(c, credit_id=ids.credit_id(i))
        for i, c in enumerate(
            sorted(story.credits, key=lambda c: (c.value_date, by_pay_ix[c.payment_ids[0]])),
            start=1,
        )
    ]
    must_fail("I8b", dataclasses.replace(story, credits=time_ordered))

    # --- Phase 5 step 1: batching, and the checks that replace what it suspends ---
    bat_cfg = GenConfig(seed=42, n=200, flags=MessFlags(batching=True))
    bat = build(bat_cfg)
    bat_rep = check_story(bat, bat_cfg)
    # The suspension is announced, and it is the *only* one --batching earns. In particular
    # I3.unique_date_amount stays strict (step 4): the batched-net nudge in story.py is what
    # keeps it satisfiable rather than a relaxation of the check.
    assert bat_rep["checks_skipped"] == {"I3.cardinality": ["batching"]}, bat_rep
    assert len(bat.settlements) < len(bat.payments), "--batching batched nothing"
    assert len(bat.settlements) == len(bat.credits), "one settlement, one credit"
    # Decision 2: both tiers must have rows to find, or the tier gate reads a swap.
    _sizes = [len(s.payment_ids) for s in bat.settlements]
    assert 1 in _sizes and max(_sizes) > 1, f"a mix of batch sizes is required, got {set(_sizes)}"
    assert sum(_sizes) == bat_cfg.n, "the member lists are not a partition of payments.csv"

    # I13 — the successor to the suspended cardinality check. Probed by lying about --n,
    # because that is the one thing I3 asserted that nothing else in this module covers:
    # measurement showed I2's unconditional clauses already catch a dropped or duplicated
    # member (see the comment at I13 itself), so a payment-count check is all that was owed.
    must_fail("I13", bat, GenConfig(seed=42, n=201, flags=MessFlags(batching=True)))

    # I14 — batch membership must not be a consecutive run of payment ids. Probed on the
    # standalone function rather than through a story: constructing a story whose batches are
    # all consecutive means reimplementing the grouping loop inside the test, which would
    # assert that the test agrees with itself. Population is above the floor on purpose --
    # 40 batches is the point at which the rate becomes assertable.
    must_raise(
        "I14",
        "consecutive-id batches",
        lambda: check_batch_adjacency([[i, i + 1] for i in range(0, 120, 2)]),
    )
    # ... and the honest case passes, or the probe above proves nothing. Interleaved pairs at
    # the same population: none is a consecutive run.
    _ok = check_batch_adjacency([[i, i + 60] for i in range(60)])
    assert _ok == (0, 60), _ok
    # Below the floor nothing is asserted, however bad the rate looks -- which is why n=60 is
    # not covered by this check. 100% of 10 batches must not raise.
    assert check_batch_adjacency([[i, i + 1] for i in range(0, 20, 2)]) == (10, 10)
    # Real batched data sits far below the ceiling: measured 8.9%-19.6% at n=200 against 100%
    # for capture-order slicing. Asserted as a *number* rather than trusting the ceiling,
    # because a ceiling nothing approaches is indistinguishable from a check that cannot fail.
    _c, _pop = (int(x) for x in str(bat_rep["consecutive_id_batches"]).split("/"))
    assert _pop >= MIN_MULTI_BATCH_POPULATION_FOR_RATE, (
        f"only {_pop} multi-member batches at n=200 -- below the floor, so I14 did not "
        f"actually assert anything on this run"
    )
    assert _c / _pop < 0.30, f"consecutive-id rate {_c}/{_pop} is far above the measured 8.9-19.6%"

    # I14 must also *fire* end to end, not only as a unit probe -- the regression it guards
    # is someone slicing a date's pool without shuffling it first, and that arrives as a whole
    # story rather than as a call into one function.
    #
    # **Regrouped within each settlement date, not across the file.** The first version of this
    # probe handed settlements the next k payments in global id order, and I11 caught it before
    # I14 could: reassigning members across dates moves a settlement's date away from its
    # members' capture dates, which is exactly what I11 asserts. So the probe reproduces the
    # bug faithfully -- consecutive ids *within* the date, which is what an unshuffled slice
    # actually produces -- and every member keeps the settlement date it had.
    _p_by_date: dict[date, list[str]] = defaultdict(list)
    for _p in bat.payments:  # capture order, which is id order
        _p_by_date[_p.business_date].append(_p.payment_id)
    _s_by_date: dict[date, list[object]] = defaultdict(list)
    for _s in sorted(bat.settlements, key=lambda s: s.settlement_id):
        _s_by_date[_s.settled_on].append(_s)

    _gross_of = {p.payment_id: p.gross_paise for p in bat.payments}
    _regrouped: dict[str, list[str]] = {}
    for _when, _setls in _s_by_date.items():
        _pool = _p_by_date[_when]
        # Batches never span a settlement date, so a date's settlements hold exactly that
        # date's payments. Asserted, because if it were false the probe would be silently
        # testing something else.
        assert sum(len(s.payment_ids) for s in _setls) == len(_pool), (
            f"{_when}: batches do not partition their date"
        )
        _cur = 0
        for _s in _setls:
            _k = len(_s.payment_ids)
            _regrouped[_s.settlement_id] = _pool[_cur : _cur + _k]
            _cur += _k

    # The nets have to be rewritten to match the new member lists, or I4 fires first and the
    # probe tests the wrong assertion.
    _contig_settlements = [
        dataclasses.replace(
            _s,
            payment_ids=_regrouped[_s.settlement_id],
            net_paise=sum(_gross_of[pid] for pid in _regrouped[_s.settlement_id]),
        )
        for _s in bat.settlements
    ]
    _contig_by_id = {s.settlement_id: s for s in _contig_settlements}
    _contig_credits = [
        dataclasses.replace(
            c,
            amount_paise=_contig_by_id[c.settlement_ids[0]].net_paise,
            payment_ids=list(_contig_by_id[c.settlement_ids[0]].payment_ids),
            decomposition=Decomposition(
                gross_paise=_contig_by_id[c.settlement_ids[0]].net_paise
            ),
        )
        for c in bat.credits
    ]
    must_fail(
        "I14",
        dataclasses.replace(bat, settlements=_contig_settlements, credits=_contig_credits),
        bat_cfg,
    )

    # --- Phase 6 step 6: I15 must fire, not merely exist ----------------------
    # I15 landed in step 6 with no negative case at all, which by this file's own standard
    # makes it decoration -- every other check here is mutant-tested. Added with step 7.
    ref_cfg = GenConfig(seed=42, n=60, flags=MessFlags(netted_refunds=True))
    ref = build(ref_cfg)
    ref_rep = check_story(ref, ref_cfg)
    assert ref_rep["checks_skipped"] == {"I3.no_refunds": ["netted_refunds"]}, ref_rep
    assert ref.refunds, "--netted-refunds emitted no refunds"

    # **Cross-attribution: two credits swap the refund ids they cite, keeping every
    # ``refunds_paise`` term exactly where it was.** This is the mutant I15 was built for and
    # the one nothing else in this file can see: the run-wide refund total is unchanged, every
    # settlement's net is unchanged, and I4 passes in full -- both in aggregate and per
    # settlement -- while truth now says the wrong refund paid for the wrong credit. A total
    # cannot detect it, which is precisely why I15 compares term by term.
    _cited = [c for c in ref.credits if c.refunds_netted]
    assert len(_cited) >= 2, f"need two refund-citing credits to swap, got {len(_cited)}"
    _a, _b = _cited[0], _cited[1]
    must_fail(
        "I15",
        dataclasses.replace(
            ref,
            credits=[
                dataclasses.replace(_a, refunds_netted=list(_b.refunds_netted))
                if c is _a
                else dataclasses.replace(_b, refunds_netted=list(_a.refunds_netted))
                if c is _b
                else c
                for c in ref.credits
            ],
        ),
        ref_cfg,
    )
    # And the orphan-holder pin: marking that row unresolvable is the "fix" a later phase would
    # reach for, and it would move a real miss into the correct-abstention cell.
    _orphan_holder = next(
        (c for c in ref.credits if c.note and "not in this month's payments.csv" in c.note),
        None,
    )
    assert _orphan_holder is not None, "the --netted-refunds run planted no orphan refund"
    must_fail(
        "I15",
        dataclasses.replace(
            ref,
            credits=[
                dataclasses.replace(c, resolvable=False, reason=str(Reason.REFUND_UNLINKED))
                if c is _orphan_holder
                else c
                for c in ref.credits
            ],
        ),
        ref_cfg,
    )

    # --- Phase 6 step 7: the reserve, and I16's four mutants ------------------
    res_cfg = GenConfig(seed=42, n=60, flags=MessFlags(reserve=True))
    res = build(res_cfg)
    res_rep = check_story(res, res_cfg)
    # **No suspension at all, which is the correction step 7 is built on.** ``SUSPENDED_BY``
    # lists ``I2.every_settlement_credited`` and ``I3.no_orphans`` under ``reserve`` -- both
    # written in earlier phases as predictions, before any reserve code existed, and both
    # wrong: under design B a reserved settlement still has its credit, merely a short one.
    # This assertion is what would fail if a future edit made the flag start dropping credits,
    # and it is stronger than the entries it supersedes because those checks now *run*.
    assert res_rep["checks_skipped"] == {}, (
        f"--reserve is expected to suspend nothing -- design B keeps every settlement "
        f"credited and every per-settlement wedge intact: {res_rep['checks_skipped']}"
    )
    _res_held = [c for c in res.credits if c.decomposition.reserve_paise]
    assert _res_held, "--reserve held nothing back"
    assert len(_res_held) < len(res.credits), "--reserve held from every credit, not partially"
    # The aggregate really is short, or the flag is a no-op that the invariants happen to pass.
    assert res.total_net_paise() > res.total_credited_paise(), (
        "--reserve left the credited total equal to the net -- nothing was held"
    )

    _net_by_sid = {s.settlement_id: s for s in res.settlements}
    _held_credit = _res_held[0]
    _own_net = _net_by_sid[_held_credit.settlement_ids[0]].net_paise

    # Mutant 1 -- **the silent wrong match, and the reason I16 exists at all.** The reserved
    # credit's short amount is moved onto *another settlement's net*, with the held figure
    # adjusted so the arithmetic still closes perfectly. I4 cannot see this: ``expected ==
    # amount`` holds, the aggregate holds, every per-settlement wedge holds. But a matcher now
    # finds exactly one candidate for this row -- the wrong settlement -- with a gap of zero,
    # and resolves it with total confidence. That is a WRONG_MATCH on the line this project
    # says never bends, and it is invisible to every check that existed before I16.
    _other_net = min(s.net_paise for s in res.settlements)
    assert _other_net < _own_net, "need a smaller net to collide with"
    _collide = _own_net - _other_net
    must_fail(
        "I16",
        dataclasses.replace(
            res,
            credits=[
                dataclasses.replace(
                    c,
                    amount_paise=_other_net,
                    reserve_held_paise=_collide,
                    decomposition=dataclasses.replace(
                        c.decomposition, reserve_paise=_collide
                    ),
                )
                if c is _held_credit
                else c
                for c in res.credits
            ],
        ),
        res_cfg,
    )

    # Mutant 2 -- a reserved row marked unresolvable. The mirror of I15's orphan pin, and the
    # "fix" that looks most like an improvement: it would inflate correct_abstention with rows
    # a tail-only join separates, and make LUCKY_GUESS fire on a matcher that got it right.
    must_fail(
        "I16",
        dataclasses.replace(
            res,
            credits=[
                dataclasses.replace(
                    c, resolvable=False, reason=str(Reason.PARTIAL_SETTLEMENT_PENDING)
                )
                if c is _held_credit
                else c
                for c in res.credits
            ],
        ),
        res_cfg,
    )

    # Mutant 3 -- the two fields that carry this one fact disagree. A reader may consult
    # either, so a divergence makes the answer key self-contradictory about the only record of
    # this money that exists anywhere.
    must_fail(
        "I16",
        dataclasses.replace(
            res,
            credits=[
                dataclasses.replace(c, reserve_held_paise=c.reserve_held_paise + 1)
                if c is _held_credit
                else c
                for c in res.credits
            ],
        ),
        res_cfg,
    )

    # Mutant 4 -- the flag is on and nothing is held. Every reserve is zeroed and every credit
    # restored to its net, so I4 passes in full and the run is arithmetically perfect -- it is
    # simply not the run it says it is. A dataset labelled with a mess it does not contain is
    # the mislabelling ``MessFlags.IMPLEMENTED`` exists to prevent, arriving by a different
    # door.
    must_fail(
        "I16",
        dataclasses.replace(
            res,
            credits=[
                dataclasses.replace(
                    c,
                    amount_paise=_net_by_sid[c.settlement_ids[0]].net_paise,
                    reserve_held_paise=0,
                    decomposition=dataclasses.replace(c.decomposition, reserve_paise=0),
                )
                for c in res.credits
            ],
        ),
        res_cfg,
    )

    # Mutant 5 -- a held reserve recorded on a run with the flag off.
    #
    # **Mutated via ``reserve_held_paise`` rather than the decomposition, and the reason is a
    # measurement worth keeping.** The obvious version sets ``decomposition.reserve_paise=1``,
    # and that fires **I4**, not I16: ``check_totals`` runs first and the aggregate no longer
    # balances ("net=55982339 - reserve=1 = 55982338 credited=55982339"). Two independent checks
    # catching one corruption is defence in depth and exactly what should happen -- but it means
    # that mutant cannot demonstrate I16's own zero gate, since the story dies before reaching
    # it. ``reserve_held_paise`` is the field **no other check reads** (it is absent from
    # ``expected_credit_paise`` and from every wedge), so it is the one that isolates this
    # branch: nothing else in the file can tell that this credit claims a reserve.
    must_fail(
        "I16",
        dataclasses.replace(
            story,
            credits=[
                dataclasses.replace(story.credits[0], reserve_held_paise=1),
                *story.credits[1:],
            ],
        ),
    )

    # Counted, not written down: the literal "6" here went stale the moment Phase 4
    # added a seventh case, and a self-check that misreports its own coverage is a
    # small version of exactly the problem this phase is fixing.
    print(
        f"invariants.py self-check ok  ({len(fired)} negative cases fire correctly: "
        f"{', '.join(sorted(set(fired)))})"
    )
