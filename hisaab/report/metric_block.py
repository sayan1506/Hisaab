"""Renders the metric block by calling ``hisaab.scoring.report.metric_block`` unmodified.

**Why this imports ``hisaab.scoring`` at all, and what that import actually buys.**
``.plan/phase11.md`` step 4 says to quote ``metric_block()``'s text output verbatim, not a
table re-derived from the JSON fields. That function's signature is ``metric_block(m:
Metrics) -> str``, and it calls ``roi(m)``, which reads ``m.landings`` -- a Python-only field
on the in-memory dataclass (``field(repr=False)``), never serialised by ``Metrics.as_json()``
(measured in `.plan/phase11.md` correction (1)). So "verbatim" and "read only the persisted
JSON" cannot both hold without one more piece: a way to rebuild a ``Metrics`` object whose
``roi()`` split agrees with the run that produced the JSON, without importing the run's actual
per-row landings, which were never written to disk in the first place.

This module is that piece, and importing ``Metrics``/``Cell``/``metric_block`` for it costs
this package a deliberate ``tools/check_isolation.py`` ``TRUTH_READERS`` entry -- the same
trade ``hisaab/scoring/report.py`` itself already makes, for the same reason its own docstring
gives: that check treats importing *anything* under ``hisaab.scoring`` as reaching truth,
because a module that can import the package can import the loader, so the allowlist cannot
express "formats but never reads." The property that actually matters is enforced structurally
instead -- this module never touches ``hisaab.scoring.truth_io`` and is handed nothing but a
JSON document a run already wrote to disk.

**How the easy/hard split is rebuilt exactly, from aggregates alone.**
``roi()`` partitions bank rows into "easy" (cleared on sight) and "hard" (chased), counting a
row hard exactly when ``outcome is EXCEPTION and cell is not NOISE_MISHANDLED``. Two shortcuts
that look equivalent were tried and measured wrong before this one:

  * **Cell-only** (``hard = cells[MISSED] + cells[CORRECT_ABSTENTION]``) is wrong on a planted
    unresolvable that was *dismissed* (``IGNORED``) rather than raised as an ``EXCEPTION``:
    ``CORRECT_ABSTENTION`` fires on both outcomes, and a dismissed one is easy, not hard.
  * **The shortcut ``hisaab/scoring/report.py``'s own docstring measures** (``hard =
    exceptions - cells[NOISE_MISHANDLED]``) is wrong the direction its own docstring warns
    about: a noise row wrongly ``RESOLVED`` lands in ``NOISE_MISHANDLED`` without ever being
    counted in ``exceptions`` at all, so subtracting it *undercounts* hard by double.

The exact fix: ``CORRECT_ABSTENTION`` is the only cell whose outcome is ambiguous from the
cells dict alone, and the dismissals block's own count resolves it. Every ``IGNORED`` verdict
lands in exactly one of ``WRONG_IGNORE``, ``NOISE_CORRECTLY_IGNORED`` or
``CORRECT_ABSTENTION`` (``_classify``'s three ``IGNORED``-reachable branches), so::

    abstained_via_ignored    = dismissals.count - cells[WRONG_IGNORE] - cells[NOISE_CORRECTLY_IGNORED]
    abstained_via_exception  = cells[CORRECT_ABSTENTION] - abstained_via_ignored
    hard                     = cells[MISSED] + abstained_via_exception
    easy                     = totals.bank_rows - hard

Verified against ``hisaab.scoring.report.roi()`` on eleven constructed runs (this module's own
self-check), including the two edge cases above and a run combining both in one file. Proven,
not merely tested: every ``IGNORED`` verdict is classified by exactly one of ``_classify``'s
three ``IGNORED`` branches (``hisaab/scoring/metrics.py``), so the subtraction cannot double-count
or miss a row -- the self-check exists to catch a future change to that partition, not because
the identity itself is in doubt.

**What is synthesised, and why it cannot mislead.** ``metric_block()`` touches ``m.landings``
only inside ``roi()``, and only for ``.outcome`` and ``.cell`` -- never for ``credit_id``,
``value_paise`` or any other field ``exception_queue()`` would print. So the ``Landing`` tuple
built here carries exactly enough real landings to reproduce the split's *count*, with every
other field a placeholder. No id, no amount and no reason code in this module's synthetic
landings is ever real, and none of them reaches the page: this module renders only the string
``metric_block()`` returns.
"""

from __future__ import annotations

from typing import Any

from ..common.verdict import Outcome

#: Duplicated from ``hisaab.report.assemble`` rather than imported, for the reason that
#: module's own comment gives: a schema constant is safe to copy, and copying it here keeps
#: this module's only ``hisaab.scoring`` import to the two symbols that need the allowlist
#: entry -- ``Metrics``, ``Cell`` and ``metric_block`` -- rather than reaching back into
#: ``assemble`` for a version number that has nothing to do with rendering.
METRICS_SCHEMA_VERSION = 5

# The two symbols this module exists to import. Importing *anything* under ``hisaab.scoring``
# counts as reaching truth under ``tools/check_isolation.py``'s own rule, so this module is
# listed on ``TRUTH_READERS`` -- transitively, the same way ``hisaab/scoring/report.py`` is:
# these two names are types and a formatter, never ``Truth`` or ``truth_io``.
from ..scoring.metrics import Cell, Landing, Metrics  # noqa: E402
from ..scoring.report import metric_block as _metric_block  # noqa: E402


class MetricBlockError(Exception):
    """The metrics document's schema version is not one this module knows how to rebuild."""


def _cell_counts(doc: dict[str, Any]) -> dict[Cell, int]:
    raw = doc["cells"]
    counts: dict[Cell, int] = {}
    for c in Cell:
        try:
            counts[c] = int(raw[str(c)])
        except KeyError:
            raise MetricBlockError(
                f"the metrics document's cells block has no entry for {c!r}"
            ) from None
    return counts


def _synthetic_landings(cells: dict[Cell, int], total_bank_rows: int, dismissals_count: int) -> tuple[Landing, ...]:
    """Just enough ``Landing``s for ``roi()``'s split to agree with the real run.

    See the module docstring for the derivation. ``hard`` counts rows ``roi()`` would chase;
    the rest are easy. Every field but ``outcome`` and ``cell`` is a placeholder -- see the
    module docstring's "what is synthesised" section for why that is safe.
    """
    abstained_via_ignored = (
        dismissals_count - cells[Cell.WRONG_IGNORE] - cells[Cell.NOISE_CORRECTLY_IGNORED]
    )
    abstained_via_exception = cells[Cell.CORRECT_ABSTENTION] - abstained_via_ignored
    hard = cells[Cell.MISSED] + abstained_via_exception
    easy = total_bank_rows - hard
    if hard < 0 or easy < 0:
        raise MetricBlockError(
            f"the easy/hard split went negative (easy={easy}, hard={hard}) -- the metrics "
            f"document's cells and dismissals counts are inconsistent with each other"
        )

    hard_landing = Landing(
        credit_id="", cell=Cell.MISSED, outcome=Outcome.EXCEPTION,
        reason=None, residual_paise=None, value_paise=0,
    )
    easy_landing = Landing(
        credit_id="", cell=Cell.CORRECT, outcome=Outcome.RESOLVED,
        reason=None, residual_paise=None, value_paise=0,
    )
    return (hard_landing,) * hard + (easy_landing,) * easy


def reconstruct(doc: dict[str, Any]) -> Metrics:
    """Rebuild a ``Metrics`` object from the scoring ``--out`` document.

    Every field is read straight from the document except ``landings``, which is
    synthesised (see ``_synthetic_landings``) because it was never serialised in the first
    place. The result reproduces ``metric_block()``'s text exactly for the run that produced
    ``doc`` -- verified for eleven runs in this module's self-check -- but it is not a real
    ``Metrics``: nothing downstream of this function should read ``.landings`` for anything
    other than what ``roi()`` reads it for.

    ``wrong_match_value_paise`` (Phase 12) is the one money field this function does not
    have to reconstruct from a shortcut: ``score()`` already prices it directly onto
    ``Metrics`` and serialises it in the document's own ``risk`` block, so it is read here
    the same way ``exception_value_paise`` is -- a real number, not a placeholder tied to
    ``_synthetic_landings``'s zeroed ``value_paise``.
    """
    version = doc.get("schema_version")
    if version != METRICS_SCHEMA_VERSION:
        raise MetricBlockError(
            f"the scoring document is schema v{version!r}, but this module rebuilds v"
            f"{METRICS_SCHEMA_VERSION}. Re-run the scorer, or update this module -- do not "
            f"guess at the difference."
        )

    run, totals = doc["run"], doc["totals"]
    cells = _cell_counts(doc)
    landings = _synthetic_landings(
        cells, totals["bank_rows"], doc["dismissals"]["count"],
    )

    return Metrics(
        seed=run["seed"],
        month=run["month"],
        clean_mode=run["clean_mode"],
        flags_enabled=tuple(run["flags"]),
        matcher=run["matcher"],
        wall_clock_seconds=doc["timing"]["wall_clock_seconds"],
        total_bank_rows=totals["bank_rows"],
        gateway_credits=totals["gateway_credits"],
        non_gateway_credits=totals["non_gateway_credits"],
        planted_unresolvable=totals["planted_unresolvable"],
        total_payments=totals["payments"],
        cells=cells,
        landings=landings,
        exceptions=doc["exceptions"]["count"],
        exception_value_paise=doc["exceptions"]["value_paise"],
        exception_minutes=doc["exceptions"]["estimated_minutes"],
        wrong_match_value_paise=doc["risk"]["wrong_match_value_paise"],
        ignores_total=doc["dismissals"]["count"],
        dismissal_minutes=doc["dismissals"]["estimated_minutes"],
        decomposition_checked=doc["decomposition"]["checked"],
        decomposition_mismatches=doc["decomposition"]["mismatches"],
    )


def render(doc: dict[str, Any]) -> str:
    """The metric block, as text, for the report's own section -- ``metric_block()`` unmodified."""
    return _metric_block(reconstruct(doc))


if __name__ == "__main__":
    from ..common.reasons import Reason
    from ..common.verdict import Decomposition, Outcome as VOutcome, Verdict, VerdictFile
    from ..scoring.metrics import score
    from ..scoring.truth_io import Truth, TruthCredit, TruthDecomposition

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
        return Verdict(
            cid, VOutcome.RESOLVED, (f"setl_{cid[1:]}",), pids, tier=1,
            residual_paise=residual,
            credit_amount_paise=gross - fee - gst + residual,
            decomposition=Decomposition(gross, fee_paise=fee, gst_paise=gst),
        )

    def _check(label: str, m) -> None:
        want = _metric_block(m)
        got = render(m.as_json())
        assert got == want, f"{label}: reconstruction diverged\n--- real ---\n{want}\n--- rebuilt ---\n{got}"

    three = tuple(_credit(f"C{i:04d}", (f"pay_{i:04d}",)) for i in (1, 2, 3))
    _check("oracle", score(_run(tuple(_resolved(c.credit_id, c.payment_ids) for c in three)), _truth(three)))

    _check("stub", score(_run(tuple(
        Verdict(c.credit_id, VOutcome.EXCEPTION, reason=Reason.NO_CANDIDATE) for c in three
    )), _truth(three)))

    _check("losing", score(
        _run((Verdict("C0001", VOutcome.EXCEPTION, reason=Reason.FX_RATE_GAP),)),
        _truth((_credit("C0001", ("pay_0001",)),)),
    ))

    noisy = tuple(_credit(f"C{i:04d}", (f"pay_{i:04d}",)) for i in (1,))
    _check("all_noise", score(
        _run((
            _resolved("C0001", ("pay_0001",)),
            Verdict("C0007", VOutcome.IGNORED, reason=Reason.NON_GATEWAY_CREDIT),
            Verdict("C0008", VOutcome.IGNORED, reason=Reason.NON_GATEWAY_CREDIT),
            Verdict("C0009", VOutcome.IGNORED, reason=Reason.NON_GATEWAY_CREDIT),
        )),
        _truth(noisy, noise=("C0007", "C0008", "C0009")),
    ))

    easy_ids = tuple(f"C{i:04d}" for i in range(1, 11))
    _check("unconditional (negative crossing)", score(
        _run(
            tuple(_resolved(cid, (f"pay_{cid[1:]}",)) for cid in easy_ids)
            + (Verdict("C0011", VOutcome.EXCEPTION, reason=Reason.PARTIAL_SETTLEMENT_PENDING),)
        ),
        _truth(
            tuple(_credit(cid, (f"pay_{cid[1:]}",)) for cid in easy_ids)
            + (_credit("C0011", ("pay_0011",)),)
        ),
    ))

    _check("wrong (claim withheld)", score(
        _run((_resolved("C0001", ("pay_0002",)), _resolved("C0002", ("pay_0002",)))),
        _truth((_credit("C0001", ("pay_0001",)), _credit("C0002", ("pay_0002",)))),
    ))

    _check("planted, raised as EXCEPTION", score(
        _run((Verdict("C0001", VOutcome.EXCEPTION, reason=Reason.AMBIGUOUS_DUPLICATE_AMOUNT),)),
        _truth((_credit("C0001", ("pay_0001",), resolvable=False,
                        reason=Reason.AMBIGUOUS_DUPLICATE_AMOUNT),)),
    ))

    _check("mixed (noise apart from headline)", score(
        _run((
            _resolved("C0001", ("pay_0001",)),
            Verdict("C0002", VOutcome.IGNORED, reason=Reason.NON_GATEWAY_CREDIT),
        )),
        _truth((_credit("C0001", ("pay_0001",)),), noise=("C0002",)),
    ))

    # --- the edge case that breaks the cell-only shortcut: a planted unresolvable
    # dismissed via IGNORED rather than raised as an EXCEPTION. CORRECT_ABSTENTION fires
    # either way; only the dismissals count disambiguates which side of "hard" it is on.
    _check("planted, dismissed via IGNORED", score(
        _run((Verdict("C0001", VOutcome.IGNORED, reason=Reason.AMBIGUOUS_DUPLICATE_AMOUNT),)),
        _truth((_credit("C0001", ("pay_0001",), resolvable=False,
                        reason=Reason.AMBIGUOUS_DUPLICATE_AMOUNT),)),
    ))

    # --- the edge case that breaks report.py's own named shortcut: a noise row wrongly
    # RESOLVED lands in NOISE_MISHANDLED without ever being counted in `exceptions` --
    # subtracting it from `exceptions` undercounts hard by double. This module's formula
    # never looks at `exceptions` at all, so it is not exposed to that trap.
    _check("noise row wrongly RESOLVED", score(
        _run((_resolved("C0001", ("pay_0001",)), _resolved("C0002", ("pay_0002",)))),
        _truth((_credit("C0001", ("pay_0001",)),), noise=("C0002",)),
    ))

    # --- both edge cases in one run, to prove the two corrections compose -------------
    _check("planted via EXCEPTION and via IGNORED together", score(
        _run((
            Verdict("C0001", VOutcome.EXCEPTION, reason=Reason.AMBIGUOUS_DUPLICATE_AMOUNT),
            Verdict("C0002", VOutcome.IGNORED, reason=Reason.AMBIGUOUS_DUPLICATE_AMOUNT),
        )),
        _truth((
            _credit("C0001", ("pay_0001",), resolvable=False, reason=Reason.AMBIGUOUS_DUPLICATE_AMOUNT),
            _credit("C0002", ("pay_0002",), resolvable=False, reason=Reason.AMBIGUOUS_DUPLICATE_AMOUNT),
        )),
    ))

    # --- schema version refusal --------------------------------------------------------
    stub_doc = score(_run(tuple(
        Verdict(c.credit_id, VOutcome.EXCEPTION, reason=Reason.NO_CANDIDATE) for c in three
    )), _truth(three)).as_json()
    bad = {**stub_doc, "schema_version": 999}
    try:
        render(bad)
    except MetricBlockError as e:
        assert "v999" in str(e), e
    else:
        raise AssertionError("accepted an unknown schema version")

    print("report/metric_block.py self-check ok -- 11 runs reconstruct byte-identical text")
