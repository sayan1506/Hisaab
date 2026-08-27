"""The stages, in order: load -> index -> resolve -> one verdict per bank row.

Phase 3 has exactly one matching stage, so this module is thin on purpose. Its job is
to be the place a stage gets *added* -- Phase 5's tier 2 (subset-sum) and Phase 7's
orphan sweep slot in here, after tier 1 and before the verdict list is sealed, without
any of them having to know how the others are wired.

Two things the engine owns that no single stage can:

  * **Completeness.** Every bank row gets exactly one verdict, and the count is asserted
    before the file is built. ``verdict_io.reconcile`` refuses a file that drops a row;
    finding that out here costs a line, and finding it out at review time costs the run.
  * **Determinism.** Verdicts are emitted in bank-file order, and candidate lists are
    already sorted by ``settlement_id`` inside ``blocking``. Nothing iterates a ``set``
    on the path from input to output.

**Provenance is passed in, never discovered.** ``seed`` and ``month`` are carried into
``matches.json`` so the scorer can refuse to grade run A against run B's answer key --
the most expensive available bug, because it produces a plausible number rather than a
crash. They arrive as arguments because the matcher genuinely cannot derive them: the
seed is nowhere in ``data/``, and ``run_manifest.json`` lives under ``truth/``, which
this package may not read. ``cli.py`` defaults the month from the bank statement's own
dates; the seed defaults to the generator's default and is a stated claim about which
run this is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..common.verdict import Outcome, Verdict, VerdictFile
from .blocking import DEFAULT_MAX_ADJUSTMENT_PAISE, DEFAULT_WINDOW_DAYS, SettlementIndex
from .fees import FeeSchedule, unpriced_methods
from .load import Dataset
from .tier1 import TIER, resolve_credit

#: Bumped when the matching *behaviour* changes, so a verdict file names the engine that
#: produced it. Tier 2 arriving in Phase 5 makes this ``tier1+2@0.5.0``.
MATCHER_NAME = "tier1@0.3.0"

#: The fields that constitute a *decision*. ``note`` is deliberately absent: it carries
#: the UTR corroboration, so blanking every narration legitimately changes it while
#: leaving every decision untouched. That distinction is what acceptance item 3 actually
#: tests -- see ``tools/acceptance.py`` gate 9 -- and keeping the list here means the
#: gate and the matcher cannot drift about what counts as a decision.
DECISION_FIELDS: tuple[str, ...] = (
    "credit_id", "outcome", "settlement_ids", "payment_ids", "tier", "reason",
    "residual_paise",
)


def decision_signature(verdict: Verdict) -> tuple[object, ...]:
    """Everything about a verdict except its prose. See ``DECISION_FIELDS``."""
    return tuple(getattr(verdict, field) for field in DECISION_FIELDS)


@dataclass(frozen=True, slots=True)
class RunSummary:
    """What the CLI prints. Counts only -- no per-row answers, nothing from truth."""

    bank_rows: int
    resolved: int
    exceptions: int
    ignored: int
    residual_nonzero: int
    settlements_indexed: int
    amount_collisions: int
    window_days: int
    max_adjustment_paise: int
    wall_clock_seconds: float
    #: The fee rates this run assumed, as one line. Printed by the CLI because the rates
    #: are an *assumption* about a counterparty (ASSUMPTIONS.md #5-#9), and a reconciliation
    #: result is only interpretable next to the rates that produced it.
    fee_rates: str = ""
    #: Payment methods present in the data that the schedule cannot price. Every row using
    #: one is unexplainable, which is worth saying once at the top rather than leaving a
    #: reader to infer it from a pile of identical exceptions.
    unpriced: tuple[str, ...] = ()

    @property
    def coverage_claimed(self) -> float | None:
        """The matcher's *own* count of how often it committed.

        Deliberately named ``claimed``: this is not a score. Whether those commitments
        were right is ``hisaab.scoring``'s answer and requires the answer key, which
        this package cannot read. A matcher reporting its own accuracy would be grading
        its own homework.
        """
        if not self.bank_rows:
            return None
        return self.resolved / self.bank_rows


def run(
    dataset: Dataset,
    seed: int,
    month: str,
    window_days: int = DEFAULT_WINDOW_DAYS,
    max_adjustment_paise: int = DEFAULT_MAX_ADJUSTMENT_PAISE,
    schedule: FeeSchedule | None = None,
) -> tuple[VerdictFile, RunSummary]:
    """Match every bank row. Returns the verdict file and a summary for the CLI."""
    started = time.perf_counter()

    rates = schedule or FeeSchedule()
    index = SettlementIndex(dataset.settlements)
    verdicts = tuple(
        resolve_credit(
            credit, index, dataset,
            window_days=window_days,
            max_adjustment_paise=max_adjustment_paise,
            schedule=rates,
        )
        for credit in dataset.credits
    )

    elapsed = time.perf_counter() - started

    # One verdict per bank row, asserted here rather than discovered by the scorer.
    if len(verdicts) != len(dataset.credits):
        raise AssertionError(
            f"internal: {len(verdicts)} verdicts for {len(dataset.credits)} bank rows"
        )

    run_file = VerdictFile(
        seed=seed,
        month=month,
        matcher=MATCHER_NAME,
        verdicts=verdicts,
        wall_clock_seconds=elapsed,
    )

    counts = run_file.counts()
    summary = RunSummary(
        bank_rows=len(dataset.credits),
        resolved=counts["RESOLVED"],
        exceptions=counts["EXCEPTION"],
        ignored=counts["IGNORED"],
        residual_nonzero=sum(
            1 for v in verdicts
            if v.outcome is Outcome.RESOLVED and v.residual_paise
        ),
        settlements_indexed=len(index),
        amount_collisions=index.amount_collisions(),
        window_days=window_days,
        max_adjustment_paise=max_adjustment_paise,
        wall_clock_seconds=elapsed,
        fee_rates=rates.describe(),
        unpriced=tuple(unpriced_methods((p.method for p in dataset.payments), rates)),
    )
    return run_file, summary


if __name__ == "__main__":
    from pathlib import Path

    from .load import load

    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    if not (data_dir / "bank_statement.csv").exists():
        print("engine.py self-check skipped  (no committed data/ -- generate a run first)")
        raise SystemExit(0)

    dataset = load(data_dir)
    run_file, summary = run(dataset, seed=42, month="2026-08")

    assert len(run_file.verdicts) == summary.bank_rows == len(dataset.credits)
    assert run_file.matcher == MATCHER_NAME
    assert summary.resolved == summary.bank_rows, (
        f"{summary.resolved}/{summary.bank_rows} resolved -- clean mode must be fully "
        f"resolvable, so a shortfall here is a matcher bug"
    )
    assert summary.exceptions == 0 and summary.ignored == 0
    assert summary.residual_nonzero == 0, "clean mode has no fees, so every residual is 0"
    assert summary.coverage_claimed == 1.0
    assert summary.window_days == 0 and summary.max_adjustment_paise == 0

    # Every verdict is a real Verdict object, so the contract's guards all ran.
    assert all(isinstance(v, Verdict) for v in run_file.verdicts)
    assert all(v.tier == TIER for v in run_file.verdicts)

    # --- determinism: two runs agree everywhere except the wall clock ------
    again, _ = run(load(data_dir), seed=42, month="2026-08")
    assert [decision_signature(v) for v in run_file.verdicts] == [
        decision_signature(v) for v in again.verdicts
    ], "two runs over the same data disagreed on a decision"
    left = {k: v for k, v in run_file.as_json().items() if k != "timing"}
    right = {k: v for k, v in again.as_json().items() if k != "timing"}
    assert left == right, "the verdict document moved outside its timing block"
    # ...and the clock is genuinely carried, so the comparison above means something.
    assert run_file.wall_clock_seconds is not None and run_file.wall_clock_seconds > 0

    # Verdicts follow bank-file order, which is what makes the bytes stable.
    assert [v.credit_id for v in run_file.verdicts] == [c.credit_id for c in dataset.credits]

    # --- the window is real, not decorative -------------------------------
    # Nudging the window must not change a clean-mode result (every distance is 0),
    # but a negative one must be refused rather than silently clamped.
    wider, _ = run(dataset, seed=42, month="2026-08", window_days=3)
    assert [decision_signature(v) for v in wider.verdicts] == [
        decision_signature(v) for v in run_file.verdicts
    ], "a wider window changed a clean-mode verdict -- (date, amount) is not unique"
    try:
        run(dataset, seed=42, month="2026-08", window_days=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("run() accepted a negative window")

    print(
        f"engine.py self-check ok  ({summary.resolved}/{summary.bank_rows} resolved, "
        f"{summary.settlements_indexed} settlements indexed, "
        f"{summary.wall_clock_seconds * 1000:.0f} ms)"
    )
