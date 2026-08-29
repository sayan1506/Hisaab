"""Writers — CSVs to ``--out``, truth to ``--truth``, deterministic throughout.

Two directories, one rule: ``data/`` is what the matcher reads, ``truth/`` is what
it must never read. The split is structural, not a promise, and it is the thing to
point at when a judge asks how we know the answer key did not leak.

**Trap 5 — Windows newlines.** ``csv.writer`` defaults to ``\\r\\n``, and a file
opened without ``newline=""`` on Windows turns that into ``\\r\\r\\n``. Output stops
being byte-identical across platforms, so a judge on macOS cannot reproduce our
hashes. Every write here is ``newline=""`` plus an explicit ``lineterminator="\\n"``.

**Determinism.** Row order is fixed by the caller (the story is already ordered),
key order is fixed by explicit construction, and hashes are computed by reading the
file back off disk rather than from the buffer we meant to write -- so the reported
hash is what a judge will actually get.

``sort_keys`` is deliberately **off** for ``truth.json``. Determinism comes from
explicit key construction (verified by ``tools/repro_check.py``), and leaving keys
in declaration order keeps ``credit_id`` first and the mess flags in mess-dial
order, which matters because acceptance gate 4 is a human reading this file.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..common import ids
from .config import GenConfig
from .model import (
    BANK_HEADER,
    PAYMENTS_HEADER,
    REFUNDS_HEADER,
    SETTLEMENT_ITEMS_HEADER,
    SETTLEMENTS_HEADER,
    Story,
)

#: Bumped only when truth.json's shape changes. Phase 2's reader pins it so a
#: schema change fails loudly instead of being mis-parsed.
TRUTH_SCHEMA_VERSION = 1

PAYMENTS_CSV = "payments.csv"
SETTLEMENTS_CSV = "settlements.csv"
SETTLEMENT_ITEMS_CSV = "settlement_items.csv"
BANK_CSV = "bank_statement.csv"
REFUNDS_CSV = "refunds.csv"
TRUTH_JSON = "truth.json"
MANIFEST_JSON = "run_manifest.json"

#: What the matcher is allowed to read.
DATA_FILES: tuple[str, ...] = (
    PAYMENTS_CSV, SETTLEMENTS_CSV, SETTLEMENT_ITEMS_CSV, BANK_CSV, REFUNDS_CSV,
)
#: What only the scorer may read.
TRUTH_FILES: tuple[str, ...] = (TRUTH_JSON, MANIFEST_JSON)

#: Files that must be byte-identical between two runs at the same seed. The
#: manifest is excluded because it records the wall clock, which is inherently
#: non-deterministic; ``repro_check.py`` compares it with ``timing`` stripped.
DETERMINISTIC_FILES: tuple[str, ...] = DATA_FILES + (TRUTH_JSON,)


@dataclass(frozen=True)
class EmitResult:
    data_dir: Path
    truth_dir: Path
    hashes: dict[str, str] = field(default_factory=dict)
    rows_written: dict[str, int] = field(default_factory=dict)

    def path_of(self, name: str) -> Path:
        return (self.truth_dir if name in TRUTH_FILES else self.data_dir) / name


def _sha256(path: Path) -> str:
    """Hash the bytes actually on disk, not the buffer we intended to write."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    """Write one CSV and return its sha256.

    A header-only file is a legitimate outcome: ``refunds.csv`` in clean mode. An
    empty file that exists is better than a missing one -- the loader gets
    exercised from Phase 3 instead of blowing up in Phase 6.
    """
    for i, row in enumerate(rows):
        assert len(row) == len(header), (
            f"{path.name} row {i} has {len(row)} cells, header has {len(header)}"
        )
        assert all(isinstance(cell, str) for cell in row), (
            f"{path.name} row {i} contains a non-string cell; models must "
            f"stringify in csv_row() so the writer never formats anything"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:  # trap 5
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return _sha256(path)


def write_json(path: Path, payload: dict[str, object]) -> str:
    """Write one JSON document with a trailing newline; return its sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return _sha256(path)


def build_truth(story: Story, cfg: GenConfig) -> dict[str, object]:
    """The answer key.

    Every field exists in Phase 1, most of them zero or empty -- that is the point:
    Phases 4 through 8 populate them without a schema migration, and Phase 2's
    reader never has to branch on key presence.
    """
    return {
        "schema_version": TRUTH_SCHEMA_VERSION,
        "seed": cfg.seed,
        "month": cfg.month_label,
        "clean_mode": cfg.clean_mode,
        "flags": cfg.flags.as_dict(),
        "counts": story.counts(),
        "credits": [c.as_truth() for c in story.credits],
        "orphans": {
            "unsettled_payment_ids": list(story.unsettled_payment_ids),
            "settlements_without_credit": list(story.settlements_without_credit),
            "non_gateway_credit_ids": list(story.non_gateway_credit_ids),
        },
    }


def build_manifest(
    cfg: GenConfig,
    story: Story,
    hashes: dict[str, str],
    invariant_report: dict[str, object],
    elapsed_seconds: float,
    generated_at: datetime,
) -> dict[str, object]:
    """Run provenance: config, hashes, invariant results, throughput.

    Lives in ``truth/`` rather than ``data/`` -- which flags are on is a hint about
    what mess is present, harmless in a report header but not something the matcher
    should be able to read. Phase 11 reads it from here.

    Everything non-deterministic is confined to ``timing`` so the rest of the
    document is byte-identical between two runs at the same seed.
    """
    return {
        "schema_version": TRUTH_SCHEMA_VERSION,
        "generator_version": __import__("hisaab").__version__,
        "config": cfg.resolved(),
        "counts": story.counts(),
        # Named, not just counted. A withheld membership is the disk equivalent of a suspended
        # invariant: it removes information the matcher would otherwise read, and the rule
        # this codebase settled after the ``clean_mode`` footgun is that such a removal must be
        # announced rather than left to be inferred from a row count. ``[]`` on every run
        # without ``--settlement-report-late``.
        "membership_withheld": list(story.membership_withheld),
        # The **realised** strata counts for ``--noise-rows``, not the declared split. Gate 14's
        # ``noise_recall`` floor is the plainly-foreign share, and it must be read from what the
        # run actually produced rather than recomputed from ``NOISE_STRATA_SPLIT``: a gate that
        # re-derives the allocation would agree with a broken allocator about a wrong answer.
        # ``{}`` on every run without the flag, like ``membership_withheld``'s ``[]``.
        #
        # Here rather than in ``truth.json`` because it is diagnostic rather than scored -- the
        # scorer keys ``noise_recall`` on ``orphans.non_gateway_credit_ids``, which is a list of
        # ids and needs no stratum. Both files are truth-side, so neither reaches the matcher.
        "noise_strata": {
            stratum: sum(1 for r in story.noise_rows if r.stratum == stratum)
            for stratum in sorted({r.stratum for r in story.noise_rows})
        },
        "totals_paise": {
            "gross": story.total_gross_paise(),
            "net": story.total_net_paise(),
            "credited": story.total_credited_paise(),
        },
        "invariants": {"status": "pass", **invariant_report},
        "file_sha256": dict(sorted(hashes.items())),
        "timing": {
            "generated_at": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_seconds": round(elapsed_seconds, 4),
            "records_per_second": (
                round(len(story.payments) / elapsed_seconds) if elapsed_seconds > 0 else None
            ),
        },
    }


def emit(
    story: Story,
    cfg: GenConfig,
    invariant_report: dict[str, object] | None = None,
    elapsed_seconds: float = 0.0,
) -> EmitResult:
    """Write all seven files. Call only after ``invariants.check_story`` has passed."""
    data_dir, truth_dir = Path(cfg.out_dir), Path(cfg.truth_dir)
    if data_dir.resolve() == truth_dir.resolve():
        raise ValueError(
            f"--out and --truth must be different directories; the data/truth split "
            f"is what proves the answer key did not leak (both were {data_dir})"
        )

    hashes: dict[str, str] = {}
    rows: dict[str, int] = {}

    payments = [p.csv_row() for p in story.payments]
    hashes[PAYMENTS_CSV] = write_csv(data_dir / PAYMENTS_CSV, PAYMENTS_HEADER, payments)
    rows[PAYMENTS_CSV] = len(payments)

    settlements = [s.csv_row() for s in story.settlements]
    hashes[SETTLEMENTS_CSV] = write_csv(
        data_dir / SETTLEMENTS_CSV, SETTLEMENTS_HEADER, settlements
    )
    rows[SETTLEMENTS_CSV] = len(settlements)

    # ``--settlement-report-late`` omits the withheld settlements' rows. **The file is still
    # written, with its header and every row that was not withheld** -- ``load.py`` raises
    # ``LoadError`` on a missing file, so omitting it would fail the run for the wrong reason
    # and read as a loader bug rather than as data the report has not caught up with (#22
    # settled the same question for an empty ``refunds.csv``: header-only, never absent).
    #
    # This is the only place the withholding exists. ``story.settlements`` still lists every
    # member and ``truth.json`` still publishes them, so the answer key stays complete and a
    # searched payment set can still be graded against the real one.
    withheld = set(story.membership_withheld)
    items = sorted(
        row
        for s in story.settlements
        if s.settlement_id not in withheld
        for row in s.item_rows()
    )
    hashes[SETTLEMENT_ITEMS_CSV] = write_csv(
        data_dir / SETTLEMENT_ITEMS_CSV, SETTLEMENT_ITEMS_HEADER, items
    )
    rows[SETTLEMENT_ITEMS_CSV] = len(items)

    # **Gateway credits and ``--noise-rows`` rows go into one file, ordered by the id they were
    # both numbered from.** ``story.build`` assigned those ids from a single counter over the
    # merged, date-and-amount-sorted draft list, so re-sorting on the sequence number here
    # restores exactly that order -- and a noise row sits wherever its date and amount put it
    # rather than in a block at the end. That is the property that keeps the row id from being
    # the answer key; see ``model.NoiseRow`` and the merge comment in ``story.build``.
    #
    # Sorted on the **integer** sequence rather than the id string: ``ids.credit_id`` pads to
    # four digits, so a run large enough to reach C10000 would sort lexically before C9999 and
    # silently reorder the file. The generator's own ceiling makes that unreachable today, which
    # is precisely why a lexical sort would have looked correct indefinitely.
    bank_rows: list[tuple[str, tuple[str, ...]]] = [
        (c.credit_id, c.csv_row()) for c in story.credits
    ] + [(r.row_id, r.csv_row()) for r in story.noise_rows]
    # ``removeprefix``, not ``lstrip``: ``lstrip`` takes a character *set* and strips greedily,
    # so it happens to work for a one-character prefix and would silently eat digits the day
    # ``CREDIT_PREFIX`` gained a second character.
    bank_rows.sort(key=lambda pair: int(pair[0].removeprefix(ids.CREDIT_PREFIX)))
    bank = [row for _id, row in bank_rows]
    hashes[BANK_CSV] = write_csv(data_dir / BANK_CSV, BANK_HEADER, bank)
    rows[BANK_CSV] = len(bank)

    # Header and no rows in clean mode, on purpose.
    refunds = [r.csv_row() for r in story.refunds]
    hashes[REFUNDS_CSV] = write_csv(data_dir / REFUNDS_CSV, REFUNDS_HEADER, refunds)
    rows[REFUNDS_CSV] = len(refunds)

    hashes[TRUTH_JSON] = write_json(truth_dir / TRUTH_JSON, build_truth(story, cfg))
    rows[TRUTH_JSON] = len(story.credits)

    manifest = build_manifest(
        cfg,
        story,
        {k: v for k, v in hashes.items()},
        invariant_report or {},
        elapsed_seconds,
        datetime.now(timezone.utc),
    )
    hashes[MANIFEST_JSON] = write_json(truth_dir / MANIFEST_JSON, manifest)
    rows[MANIFEST_JSON] = 1

    return EmitResult(data_dir=data_dir, truth_dir=truth_dir, hashes=hashes, rows_written=rows)


if __name__ == "__main__":
    import tempfile

    from .invariants import check_story
    from .story import build

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = GenConfig(seed=42, n=12, out_dir=root / "data", truth_dir=root / "truth")
        story = build(cfg)
        report = check_story(story, cfg)
        result = emit(story, cfg, report, elapsed_seconds=0.01)

        # All seven files exist and are hashed.
        for name in DATA_FILES + TRUTH_FILES:
            assert result.path_of(name).exists(), f"{name} was not written"
            assert name in result.hashes, f"{name} was not hashed"
        assert len(result.hashes) == 7

        # Trap 5: no bare or doubled CR anywhere in a CSV.
        for name in DATA_FILES:
            raw = result.path_of(name).read_bytes()
            assert b"\r" not in raw, f"{name} contains CR -- newline handling regressed"

        # refunds.csv is header-only, and the header is present.
        refund_lines = result.path_of(REFUNDS_CSV).read_text("utf-8").splitlines()
        assert refund_lines == [",".join(REFUNDS_HEADER)], refund_lines

        # The bank file is four columns wide and mentions no gateway identifier.
        bank_text = result.path_of(BANK_CSV).read_text("utf-8")
        for line in bank_text.splitlines():
            assert len(line.split(",")) == 4, line
        assert "setl_" not in bank_text and "pay_" not in bank_text

        # truth.json round-trips, keeps reason/note present-but-null, and holds
        # every flag.
        truth = json.loads(result.path_of(TRUTH_JSON).read_text("utf-8"))
        assert truth["schema_version"] == TRUTH_SCHEMA_VERSION
        assert truth["seed"] == 42 and truth["clean_mode"] is True
        assert len(truth["flags"]) == 13 and not any(truth["flags"].values())
        assert truth["counts"]["credits"] == 12
        first = truth["credits"][0]
        assert list(first)[0] == "credit_id", "key order lost; the human gate needs it readable"
        assert first["reason"] is None and "note" in first
        assert first["decomposition"]["expected_credit_paise"] == first["decomposition"]["gross_paise"]
        assert truth["orphans"] == {"unsettled_payment_ids": [],
                                    "settlements_without_credit": [],
                                    "non_gateway_credit_ids": []}

        # The manifest carries hashes for the six deterministic files plus itself.
        manifest = json.loads(result.path_of(MANIFEST_JSON).read_text("utf-8"))
        assert manifest["invariants"]["status"] == "pass"
        assert set(manifest["file_sha256"]) == set(DATA_FILES + (TRUTH_JSON,))
        assert manifest["totals_paise"]["gross"] == manifest["totals_paise"]["credited"]

        # Both JSON files end with exactly one newline.
        for name in TRUTH_FILES:
            assert result.path_of(name).read_bytes().endswith(b"}\n")

        # Re-emitting the same story must reproduce every deterministic file.
        again = emit(story, cfg, report, elapsed_seconds=0.02)
        for name in DETERMINISTIC_FILES:
            assert again.hashes[name] == result.hashes[name], f"{name} is not deterministic"

        # A shared data/truth directory must be refused outright.
        try:
            emit(story, GenConfig(out_dir=root / "same", truth_dir=root / "same"), report)
        except ValueError:
            pass
        else:
            raise AssertionError("emit() allowed --out and --truth to be the same directory")

    print("emit.py self-check ok  (7 files, 6 deterministic)")
