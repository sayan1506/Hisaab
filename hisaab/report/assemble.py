"""Reads the (up to) five documents a run produces, and refuses to render across two runs.

**Why this reads JSON directly rather than importing the packages that wrote it.**
``hisaab.explain`` is a leaf (``tools/check_isolation.py``'s ``EXPLAIN_IMPORT_READERS`` is
empty) -- nothing shipped under ``hisaab/`` may import it, this package included, so the
explain artifact has to be read off disk. Doing the same for ``matches.json``, the scoring
``--out`` document and the triage ``--out`` document keeps the four documents on equal
footing rather than reading three of them through a validating loader and the fourth by
hand: this package renders what a run wrote, not what a Python object says about itself.

It is also the cheaper way to stay off ``tools/check_isolation.py``'s ``TRUTH_READERS``.
Importing ``hisaab.scoring`` -- even ``verdict_io``, which never opens ``truth.json`` --
counts as reaching truth under that check's own rule (a module that can import the package
can import the loader), so pulling in ``hisaab.scoring.verdict_io.load_verdicts`` for a
sturdier parse would cost this package a deliberate addition to that allowlist for a report
that has no business anywhere near the answer key. ``hisaab.triage`` is not on
``TRUTH_READERS`` and carries no such tax, but its readers (``triage/read.py``) validate
against the verdict *contract*, which is one degree stricter than a renderer needs: a
document that already reconciled once, when the matcher or scorer wrote it, does not need
its balance re-proved a second time to be displayed. So this module owns its own narrow
read, the same choice ``triage/read.py``'s docstring explains and defends: shared *logic*
would move to ``hisaab/common``, but a *schema* is duplicated on purpose, with a comment
saying why, so drift fails loudly instead of hiding behind a shared symbol.

**What is checked, and what is not.** Every document must carry its declared top-level
keys and a schema version this module knows. Cross-run provenance is checked the way
``hisaab.scoring.verdict_io.reconcile`` checks it: seed and month must agree between
``matches.json`` and the metrics document, because scoring one run's verdicts against
another run's numbers produces a plausible-looking report rather than an error. The triage
document carries no seed or month at all (``hisaab/triage/cli.py``'s ``as_json()`` never
reads truth, so it has none to state) -- its only provenance field is the matches path it
was pointed at, so that is what gets compared instead. The explain artifact carries neither
seed, month, nor a matches path (see its own schema in ``hisaab/explain/cli.py``); its only
link to the run is the ``cause`` string each explanation carries, and that join is exercised
at render time, not here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common.verdict import MATCHES_JSON, VERDICT_SCHEMA_VERSION

#: Duplicated literals, not imports -- the same choice ``triage/read.py`` makes for the
#: verdict contract, for the reason its own docstring gives: the constant this module would
#: import (``hisaab.scoring.metrics.METRICS_SCHEMA_VERSION``) lives inside the one package
#: ``tools/check_isolation.py`` treats as reaching truth by import alone. A schema is safe to
#: duplicate; the read path that could reach the answer key is not.
#:
#: ``hisaab.triage.cli.TRIAGE_SCHEMA_VERSION`` carries no such tax -- triage is not on
#: ``TRUTH_READERS`` -- so it is imported rather than copied, one line below.
METRICS_SCHEMA_VERSION = 5
EXPLAIN_SCHEMA_VERSION = 1

from ..triage.cli import TRIAGE_SCHEMA_VERSION  # noqa: E402  (after the constants it sits beside)

REQUIRED_MATCHES_KEYS: tuple[str, ...] = ("schema_version", "seed", "month", "matcher", "verdicts")
REQUIRED_METRICS_KEYS: tuple[str, ...] = (
    "schema_version", "run", "timing", "totals", "cells", "rates",
    "exceptions", "dismissals", "decomposition", "risk",
)
REQUIRED_TRIAGE_KEYS: tuple[str, ...] = ("schema_version", "inputs", "totals", "groups")
REQUIRED_EXPLAIN_KEYS: tuple[str, ...] = (
    "schema_version", "model", "endpoint", "groups", "explained",
    "citations_clean", "usage_total", "explanations",
)
#: The Q&A artifact carries no schema version -- ``Answer.as_json()`` (``hisaab/explain/qa.py``)
#: was added to reuse the dataclass's own fields, not as a document with its own versioning
#: story. Required keys are checked instead, and there is no version to refuse on.
REQUIRED_QA_KEYS: tuple[str, ...] = (
    "credit_id", "question", "answer", "cited_row_ids", "cited_amounts_paise",
    "arithmetic", "ok", "findings", "usage",
)


class ReportError(Exception):
    """One of the four documents is missing, malformed, or from a different run."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise ReportError(f"{label} not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ReportError(f"{label} ({path}) is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ReportError(f"{label} ({path}): expected a JSON object at the top level")
    return raw


def _require_keys(doc: dict[str, Any], required: tuple[str, ...], label: str, path: Path) -> None:
    missing = [k for k in required if k not in doc]
    if missing:
        raise ReportError(f"{label} ({path}): missing key(s) {missing}")


def _require_version(doc: dict[str, Any], expected: int, label: str, path: Path) -> None:
    version = doc.get("schema_version")
    if version != expected:
        raise ReportError(
            f"{label} ({path}): schema v{version!r}, but this report reads v{expected}. "
            f"Re-run the tool that wrote it, or update this module -- do not guess at the "
            f"difference."
        )


def load_matches(path: Path | str) -> dict[str, Any]:
    """``matches.json`` (a file, or a directory holding one)."""
    p = Path(path)
    if p.is_dir():
        p = p / MATCHES_JSON
    doc = _load_json(p, "matches.json")
    _require_keys(doc, REQUIRED_MATCHES_KEYS, "matches.json", p)
    _require_version(doc, VERDICT_SCHEMA_VERSION, "matches.json", p)
    return doc


def load_metrics(path: Path | str) -> dict[str, Any]:
    """The scoring ``--out`` document."""
    p = Path(path)
    doc = _load_json(p, "the scoring document")
    _require_keys(doc, REQUIRED_METRICS_KEYS, "the scoring document", p)
    _require_version(doc, METRICS_SCHEMA_VERSION, "the scoring document", p)
    return doc


def load_triage(path: Path | str) -> dict[str, Any]:
    """The triage ``--out`` document."""
    p = Path(path)
    doc = _load_json(p, "the triage document")
    _require_keys(doc, REQUIRED_TRIAGE_KEYS, "the triage document", p)
    _require_version(doc, TRIAGE_SCHEMA_VERSION, "the triage document", p)
    return doc


def load_explain(path: Path | str) -> dict[str, Any]:
    """The explain artifact. Caller decides whether it exists at all -- see ``assemble``."""
    p = Path(path)
    doc = _load_json(p, "the explain artifact")
    _require_keys(doc, REQUIRED_EXPLAIN_KEYS, "the explain artifact", p)
    _require_version(doc, EXPLAIN_SCHEMA_VERSION, "the explain artifact", p)
    return doc


def load_qa(path: Path | str) -> dict[str, Any]:
    """The Q&A artifact (``Answer.as_json()``, via ``hisaab.explain --ask ... --out``).

    No schema version to check -- see ``REQUIRED_QA_KEYS``'s comment. Caller decides whether
    it exists at all, the same as ``load_explain``.
    """
    p = Path(path)
    doc = _load_json(p, "the Q&A artifact")
    _require_keys(doc, REQUIRED_QA_KEYS, "the Q&A artifact", p)
    return doc


@dataclass(frozen=True, slots=True)
class ReportInput:
    """The five documents one report renders, already refused if they disagree.

    ``explain`` and ``qa`` are ``None`` exactly when no artifact was given -- checked by
    ``Path.exists()`` in ``assemble``, not by catching a missing-file error. A run with
    neither is a legitimate state (Phase 10's layer is optional, and ``--ask`` is a one-shot
    a caller may never have run), not a degraded one.
    """

    matches: dict[str, Any]
    metrics: dict[str, Any]
    triage: dict[str, Any]
    explain: dict[str, Any] | None
    qa: dict[str, Any] | None = None

    @property
    def seed(self) -> int:
        return self.matches["seed"]

    @property
    def month(self) -> str:
        return self.matches["month"]

    @property
    def matcher(self) -> str:
        return self.matches["matcher"]


def assemble(
    matches_path: Path | str,
    metrics_path: Path | str,
    triage_path: Path | str,
    explain_path: Path | str | None,
    qa_path: Path | str | None = None,
) -> ReportInput:
    """Load all five documents and refuse unless they describe one run.

    ``matches_path``, ``metrics_path`` and ``triage_path`` are required -- a report cannot
    render a metric block or an exception queue at all without them. ``explain_path`` and
    ``qa_path`` may each be ``None`` or point at a file that does not exist; either way the
    corresponding field comes back ``None`` rather than raising, per plan correction (3) --
    extended here to the Q&A artifact for the same reason: a caller who never ran ``--ask``
    has produced a complete report, not an incomplete one.

    The Q&A artifact carries no seed, month or matches path to check provenance against (see
    ``REQUIRED_QA_KEYS``'s comment) -- its only link to the run is the ``credit_id`` it names,
    and that is a fact about one row, not about which run it came from, so there is nothing
    here for this function to refuse on. A caller pointing ``--qa`` at a stale artifact from a
    different run gets a report that renders it anyway; that is a caller error this function
    has no evidence to detect.
    """
    matches = load_matches(matches_path)
    metrics = load_metrics(metrics_path)
    triage = load_triage(triage_path)

    # --- provenance, checked the way verdict_io.reconcile checks it -----------------
    # Scoring one run's verdicts against another run's numbers is the failure that produces
    # a plausible report instead of an error, so it is refused here exactly as it would be
    # refused before the numbers were ever computed.
    m_seed, m_month = metrics["run"]["seed"], metrics["run"]["month"]
    if (m_seed, m_month) != (matches["seed"], matches["month"]):
        raise ReportError(
            f"matches.json describes seed {matches['seed']}, {matches['month']}, but the "
            f"scoring document describes seed {m_seed}, {m_month}. Rendering one run's "
            f"verdicts beside another run's score would produce a plausible report about a "
            f"run that never happened."
        )

    # The triage document carries no seed or month -- its only provenance field is the
    # matches path it was pointed at. Paths are compared resolved, since a caller may spell
    # the same file two ways (a bare filename vs. a directory), and only when the stated
    # path actually exists to resolve -- a stale or relocated path is not evidence of a
    # mismatched run, only of a moved file, which is not this check's job to flag.
    stated = Path(triage["inputs"]["matches"])
    if stated.is_dir():
        stated = stated / MATCHES_JSON
    given = Path(matches_path)
    if given.is_dir():
        given = given / MATCHES_JSON
    if stated.exists() and given.exists() and stated.resolve() != given.resolve():
        raise ReportError(
            f"the triage document was built from {stated}, but this report was pointed at "
            f"{given} for matches.json. Two different matcher runs cannot be rendered as one."
        )

    explain: dict[str, Any] | None = None
    if explain_path is not None and Path(explain_path).exists():
        explain = load_explain(explain_path)

    qa: dict[str, Any] | None = None
    if qa_path is not None and Path(qa_path).exists():
        qa = load_qa(qa_path)

    return ReportInput(matches=matches, metrics=metrics, triage=triage, explain=explain, qa=qa)


if __name__ == "__main__":
    import tempfile

    def refuses(fn, label: str, expect_in: str) -> None:
        try:
            fn()
        except ReportError as e:
            assert expect_in in str(e), f"{label}: message lacks {expect_in!r}\n  got: {e}"
            return
        raise AssertionError(f"accepted {label}")

    MATCHES_DOC = {
        "schema_version": VERDICT_SCHEMA_VERSION, "seed": 42, "month": "2026-08",
        "matcher": "fixture:selfcheck@1", "timing": {"wall_clock_seconds": 0.02},
        "verdicts": [],
    }
    METRICS_DOC = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "run": {"seed": 42, "month": "2026-08", "clean_mode": True, "flags": [], "matcher": "x"},
        "timing": {"wall_clock_seconds": 0.01}, "totals": {}, "cells": {}, "rates": {},
        "exceptions": {}, "dismissals": {}, "decomposition": {}, "risk": {},
    }
    EXPLAIN_DOC = {
        "schema_version": EXPLAIN_SCHEMA_VERSION, "model": "m", "endpoint": "e",
        "groups": 0, "explained": 0, "citations_clean": 0,
        "usage_total": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0},
        "explanations": [],
    }
    QA_DOC = {
        "credit_id": "C0001", "question": "why is this less than the gross?",
        "answer": "fees and taxes were withheld", "cited_row_ids": ["C0001"],
        "cited_amounts_paise": [20679], "arithmetic": None, "ok": True, "findings": [],
        "usage": {"input_tokens": 100, "output_tokens": 20,
                  "cache_read_input_tokens": 50, "cache_creation_input_tokens": 0},
    }

    with tempfile.TemporaryDirectory(prefix="hisaab-report-assemble-") as tmp:
        root = Path(tmp)

        def write(name: str, doc: dict[str, Any]) -> Path:
            p = root / name
            p.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
            return p

        matches_path = write("matches.json", MATCHES_DOC)
        metrics_path = write("metrics.json", METRICS_DOC)
        triage_path = write(
            "triage.json",
            {
                "schema_version": TRIAGE_SCHEMA_VERSION,
                "inputs": {"matches": str(matches_path), "data": str(root / "data")},
                "totals": {}, "groups": [],
            },
        )
        explain_path = write("explain.json", EXPLAIN_DOC)
        qa_path = write("qa.json", QA_DOC)

        # --- the positive control: all five present and agreeing -------------------
        ri = assemble(matches_path, metrics_path, triage_path, explain_path, qa_path)
        assert ri.seed == 42 and ri.month == "2026-08" and ri.matcher == "fixture:selfcheck@1"
        assert ri.explain == EXPLAIN_DOC
        assert ri.qa == QA_DOC

        # --- explain and qa are optional-by-path-existence, not an error -----------
        no_explain = assemble(matches_path, metrics_path, triage_path, None)
        assert no_explain.explain is None and no_explain.qa is None
        missing_explain = assemble(matches_path, metrics_path, triage_path, root / "nope.json")
        assert missing_explain.explain is None
        missing_qa = assemble(matches_path, metrics_path, triage_path, None, root / "nope.json")
        assert missing_qa.qa is None
        with_qa_no_explain = assemble(matches_path, metrics_path, triage_path, None, qa_path)
        assert with_qa_no_explain.explain is None and with_qa_no_explain.qa == QA_DOC

        # --- the three required documents refuse when missing ---------------------
        refuses(
            lambda: assemble(root / "nope.json", metrics_path, triage_path, None),
            "a missing matches.json", "not found",
        )
        refuses(
            lambda: assemble(matches_path, root / "nope.json", triage_path, None),
            "a missing scoring document", "not found",
        )
        refuses(
            lambda: assemble(matches_path, metrics_path, root / "nope.json", None),
            "a missing triage document", "not found",
        )

        # --- schema version and required-key refusals, one per document -----------
        bad_version = write("bad_version.json", {**MATCHES_DOC, "schema_version": 999})
        refuses(
            lambda: assemble(bad_version, metrics_path, triage_path, None),
            "an unknown matches schema version", "v999",
        )
        no_key = write("no_key.json", {k: v for k, v in MATCHES_DOC.items() if k != "seed"})
        refuses(
            lambda: assemble(no_key, metrics_path, triage_path, None),
            "matches.json with no seed", "seed",
        )

        # --- the provenance refusal: matches and metrics disagree ------------------
        wrong_seed_metrics = write(
            "wrong_seed.json",
            {**METRICS_DOC, "run": {**METRICS_DOC["run"], "seed": 43}},
        )
        refuses(
            lambda: assemble(matches_path, wrong_seed_metrics, triage_path, None),
            "metrics from a different seed", "seed 43",
        )
        wrong_month_metrics = write(
            "wrong_month.json",
            {**METRICS_DOC, "run": {**METRICS_DOC["run"], "month": "2026-09"}},
        )
        refuses(
            lambda: assemble(matches_path, wrong_month_metrics, triage_path, None),
            "metrics from a different month", "2026-09",
        )

        # --- the provenance refusal: triage was built from a different matches file -
        other_matches = write("other_matches.json", {**MATCHES_DOC, "seed": 43})
        skewed_triage = write(
            "skewed_triage.json",
            {
                "schema_version": TRIAGE_SCHEMA_VERSION,
                "inputs": {"matches": str(other_matches), "data": str(root / "data")},
                "totals": {}, "groups": [],
            },
        )
        refuses(
            lambda: assemble(matches_path, metrics_path, skewed_triage, None),
            "triage built from a different matches file", "Two different matcher runs",
        )

        # --- a directory works for matches.json, same as verdict_io -----------------
        subdir = root / "sub"
        subdir.mkdir()
        (subdir / MATCHES_JSON).write_text(json.dumps(MATCHES_DOC), encoding="utf-8")
        triage_for_subdir = write(
            "triage_subdir.json",
            {
                "schema_version": TRIAGE_SCHEMA_VERSION,
                "inputs": {"matches": str(subdir), "data": str(root / "data")},
                "totals": {}, "groups": [],
            },
        )
        ri2 = assemble(subdir, metrics_path, triage_for_subdir, None)
        assert ri2.seed == 42

    print("report/assemble.py self-check ok")
