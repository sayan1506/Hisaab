"""The **only** reader of ``truth.json`` in this codebase.

Structural rule 1 from section 12 of the track spec: ``truth.json`` feeds the
scoring script and nothing else. Nothing on the matching path may read it, ever.
That rule is enforced three ways:

  1. The file is written to a separate directory from the CSVs.
  2. It is read only through this module, which lives in ``hisaab.scoring`` --
     a package the matcher does not import.
  3. ``tools/check_isolation.py`` fails the build if anything on the matching path
     imports this module or names the truth file.

Phase 2's scoring harness builds on the readers here. Phase 1's job is to make the
schema stable enough to write that harness against, and to pin the version so a
later change fails loudly instead of being silently mis-parsed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: Must match ``hisaab.generator.emit.TRUTH_SCHEMA_VERSION``. Deliberately
#: duplicated rather than imported: the scorer must not depend on the generator,
#: so that a schema drift between them surfaces as a loud version mismatch
#: instead of being hidden by a shared constant.
#:
#: **v2 (Phase 8 step 2b):** the decomposition block gains ``fx_paise``. Bumped here in the same
#: change that bumps the generator's copy -- the duplication above only pays off if both move
#: together, and a scorer left at v1 would reject every regenerated run with the version error
#: below, which is the loud failure the duplication is designed to produce.
SUPPORTED_SCHEMA_VERSION = 2

TRUTH_JSON = "truth.json"
MANIFEST_JSON = "run_manifest.json"


class TruthError(Exception):
    """``truth.json`` is missing, malformed, or a version this scorer cannot read."""


@dataclass(frozen=True, slots=True)
class TruthDecomposition:
    """The expected balance for one credit, per the answer key."""

    gross_paise: int
    fee_paise: int
    gst_paise: int
    tds_paise: int
    refunds_paise: int
    reserve_paise: int
    expected_credit_paise: int
    #: Phase 8 (``--fx``): the rate movement between capture and settlement, in paise. The one
    #: **signed** term, and the one that is **added** -- design (b) leaves ``payments.csv``'s
    #: gross stale at the capture rate while the payout is right at the settlement rate.
    #:
    #: Last in declaration order, and with a default, so this stays a widening rather than a
    #: rewrite: every positional ``TruthDecomposition(value, 0, 0, 0, 0, 0, value)`` in the
    #: scorer's own fixtures keeps meaning what it meant. The generator's copy carries the same
    #: field; ``common/verdict.Decomposition`` deliberately does **not** -- a term the matcher
    #: could populate is a term it could fit any residual into. That asymmetry is why
    #: ``metrics.DECOMPOSITION_TERMS`` does not list this one.
    fx_paise: int = 0

    def closes(self) -> bool:
        return self.expected_credit_paise == (
            self.gross_paise + self.fx_paise - self.fee_paise - self.gst_paise
            - self.tds_paise - self.refunds_paise - self.reserve_paise
        )


@dataclass(frozen=True, slots=True)
class TruthCredit:
    """What actually produced one bank credit."""

    credit_id: str
    settlement_ids: tuple[str, ...]
    payment_ids: tuple[str, ...]
    refunds_netted: tuple[str, ...]
    reserve_held_paise: int
    decomposition: TruthDecomposition
    resolvable: bool
    reason: str | None
    note: str | None

    @property
    def is_planted_unresolvable(self) -> bool:
        """True for cases the generator planted as unresolvable from the inputs.

        Phase 8's correct-abstention count is built on this: abstaining here is
        the right answer, not a limitation to apologise for.
        """
        return not self.resolvable


@dataclass(frozen=True, slots=True)
class Truth:
    """The parsed answer key."""

    schema_version: int
    seed: int
    month: str
    clean_mode: bool
    flags: dict[str, bool]
    counts: dict[str, int]
    credits: tuple[TruthCredit, ...]
    unsettled_payment_ids: tuple[str, ...]
    settlements_without_credit: tuple[str, ...]
    non_gateway_credit_ids: tuple[str, ...]

    def credit(self, credit_id: str) -> TruthCredit:
        try:
            return self._by_id[credit_id]
        except KeyError:
            raise TruthError(f"no such credit in truth: {credit_id!r}") from None

    @property
    def _by_id(self) -> dict[str, TruthCredit]:
        return {c.credit_id: c for c in self.credits}

    @property
    def resolvable_credits(self) -> tuple[TruthCredit, ...]:
        return tuple(c for c in self.credits if c.resolvable)

    @property
    def planted_unresolvable(self) -> tuple[TruthCredit, ...]:
        return tuple(c for c in self.credits if not c.resolvable)

    @property
    def flags_enabled(self) -> tuple[str, ...]:
        return tuple(name for name, on in self.flags.items() if on)


def _require_int(obj: dict[str, object], key: str, where: str) -> int:
    v = obj.get(key)
    if isinstance(v, bool) or not isinstance(v, int):
        raise TruthError(f"{where}: {key} must be an int, got {type(v).__name__} ({v!r})")
    return v


def _require_str_list(obj: dict[str, object], key: str, where: str) -> tuple[str, ...]:
    v = obj.get(key)
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise TruthError(f"{where}: {key} must be a list of strings, got {v!r}")
    return tuple(v)


def load_truth(truth_dir: Path | str) -> Truth:
    """Parse ``truth.json`` from ``truth_dir``. Raises ``TruthError`` on any drift.

    Validation is strict on purpose: this is the reference the match rate is
    measured against, so a malformed answer key must stop the run rather than
    quietly produce a plausible-looking score.
    """
    path = Path(truth_dir)
    if path.is_dir():
        path = path / TRUTH_JSON
    if not path.exists():
        raise TruthError(f"truth file not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise TruthError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise TruthError(f"{path}: expected a JSON object at the top level")

    version = _require_int(raw, "schema_version", str(path))
    if version != SUPPORTED_SCHEMA_VERSION:
        raise TruthError(
            f"{path}: truth schema v{version}, but this scorer reads "
            f"v{SUPPORTED_SCHEMA_VERSION}. Regenerate the data or update "
            f"hisaab/scoring/truth_io.py -- do not guess at the difference."
        )

    for key in ("seed", "month", "clean_mode", "flags", "counts", "credits", "orphans"):
        if key not in raw:
            raise TruthError(f"{path}: missing required key {key!r}")

    flags = raw["flags"]
    if not isinstance(flags, dict) or not all(isinstance(v, bool) for v in flags.values()):
        raise TruthError(f"{path}: flags must be a flat object of booleans")

    counts = raw["counts"]
    if not isinstance(counts, dict):
        raise TruthError(f"{path}: counts must be an object")

    credits: list[TruthCredit] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw["credits"]):
        where = f"{path}: credits[{i}]"
        if not isinstance(entry, dict):
            raise TruthError(f"{where}: expected an object")
        credit_id = entry.get("credit_id")
        if not isinstance(credit_id, str) or not credit_id:
            raise TruthError(f"{where}: credit_id must be a non-empty string")
        if credit_id in seen:
            raise TruthError(f"{where}: duplicate credit_id {credit_id!r}")
        seen.add(credit_id)

        d = entry.get("decomposition")
        if not isinstance(d, dict):
            raise TruthError(f"{where}: decomposition must be an object")
        dec = TruthDecomposition(
            gross_paise=_require_int(d, "gross_paise", where),
            fee_paise=_require_int(d, "fee_paise", where),
            gst_paise=_require_int(d, "gst_paise", where),
            tds_paise=_require_int(d, "tds_paise", where),
            refunds_paise=_require_int(d, "refunds_paise", where),
            reserve_paise=_require_int(d, "reserve_paise", where),
            expected_credit_paise=_require_int(d, "expected_credit_paise", where),
            # **Required, not ``.get``-with-default**, even though the dataclass defaults it.
            # This file refuses absent-versus-null branches by policy (see the ``reason``/``note``
            # comment below): a reader that tolerates a missing key grows a branch per field, and
            # one of those branches is eventually wrong. A v2 answer key always writes this term,
            # at 0 without ``--fx``, so a missing one means a v1 document that the version pin
            # above has already rejected -- or a generator that forgot the term, which is exactly
            # what should fail loudly here rather than score as agreement.
            fx_paise=_require_int(d, "fx_paise", where),
        )
        if not dec.closes():
            raise TruthError(f"{where}: decomposition does not close to expected_credit_paise")

        resolvable = entry.get("resolvable")
        if not isinstance(resolvable, bool):
            raise TruthError(f"{where}: resolvable must be a bool")
        # reason/note must be *present*, even as null -- Phase 1 relies on this so
        # Phase 8 adds planted unresolvables without a schema migration, and so
        # this reader never has to branch on key presence.
        for key in ("reason", "note"):
            if key not in entry:
                raise TruthError(f"{where}: {key!r} must be present (null is fine)")
            if entry[key] is not None and not isinstance(entry[key], str):
                raise TruthError(f"{where}: {key} must be a string or null")
        if not resolvable and not entry["reason"]:
            raise TruthError(
                f"{where}: an unresolvable credit must carry a reason code -- "
                f"an unexplained unresolvable is a silent drop in the answer key"
            )

        credits.append(
            TruthCredit(
                credit_id=credit_id,
                settlement_ids=_require_str_list(entry, "settlement_ids", where),
                payment_ids=_require_str_list(entry, "payment_ids", where),
                refunds_netted=_require_str_list(entry, "refunds_netted", where),
                reserve_held_paise=_require_int(entry, "reserve_held_paise", where),
                decomposition=dec,
                resolvable=resolvable,
                reason=entry["reason"],
                note=entry["note"],
            )
        )

    declared = counts.get("credits")
    if isinstance(declared, int) and declared != len(credits):
        raise TruthError(
            f"{path}: counts.credits is {declared} but {len(credits)} credit entries exist"
        )

    orphans = raw["orphans"]
    if not isinstance(orphans, dict):
        raise TruthError(f"{path}: orphans must be an object")

    return Truth(
        schema_version=version,
        seed=_require_int(raw, "seed", str(path)),
        month=str(raw["month"]),
        clean_mode=bool(raw["clean_mode"]),
        flags=dict(flags),
        counts={k: int(v) for k, v in counts.items()},
        credits=tuple(credits),
        unsettled_payment_ids=_require_str_list(orphans, "unsettled_payment_ids", str(path)),
        settlements_without_credit=_require_str_list(
            orphans, "settlements_without_credit", str(path)
        ),
        non_gateway_credit_ids=_require_str_list(orphans, "non_gateway_credit_ids", str(path)),
    )


def load_manifest(truth_dir: Path | str) -> dict[str, object]:
    """Parse ``run_manifest.json`` -- run provenance, hashes and throughput.

    Phase 11's report header reads this. It lives in ``truth/`` because which mess
    flags are on is a hint about what mess is present.
    """
    path = Path(truth_dir)
    if path.is_dir():
        path = path / MANIFEST_JSON
    if not path.exists():
        raise TruthError(f"manifest not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise TruthError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise TruthError(f"{path}: expected a JSON object")
    return raw


if __name__ == "__main__":
    import sys

    truth_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "truth")
    t = load_truth(truth_dir)
    print(
        f"truth v{t.schema_version}: seed {t.seed}, {t.month}, "
        f"{'clean' if t.clean_mode else 'mess[' + ','.join(t.flags_enabled) + ']'}"
    )
    print(f"  credits {len(t.credits)}  resolvable {len(t.resolvable_credits)}  "
          f"planted-unresolvable {len(t.planted_unresolvable)}")
    print(f"  counts {t.counts}")
    total = sum(c.decomposition.expected_credit_paise for c in t.credits)
    print(f"  expected credited total {total} paise")
    m = load_manifest(truth_dir)
    timing = m.get("timing", {})
    print(f"  manifest: {len(m.get('file_sha256', {}))} hashes, "
          f"{timing.get('elapsed_seconds')}s, {timing.get('records_per_second')} rec/s")
