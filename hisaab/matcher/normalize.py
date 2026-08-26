"""Bank narration -> ``{channel, counterparty, ref_tail}``, raw string always kept.

Four templates are in play in clean mode, and the reference tail appears four ways::

    NEFT-RAZORPAYSOFT-XXXX4471             -> NEFT, RAZORPAYSOFT,      4471
    IMPS CR/RAZORPAY SOFTWARE/4451         -> IMPS, RAZORPAY SOFTWARE, 4451
    RTGS/RAZORPAYSOFT/XXXX3093/SETTLEMENT  -> RTGS, RAZORPAYSOFT,      3093
    NEFT-RZRPAY-3532                       -> NEFT, RZRPAY,            3532

Three channels, three counterparty spellings, and a tail that is sometimes prefixed
with ``XXXX`` masking. Real statements vary exactly this much between banks and
channels, which is why the generator emits four styles rather than one.

**Nothing on the match path consumes ``ref_tail`` in Phase 3, deliberately.** The tail
is a second, independent 100% join on this data -- 60 distinct tails for 60
settlements -- and a matcher that keys on it scores a perfect number while the amount
arithmetic is never exercised, then *stays* perfect through Phase 4 with no fee model
ever written. So the tail is recorded in the verdict's ``note`` as corroboration and
nothing branches on it. The mechanical guard is in ``tools/acceptance.py`` gate 9:
blanking every narration to a constant must change no verdict.

**This module never raises.** A narration is free text from an external system, and
Phase 8's ``--utr-patchy`` truncates or removes the tail on purpose. A parser that
threw on a degraded string would turn "cannot corroborate this row" into "cannot
process this file", and the matcher would fail on data it is supposed to survive.
Every field is therefore ``str | None``, and an unparseable narration yields a
``Narration`` whose ``raw`` is intact and whose parts are ``None``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..common.ids import digit_runs

#: Bank channels seen in this data. Matched on a word boundary so ``NEFT`` inside a
#: counterparty name cannot be mistaken for the channel field.
CHANNELS: tuple[str, ...] = ("NEFT", "IMPS", "RTGS")

#: Counterparty spellings, **longest first**. Order is load-bearing: matched the other
#: way round, ``RAZORPAY SOFTWARE`` would report as ``RAZORPAYSOFT`` on the prefix.
COUNTERPARTY_SPELLINGS: tuple[str, ...] = (
    "RAZORPAY SOFTWARE",
    "RAZORPAYSOFT",
    "RZRPAY",
)

#: All three spellings mean one entity. Phase 10 explains rows and Phase 11 groups
#: them, and neither should have to know that the bank writes the name three ways.
CANONICAL_COUNTERPARTY = "RAZORPAYSOFT"

CHANNEL_RE = re.compile(r"\b(" + "|".join(CHANNELS) + r")\b")

#: The masking prefix on a tail, as in ``XXXX4471``. Stripped by ``digit_runs``
#: naturally, since ``X`` is not a digit -- this constant exists for the docstring
#: above and for Phase 11, which shows the raw form.
MASK_PREFIX = "XXXX"


@dataclass(frozen=True, slots=True)
class Narration:
    """One parsed bank narration. ``raw`` is always present and never altered.

    ``raw`` is kept because Phase 11 displays it and Phase 10 is graded on parsing
    it -- an LLM citation that quotes a normalised string cannot be checked against
    the source, which defeats the point of the citation.
    """

    raw: str
    channel: str | None = None
    counterparty: str | None = None
    ref_tail: str | None = None

    @property
    def is_gateway_counterparty(self) -> bool:
        """Does this row name the gateway at all?

        Not consumed in Phase 3 -- clean mode has no non-gateway rows, so there is
        nothing to tell apart. Phase 7's ``--noise-rows`` is what makes this useful,
        and it is a property rather than a stored field so it cannot fall out of sync
        with ``counterparty``.
        """
        return self.counterparty is not None

    @property
    def canonical_counterparty(self) -> str | None:
        """The one entity behind three spellings, or ``None`` if none was found."""
        return CANONICAL_COUNTERPARTY if self.counterparty is not None else None

    @property
    def is_fully_parsed(self) -> bool:
        """All three parts recovered. True for every row in clean mode."""
        return None not in (self.channel, self.counterparty, self.ref_tail)

    def evidence(self) -> str:
        """A short corroboration string for a verdict's ``note``.

        Corroboration only: this text records what the narration *said*, and no
        matching decision reads it. See the module docstring.
        """
        parts = [
            f"channel={self.channel or '?'}",
            f"counterparty={self.counterparty or '?'}",
            f"ref_tail={self.ref_tail or '?'}",
        ]
        return " ".join(parts)


def parse(text: str) -> Narration:
    """Parse one narration into its parts. Never raises; missing parts are ``None``."""
    raw = text if isinstance(text, str) else ""

    channel_match = CHANNEL_RE.search(raw)
    channel = channel_match.group(1) if channel_match else None

    counterparty = next((s for s in COUNTERPARTY_SPELLINGS if s in raw), None)

    # The reference tail is the *last* digit run: every template puts it last, and
    # taking the last run rather than the first is what keeps a channel or a masked
    # prefix from being read as the tail. Whole-run matching (not substring) is the
    # same discipline invariant I7 uses -- see hisaab/common/ids.py.
    runs = digit_runs(raw)
    ref_tail = runs[-1] if runs else None

    return Narration(raw=raw, channel=channel, counterparty=counterparty, ref_tail=ref_tail)


if __name__ == "__main__":
    from pathlib import Path

    # --- the four templates, across channels and spellings -----------------
    cases: tuple[tuple[str, str | None, str | None, str | None], ...] = (
        ("NEFT-RAZORPAYSOFT-XXXX4471", "NEFT", "RAZORPAYSOFT", "4471"),
        ("IMPS CR/RAZORPAY SOFTWARE/4451", "IMPS", "RAZORPAY SOFTWARE", "4451"),
        ("RTGS/RAZORPAYSOFT/XXXX3093/SETTLEMENT", "RTGS", "RAZORPAYSOFT", "3093"),
        ("NEFT-RZRPAY-3532", "NEFT", "RZRPAY", "3532"),
        # The same four styles with the other channels, since the generator mixes them.
        ("RTGS CR/RAZORPAY SOFTWARE/5745", "RTGS", "RAZORPAY SOFTWARE", "5745"),
        ("IMPS/RAZORPAYSOFT/XXXX1933/SETTLEMENT", "IMPS", "RAZORPAYSOFT", "1933"),
        ("IMPS-RZRPAY-8104", "IMPS", "RZRPAY", "8104"),
    )
    for raw, channel, counterparty, tail in cases:
        got = parse(raw)
        assert got.raw == raw, "raw must survive verbatim"
        assert got.channel == channel, (raw, got.channel)
        assert got.counterparty == counterparty, (raw, got.counterparty)
        assert got.ref_tail == tail, (raw, got.ref_tail)
        assert got.is_fully_parsed and got.is_gateway_counterparty
        assert got.canonical_counterparty == CANONICAL_COUNTERPARTY

    # The spelling order matters: the spaced form must not report as the prefix.
    assert parse("NEFT CR/RAZORPAY SOFTWARE/1234").counterparty == "RAZORPAY SOFTWARE"

    # --- degradation must not raise (Phase 8's --utr-patchy) ---------------
    patchy = parse("NEFT-RAZORPAYSOFT-XXXX")
    assert patchy.ref_tail is None and patchy.channel == "NEFT"
    assert not patchy.is_fully_parsed
    assert patchy.is_gateway_counterparty, "a missing tail says nothing about the payer"
    truncated = parse("NEFT-RAZORPAYSOFT-XX93")
    assert truncated.ref_tail == "93", "a truncated tail is reported, not discarded"

    # A row that is not gateway income at all (Phase 7's shape).
    noise = parse("IMPS-JOHNDOE-DIRECTTRANSFER")
    assert noise.channel == "IMPS" and noise.counterparty is None
    assert not noise.is_gateway_counterparty and noise.ref_tail is None

    # Unparseable, empty, and non-string inputs all yield an intact object.
    for bad in ("", "   ", "!!!", "SOME UNRELATED CREDIT"):
        got = parse(bad)
        assert got.raw == bad and not got.is_fully_parsed
    assert parse(None).raw == ""  # type: ignore[arg-type]

    # The blanked-narration case gate 9 relies on: parses to nothing, raises nothing.
    blanked = parse("X")
    assert blanked.channel is None and blanked.ref_tail is None
    assert "?" in blanked.evidence()

    # --- the committed run: every narration must parse completely ----------
    bank = Path(__file__).resolve().parent.parent.parent / "data" / "bank_statement.csv"
    if bank.exists():
        import csv

        with bank.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        parsed = [parse(r["narration"]) for r in rows]
        full = sum(1 for p in parsed if p.is_fully_parsed)
        assert full == len(rows), f"only {full}/{len(rows)} narrations parsed completely"
        # The tail is a second independent join on this data -- which is exactly why
        # nothing on the match path is allowed to read it.
        tails = {p.ref_tail for p in parsed}
        assert len(tails) == len(rows), (
            f"{len(tails)} distinct tails for {len(rows)} rows"
        )
        channels = {p.channel for p in parsed}
        assert channels <= set(CHANNELS), channels
        print(
            f"normalize.py self-check ok  ({len(rows)} committed narrations, "
            f"{len(tails)} distinct tails, channels {sorted(channels)})"
        )
    else:
        print("normalize.py self-check ok  (no committed data/ to cross-read)")
