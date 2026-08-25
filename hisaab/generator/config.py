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

PAYMENT_METHODS: dict[str, int] = {"card": 45, "upi": 40, "netbanking": 10, "wallet": 5}

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
    dup_amounts: bool = False              # Phase 8  planted unresolvable
    fx: bool = False                       # Phase 8  rate moves between capture/settle
    rounding_edge: bool = False            # Phase 8  fee x GST on a half-paisa
    settlement_report_late: bool = False   # Phase 8  withhold settlement_items.csv
    utr_patchy: bool = False               # Phase 8  UTR missing/truncated on some rows

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

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

    **Unused in Phase 1** -- clean mode has zero fees, not "fees applied
    consistently". It exists now so the rates are a declared assumption rather
    than a number invented under pressure in Phase 4.

    These are ASSUMPTIONS, flagged as such in ASSUMPTIONS.md and in the write-up.
    Verify against Razorpay's current published pricing before relying on them;
    real rates vary by method and by plan.
    """

    fee_bps_by_method: dict[str, int] = field(
        default_factory=lambda: {"card": 200, "upi": 0, "netbanking": 190, "wallet": 200}
    )
    gst_bps: int = 1800   # 18% GST, charged on the fee, not on the gross
    tds_bps: int = 100    # 1% -- only applies where --tds is on

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

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"--n must be at least 1, got {self.n}")
        if not 1 <= self.month <= 12:
            raise ValueError(f"--month has an invalid month: {self.month}")
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
    assert not GenConfig(flags=MessFlags.all_on()).clean_mode
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
