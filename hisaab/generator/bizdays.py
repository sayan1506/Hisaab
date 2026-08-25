"""Business-day calendar.

Phase 1 needs only "is this a weekday" (payments are captured on business days),
but the interface is the real one from the start, with an injectable holiday set,
because Phase 4's ``--settlement-delay`` computes T+n over it and Phase 3's
blocking window counts business days. A stub with the wrong shape here means
rewriting call sites in Phase 4.

Phase 1 ships an **empty holiday set**: weekends only. That is a stated
assumption in ASSUMPTIONS.md, not an oversight — a real Indian bank calendar has
gazetted holidays that vary by state, and inventing a list would be bluffing a
number we did not verify.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta

SATURDAY = 5
SUNDAY = 6


class BusinessCalendar:
    """Weekday calendar with an optional holiday set."""

    def __init__(self, holidays: frozenset[date] | None = None) -> None:
        self.holidays: frozenset[date] = frozenset(holidays or ())

    def is_business_day(self, d: date) -> bool:
        return d.weekday() < SATURDAY and d not in self.holidays

    def next_business_day(self, d: date) -> date:
        """The first business day strictly after ``d``."""
        out = d + timedelta(days=1)
        while not self.is_business_day(out):
            out += timedelta(days=1)
        return out

    def add_business_days(self, d: date, n: int) -> date:
        """``d`` plus ``n`` business days.

        ``n == 0`` rolls forward to the next business day if ``d`` is not one --
        which is the T+0 settlement semantics a gateway actually uses: money
        captured on Saturday settles on Monday, not on Saturday.
        """
        assert n >= 0, f"add_business_days is forward-only in this system: {n}"
        out = d
        while not self.is_business_day(out):
            out += timedelta(days=1)
        for _ in range(n):
            out = self.next_business_day(out)
        return out

    def business_days_between(self, start: date, end: date) -> int:
        """Count of business days in the half-open interval ``[start, end)``.

        Negative if ``end`` precedes ``start``. Phase 3's date window uses this.
        """
        if end < start:
            return -self.business_days_between(end, start)
        count, cur = 0, start
        while cur < end:
            if self.is_business_day(cur):
                count += 1
            cur += timedelta(days=1)
        return count

    def business_days_in_month(self, year: int, month: int) -> list[date]:
        last = calendar.monthrange(year, month)[1]
        return [
            d
            for day in range(1, last + 1)
            if self.is_business_day(d := date(year, month, day))
        ]


if __name__ == "__main__":
    cal = BusinessCalendar()
    mon = date(2026, 8, 10)   # Monday
    sat = date(2026, 8, 15)   # Saturday
    assert cal.is_business_day(mon)
    assert not cal.is_business_day(sat)
    assert not cal.is_business_day(date(2026, 8, 16))          # Sunday
    assert cal.next_business_day(date(2026, 8, 14)) == date(2026, 8, 17)  # Fri -> Mon
    assert cal.add_business_days(mon, 2) == date(2026, 8, 12)  # T+2 midweek
    assert cal.add_business_days(date(2026, 8, 13), 2) == date(2026, 8, 17)  # over a weekend
    assert cal.add_business_days(sat, 0) == date(2026, 8, 17)  # T+0 rolls off a weekend
    assert cal.business_days_between(mon, date(2026, 8, 12)) == 2
    assert cal.business_days_between(date(2026, 8, 12), mon) == -2
    assert cal.business_days_between(mon, mon) == 0
    aug = cal.business_days_in_month(2026, 8)
    assert len(aug) == 21, len(aug)
    assert aug[0] == date(2026, 8, 3) and aug[-1] == date(2026, 8, 31)
    # An injected holiday must be respected by every method.
    holiday = date(2026, 8, 11)
    h = BusinessCalendar(frozenset({holiday}))
    assert not h.is_business_day(holiday)
    assert h.add_business_days(mon, 1) == date(2026, 8, 12)
    assert holiday not in h.business_days_in_month(2026, 8)
    print("bizdays.py self-check ok")
