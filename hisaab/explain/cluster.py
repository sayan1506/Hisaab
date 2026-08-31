"""Sub-causes inside one group -- found deterministically, described by the model.

Step 7 asked for clustering within groups. **The clusters are computed without a model**, and
that is a finding rather than a shortcut: the matcher writes each note from a template, so
stripping the ids and figures out of a note leaves the template it came from, and rows sharing
a template share a sub-cause exactly. A model asked to cluster these would be guessing at
something a regex knows for certain -- and its guess would be unverifiable, which is the one
thing this project will not spend on.

Measured on seed 1, n=1000, ``--all-mess --window 1``:

    group                        rows   shapes   split
    UNEXPLAINED_RESIDUAL           47        2   24 / 23
    FX_RATE_GAP                   141        2   126 / 15
    PARTIAL_SETTLEMENT_PENDING     66        1   66

So two of the three largest groups **do** carry real sub-structure and the third carries none.
Both halves matter: a clusterer that found sub-causes in all three would be inventing them,
which is why ``PARTIAL_SETTLEMENT_PENDING`` is the control this module is checked against.

**What the 141-row split exposes in the prompt.** ``FX_RATE_GAP`` divides into 126 rows where
exactly one subset of the captured payments sums to the credit at gross, and 15 where no subset
does -- genuinely different situations for an operator, since the first names a rate to ask for
and the second says the export is short. ``prompt.ROWS_PER_GROUP`` sends the 12 largest rows by
value, so a 15-row minority can be **entirely absent from the sample** while the model writes a
summary about "the group". Sampling across clusters instead is what this module is for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: How much of the skeleton identifies a sub-cause. The templates diverge early -- "and the"
#: versus "but the" at around 110 chars in ``UNEXPLAINED_RESIDUAL`` -- but two templates that
#: agree for 400 characters and differ at the end are still two templates, so this is
#: deliberately generous rather than tuned to the shortest distinguishing prefix.
SKELETON_CHARS = 400

_ID = re.compile(r"\b(?:setl_|pay_|rfnd_)\d+\b")
_PAISE = re.compile(r"\d+p\b")
_NUM = re.compile(r"\d+")


def skeleton(note: str) -> str:
    """A note with every id and figure replaced, leaving the template it was written from.

    Order matters: ids first (they contain digits), then paise figures (they carry a suffix
    that ``_NUM`` would eat), then bare numbers. Reversing the first two turns ``setl_0164``
    into ``setl_<n>`` and loses the distinction between an id and a count.
    """
    text = " ".join((note or "").split())
    text = _ID.sub("<id>", text)
    text = _PAISE.sub("<amt>", text)
    text = _NUM.sub("<n>", text)
    return text[:SKELETON_CHARS]


@dataclass(frozen=True, slots=True)
class Cluster:
    """One sub-cause within a group: the rows sharing a note template."""

    shape: str
    rows: tuple[dict[str, Any], ...]

    @property
    def count(self) -> int:
        return len(self.rows)

    @property
    def value_paise(self) -> int:
        return sum(int(r.get("bank_amount_paise", 0)) for r in self.rows)


def clusters(group: dict[str, Any]) -> tuple[Cluster, ...]:
    """A group's sub-causes, largest first, ordered so two runs agree.

    Ties break on the shape text rather than on insertion order: a dict preserves insertion
    order, but the input order is the bank statement's, so a tie would silently reorder
    between two runs on differently-ordered input. Everything downstream of this feeds a
    prompt, and a reordered prompt is an uncached prompt.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in group.get("credits", ()):
        buckets.setdefault(skeleton(row.get("note", "")), []).append(row)
    return tuple(
        Cluster(shape=shape, rows=tuple(rows))
        for shape, rows in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    )


def sample(group: dict[str, Any], limit: int) -> tuple[dict[str, Any], ...]:
    """Up to ``limit`` rows, guaranteeing every sub-cause is represented if it fits.

    **Why this exists.** Taking the ``limit`` largest rows by value -- the queue's own ranking,
    and what this module replaced -- can omit a whole sub-cause: ``FX_RATE_GAP``'s 15-row
    minority need not contain any of the group's 12 biggest rows, and a model shown only the
    other 126 will describe the group as though the minority were not in it.

    So: one round-robin pass taking each cluster's largest row in turn, then value order within
    what remains. Every cluster is present while ``len(clusters) <= limit``; past that the
    largest clusters win, which is the same value-first principle applied one level up.

    Ordering within the result stays value-descending, so the prompt still reads biggest-first
    and the change is invisible to a reader who does not care about clusters.
    """
    groups = clusters(group)
    if not groups:
        return ()

    chosen: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for cluster in groups:
        ranked = sorted(
            cluster.rows, key=lambda r: int(r.get("bank_amount_paise", 0)), reverse=True
        )
        if len(chosen) < limit:
            chosen.append(ranked[0])
            remaining.extend(ranked[1:])
        else:
            remaining.extend(ranked)

    remaining.sort(key=lambda r: int(r.get("bank_amount_paise", 0)), reverse=True)
    chosen.extend(remaining[: max(0, limit - len(chosen))])
    chosen.sort(key=lambda r: int(r.get("bank_amount_paise", 0)), reverse=True)
    return tuple(chosen)


def describe(group: dict[str, Any], shown: tuple[dict[str, Any], ...]) -> str:
    """The sub-cause census, as a line for the prompt -- or empty when there is one cause.

    Says how many rows each sub-cause holds and how many of them are in the sample, because
    the model is being asked to describe a group it is seeing a fraction of. A group with one
    shape returns "" rather than a line saying "1 sub-cause": a census of one is noise, and it
    would also tell the model to look for structure that is not there.
    """
    found = clusters(group)
    if len(found) < 2:
        return ""
    shown_ids = {r["credit_id"] for r in shown}
    parts = []
    for i, cluster in enumerate(found, 1):
        in_sample = sum(1 for r in cluster.rows if r["credit_id"] in shown_ids)
        parts.append(
            f"  sub-cause {i}: {cluster.count} row(s), {cluster.value_paise}p, "
            f"{in_sample} shown below"
        )
    return (
        f"This group holds {len(found)} distinct sub-causes -- the reconciler's notes for these "
        f"rows were written from {len(found)} different templates, so they are not all the same "
        f"situation. Describe each one:\n" + "\n".join(parts)
    )


def _self_check() -> None:
    """Two shapes must split, one shape must not, and no sub-cause may vanish from a sample."""
    def row(cid: str, amount: int, note: str) -> dict[str, Any]:
        return {"credit_id": cid, "bank_amount_paise": amount, "note": note}

    # The measured FX_RATE_GAP shape: a large majority and a small minority whose rows are all
    # SMALLER than the majority's. This is the case that defeats value-ranked sampling.
    majority = [
        row(f"C{i:04d}", 90000 + i, f"matched setl_{i:04d} on date, one subset of {i} sums")
        for i in range(20)
    ]
    minority = [
        row(f"C1{i:03d}", 100 + i, f"matched setl_1{i:03d} on date, no subset of {i} sums")
        for i in range(3)
    ]
    group = {
        "reason": "FX_RATE_GAP", "cause": "fx", "rows": 23, "value_paise": 1,
        "credits": [*majority, *minority],
    }

    found = clusters(group)
    assert len(found) == 2, f"expected 2 shapes, got {len(found)}: {[c.shape[:40] for c in found]}"
    assert found[0].count == 20 and found[1].count == 3, "clusters must be largest-first"
    assert found[0].value_paise == sum(r["bank_amount_paise"] for r in majority)

    # Determinism: the same group in a different row order must cluster identically.
    shuffled = {**group, "credits": [*minority, *majority]}
    assert [c.shape for c in clusters(shuffled)] == [c.shape for c in found], (
        "clustering depends on input order, so the prompt would differ between two runs on "
        "the same data and never cache"
    )

    # **The assertion this module exists for.** Value-ranked sampling misses the minority
    # entirely; cluster-aware sampling does not.
    by_value = sorted(
        group["credits"], key=lambda r: r["bank_amount_paise"], reverse=True
    )[:12]
    assert not any(r["credit_id"].startswith("C1") for r in by_value), (
        "the value-ranked sample happens to include the minority here, so this self-check "
        "no longer demonstrates the problem -- pick amounts that separate them again"
    )
    picked = sample(group, 12)
    assert len(picked) == 12, f"expected 12 rows, got {len(picked)}"
    assert any(r["credit_id"].startswith("C1") for r in picked), (
        "cluster-aware sampling still dropped the 3-row sub-cause, which is the whole point"
    )
    assert [r["bank_amount_paise"] for r in picked] == sorted(
        (r["bank_amount_paise"] for r in picked), reverse=True
    ), "the sample must still read value-descending"
    assert len({r["credit_id"] for r in picked}) == 12, "the sample repeats a row"

    # The control: one shape must NOT be split, and must produce no census line. Without this,
    # a clusterer that split on anything would look like it worked.
    single = {
        "reason": "PARTIAL_SETTLEMENT_PENDING", "cause": "reserve", "rows": 4, "value_paise": 1,
        "credits": [
            row(f"C{i:04d}", 500 + i,
                f"no settlement pays {900 + i}p exactly, but {i} within 2bd declare a net: "
                f"setl_{i:04d}")
            for i in range(4)
        ],
    }
    assert len(clusters(single)) == 1, (
        f"4 rows from one template split into {len(clusters(single))} clusters -- the skeleton "
        f"is not removing everything row-specific, so every group would look sub-structured"
    )
    assert describe(single, sample(single, 12)) == "", "a single sub-cause needs no census"

    census = describe(group, picked)
    assert "2 distinct sub-causes" in census and "20 row(s)" in census and "3 row(s)" in census
    assert "shown below" in census, "the census must say how many of each are in the sample"

    # A sample smaller than the cluster count keeps the largest clusters rather than crashing.
    tiny = sample(group, 1)
    assert len(tiny) == 1 and tiny[0]["credit_id"] in {r["credit_id"] for r in majority}

    # Skeleton hygiene: ids, paise figures and bare counts must all go, and an id must not be
    # mistaken for a number.
    sk = skeleton("setl_0164 agrees but falls 526p short of 19600p gross of 1 payment(s)")
    assert "<id>" in sk and "0164" not in sk, f"an id survived: {sk}"
    assert "526" not in sk and "19600" not in sk, f"a paise figure survived: {sk}"
    assert sk.count("<amt>") == 2 and "<n>" in sk, f"unexpected skeleton: {sk}"
    assert skeleton("a  b\n c") == "a b c", "whitespace must be normalised"

    print(
        f"cluster: ok -- 2 shapes split and 1 does not, order-independent, and a "
        f"{len(minority)}-row sub-cause survives a {len(majority)}-row majority in a 12-row "
        f"sample (value ranking drops it)"
    )


if __name__ == "__main__":
    _self_check()
