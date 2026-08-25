"""Seeded randomness, one substream per concern.

Trap 4 (see .plan/phase1.md): with a single shared ``Random``, turning on
``--fees`` in Phase 4 consumes extra draws and shifts every subsequent value, so
the clean-mode payment amounts silently change too. Phase-to-phase diffs become
unreadable and "same seed, same numbers" stops being true across flag changes.

Fix: name every concern and give it its own stream derived from the master seed.
Flipping a flag then perturbs only the streams that flag touches.

Two rules that keep this honest:

  1. Never call module-level ``random.*`` anywhere in the codebase. A single stray
     call makes the run irreproducible, and "run with --seed 42 and you get my
     numbers" is a claim you do not want to walk back.
  2. Each stream should draw a **fixed number of values per record**. A variable
     draw count reintroduces trap 4 inside a single stream. Anything conditional
     gets its own stream — which is why the reserved names below exist now.

``random.Random`` seeded with a ``str`` hashes it with SHA-512 internally, so it
is stable across runs, processes, platforms and Python versions. Do NOT use the
builtin ``hash()`` to derive seeds: ``PYTHONHASHSEED`` randomises it per process
and the reproducibility claim evaporates.
"""

from __future__ import annotations

import random

#: Streams used in Phase 1.
PHASE1_STREAMS = (
    "amounts",
    "timestamps",
    "methods",
    "utr",
    "narration",
    "settlement_order",  # shuffle before setl_NNNN assignment (trap 1)
    "bank_order",        # sort tiebreak before CNNNN assignment (trap 2)
)

#: Reserved for later phases. Listed now so they are not invented mid-phase, and
#: so nobody reuses a Phase 1 stream for new randomness (which would perturb
#: existing data and break the phase-to-phase diff).
RESERVED_STREAMS = (
    "delay",      # Phase 4  --settlement-delay
    "batching",   # Phase 5  --batching
    "refunds",    # Phase 6  --netted-refunds
    "reserve",    # Phase 6  --reserve
    "tds",        # Phase 6  --tds
    "noise",      # Phase 7  --noise-rows
    "unsettled",  # Phase 7  --unsettled
    "dup",        # Phase 8  --dup-amounts
    "fx",         # Phase 8  --fx
    "utr_patchy",  # Phase 8  --utr-patchy
)

KNOWN_STREAMS = frozenset(PHASE1_STREAMS + RESERVED_STREAMS)


def substream(master_seed: int, name: str) -> random.Random:
    """An independent, reproducible ``Random`` for one named concern."""
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise TypeError(f"seed must be int, got {type(master_seed).__name__}")
    if name not in KNOWN_STREAMS:
        raise ValueError(
            f"unknown rng stream {name!r}; add it to PHASE1_STREAMS or "
            f"RESERVED_STREAMS in hisaab/generator/rng.py so the set stays auditable"
        )
    return random.Random(f"{master_seed}:{name}")


def weighted_choice(rng: random.Random, choices: dict[str, int]) -> str:
    """Pick one key from ``{value: weight}`` using exactly one draw.

    Uses one ``randrange`` call rather than ``random.choices`` so the draw count
    per record is fixed and obvious (see rule 2 above), and iterates a sorted
    key order so the result never depends on dict insertion order.
    """
    assert choices, "weighted_choice needs at least one option"
    items = sorted(choices.items())
    total = sum(w for _, w in items)
    assert total > 0, f"weights must sum to a positive number: {choices}"
    ticket = rng.randrange(total)
    upto = 0
    for value, weight in items:
        upto += weight
        if ticket < upto:
            return value
    raise AssertionError("unreachable: weights did not cover the ticket")


if __name__ == "__main__":
    # Determinism: same seed and name -> same sequence.
    a = [substream(42, "amounts").random() for _ in range(3)]
    b = [substream(42, "amounts").random() for _ in range(3)]
    assert a == b
    # Independence: different names -> different sequences.
    assert substream(42, "amounts").random() != substream(42, "methods").random()
    # Different seeds -> different sequences.
    assert substream(42, "amounts").random() != substream(43, "amounts").random()
    # Pin the derivation itself. If anyone changes how a stream seed is built --
    # a different separator, or switching to the builtin hash() -- these fail
    # loudly instead of silently producing a different dataset at the same seed.
    pinned = {
        "amounts": 0.46347584108949025,
        "timestamps": 0.03898529203055623,
        "methods": 0.6352407122084595,
    }
    for name, expected in pinned.items():
        got = substream(42, name).random()
        assert got == expected, f"stream {name!r} derivation changed: {got!r}"
    # A typo'd stream name must fail loudly, not silently create a new stream.
    try:
        substream(42, "amonuts")
    except ValueError:
        pass
    else:
        raise AssertionError("substream() accepted an unknown stream name")
    # weighted_choice: a zero-weight option is never selected; one draw only.
    r = substream(1, "methods")
    picks = {weighted_choice(r, {"card": 5, "upi": 5, "never": 0}) for _ in range(200)}
    assert "never" not in picks and picks == {"card", "upi"}, picks
    print("rng.py self-check ok")
