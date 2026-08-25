"""Acceptance — every gate, one command.

    python tools/acceptance.py

Running the gates individually means remembering N commands and, eventually,
forgetting one. This runs the lot and exits non-zero if any fails, so "the phase is
done" is a claim with a check behind it rather than a feeling.

  0. every module's own self-check passes
  1. byte-identical output at a fixed seed, across two processes
  2. a different seed changes the data but not its shape
  3. all invariants pass on three seeds, in memory AND re-read from disk
  4. the leak audit -- structural strategies fail, the arithmetic join succeeds
  5. the answer key is unreachable from the matching path
  6. n=200 stays fast enough to rerun after every change
  7. ASSUMPTIONS.md exists and covers what the write-up has to state
  8. the scorer reports the known answer on four known-answer fixtures  [Phase 2]

**This file grows a gate per phase; it never turns over.** Gates 0-7 are Phase 1's and
still run, because row 1 of the mess dial is the regression check -- "if clean mode is
not 100%, the code is broken" only works if clean mode keeps being measured. Gate 8
arrived with Phase 2's scoring harness.

Gates 1, 2 and 6 live in ``repro_check.py``; gates 4 and part of 3 in
``verify_output.py``; gate 5 in ``check_isolation.py``; gate 8 in ``fixtures.py``. This
module owns gates 0, 3 and 7, and sequences the rest.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hisaab.generator.config import GenConfig  # noqa: E402
from hisaab.generator.emit import emit  # noqa: E402
from hisaab.generator.invariants import InvariantError, check_story  # noqa: E402
from hisaab.generator.story import build  # noqa: E402

#: Modules with a ``__main__`` self-check, in dependency order -- the order they
#: were built in, so the first failure is the deepest one.
SELF_CHECK_MODULES: tuple[str, ...] = (
    "hisaab.common.ids",
    "hisaab.common.reasons",
    "hisaab.common.money",
    "hisaab.generator.rng",
    "hisaab.generator.bizdays",
    "hisaab.generator.config",
    "hisaab.generator.model",
    "hisaab.generator.story",
    "hisaab.generator.invariants",
    "hisaab.generator.emit",
    # --- Phase 2, the scoring harness. Contract first, then the reader that
    # validates it, then the join, then the formatter -- so a failure points at the
    # deepest broken thing rather than at whatever imported it.
    "hisaab.common.verdict",
    "hisaab.scoring.verdict_io",
    "hisaab.scoring.metrics",
    "hisaab.scoring.report",
)

#: Gate 3's seed matrix. Seed 99 is absent on purpose: it is the holdout, and it is
#: not run until Phase 12 (decision #11). Reporting a holdout number we have been
#: quietly checking all along would be the reconciliation equivalent of test-set
#: leakage, and it is detectable when a judge asks how we chose our tolerances.
DEV_SEEDS: tuple[int, ...] = (1, 2, 3)
HOLDOUT_SEED = 99

#: Gate 7: topics the final write-up must state rather than bluff. Checked by
#: keyword so the file cannot pass while silently dropping one.
REQUIRED_ASSUMPTION_TOPICS: dict[str, tuple[str, ...]] = {
    "rounding rule": ("half-up", "rounding"),
    "fee and GST rates": ("basis point", "GST"),
    "settlement cycle": ("T+", "settlement cycle"),
    "Tier 3 tolerances": ("tolerance",),
    "business-day calendar": ("holiday", "weekend"),
    "timezone handling": ("IST", "UTC"),
    "money representation": ("paise",),
    "ID widths": ("pay_0001", "ID width"),
    # --- Phase 2. Both are figures nobody measured, which is exactly why the file
    # has to keep carrying them: "est. human time to clear" is the one number in the
    # submission that is neither derived nor verified.
    "exception effort estimate": ("minutes per exception", "per exception"),
    "by-hand baseline": ("by hand",),
    "match definition": ("set equality", "unit of account"),
}


class GateFailure(Exception):
    """A gate did not pass."""


def _run(argv: list[str], label: str) -> str:
    """Run a subprocess, returning stdout. Raises ``GateFailure`` on non-zero."""
    proc = subprocess.run(
        argv, cwd=ROOT, capture_output=True, text=True,
        env={**_env(), "PYTHONUTF8": "1"},
    )
    if proc.returncode != 0:
        raise GateFailure(
            f"{label} failed (exit {proc.returncode})\n"
            f"{proc.stdout.rstrip()}\n{proc.stderr.rstrip()}"
        )
    return proc.stdout


def _env() -> dict[str, str]:
    import os

    return dict(os.environ)


def gate_0_self_checks() -> None:
    print("gate 0 -- module self-checks")
    for module in SELF_CHECK_MODULES:
        out = _run([sys.executable, "-m", module], f"{module} self-check")
        last = out.strip().splitlines()[-1] if out.strip() else "(no output)"
        print(f"    {module:<34} {last}")


def gate_3_invariants_across_seeds(sizes: tuple[int, ...] = (12, 60, 200)) -> None:
    """Invariants on three seeds, in memory and again on the written files.

    The read-back half matters because the write step itself can corrupt -- column
    order, quoting, encoding, an int that became a string -- and none of that is
    visible to a check that runs on dataclasses.
    """
    print(f"\ngate 3 -- invariants on seeds {list(DEV_SEEDS)} x sizes {list(sizes)}")
    checked = 0
    with tempfile.TemporaryDirectory(prefix="hisaab-accept-") as tmp:
        root = Path(tmp)
        for seed in DEV_SEEDS:
            for n in sizes:
                cfg = GenConfig(
                    seed=seed, n=n,
                    out_dir=root / f"s{seed}n{n}/data",
                    truth_dir=root / f"s{seed}n{n}/truth",
                )
                try:
                    report = check_story(build(cfg), cfg)
                except InvariantError as e:
                    raise GateFailure(f"in-memory invariants failed at seed {seed}, n={n}: {e}")
                story = build(cfg)
                emit(story, cfg, report)
                # Re-read from disk with the independent parser.
                _run(
                    [
                        sys.executable, "tools/verify_output.py",
                        "--data", str(cfg.out_dir), "--truth", str(cfg.truth_dir), "--quiet",
                    ],
                    f"read-back verification at seed {seed}, n={n}",
                )
                checked += 1
                print(
                    f"    seed {seed}, n={n:<4} in-memory ok, read-back ok  "
                    f"({report['date_blocks']} date blocks, "
                    f"numbering fixed {report['numbering_fixed_points']}, "
                    f"within-block {report['within_block_aligned']})"
                )
    print(f"    {checked} configurations pass both passes")

    if HOLDOUT_SEED in DEV_SEEDS:
        raise GateFailure(
            f"seed {HOLDOUT_SEED} is the holdout and must not be in the dev matrix"
        )
    print(f"    seed {HOLDOUT_SEED} deliberately untouched -- holdout for Phase 12")


def gate_4_and_leak_audit() -> None:
    """The honest gate 4: measure what the answer is readable from."""
    print("\ngate 4 -- leak audit on the committed run (data/, truth/)")
    if not (ROOT / "data" / "bank_statement.csv").exists():
        raise GateFailure(
            "no run found at data/ -- generate one first:\n"
            "    python -m hisaab.generator --seed 42 --n 60"
        )
    out = _run([sys.executable, "tools/verify_output.py"], "verify_output")
    for line in out.splitlines():
        stripped = line.strip()
        if stripped and ("/" in stripped or "verified" in stripped) and "proof" not in stripped:
            print(f"    {stripped}")


def gate_5_isolation() -> None:
    print("\ngate 5 -- truth isolation")
    out = _run([sys.executable, "tools/check_isolation.py", "--quiet"], "check_isolation")
    print(f"    the answer key is unreachable from the matching path{out and ''}")


def gates_1_2_6_reproducibility() -> None:
    print("\ngates 1, 2, 6 -- reproducibility, seed sensitivity, throughput")
    out = _run([sys.executable, "tools/repro_check.py"], "repro_check")
    for line in out.splitlines():
        if line.strip():
            print(f"    {line.rstrip()}" if line.startswith("    ") else f"    {line.strip()}")


def gate_8_fixtures() -> None:
    """Phase 2: the scorer reports the known answer on four known-answer fixtures.

    Shells out to ``tools/fixtures.py --check``, which in turn scores each fixture by
    running ``python -m hisaab.scoring``. Two levels of subprocess is deliberate: it
    means the gate exercises the CLI's exit codes and the promise that line 1 of stdout
    is the metric JSON, which is the contract Phase 11 parses. An in-process call would
    leave all of that untested.

    It also keeps this module off ``check_isolation.py``'s truth allowlist, the same way
    gates 1, 2, 4, 5 and 6 do.
    """
    print("\ngate 8 -- the scorer, against four known-answer fixtures")
    out = _run([sys.executable, "tools/fixtures.py", "--check"], "fixtures --check")
    for line in out.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("gate 8"):
            print(f"    {stripped}")


def gate_7_assumptions() -> None:
    """ASSUMPTIONS.md must exist and cover every topic the write-up has to state."""
    print("\ngate 7 -- ASSUMPTIONS.md")
    path = ROOT / "ASSUMPTIONS.md"
    if not path.exists():
        raise GateFailure(
            "ASSUMPTIONS.md is missing. Every number in it is something a judge can "
            "ask about, and 'stated, not bluffed' is a submission requirement."
        )
    text = path.read_text(encoding="utf-8")
    missing = [
        topic
        for topic, keywords in REQUIRED_ASSUMPTION_TOPICS.items()
        if not any(k.lower() in text.lower() for k in keywords)
    ]
    if missing:
        raise GateFailure(
            f"ASSUMPTIONS.md does not cover: {missing}. These are the topics a judge "
            f"asks about; a missing one is a number we would end up bluffing."
        )
    print(f"    {len(REQUIRED_ASSUMPTION_TOPICS)} required topics all covered")
    print(f"    {len(text.splitlines())} lines at {path.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run every acceptance gate (Phases 1-2).")
    p.add_argument("--skip-slow", action="store_true",
                   help="skip the n=200 sweeps in gates 3 and 6")
    args = p.parse_args(argv)

    print("Acceptance -- generator (clean mode) + scoring harness\n" + "=" * 62)
    gates = [
        gate_0_self_checks,
        lambda: gate_3_invariants_across_seeds((12, 60) if args.skip_slow else (12, 60, 200)),
        gate_4_and_leak_audit,
        gate_5_isolation,
        gates_1_2_6_reproducibility,
        gate_7_assumptions,
        gate_8_fixtures,
    ]
    try:
        for gate in gates:
            gate()
    except GateFailure as e:
        print(f"\n{'=' * 62}\nACCEPTANCE FAILED\n\n{e}", file=sys.stderr)
        return 1

    print("\n" + "=" * 62)
    print("all eight gates pass -- Phases 1 and 2 are complete")
    print("\nPhase 3 can start. It gets a contract (hisaab/common/verdict.py), a target")
    print("(the oracle scores 100% coverage with 0 wrong matches on this exact data, so a")
    print("shortfall is the matcher's fault and not the data's), and a feedback loop in")
    print("seconds. First move: run the stub through the scorer, confirm 0%, make it climb.")
    print("\n    python tools/fixtures.py --fixture stub --out out/matches.json")
    print("    python -m hisaab.scoring --matches out/matches.json --truth truth/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
