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
  9. the matcher scores 100/100/0 across seeds and sizes, without the UTR shortcut  [Phase 3]
 10. --fees and --settlement-delay: correctness holds, the arithmetic is proved per row
     against truth term by term, and the window is shown to be load-bearing  [Phase 4]
 11. planted unresolvable pairs abstain, and the cheap UTR-tail attack is re-run on
     every planted row and required to come back ambiguous  [Phase 4b]
 12. both tiers carry rows, so a Tier 1 regression cannot hide behind a Tier 2
     success -- and the subset search's bound refuses rather than guessing  [Phase 5]
 13. all seven of Phase 6's flags at once: the three new deduction terms are non-zero
     and re-derived, and the reserve is diagnosed but never resolved  [Phase 6]

**This file grows a gate per phase; it never turns over.** Gates 0-7 are Phase 1's and
still run, because row 1 of the mess dial is the regression check -- "if clean mode is
not 100%, the code is broken" only works if clean mode keeps being measured. Gate 8
arrived with Phase 2's scoring harness, gate 9 with Phase 3's matcher, gate 10 with
Phase 4's fee model and date wedge, gate 11 with Phase 4b's plants, gate 12 with
Phase 5's batching and subset search, gate 13 with Phase 6's three new terms.

**Gate 13 is the argument for that no-turnover rule, in one gate.** It is the first
thing here to run ``--netted-refunds`` alongside ``--batching`` and
``--settlement-report-late``, and it found three defects sitting in already-green code
-- including two wrong matches per run at n=1000, the one number this project says never
moves. Every earlier gate passed throughout. A suite that had replaced its old gates
with new ones would have had no way to notice, because the failures live in the
*interaction* between flags rather than in any single one.

Note what gate 10 does **not** assert: a flat 100% coverage. The plan asked for it, and
measurement said the plan was wrong -- one credit on one seed is genuinely ambiguous, and
abstaining there is the correct answer. Coverage may fall only onto an honest abstention
(``ABSTENTION_REASONS``); correctness and the wrong-match count never bend. That asymmetry
is the project's central claim expressed as a test.

Gates 1, 2 and 6 live in ``repro_check.py``; gates 4 and part of 3 in
``verify_output.py``; gate 5 in ``check_isolation.py``; gate 8 in ``fixtures.py``. This
module owns gates 0, 3 and 7, and sequences the rest.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import io
import json
import re
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
# Gate 9 asks the matcher what counts as a *decision* rather than re-listing the fields,
# so the gate and the matcher cannot drift about whether ``note`` is one. (This module
# is a test harness and imports both halves by design; it is not on the matching path,
# which is why check_isolation.py scans hisaab/matcher/ and not tools/.)
from hisaab.matcher.engine import DECISION_FIELDS  # noqa: E402
from hisaab.matcher.tier2 import MAX_POOL as TIER2_MAX_POOL  # noqa: E402

#: Modules with a ``__main__`` self-check, in dependency order -- the order they
#: were built in, so the first failure is the deepest one.
SELF_CHECK_MODULES: tuple[str, ...] = (
    "hisaab.common.ids",
    "hisaab.common.reasons",
    # Phase 9 moved the effort table here from ``scoring/metrics.py`` so ``hisaab/triage``
    # could read it without importing the scoring package. Listed straight after
    # ``reasons`` -- the only thing it depends on -- and well ahead of ``metrics``, which
    # now imports it: a broken price table should be reported as a broken price table.
    "hisaab.common.effort",
    "hisaab.common.money",
    "hisaab.common.bizdays",
    "hisaab.generator.rng",
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
    # --- Phase 3, the matcher. Dependency order again: the loader, then the parser
    # that reads what it loaded, then candidate generation, then the resolution that
    # counts candidates, then the engine that sequences the lot.
    "hisaab.matcher.load",
    "hisaab.matcher.normalize",
    "hisaab.matcher.blocking",
    # ``fees`` (Phase 4) and ``tier2`` (Phase 5) were **absent from this list until Phase
    # 6**, and both have had real self-checks the whole time -- the fee arithmetic and the
    # subset search, two of the least forgiving modules here, verified only when someone
    # ran them by hand. Nothing failed as a result, which is the uncomfortable part: gate 0
    # printed a clean sweep while skipping them, so the omission was invisible in exactly
    # the output that exists to make omissions visible. Both are leaves, so they sit ahead
    # of ``tier1``, which imports them: a broken fee model should be reported as a broken
    # fee model and not as whatever failed downstream of it.
    "hisaab.matcher.fees",
    "hisaab.matcher.tier2",
    "hisaab.matcher.tier1",
    "hisaab.matcher.engine",
    # --- Phase 6. Last on purpose, and not because it is newest: this one is off the
    # resolution path by design (``check_isolation.py`` check 7), so a failure here means
    # the declared-vs-derived *report* is wrong, never that a verdict is.
    "hisaab.matcher.adjustments",
    # --- Phase 9, triage. Last because it reads what everything above produces, and it
    # reads it with the least access of any package here: ``hisaab/triage`` is in
    # ``MATCHER_PACKAGES``, so it may not import the scoring package, the generator, or
    # the adjustment report. Its reader is a deliberate narrow duplicate of
    # ``verdict_io``'s parse half -- see that module's docstring for why the alternative
    # was rejected -- so this self-check is what keeps the two from drifting silently.
    "hisaab.triage.read",
    "hisaab.triage.group",
    "hisaab.triage.value",
    "hisaab.triage.hint",
    # --- Phase 10, the LLM layer. Listed for the reason the ``fees``/``tier2`` note above
    # records: four new modules with working self-checks that this list skipped would repeat
    # exactly that omission, and it would be invisible in the output that exists to make
    # omissions visible.
    #
    # **These four must import with ``anthropic`` absent**, because it is an optional extra
    # and gate 0 is the gate that proves the core does not need it. That is what
    # ``client.py``'s in-function import buys, and it is verified rather than assumed: the
    # self-checks drive ``explain_group`` with a recorded double, so no path here constructs
    # a real client or touches a network.
    #
    # ``hisaab.explain.cli`` is deliberately absent, matching ``hisaab.triage.cli``: a CLI
    # module's ``__main__`` runs the CLI, so listing it here would have gate 0 attempt a
    # model call. Dependency order again -- schema and prompt are leaves, verify reads the
    # prompt's universe, client sends what both build.
    "hisaab.explain.schema",
    # ``cluster`` before ``prompt`` because ``prompt`` imports it: the sub-cause split decides
    # which rows a request sends, so a broken clusterer should be reported as a broken
    # clusterer rather than as whatever the prompt did with its output.
    "hisaab.explain.cluster",
    "hisaab.explain.prompt",
    "hisaab.explain.verify",
    "hisaab.explain.client",
    # ``qa`` last: it imports ``client`` for ``Usage``/``_client`` and reuses ``prompt``'s id
    # and paise regexes, so a break in either should be reported there, not here. This entry
    # was the omission gate 17's own module-count check exists to catch -- see that check's
    # comment on ``cluster.py`` slipping past a hardcoded tuple the same way.
    "hisaab.explain.qa",
    # --- Phase 11, the report. ``assemble`` first: it defines ``ReportInput`` and reads
    # nothing else in this package. The five section renderers after it (``header``,
    # ``metric_block``, ``exceptions``, ``matched``, ``qa``) are leaves with respect to each
    # other -- none imports another -- so they are listed in the order Step 3-7 built them
    # in, not by any dependency among themselves. ``html`` last: it is the only module here
    # that imports the other six, so a broken section should be reported as that section
    # breaking, not as ``html`` breaking.
    "hisaab.report.assemble",
    "hisaab.report.header",
    # ``metric_block`` is the one entry in this block that also appears in
    # ``tools/check_isolation.py``'s ``TRUTH_READERS`` -- see that module's docstring for
    # why reconstructing ``roi()``'s easy/hard split needs the real, unmodified
    # ``hisaab.scoring.report.metric_block()`` rather than a re-derivation from JSON alone.
    "hisaab.report.metric_block",
    "hisaab.report.exceptions",
    "hisaab.report.matched",
    "hisaab.report.qa",
    "hisaab.report.html",
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


#: Gate 5's self-check battery for **check 8** (Phase 10) -- the rule that nothing which
#: ships can reach a network or a model. Each entry is
#: ``(label, path to plant, file body, expect_refused, marker the message must carry)``.
#:
#: The marker is the point. A mutant that trips *some other* check looks identical from
#: outside to one check 8 caught, and proves nothing about check 8 -- so each case names
#: the phrase its owning half raises, and a refusal without that phrase fails the gate as
#: loudly as no refusal at all.
_CHECK_8_MUTANTS: tuple[tuple[str, str, str, bool, str], ...] = (
    (
        "urllib under hisaab/matcher/",
        "hisaab/matcher/_m.py",
        "import urllib.request\n",
        True,
        "can reach a network or a model",
    ),
    (
        "the anthropic SDK under hisaab/matcher/",
        "hisaab/matcher/_m.py",
        "import anthropic\n",
        True,
        "can reach a network or a model",
    ),
    (
        "urllib imported inside a function body",
        "hisaab/matcher/_m.py",
        "def go():\n    from urllib import request\n    return request\n",
        True,
        "can reach a network or a model",
    ),
    (
        "urllib under hisaab/common/ -- outside MATCHER_PACKAGES, imported by it",
        "hisaab/common/_m.py",
        "import urllib.request\n",
        True,
        "can reach a network or a model",
    ),
    (
        "relative `from ..explain import ask` inside the matcher",
        "hisaab/matcher/_m.py",
        "from ..explain import ask\n",
        True,
        "imports the model layer",
    ),
    (
        "absolute `import hisaab.explain.client` inside triage",
        "hisaab/triage/_m.py",
        "import hisaab.explain.client\n",
        True,
        "imports the model layer",
    ),
    (
        "subprocess inside hisaab/explain/ -- the network-exempt tree",
        "hisaab/explain/_m.py",
        "import subprocess\n",
        True,
        "can reach a network or a model",
    ),
    # --- the two controls. Without these the gate cannot tell a working check from one
    #     that refuses everything, and "refuses everything" is the cheapest way to make a
    #     mutation battery look green.
    (
        "CONTROL: urllib inside hisaab/explain/ -- the exemption must work",
        "hisaab/explain/_m.py",
        "import urllib.request\n",
        False,
        "",
    ),
    (
        "CONTROL: hisaab/explain/ importing its own siblings -- must not self-trip 8b",
        "hisaab/explain/_m.py",
        "from hisaab.explain import client\nimport hisaab.explain.prompt\n",
        False,
        "",
    ),
)


def _gate_5_self_check() -> int:
    """Attack check 8 before trusting it, and return the number of cases verified.

    Check 8 went green on its first run across all 42 files under ``hisaab/``, which is
    exactly the state in which a new default-deny rule is most likely to be silently
    vacuous -- it is also what a rule that matched *nothing at all* would print. The
    claim it backs ("the matching engine is deliberately not AI") is this project's
    headline design commitment, so it gets the same treatment as gates 12-15: plant the
    violation, require the check to refuse it, and require the refusal to come from the
    assertion that owns it.

    **Synthetic tree, patched ``ROOT``, restored in a finally.** ``check_isolation``
    resolves every path from its module-global ``ROOT``, so pointing that at a temp
    directory makes the tool scan a four-file fake package instead of this repo. Planting
    a network import into real source and restoring it afterwards would work until the one
    run that dies between the two, and leaving ``import anthropic`` in ``matcher/`` is the
    single worst artefact this build could leave behind. The patch is asserted not to leak.

    The split mirrors ``_gate_12_self_check``: the synthetic cases prove the check *can*
    fail, and the real run below proves this tree *does* pass. Neither claim substitutes
    for the other -- a check that only ever ran on passing input reports its own silence.
    """
    import importlib

    ci = importlib.import_module("check_isolation")
    real_root = ci.ROOT

    # The four benign files the mutants are planted alongside. Two of them matter beyond
    # scaffolding: hisaab/common/ is outside MATCHER_PACKAGES but imported by the matcher,
    # and hisaab/explain/ is the one tree with an exemption.
    scaffold = {
        "hisaab/matcher/engine.py": "from hisaab.common import money\n",
        "hisaab/common/money.py": "PAISE = 100\n",
        "hisaab/triage/group.py": "from hisaab.common import money\n",
        "hisaab/explain/client.py": "import json\n",
    }

    verified = 0
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for rel, body in scaffold.items():
                target = tmp / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(body, encoding="utf-8")
            (tmp / "tools").mkdir()
            ci.ROOT = tmp

            # The vacuity guard, first: if the synthetic tree fails for any unrelated
            # reason, every "caught" verdict below is noise rather than evidence.
            try:
                ci.check(verbose=False)
            except ci.IsolationError as e:
                raise GateFailure(
                    f"gate 5's self-check baseline failed on its own synthetic tree "
                    f"({e}). Every mutant result after this would be attributable to "
                    f"the scaffold rather than to the planted violation."
                ) from e

            planted = tmp / "hisaab" / "matcher" / "_m.py"
            for label, rel, body, expect_refused, marker in _CHECK_8_MUTANTS:
                target = tmp / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(body, encoding="utf-8")
                try:
                    ci.check(verbose=False)
                    refused, message = False, ""
                except ci.IsolationError as e:
                    refused, message = True, str(e)
                finally:
                    target.unlink()
                    if planted.exists():
                        planted.unlink()

                if expect_refused and not refused:
                    raise GateFailure(
                        f"check 8 did NOT refuse a planted violation: {label}.\n"
                        f"  Planted {rel} containing {body.strip()!r}, and "
                        f"check_isolation passed. The rule that the matching engine "
                        f"cannot reach a model is unenforced for this case."
                    )
                if expect_refused and marker not in message:
                    raise GateFailure(
                        f"check 8 refused {label}, but not by its own assertion.\n"
                        f"  expected the message to carry {marker!r}\n"
                        f"  got: {message.splitlines()[0] if message else '(empty)'}\n"
                        f"A mutant caught by some other check looks identical to one "
                        f"check 8 caught, and proves nothing about check 8."
                    )
                if not expect_refused and refused:
                    raise GateFailure(
                        f"check 8 raised a FALSE POSITIVE on {label}.\n"
                        f"  {message.splitlines()[0] if message else ''}\n"
                        f"  hisaab/explain is exempt from the network half by design -- "
                        f"calling the model is its entire job. A check that refuses this "
                        f"forbids the feature it was written to make safe."
                    )
                verified += 1
    finally:
        ci.ROOT = real_root

    if ci.ROOT != real_root:  # pragma: no cover -- the patch must not survive
        raise GateFailure("gate 5's self-check leaked its patched ROOT")
    return verified


def gate_5_isolation() -> None:
    print("\ngate 5 -- truth isolation, and the matching engine cannot reach a model")
    verified = _gate_5_self_check()
    print(f"    check 8 self-check: {verified} planted cases behave")
    print(
        "      7 violations refused, each naming its own assertion; 2 controls pass "
        "(hisaab/explain may call out, and may import itself)"
    )
    out = _run([sys.executable, "tools/check_isolation.py", "--quiet"], "check_isolation")
    print(f"    the answer key is unreachable from the matching path{out and ''}")
    print(
        "    nothing under hisaab/ imports an HTTP client, a model SDK, subprocess, "
        "importlib or ctypes"
    )


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


def _matcher_and_score(
    data: Path, truth: Path, out: Path, seed: int, month: str = "2026-08",
    extra: list[str] | None = None,
) -> dict[str, object]:
    """Run the matcher CLI, then the scorer CLI. Returns the scorer's metric JSON.

    Two subprocesses rather than two function calls, for the reason gate 8 shells out:
    it exercises the exit codes and the promise that **line 1 of stdout is JSON** on
    both CLIs. That line is the contract Phase 11 parses, and an in-process call would
    leave it untested.
    """
    _run(
        [
            sys.executable, "-m", "hisaab.matcher",
            "--data", str(data), "--out", str(out),
            "--seed", str(seed), "--month", month, "--quiet",
            *(extra or []),
        ],
        f"matcher at seed {seed}",
    )
    stdout = _run(
        [
            sys.executable, "-m", "hisaab.scoring",
            "--matches", str(out), "--truth", str(truth), "--quiet",
        ],
        f"scorer at seed {seed}",
    )
    first = stdout.splitlines()[0] if stdout.strip() else ""
    try:
        return json.loads(first)
    except json.JSONDecodeError as e:
        raise GateFailure(f"line 1 of the scorer's stdout is not JSON: {first!r} ({e})")


def _decisions(matches: Path) -> list[tuple[object, ...]]:
    """Every verdict's decision fields, in file order -- ``note`` deliberately excluded.

    ``note`` carries the UTR corroboration, so blanking the narrations legitimately
    changes it. Excluding it here is what makes the blanked-narration comparison a test
    of *decisions* rather than of prose. ``DECISION_FIELDS`` is imported from the engine
    so the gate and the matcher cannot drift about what a decision is.
    """
    doc = json.loads(matches.read_text(encoding="utf-8"))
    return [tuple(v[f] for f in DECISION_FIELDS) for v in doc["verdicts"]]


def _without_timing(matches: Path) -> str:
    """The verdict document minus ``timing``, canonically serialised.

    **Why not a byte comparison.** ``.plan/phase3.md`` acceptance item 5 asks for a
    byte-identical ``matches.json`` across two runs, while step 5 requires a *measured*
    ``wall_clock_seconds`` inside it. Both cannot hold, and the codebase already settled
    which one gives: ``hisaab/common/verdict.py`` confines all non-determinism to
    ``timing`` and says to compare the document without it, exactly as
    ``emit.build_manifest`` and ``Metrics.as_json`` do. So this compares everything that
    is a *decision* and ignores the clock -- and the caller separately asserts the clock
    is really there, so the comparison cannot pass by the field being dropped.
    """
    doc = json.loads(matches.read_text(encoding="utf-8"))
    return json.dumps({k: v for k, v in doc.items() if k != "timing"}, sort_keys=True)


def _blank_narrations(src: Path, dst: Path, constant: str = "BANK CREDIT") -> None:
    """Copy a run, replacing every bank narration with one constant string."""
    import csv as _csv
    import shutil

    shutil.copytree(src, dst)
    bank = dst / "bank_statement.csv"
    with bank.open("r", encoding="utf-8", newline="") as f:
        reader = _csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]
    narration_at = header.index("narration")
    for row in rows:
        row[narration_at] = constant
    with bank.open("w", encoding="utf-8", newline="") as f:
        writer = _csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def gate_9_matcher(sizes: tuple[int, ...] = (60, 200)) -> None:
    """Phase 3: the matcher scores 100/100/0 across seeds and sizes, deterministically.

    Four things, and the last two are the ones a passing coverage number cannot show:

      * **100% coverage, 100% correctness, 0 wrong matches** on every seed x size. The
        oracle already proved the target reachable on this data, so a shortfall is the
        matcher's fault rather than the dataset's.
      * **n=200 as well as n=60.** A bare ``net_paise`` is unique at n=60 and collides
        from n=200 up (1-2 collisions at n=200, 42-64 at n=1000), so testing only the
        default size would hide a key that breaks at Phase 12's scale.
      * **Determinism.** Two runs over one input agree everywhere outside ``timing``.
      * **The UTR shortcut did not get in.** Blanking every narration to a constant must
        change no decision. If it does, the narration was load-bearing -- and a matcher
        that keys on the UTR tail scores 100% here while never exercising the amount
        arithmetic, then stays at 100% through Phase 4 with no fee model written.
    """
    print(f"\ngate 9 -- the matcher on seeds {list(DEV_SEEDS)} x sizes {list(sizes)}")
    with tempfile.TemporaryDirectory(prefix="hisaab-matcher-") as tmp:
        root = Path(tmp)
        for seed in DEV_SEEDS:
            for n in sizes:
                base = root / f"s{seed}n{n}"
                data, truth = base / "data", base / "truth"
                _run(
                    [
                        sys.executable, "-m", "hisaab.generator",
                        "--seed", str(seed), "--n", str(n), "--month", "2026-08",
                        "--out", str(data), "--truth", str(truth), "--quiet",
                    ],
                    f"generator at seed {seed}, n={n}",
                )
                doc = _matcher_and_score(data, truth, base / "matches.json", seed)
                rates = doc["rates"]  # type: ignore[index]
                cells = doc["cells"]  # type: ignore[index]
                wrong = (
                    cells["wrong_match"] + cells["wrong_match_invented"]  # type: ignore[index]
                    + cells["lucky_guess"]  # type: ignore[index]
                )
                if (rates["coverage"], rates["correctness"], wrong) != (1.0, 1.0, 0):  # type: ignore[index]
                    raise GateFailure(
                        f"seed {seed}, n={n}: coverage {rates['coverage']}, "  # type: ignore[index]
                        f"correctness {rates['correctness']}, "  # type: ignore[index]
                        f"{wrong} wrong matches -- expected 1.0/1.0/0. Clean mode is row "
                        f"1 of the mess dial: anything less means the matcher is broken, "
                        f"not that the data is hard.\n  cells: {cells}"
                    )
                print(
                    f"    seed {seed}, n={n:<4} coverage 100.0%  correctness 100.0%  "
                    f"wrong 0  exceptions "
                    f"{doc['exceptions']['count']}"  # type: ignore[index]
                )

        # --- determinism, on one input ------------------------------------
        base = root / f"s{DEV_SEEDS[0]}n{sizes[0]}"
        data, truth = base / "data", base / "truth"
        first, second = base / "run_a.json", base / "run_b.json"
        for out in (first, second):
            _matcher_and_score(data, truth, out, DEV_SEEDS[0])
        if _without_timing(first) != _without_timing(second):
            raise GateFailure(
                "two matcher runs over the same data disagree outside their timing "
                "block, so something non-deterministic reached a verdict -- a set "
                "iteration, a dict insertion order, or an unsorted candidate list"
            )
        clocks = [
            json.loads(p.read_text(encoding="utf-8"))["timing"]["wall_clock_seconds"]
            for p in (first, second)
        ]
        if any(c is None for c in clocks):
            raise GateFailure(
                f"a run reported no wall clock ({clocks}), so the comparison above "
                f"passed by the field being absent rather than quarantined"
            )
        print(
            f"    determinism: two runs identical outside timing/ "
            f"(clocks {clocks[0] * 1000:.1f} ms, {clocks[1] * 1000:.1f} ms)"
        )

        # --- the UTR shortcut did not get in -------------------------------
        blanked_data = base / "blanked_data"
        _blank_narrations(data, blanked_data)
        blanked = base / "blanked.json"
        doc = _matcher_and_score(blanked_data, truth, blanked, DEV_SEEDS[0])
        if _decisions(first) != _decisions(blanked):
            raise GateFailure(
                "blanking every bank narration changed a matcher decision. The "
                "narration is therefore load-bearing on the match path -- which means "
                "the UTR-tail shortcut got in. That shortcut scores 100% on clean mode "
                "while never exercising the amount arithmetic, and would stay at 100% "
                "through Phase 4 with no fee model ever written. Tier 1 must key on "
                "(value_date, net_paise); the tail is corroboration only."
            )
        if doc["rates"]["coverage"] != 1.0:  # type: ignore[index]
            raise GateFailure(
                f"the matcher scored {doc['rates']['coverage']} with blanked "  # type: ignore[index]
                f"narrations -- it should be unaffected at 1.0"
            )
        print(
            "    the UTR shortcut is absent: every narration blanked to a constant, "
            "no decision moved, still 100%"
        )

        # --- provenance is real, not decorative ----------------------------
        # The seed in matches.json must be the one the run claims. If it were ignored,
        # the scorer could be handed any run's answer key and would report a plausible
        # number instead of refusing.
        # The matcher itself must succeed: it has no way to know the seed is wrong, and
        # inventing one would be worse. The refusal belongs to the scorer.
        _run(
            [
                sys.executable, "-m", "hisaab.matcher",
                "--data", str(data), "--out", str(base / "mismatch.json"),
                "--seed", str(DEV_SEEDS[0] + 100), "--month", "2026-08", "--quiet",
            ],
            "matcher with a deliberately wrong seed",
        )
        proc = subprocess.run(
            [
                sys.executable, "-m", "hisaab.scoring",
                "--matches", str(base / "mismatch.json"), "--truth", str(truth), "--quiet",
            ],
            cwd=ROOT, capture_output=True, text=True,
            env={**_env(), "PYTHONUTF8": "1"},
        )
        if proc.returncode == 0:
            raise GateFailure(
                "the scorer accepted a verdict file claiming the wrong seed. Scoring one "
                "run against another's answer key yields a plausible number rather than "
                "an error, which makes it the most expensive available bug."
            )
        print("    provenance: the scorer refuses a verdict file naming the wrong seed")


#: Gate 10's window, in **business days**. The posting lag is 1 business day
#: (ASSUMPTIONS.md #15a) and the bank credit is derived from the settlement, so a
#: credit-to-settlement join needs exactly that much room -- **not** the T+2 settlement
#: cycle, which shifts ``settled_on`` and ``value_date`` together and is therefore invisible
#: to this join (#15b). Measured rather than reasoned: under the delay, ``--window 0`` scores
#: 0.0 coverage and ``--window 1`` scores 1.0, while ``--fees`` alone passes at window 0.
MESS_WINDOW_DAYS = 1

#: Reasons that are an **honest abstention**: the matcher looked, could not separate the
#: candidates from the inputs alone, and said so. Gate 10 permits coverage below 100% only
#: when every shortfall carries one of these, because "did not resolve" and "resolved
#: wrongly" are the two outcomes this project refuses to average together.
ABSTENTION_REASONS: frozenset[str] = frozenset(
    {
        "AMBIGUOUS_DUPLICATE_AMOUNT",
        "AMBIGUOUS_MULTI_SUBSET",
        "UNEXPLAINED_RESIDUAL",
        # Phase 6. An honest refusal in the same sense as the two above: two declared rules
        # close the gap with *different* component splits, so committing to one would be a
        # coin flip on the decomposition -- which the scorer grades term by term, not on the
        # total. Admitted here knowing it cannot fire at the declared rates (see
        # ``Reason.AMBIGUOUS_ADJUSTMENT``); a code that is unreachable today and *not*
        # listed would turn into a spurious gate failure the first time a rate is re-pointed.
        "AMBIGUOUS_ADJUSTMENT",
        # Phase 6 steps 6 and 7, and both were **missing until gate 13 was written** -- worth
        # recording, because the reason they went unnoticed is structural rather than careless.
        # No gate before 13 runs ``--netted-refunds`` or ``--reserve``, so the reason-code
        # check at the bottom of this file had no run that could emit either one. A vocabulary
        # this list does not know about does not fail loudly; it fails the first time a gate
        # exercises the flag, which is exactly what gate 13 is for.
        #
        # ``REFUND_UNLINKED``: a refund whose payment is outside this month's file. Its amount
        # *is* declared in ``refunds.csv``, so the row is resolvable in principle -- the
        # abstention scores as a MISSED, not a correct abstention, and truth marks the holder
        # ``resolvable: true`` for precisely that reason.
        "REFUND_UNLINKED",
        # ``PARTIAL_SETTLEMENT_PENDING``: a credit short of a settlement's net by a plausible
        # rolling reserve. The held amount is declared in **no input file at all**, so the
        # arithmetic cannot be closed from the inputs -- while the payment set remains
        # recoverable from the untouched UTR, which is why these rows are also
        # ``resolvable: true`` and also score as misses.
        "PARTIAL_SETTLEMENT_PENDING",
        # Phase 8, and **the admission is argued rather than made to get a gate green** --
        # which is the only reason it is admissible at all. ``FX_RATE_GAP`` fires where a
        # settlement's membership is withheld, no subset of the pool sums to the credit, and a
        # payment in that pool was captured in a foreign currency: its ``gross_paise`` is fixed
        # at the capture-day rate while the settlement's net and its credit are not.
        #
        # It belongs here for the same reason ``PARTIAL_SETTLEMENT_PENDING`` does, and the
        # parallel is exact. Both name a quantity **declared in no input file** -- a rolling
        # reserve's held amount there, a settlement-day rate here -- so the arithmetic cannot be
        # closed from the inputs, only recognised. Both leave the payment set recoverable
        # through the untouched UTR, so truth marks the holder ``resolvable: true`` and the
        # abstention scores as a **miss, not a correct abstention**. Neither buys coverage.
        #
        # **What must not happen instead: admitting ``NO_CANDIDATE`` to make those rows pass.**
        # That code means "I looked and found nothing", which covers a missing payment and an
        # unmodelled deduction as well -- capability gaps. Admitting it would let every failed
        # search anywhere score as an honest refusal, and separating those two was Phase 7's
        # entire job. The producer is correspondingly narrow: with no foreign-currency payment
        # in the pool the row still falls through to ``NO_CANDIDATE``, and a **declared** cause
        # is checked first (an orphan refund, whose amount ``refunds.csv`` states), so this code
        # is reached only when nothing in the inputs can account for the gap.
        #
        # Admitted while **unreachable on today's data**, and deliberately: no generator flag
        # emits a non-INR payment until ``--fx`` lands two steps from here. Same treatment as
        # ``AMBIGUOUS_ADJUSTMENT`` above, for the same reason -- a code that is unreachable and
        # *not* listed turns into a spurious gate failure the moment it first fires, which is
        # precisely how ``REFUND_UNLINKED`` and ``PARTIAL_SETTLEMENT_PENDING`` went missing
        # until gate 13 was written. ``tier1.py``'s self-check reaches the branch with a
        # fixture, and all five plausible mis-implementations of it are caught by mutation
        # (`.plan/probe_phase8_branch_mutants.py`).
        "FX_RATE_GAP",
    }
)


def _corrupt_one_net(src: Path, dst: Path, delta: int = 307) -> tuple[str, int]:
    """Copy a run, shifting one settlement's ``net_paise`` **and its bank credit** by ``delta``.

    **Corrupting ``fee_paise`` would prove nothing**, and that is worth stating because it is
    the obvious thing to reach for. The matcher never reads that column -- re-deriving the fee
    from an independently declared rate is the whole point of ``matcher/fees.py`` -- so a
    poisoned ``fee_paise`` would sail through unnoticed and the gate would pass while testing
    nothing.

    So the corruption is to the **money**, and it moves the credit too. That keeps the join
    intact (Tier 1 keys on net == credit amount, and both moved together) while making the
    arithmetic impossible: the gross is unchanged, so no declared rule can account for a gap
    that is now 307p wrong. Matched-but-unproven is exactly the case #25 refuses, and this is
    the analogue of Phase 2's saboteur fixture -- it corrupts one number and expects the
    damage to land on one row, not to smear across the batch.
    """
    import shutil

    shutil.copytree(src, dst)

    def read(p: Path) -> tuple[list[str], list[list[str]]]:
        import csv as _csv

        with p.open(newline="", encoding="utf-8") as f:
            rd = _csv.reader(f)
            return next(rd), [r for r in rd if r]

    def write(p: Path, header: list[str], rows: list[list[str]]) -> None:
        import csv as _csv

        with p.open("w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f, lineterminator="\n")
            w.writerow(header)
            w.writerows(rows)

    sh, srows = read(dst / "settlements.csv")
    bh, brows = read(dst / "bank_statement.csv")
    net_at, sid_at, amt_at = sh.index("net_paise"), sh.index("settlement_id"), bh.index("amount_paise")

    was = int(srows[0][net_at])
    victim = srows[0][sid_at]
    srows[0][net_at] = str(was + delta)
    moved = 0
    for row in brows:
        if int(row[amt_at]) == was:
            row[amt_at] = str(was + delta)
            moved += 1
    if moved != 1:
        raise GateFailure(
            f"expected exactly one bank row at {was}p to move with settlement {victim}, "
            f"found {moved}. The corruption would then test batching rather than arithmetic."
        )
    write(dst / "settlements.csv", sh, srows)
    write(dst / "bank_statement.csv", bh, brows)
    return victim, was + delta


def gate_10_mess(sizes: tuple[int, ...] = (60, 200)) -> None:
    """Phase 4: the fee model and the date wedge, with the arithmetic proved per row.

    Gate 9 keeps clean mode honest; this one turns on the first two rows of the mess dial.
    What it asserts, and one thing it deliberately does **not**:

      * **Correctness 100% and 0 wrong matches, unconditionally.** This is the line that
        never bends.
      * **Coverage 100%, or a shortfall that is entirely honest abstention.** The plan asked
        for a flat 100/100/0 across seeds 1/2/3 x n=60/200. Measured, seed 3 at n=200 gives
        199/200 -- and the missing row is *correct behaviour*, so the expectation was wrong
        rather than the matcher. Two settlements genuinely share the net 417,899p (2 of 198
        nets are non-unique at that size), and once the window opens far enough to admit the
        posting lag, both are candidates for one credit. The inputs cannot separate them.
      * **The arithmetic agrees with truth term by term**, over a denominator the gate also
        checks. ``decomposition_agreement`` is compared against truth's own six-term block,
        never against the total: a fee 307p high and a GST 307p low close the identical gap,
        so a total-only comparison would score that CORRECT.
      * **A corrupted row abstains with ``UNEXPLAINED_RESIDUAL``**, and only that row.
      * **``--window 0`` fails under the posting lag**, proving the window is load-bearing,
        while ``--fees`` alone passes at window 0 -- which locates the requirement in the lag
        rather than in the fees.

    **Why it does not assert a tie-break**, which is the fix that suggests itself the moment
    a row abstains. Truth's answer for that credit is the settlement at **+1bd** -- the
    posting lag -- while the decoy sits at **+0bd**, same-day. ``prefer_closest`` keeps the
    **minimum** distance, so "nearest date wins" picks the decoy and reports a confident wrong
    match. That is not a near-miss heuristic: at a constant non-zero lag the closest candidate
    is the *least* likely one, so it is wrong every time it fires rather than occasionally.
    ``blocking.py``'s self-check carries the full measurement (5,040 pairs, all at +1; 5-10
    wrong matches per seed at n=1000 when it was live) and this gate is the end-to-end
    consequence of it. Trading a confident guess for an honest abstention is the trade this
    whole submission argues for, so the gate encodes the abstention as *acceptable* and a
    wrong match as *never*.

    Sign convention, since it is the easy thing to get backwards: distances here are
    ``date_distance`` -- ``business_days_between(settled_on, value_date)``, **positive when
    the credit lands after the settlement**. The verdict note on the abstaining row prints
    the same way, so the two cannot drift.
    """
    print(f"\ngate 10 -- the mess dial: --fees and --settlement-delay, seeds {list(DEV_SEEDS)}")
    with tempfile.TemporaryDirectory(prefix="hisaab-mess-") as tmp:
        root = Path(tmp)
        abstentions = 0
        for seed in DEV_SEEDS:
            for n in sizes:
                base = root / f"s{seed}n{n}"
                data, truth = base / "data", base / "truth"
                _run(
                    [
                        sys.executable, "-m", "hisaab.generator",
                        "--seed", str(seed), "--n", str(n), "--month", "2026-08",
                        "--out", str(data), "--truth", str(truth), "--quiet",
                        "--fees", "--settlement-delay",
                    ],
                    f"generator with --fees --settlement-delay at seed {seed}, n={n}",
                )
                out = base / "matches.json"
                doc = _matcher_and_score(
                    data, truth, out, seed,
                    extra=["--window", str(MESS_WINDOW_DAYS)],
                )
                rates, cells = doc["rates"], doc["cells"]  # type: ignore[index]
                wrong = (
                    cells["wrong_match"] + cells["wrong_match_invented"]  # type: ignore[index]
                    + cells["lucky_guess"]  # type: ignore[index]
                )
                if wrong or rates["correctness"] != 1.0:  # type: ignore[index]
                    raise GateFailure(
                        f"seed {seed}, n={n}: correctness {rates['correctness']}, "  # type: ignore[index]
                        f"{wrong} wrong matches. A wedge that costs *correctness* is a broken "
                        f"fee model or a broken window, not a harder dataset.\n  cells: {cells}"
                    )

                # A coverage shortfall is permitted only as honest abstention -- and the
                # verdicts are inspected rather than inferred from the count, because
                # "one missed row" and "one guessed row" produce the same coverage number.
                verdicts = json.loads(out.read_text(encoding="utf-8"))["verdicts"]
                unresolved = [v for v in verdicts if v["outcome"] != "RESOLVED"]
                bad = [v for v in unresolved if v.get("reason") not in ABSTENTION_REASONS]
                if bad:
                    raise GateFailure(
                        f"seed {seed}, n={n}: {len(bad)} row(s) failed for a reason that is "
                        f"not an honest abstention: "
                        f"{sorted({str(v.get('reason')) for v in bad})}. Expected one of "
                        f"{sorted(ABSTENTION_REASONS)}."
                    )
                abstentions += len(unresolved)

                # The arithmetic, and its denominator. A rate that agrees on 3 of 200 rows
                # also reports 100% agreement, so the count is the half that matters.
                agreement = rates["decomposition_agreement"]  # type: ignore[index]
                checked = doc["decomposition"]["checked"]  # type: ignore[index]
                if agreement != 1.0:
                    raise GateFailure(
                        f"seed {seed}, n={n}: decomposition_agreement {agreement} over "
                        f"{checked} rows -- the matcher's independently derived fee and GST "
                        f"disagree with truth's own arithmetic on at least one term. "
                        f"Mismatches: {doc['decomposition'].get('mismatches')}"  # type: ignore[index]
                    )
                if checked != cells["correct"]:  # type: ignore[index]
                    raise GateFailure(
                        f"seed {seed}, n={n}: {checked} rows had their arithmetic checked but "
                        f"{cells['correct']} resolved correctly. A resolved row whose "  # type: ignore[index]
                        f"decomposition went unscored is a check that has gone inert, which "
                        f"is worse than one never written -- the report still prints a rate."
                    )
                print(
                    f"    seed {seed}, n={n:<4} coverage {rates['coverage']:>7.2%}  "  # type: ignore[index]
                    f"correctness 100.0%  wrong 0  arithmetic {checked}/{checked}"
                    + (f"  ({len(unresolved)} honest abstention)" if unresolved else "")
                )
        print(
            f"    {abstentions} abstention(s) across the matrix, 0 wrong matches -- the "
            f"shortfall is refusal, not error"
        )

        # --- the window is load-bearing, and the lag is why -----------------
        base = root / f"s{DEV_SEEDS[0]}n{sizes[0]}"
        data, truth = base / "data", base / "truth"
        zero = _matcher_and_score(data, truth, base / "w0.json", DEV_SEEDS[0],
                                  extra=["--window", "0"])
        if zero["rates"]["coverage"] != 0.0:  # type: ignore[index]
            raise GateFailure(
                f"--window 0 scored {zero['rates']['coverage']} coverage under a "  # type: ignore[index]
                f"non-zero posting lag. It must score 0.0: the credit posts a business day "
                f"after the settlement, so a +/-0 join cannot reach it. Anything else means "
                f"the window is not being applied and D3's claim is untested."
            )
        fees_only = base / "fees_only"
        _run(
            [
                sys.executable, "-m", "hisaab.generator",
                "--seed", str(DEV_SEEDS[0]), "--n", str(sizes[0]), "--month", "2026-08",
                "--out", str(fees_only / "data"), "--truth", str(fees_only / "truth"),
                "--quiet", "--fees",
            ],
            "generator with --fees alone",
        )
        alone = _matcher_and_score(
            fees_only / "data", fees_only / "truth", fees_only / "m.json", DEV_SEEDS[0],
            extra=["--window", "0"],
        )
        if alone["rates"]["coverage"] != 1.0:  # type: ignore[index]
            raise GateFailure(
                f"--fees alone scored {alone['rates']['coverage']} at --window 0, "  # type: ignore[index]
                f"expected 1.0. Fees wedge gross against net and move no date, so they must "
                f"cost nothing at +/-0 -- otherwise the two wedges are entangled and the "
                f"window result above cannot be attributed to the posting lag."
            )
        print(
            "    the window is load-bearing: --window 0 scores 0% under the posting lag, "
            "while --fees alone still scores 100% there"
        )

        # --- one corrupted number, one refused row -------------------------
        bad_dir = base / "corrupt"
        victim, now = _corrupt_one_net(data, bad_dir, delta=307)
        broken = _matcher_and_score(bad_dir, truth, base / "corrupt.json", DEV_SEEDS[0],
                                    extra=["--window", str(MESS_WINDOW_DAYS)])
        verdicts = json.loads((base / "corrupt.json").read_text(encoding="utf-8"))["verdicts"]
        residual = [v for v in verdicts if v.get("reason") == "UNEXPLAINED_RESIDUAL"]
        if len(residual) != 1:
            raise GateFailure(
                f"shifting settlement {victim}'s net to {now}p (and its credit with it) "
                f"produced {len(residual)} UNEXPLAINED_RESIDUAL rows, expected exactly 1. "
                f"The gap is 307p that no declared rule can name, so the row must be matched "
                f"but refused -- and the damage must not spread to rows that are still "
                f"arithmetically sound.\n  reasons: "
                f"{sorted({str(v.get('reason')) for v in verdicts if v.get('reason')})}"
            )
        broken_wrong = (
            broken["cells"]["wrong_match"] + broken["cells"]["wrong_match_invented"]  # type: ignore[index]
            + broken["cells"]["lucky_guess"]  # type: ignore[index]
        )
        if broken_wrong or broken["rates"]["correctness"] != 1.0:  # type: ignore[index]
            raise GateFailure(
                f"the corrupted run reported {broken_wrong} wrong matches at correctness "
                f"{broken['rates']['correctness']} -- a bad number must cost coverage, "  # type: ignore[index]
                f"never correctness"
            )
        print(
            f"    a 307p corruption to {victim} is refused as UNEXPLAINED_RESIDUAL: "
            f"1 row abstains, the other {broken['cells']['correct']} still prove their "  # type: ignore[index]
            f"arithmetic"
        )

        # --- determinism, with both wedges on ------------------------------
        first, second = base / "det_a.json", base / "det_b.json"
        for out in (first, second):
            _matcher_and_score(data, truth, out, DEV_SEEDS[0],
                               extra=["--window", str(MESS_WINDOW_DAYS)])
        if _without_timing(first) != _without_timing(second):
            raise GateFailure(
                "two runs over the same --fees --settlement-delay data disagree outside "
                "their timing block. The fee model added arithmetic to the match path, and "
                "a derived number is a new place for iteration order to leak in."
            )
        print("    determinism holds with both wedges on, outside timing/")


#: Phase 4b: pairs ``--dup-amounts`` plants by default (``GenConfig.dup_pairs``). Two rather
#: than one, so the ``correct_abstention`` denominator is never 1 -- a rate of 1/1 cannot be
#: told apart from a coincidence.
DUP_PAIRS = 2

#: A UTR tail as it reaches a bank narration. ``settlements.csv`` carries the masked form
#: (``XXXX8928``); the narration carries the bare four digits, in any of the four templates
#: (``NEFT-RZRPAY-8104``, ``IMPS CR/RAZORPAY SOFTWARE/4451``). I7 already guarantees no
#: amount is echoed into a narration, and every narration in the generated data was measured
#: to hold exactly one run of digits, exactly four long -- so this pattern cannot pick up a
#: date fragment or an amount by accident.
TAIL_RE = re.compile(r"\d{4}")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    """One CSV as a list of dicts. Gate-local: gates read files, they do not import loaders."""
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _tail_of(narration: str) -> str | None:
    runs = TAIL_RE.findall(narration)
    return runs[-1] if runs else None


def gate_11_planted(sizes: tuple[int, ...] = (60, 200)) -> None:
    """Phase 4b: the exception list is **measured**, not asserted.

    Every earlier gate scores a run in which every row is resolvable, so
    ``correct_abstention`` has never held a number -- it printed as 0/0 while carrying this
    project's central claim, that the honest-abstention count is a measurement rather than a
    promise. This gate is where that cell gets a denominator, and it is why ``--dup-amounts``
    was pulled out of Phase 8 into a phase of its own.

    **The load-bearing assertion is the UTR one, and it is not the obvious one.** Before this
    flag was built, a tail-only strategy -- reading no date and no amount, joining the bank
    narration straight onto ``settlements.csv``'s ``utr`` column -- was measured resolving
    60/60, 200/200 and 1000/1000 credits *correctly* on every dev seed, clean and under both
    wedges alike, because tails are drawn without replacement. So a pair colliding on
    ``(date, amount)`` while keeping distinct tails is **still separable**, by a strategy no
    more sophisticated than exhaustive string matching. Marking such a pair
    ``resolvable=false`` would be a false statement about the data, and every
    ``correct_abstention`` counted on it would be counting a fiction.

    That is not a hypothetical: it is the failure a comparable Track 04 build recorded
    against itself, when the tier it had designed as its AI showcase fell to enumeration of
    narration substrings -- 200/200, zero wrong. So this gate re-runs that cheap attack on
    every planted row and requires it to come back *ambiguous*. A flag that survives only
    because nobody tried the cheap attack is not testing the capability its name claims.

    What it asserts, per dev seed and size, with ``--fees``, ``--settlement-delay`` and
    ``--dup-amounts`` on:

      * ``planted_pairs`` in the manifest, and ``2 x`` that many unresolvable rows in truth,
        each carrying ``AMBIGUOUS_DUPLICATE_AMOUNT``. The count is the *denominator*, so a
        wrong one silently rescales the claim rather than failing it.
      * ``I3.unique_date_amount`` suspended and **named** in the manifest, and nothing else
        suspended. An unannounced skip is what made the old ``clean_mode`` gate dangerous.
      * each planted pair shares one ``(value_date, amount_paise)`` **and one UTR**, and its
        two narrations parse to the same tail.
      * the tail-only join is ambiguous on every planted row and still resolves every other
        row -- so the plant is doing the work, and the file has not simply been degraded.
      * the matcher abstains on both members with that same code, and ``correct_abstention``
        equals the planted count exactly.
      * ``lucky_guess`` is 0. A matcher that commits to one member of a pair has even odds of
        naming the right payment set, and crediting that would reward guessing over
        abstaining -- the inversion the scorer exists to prevent.
      * correctness 100% and 0 wrong matches, **unconditionally**, exactly as in gate 10.
      * a negative control: the same seed and size *without* the flag must report 0 planted
        and ``correct_abstention`` 0. A rate is only attributable to a cause if the run
        without that cause reads zero.

    A coverage shortfall beyond the planted rows is permitted, for gate 10's reason: with
    I3's uniqueness check suspended a *natural* collision may also occur, and abstaining
    there is correct behaviour that the scorer records as ``MISSED``.

    What a pass does **not** prove: that a planted pair is unresolvable by every conceivable
    strategy -- only that it defeats the two this data supports (date+amount, and the UTR
    tail) plus the amount arithmetic. It also says nothing about three-way collisions, which
    I12 refuses at generation time rather than scoring.
    """
    print(
        f"\ngate 11 -- Phase 4b: {DUP_PAIRS} planted unresolvable pair(s), "
        f"seeds {list(DEV_SEEDS)}"
    )
    with tempfile.TemporaryDirectory(prefix="hisaab-planted-") as tmp:
        root = Path(tmp)
        for seed in DEV_SEEDS:
            for n in sizes:
                base = root / f"s{seed}n{n}"
                data, truth = base / "data", base / "truth"
                _run(
                    [
                        sys.executable, "-m", "hisaab.generator",
                        "--seed", str(seed), "--n", str(n), "--month", "2026-08",
                        "--out", str(data), "--truth", str(truth), "--quiet",
                        "--fees", "--settlement-delay", "--dup-amounts",
                    ],
                    f"generator with --dup-amounts at seed {seed}, n={n}",
                )

                manifest = json.loads(
                    (truth / "run_manifest.json").read_text(encoding="utf-8")
                )
                stated = manifest["config"]["planted_pairs"]
                if stated != DUP_PAIRS:
                    raise GateFailure(
                        f"seed {seed}, n={n}: run_manifest states planted_pairs={stated}, "
                        f"expected {DUP_PAIRS}. This number is the correct_abstention "
                        f"denominator, and the run describes its own answer key with it."
                    )
                skipped = manifest["invariants"]["checks_skipped"]
                if skipped != {"I3.unique_date_amount": ["dup_amounts"]}:
                    raise GateFailure(
                        f"seed {seed}, n={n}: expected exactly I3.unique_date_amount to be "
                        f"suspended and named in the manifest, got {skipped}. A check that "
                        f"stands down invisibly is what made the old clean_mode gate "
                        f"dangerous -- the checks vanished and the output looked identical."
                    )

                # --- truth's side of the plant --------------------------------------
                truth_doc = json.loads((truth / "truth.json").read_text(encoding="utf-8"))
                planted = [c for c in truth_doc["credits"] if not c["resolvable"]]
                #: Read off the answer key rather than hardcoded ``False``, so this gate keeps
                #: telling the truth if a later step ever runs it with the flag on. Phase 8
                #: step 7 measured both settings (`.plan/probe_phase8_gate11_utr_patchy.py`);
                #: the flag is not in this gate's own flag list, so today this is always False
                #: and the two assertions below it are equivalent to the single one they
                #: replaced -- which is the point. The gate does not weaken now, and it does
                #: not misdiagnose later.
                patchy_on = bool(truth_doc.get("flags", {}).get("utr_patchy", False))
                if len(planted) != 2 * DUP_PAIRS:
                    raise GateFailure(
                        f"seed {seed}, n={n}: truth marks {len(planted)} credit(s) "
                        f"unresolvable, expected {2 * DUP_PAIRS} ({DUP_PAIRS} pair(s) x 2). "
                        f"A miscount rescales the central claim instead of failing it."
                    )
                for c in planted:
                    if c["reason"] != "AMBIGUOUS_DUPLICATE_AMOUNT":
                        raise GateFailure(
                            f"seed {seed}, n={n}: planted {c['credit_id']} carries reason "
                            f"{c['reason']!r}. The generator's intent and the matcher's "
                            f"verdict must come from one vocabulary, or 'correct abstention' "
                            f"is a judgement call rather than a count."
                        )

                # --- the pair must be identical on every field that could link it ----
                # Joined through the CSVs rather than read off truth: truth.json carries no
                # value_date and no amount_paise per credit, because the answer key does not
                # duplicate the bank statement. The linkage is what it adds.
                bank = {r["row_id"]: r for r in _csv_rows(data / "bank_statement.csv")}
                utr_of = {
                    r["settlement_id"]: r["utr"]
                    for r in _csv_rows(data / "settlements.csv")
                }
                groups: dict[tuple[str, str], list[dict]] = {}
                for c in planted:
                    row = bank[c["credit_id"]]
                    groups.setdefault((row["value_date"], row["amount_paise"]), []).append(c)
                if len(groups) != DUP_PAIRS:
                    raise GateFailure(
                        f"seed {seed}, n={n}: the {len(planted)} planted rows form "
                        f"{len(groups)} (date, amount) group(s), expected {DUP_PAIRS}. Three "
                        f"credits sharing a key is a harder case than the pair this flag "
                        f"documents, and it would be scored as though it were the pair."
                    )
                for key, members in sorted(groups.items()):
                    if len(members) != 2:
                        raise GateFailure(
                            f"seed {seed}, n={n}: planted group {key} has {len(members)} "
                            f"members, not 2."
                        )
                    utrs = {utr_of[sid] for c in members for sid in c["settlement_ids"]}
                    if len(utrs) != 1:
                        raise GateFailure(
                            f"seed {seed}, n={n}: planted group {key} spans {len(utrs)} "
                            f"distinct UTRs {sorted(utrs)}. The tail reaches the bank "
                            f"narration and resolves 100% of rows on its own, so this pair "
                            f"is still separable by exhaustive narration matching -- it is "
                            f"NOT unresolvable, and truth calling it so is a false statement "
                            f"about the data."
                        )
                    # **Phase 8 step 7: split by cause.** This was one assertion --
                    # ``len(tails) != 1 or None in tails`` -- and its message named exactly one
                    # cause: "story.build's echo fixup must be memoised". That reading is right
                    # for two *different* tails and wrong for a *missing* one, and under
                    # ``--utr-patchy`` a planted member legitimately parses to ``None``. Left
                    # merged, the gate would fail on an honest run advising a maintainer to fix
                    # a memoisation bug that is not there -- and the fix they would reach for
                    # (making the mask skip planted rows) is the one I12 forbids, because it
                    # makes absence-of-tail a tell for unresolvability.
                    tails = {_tail_of(bank[c["credit_id"]]["narration"]) for c in members}
                    distinct = {t for t in tails if t is not None}
                    if len(distinct) > 1:
                        raise GateFailure(
                            f"seed {seed}, n={n}: planted group {key}'s narrations parse to "
                            f"{len(distinct)} different tails {sorted(distinct)} despite one "
                            f"shared UTR. story.build's echo fixup must be memoised, or each "
                            f"member draws its own spare tail and the pair is separated again."
                        )
                    if None in tails and not patchy_on:
                        raise GateFailure(
                            f"seed {seed}, n={n}: planted group {key} has a member whose "
                            f"narration carries no reference tail, without --utr-patchy. Every "
                            f"genuine template renders a 4-digit tail, so a missing one here is "
                            f"the mask firing on a run that did not ask for it -- a different "
                            f"defect from the echo fixup above, which is why the two are "
                            f"separate assertions."
                        )

                # --- the brute-force attack, actually run rather than argued away ----
                by_tail: dict[str, list[str]] = {}
                for sid, utr in utr_of.items():
                    by_tail.setdefault(utr.removeprefix("XXXX"), []).append(sid)
                planted_ids = {c["credit_id"] for c in planted}
                separated: list[str] = []
                # **Phase 8 step 7: one counter became three, by cause.** This was a single
                # ``ambiguous`` tally over every row the tail-only join failed to resolve
                # uniquely, asserted equal to the planted count -- and its own message named the
                # flag that would break it: "a tail missing or colliding elsewhere is
                # --utr-patchy's job in Phase 8, not this flag's". It was right. A masked row has
                # no tail, so under that flag the tally goes from 4 to ~34 at n=200 and the gate
                # fails reporting "the file has been degraded beyond the plant" about a run that
                # is behaving exactly as designed.
                #
                # The three causes are genuinely different claims, and only the third belongs to
                # ``--dup-amounts``:
                #
                #   * **no tail at all** -- the mask. Legitimate under ``--utr-patchy``, and a
                #     defect on any other run, since every genuine template renders a 4-digit
                #     tail.
                #   * **a tail that hits no settlement** -- a generator defect here, because
                #     every gateway credit's tail *is* its settlement's UTR tail. This gate runs
                #     without ``--noise-rows``, so there is no legitimate source of one; a future
                #     flag that adds ``look_alike`` rows would make this bucket honest and would
                #     need the same split treatment rather than a relaxed number.
                #   * **a tail hitting two or more settlements** -- the plant itself, and the
                #     only bucket whose size this flag controls.
                no_tail, tail_hits_none, collision = 0, 0, 0
                for cid, row in bank.items():
                    tail = _tail_of(row["narration"])
                    hits = by_tail.get(tail, []) if tail is not None else []
                    if len(hits) == 1:
                        if cid in planted_ids:
                            separated.append(cid)
                    elif tail is None:
                        no_tail += 1
                    elif not hits:
                        tail_hits_none += 1
                    else:
                        collision += 1
                if separated:
                    raise GateFailure(
                        f"seed {seed}, n={n}: a tail-only strategy -- no date, no amount, "
                        f"just the narration joined onto settlements.csv -- uniquely resolves "
                        f"planted row(s) {sorted(separated)}. The plant is separable by brute "
                        f"force, so it does not test the capability its name claims and "
                        f"resolvable=false is false for those rows."
                    )
                # The plant's own bucket, and the assertion that used to be made against the
                # merged tally. Strictly stronger now: the old form could be satisfied by a
                # missing tail cancelling out a collision that never happened.
                #
                # Under ``--utr-patchy`` a masked planted member carries no tail, so it moves to
                # the bucket above rather than colliding -- measured at seed 2 and seed 3, n=200,
                # the two cells where a pair is split by the mask
                # (`.plan/probe_phase8_gate11_utr_patchy.py`). That is not the plant weakening:
                # the two settlements still share one UTR, the credit simply no longer carries a
                # tail to reach it with, and the pair stayed unseparated in both cells.
                masked_planted = sum(
                    1 for cid in planted_ids if _tail_of(bank[cid]["narration"]) is None
                )
                if collision != len(planted) - masked_planted:
                    raise GateFailure(
                        f"seed {seed}, n={n}: the tail-only join collides on {collision} row(s) "
                        f"against {len(planted)} planted less {masked_planted} masked. Every "
                        f"planted row that still carries a tail must land on a shared UTR, and "
                        f"nothing else may: a collision outside the plant means two settlements "
                        f"share a UTR by accident, which makes an unplanted row unresolvable "
                        f"while truth calls it resolvable."
                    )
                if no_tail and not patchy_on:
                    raise GateFailure(
                        f"seed {seed}, n={n}: {no_tail} bank row(s) carry no reference tail "
                        f"without --utr-patchy. Every genuine narration template renders a "
                        f"4-digit tail, so this is the mask firing on a run that did not ask "
                        f"for it -- reported separately from a collision because the two have "
                        f"opposite causes and opposite fixes."
                    )
                if tail_hits_none:
                    raise GateFailure(
                        f"seed {seed}, n={n}: {tail_hits_none} bank row(s) carry a tail that "
                        f"matches no settlement's UTR. This gate runs without --noise-rows, so "
                        f"every bank row is gateway money and its tail is its own settlement's "
                        f"-- a tail pointing nowhere means the narration and the settlement "
                        f"disagree about the reference."
                    )

                # --- and now the number this whole phase exists to produce -----------
                out = base / "matches.json"
                doc = _matcher_and_score(
                    data, truth, out, seed, extra=["--window", str(MESS_WINDOW_DAYS)],
                )
                cells = doc["cells"]  # type: ignore[index]
                rates = doc["rates"]  # type: ignore[index]
                totals = doc["totals"]  # type: ignore[index]
                wrong = (
                    cells["wrong_match"] + cells["wrong_match_invented"]
                    + cells["lucky_guess"]
                )
                if wrong or rates["correctness"] != 1.0:
                    raise GateFailure(
                        f"seed {seed}, n={n}: correctness {rates['correctness']}, {wrong} "
                        f"wrong match(es). This line never bends -- and a lucky_guess here "
                        f"means the matcher committed on a row the inputs cannot separate, "
                        f"which is either a guess or a leak.\n  cells: {cells}"
                    )
                if totals["planted_unresolvable"] != 2 * DUP_PAIRS:
                    raise GateFailure(
                        f"seed {seed}, n={n}: the scorer read "
                        f"{totals['planted_unresolvable']} planted unresolvable row(s), "
                        f"expected {2 * DUP_PAIRS}."
                    )
                if cells["correct_abstention"] != 2 * DUP_PAIRS:
                    raise GateFailure(
                        f"seed {seed}, n={n}: correct_abstention is "
                        f"{cells['correct_abstention']}, expected {2 * DUP_PAIRS}. Every "
                        f"planted row must be abstained on: this cell IS the claim that the "
                        f"exception list is measured rather than asserted.\n"
                        f"  cells: {cells}"
                    )

                verdicts = json.loads(out.read_text(encoding="utf-8"))["verdicts"]
                by_id = {v["credit_id"]: v for v in verdicts}
                for cid in sorted(planted_ids):
                    v = by_id[cid]
                    if (
                        v["outcome"] != "EXCEPTION"
                        or v["reason"] != "AMBIGUOUS_DUPLICATE_AMOUNT"
                    ):
                        raise GateFailure(
                            f"seed {seed}, n={n}: planted {cid} got "
                            f"{v['outcome']}/{v['reason']}, expected an EXCEPTION carrying "
                            f"AMBIGUOUS_DUPLICATE_AMOUNT. Truth and the matcher agree on the "
                            f"vocabulary, or the count means nothing."
                        )
                bad = [
                    v for v in verdicts
                    if v["outcome"] != "RESOLVED"
                    and v.get("reason") not in ABSTENTION_REASONS
                ]
                if bad:
                    raise GateFailure(
                        f"seed {seed}, n={n}: {len(bad)} row(s) failed for a reason that is "
                        f"not an honest abstention: "
                        f"{sorted({str(v.get('reason')) for v in bad})}"
                    )
                natural = cells["missed"]
                print(
                    f"    seed {seed}, n={n:<4} coverage {rates['coverage']:>7.2%}  "
                    f"correct abstentions {cells['correct_abstention']}/"
                    f"{totals['planted_unresolvable']}  wrong 0  lucky 0  "
                    f"tail-join blind on all {len(planted)} planted"
                    + (f"  (+{natural} natural collision)" if natural else "")
                )

        # --- the negative control -------------------------------------------------
        # Without it, "correct_abstention is 4" could be reporting something the flag had no
        # part in. A rate is attributable only if the run without the cause reads zero.
        seed, n = DEV_SEEDS[0], sizes[0]
        base = root / "control"
        data, truth = base / "data", base / "truth"
        _run(
            [
                sys.executable, "-m", "hisaab.generator",
                "--seed", str(seed), "--n", str(n), "--month", "2026-08",
                "--out", str(data), "--truth", str(truth), "--quiet",
                "--fees", "--settlement-delay",
            ],
            "generator without --dup-amounts (the negative control)",
        )
        control = _matcher_and_score(
            data, truth, base / "matches.json", seed,
            extra=["--window", str(MESS_WINDOW_DAYS)],
        )
        if (
            control["totals"]["planted_unresolvable"]  # type: ignore[index]
            or control["cells"]["correct_abstention"]  # type: ignore[index]
        ):
            raise GateFailure(
                f"without --dup-amounts the run still reports "
                f"{control['totals']['planted_unresolvable']} planted unresolvable row(s) "  # type: ignore[index]
                f"and correct_abstention {control['cells']['correct_abstention']}. Then the "  # type: ignore[index]
                f"flagged run's count is not attributable to the flag."
            )
        control_manifest = json.loads(
            (truth / "run_manifest.json").read_text(encoding="utf-8")
        )
        if control_manifest["config"]["planted_pairs"] != 0:
            raise GateFailure(
                "a run without --dup-amounts reports a non-zero planted_pairs, which "
                "describes an answer key it does not have"
            )
        print(
            "    control: the same seed and size without --dup-amounts reports 0 planted "
            "and correct_abstention 0 -- the count above is the flag's"
        )


#: Seeds for gate 12. Wider than ``DEV_SEEDS`` on purpose: a tier *distribution* is a
#: claim about a mix, and a mix measured on three seeds is a mix that can be one seed's
#: accident. `.plan/phase5.md` step 2 specifies seeds 1-5.
TIER_MIX_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)

#: The size gate 12's over-cap probe runs at. The tier 2 pool cap (``matcher/tier2.py``
#: ``MAX_POOL``) never binds at the gate's own sizes -- the pool tops out near 20 there against
#: a cap of 64 -- so without this the bound is a number in a docstring rather than a behaviour.
#:
#: Measured before choosing: at n=1200 the cap never fires; n=1500 fires it on 129 rows and
#: costs 8.7s; **n=2000 fires it on 345 rows and costs 3.2s**. The larger size being the
#: *cheaper* one is not a fluke and is the cap doing its job -- a refused pool skips the
#: enumeration entirely, so past the point where the cap binds, more data means less work. That
#: is the whole claim behind a bounded refusal, and it is why this probe is affordable on every
#: acceptance run.
TIER_MIX_CAP_N: int = 2000

#: The flag set gate 12 scores. ``--batching`` alone does not force a subset search:
#: ``settlement_items.csv`` declares which payments belong to which settlement, so Tier 1
#: resolves a 4-payment batch by lookup and Tier 2 is never reached -- which is why the gate
#: was landed red in step 2 with only that flag. ``--settlement-report-late`` withholds that
#: membership for *some* settlements, and a withheld batch is work only a search can do.
#:
#: Deliberately **not** adding ``--fees`` and ``--settlement-delay``. They are gate 10's
#: subject, they would need ``--window 1`` here, and this gate reads a tier *distribution* --
#: one variable at a time is the discipline that put ``--settlement-delay`` before ``--fees``
#: in Phase 4. Measured at both settings before choosing: the mix, correctness and wrong-match
#: count are identical with and without them across seeds 1-5 at n=60 and n=200, so the extra
#: flags would add a second moving part and no signal.
TIER_MIX_FLAGS: tuple[str, ...] = ("--batching", "--settlement-report-late")


def _tier_mix_failure(
    verdicts: list[dict[str, object]], label: str
) -> str | None:
    """Describe what is wrong with a run's tier distribution, or ``None`` if it is healthy.

    Factored out of the gate so that the **same predicate** can be run against synthetic
    inputs whose answer is known. That is not tidiness: gate 12 is landed in a state where it
    *fails*, and a check that fails on every input it will ever see is indistinguishable from
    a ``raise``. The probe in ``_gate_12_self_check`` feeds this function a healthy mix, an
    all-Tier-1 run and an all-Tier-2 run, and requires it to accept the first and reject the
    other two -- so the red state below is a **measurement**, not a hard-coded refusal.

    The predicate, and why each half is here:

      * **Tier 2 must be non-zero.** This is the half that fails today. A green 100/100/0
        under ``--batching`` says nothing about the search, because Tier 1 resolves batches
        by declared membership; the *numbers* are right and the *capability* is absent.
      * **Tier 1 must also be non-zero.** The failure this half catches is the opposite one
        and it is easy to walk into: withhold membership for *every* settlement and the
        distribution becomes a swap rather than a mix, so a Tier 1 regression can hide behind
        a Tier 2 success. It is the reason step 5 withholds *partially*.
      * **Every resolved row carries a tier.** A resolved row with ``tier=None`` is a row no
        rule claims, which makes per-rule attribution unauditable.
    """
    resolved = [v for v in verdicts if v.get("outcome") == "RESOLVED"]
    if not resolved:
        return f"{label}: no row resolved at all, so there is no distribution to check"

    untiered = [str(v.get("credit_id")) for v in resolved if v.get("tier") is None]
    if untiered:
        return (
            f"{label}: {len(untiered)} resolved row(s) carry no tier "
            f"(e.g. {untiered[:3]}). Per-rule attribution cannot be audited if a resolved "
            f"row belongs to no rule."
        )

    by_tier = collections.Counter(int(v["tier"]) for v in resolved)  # type: ignore[arg-type]
    tier1, tier2 = by_tier.get(1, 0), by_tier.get(2, 0)

    if tier1 and not tier2:
        return (
            f"{label}: Tier 1 resolved {tier1} row(s) and Tier 2 resolved 0. The run is "
            f"green and the search is never exercised. This is the state the gate was built "
            f"to refuse, and with the search present it is a regression rather than a "
            f"pending step: either membership is being declared for every settlement (check "
            f"that TIER_MIX_FLAGS still carries --settlement-report-late and that the "
            f"withholding share is non-zero), or every withheld row is abstaining -- in "
            f"which case the reason codes on the unresolved rows say which of the tier 2 "
            f"refusals fired."
        )
    if tier2 and not tier1:
        return (
            f"{label}: Tier 2 resolved {tier2} row(s) and Tier 1 resolved 0. Membership was "
            f"withheld from every settlement, so the distribution is a *swap* rather than a "
            f"mix and a Tier 1 regression would hide behind a Tier 2 success. Withhold "
            f"partially (decision 3)."
        )
    if not tier1 and not tier2:
        return f"{label}: nothing resolved at Tier 1 or Tier 2, only {dict(by_tier)}"
    return None


def _gate_12_self_check() -> None:
    """Prove ``_tier_mix_failure`` is satisfiable before using it to fail the suite.

    Gate 12 lands red. The one thing that would make that worthless is a predicate that
    cannot go green, so this runs it against three hand-built inputs whose verdict is known.
    """
    def rows(*tiers: int | None) -> list[dict[str, object]]:
        return [
            {"credit_id": f"C{i:04d}", "outcome": "RESOLVED", "tier": t}
            for i, t in enumerate(tiers, start=1)
        ]

    healthy = rows(1, 1, 2, 1, 2)
    if _tier_mix_failure(healthy, "probe") is not None:
        raise GateFailure(
            "gate 12's own predicate rejects a healthy Tier 1 / Tier 2 mix, so the failure "
            "it reports below would be unconditional and would prove nothing"
        )
    for bad, want in ((rows(1, 1, 1), "Tier 2 resolved 0"), (rows(2, 2), "Tier 1 resolved 0")):
        got = _tier_mix_failure(bad, "probe")
        if got is None or want not in got:
            raise GateFailure(
                f"gate 12's predicate failed to reject a one-sided distribution: "
                f"expected a complaint containing {want!r}, got {got!r}"
            )
    # An abstaining row legitimately carries no tier, and must not be read as untiered.
    mixed = healthy + [{"credit_id": "C9999", "outcome": "EXCEPTION", "tier": None}]
    if _tier_mix_failure(mixed, "probe") is not None:
        raise GateFailure(
            "gate 12's predicate counts an abstaining row as untiered. Only RESOLVED rows "
            "carry a tier -- verdict.py refuses an EXCEPTION that names one."
        )


def _gate_12_cap_probe(root: Path) -> str:
    """Run one deliberately over-cap dataset and require the tier 2 refusal to fire.

    The cap is the answer to "what happens at 10,000 records?", and the answer is only worth
    something if the refusal is a behaviour rather than a branch nobody has taken. At the
    gate's own sizes the pool never approaches it, so this generates a run big enough that it
    binds.

    Two things are asserted, and the second matters more than the first:

      * the refusal fires at all, with a note that **names the bound** -- an exception saying
        "the pool of 99 exceeds the cap of 80" is triage-able, and "could not resolve" is not;
      * refusing costs only coverage. Correctness stays 1.0 and wrong matches stay 0 while
        hundreds of rows are refused, which is the property that separates a bounded refusal
        from a search that gives up and guesses.

    Returns the line to print, so the gate reports the measurement rather than only its pass.
    """
    data, truth = root / "capdata", root / "captruth"
    out = root / "capmatches.json"
    _run(
        [
            sys.executable, "-m", "hisaab.generator",
            "--seed", "1", "--n", str(TIER_MIX_CAP_N), "--month", "2026-08",
            "--out", str(data), "--truth", str(truth), "--quiet", *TIER_MIX_FLAGS,
        ],
        f"generator at n={TIER_MIX_CAP_N} for the tier 2 cap probe",
    )
    doc = _matcher_and_score(data, truth, out, 1)
    verdicts = json.loads(out.read_text(encoding="utf-8"))["verdicts"]

    refused = [
        v for v in verdicts
        if v.get("outcome") != "RESOLVED"
        and str(v.get("reason")) == "MEMBERSHIP_UNDECLARED"
    ]
    if not refused:
        raise GateFailure(
            f"the tier 2 pool cap never fired at n={TIER_MIX_CAP_N}, so the bound in "
            f"matcher/tier2.py is a declared number rather than a demonstrated behaviour. "
            f"Either the cap moved, or the pool no longer grows with n -- raise "
            f"TIER_MIX_CAP_N until it binds, and check the note names the bound."
        )
    named = [v for v in refused if str(TIER2_MAX_POOL) in str(v.get("note") or "")]
    if not named:
        raise GateFailure(
            f"{len(refused)} row(s) were refused for an undeclared membership at "
            f"n={TIER_MIX_CAP_N}, but no note names the cap of {TIER2_MAX_POOL}. A refusal "
            f"that does not state its bound is indistinguishable from a search that gave up."
        )
    cells, rates = doc["cells"], doc["rates"]  # type: ignore[index]
    if rates["correctness"] != 1.0 or cells["wrong_match"]:  # type: ignore[index]
        raise GateFailure(
            f"at n={TIER_MIX_CAP_N} the cap refused {len(refused)} row(s) and correctness "
            f"fell to {rates['correctness']} with "  # type: ignore[index]
            f"{cells['wrong_match']} wrong match(es). "  # type: ignore[index]
            f"Refusing an over-cap pool must cost coverage and nothing else."
        )
    return (
        f"    the cap is demonstrated, not declared: at n={TIER_MIX_CAP_N} the pool exceeds "
        f"{TIER2_MAX_POOL} on {len(refused)} row(s), each naming the bound, and correctness "
        f"stays 100% with 0 wrong matches"
    )


def gate_12_tier_mix(sizes: tuple[int, ...] = (60, 200)) -> None:
    """Phase 5: **both tiers must carry rows**, not just the totals.

    Every gate before this one scores a run in which one rule resolves everything, so
    "the search works" has never been a claim any run could contradict. This gate is the
    per-rule attribution check, and Phase 5 is the first phase where it can say anything.

    **It was landed deliberately red in step 2, and step 6 turned it green.** The state it
    refuses was measured, not hypothetical: at seeds 1-5 and 42, n=60 and n=200,
    ``--batching`` alone scored coverage 1.0, correctness 1.0 and decomposition agreement 1.0
    with **Tier 2 at zero on every one of them** -- 120 of 120 rows at Tier 1 at seed 42,
    n=200, 49 of them citing more than one payment. So a batched run already looked like a
    passing phase while the subset search did not exist, because ``settlement_items.csv``
    declares membership and turns the search into a lookup. Unlike Phase 4's version of this
    trap the numbers were *correct*; only the mechanism claim was false, which is what made it
    worth a gate rather than a note. It is also the shape of a comparable Track 04 build's
    recorded near-miss, where the tier designed as the showcase was resolved by enumerating
    narration substrings instead.

    That history is why the predicate is separate from the data. A check that cannot pass is a
    ``raise`` wearing a gate's clothes, so ``_gate_12_self_check`` runs the predicate against a
    healthy mix and against both one-sided distributions before the gate reads a single run --
    which is what made the red state attributable to the data, and makes the green one worth
    something now.

    The unconditional lines are asserted here too, so that step 6 cannot buy a Tier 2 number
    with a wrong match: correctness 100%, 0 wrong matches, and every coverage shortfall
    carrying an ``ABSTENTION_REASONS`` code.
    """
    print(
        f"\ngate 12 -- the tier distribution: both tiers must carry rows, seeds "
        f"{list(TIER_MIX_SEEDS)}"
    )
    _gate_12_self_check()
    print("    predicate accepts a healthy mix and rejects either one-sided run")

    with tempfile.TemporaryDirectory(prefix="hisaab-tiers-") as tmp:
        root = Path(tmp)
        failures: list[str] = []
        for seed in TIER_MIX_SEEDS:
            for n in sizes:
                base = root / f"s{seed}n{n}"
                data, truth = base / "data", base / "truth"
                _run(
                    [
                        sys.executable, "-m", "hisaab.generator",
                        "--seed", str(seed), "--n", str(n), "--month", "2026-08",
                        "--out", str(data), "--truth", str(truth), "--quiet",
                        *TIER_MIX_FLAGS,
                    ],
                    f"generator with {' '.join(TIER_MIX_FLAGS)} at seed {seed}, n={n}",
                )
                out = base / "matches.json"
                doc = _matcher_and_score(data, truth, out, seed)
                verdicts = json.loads(out.read_text(encoding="utf-8"))["verdicts"]

                # The lines that never bend, checked before the distribution: a Tier 2 count
                # bought with a wrong match is worse than a Tier 2 count of zero.
                cells, rates = doc["cells"], doc["rates"]  # type: ignore[index]
                if rates["correctness"] != 1.0 or cells["wrong_match"]:  # type: ignore[index]
                    raise GateFailure(
                        f"seed {seed}, n={n}: correctness "
                        f"{rates['correctness']} with "  # type: ignore[index]
                        f"{cells['wrong_match']} wrong match(es). "  # type: ignore[index]
                        f"Correctness 100% and 0 wrong matches are unconditional."
                    )
                # The third axis, with the denominator that stops it being vacuous.
                # ``decomposition_agreement`` is a ratio, so 1.0 over three checked rows would
                # pass a 200-row run while the arithmetic went unexamined; tying ``checked`` to
                # the correct-cell count is what makes the number mean "every row it got right
                # also proved its money". Measured equal on all ten gate runs before being
                # asserted here.
                dec = doc["decomposition"]  # type: ignore[index]
                if rates["decomposition_agreement"] != 1.0:  # type: ignore[index]
                    raise GateFailure(
                        f"seed {seed}, n={n}: decomposition agreement "
                        f"{rates['decomposition_agreement']} with "  # type: ignore[index]
                        f"{dec['mismatches']} mismatch(es) over "  # type: ignore[index]
                        f"{dec['checked']} checked row(s). A resolved row must prove "  # type: ignore[index]
                        f"its arithmetic term by term, and a batched settlement is where "
                        f"per-member rounding would show up."
                    )
                if dec["checked"] != cells["correct"]:  # type: ignore[index]
                    raise GateFailure(
                        f"seed {seed}, n={n}: decomposition agreement is 1.0 but only "
                        f"{dec['checked']} row(s) were checked against "  # type: ignore[index]
                        f"{cells['correct']} correct match(es). "  # type: ignore[index]
                        f"Agreement is a ratio: a perfect score over a subset of the "
                        f"correct rows is satisfied by not looking."
                    )

                unresolved = [v for v in verdicts if v.get("outcome") != "RESOLVED"]
                dishonest = [
                    str(v.get("credit_id")) for v in unresolved
                    if str(v.get("reason")) not in ABSTENTION_REASONS
                ]
                if dishonest:
                    raise GateFailure(
                        f"seed {seed}, n={n}: {len(dishonest)} unresolved row(s) carry a "
                        f"reason outside ABSTENTION_REASONS (e.g. {dishonest[:3]}). A "
                        f"coverage shortfall is only acceptable as an honest abstention."
                    )

                # Batching must actually be producing multi-payment settlements, or the
                # distribution below is a claim about data that does not exist.
                multi = sum(1 for v in verdicts if len(v.get("payment_ids") or []) > 1)
                if not multi:
                    raise GateFailure(
                        f"seed {seed}, n={n}: no resolved row cites more than one payment, "
                        f"so {' '.join(TIER_MIX_FLAGS)} batched nothing and a tier mix "
                        f"cannot be read off this run"
                    )

                problem = _tier_mix_failure(verdicts, f"seed {seed}, n={n}")
                if problem:
                    failures.append(problem)
                else:
                    by_tier = collections.Counter(
                        int(v["tier"]) for v in verdicts if v.get("outcome") == "RESOLVED"
                    )
                    print(
                        f"    seed {seed}, n={n:<4} tier1={by_tier.get(1, 0):<4} "
                        f"tier2={by_tier.get(2, 0):<4} multi-payment rows={multi}"
                    )

        if failures:
            raise GateFailure(
                f"{len(failures)} of {len(TIER_MIX_SEEDS) * len(sizes)} scored run(s) do "
                f"not exercise both tiers.\n\n  " + "\n  ".join(failures[:4])
                + (f"\n  ... and {len(failures) - 4} more" if len(failures) > 4 else "")
            )

        # Last, because it is the most expensive line in the gate and there is no reason to
        # pay for it when the distribution above is already wrong.
        print(_gate_12_cap_probe(root))


def _half_up_bps(amount_paise: int, bps: int) -> int:
    """``amount_paise * bps / 10_000``, rounded half-up at the paisa.

    Deliberately **re-implemented** here rather than imported from ``hisaab.common.money``.
    Gate 13 re-derives the TDS term to check the generator's arithmetic, and calling the
    generator's own rounding helper would make that a comparison of a function with itself:
    the two would agree on any rounding bug they shared, which is precisely the bug worth
    catching. One line of integer arithmetic is a cheap price for an independent witness.

    ``floor(x + 1/2)`` on ``x = a*b/10_000``, done in integers as
    ``(2ab + 10_000) // 20_000``. The track spec's worked example: a fee of 2,222p at 18%
    GST is 399.96p, which must land on 400.
    """
    assert amount_paise >= 0 and bps >= 0, (amount_paise, bps)
    return (2 * amount_paise * bps + 10_000) // 20_000


def _orphan_bearing_undeclared(data: Path, credits: list[dict[str, object]]) -> set[str]:
    """Credits that net an **orphan** refund *and* whose settlement membership is withheld.

    The one shape on a Phase 6 run where ``NO_CANDIDATE`` is the honest answer, and it needs
    both halves at once:

      * An *orphan* refund cites a payment outside this month's ``payments.csv``, so the
        matcher cannot attribute it -- ``refunds_by_payment`` rightly excludes it, and no
        per-member reading can price it.
      * With membership **declared**, that is not fatal: the settlement is matched on amount
        and the gap is named, which is ``REFUND_UNLINKED`` -- an honest abstention.
      * With membership **withheld**, the subset search's target is short by an amount nothing
        in the inputs explains, so the true set is not in the search space at all. The search
        looked and found nothing: ``NO_CANDIDATE``, meaning "a deduction is unmodelled", which
        is exactly what its note says.

    Returned as an identity set rather than admitted as a reason code, because widening
    ``ABSTENTION_REASONS`` to include ``NO_CANDIDATE`` would let *every* failure to find a
    candidate score as an honest refusal -- and telling a gateway credit it cannot explain from
    a non-gateway row is Phase 7's entire job. Gate 13 asserts containment in this set, so a
    second ``NO_CANDIDATE`` arriving for any other reason still fails the suite.
    """
    known = {row["payment_id"] for row in _rows(data / "payments.csv")}
    orphan_refund_ids = {
        row["refund_id"] for row in _rows(data / "refunds.csv")
        if row["payment_id"] not in known
    }
    declared = {row["settlement_id"] for row in _rows(data / "settlement_items.csv")}
    return {
        str(c["credit_id"]) for c in credits
        if set(c.get("refunds_netted") or ()) & orphan_refund_ids  # type: ignore[arg-type]
        and not (set(c["settlement_ids"]) & declared)  # type: ignore[arg-type]
    }


def _rows(path: Path) -> list[dict[str, str]]:
    """Every row of a CSV as a dict. An absent file reads as no rows, never as an error."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _gross_by_payment(data: Path) -> dict[str, int]:
    """``payment_id -> gross_paise``, read from ``payments.csv``.

    From the *inputs* rather than from truth, which is the whole point: it makes the TDS
    re-derivation a comparison between truth's declared term and the file the matcher
    reads. Taking both sides out of truth would pass a generator that wrote a wrong gross
    and a TDS that agreed with it.
    """
    with (data / "payments.csv").open("r", encoding="utf-8", newline="") as f:
        return {row["payment_id"]: int(row["gross_paise"]) for row in csv.DictReader(f)}


#: Phase 6's full flag set. ``--dup-amounts`` is deliberately absent and cannot be added:
#: ``GenConfig`` refuses it alongside either ``--netted-refunds`` or ``--reserve``, because
#: moving one member of a planted pair's credit away from its partner's makes the pair separable
#: and turns truth's ``resolvable: false`` into a false statement. Gate 11 owns the planted rows.
PHASE6_FLAGS: tuple[str, ...] = (
    "--fees", "--settlement-delay", "--batching", "--settlement-report-late",
    "--netted-refunds", "--tds", "--reserve",
)

#: The TDS rate the generator withholds, in basis points. §194-O was cut from 1% to 0.1%
#: effective 2024-10-01 (ASSUMPTIONS #9); the gate re-derives the term rather than trusting it.
PHASE6_TDS_BPS = 10


def _reserve_failure(
    verdicts: list[dict[str, object]],
    reserved_ids: set[str],
    label: str,
) -> str | None:
    """Gate 13's predicate: did the reserve land as a *diagnosis* and never as a match?

    Returns a complaint or ``None``. Separate from the data for the reason
    ``_gate_12_self_check`` records: a predicate that cannot go green is a ``raise`` wearing a
    gate's clothes, so ``_gate_13_self_check`` runs this against a healthy shape and against
    every failing shape before the gate reads a real run.

    Three things, and the middle one is the regression test the plan's correction (c) asked for:

      * **No resolved row carries a non-zero ``reserve_paise``.** Decision 4: the reserve is not
        modelled, because a deduction with a free magnitude closes every gap by construction --
        it would convert ``UNEXPLAINED_RESIDUAL`` rows into resolved ones while every arithmetic
        gate stayed green. This is the assertion that would catch someone "improving" coverage by
        fitting the shortfall.
      * **Every reserved credit abstains as ``PARTIAL_SETTLEMENT_PENDING``, never
        ``NO_CANDIDATE``.** A reserved credit is short of its settlement's net, so at an exact
        amount band it reaches blocking and finds nothing -- indistinguishable from a
        non-gateway row, which is exactly the distinction Phase 7 exists to make. So Phase 6
        owes Phase 7 a ``NO_CANDIDATE`` that is not silently carrying reserved rows.
      * **No reserved credit is resolved at all.** The held amount is declared in no input file,
        so committing to a settlement means fitting a magnitude nothing can verify.
    """
    by_id = {str(v.get("credit_id")): v for v in verdicts}

    fitted = [
        str(v.get("credit_id")) for v in verdicts
        if v.get("outcome") == "RESOLVED"
        and int((v.get("decomposition") or {}).get("reserve_paise", 0) or 0)  # type: ignore[union-attr]
    ]
    if fitted:
        return (
            f"{label}: {len(fitted)} resolved row(s) carry a non-zero reserve_paise "
            f"(e.g. {fitted[:3]}). The reserve is deliberately NOT modelled (decision 4): its "
            f"magnitude is declared in no input file, so a rule that fits it closes every gap "
            f"by construction and turns unexplained residuals into confident matches while "
            f"every arithmetic gate stays green"
        )

    resolved_reserved = sorted(
        cid for cid in reserved_ids
        if by_id.get(cid, {}).get("outcome") == "RESOLVED"
    )
    if resolved_reserved:
        return (
            f"{label}: {len(resolved_reserved)} reserved credit(s) were RESOLVED "
            f"(e.g. {resolved_reserved[:3]}). The held amount appears in no input file, so "
            f"committing to a settlement here means fitting a magnitude nothing can verify"
        )

    miscoded = sorted(
        (cid, str(by_id.get(cid, {}).get("reason")))
        for cid in reserved_ids
        if str(by_id.get(cid, {}).get("reason")) != "PARTIAL_SETTLEMENT_PENDING"
    )
    if miscoded:
        no_candidate = [cid for cid, reason in miscoded if reason == "NO_CANDIDATE"]
        return (
            f"{label}: {len(miscoded)} reserved credit(s) abstain with the wrong reason "
            f"(e.g. {miscoded[:3]})."
            + (
                f" {len(no_candidate)} came back NO_CANDIDATE, which is the specific failure "
                f"this assertion exists for: a reserved row hiding inside NO_CANDIDATE is "
                f"indistinguishable from a non-gateway credit, and telling those two apart is "
                f"Phase 7's entire job."
                if no_candidate
                else ""
            )
        )
    return None


def _gate_13_self_check() -> None:
    """Prove ``_reserve_failure`` is satisfiable, and rejects each shape it claims to.

    Same discipline as ``_gate_12_self_check`` and for the same reason: gate 13's value is
    entirely in what it refuses, so the predicate is exercised against a known-good input and
    against every known-bad one before a single real run is read.
    """
    def row(cid: str, outcome: str, reason: str | None = None,
            reserve: int = 0) -> dict[str, object]:
        v: dict[str, object] = {"credit_id": cid, "outcome": outcome, "reason": reason}
        if outcome == "RESOLVED":
            v["decomposition"] = {"reserve_paise": reserve}
        return v

    healthy = [
        row("C0001", "RESOLVED"),
        row("C0002", "RESOLVED"),
        row("C0003", "EXCEPTION", "PARTIAL_SETTLEMENT_PENDING"),
        # An unrelated abstention must not be read as a reserve failure.
        row("C0004", "EXCEPTION", "AMBIGUOUS_MULTI_SUBSET"),
    ]
    if (got := _reserve_failure(healthy, {"C0003"}, "probe")) is not None:
        raise GateFailure(
            f"gate 13's predicate rejects a healthy reserved run, so the failure it reports "
            f"below would be unconditional and would prove nothing: {got}"
        )
    # No reserved rows at all (every gate before this one) must also be clean, or the gate
    # cannot be run on an unreserved control.
    if _reserve_failure(healthy[:2], set(), "probe") is not None:
        raise GateFailure("gate 13's predicate fails on a run with no reserve at all")

    for bad, reserved, want in (
        # Decision 4's trap: a resolved row that priced the reserve.
        ([row("C0001", "RESOLVED", reserve=500)], set(), "non-zero reserve_paise"),
        # A reserved row resolved anyway.
        ([row("C0003", "RESOLVED")], {"C0003"}, "were RESOLVED"),
        # Correction (c)'s regression: the reserved row hid inside NO_CANDIDATE.
        ([row("C0003", "EXCEPTION", "NO_CANDIDATE")], {"C0003"}, "NO_CANDIDATE"),
        # ...or any other wrong code.
        ([row("C0003", "EXCEPTION", "UNEXPLAINED_RESIDUAL")], {"C0003"}, "wrong reason"),
    ):
        got = _reserve_failure(bad, reserved, "probe")
        if got is None or want not in got:
            raise GateFailure(
                f"gate 13's predicate failed to reject a bad shape: expected a complaint "
                f"containing {want!r}, got {got!r}"
            )


def gate_13_phase6(sizes: tuple[int, ...] = (200, 1000)) -> None:
    """Phase 6: three new deduction terms, and the one that must never be resolved.

    Every gate before this one scores runs where the whole gross/net wedge is fee and GST. This
    gate turns on all seven flags implemented as of Phase 6 at once and asserts the properties its three
    terms are supposed to have -- two of which are *arithmetic* and one of which is a refusal.

    **It runs at n=1000, and that size is not incidental.** Phase 6 step 5 swept the amount band
    and found its coverage cost severe and asymmetric with size: free up to 1,000p at n=200,
    while at n=1000 even 100p costs 6 rows and 1,000p costs 41. A property justified only at
    n=200 is justified on the size where it cannot fail.

    What it asserts, per size:

      * **All three new terms are non-zero somewhere**, and TDS on *every* settlement. At 10 bps
        against a floor of ₹100 the smallest drawable TDS is 10p, so a zero term is unreachable
        -- which inverts the plan's weaker "some row carries each term" into "no row escapes
        the TDS term". Re-derived at ``PHASE6_TDS_BPS`` rather than trusted, so a rate change in
        the generator cannot quietly agree with itself.
      * **The reserve is diagnosed and never resolved** -- ``_reserve_failure``, above, whose
        three clauses carry their own reasoning.
      * **Every resolved row names the rule that closed it.** A decomposition without a rule is
        a number that balances for unstated reasons, and Phase 6 adds two more ways for that to
        happen.
      * **Correctness 100% and 0 wrong matches, unconditionally**, plus decomposition agreement
        1.0 over a denominator equal to the correct-cell count -- gate 12's argument for why a
        ratio needs its denominator pinned applies unchanged.
      * **Every coverage shortfall is an honest abstention** from ``ABSTENTION_REASONS``. Both
        Phase 6 codes had to be *added* to that set to write this gate, which is the gate
        earning its keep before it ever ran: no earlier gate exercises these flags, so an
        unknown code would have failed silently until something looked.
      * **A negative control**: the same seed and size with no flags at all resolves everything
        and diagnoses no reserve. A rate is only attributable to a cause if the run without the
        cause reads zero.

    What a pass does **not** prove: that a reserved row *should* have abstained rather than
    resolved. With the held amount in no input file there is nothing to check against but
    truth's own record, so this gate can show the matcher declined to guess -- not that
    declining was the only available answer. That is a genuine limit of design B and it belongs
    in the write-up rather than in a reviewer's notes.
    """
    print(f"\ngate 13 -- Phase 6: all seven flags, seeds {list(DEV_SEEDS)}")
    _gate_13_self_check()

    with tempfile.TemporaryDirectory(prefix="hisaab-phase6-") as tmp:
        root = Path(tmp)
        for seed in DEV_SEEDS:
            for n in sizes:
                base = root / f"s{seed}n{n}"
                data, truth = base / "data", base / "truth"
                _run(
                    [
                        sys.executable, "-m", "hisaab.generator",
                        "--seed", str(seed), "--n", str(n), "--month", "2026-08",
                        "--out", str(data), "--truth", str(truth), "--quiet", *PHASE6_FLAGS,
                    ],
                    f"generator with all Phase 6 flags at seed {seed}, n={n}",
                )
                out = base / "matches.json"
                doc = _matcher_and_score(
                    data, truth, out, seed, extra=["--window", str(MESS_WINDOW_DAYS)],
                )
                verdicts = json.loads(out.read_text(encoding="utf-8"))["verdicts"]
                truth_doc = json.loads((truth / "truth.json").read_text(encoding="utf-8"))
                credits = truth_doc["credits"]

                # --- the three terms exist, and TDS is re-derived rather than trusted ----
                terms = {
                    name: sum(int(c["decomposition"][f"{name}_paise"]) for c in credits)
                    for name in ("tds", "refunds", "reserve")
                }
                for name, total in sorted(terms.items()):
                    if not total:
                        raise GateFailure(
                            f"seed {seed}, n={n}: the {name} term is zero across every credit "
                            f"while its flag is on -- the run is labelled with a mess it does "
                            f"not have, which is the mislabelling MessFlags.IMPLEMENTED exists "
                            f"to prevent arriving by a different door"
                        )
                gross_of = _gross_by_payment(data)
                zero_tds = [
                    c["credit_id"] for c in credits
                    if not int(c["decomposition"]["tds_paise"])
                ]
                if zero_tds:
                    raise GateFailure(
                        f"seed {seed}, n={n}: {len(zero_tds)} credit(s) carry a zero TDS term "
                        f"(e.g. {zero_tds[:3]}). At {PHASE6_TDS_BPS}bps against a ₹100 floor "
                        f"the smallest reachable TDS is 10p, so a zero term means the rate "
                        f"moved or the amount floor did -- and --tds would be a no-op on those "
                        f"rows while the flag claimed otherwise"
                    )
                # Re-derived per credit, exactly, from the members' own grosses in
                # payments.csv. Per credit rather than in total because two rows wrong in
                # opposite directions sum correctly -- the same reason I15 compares refunds
                # term by term.
                #
                # A *batch's* term is the sum of its members' terms, never a rate on the
                # batch total, and the two genuinely differ: three ₹104 members each round
                # 10.4p down to 10p for 30p, while the same rate on the ₹312 total is 31p.
                # So this re-derivation has to walk the members, and asserting anything about
                # ``decomposition.gross_paise`` alone would either be wrong on batches or
                # loose enough to prove nothing.
                for c in credits:
                    got_tds = int(c["decomposition"]["tds_paise"])
                    members = [str(pid) for pid in c["payment_ids"]]
                    missing = [pid for pid in members if pid not in gross_of]
                    if missing:
                        raise GateFailure(
                            f"seed {seed}, n={n}: {c['credit_id']} names payment(s) "
                            f"{missing[:3]} that payments.csv does not contain"
                        )
                    want = sum(_half_up_bps(gross_of[pid], PHASE6_TDS_BPS) for pid in members)
                    if got_tds != want:
                        raise GateFailure(
                            f"seed {seed}, n={n}: {c['credit_id']} declares tds_paise="
                            f"{got_tds}, but {PHASE6_TDS_BPS}bps half-up on each of its "
                            f"{len(members)} member gross(es) sums to {want}p. Either the "
                            f"rate is not what ASSUMPTIONS #9 states, or the term is being "
                            f"taken on the batch total rather than member by member"
                        )

                # --- the reserve is diagnosed, never resolved ---------------------------
                reserved_ids = {
                    str(c["credit_id"]) for c in credits
                    if int(c["decomposition"]["reserve_paise"])
                }
                if not reserved_ids:
                    raise GateFailure(
                        f"seed {seed}, n={n}: --reserve held nothing back, so the refusal this "
                        f"gate exists to check has no row to fire on"
                    )
                if problem := _reserve_failure(verdicts, reserved_ids, f"seed {seed}, n={n}"):
                    raise GateFailure(problem)

                # --- every resolved row names its rule ----------------------------------
                unruled = [
                    str(v.get("credit_id")) for v in verdicts
                    if v.get("outcome") == "RESOLVED"
                    and not (v.get("decomposition") or {}).get("rule")  # type: ignore[union-attr]
                ]
                if unruled:
                    raise GateFailure(
                        f"seed {seed}, n={n}: {len(unruled)} resolved row(s) publish no rule "
                        f"(e.g. {unruled[:3]}). A decomposition that balances without naming "
                        f"what closed it is a number a reader cannot audit, and Phase 6 adds "
                        f"two more terms that could be doing the closing"
                    )

                # --- the lines that never bend ------------------------------------------
                cells, rates = doc["cells"], doc["rates"]  # type: ignore[index]
                wrong = (
                    cells["wrong_match"] + cells["wrong_match_invented"]  # type: ignore[index]
                    + cells["lucky_guess"]  # type: ignore[index]
                )
                if wrong or rates["correctness"] != 1.0:  # type: ignore[index]
                    raise GateFailure(
                        f"seed {seed}, n={n}: correctness "
                        f"{rates['correctness']}, {wrong} wrong match(es). "  # type: ignore[index]
                        f"This line never bends -- and with three new deduction terms in play "
                        f"a wrong match most likely means one of them closed a gap it had no "
                        f"business closing.\n  cells: {cells}"
                    )
                dec = doc["decomposition"]  # type: ignore[index]
                if rates["decomposition_agreement"] != 1.0:  # type: ignore[index]
                    raise GateFailure(
                        f"seed {seed}, n={n}: decomposition agreement "
                        f"{rates['decomposition_agreement']} with "  # type: ignore[index]
                        f"{dec['mismatches']} mismatch(es). Six terms now, and the "  # type: ignore[index]
                        f"comparison is term by term precisely because a fee too high and a "
                        f"TDS too low land on the same total"
                    )
                if dec["checked"] != cells["correct"]:  # type: ignore[index]
                    raise GateFailure(
                        f"seed {seed}, n={n}: agreement is 1.0 over only "
                        f"{dec['checked']} of {cells['correct']} correct row(s) -- "  # type: ignore[index]
                        f"a ratio is satisfied by not looking"
                    )
                unresolved = [v for v in verdicts if v.get("outcome") != "RESOLVED"]
                # **The one-row ``NO_CANDIDATE`` exemption was RETIRED in Phase 8, and the rule
                # below is now unconditional.** It used to tolerate exactly one such row by
                # identity -- the credit netting an orphan refund whose settlement membership is
                # withheld -- because on a withheld settlement the orphan is subtracted from no
                # member, so the true subset sat outside the search space and the search honestly
                # found nothing.
                #
                # Phase 8 step 1 gave that row a real code instead. ``tier1.py``'s withheld-
                # membership branch now offers the shortfall back as ``credit + orphan``, and
                # where a subset appears it abstains as ``REFUND_UNLINKED`` -- naming an amount
                # ``refunds.csv`` declares rather than one the matcher fitted. Measured across
                # seeds 1/2/3/42 at n=200 and n=1000 on both this flag set and gate 14's: the
                # bump reveals **truth's own membership on 3 of 4 rows and an ambiguity on the
                # 4th, with zero coincidences** (`.plan/probe_phase8_refunds_first.py`).
                #
                # **So keeping the exemption would make this check vacuous, which is why it is
                # gone rather than merely unused.** With no row landing ``NO_CANDIDATE`` on this
                # flag set, an ``and not (...)`` clause guarding against one is a condition that
                # can no longer fire -- the decoration class Phase 7's own commit is named for.
                # Unconditional, it is strictly *stronger* than the exempted form and it earns a
                # second job: if the refunds-first ordering ever regresses, those rows fall back
                # to ``NO_CANDIDATE``, which is outside ``ABSTENTION_REASONS``, and this gate
                # fails. The exemption tolerated exactly that regression; the rule now catches
                # it. Same treatment ``SUSPENDED_BY`` gave its wrong predictions -- strengthened
                # rather than stood down.
                #
                # ``_orphan_bearing_undeclared`` is still called, but only to *name* that
                # population in the failure message. It is diagnostic now, never an excuse: the
                # rows it lists get no special permission, and one of them landing outside
                # ``ABSTENTION_REASONS`` fails the gate like any other.
                exempt = _orphan_bearing_undeclared(data, credits)
                dishonest = [
                    str(v.get("credit_id")) for v in unresolved
                    if str(v.get("reason")) not in ABSTENTION_REASONS
                ]
                if dishonest:
                    stale = sorted(set(dishonest) & exempt)
                    raise GateFailure(
                        f"seed {seed}, n={n}: {len(dishonest)} unresolved row(s) carry a "
                        f"reason outside ABSTENTION_REASONS (e.g. {dishonest[:3]}). A "
                        f"coverage shortfall is only acceptable as an honest abstention, and "
                        f"Phase 8 retired the one exemption this gate used to grant.\n"
                        + (
                            f"  {len(stale)} of them ({stale}) net an orphan refund on a "
                            f"settlement whose membership is withheld -- the population that "
                            f"USED to be exempt. Phase 8's refunds-first branch in "
                            f"tier1._search_membership should give those REFUND_UNLINKED, so a "
                            f"NO_CANDIDATE here means that ordering regressed rather than that "
                            f"the exemption is needed back.\n"
                            if stale
                            else ""
                        )
                        + f"  reasons seen: "
                        f"{dict(sorted(collections.Counter(str(v.get('reason')) for v in unresolved).items()))}"
                    )

                by_reason = collections.Counter(
                    str(v.get("reason")) for v in unresolved
                )
                print(
                    f"    seed {seed}, n={n:<5} resolved={cells['correct']:<5} "  # type: ignore[index]
                    f"reserved={len(reserved_ids):<4} "
                    f"tds={terms['tds']}p refunds={terms['refunds']}p "
                    f"reserve={terms['reserve']}p"
                )
                print(f"      abstentions: {dict(sorted(by_reason.items()))}")

        # --- the negative control -------------------------------------------------
        # A rate is only attributable to a cause if the run without that cause reads zero.
        # Same seed, same size, no flags: everything resolves and no reserve is diagnosed.
        seed, n = DEV_SEEDS[0], sizes[0]
        base = root / "control"
        data, truth = base / "data", base / "truth"
        _run(
            [
                sys.executable, "-m", "hisaab.generator",
                "--seed", str(seed), "--n", str(n), "--month", "2026-08",
                "--out", str(data), "--truth", str(truth), "--quiet",
            ],
            f"clean-mode control at seed {seed}, n={n}",
        )
        out = base / "matches.json"
        doc = _matcher_and_score(data, truth, out, seed)
        verdicts = json.loads(out.read_text(encoding="utf-8"))["verdicts"]
        cells, rates = doc["cells"], doc["rates"]  # type: ignore[index]
        if rates["coverage"] != 1.0 or rates["correctness"] != 1.0 or cells["missed"]:  # type: ignore[index]
            raise GateFailure(
                f"the clean-mode control at seed {seed}, n={n} no longer scores 100/100/0: "
                f"coverage {rates['coverage']}, correctness "  # type: ignore[index]
                f"{rates['correctness']}, {cells['missed']} missed. "  # type: ignore[index]
                f"Phase 6's reserve probe runs on every abstaining row, so a clean-mode "
                f"regression here would mean it is firing where nothing was held"
            )
        diagnosed = [
            str(v.get("credit_id")) for v in verdicts
            if str(v.get("reason")) == "PARTIAL_SETTLEMENT_PENDING"
        ]
        if diagnosed:
            raise GateFailure(
                f"the clean-mode control diagnosed {len(diagnosed)} reserve(s) "
                f"(e.g. {diagnosed[:3]}) on a run where nothing was held back. The probe's "
                f"plausibility floor is what should prevent this, so it is either too low or "
                f"being applied to the wrong base"
            )
        print(f"    control  seed {seed}, n={n}: 100/100/0, 0 reserves diagnosed")


#: Gate 13's seven plus Phase 7's two -- every flag implemented as of Phase 7 except ``--dup-amounts``
#: (gate 11 owns the planted rows, and combining it with ``--batching`` is refused outright).
PHASE7_FLAGS: tuple[str, ...] = (*PHASE6_FLAGS, "--noise-rows", "--unsettled")

#: The three noise strata, in the order ``run_manifest.json`` sorts them. Named here rather than
#: read from the manifest because a gate that learns the stratum names from the file it is
#: auditing cannot notice a stratum disappearing.
NOISE_STRATA: tuple[str, ...] = ("gateway_plausible", "look_alike", "plainly_foreign")


def _noise_failure(
    cells: dict[str, int],
    strata: dict[str, int],
    label: str,
) -> str | None:
    """Gate 14's predicate: did the non-gateway rows land on their own axis, correctly?

    Three clauses, and the first is the one carrying the phase.

    **The count identity.** ``noise_correctly_ignored`` must equal the manifest's realised
    ``plainly_foreign`` count -- not a rate, and not a threshold. Step 4 measured
    ``noise_recall`` as size-dependent (50%/41.7%/40.0% at n=60/200/1000) because the strata are
    allocated by largest remainder over a share that is itself a share of a varying row count, so
    any flat floor would be a seed-and-size-fitted number wearing a property's clothes. The
    identity is the property, and it closes **both** directions at once: below it, a
    ``plainly_foreign`` row the matcher should have set aside was not; above it, one of the two
    gateway-spelled strata leaked into ``IGNORED``, which is the shape that becomes Phase 8's
    ``WRONG_IGNORE`` the moment ``--utr-patchy`` strips a genuine credit's UTR.

    **What this is and is not.** It is a count identity, so on its own it permits an exchange --
    one ``plainly_foreign`` row diagnosed as pending while one ``look_alike`` row is ignored keeps
    the total. That exchange cannot happen, and the reason lives in a different check rather than
    this one: ``look_alike`` and ``gateway_plausible`` rows carry a gateway counterparty **by
    construction**, and I18 asserts precisely that masking on every generated story -- including
    the ones this gate generates, since ``check_story`` runs inside the generator subprocess. So
    set equality follows from the identity *composed with* I18, not from the identity alone. That
    composition is the honest claim; asserting set equality here would mean re-deriving each row's
    stratum from its narration using the same two evidence tests the matcher's gate uses, which
    would pass by construction and prove nothing. The per-row stratum is deliberately not on disk
    (``emit.build_manifest``: the id list needs none), and this is why it does not need to be.

    **``WRONG_IGNORE`` at zero, unconditionally.** Decision 6. Ignoring a genuine gateway credit
    drops real money out of the books; unlike a wrong match it leaves no residual behind to
    notice it by. There is no size, seed or flag set where a non-zero count here is acceptable.

    **Every stratum populated.** A stratum that drew no rows is a stratum whose handling this run
    did not test, and n=200 under ``--batching`` is where that would first happen -- the noise
    share is taken against a bank-row count that batching shrinks. A gate reporting "all three
    strata behave" over a run containing two of them is the vacuous pass this project keeps
    finding; measured minimum at n=200 is 2 rows in the smallest stratum, so the floor is real
    but thin.
    """
    ignored = cells["noise_correctly_ignored"]
    plainly_foreign = strata.get("plainly_foreign", 0)
    if ignored != plainly_foreign:
        direction = (
            f"{plainly_foreign - ignored} plainly-foreign row(s) were NOT set aside"
            if ignored < plainly_foreign
            else f"{ignored - plainly_foreign} gateway-spelled noise row(s) leaked into IGNORED"
        )
        return (
            f"{label}: {ignored} noise row(s) correctly ignored against "
            f"{plainly_foreign} plainly_foreign in the manifest -- {direction}. Only "
            f"plainly_foreign is ignorable: the other two strata carry a gateway counterparty by "
            f"construction and must fall through to a diagnosis. A leak in this direction is "
            f"Phase 8's WRONG_IGNORE arriving early, since --utr-patchy makes a genuine credit "
            f"look exactly like gateway_plausible.\n  strata: {dict(sorted(strata.items()))}"
        )

    if cells["wrong_ignore"]:
        return (
            f"{label}: {cells['wrong_ignore']} genuine gateway credit(s) were discarded as "
            f"non-gateway income. This is the one cell with no acceptable non-zero value "
            f"(decision 6) -- a wrong match leaves a residual a human can find, while a wrongly "
            f"ignored credit leaves nothing behind at all"
        )

    if empty := [s for s in NOISE_STRATA if not strata.get(s)]:
        return (
            f"{label}: stratum/strata {empty} drew no rows, so this run does not exercise "
            f"the handling this gate claims to check. The noise share is taken against a bank-row "
            f"count that --batching shrinks, so the small size is where a stratum empties first"
            f"\n  strata: {dict(sorted(strata.items()))}"
        )
    return None


def _gate_14_self_check() -> None:
    """Prove ``_noise_failure`` is satisfiable, and rejects each shape it claims to.

    Gate 13's discipline, for gate 13's reason: this gate's value is entirely in what it refuses,
    so the predicate meets a known-good input and every known-bad one before a real run is read.
    Both directions of the identity are exercised separately -- a predicate that caught only the
    shortfall would miss the leak, and the leak is the one that becomes a wrong answer next phase.
    """
    def cells(ignored: int, wrong_ignore: int = 0) -> dict[str, int]:
        return {"noise_correctly_ignored": ignored, "noise_mishandled": 0,
                "wrong_ignore": wrong_ignore}

    healthy_strata = {"gateway_plausible": 11, "look_alike": 11, "plainly_foreign": 15}
    if (got := _noise_failure(cells(15), healthy_strata, "probe")) is not None:
        raise GateFailure(
            f"gate 14's predicate rejects a healthy noisy run, so every failure it reports below "
            f"would be unconditional and would prove nothing: {got}"
        )

    for bad_cells, bad_strata, want in (
        # A plainly-foreign row the matcher failed to set aside.
        (cells(14), healthy_strata, "were NOT set aside"),
        # The direction that matters more: a gateway-spelled row wrongly ignored.
        (cells(16), healthy_strata, "leaked into IGNORED"),
        # Decision 6.
        (cells(15, wrong_ignore=1), healthy_strata, "no acceptable non-zero value"),
        # A stratum that drew nothing -- checked with the identity satisfied, so the
        # complaint must be attributable to the stratum rather than to the count.
        (cells(15), {**healthy_strata, "look_alike": 0}, "drew no rows"),
    ):
        got = _noise_failure(bad_cells, bad_strata, "probe")
        if got is None or want not in got:
            raise GateFailure(
                f"gate 14's predicate failed to reject a bad shape: expected a complaint "
                f"containing {want!r}, got {got!r}"
            )


def gate_14_phase7(sizes: tuple[int, ...] = (200, 1000)) -> None:
    """Phase 7: bank rows that are not gateway credits, and payments that never pay out.

    Every gate before this one scores a file where **every bank row is gateway money** and every
    payment eventually settles. Both assumptions are false in a real bank statement, and this gate
    turns on all nine flags implemented as of Phase 7 at once to assert what the matcher must
    do without them.

    **It runs at n=1000 even under ``--skip-slow``, and for gate 13's reason rather than by
    imitation.** Two of the properties below are invisible at n=200: ``AMBIGUOUS_MULTI_SUBSET``
    does not appear at all until the candidate pool is large enough for a coincidental subset to
    exist (measured: 0 at n=200 on all three seeds, 21-34 at n=1000), and Tier 2's pool cap is
    what Phase 7 step 0 raised to 80 -- a size that never presents a pool above 64 cannot notice
    a cap regression. A fast run that dropped n=1000 would report this phase green while blind to
    both.

    What it asserts, per seed and size:

      * **The noise axis lands correctly** -- ``_noise_failure``, above, whose three clauses carry
        their own reasoning and whose central claim is a count identity against the manifest's
        realised strata rather than a rate.
      * **Correctness 1.0 with zero wrong matches.** Gate 13 asserts this on seven flags; nothing
        before this gate asserts it with non-gateway rows and unsettled payments on the file, and
        both are new ways for a match to go wrong: a noise row sits in the same date-and-amount
        space as a settlement's net, and an orphaned payment is claimed by no settlement, so
        Tier 2's partition filter cannot remove it from the pool.
      * **Every unresolved *gateway* row abstains honestly**, with the noise rows excluded from
        that check rather than swept into it. Their outcome is scored on its own axis, and folding
        ``IGNORED`` into an abstention audit would let the easiest rows in the file vouch for the
        hardest. **Gate 13's single ``NO_CANDIDATE`` exemption was retired in Phase 8 and is no
        longer inherited here** -- this check is unconditional. It never fired on this gate's own
        seeds even before that (measured: the exempt row appears on seed 42, which this gate does
        not run), so it was already a permission carried rather than a clause depended on; Phase 8
        removed the permission itself, because the withheld-membership branch now abstains as
        ``REFUND_UNLINKED`` on that shape and a clause guarding a code that can no longer arrive
        is decoration. Unconditional is strictly stronger: a regression in that ordering drops
        those rows back to ``NO_CANDIDATE`` and this check fails on them.
      * **A negative control**: the same seed and size with no flags resolves everything, ignores
        nothing and reports ``noise_recall`` as ``n/a``. A rate is only attributable to a cause
        when the run without the cause reads zero -- and an ``IGNORED`` in clean mode would mean
        the gate from step 5 fires on rows that are plain gateway credits.

    What a pass does **not** prove. ``noise_recall`` sits near 40% by construction, not by
    difficulty: only ``plainly_foreign`` is ignorable, so the ceiling on this metric is the
    stratum split itself. The other two strata are *designed* to be indistinguishable from a
    genuine credit that has lost its UTR, because Phase 8 makes exactly that row real -- and a
    rule that ignored them would convert this phase's recall into next phase's wrong answers. The
    17-of-36 misdiagnosed as pending reserves at n=1000 are a measured limit of the band (step 6),
    not a defect this gate is failing to catch.
    """
    print(f"\ngate 14 -- Phase 7: all nine flags, seeds {list(DEV_SEEDS)}")
    _gate_14_self_check()

    with tempfile.TemporaryDirectory(prefix="hisaab-phase7-") as tmp:
        root = Path(tmp)
        for seed in DEV_SEEDS:
            for n in sizes:
                base = root / f"s{seed}n{n}"
                data, truth = base / "data", base / "truth"
                _run(
                    [
                        sys.executable, "-m", "hisaab.generator",
                        "--seed", str(seed), "--n", str(n), "--month", "2026-08",
                        "--out", str(data), "--truth", str(truth), "--quiet", *PHASE7_FLAGS,
                    ],
                    f"generator with all Phase 7 flags at seed {seed}, n={n}",
                )
                out = base / "matches.json"
                doc = _matcher_and_score(
                    data, truth, out, seed, extra=["--window", str(MESS_WINDOW_DAYS)],
                )
                verdicts = json.loads(out.read_text(encoding="utf-8"))["verdicts"]
                truth_doc = json.loads((truth / "truth.json").read_text(encoding="utf-8"))
                credits = truth_doc["credits"]
                manifest = json.loads(
                    (truth / "run_manifest.json").read_text(encoding="utf-8")
                )
                strata: dict[str, int] = manifest["noise_strata"]
                cells, rates = doc["cells"], doc["rates"]  # type: ignore[index]

                # --- the flags produced their mess at all --------------------------------
                noise_ids = set(truth_doc["orphans"]["non_gateway_credit_ids"])
                if not noise_ids:
                    raise GateFailure(
                        f"seed {seed}, n={n}: --noise-rows put nothing on the file, so every "
                        f"noise property below would be satisfied by an empty set"
                    )
                if sum(strata.values()) != len(noise_ids):
                    raise GateFailure(
                        f"seed {seed}, n={n}: the manifest's strata sum to "
                        f"{sum(strata.values())} but truth lists {len(noise_ids)} non-gateway "
                        f"row(s) -- the two truth-side files disagree about the same rows, and "
                        f"the identity below is read off the manifest"
                    )
                if not truth_doc["orphans"]["unsettled_payment_ids"]:
                    raise GateFailure(
                        f"seed {seed}, n={n}: --unsettled orphaned no payment, so the run is "
                        f"labelled with a mess it does not have"
                    )

                # --- the noise axis ------------------------------------------------------
                if problem := _noise_failure(cells, strata, f"seed {seed}, n={n}"):
                    raise GateFailure(problem)

                # --- the line that never bends ------------------------------------------
                wrong = (
                    cells["wrong_match"] + cells["wrong_match_invented"]
                    + cells["lucky_guess"]
                )
                if wrong or rates["correctness"] != 1.0:
                    raise GateFailure(
                        f"seed {seed}, n={n}: correctness {rates['correctness']}, {wrong} wrong "
                        f"match(es). Two new ways for this to break: a noise row occupies the "
                        f"same date-and-amount space as a settlement's net, and an orphaned "
                        f"payment cannot be filtered out of Tier 2's pool by partition.\n"
                        f"  cells: {cells}"
                    )

                # --- honest abstentions, gateway rows only -------------------------------
                # **Unconditional since Phase 8, for the reason gate 13's copy of this carries
                # in full.** The ``NO_CANDIDATE``-by-identity exemption is retired: the withheld-
                # membership branch in ``tier1._search_membership`` now offers an orphan refund's
                # declared amount back to the search and abstains as ``REFUND_UNLINKED``, so no
                # gateway row on this flag set lands ``NO_CANDIDATE`` any more and a clause
                # guarding one could not fire. Retired rather than left unused, and it gains a
                # second job that way: a regression in that ordering drops those rows back to
                # ``NO_CANDIDATE``, and this check now fails on them instead of excusing them.
                #
                # The noise-row exclusion is a different thing entirely and it **stays**. Those
                # rows are scored on their own axis (``noise_recall``, and ``_noise_failure``
                # above), and letting ``IGNORED`` count as an honest abstention would have the
                # easiest rows in the file vouch for the hardest. Measured: on this flag set the
                # noise strata are where ``NO_CANDIDATE`` legitimately lives -- 44 of the 48 such
                # rows across seeds 1/2/3/42 at both sizes are non-gateway
                # (`.plan/probe_phase8_no_candidate_control.py`), which is exactly why this
                # check is scoped to gateway rows rather than widened to accept the code.
                exempt = _orphan_bearing_undeclared(data, credits)
                dishonest = [
                    str(v.get("credit_id")) for v in verdicts
                    if v.get("outcome") != "RESOLVED"
                    and str(v.get("credit_id")) not in noise_ids
                    and str(v.get("reason")) not in ABSTENTION_REASONS
                ]
                if dishonest:
                    unresolved = [
                        v for v in verdicts
                        if v.get("outcome") != "RESOLVED"
                        and str(v.get("credit_id")) not in noise_ids
                    ]
                    stale = sorted(set(dishonest) & exempt)
                    raise GateFailure(
                        f"seed {seed}, n={n}: {len(dishonest)} unresolved gateway row(s) carry a "
                        f"reason outside ABSTENTION_REASONS (e.g. {dishonest[:3]}). The noise "
                        f"rows are excluded from this check on purpose -- they are scored on "
                        f"their own axis, and letting IGNORED count as an abstention would have "
                        f"the easiest rows in the file vouch for the hardest.\n"
                        + (
                            f"  {len(stale)} of them ({stale}) net an orphan refund on a "
                            f"settlement whose membership is withheld -- the population gate 13 "
                            f"used to exempt by identity. Phase 8's refunds-first branch in "
                            f"tier1._search_membership should give those REFUND_UNLINKED, so a "
                            f"NO_CANDIDATE here means that ordering regressed.\n"
                            if stale
                            else ""
                        )
                        + f"  reasons seen: "
                        f"{dict(sorted(collections.Counter(str(v.get('reason')) for v in unresolved).items()))}"
                    )

                ignored_total = sum(
                    1 for v in verdicts if v.get("outcome") == "IGNORED"
                )
                print(
                    f"    seed {seed}, n={n:<5} rows={len(verdicts):<5} "
                    f"noise={len(noise_ids):<4} ignored={ignored_total:<4} "
                    f"recall={rates['noise_recall']:.3f} "
                    f"mishandled={cells['noise_mishandled']:<4} "
                    f"correct={cells['correct']:<5} wrong={wrong} "
                    f"cov={rates['coverage']:.4f}  strata={dict(sorted(strata.items()))}"
                )

        # --- the negative control ----------------------------------------------------
        # Same seed and size, no flags. Two things must read zero, and the second is the
        # one step 5 could plausibly have broken: if the IGNORED gate fired on plain
        # gateway credits, it would show up here and nowhere else.
        seed, n = DEV_SEEDS[0], sizes[0]
        base = root / f"control-s{seed}n{n}"
        data, truth = base / "data", base / "truth"
        _run(
            [
                sys.executable, "-m", "hisaab.generator",
                "--seed", str(seed), "--n", str(n), "--month", "2026-08",
                "--out", str(data), "--truth", str(truth), "--quiet",
            ],
            f"clean-mode control at seed {seed}, n={n}",
        )
        out = base / "matches.json"
        doc = _matcher_and_score(data, truth, out, seed)
        cells, rates = doc["cells"], doc["rates"]  # type: ignore[index]
        # The control's **own** truth, read fresh. Reusing the loop's ``truth_doc`` here would
        # compare the clean run's cells against the last noisy run's credit count -- two
        # different files, and the mismatch would read as a control failure.
        control_credits = json.loads(
            (truth / "truth.json").read_text(encoding="utf-8")
        )["credits"]
        control_ignored = [
            str(v.get("credit_id"))
            for v in json.loads(out.read_text(encoding="utf-8"))["verdicts"]
            if v.get("outcome") == "IGNORED"
        ]
        if control_ignored:
            raise GateFailure(
                f"the clean-mode control at seed {seed}, n={n} set aside "
                f"{len(control_ignored)} row(s) as non-gateway (e.g. {control_ignored[:3]}) on a "
                f"file where every row is a gateway credit. Step 5's gate requires **both** "
                f"evidence tests to fail, and a clean credit names the counterparty, so this "
                f"means the gate is reading something other than what it claims to"
            )
        if rates["noise_recall"] is not None or cells["noise_correctly_ignored"]:
            raise GateFailure(
                f"the clean-mode control reports noise_recall={rates['noise_recall']} over "
                f"{cells['noise_correctly_ignored']} ignored row(s); with no non-gateway rows on "
                f"the file the rate has to be n/a rather than a number -- a 0.0 or a 1.0 here "
                f"would make the noisy runs' rate unattributable"
            )
        if rates["correctness"] != 1.0 or cells["correct"] != len(control_credits):
            raise GateFailure(
                f"the clean-mode control resolves {cells['correct']} of "
                f"{len(control_credits)} at correctness {rates['correctness']} -- the control is "
                f"what makes the noisy numbers attributable, so it has to be perfect"
            )
        print(
            f"    control  seed {seed}, n={n:<5} no flags: resolved={cells['correct']}"
            f"/{len(control_credits)} ignored=0 recall=n/a"
        )


#: The eleven flags ``--all-mess`` resolves to after Phase 8 step 9 -- every implemented flag
#: except ``dup_amounts``, which is mutually exclusive with four of the others. Spelled out rather
#: than read from ``MessFlags.composable()``: a gate that asked the config layer what to run would
#: agree with a broken reduction about a wrong flag set. The generator refuses any illegal
#: combination, so a drift between this tuple and ``composable()`` surfaces as a *failure* here.
PHASE8_FLAGS: tuple[str, ...] = (*PHASE7_FLAGS, "--fx", "--utr-patchy")


def _fx_and_mask_failure(
    cells: dict[str, int],
    fx_total: int,
    fx_resolved: list[str],
    masked_total: int,
    masked_ignored: list[str],
    fx_rate_gap: int,
    label: str,
) -> str | None:
    """What is wrong with a run's FX and masking outcomes, or ``None`` if it is healthy.

    Factored out for gates 13 and 14's reason: this gate's value is in what it refuses, so the
    predicate meets a known-good input and every known-bad one before a real run is read.

    **Each clause carries its own denominator, and three of the six exist only to refuse a
    vacuous pass.** ``WRONG_IGNORE == 0`` is trivially true on a file where the mask fired on
    nothing genuine, and Phase 8 step 6 measured exactly that: ``--utr-patchy --noise-rows``
    alone gets **zero** masked credits to the ignore gate on all three seeds, because only
    ``--reserve`` makes a credit differ from its settlement's declared net. So the population is
    asserted non-empty beside the outcome, never inferred from the flag being on.

    **``fx_resolved`` is driven off truth's ``fx_paise``, not off the verdict.** The plan asked
    for gate 13's shape -- "no resolved row carries an FX term" -- and that is not writable here:
    ``hisaab/common/verdict.Decomposition`` has **no** ``fx_paise`` field, deliberately
    (``generator/model.py:209`` -- a field the matcher could populate is a residual it could fit
    anything into). ``v["decomposition"].get("fx_paise", 0)`` therefore reads a key that can never
    exist, returns 0 on every row of every run, and would pass forever. Measured: 0 of ~629
    verdicts carry the key on any cell. The property ``model.py`` names as what makes the
    asymmetry safe is the one asserted instead -- *an FX-bearing credit is never RESOLVED* -- and
    it has a denominator: 15-16 such credits at n=200, 72-76 at n=1000.
    """
    if fx_total == 0:
        return (
            f"{label}: no credit carries a non-zero fx_paise, so the FX assertions below have no "
            f"subject -- 'none resolved' is vacuous on a file with no FX row. --fx is on, so this "
            f"is a generator regression rather than a permitted shape"
        )
    if fx_resolved:
        return (
            f"{label}: {len(fx_resolved)} credit(s) carrying an FX term were RESOLVED "
            f"(e.g. {fx_resolved[:3]}), out of {fx_total}. The rate movement hides inside a gross "
            f"the matcher reads as authoritative and is declared in no input file, so a resolved "
            f"FX row means the arithmetic closed on a term it cannot see -- and the matcher's own "
            f"Decomposition has no fx_paise to have priced it with"
        )
    if fx_rate_gap == 0:
        return (
            f"{label}: FX_RATE_GAP never fired across {fx_total} FX-bearing credit(s). It is in "
            f"ABSTENTION_REASONS, so a run where it never appears lets those rows pass the "
            f"honesty audit below on a code that was never exercised"
        )
    if masked_total == 0:
        return (
            f"{label}: no genuine credit lost its UTR tail, so WRONG_IGNORE == 0 is vacuous -- "
            f"the ignore conjunction is never asked to keep a masked genuine credit. Phase 8 "
            f"step 6 measured this exact shape on --utr-patchy --noise-rows without --reserve"
        )
    if masked_ignored:
        return (
            f"{label}: {len(masked_ignored)} masked genuine credit(s) were IGNORED "
            f"(e.g. {masked_ignored[:3]}), out of {masked_total}. Truth marks these resolvable, "
            f"so discarding one is money dropped from the books -- the failure Phase 7's "
            f"conjunction exists to prevent, and the one --utr-patchy is designed to provoke"
        )
    if cells["wrong_ignore"] != 0:
        return (
            f"{label}: wrong_ignore is {cells['wrong_ignore']} over {masked_total} masked genuine "
            f"credit(s). There is no acceptable non-zero value: a real credit set aside as "
            f"non-gateway is unrecoverable without a human noticing the shortfall"
        )
    return None


def _gate_15_self_check() -> None:
    """Prove ``_fx_and_mask_failure`` accepts a healthy run and rejects each shape it names.

    The healthy fixture is the measured seed-1 n=200 cell, so a pass here is attributable to a
    run this gate actually produces rather than to a number chosen to satisfy the predicate.
    Every bad case is checked by **wording**, because six clauses share one predicate and a
    truthy return would otherwise not say which one fired.
    """
    def cells(wrong_ignore: int = 0) -> dict[str, int]:
        return {"wrong_ignore": wrong_ignore}

    healthy = dict(
        cells=cells(), fx_total=16, fx_resolved=[], masked_total=19, masked_ignored=[],
        fx_rate_gap=34, label="probe",
    )
    if (got := _fx_and_mask_failure(**healthy)) is not None:  # type: ignore[arg-type]
        raise GateFailure(
            f"gate 15's predicate rejects the healthy run it was measured on, so every failure "
            f"it reports below would be unconditional and would prove nothing: {got}"
        )

    for override, want in (
        # The three vacuity refusals: each is a run where the *outcome* is perfect and the
        # population is empty, which is the shape this gate exists to refuse.
        ({"fx_total": 0}, "no subject"),
        ({"masked_total": 0}, "vacuous"),
        ({"fx_rate_gap": 0}, "never fired"),
        # The three real defects.
        ({"fx_resolved": ["C0007"]}, "were RESOLVED"),
        ({"masked_ignored": ["C0031"]}, "were IGNORED"),
        ({"cells": cells(wrong_ignore=1)}, "no acceptable non-zero value"),
    ):
        got = _fx_and_mask_failure(**{**healthy, **override})  # type: ignore[arg-type]
        if got is None or want not in got:
            raise GateFailure(
                f"gate 15's predicate failed to reject {override}: expected a complaint "
                f"containing {want!r}, got {got!r}"
            )


def gate_15_phase8(sizes: tuple[int, ...] = (200, 1000)) -> None:
    """Phase 8: the eleven-flag run -- foreign currency and a bank statement missing UTRs.

    The capstone. Every implemented flag except ``dup_amounts`` (mutually exclusive with four of
    the others), which is what ``--all-mess`` resolves to after step 9.

    **Coverage is deliberately NOT asserted at 1.0, and that is a correction to this phase's own
    plan.** Step 5 measured that ``--fx`` voids Tier 2's uniqueness inference whenever a foreign
    payment is in the candidate pool: the subset search can no longer argue a unique membership,
    so ~19% of credits abstain that would otherwise resolve. Measured coverage on this flag set is
    **56.11%-71.20%** across the six cells, with ``FX_RATE_GAP`` alone taking 137-148 rows at
    n=1000. A gate asserting 1.0 here would fail on correct code. What never bends is the pair
    this project refuses to average: **correctness 1.0 and zero wrong matches**, on every cell.

    What it asserts, per seed and size:

      * ``_fx_and_mask_failure``, above -- six clauses, three of which refuse a *vacuous* pass by
        asserting the population beside the outcome.
      * **Correctness 1.0 with zero wrong matches**, on eleven flags. Nothing before this gate
        scores a file where a term the matcher cannot see (the rate movement) hides inside a gross
        it reads as authoritative.
      * **No gateway row carries ``NO_CANDIDATE``** -- and this is *stronger* than the plan asked
        for. The plan permitted ``NO_CANDIDATE <= 1`` by the characterised identity gate 13 used;
        measured, gateway ``NO_CANDIDATE`` is **0** on all six cells, and all 35 such verdicts
        across the matrix are noise rows whose narration names a gateway counterparty -- Phase 7's
        conjunction correctly refusing to ignore them, after which the search honestly finds
        nothing. So the exemption is not inherited: the assertion is equality with zero.
      * **The partition, in both directions.** Every bank row is a gateway credit or a noise row,
        never neither and never both, and every one carries a verdict. Without it, "the
        ``NO_CANDIDATE`` rows are noise" would be an inference from *absence* from truth's
        ``credits`` -- which is not the same claim as membership in the noise set.
      * **Every unresolved gateway row abstains honestly.** Noise rows are excluded, for gate 14's
        reason: they are scored on their own axis, and letting ``IGNORED`` count as an abstention
        would have the easiest rows in the file vouch for the hardest.
      * **``noise_correctly_ignored == noise_strata["plainly_foreign"]``.** Gate 14's identity,
        re-run here because it is what rules out a ``plainly_foreign`` row hiding inside the
        ``NO_CANDIDATE`` population **without re-deriving per-row strata** -- which are
        deliberately off-disk, and which a gate must not reconstruct, since doing so would use the
        same two evidence tests the matcher uses and agree with a broken matcher by construction.

    **It honours ``--skip-slow``, and that is a decision by measurement rather than by imitation
    of its two predecessors.** Gates 13 and 14 ignore the flag because their subjects read
    literally zero at n=200 -- the reserve shortfall was invisible there, and
    ``AMBIGUOUS_MULTI_SUBSET`` does not occur at all. Every subject *this* gate adds is
    non-vacuous at n=200: 15-16 FX-bearing credits, 19 masked genuine credits, ``FX_RATE_GAP`` at
    12-34, the plainly-foreign identity at 3=3, gateway ``NO_CANDIDATE`` at 0. The n=1000 cells
    are kept in a full run because the eleven-flag matcher is the most expensive configuration in
    this suite (~10.2s per cell, against 95s for the whole suite before this gate) and a size that
    costs that much should not be paid twice for properties already covered at n=200.

    What a pass does **not** prove. That an FX row *could not* be resolved -- only that none was.
    The rate movement is declared in no input file and is not even bounded by one, so nothing but
    truth's own record could say otherwise, and that limit belongs in the write-up. Nor does it
    prove the ~19% Tier 2 abstention is minimal: it is the price of refusing to infer uniqueness
    from a pool containing a payment whose home-currency amount is stale, and a matcher that
    guessed there would score wrong matches rather than abstentions.
    """
    print(f"\ngate 15 -- Phase 8: all eleven flags, seeds {list(DEV_SEEDS)}")
    _gate_15_self_check()

    with tempfile.TemporaryDirectory(prefix="hisaab-phase8-") as tmp:
        root = Path(tmp)
        for seed in DEV_SEEDS:
            for n in sizes:
                base = root / f"s{seed}n{n}"
                data, truth = base / "data", base / "truth"
                _run(
                    [
                        sys.executable, "-m", "hisaab.generator",
                        "--seed", str(seed), "--n", str(n), "--month", "2026-08",
                        "--out", str(data), "--truth", str(truth), "--quiet", *PHASE8_FLAGS,
                    ],
                    f"generator with all Phase 8 flags at seed {seed}, n={n}",
                )
                out = base / "matches.json"
                doc = _matcher_and_score(
                    data, truth, out, seed, extra=["--window", str(MESS_WINDOW_DAYS)],
                )
                verdicts = json.loads(out.read_text(encoding="utf-8"))["verdicts"]
                truth_doc = json.loads((truth / "truth.json").read_text(encoding="utf-8"))
                manifest = json.loads(
                    (truth / "run_manifest.json").read_text(encoding="utf-8")
                )
                credits = truth_doc["credits"]
                cells, rates = doc["cells"], doc["rates"]  # type: ignore[index]

                # ``outcome``, subscripted rather than ``.get``-ed, and the vocabulary pinned.
                # A probe written for this gate read ``v.get("status")`` -- a key verdicts do not
                # carry -- so every comparison against "RESOLVED" was ``None == "RESOLVED"`` and
                # three of its measurements were arithmetic on a missing key that could only
                # come out clean. A missing key must raise here, and an unknown outcome value
                # must not silently bucket as "not resolved".
                outcome = {str(v["credit_id"]): str(v["outcome"]) for v in verdicts}
                reason = {str(v["credit_id"]): str(v.get("reason")) for v in verdicts}
                if unknown := sorted(set(outcome.values()) - {"RESOLVED", "EXCEPTION", "IGNORED"}):
                    raise GateFailure(
                        f"seed {seed}, n={n}: unknown outcome value(s) {unknown}. Every check "
                        f"below buckets on this field, so a value they do not know about would "
                        f"be counted as 'not resolved' without a word"
                    )

                gateway_ids = {str(c["credit_id"]) for c in credits}
                noise_ids = {str(i) for i in truth_doc["orphans"]["non_gateway_credit_ids"]}
                bank_ids = {r["row_id"] for r in _csv_rows(data / "bank_statement.csv")}

                # **The denominators, stated before anything is compared against them.** The five
                # checks that follow are emptiness tests on set differences, and every one of them
                # passes on an empty population: no bank row is in neither set when there are no
                # bank rows, and no gateway credit carries NO_CANDIDATE when truth lists no
                # credits. That is the failure mode this suite keeps finding -- a check that holds
                # because it measured nothing -- so the populations are asserted rather than
                # assumed from the flags being on.
                if not bank_ids or not gateway_ids:
                    raise GateFailure(
                        f"seed {seed}, n={n}: {len(bank_ids)} bank row(s) and "
                        f"{len(gateway_ids)} truth credit(s). Every partition and NO_CANDIDATE "
                        f"check below is an emptiness test that an empty population satisfies, "
                        f"so neither may be zero for their passes to mean anything"
                    )

                # --- the partition, both directions ------------------------------------------
                if neither := sorted(bank_ids - gateway_ids - noise_ids):
                    raise GateFailure(
                        f"seed {seed}, n={n}: {len(neither)} bank row(s) are neither a gateway "
                        f"credit nor a listed noise row (e.g. {neither[:3]}). Every check below "
                        f"splits on that membership, so a row in neither set is audited by none "
                        f"of them"
                    )
                if both := sorted(gateway_ids & noise_ids):
                    raise GateFailure(
                        f"seed {seed}, n={n}: {len(both)} row(s) are listed as BOTH a gateway "
                        f"credit and a noise row (e.g. {both[:3]}), so the noise exclusion below "
                        f"would silently excuse genuine credits from the honesty audit"
                    )
                # The reverse direction of the same pairing, and not decoration: the masking check
                # below indexes ``bank_by_id[cid]`` for every gateway credit, so a credit with no
                # bank row would surface as a KeyError traceback rather than a gate failure.
                if bodiless := sorted(gateway_ids - bank_ids):
                    raise GateFailure(
                        f"seed {seed}, n={n}: {len(bodiless)} truth credit(s) have no bank row "
                        f"(e.g. {bodiless[:3]}). Truth lists them as gateway money that reached "
                        f"the account, so the statement has to carry them"
                    )
                if missing := sorted(bank_ids - set(outcome)):
                    raise GateFailure(
                        f"seed {seed}, n={n}: {len(missing)} bank row(s) carry no verdict "
                        f"(e.g. {missing[:3]}) -- the matcher must answer every row, and a row "
                        f"with no verdict is invisible to every count in this gate"
                    )

                # --- the FX and masking axis, via the factored predicate ----------------------
                fx_ids = {
                    str(c["credit_id"]) for c in credits
                    if int(c["decomposition"].get("fx_paise", 0) or 0) != 0
                }
                bank_by_id = {r["row_id"]: r for r in _csv_rows(data / "bank_statement.csv")}
                # A genuine credit whose narration lost its four-digit tail. ``_tail_of`` is the
                # gates' own definition, shared rather than re-spelled: a second regex would be a
                # second definition of "still carries a UTR", and the two could disagree.
                masked = sorted(
                    cid for cid in gateway_ids
                    if _tail_of(bank_by_id[cid]["narration"]) is None
                )
                if problem := _fx_and_mask_failure(
                    cells,  # type: ignore[arg-type]
                    fx_total=len(fx_ids),
                    fx_resolved=sorted(c for c in fx_ids if outcome[c] == "RESOLVED"),
                    masked_total=len(masked),
                    masked_ignored=[c for c in masked if outcome[c] == "IGNORED"],
                    fx_rate_gap=sum(
                        1 for v in verdicts if str(v.get("reason")) == "FX_RATE_GAP"
                    ),
                    label=f"seed {seed}, n={n}",
                ):
                    raise GateFailure(problem)

                # --- correctness never bends --------------------------------------------------
                wrong = (
                    cells["wrong_match"] + cells["wrong_match_invented"]  # type: ignore[index]
                    + cells["lucky_guess"]  # type: ignore[index]
                )
                if wrong or rates["correctness"] != 1.0:  # type: ignore[index]
                    raise GateFailure(
                        f"seed {seed}, n={n}: correctness {rates['correctness']}, "  # type: ignore[index]
                        f"{wrong} wrong match(es). Coverage is permitted to fall on this flag "
                        f"set -- --fx voids Tier 2's uniqueness inference and those rows abstain "
                        f"-- but a wrong match is a wrong answer that looks like a result, which "
                        f"is the one outcome no flag excuses"
                    )
                # A collapse floor rather than a target: measured 56.11%-71.20% across six cells,
                # so 0.45 is below every observed value and still catches a matcher that stopped
                # resolving. Asserted because "coverage may fall" must not become "coverage may
                # be anything" -- the FX abstentions are a bounded, argued cost, not a licence.
                if float(rates["coverage"]) < 0.45:  # type: ignore[index, arg-type]
                    raise GateFailure(
                        f"seed {seed}, n={n}: coverage {rates['coverage']} is below the 0.45 "  # type: ignore[index]
                        f"collapse floor (measured range on this flag set: 56.11%-71.20%). The "
                        f"FX abstentions are an argued cost with a measured size, so a coverage "
                        f"far under that band is a regression rather than the flag working"
                    )

                # --- no gateway row honestly found nothing ------------------------------------
                # Equality with zero, stronger than the plan's "<= 1 by the characterised
                # identity". Measured: 0 on all six cells, and all 35 NO_CANDIDATE verdicts
                # across the matrix are noise rows naming a gateway counterparty in their
                # narration -- Phase 7's conjunction refusing to ignore them, after which the
                # search honestly finds nothing. Gate 13's exemption is deliberately not
                # inherited: a permission carried past the shape that needed it is decoration.
                if stranded := sorted(
                    cid for cid in gateway_ids if reason[cid] == "NO_CANDIDATE"
                ):
                    raise GateFailure(
                        f"seed {seed}, n={n}: {len(stranded)} gateway credit(s) carry "
                        f"NO_CANDIDATE (e.g. {stranded[:3]}). That code is outside "
                        f"ABSTENTION_REASONS on purpose -- admitting it would let every failed "
                        f"search score as an honest refusal -- and on this flag set it belongs "
                        f"to the noise strata, never to gateway money"
                    )

                # --- every unresolved gateway row abstains honestly ---------------------------
                dishonest = [
                    str(v["credit_id"]) for v in verdicts
                    if str(v["outcome"]) != "RESOLVED"
                    and str(v["credit_id"]) not in noise_ids
                    and str(v.get("reason")) not in ABSTENTION_REASONS
                ]
                if dishonest:
                    raise GateFailure(
                        f"seed {seed}, n={n}: {len(dishonest)} unresolved gateway row(s) carry a "
                        f"reason outside ABSTENTION_REASONS (e.g. {dishonest[:3]}). The noise "
                        f"rows are excluded on purpose -- they are scored on their own axis, and "
                        f"letting IGNORED count as an abstention would have the easiest rows in "
                        f"the file vouch for the hardest"
                    )

                # --- the composed plainly-foreign argument ------------------------------------
                # ``_noise_failure`` rather than a hand-rolled equality, which is what this was
                # first written as. Three reasons, and the first is the one that matters: gate
                # 14's predicate is **already self-checked**, so reusing it inherits a control
                # instead of adding a second uncontrolled assertion. It also separates the two
                # directions -- a shortfall means a plainly-foreign row was not set aside, a
                # surplus means a gateway-spelled row leaked into IGNORED, and only the second
                # becomes a wrong answer next phase -- where the merged form said neither. And it
                # keeps **one** definition of the identity: a private copy here could drift from
                # gate 14's and the two would then disagree about what a healthy noisy run is.
                strata: dict[str, int] = manifest["noise_strata"]
                if problem := _noise_failure(
                    cells, strata, f"seed {seed}, n={n}"  # type: ignore[arg-type]
                ):
                    raise GateFailure(
                        f"{problem}\n"
                        f"  This identity is also what rules out a plainly-foreign row hiding in "
                        f"the NO_CANDIDATE population without re-deriving per-row strata -- which "
                        f"a gate must not do, since it would use the matcher's own two evidence "
                        f"tests and agree with a broken matcher by construction."
                    )

                print(
                    f"    seed {seed}, n={n:<5} coverage {float(rates['coverage']):>7.2%}  "  # type: ignore[index, arg-type]
                    f"correctness {rates['correctness']}  wrong 0  "  # type: ignore[index]
                    f"fx {len(fx_ids):>3} (0 resolved)  masked {len(masked):>3} (0 ignored)  "
                    f"wrong_ignore 0"
                )


#: Row 34 of ASSUMPTIONS.md prices every reason code in prose -- ``10 (`NO_CANDIDATE`,
#: `REFUND_UNLINKED`)`` -- and this parses it back into a dict so gate 7 can compare the prose
#: against ``MINUTES_PER_EXCEPTION`` itself.
#:
#: The dict already asserts its own exhaustiveness over ``Reason``, so a *new* code cannot
#: silently price at zero. What nothing guarded was the **transcription**, and the gap was
#: live: Phase 8 found row 34 listing 11 of the 13 codes, missing ``AMBIGUOUS_ADJUSTMENT`` (10)
#: and ``MEMBERSHIP_UNDECLARED`` (20) -- the joint-most-expensive code in the table. Two
#: sources of truth for one number, and the prose one is what a judge actually reads.
ROW_34_MINUTES_RE = re.compile(r"(\d+)(?:\s*min[a-z]*)?\s*\(((?:`[A-Z_]+`(?:,\s*)?)+)\)")


#: Reads ``MINUTES_PER_EXCEPTION`` out of the scorer **in a subprocess**, the way every other
#: scorer number in this file arrives.
#:
#: The obvious spelling -- ``from hisaab.scoring.metrics import MINUTES_PER_EXCEPTION`` at the
#: top of this file -- was written first and gate 5 rejected it, correctly. ``check_isolation``
#: treats importing *anything* under ``hisaab.scoring`` as reaching the answer key, because a
#: module that can import the package can import the loader. Putting the acceptance harness on
#: the ``TRUTH_READERS`` allowlist to price one constant would trade a structural guarantee for
#: a convenience: this file is the *verifier*, and it currently cannot reach truth in-process at
#: all. So the constant crosses the same boundary the scored metrics already cross.
#:
#: Do not "simplify" this back into an import. The subprocess is the point.
_MINUTES_SNIPPET = (
    "import json;"
    "from hisaab.scoring.metrics import MINUTES_PER_EXCEPTION as m;"
    "print(json.dumps({r.name: v for r, v in m.items()}))"
)


def _priced_minutes() -> dict[str, int]:
    """``{reason code: minutes}`` as ``metrics.MINUTES_PER_EXCEPTION`` declares it."""
    out = _run([sys.executable, "-c", _MINUTES_SNIPPET], "reading MINUTES_PER_EXCEPTION")
    try:
        table = json.loads(out.splitlines()[0])
    except (IndexError, json.JSONDecodeError) as e:
        raise GateFailure(
            f"could not read MINUTES_PER_EXCEPTION out of the scorer: {out!r} ({e})"
        )
    if not table:
        raise GateFailure(
            "MINUTES_PER_EXCEPTION came back empty, so comparing row 34 against it would "
            "compare two empty tables and pass. The scorer's own assertion should make this "
            "impossible; reaching it means the read is wrong, not that the table is."
        )
    return table


def _row_34_minutes(text: str) -> dict[str, int]:
    """``{reason code: minutes}`` as ASSUMPTIONS.md row 34 states it, in prose."""
    rows = [ln for ln in text.splitlines() if ln.startswith("| 34 |")]
    if len(rows) != 1:
        raise GateFailure(
            f"ASSUMPTIONS.md has {len(rows)} rows numbered 34, expected exactly one. The "
            f"effort-estimate comparison reads that row, so it cannot run at all."
        )
    out: dict[str, int] = {}
    for m in ROW_34_MINUTES_RE.finditer(rows[0]):
        for code in re.findall(r"`([A-Z_]+)`", m.group(2)):
            out[code] = int(m.group(1))
    return out


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

    # --- row 34's prose must price every reason code exactly as the scorer does ----------
    stated = _row_34_minutes(text)
    priced = _priced_minutes()

    # The vacuity guard, first: if reformatting row 34 stops the regex matching, every
    # comparison below is between an empty dict and itself and would pass forever.
    if not stated:
        raise GateFailure(
            "gate 7 parsed 0 reason codes out of ASSUMPTIONS.md row 34, so the comparison "
            "against MINUTES_PER_EXCEPTION measured nothing. Either the row was reformatted "
            "away from the `<minutes> (`CODE`, `CODE`)` shape ROW_34_MINUTES_RE expects, or "
            "the codes were dropped. Both are failures -- an empty parse must never read as "
            "agreement."
        )
    # And the control: the comparison must be able to fail. A code deleted from the parsed
    # copy has to be reported, or the check cannot distinguish agreement from blindness.
    _sabotaged = dict(stated)
    _sabotaged.pop(sorted(_sabotaged)[0])
    if _sabotaged == priced:
        raise GateFailure(
            "gate 7's own control failed: removing a code from row 34's parsed table still "
            "compared equal to MINUTES_PER_EXCEPTION, so this check cannot detect a missing "
            "code and its pass proves nothing."
        )

    if stated != priced:
        unstated = {c: priced[c] for c in sorted(set(priced) - set(stated))}
        unknown = sorted(set(stated) - set(priced))
        disagree = {
            c: (stated[c], priced[c])
            for c in sorted(set(stated) & set(priced))
            if stated[c] != priced[c]
        }
        raise GateFailure(
            f"ASSUMPTIONS.md row 34 and metrics.MINUTES_PER_EXCEPTION disagree about the "
            f"effort estimate.\n"
            f"  priced in code but not stated in row 34: {unstated or '{}'}\n"
            f"  stated in row 34 but not a real code   : {unknown or '[]'}\n"
            f"  stated with different minutes           : {disagree or '{}'}\n"
            f"The dict asserts its own exhaustiveness over Reason, so a new code cannot price "
            f"at zero -- but nothing guarded this transcription, and row 34 is the copy a "
            f"judge reads. Phase 8 found it listing 11 of 13 codes, omitting the "
            f"joint-most-expensive one. Update whichever is wrong; do not let them drift."
        )
    print(f"    row 34 prices all {len(priced)} reason codes exactly as the scorer does")
    print(f"    {len(text.splitlines())} lines at {path.relative_to(ROOT)}")


def _duration_minutes(text: str) -> int:
    """Read ``report.duration``'s output back to an integer.

    The gate compares the two ROI totals itself rather than trusting the percentage beside
    them, which is the whole reason this parser exists: the inverted claim survived eight
    phases because no assertion ever subtracted the two printed numbers.
    """
    import re

    t = text.strip()
    if t.endswith(" min") and " h " not in t:
        return int(t[: -len(" min")])
    m = re.fullmatch(r"(\d+) h (\d+) min", t)
    if not m:
        raise GateFailure(f"cannot read a duration from {text!r}")
    return int(m.group(1)) * 60 + int(m.group(2))


def _metric_line(block: str, label: str) -> str:
    """The value on one line of the metric block, without its parenthesised note."""
    for line in block.splitlines():
        if line.startswith(label):
            return line[len(label):].strip().split("(")[0].strip()
    raise GateFailure(f"no {label!r} line in the metric block:\n{block}")


def _bank_amounts(data: Path) -> dict[str, int]:
    """``row_id -> amount_paise``, read straight from the CSV.

    Deliberately **not** through ``matcher.load``, which is what triage itself uses. A gate
    that validated the join with the same reader the join is built on could not notice the
    reader being wrong -- it would only notice the two of them disagreeing, which they never
    would. Six lines of ``csv`` is the independent second opinion.
    """
    import csv as _csv

    with (data / "bank_statement.csv").open("r", encoding="utf-8", newline="") as f:
        reader = _csv.DictReader(f)
        return {row["row_id"]: int(row["amount_paise"]) for row in reader}


def _triage(matches: Path, data: Path) -> tuple[dict[str, object], str]:
    """Run the triage CLI. Returns (line 1 as JSON, the text block below it)."""
    stdout = _run(
        [
            sys.executable, "-m", "hisaab.triage",
            "--matches", str(matches), "--data", str(data),
        ],
        f"triage on {matches.name}",
    )
    lines = stdout.splitlines()
    if not lines:
        raise GateFailure("triage printed nothing")
    try:
        doc = json.loads(lines[0])
    except json.JSONDecodeError as e:
        raise GateFailure(f"line 1 of triage's stdout is not JSON: {lines[0]!r} ({e})")
    return doc, "\n".join(lines[1:])


def gate_16_triage(sizes: tuple[int, ...] = (60, 200)) -> None:
    """Phase 9: the exception queue is complete, correctly valued, ranked, and honest about ROI.

    Six properties, and the order matters -- the control comes first, because every check
    below it would also pass on a queue that silently dropped rows:

      * **The empty queue, first.** Clean mode resolves every row, so triage must print an
        empty queue and exit 0. Without this, "no row appears twice" and "every value matches
        its bank row" are both satisfied by a tool that emits nothing at all.
      * **Every queued row exactly once, and no resolved row.** The partition is against
        ``matches.json`` itself: exceptions plus dismissals, no more and no less.
      * **Every total is the sum of its parts.** Group value against its own credits, group
        minutes against ``rows x minutes_per_row``, and the totals block against the groups.
        Three sums that could each be computed independently, so a disagreement localises.
      * **Every value is its own bank row's**, checked against the CSV read directly rather
        than through the loader triage uses.
      * **Genuinely ranked by money, and the ranking is not vacuous.** Descending value is
        asserted on every cell; separately, at least one cell across the sweep must order two
        groups differently than effort would. Otherwise "ranked by value" is untestable --
        every ordering agrees when the two keys never disagree.
      * **The ROI claim points the right way.** The gate reads both totals out of the metric
        block, subtracts them itself, and requires the printed claim to agree. This is the
        assertion whose absence let the report state a saving on all six measured cells while
        the tool's own total was 2-3x larger than doing the batch by hand.

    And two refusals, because a queue that cannot say "these inputs are not from the same
    run" would present numbers computed across two months as a month's work.
    """
    print(f"\ngate 16 -- the exception queue on seeds {list(DEV_SEEDS)} x sizes {list(sizes)}")
    inversion_seen: str | None = None

    with tempfile.TemporaryDirectory(prefix="hisaab-triage-") as tmp:
        root = Path(tmp)

        # --- the control: clean mode leaves nothing for a human --------------------------
        clean = root / "clean"
        data, truth = clean / "data", clean / "truth"
        _run(
            [
                sys.executable, "-m", "hisaab.generator",
                "--seed", "1", "--n", "60", "--month", "2026-08",
                "--out", str(data), "--truth", str(truth), "--quiet",
            ],
            "generator, clean mode",
        )
        matches = clean / "matches.json"
        _run(
            [
                sys.executable, "-m", "hisaab.matcher",
                "--data", str(data), "--out", str(matches),
                "--seed", "1", "--month", "2026-08", "--quiet",
            ],
            "matcher, clean mode",
        )
        doc, text = _triage(matches, data)
        totals = doc["totals"]  # type: ignore[index]
        if doc["groups"] != [] or totals["rows"] != 0:  # type: ignore[index]
            raise GateFailure(
                f"clean mode resolves every row, so the queue must be empty -- got "
                f"{totals['rows']} row(s) in {len(doc['groups'])} group(s).\n"  # type: ignore[index,arg-type]
                f"Either the matcher regressed or the queue is inventing work."
            )
        if "empty" not in text:
            raise GateFailure(f"an empty queue must say so in the text block, got:\n{text}")
        print("    clean mode        empty queue, exit 0 -- nothing needs a human")

        # --- the real thing: an eleven-flag run at every seed and size -------------------
        for seed in DEV_SEEDS:
            for n in sizes:
                base = root / f"s{seed}n{n}"
                data, truth = base / "data", base / "truth"
                _run(
                    [
                        sys.executable, "-m", "hisaab.generator",
                        "--seed", str(seed), "--n", str(n), "--month", "2026-08",
                        "--all-mess", "--out", str(data), "--truth", str(truth), "--quiet",
                    ],
                    f"generator --all-mess at seed {seed}, n={n}",
                )
                matches = base / "matches.json"
                # ``--window 1`` for the reason gate 10 measured: ``--all-mess`` includes the
                # posting lag, and at the default window 0 nothing resolves at all. A queue
                # holding every row would pass most checks here while testing nothing.
                metrics = _matcher_and_score(
                    data, truth, matches, seed, extra=["--window", str(MESS_WINDOW_DAYS)],
                )
                doc, text = _triage(matches, data)
                groups = doc["groups"]  # type: ignore[assignment]
                totals = doc["totals"]  # type: ignore[index]
                where = f"seed {seed}, n={n}"

                # --- the partition, against matches.json ----------------------------------
                verdicts = json.loads(matches.read_text(encoding="utf-8"))["verdicts"]
                queued = {v["credit_id"] for v in verdicts if v["outcome"] != "RESOLVED"}
                resolved = {v["credit_id"] for v in verdicts if v["outcome"] == "RESOLVED"}
                listed = [c["credit_id"] for g in groups for c in g["credits"]]  # type: ignore[index,union-attr]
                if len(listed) != len(set(listed)):
                    dupes = sorted({c for c in listed if listed.count(c) > 1})
                    raise GateFailure(
                        f"{where}: {dupes} appear in more than one group. A duplicated row is "
                        f"counted twice in its group's money and its minutes."
                    )
                if set(listed) != queued:
                    raise GateFailure(
                        f"{where}: the queue does not match the verdicts.\n"
                        f"  missing from the queue: {sorted(queued - set(listed))[:8]}\n"
                        f"  in the queue but resolved: {sorted(set(listed) & resolved)[:8]}\n"
                        f"A row missing from every group is a row that quietly stops being "
                        f"somebody's job."
                    )

                # --- three sums, each computed independently ------------------------------
                bank = _bank_amounts(data)
                for g in groups:  # type: ignore[union-attr]
                    own = sum(c["value_paise"] for c in g["credits"])
                    if g["value_paise"] != own:
                        raise GateFailure(
                            f"{where}, group {g['cause']}: claims {g['value_paise']}p but its "
                            f"rows sum to {own}p"
                        )
                    if g["estimated_minutes"] != g["rows"] * g["minutes_per_row"]:
                        raise GateFailure(
                            f"{where}, group {g['cause']}: {g['estimated_minutes']} min for "
                            f"{g['rows']} rows at {g['minutes_per_row']} min each"
                        )
                    if g["rows"] != len(g["credits"]):
                        raise GateFailure(
                            f"{where}, group {g['cause']}: says {g['rows']} rows, lists "
                            f"{len(g['credits'])}"
                        )
                    if not g["action"]:
                        raise GateFailure(
                            f"{where}, group {g['cause']}: no next action. A heading with no "
                            f"action reads as 'nothing can be done about these'."
                        )
                    # --- every value is its own bank row's --------------------------------
                    for c in g["credits"]:
                        if c["credit_id"] not in bank:
                            raise GateFailure(
                                f"{where}: {c['credit_id']} is queued but is not a bank row"
                            )
                        if c["value_paise"] != bank[c["credit_id"]]:
                            raise GateFailure(
                                f"{where}: {c['credit_id']} is ranked at {c['value_paise']}p "
                                f"but the bank statement says {bank[c['credit_id']]}p -- the "
                                f"join is attaching the wrong money to a row"
                            )
                for key, got in (
                    ("groups", len(groups)),  # type: ignore[arg-type]
                    ("rows", len(listed)),
                    ("value_paise", sum(g["value_paise"] for g in groups)),  # type: ignore[union-attr]
                    ("estimated_minutes", sum(g["estimated_minutes"] for g in groups)),  # type: ignore[union-attr]
                ):
                    if totals[key] != got:  # type: ignore[index]
                        raise GateFailure(
                            f"{where}: totals.{key} is {totals[key]} but the groups sum to "  # type: ignore[index]
                            f"{got}"
                        )

                # --- genuinely ranked by money ------------------------------------------
                values = [g["value_paise"] for g in groups]  # type: ignore[union-attr]
                if values != sorted(values, reverse=True):
                    raise GateFailure(
                        f"{where}: groups are not ordered by money at risk: {values}"
                    )
                for a, b in zip(groups, groups[1:]):  # type: ignore[index,arg-type]
                    # Value already leads; this is the pair that proves it *led* rather than
                    # merely agreed with effort. ``a`` is worth more than ``b`` by the check
                    # above, so ``a`` costing fewer minutes is a pair the two keys order
                    # differently.
                    if a["estimated_minutes"] < b["estimated_minutes"]:
                        inversion_seen = (
                            f"{where}: {a['cause']} ({a['estimated_minutes']} min) outranks "
                            f"{b['cause']} ({b['estimated_minutes']} min) on money"
                        )

                # --- the ROI claim, subtracted here rather than trusted -------------------
                block = _run(
                    [
                        sys.executable, "-m", "hisaab.scoring",
                        "--matches", str(matches), "--truth", str(truth),
                    ],
                    f"scorer text block at seed {seed}, n={n}",
                )
                tool = _duration_minutes(_metric_line(block, "Est. human time to clear"))
                by_hand = _duration_minutes(_metric_line(block, "Same batch by hand"))
                claim = _metric_line(block, "Time saved")
                # **Three states, not two.** Reading this line as "COSTS MORE or a percentage"
                # was wrong the moment the wrong-match branch existed: a run that books a wrong
                # match withholds the claim entirely, and on such a run the minute totals may
                # still favour the tool -- so a two-valued reading would fail this gate on a
                # block that is behaving correctly. The states are asserted separately because
                # each one is a different promise about the same two numbers.
                wrong = (
                    metrics["cells"]["wrong_match"]  # type: ignore[index]
                    + metrics["cells"]["wrong_match_invented"]  # type: ignore[index]
                    + metrics["cells"]["lucky_guess"]  # type: ignore[index]
                )
                withheld = "not claimable" in claim
                costs_more = "COSTS MORE" in claim
                if withheld != bool(wrong):
                    raise GateFailure(
                        f"{where}: the block says {claim!r} with {wrong} wrong match(es). The "
                        f"claim must be withheld exactly when a wrong match exists -- a wrong "
                        f"match raises no exception, so it costs the queue nothing and is "
                        f"invisible to this comparison. The `zip` fixture printed 'Time saved "
                        f"100.0%' at 35% correctness before that branch existed."
                    )
                if not withheld and costs_more != (tool >= by_hand):
                    raise GateFailure(
                        f"{where}: the ROI claim contradicts its own numbers -- the block says "
                        f"{claim!r} while the tool costs {tool} min against {by_hand} min by "
                        f"hand. This is the defect the four-line block was written to make "
                        f"impossible: a claim nothing subtracted."
                    )
                if withheld and "%" in claim:
                    raise GateFailure(
                        f"{where}: the claim is withheld but still prints a percentage "
                        f"({claim!r}). A number beside a caveat is the number a reader takes "
                        f"away."
                    )
                # The tool's own total must also be the queue's, since they price the same
                # work from the same table by two routes.
                if tool != totals["estimated_minutes"]:  # type: ignore[index]
                    raise GateFailure(
                        f"{where}: the scorer charges {tool} min but the queue charges "
                        f"{totals['estimated_minutes']} min for the same rows -- one of them "  # type: ignore[index]
                        f"is pricing work the other cannot see"
                    )
                # Three states here too. The sweep's cells all hold 100% correctness today, so
                # the withheld case never prints -- but a gate whose own summary line could
                # report "saves 88 min" for a run that refused to claim a saving would be
                # narrating the opposite of what it just asserted.
                if withheld:
                    verdict = f"not claimable ({wrong} wrong)"
                elif costs_more:
                    verdict = "COSTS MORE"
                else:
                    verdict = f"saves {by_hand - tool} min"
                print(
                    f"    seed {seed}, n={n:<4} {totals['rows']:>3} row(s) in "  # type: ignore[index]
                    f"{len(groups):>2} group(s)   "  # type: ignore[arg-type]
                    f"{totals['estimated_minutes']:>4} min vs {by_hand:>4} by hand   "  # type: ignore[index]
                    f"{verdict}   exceptions {metrics['exceptions']['count']}"  # type: ignore[index]
                )

        if inversion_seen is None:
            raise GateFailure(
                "no cell in this sweep ordered two groups differently than effort would, so "
                "'ranked by money at risk' is untested here -- every ordering agrees when the "
                "two keys never disagree. Widen the sweep rather than deleting this check."
            )
        print(f"    ranking is load-bearing: {inversion_seen}")

        # --- and it refuses two runs pretending to be one -------------------------------
        # Both directions. Seed 1's verdicts against seed 2's data: some credit ids exist in
        # both files with different amounts, which is precisely the mismatch that would
        # produce a plausible, wrong queue rather than an error.
        a, b = root / f"s{DEV_SEEDS[0]}n{sizes[0]}", root / f"s{DEV_SEEDS[1]}n{sizes[0]}"
        for matches_dir, data_dir, label in (
            (a, b, "seed 1 verdicts against seed 2 data"),
            (b, a, "seed 2 verdicts against seed 1 data"),
        ):
            proc = subprocess.run(
                [
                    sys.executable, "-m", "hisaab.triage",
                    "--matches", str(matches_dir / "matches.json"), "--data", str(data_dir / "data"),
                ],
                cwd=ROOT, capture_output=True, text=True,
                env={**_env(), "PYTHONUTF8": "1"},
            )
            if proc.returncode != 1:
                raise GateFailure(
                    f"{label}: exit {proc.returncode}, expected 1. Two runs' files must not "
                    f"produce a queue -- every number in it would be about the wrong month.\n"
                    f"{proc.stdout[:400]}"
                )
            if "REFUSING TO BUILD A QUEUE" not in proc.stderr:
                raise GateFailure(f"{label}: refused without saying so:\n{proc.stderr[:400]}")
        print("    mismatched runs refused in both directions (exit 1)")

        # --- and the mismatch no id check can see ---------------------------------------
        # Both refusals above fire on the id sets: seeds 1 and 2 produce 48 and 50 bank rows at
        # n=60, so one file simply has rows the other lacks. **The dangerous case is two runs of
        # the same size**, where the ids coincide exactly, both id checks pass, and the queue
        # comes out plausible and wrong. Rather than hunting for two seeds that happen to agree
        # on row count, this changes one amount by one paisa in a copy of the data -- the
        # smallest possible same-size divergence, and the one a byte-level diff would find but
        # an id comparison never will.
        import shutil

        skewed = root / "skewed"
        shutil.copytree(a / "data", skewed)
        bank_csv = skewed / "bank_statement.csv"
        rows = bank_csv.read_text(encoding="utf-8").splitlines()
        if len(rows) < 2:
            raise GateFailure("the copied bank statement has no rows to skew")
        fields = rows[1].split(",")
        amount_at = rows[0].split(",").index("amount_paise")
        before = int(fields[amount_at])
        fields[amount_at] = str(before + 1)
        rows[1] = ",".join(fields)
        bank_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

        proc = subprocess.run(
            [
                sys.executable, "-m", "hisaab.triage",
                "--matches", str(a / "matches.json"), "--data", str(skewed),
            ],
            cwd=ROOT, capture_output=True, text=True,
            env={**_env(), "PYTHONUTF8": "1"},
        )
        if proc.returncode != 1:
            raise GateFailure(
                f"a bank amount changed by 1 paisa produced exit {proc.returncode}, expected 1. "
                f"Every credit id still matches, so this is the mismatch the id checks cannot "
                f"see -- and the queue it built ranks {before + 1}p as though the matcher had "
                f"judged it.\n{proc.stdout[:400]}"
            )
        if "wrong month" not in proc.stderr:
            raise GateFailure(
                f"refused, but not by comparing the stated amount -- so the same-size case is "
                f"still uncovered:\n{proc.stderr[:400]}"
            )
        print(f"    one amount off by 1p refused (ids all match, {before}p vs {before + 1}p)")


#: Gate 17's recorded double. A hand-built object rather than a mock framework, for the
#: reason ``client.py``'s own double states: this asserts the exact shapes the client reads
#: off a response, and writing them out makes them reviewable.
#:
#: It answers each group with citations drawn from **that group's own universe**, so the
#: clean case is checked against real fixture data rather than against something invented to
#: satisfy the checker. In ``fabricate`` mode every citation is replaced with a figure and an
#: id that appear in no row.
class _ExplainDouble:
    def __init__(self, groups: list[dict], *, fabricate: bool = False) -> None:
        from hisaab.explain import prompt as prompt_mod

        self._universe = {
            id(g): tuple(sorted(u)[:3] for u in prompt_mod.cited_universe(g))
            for g in groups
        }
        self._groups = groups
        self._fabricate = fabricate
        self.messages = self
        self.calls = 0
        self.systems_seen: list[str] = []

    def create(self, **kwargs: object) -> object:
        from hisaab.common.reasons import Reason
        from hisaab.triage.hint import HINTS

        text = kwargs["messages"][0]["content"]  # type: ignore[index]
        self.systems_seen.append(
            "".join(b["text"] for b in kwargs["system"])  # type: ignore[index,union-attr]
        )
        group = next(
            (g for g in self._groups
             if f"Rows in this group: {g['rows']}" in text and g["cause"] in text),
            self._groups[0],
        )
        ids, amounts = self._universe[id(group)]
        reason = group.get("reason")
        hint = HINTS.get(Reason(reason)) if reason else None
        if self._fabricate:
            ids, amounts = ("setl_999999",), (123456789,)
        payload = {
            "summary": f"{group['rows']} rows share one cause.",
            "why_unresolved": "The input files do not carry the evidence needed.",
            "next_step": hint.action if hint else "Confirm with whoever knows the account.",
            "cited_row_ids": list(ids),
            "cited_amounts_paise": list(amounts),
        }
        self.calls += 1
        # First call writes the cache, later calls read it -- so run()'s step-8 branch is
        # exercised in both directions rather than only the one this environment produces.
        return type("R", (), {
            "content": [type("B", (), {"type": "text", "text": json.dumps(payload)})()],
            "stop_reason": "end_turn",
            "parsed_output": None,
            "usage": type("U", (), {
                "input_tokens": 1770 + len(text) // 4,
                "output_tokens": 180,
                "cache_creation_input_tokens": 1770 if self.calls == 1 else 0,
                "cache_read_input_tokens": 0 if self.calls == 1 else 1770,
            })(),
        })()


#: The meta-path blocker gate 17 uses to prove the four ``hisaab.explain`` modules work with
#: ``anthropic`` absent. Stronger than uninstalling: it refuses the import however the module
#: is reached, and it proves it blocks before testing anything.
_NO_SDK_HARNESS = '''
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name == "anthropic" or name.startswith("anthropic."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None

sys.meta_path.insert(0, _Block())
try:
    import anthropic
except ModuleNotFoundError:
    pass
else:
    raise SystemExit("BLOCKER FAILED: anthropic imported anyway")

import runpy
runpy.run_module(sys.argv[1], run_name="__main__")
'''


def gate_17_explain(full: bool = True) -> None:
    """Phase 10: the LLM layer explains the queue, and every claim it makes is checked.

    **No live call, and that is a design property rather than a convenience.** The whole
    argument for putting a model near a reconciliation is that its output is verified, so the
    verification has to be exercised by the gate -- and a gate that needed a key and a
    network would be skipped on a clean checkout, which is where it matters most. Every check
    here runs against the frozen fixture with a recorded double.

    Nine properties:

      * **The fixture is what it says it is.** ``explain_fixture.py --check`` rebuilds it from
        the generator and the matcher and requires it byte-identical. Skipped under
        ``--skip-slow`` (it costs ~11s); the committed file's shape is still validated.
      * **The clean case passes, first.** Every check below would also pass on a pipeline that
        refused everything, so the vacuity control leads: 8 groups explained, every citation
        verified, exit 0, artifact written.
      * **A fabricated citation is fatal.** The same 8 groups with every id and amount
        replaced by figures in no row must exit 1. This is the assertion the phase rests on,
        and without it "the model's claims are checked" is a sentence rather than a property.
      * **The cached prefix is byte-identical across every request in a run.** Read off what
        the double actually received, not from ``system_blocks()`` twice -- prompt caching
        matches an exact byte prefix, so the claim is about what was *sent*.
      * **The dry-run's arithmetic is internally consistent.** It reports the uncached input
        total and the prefix re-sent within it, and the second must not exceed the first.
        Written because the first version printed 12,390 tokens as a component of 11,536 -- a
        part larger than its whole, in the output, the same way ASSUMPTIONS.md #38's ROI claim
        printed backwards for eight phases: two figures side by side with nothing relating
        them.
      * **Every module under ``hisaab/explain/`` that self-checks works with ``anthropic``
        absent.** The count is derived from the package, not written here as a number --
        this docstring said "the four modules" until a fifth (``cluster``) and then a sixth
        (``qa``) landed, each time silently untested by this line while the code below
        still ran correctly. The core is stdlib-only and the SDK is an optional extra;
        ``anthropic`` is installed in the shell this was built in, so gate 0 passing here
        proves nothing about a clean checkout. Each module's self-check is re-run with the
        import blocked at the meta-path level, and ``_client()`` is required to refuse with
        install instructions rather than fail obscurely.
      * **A resolved row's Q&A is checked by arithmetic, not containment alone.** An
        exception row has no decomposition to verify a claim against; a resolved row does,
        and ``qa.ask`` is the one place in this package a claim is checked against a
        computation rather than merely matched against a string. A correct answer must
        verify and an answer with an invented term in an otherwise-closing sum must not.
      * **The hint comparison actually runs, and its number means what it says.** Every
        fixture group with a declared code carries a hint; ``compare_to_hint`` must return
        ``has_hint`` for each and must score 1.0 on the hint's own text and near-zero on an
        unrelated sentence -- the two points ``cli.py``'s own docstring uses to say a
        reproduction score is not an agreement score. Nothing else in this gate calls
        ``compare_to_hint`` at all, so without this it ships unexercised.
      * **Clustering leaves Phase 9's partition untouched.** ``cluster.sample`` picks which
        rows a request sees; it must not change which rows belong to which reason-coded
        group. Checked against the same fixture gate 17 already loads: every credit id
        clustering touches is still in the group ``groups_from_fixture`` put it in.

    What this does **not** prove: that the model says anything true. The double supplies the
    text, so this gate covers the plumbing, the citation check and the exit codes -- never
    output quality. And the citation check itself proves every figure is *real*, not that each
    is used correctly; ``verify.py``'s docstring lists that limit and two others.
    """
    from hisaab.explain import EXPLAIN_SCHEMA_VERSION
    from hisaab.explain import cli as cli_mod

    print("\ngate 17 -- the LLM layer, against the frozen fixture with no live call")

    fixture = ROOT / "fixtures" / "explain" / "fixture.json"
    if not fixture.exists():
        raise GateFailure(
            f"{fixture.relative_to(ROOT).as_posix()} is missing. It is the recorded input this "
            f"whole gate runs on -- build it with `python tools/explain_fixture.py`."
        )

    # --- 1. the fixture is what it says it is ----------------------------------------
    if full:
        _run([sys.executable, str(ROOT / "tools" / "explain_fixture.py"), "--check"],
             "explain fixture rebuild")
        print("    fixture rebuilds byte-identical from the generator and the matcher")
    else:
        print("    fixture rebuild SKIPPED (--skip-slow, ~11s); shape still checked below")

    groups = cli_mod.groups_from_fixture(fixture)
    codes = {g["reason"] for g in groups if g.get("reason")}
    dismissals = [g for g in groups if g.get("reason") is None]
    if len(groups) < 8 or len(codes) < 7:
        raise GateFailure(
            f"the fixture holds {len(groups)} group(s) covering {len(codes)} reason code(s); "
            f"expected at least 8 and 7. A single seed x size cell tops out at 7 of 13 codes, "
            f"which is why the fixture uses two flag sets -- if this shrank, one cell was lost."
        )
    if not dismissals:
        raise GateFailure(
            "the fixture has no dismissal group, so the uncoded path is untested. That path "
            "printed 'Reason code: None' into the prompt and 'citations: None:' into the "
            "report until this fixture caught both."
        )

    # ``run`` prints a full explanation for every group -- ~100 lines per call, three calls
    # here. Captured rather than printed so this gate reads like the sixteen before it, and
    # handed to the failure message when there is one: the output matters exactly when
    # something is wrong, which is the opposite of printing it always.
    # Both streams: ``run`` prints explanations to stdout and its failure list to stderr, so
    # capturing only stdout left the fabrication report printing twice into the suite -- the
    # exact noise this exists to remove. Kept SEPARATE rather than merged, because the phrase
    # "NOT FOUND in the input" appears on both, and counting a merged buffer would report 16
    # findings for 8 groups.
    def _quiet(**kwargs: object) -> tuple[int, str, str]:
        out_buf, err_buf = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            code = cli_mod.run(groups, model="recorded-double", **kwargs)
        return code, out_buf.getvalue(), err_buf.getvalue()

    # --- 2. the clean case, first: the vacuity control for everything below -----------
    with tempfile.TemporaryDirectory(prefix="hisaab-explain-") as tmp:
        out = Path(tmp) / "explanations.json"
        double = _ExplainDouble(groups)
        code, said, said_err = _quiet(out=out, strict=True, client=double)
        if code != 0:
            raise GateFailure(
                f"the clean case exited {code}. Every citation the double makes is drawn from "
                f"its own group's rows, so a failure here is the checker refusing valid input "
                f"-- and it would make the fabrication check below pass for the wrong reason."
                f"\n\n{said_err.strip() or said[-1200:]}"
            )
        # Nothing may reach stderr on a clean run: that stream is where run() reports
        # unverifiable citations, so anything on it here means a finding was raised and then
        # not reflected in the exit code.
        if said_err.strip():
            raise GateFailure(
                f"the clean case exited 0 but wrote to stderr, so something was flagged and "
                f"then not counted:\n{said_err[-800:]}"
            )
        if double.calls != len(groups):
            raise GateFailure(
                f"{double.calls} request(s) for {len(groups)} group(s) -- a group was skipped "
                f"silently, which is how a queue stops being somebody's job."
            )

        doc = json.loads(out.read_text(encoding="utf-8"))
        if doc["schema_version"] != EXPLAIN_SCHEMA_VERSION:
            raise GateFailure(f"artifact schema v{doc['schema_version']} != code's")
        if doc["citations_clean"] != len(groups) or doc["explained"] != len(groups):
            raise GateFailure(
                f"{doc['citations_clean']} of {doc['explained']} explanation(s) verified "
                f"clean, expected {len(groups)}/{len(groups)}"
            )
        checked = sum(e["citation_check"]["checked"] for e in doc["explanations"])
        if checked < len(groups):
            raise GateFailure(
                f"only {checked} citation(s) checked across {len(groups)} group(s). The check "
                f"is passing because it is looking at almost nothing."
            )
        # No absolute path may reach the artifact: it is committed, and a temp dir in it would
        # differ on every machine.
        if tmp.replace("\\", "/") in json.dumps(doc):
            raise GateFailure("the artifact carries an absolute temp path")
        print(f"    {len(groups)} group(s) explained, {checked} citation(s) verified, exit 0")

        # --- 3. the load-bearing assertion: a fabrication must be fatal --------------
        bad = _ExplainDouble(groups, fabricate=True)
        code_bad, said_bad, said_bad_err = _quiet(out=None, strict=True, client=bad)
        if code_bad != 1:
            raise GateFailure(
                f"every citation in every group was replaced with an id and an amount that "
                f"appear in no row, and the run exited {code_bad} instead of 1.\n"
                f"This is the property the whole phase rests on: a model near a "
                f"reconciliation is defensible only because its claims are checked. If this "
                f"passes, the citation check is decoration.\n\n{said_bad[-1200:]}"
            )
        # **Exit 1 alone does not prove the citation check fired.** Any ExplainError exits 1
        # too, so a client that simply broke would produce this same code and look like a
        # working checker. So the fabricated values must be named in the output, and every
        # group must be accounted for -- the same reasoning as gate 5's marker phrases, where
        # a mutant caught by the wrong assertion is indistinguishable from a working check.
        if "setl_999999" not in said_bad or "123456789" not in said_bad:
            raise GateFailure(
                f"the run exited 1, but neither fabricated value is named in its output -- so "
                f"something else failed and the citation check may not have run at all.\n"
                f"{said_bad[-1200:]}"
            )
        if said_bad.count("NOT FOUND in the input") != len(groups):
            raise GateFailure(
                f"{said_bad.count('NOT FOUND in the input')} of {len(groups)} group(s) "
                f"reported a fabrication. Every group's citations were replaced, so every "
                f"group must refuse them -- a subset means some groups are not being checked."
            )
        # And --permissive must still REPORT the finding while exiting 0, or the flag is a way
        # to make fabrications invisible rather than non-fatal.
        # The findings must also reach stderr, which is where a person or a CI log sees them.
        # Checked separately from stdout because the two streams carry the same phrase for
        # different reasons: stdout reports each group as it goes, stderr summarises at the end.
        if said_bad_err.count("NOT FOUND in the input") != len(groups):
            raise GateFailure(
                f"stderr named {said_bad_err.count('NOT FOUND in the input')} fabrication(s) "
                f"for {len(groups)} group(s). A finding printed only to stdout is one a CI "
                f"log filtering for errors would miss.\n{said_bad_err[-800:]}"
            )
        code_perm, said_perm, said_perm_err = _quiet(
            out=None, strict=False, client=_ExplainDouble(groups, fabricate=True)
        )
        if code_perm != 0:
            raise GateFailure("--permissive did not exit 0 on a fabricated citation")
        if "NOT FOUND in the input" not in said_perm:
            raise GateFailure(
                "--permissive exited 0 without reporting the fabrication, which makes it a "
                "way to hide a bad citation rather than a way to continue past one."
            )
        print(f"    every citation fabricated -> exit 1, all {len(groups)} groups naming it; "
              f"--permissive reports it and exits 0")

        # --- 4. the cached prefix, read off what was actually sent -------------------
        prefixes = set(double.systems_seen)
        if len(prefixes) != 1:
            raise GateFailure(
                f"the system prefix differed across {len(prefixes)} of {double.calls} "
                f"requests, so prompt caching can never hit. Caching matches an exact byte "
                f"prefix -- this is checked against what the client SENT, because "
                f"system_blocks() agreeing with itself does not prove the request did."
            )
        prefix_len = len(next(iter(prefixes)))
        print(f"    the {prefix_len:,}-char prefix was byte-identical on all "
              f"{double.calls} requests")

    # --- 5. the dry-run's arithmetic must be internally consistent -------------------
    dry = _run(
        [sys.executable, "-m", "hisaab.explain", "--fixture", "--dry-run"],
        "explain --dry-run",
    )
    uncached = re.search(r"input if nothing caches: ~([\d,]+) tokens", dry)
    fewer = re.search(r"~([\d,]+) fewer", dry)
    if not (uncached and fewer):
        raise GateFailure(f"--dry-run no longer reports its token arithmetic:\n{dry[-600:]}")
    total, saved = (int(m.group(1).replace(",", "")) for m in (uncached, fewer))
    if not 0 <= saved <= total:
        raise GateFailure(
            f"the dry-run says {saved:,} tokens of a {total:,}-token total are re-sent "
            f"prefix -- a part larger than its whole.\n"
            f"The first version of this printed exactly that, because it divided the prefix "
            f"in once while sending it 8 times. Nothing caught it but a reader, which is how "
            f"the ROI claim survived eight phases."
        )
    if "nothing will be sent" not in dry:
        raise GateFailure("--dry-run must say plainly that it sends nothing")
    print(f"    --dry-run consistent: {saved:,} re-sent prefix tokens of {total:,} total")

    # --- 6. the four modules must work with the optional extra absent ----------------
    with tempfile.TemporaryDirectory(prefix="hisaab-nosdk-") as tmp:
        harness = Path(tmp) / "_no_sdk.py"
        # The harness puts its OWN directory on sys.path, so the repo root is linked in
        # rather than assumed -- running a script by path does not put the cwd on sys.path,
        # and getting that wrong reports five import failures that are all the harness's.
        harness.write_text(
            _NO_SDK_HARNESS.replace(
                "Path(__file__).resolve().parent", repr(str(ROOT))
            ),
            encoding="utf-8",
        )
        control = Path(tmp) / "_no_sdk_control.py"
        control.write_text(
            "from hisaab.explain import client\n"
            "try:\n"
            "    client._client()\n"
            "except client.ExplainError as e:\n"
            "    assert 'optional extra' in str(e), f'wrong message: {e}'\n"
            "    print('refused with install instructions')\n"
            "else:\n"
            "    raise SystemExit('built a client with no SDK installed')\n",
            encoding="utf-8",
        )
        # **Derived from the package, not listed.** This was a hardcoded tuple of four names
        # until `cluster.py` was added, at which point the gate went on reporting "all 4
        # modules self-check with anthropic blocked" while silently skipping the fifth -- a
        # list that prints its own count and is wrong, which is the `fees`/`tier2` omission
        # from Phase 6 recurring in the gate written to prevent that class of thing.
        # `cli.py` and `__main__.py` are excluded by having no `_self_check`: running them as
        # __main__ would invoke the CLI and attempt a model call.
        modules = tuple(
            f"hisaab.explain.{p.stem}"
            for p in sorted((ROOT / "hisaab" / "explain").glob("*.py"))
            if "def _self_check" in p.read_text(encoding="utf-8")
        )
        if len(modules) < 6:
            raise GateFailure(
                f"only {len(modules)} module(s) under hisaab/explain/ define a _self_check: "
                f"{list(modules)}. A module in this package with no self-check is a module "
                f"gate 0 cannot run."
            )
        # And gate 0 must run exactly these. Two lists that can disagree is how a module with
        # a working self-check ends up never being run by the suite that reports a clean sweep.
        in_gate_0 = {m for m in SELF_CHECK_MODULES if m.startswith("hisaab.explain.")}
        if in_gate_0 != set(modules):
            raise GateFailure(
                f"gate 0 and hisaab/explain/ disagree about which modules self-check.\n"
                f"  on disk but not in SELF_CHECK_MODULES: {sorted(set(modules) - in_gate_0)}\n"
                f"  in SELF_CHECK_MODULES but not on disk: {sorted(in_gate_0 - set(modules))}\n"
                f"A module missing from gate 0 has a self-check nothing runs, and gate 0 "
                f"prints a clean sweep while skipping it -- exactly what happened to "
                f"matcher.fees and matcher.tier2 for three phases."
            )
        for name in (*modules, "_no_sdk_control"):
            proc = subprocess.run(
                [sys.executable, str(harness), name],
                cwd=str(tmp) if name == "_no_sdk_control" else str(ROOT),
                capture_output=True, text=True, env={**_env(), "PYTHONUTF8": "1"},
            )
            if proc.returncode != 0:
                raise GateFailure(
                    f"{name} fails with `anthropic` absent (exit {proc.returncode}).\n"
                    f"The SDK is an optional extra and the core is stdlib-only -- that is "
                    f"the claim check 8 and pyproject.toml both make. It is installed in "
                    f"this shell, so nothing else here would notice.\n"
                    f"{(proc.stderr or proc.stdout)[-700:]}"
                )
        print(f"    all {len(modules)} modules self-check with `anthropic` blocked; "
              f"_client() refuses with install instructions")

    # --- 7. the hint comparison runs, and its number means what cli.py claims ---------
    #
    # Nothing above calls compare_to_hint: the clean/fabricated runs in step 2/3 check the
    # citation path, not this one. Without this block the feature ships in cli.py's own
    # docstring and nowhere else -- exercised by hand once, never again.
    from hisaab.common.reasons import Reason as _Reason
    from hisaab.triage.hint import HINTS as _HINTS

    coded = [g for g in groups if g.get("reason")]
    if not coded:
        raise GateFailure("no fixture group carries a reason code, so the hint comparison "
                           "cannot be exercised at all")
    for group in coded:
        reason = _Reason(group["reason"])
        hint = _HINTS[reason]
        exact = cli_mod.compare_to_hint(group, hint.action)
        if not exact["has_hint"] or exact["hint_terms_reproduced"] != 1.0:
            raise GateFailure(
                f"{reason}: comparing the hint's own text against itself scored "
                f"{exact.get('hint_terms_reproduced')}, expected 1.0. compare_to_hint's "
                f"docstring measured this exact case at 1.00 on all 7 coded groups; a lower "
                f"score here means the term-overlap arithmetic broke."
            )
        unrelated = cli_mod.compare_to_hint(group, "nothing to do; ignore these rows")
        if unrelated["hint_terms_reproduced"] > 0.10:
            raise GateFailure(
                f"{reason}: a flat contradiction scored "
                f"{unrelated['hint_terms_reproduced']}, expected near zero. "
                f"compare_to_hint's docstring measured contradictions at 0.00 on all 7 "
                f"groups; this is the vacuity control for the metric."
            )
    print(f"    hint comparison: {len(coded)} coded group(s), each scores 1.0 on its own "
          f"hint's text and ~0 on a contradiction")

    # --- 8. clustering leaves the fixture's own partition untouched -------------------
    #
    # cluster.sample picks which rows a request SEES; it must never change which reason-coded
    # group a row belongs to. Checked against groups_from_fixture's own output, which is what
    # step 1 already verified matches the committed file -- so this is Phase 9's partition,
    # not a second copy of it.
    from hisaab.explain import cluster as cluster_mod
    from hisaab.explain import prompt as prompt_mod_for_gate17

    for group in groups:
        own_ids = {c["credit_id"] for c in group.get("credits", ())}
        sampled = cluster_mod.sample(group, prompt_mod_for_gate17.ROWS_PER_GROUP)
        sampled_ids = {r["credit_id"] for r in sampled}
        if not sampled_ids <= own_ids:
            raise GateFailure(
                f"{group.get('reason') or group['cause']}: cluster.sample returned "
                f"credit(s) {sorted(sampled_ids - own_ids)} not in this group's own "
                f"{len(own_ids)} credit(s) -- sampling must not move a row between "
                f"reason-coded groups."
            )
    print(f"    clustering: every sampled row across {len(groups)} group(s) stayed inside "
          f"its own group")

    # --- 9. a resolved row's Q&A is checked by arithmetic, not containment alone ------
    from hisaab.explain import qa as qa_mod

    resolved = cli_mod.resolved_rows_from_fixture(fixture)
    if not resolved:
        raise GateFailure(
            "the fixture's resolved_sample is empty, so qa.ask has nothing to be checked "
            "against here. tools/explain_fixture.py freezes RESOLVED rows with a full "
            "fee+GST+TDS deduction for exactly this."
        )
    sample_row = resolved[0]
    decomp = sample_row["decomposition"]
    terms = [{"label": "gross", "paise": decomp["gross_paise"]}]
    for field in qa_mod.DEDUCTION_FIELDS:
        if decomp.get(field):
            terms.append({"label": field.removesuffix("_paise"), "paise": -decomp[field]})
    good_payload = {
        "answer": "The gateway withheld its fee, GST on that fee and TDS.",
        "cited_row_ids": [sample_row["credit_id"]],
        "cited_amounts_paise": [decomp["gross_paise"], decomp["expected_credit_paise"]],
        "arithmetic": {"terms": terms, "total_paise": decomp["expected_credit_paise"]},
    }
    good_findings = qa_mod.verify_answer(sample_row, good_payload)
    if good_findings:
        raise GateFailure(
            f"a correct answer over {sample_row['credit_id']}'s own decomposition was "
            f"refused: {good_findings}"
        )
    bad_payload = {
        **good_payload,
        "arithmetic": {
            "terms": [{"label": "gross", "paise": decomp["gross_paise"]},
                      {"label": "fees", "paise": -321}],
            "total_paise": decomp["expected_credit_paise"],
        },
    }
    bad_findings = qa_mod.verify_answer(sample_row, bad_payload)
    if not bad_findings or not any("came from nowhere" in f for f in bad_findings):
        raise GateFailure(
            f"a term (-321p) invented for no figure in {sample_row['credit_id']}'s row, "
            f"inside a sum that still closes, was not refused: {bad_findings}. This is the "
            f"failure containment alone cannot see -- every id and amount elsewhere in the "
            f"payload is real."
        )
    print(f"    qa arithmetic check: {sample_row['credit_id']}'s own decomposition verifies; "
          f"an invented term in an otherwise-closing sum is refused")


def gate_18_report(sizes: tuple[int, ...] = (60, 200)) -> None:
    """Phase 11: the HTML report renders all five sections, reproducibly, with no truth leak.

    Five properties:

      * **Clean mode renders a complete page with an empty queue and no Q&A.** The control:
        every check below would also pass on a page that rendered nothing at all, so the
        vacuity case comes first -- all five section headers present, the match definition
        quoted, the queue's own absence note, the Q&A section's own absence note.
      * **A messy run renders a non-empty queue and at least one RESOLVED row's full
        decomposition**, re-parsed from the rendered *text* and checked to sum to the stated
        credit amount -- not trusted because ``Verdict.__post_init__`` already checked it
        once in memory, since the render step is a new place a transcription error could be
        introduced (`.plan/phase11.md` §3).
      * **Two renders of the same input are byte-identical outside the one timestamp line.**
        ``strip_generated_at`` locates that line by its own marker; the rest of the page
        must match exactly.
      * **The truth-vocabulary grep passes, scoped to two documented, measured exceptions.**
        ``truth.json`` and ``true_`` must not appear at all. Bare ``resolvable`` may appear
        only inside ``metric_block()``'s own shipped "Missed" caption -- a fact this phase's
        own build discovered empirically (``hisaab/report/html.py``'s self-check comment):
        a literal "ban resolvable" gate fails on every real report, not just a leaking one,
        because that caption is static Phase 9 prose this phase quotes verbatim rather than
        rewords. ``tier`` is not checked at all -- it is matcher-side vocabulary
        (``hisaab/report/matched.py``'s own docstring), a different leaf of the same
        reasoning check 8a's ``hisaab/explain`` exemption uses.
      * **Two runs cannot be rendered as one.** Metrics from a different run than
        matches.json must refuse with exit 1, the same provenance check
        ``verdict_io.reconcile`` already makes for the matcher and scorer.
    """
    from hisaab.report.html import GENERATED_AT_MARKER, strip_generated_at
    from hisaab.report.matched import MAX_ROWS_LISTED as REPORT_MAX_ROWS_LISTED

    print(f"\ngate 18 -- the HTML report on sizes {list(sizes)}")

    with tempfile.TemporaryDirectory(prefix="hisaab-report-") as tmp:
        root = Path(tmp)

        def _pipeline(base: Path, *, all_mess: bool, n: int, seed: int = 1) -> tuple[Path, Path, Path, Path]:
            data, truth = base / "data", base / "truth"
            gen_argv = [
                sys.executable, "-m", "hisaab.generator",
                "--seed", str(seed), "--n", str(n), "--month", "2026-08",
                "--out", str(data), "--truth", str(truth), "--quiet",
            ]
            if all_mess:
                gen_argv.append("--all-mess")
            _run(gen_argv, f"generator{' --all-mess' if all_mess else ''}, n={n}, seed={seed}")

            matches = base / "matches.json"
            match_argv = [
                sys.executable, "-m", "hisaab.matcher",
                "--data", str(data), "--out", str(matches),
                "--seed", str(seed), "--month", "2026-08", "--quiet",
            ]
            if all_mess:
                match_argv += ["--window", str(MESS_WINDOW_DAYS)]
            _run(match_argv, f"matcher{' --all-mess' if all_mess else ''}, n={n}")

            metrics_path = base / "metrics.json"
            _run(
                [
                    sys.executable, "-m", "hisaab.scoring",
                    "--matches", str(matches), "--truth", str(truth),
                    "--out", str(metrics_path), "--quiet",
                ],
                f"scorer, n={n}",
            )
            triage_path = base / "triage.json"
            _run(
                [
                    sys.executable, "-m", "hisaab.triage",
                    "--matches", str(matches), "--data", str(data),
                    "--out", str(triage_path), "--quiet",
                ],
                f"triage, n={n}",
            )
            return matches, metrics_path, triage_path, truth

        def _render(matches: Path, metrics: Path, triage: Path, out: Path) -> str:
            _run(
                [
                    sys.executable, "-m", "hisaab.report",
                    "--matches", str(matches), "--metrics", str(metrics),
                    "--triage", str(triage), "--out", str(out), "--quiet",
                ],
                f"report -> {out.name}",
            )
            return out.read_text(encoding="utf-8")

        # --- the control: clean mode, no queue, no Q&A ------------------------------------
        clean = root / "clean"
        c_matches, c_metrics, c_triage, _ = _pipeline(clean, all_mess=False, n=60)
        page = _render(c_matches, c_metrics, c_triage, clean / "report.html")
        for section in ("Header", "Metric block", "Exception queue", "Matched records", "Q&amp;A"):
            if section not in page:
                raise GateFailure(f"clean report is missing the {section!r} section")
        if "set equality" not in page:
            raise GateFailure("clean report does not quote the match definition")
        if "Exception queue: empty" not in page:
            raise GateFailure("clean mode resolves every row, so the queue must render empty")
        if "Q&amp;A: none" not in page:
            raise GateFailure("no --ask was ever run, so the Q&A section must say so")
        print("    clean mode        all 5 sections, empty queue, no Q&A -- exit 0")

        # --- the real thing: a messy run with a non-empty queue and RESOLVED rows --------
        for n in sizes:
            base = root / f"mess-n{n}"
            matches, metrics_path, triage_path, truth = _pipeline(base, all_mess=True, n=n)
            page = _render(matches, metrics_path, triage_path, base / "report.html")

            verdicts = json.loads(matches.read_text(encoding="utf-8"))["verdicts"]
            resolved = [v for v in verdicts if v["outcome"] == "RESOLVED"]
            unresolved = [v for v in verdicts if v["outcome"] != "RESOLVED"]
            if not unresolved:
                raise GateFailure(
                    f"n={n}: --all-mess produced no exception/ignored row -- the queue check "
                    f"below would be vacuous"
                )
            if "Exception queue: empty" in page:
                raise GateFailure(
                    f"n={n}: --all-mess produced {len(unresolved)} unresolved row(s), but the "
                    f"report renders an empty queue"
                )

            # --- re-parse a rendered decomposition from the page's own text, and sum it ---
            # Matched records truncates at MAX_ROWS_LISTED, in matches.json's own order --
            # so the sample must come from that same prefix, or "not found" would mean
            # "truncated" rather than "missing".
            listed_first = verdicts[:REPORT_MAX_ROWS_LISTED]
            sample = next((v for v in listed_first if v["outcome"] == "RESOLVED"), None)
            if sample is None:
                raise GateFailure(
                    f"n={n}: none of the first {REPORT_MAX_ROWS_LISTED} verdicts (the ones "
                    f"Matched records actually renders) is RESOLVED, so there is no "
                    f"decomposition on the page to check"
                )
            credit_id = sample["credit_id"]
            row_start = page.find(f"{credit_id:<8} RESOLVED")
            if row_start == -1:
                raise GateFailure(f"n={n}: {credit_id} (RESOLVED) is not rendered in the page at all")
            row_end = page.find("\n", row_start)
            row_line = page[row_start:row_end if row_end != -1 else len(page)]

            def _to_paise(sign: str, val: str) -> int:
                p = round(float(val.replace(",", "")) * 100)
                return -p if sign == "-" else p

            terms = re.findall(
                r"(gross|fee|GST|TDS|refunds|reserve) (-?)₹([\d,]+\.\d{2})", row_line
            )
            # ``html.escape`` turns the literal "->" into "-&gt;" -- the page went through
            # that escape before this gate ever reads it, so the pattern has to match what
            # is actually on the page, not what matched.py wrote before rendering.
            expected_m = re.search(r"-&gt; expected (-?)₹([\d,]+\.\d{2})", row_line)
            if not terms or not expected_m:
                raise GateFailure(
                    f"n={n}: {credit_id}'s rendered row has no decomposition bracket:\n{row_line}"
                )
            gross = None
            deductions = 0
            for label, sign, val in terms:
                p = _to_paise(sign, val)
                if label == "gross":
                    gross = p
                else:
                    deductions += p
            if gross is None:
                raise GateFailure(
                    f"n={n}: {credit_id}'s rendered decomposition has no gross term:\n{row_line}"
                )
            expected = _to_paise(expected_m.group(1), expected_m.group(2))
            if gross - deductions != expected:
                raise GateFailure(
                    f"n={n}: {credit_id}'s rendered decomposition does not sum: gross {gross} "
                    f"minus terms {deductions} = {gross - deductions}, but the row states "
                    f"expected {expected}. This is the arithmetic Verdict.__post_init__ already "
                    f"checked once in memory -- a mismatch here means the render step, not the "
                    f"verdict, introduced a transcription error.\n{row_line}"
                )
            print(
                f"    n={n:<4} queue non-empty ({len(unresolved)} row(s)), "
                f"{credit_id}'s rendered decomposition sums exactly"
            )

            # --- truth-vocabulary grep, scoped to the two measured exceptions ------------
            lowered = page.lower()
            if re.search(r"\btruth\.json\b", lowered):
                raise GateFailure(f"n={n}: 'truth.json' leaked into the rendered page")
            if re.search(r"\btrue_", lowered):
                raise GateFailure(f"n={n}: 'true_*' leaked into the rendered page")
            bare_resolvable = len(re.findall(r"\bresolvable\b", page, re.IGNORECASE))
            caption_count = page.count("resolvable, but it abstained")
            if bare_resolvable != caption_count:
                raise GateFailure(
                    f"n={n}: 'resolvable' appears {bare_resolvable} time(s) but only "
                    f"{caption_count} of them are metric_block()'s own shipped Missed-line "
                    f"caption -- something else is leaking truth-side vocabulary."
                )

            # --- reproducibility: two renders differ only in the generated-at line ------
            page2 = _render(matches, metrics_path, triage_path, base / "report2.html")
            if strip_generated_at(page) != strip_generated_at(page2):
                raise GateFailure(
                    f"n={n}: two renders of the same input differ outside the "
                    f"{GENERATED_AT_MARKER!r} line"
                )
        print("    truth-vocabulary grep clean (scoped); two renders byte-identical outside "
              "the timestamp")

        # --- and it refuses two runs pretending to be one --------------------------------
        # Seed 2, not seed 1: every pipeline above ran at seed 1, so a metrics document from
        # that same seed would agree with c_matches's provenance by accident and this check
        # would pass on a run that was never actually refused.
        other = root / "other-seed"
        _, other_metrics, _, _ = _pipeline(other, all_mess=False, n=60, seed=2)
        proc = subprocess.run(
            [
                sys.executable, "-m", "hisaab.report",
                "--matches", str(c_matches), "--metrics", str(other_metrics),
                "--triage", str(c_triage), "--out", str(root / "bad.html"),
            ],
            cwd=ROOT, capture_output=True, text=True, env={**_env(), "PYTHONUTF8": "1"},
        )
        if proc.returncode != 1:
            raise GateFailure(
                f"clean matches.json against a --all-mess metrics document exited "
                f"{proc.returncode}, expected 1. Rendering one run's verdicts beside another "
                f"run's score would produce a plausible report about a run that never "
                f"happened.\n{proc.stdout[:400]}\n{proc.stderr[:400]}"
            )
        if "REFUSING TO RENDER" not in proc.stderr:
            raise GateFailure(f"refused without saying so:\n{proc.stderr[:400]}")
        print("    mismatched runs refused (exit 1)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run every acceptance gate (Phases 1-10).")
    p.add_argument("--skip-slow", action="store_true",
                   help="skip the n=200 sweeps in gates 3, 6, 9, 10, 11, 12, 16, and gate 17's fixture rebuild. Gates 13 "
                        "and 14 ignore this flag: gate 13's wrong-match assertion and gate "
                        "14's ambiguity and pool-cap assertions are all invisible at n=200")
    args = p.parse_args(argv)

    print("Acceptance -- generator + scoring harness + matcher + exception queue\n" + "=" * 62)
    gates = [
        gate_0_self_checks,
        lambda: gate_3_invariants_across_seeds((12, 60) if args.skip_slow else (12, 60, 200)),
        gate_4_and_leak_audit,
        gate_5_isolation,
        gates_1_2_6_reproducibility,
        gate_7_assumptions,
        gate_8_fixtures,
        lambda: gate_9_matcher((60,) if args.skip_slow else (60, 200)),
        lambda: gate_10_mess((60,) if args.skip_slow else (60, 200)),
        lambda: gate_11_planted((60,) if args.skip_slow else (60, 200)),
        lambda: gate_12_tier_mix((60,) if args.skip_slow else (60, 200)),
        # **n=1000 even under --skip-slow, and this is the one gate that ignores the flag.**
        # Not a preference: the wrong-match defect this gate was written to catch is invisible
        # at n=200. Before the Tier 2 refund fix, seeds 1 and 2 at n=1000 each resolved two
        # credits to the wrong payment set (correctness 0.9962) while the *same seeds at n=200
        # read 1.0000 with zero wrong matches* -- the coincidental-subset rate scales with the
        # candidate pool, so the small size cannot see it. A --skip-slow run that dropped
        # n=1000 would report Phase 6 green while blind to its only correctness failure.
        # Measured cost of keeping it: 2.2s -> 23.5s, and the slow half is the matcher rather
        # than the generator (n=1000 generates in ~0.5s).
        lambda: gate_13_phase6((200, 1000)),
        # n=1000 under --skip-slow for the same reason as gate 13, measured separately:
        # AMBIGUOUS_MULTI_SUBSET does not occur at all at n=200 on any dev seed (0, against
        # 21-34 at n=1000), so the small size cannot show that a noisy file still abstains
        # honestly where the subset search is genuinely ambiguous -- and n=200 never presents
        # a Tier 2 pool above 64, so it cannot notice a regression in the cap Phase 7 raised.
        lambda: gate_14_phase7((200, 1000)),
        # **Gate 15 HONOURS --skip-slow, and that breaks the pattern its two predecessors set.**
        # Decided by measurement, which is what `.plan/phase8.md` step 10 asked for rather than
        # copying the exemption a third time. Gates 13 and 14 ignore the flag because their
        # subjects read literally zero at n=200 -- the reserve shortfall was invisible there, and
        # AMBIGUOUS_MULTI_SUBSET does not occur at all. Every subject gate 15 adds is already
        # non-vacuous at n=200: 15-16 FX-bearing credits, 19 masked genuine credits, FX_RATE_GAP
        # at 12-34, the plainly-foreign identity at 3=3, gateway NO_CANDIDATE at 0. And the cost
        # is the highest in this suite -- the eleven-flag matcher runs ~10.2s per n=1000 cell
        # against 95s for the whole suite before this gate, so three of them are ~32s of the
        # ~127s total. A size that expensive should not be paid twice for properties n=200
        # already covers.
        lambda: gate_15_phase8((200,) if args.skip_slow else (200, 1000)),
        # Honours --skip-slow, like gate 15 and unlike gates 13 and 14. Every property this
        # gate reads is non-vacuous at n=60: an --all-mess run there already produces several
        # groups, a dismissal group, and the value/effort inversion the ranking check needs.
        # Nothing here is a rate that only separates at scale.
        lambda: gate_16_triage((60,) if args.skip_slow else (60, 200)),
        # Honours --skip-slow, and only for the fixture rebuild: that step re-runs the
        # generator and the matcher on two cells (~11s) to prove the committed fixture is
        # reproducible. Everything else in this gate is milliseconds against a recorded file
        # and a double, so all six properties still run under the flag -- including the
        # fabrication check, which is the one the phase rests on. Nothing here needs a
        # network, a key, or the optional extra installed.
        lambda: gate_17_explain(full=not args.skip_slow),
        # Honours --skip-slow: n=200 adds nothing gate 18 needs that n=60 does not already
        # give it. The queue is non-empty and carries a RESOLVED row inside the truncated
        # prefix at both sizes; a bigger pool does not change what this gate is checking.
        lambda: gate_18_report((60,) if args.skip_slow else (60, 200)),
    ]
    try:
        for gate in gates:
            gate()
    except GateFailure as e:
        print(f"\n{'=' * 62}\nACCEPTANCE FAILED\n\n{e}", file=sys.stderr)
        return 1

    print("\n" + "=" * 62)
    print("all eighteen gates pass -- Phases 1 through 11 are complete")
    print("\nPhase 9 turned the scored run into a work queue -- grouped by cause, ranked by")
    print("money at risk, priced per group, with a next action and a named missing input each.")
    print("It reads matches.json and data/ only: hisaab/triage is on MATCHER_PACKAGES, so the")
    print("same static check that keeps the matcher away from the answer key keeps the queue")
    print("away from it too. An operator can run this on their own month.")
    print("\n  And it found that the ROI claim had been printing backwards since Phase 2. The")
    print("  metric block showed the tool's minutes beside a by-hand total and never subtracted")
    print("  them -- on all six measured cells the by-hand figure was SMALLER, so the report was")
    print("  quietly claiming a saving while the tool cost an operator 2-3x more time than")
    print("  ignoring it. Eight phases and fifteen gates missed it because no assertion anywhere")
    print("  put the two numbers on opposite sides of a comparison. Gate 16 now does, and reads")
    print("  the claim back out of the block to check it agrees with its own arithmetic.")
    print("\n  The fix was not a bigger number. The baseline was one flat 2 min/row, which prices")
    print("  a chased exception the same as a tick-off; it now splits into 2 min on sight and 15")
    print("  min chased, and dismissals are charged 3 min each where they had been priced since")
    print("  Phase 2 and never billed. Break-even sits at 13.34 min against the assumed 15 -- a")
    print("  1.66-minute margin, printed beside the claim so a reader can see how much room it")
    print("  has rather than taking the percentage on trust.")
    print("\nClean mode still resolves at 100/100/0 (gate 9), and the mess dial is now fully")
    print("built: 12 of 13 flags implemented, eleven of them composing in one run (gate 15).")
    print("Across every flag set in this suite the matcher holds 100% correctness with 0 wrong")
    print("matches while proving its arithmetic per row against truth's own six-term")
    print("decomposition -- term by term, never on the total, since a fee too high and a GST")
    print("too low close the same gap.")
    print("\nTwo things Phase 4 settled that had been open, and both were surprises:")
    print("\n  The window is no longer untested. Gate 9's summary used to say +/-1000 days")
    print("  scored the same; gate 10 now pins --window 0 at 0% coverage under the posting")
    print("  lag while --fees alone still scores 100% there. So the requirement lives in the")
    print("  1-business-day posting lag, NOT in the T+2 settlement cycle -- T+2 shifts")
    print("  settled_on and value_date together and is invisible to this join.")
    print("\n  Coverage is 199/200 on seed 3 at n=200, and that is the right answer. Two")
    print("  settlements genuinely share a net; once the window admits the posting lag both")
    print("  are candidates for one credit, and the inputs cannot separate them. The true")
    print("  one sits at +1bd (the posting lag) and the decoy at +0bd (same-day), so")
    print("  `nearest date wins` would pick the decoy -- and since the lag is constant, it")
    print("  would be wrong every time it fired, not occasionally. The abstention is kept")
    print("  and the tie-break stays retired.")
    print("\nAlso corrected in Phase 4: two published fee rates. Netbanking is not cheaper")
    print("than cards (190 -> 200 bps), and UPI is not free -- zero MDR is not zero fee, as")
    print("Razorpay's 2% platform fee still applies on the standard PG rail (0 -> 200 bps).")
    print("The zero-rated rail moved to POS UPI, which the pricing page actually verifies.")
    print("That cut the share of rows settling at their gross from 36% to ~6%, so --fees is")
    print("a materially sharper test than the flag alone suggests.")
    print("\nPhase 4b gave correct_abstention a denominator. For three phases that cell")
    print("printed 0/0 while carrying this project's central claim -- that the exception list")
    print("is measured rather than asserted -- because every row in every scored run was")
    print("resolvable. Gate 11 plants two genuinely indistinguishable pairs and reads 4/4 on")
    print("all three dev seeds at both n=60 and n=200, with 0 lucky guesses.")
    print("\n  And the plant nearly did not work. Before it was built, a tail-only strategy")
    print("  -- reading no date and no amount, joining the four digits in the bank narration")
    print("  straight onto the utr column of settlements.csv -- was measured resolving 60/60,")
    print("  200/200 and 1000/1000 credits correctly, on every dev seed, clean and under both")
    print("  wedges alike, because tails are drawn without replacement. So a pair colliding")
    print("  on (date, amount) while keeping distinct UTRs is still separable, by a strategy")
    print("  no more sophisticated than exhaustive string matching: marking it unresolvable")
    print("  would have been a false statement about the data, and every abstention counted")
    print("  on it a fiction. Each planted pair now shares one UTR, and gate 11 re-runs that")
    print("  cheap attack on every planted row and requires it to come back ambiguous.")
    print("\n  The coverage shortfall is exactly the plant and nothing else: 56/60 and")
    print("  196/200. Seed 3 at n=200 reads 195/200 because it also throws the natural")
    print("  collision gate 10 found, and that row lands in MISSED while correct_abstention")
    print("  stays 4/4 -- a coincidental ambiguity cannot inflate the planted count. The same")
    print("  seed and size without the flag reports 0 planted and 0 abstentions, so the")
    print("  number is attributable to the flag and not to the fixture.")
    print("\nPhase 6 turned on the three remaining deduction terms -- TDS, netted refunds and")
    print("a withheld reserve -- and gate 13 is the first thing in this suite to run all seven")
    print("of Phase 6's flags at once. It found three defects that had been sitting in the code,")
    print("and the reason all three hid is the same: no gate before it ran --netted-refunds")
    print("alongside --batching and --settlement-report-late.")
    print("\n  The serious one was a wrong match, which is the single number this project says")
    print("  never moves. With membership withheld, Tier 2 searches for a subset of payments")
    print("  summing to the credit, pricing each member at its gross, net of fee, or net of")
    print("  TDS -- and never net of its refund. So on a refunded settlement the true subset")
    print("  was not in the search space at all, and the search saw only coincidences. Usually")
    print("  none, and the row abstained as NO_CANDIDATE: 22 of them on seed 1 at n=1000.")
    print("  Occasionally one unrelated subset hit the shrunken target exactly and the row")
    print("  resolved WRONGLY -- two per run on seeds 1 and 2, correctness 0.9962. The fix is")
    print("  a lookup rather than a fifth hypothesis: refunds.csv names the payment each")
    print("  refund cites, so the term is declared and subtracts in every reading. Correctness")
    print("  is back to 1.0000 with 0 wrong matches, and coverage rose with it (549 correct on")
    print("  seed 1 at n=1000, up from 530).")
    print("\n  It was invisible at n=200, which is why gate 13 ignores --skip-slow. The same")
    print("  two seeds that read 0.9962 at n=1000 read a clean 1.0000 with zero wrong matches")
    print("  at n=200: coincidental subsets scale with the candidate pool, so the small size")
    print("  cannot see the failure. A fast run that dropped n=1000 would have reported Phase 6")
    print("  green while blind to its only correctness defect.")
    print("\n  The second was a crash, and it is the good kind. --netted-refunds --batching at")
    print("  seed 3 and seed 7, n=200 died before writing a byte, on an assertion that had")
    print("  predicted the failure in its own message: the linked and planted refunds were")
    print("  drawn disjoint on *payments*, and batching put both into one settlement, whose")
    print("  single refunds_paise term would then have been partly attributable. The draw now")
    print("  excludes whole settlements. An assumption instead of an assertion would have")
    print("  shipped an incoherent refund term in silence.")
    print("\n  One NO_CANDIDATE used to survive here, and Phase 8 retired it rather than")
    print("  keeping the exemption that tolerated it. The row is the credit that nets an")
    print("  orphan refund AND has its membership withheld: the refund cites a payment outside")
    print("  this month's file, so on a withheld settlement it is subtracted from no member and")
    print("  the true subset sits outside the search space entirely. The search looked and")
    print("  honestly found nothing, so gate 13 permitted exactly that row by identity.")
    print("\n  It now gets a real code instead. The withheld-membership branch offers the")
    print("  shortfall back as credit-plus-orphan, and where a subset appears the row abstains")
    print("  as REFUND_UNLINKED -- naming an amount refunds.csv declares rather than one the")
    print("  matcher fitted. Measured across seeds 1/2/3/42 at n=200 and n=1000 on both flag")
    print("  sets: the bump reveals truth's own membership on 3 of the 4 rows and an ambiguity")
    print("  on the 4th, with zero coincidences, and it reveals nothing on every row whose")
    print("  truth nets no orphan refund. So the same evidence the declared-membership path")
    print("  already used is no longer ignored one level down -- that asymmetry was the defect.")
    print("\n  The verdict still abstains. The revealed set is deliberately discarded rather")
    print("  than returned: resolving on it would subtract money the inputs say left SOME")
    print("  settlement without saying it left this one, which is what the declared path")
    print("  refuses too -- the withheld path must not be more confident than the path that")
    print("  can see its membership. And retiring the exemption made this gate stronger, not")
    print("  weaker: a regression in that ordering drops those rows back to NO_CANDIDATE,")
    print("  which is outside ABSTENTION_REASONS, so the gate now catches what it used to")
    print("  excuse. Admitting NO_CANDIDATE to that set would still be the wrong fix -- it")
    print("  would let every failed search score as an honest refusal.")
    print("\n  The reserve is the one term deliberately left unmodelled. Its magnitude appears")
    print("  in no input file, so a rule that fitted it would close every gap by construction;")
    print("  gate 13 asserts no resolved row carries a reserve term and that all ~50 reserved")
    print("  credits abstain as PARTIAL_SETTLEMENT_PENDING rather than vanishing into")
    print("  NO_CANDIDATE. What a pass does NOT prove is that abstaining was the only available")
    print("  answer there -- with the held amount in no input, nothing but truth's own record")
    print("  could say otherwise, and that limit belongs in the write-up.")
    print("\nPhase 7 put rows in the bank statement that are not gateway credits at all, and")
    print("payments that never pay out. Gate 14 runs nine flags at once. The number to read")
    print("there is not noise_recall (0.375 at n=200, 0.389-0.405 at n=1000) but the identity")
    print("beside it: rows IGNORED == plainly_foreign rows in the manifest, checked in BOTH")
    print("directions. Only one of the three noise strata is ignorable -- the other two carry")
    print("a gateway counterparty by construction -- so a recall rate looks like a shortfall")
    print("while the matcher is in fact setting aside every row it is entitled to and not one")
    print("more. A ratio would have hidden a plainly-foreign row leaking into a diagnosis")
    print("behind a look-alike row leaking into IGNORED; the identity cannot.")
    print("\nPhase 8 added foreign currency and missing UTRs, and gate 15 runs eleven flags.")
    print("It is the phase where a headline number got worse on purpose.")
    print("\n  --fx costs about 19% of coverage, and that is the correct price. A foreign")
    print("  payment settles at the rate on the settlement day; payments.csv keeps the gross")
    print("  recorded at capture. So when such a payment sits in the candidate pool, Tier 2")
    print("  can no longer argue the subset it found is the ONLY one summing to the credit --")
    print("  the inference is voided rather than widened, and rows that would have resolved")
    print("  abstain instead. Coverage reads 56.11%-71.20% against Phase 7's 86%-92%. Before")
    print("  that fix, --fx on generator-drawn data produced 7 wrong matches; the plan's")
    print("  hand-moved fixture could not see them. This hole cannot be closed the way Phase")
    print("  6's was: a refund is declared in refunds.csv and can be looked up, but the")
    print("  settlement-day rate is declared in NO input file. That is the entire flag.")
    print("\n  --utr-patchy strips the reference tail from 15% of bank narrations, and")
    print("  wrong_ignore stays 0 -- which on its own proves nothing. Measured: with")
    print("  --utr-patchy --noise-rows alone, ZERO masked credits ever reach the ignore gate,")
    print("  because everything resolves on the amount arithmetic first. The clean zero was")
    print("  vacuous. Only --reserve delivers rows there; of the 12 that arrive, the real")
    print("  two-test conjunction ignores 0 while the rejected 'no readable tail is enough'")
    print("  rule would have ignored all 12. So gate 15 prints the population (19 masked")
    print("  credits at n=200) beside the zero, rather than a zero whose denominator")
    print("  nobody stated.")
    print("\n  And one flag was declined on the strength of a measurement. --rounding-edge")
    print("  was specified to make the two sides disagree about rounding; they cannot,")
    print("  because mul_bps is defined once and shared while only the rates are duplicated.")
    print("  Nor is the behaviour dormant: over 12,000 calls the division is inexact 2,183")
    print("  times per 1,000-payment run, half-up beats truncation 1,125 times, and an exact")
    print("  half -- the only case where half-up and banker's rounding differ -- arises 81.5")
    print("  times. Declining it also keeps a check honest: rounding_edge stays permanently")
    print("  declared-and-inert, so unimplemented() keeps a real subject instead of quietly")
    print("  emptying. #23's Tier 3 tolerance is refused on the same footing -- 2 of 223")
    print("  FX rows sit within its +/-50 paise, and resolving them would mean declaring 50")
    print("  paise unexplained, which is what an exception IS here.")
    print("\nWhat this still does NOT prove: the business-day calendar is exercised only by")
    print("its own unit test, and the narration parser is still not on the match path at all")
    print("(gated, deliberately -- gates 11 and 15 read narrations to attack the data, never")
    print("to resolve it). A planted pair is shown to defeat the two strategies this data")
    print("supports -- date-plus-amount and the UTR tail -- plus the amount arithmetic, not")
    print("every conceivable one; and three-way collisions are refused by I12 at generation")
    print("time rather than scored. The masked-credit defence is exercised only where")
    print("--reserve carries rows to the ignore gate, so it is 12 rows that hold it up, not")
    print("19 per run. And a coverage figure of 56% is a claim about THIS flag set: it is")
    print("not comparable to Phase 7's 86% without naming both.")
    print("\nNext: Phase 9, exception ranking -- and Phase 8 hands it a queue with a shape it")
    print("has to be told about. A row where a withheld reserve and an FX rate move are both")
    print("consistent with the same gap is not hypothetical, so a verdict naming one cause")
    print("would have Phase 9 ranking a guess.")
    print("\nThree reason codes also close here, and they are producerless in three different")
    print("senses -- worth separating, because Phase 9 would otherwise rank codes that cannot")
    print("appear. CREDIT_MISSING has no construction site at all: Phase 7 made it structurally")
    print("unreachable by emitting one verdict per bank row, so a settlement with no credit has")
    print("no verdict slot to carry it. SETTLEMENT_MISSING IS constructed (tier1.py), but the")
    print("branch is unreachable through the loader -- load.py refuses a settlement citing an")
    print("unknown payment before the matcher runs -- so it is defence against that check being")
    print("weakened, not a code that fires. ROUNDING_DRIFT is producerless as a direct result of")
    print("declining --rounding-edge. Phase 8 is the last phase that adds flags, so it is the")
    print("last that could have given any of the three a producer.")
    print("\nPhase 10 put a model near the queue, and check 8 is the assertion that the matcher")
    print("itself never does. Before this phase check_isolation.py had seven checks and not")
    print("one mentioned a network, so the headline claim -- 'the matching engine is")
    print("deliberately not AI' -- rested on nothing a build could fail. Check 8 bans any")
    print("HTTP client or model SDK under hisaab/, scoped to every .py file rather than to")
    print("MATCHER_PACKAGES (hisaab/common/ sits one directory outside that tuple and is")
    print("reachable from the matcher), with hisaab/explain carved out as the one exempt leaf")
    print("-- exempt from the network ban and from nothing else, so the component that talks")
    print("to an LLM still cannot read truth.json, the generator, or the declared fee columns.")
    print("Gate 5 proves it by planting 7 mutants and requiring each refused by its own")
    print("assertion's phrase, not merely by exit code.")
    print("\nThe citation check turned out to have less to check than the roadmap assumed.")
    print("Exception rows carry no computed decomposition at all -- 0 of 295, measured --")
    print("because a row IS an exception for having nothing computed. So the check verifies")
    print("one bank amount plus row ids, and every id or figure the model cites that appears")
    print("in no row it was shown is fatal by default: gate 17 fabricates every citation in")
    print("every group and requires exit 1, naming both fabricated values, in both strict and")
    print("--permissive runs. qa.py closes the gap for the rows that DO have a decomposition")
    print("-- the 349 of 349 RESOLVED rows whose six-term breakdown closes exactly -- by")
    print("pulling a claimed sum out of prose into signed terms and requiring it to close,")
    print("every term to be a real figure in that row, and the total to equal the credit.")
    print("Gate 17 plants an invented term inside an otherwise-closing sum, the one failure")
    print("containment alone cannot see, and requires it refused by name.")
    print("\nStructured output could not be verified here, and the honest answer is to say so")
    print("rather than paper over it. The same request with and without output_config")
    print("returned the same markdown prose through this shell's proxy, and a deliberately")
    print("invalid format value was accepted rather than rejected -- so Phase 9's 'the model")
    print("fenced its JSON' finding is untested, not retired, and client.py's error message")
    print("names the proxy as the likelier cause rather than asserting a protocol failure it")
    print("cannot show happened. Cache telemetry came back absent rather than zero, which is")
    print("not the same fact, and cost per row is stated as unmeasurable through a proxy")
    print("rather than quoted as this project's number -- Phase 9's ~35x inflation figure did")
    print("not reproduce on remeasurement, so the caveat no longer repeats it.")
    print("\nThe dependency question resolved to an extra rather than a rewrite. pyproject.toml")
    print("adds `anthropic` under [project.optional-dependencies], not to the core -- pinned")
    print("to a major range so a 2.x cannot silently change the surface underneath. Every")
    print("module in hisaab/explain/ that self-checks (six, derived from the package rather")
    print("than counted by hand after the count itself went stale twice) is re-run with the")
    print("import blocked at the meta-path level, and _client() is required to refuse with")
    print("install instructions rather than fail obscurely. Nothing else in this suite needs")
    print("it installed.")
    print("\nPhase 11 is a renderer, not a decision: it reads the (up to) five documents a run")
    print("already wrote and prints one self-contained HTML page, stdlib only. Two gaps closed")
    print("first -- `--out` on the scorer and the queue, mirroring what `hisaab.explain` already")
    print("had -- and one gap stayed closed on purpose: `Metrics.as_json()` never serializes")
    print("`landings`, so the matched-records section reads `matches.json` alone rather than")
    print("joining a field that does not exist in any file on disk. What `Verdict.as_json()`")
    print("already carries -- the full six-term decomposition, guaranteed to balance by")
    print("`Verdict.__post_init__` at construction time -- turned out to be everything the")
    print("section needed; gate 18 re-parses that decomposition back out of the rendered TEXT")
    print("and sums it again, because the render step is a new place a transcription error")
    print("could be introduced even though the verdict itself cannot be wrong.")
    print("\nNo canonical match-definition sentence existed anywhere before this phase -- the")
    print("idea was spread across tier1.py's docstring and two ASSUMPTIONS.md rows, phrased")
    print("differently in each. MATCH_DEFINITION in hisaab/common/verdict.py is now the one")
    print("sentence a header, a future README edit, or a judge's question can all quote instead")
    print("of re-deriving a fourth phrasing at render time.")
    print("\nThe truth-vocabulary grep gate 18 runs found something the plan did not predict.")
    print("`tier` was already flagged as matcher-side vocabulary safe to render. `resolvable`")
    print("was not, and a literal ban on it fails on every real report: hisaab/scoring/report.py's")
    print("own shipped metric_block() prints a fixed caption on its Missed line --")
    print("'resolvable, but it abstained' -- static Phase 9 prose this phase is required to")
    print("quote verbatim rather than reword. Gate 18 scopes around that one exact string and")
    print("still fails on any other occurrence, so a real leak elsewhere is still caught.")
    print("\nThe report package earned one narrow, documented tools/check_isolation.py")
    print("TRUTH_READERS entry -- hisaab/report/metric_block.py -- for the same reason")
    print("hisaab/scoring/report.py itself already carries one: quoting metric_block()'s text")
    print("verbatim means calling the real function, and that function reads m.landings, a")
    print("field that exists only on the in-memory Metrics object. Two shortcuts that looked")
    print("equivalent were tried and proven wrong by construction before the exact reconstruction")
    print("formula was found: a planted-unresolvable row dismissed via IGNORED rather than raised")
    print("as an EXCEPTION breaks the cell-only split, and a noise row wrongly RESOLVED breaks")
    print("the shortcut named in report.py's own docstring. The module never touches")
    print("hisaab.scoring.truth_io and is handed nothing but a JSON document a run already")
    print("wrote to disk -- the property that matters is enforced structurally, not by the")
    print("allowlist alone.")
    print("\nA run with no explain artifact and no --ask still renders a complete page: every")
    print("group falls back to its template hint with a visible note saying so, and the Q&A")
    print("section says plainly that no question was recorded. Phase 10's optional layer stays")
    print("optional all the way to the page a person actually opens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
