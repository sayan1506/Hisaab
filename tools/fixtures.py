"""Four known-answer fixtures — the scorer's own test set.

    python tools/fixtures.py --check                     # gate 8: all four, expected values
    python tools/fixtures.py --fixture oracle --out out/matches.json

The scorer is the instrument every later phase is measured with, so it needs its own
known answers. Each fixture below has a score that is known *before* it runs:

  ========  =============================  ==========================================
  fixture   reads                          expected at seed 42, n=60
  ========  =============================  ==========================================
  stub      bank_statement.csv             0/60 coverage, correctness n/a, 60 missed
  oracle    truth.json                     60/60 coverage, 60/60 correctness, 0 wrong
  saboteur  truth.json                     60/60 coverage, 54/60 correctness, 6 wrong
  zip       payments.csv, bank_statement   60/60 coverage, 21/60 correctness, 39 wrong
  ========  =============================  ==========================================

**None of these is a matcher, and none may move into ``hisaab/matcher/``.** They live in
``tools/`` for that reason. The oracle reads the answer key deliberately: it is four
lines, perfect by construction, and can never be mistaken for the real matcher or
quietly promoted into one. The tempting alternative -- a date+amount join -- is Tier 1 in
embryo, and the day someone adds a tolerance to it to "make the fixture more useful",
Phase 3 has begun inside a test file. There is no coverage gap: whether the *data* is
resolvable from date+amount is already proven 60/60 by ``tools/verify_output.py``'s leak
audit, which is a different tool answering a different question.

**Why four and not the two Phase 1 promised.** The hand-off named a stub (0%) and a zip
(100%). The second half is now wrong: step 7 of Phase 1 deliberately broke positional
alignment, so the zip resolves 21 of 60, not all of them. That turns out to be worth
more than a second perfect fixture, because the zip is the only one of these that
produces *confident, specific, wrong answers* -- and ``wrong_matches`` is the single most
important counter in the metric block. Neither a stub (never commits) nor an oracle
(never errs) exercises it at all. The saboteur then pins the same counter to an exact
expected value rather than a measured one.

Each fixture takes ``seed`` and ``month`` as plain arguments rather than reading them
from truth itself, so that "the stub reads only the bank statement" is true of the
function and not merely of the sentence describing it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from hisaab.common.reasons import Reason  # noqa: E402
from hisaab.common.verdict import (  # noqa: E402
    Decomposition,
    Outcome,
    Verdict,
    VerdictFile,
    write_verdicts,
)
from hisaab.scoring.truth_io import Truth, TruthDecomposition, load_truth  # noqa: E402

#: The saboteur corrupts this many credits: three pairs, swapped. **Even by
#: construction** -- a swap corrupts two rows at a time, so an odd K is unreachable by
#: this mechanism and asking for one would silently give K-1.
SABOTAGE_K = 6

#: Where the fixtures expect the committed run to live.
DEFAULT_DATA = ROOT / "data"
DEFAULT_TRUTH = ROOT / "truth"


class FixtureError(Exception):
    """A fixture could not be built, or did not score what it promised."""


#: What the oracle names as the rule behind its decomposition. Deliberately not
#: ``"gateway fee + GST at declared rates"`` -- the oracle did not derive anything, it read
#: the answer. A fixture that borrowed the matcher's rule name would make a copied
#: decomposition indistinguishable from a computed one in any output that quotes the field,
#: and that field exists precisely so a reader can tell which rule earned a row.
ORACLE_RULE = "copied from the answer key"


def _from_truth(dec: TruthDecomposition) -> Decomposition:
    """Truth's decomposition as a verdict's, term for term. The oracle's only arithmetic.

    Two independently declared shapes -- ``scoring/truth_io.TruthDecomposition`` and
    ``common/verdict.Decomposition`` -- so this function is where they meet, and a term
    added to one without the other fails here rather than silently going uncompared. The
    duplication is the same deliberate kind as ``load.py``'s CSV headers: a schema is
    copied so drift is *found*, while the arithmetic is shared so drift is impossible.

    Note what the caller gets for free. ``Decomposition`` recomputes
    ``expected_credit_paise`` from the six terms rather than carrying truth's stated one, so
    an answer key whose total disagrees with its own components makes the oracle's verdict
    fail construction. The oracle is the fixture that defines what a perfect score *is*, so
    it should be unable to describe an inconsistent answer key as perfect.
    """
    return Decomposition(
        gross_paise=dec.gross_paise,
        fee_paise=dec.fee_paise,
        gst_paise=dec.gst_paise,
        tds_paise=dec.tds_paise,
        refunds_paise=dec.refunds_paise,
        reserve_paise=dec.reserve_paise,
        rule=ORACLE_RULE,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read one CSV in file order. Row order is load-bearing for the zip fixture."""
    import csv

    if not path.exists():
        raise FixtureError(
            f"{path} not found -- generate a run first:\n"
            f"    python -m hisaab.generator --seed 42 --n 60"
        )
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = tuple(next(reader))
        return [dict(zip(header, row)) for row in reader if row]


# ---------------------------------------------------------------------------
# The fixtures
# ---------------------------------------------------------------------------


def stub(seed: int, month: str, data_dir: Path) -> VerdictFile:
    """Abstain on every row. Reads the bank statement and nothing else.

    Proves the scorer survives the empty case: a real ``0%`` coverage rather than a
    crash, and ``n/a`` rather than ``0%`` for a correctness with no commitments to divide
    by. Every zero-denominator path in the report runs on this fixture.

    It reads the bank statement rather than being handed a list of IDs because that is
    what makes it a legitimate *matcher* shape -- Phase 3's first act is to run this,
    confirm 0%, and make it climb.
    """
    bank = _read_csv(data_dir / "bank_statement.csv")
    return VerdictFile(
        seed=seed,
        month=month,
        matcher="fixture:stub@1",
        verdicts=tuple(
            Verdict(row["row_id"], Outcome.EXCEPTION, reason=Reason.NO_CANDIDATE,
                    note="stub fixture: commits to nothing, by design")
            for row in bank
        ),
    )


def oracle(truth: Truth) -> VerdictFile:
    """Copy the answer key. The best score this data admits.

    Four lines of logic, perfect by construction. Its purpose is to establish that the
    target is *reachable* on this exact data, so a Phase 3 shortfall is unambiguously the
    matcher's fault rather than the dataset's.

    It abstains on planted unresolvables rather than answering them, and with truth's own
    reason code. Those rows are unresolvable *from the inputs*, so committing to one --
    even with the right answer in hand -- is not a score any matcher could honestly
    reproduce, and an oracle whose score is unreachable is no longer a target. In clean
    mode nothing is planted, so this branch does not fire until Phase 8.
    """
    noise = set(truth.non_gateway_credit_ids)
    verdicts: list[Verdict] = []
    for credit in truth.credits:
        if credit.credit_id in noise:
            verdicts.append(
                Verdict(credit.credit_id, Outcome.IGNORED, reason=Reason.NON_GATEWAY_CREDIT)
            )
        elif credit.is_planted_unresolvable:
            verdicts.append(
                Verdict(credit.credit_id, Outcome.EXCEPTION,
                        reason=Reason(credit.reason), note="oracle: planted unresolvable")
            )
        else:
            # The decomposition is copied term for term, so the oracle scores 100% on the
            # arithmetic axis as well as on the linkage. That is what makes it a *target*:
            # a shortfall on either axis in Phase 4 is then unambiguously the matcher's
            # fault and not the answer key's.
            #
            # ``residual_paise=0`` is the one thing here that is asserted rather than
            # copied, and the ``Verdict`` constructor is what checks it: the residual must
            # equal ``credit - expected``, so a truth row whose stated credit disagrees
            # with its own six terms fails to build. The fixture that defines a perfect
            # score should not be able to call an inconsistent answer key perfect.
            verdicts.append(
                Verdict(credit.credit_id, Outcome.RESOLVED,
                        settlement_ids=credit.settlement_ids,
                        payment_ids=credit.payment_ids,
                        tier=1, residual_paise=0,
                        credit_amount_paise=credit.decomposition.expected_credit_paise,
                        decomposition=_from_truth(credit.decomposition))
            )
    known = {c.credit_id for c in truth.credits}
    verdicts.extend(
        Verdict(cid, Outcome.IGNORED, reason=Reason.NON_GATEWAY_CREDIT)
        for cid in truth.non_gateway_credit_ids if cid not in known
    )
    return VerdictFile(truth.seed, truth.month, "fixture:oracle@1", tuple(verdicts))


def saboteur(truth: Truth, k: int = SABOTAGE_K) -> VerdictFile:
    """The oracle, with exactly ``k`` credits crossed over in ``k // 2`` pairs.

    The fixture that proves the scorer can *detect* wrongness, with an exact expected
    value rather than a measured one: ``k`` wrong matches, ``(n - k)/n`` correctness, and
    coverage untouched at 100% because a wrong match is still a commitment.

    ``k`` must be even. Swapping crosses two rows at a time, so an odd ``k`` is
    unreachable by this mechanism, and quietly delivering ``k - 1`` would make the
    fixture's expected value a lie -- which in a fixture is worse than a crash.

    The corruptions are *counted*, not assumed. Two credits with identical payment sets
    would swap to a no-op; impossible in clean mode where every set is a distinct
    singleton, but Phase 5's batching makes identical sets plausible, and by then this
    assertion is the only thing standing between a silent no-op and a wrong expected
    value.
    """
    if k % 2:
        raise FixtureError(
            f"k must be even -- a swap corrupts two credits at a time, so k={k} would "
            f"actually corrupt {k - 1}"
        )
    base = oracle(truth)
    by_id = {v.credit_id: v for v in base.verdicts}

    targets = [v for v in base.verdicts if v.outcome is Outcome.RESOLVED][:k]
    if len(targets) < k:
        raise FixtureError(
            f"need {k} resolved verdicts to corrupt, the oracle produced {len(targets)}"
        )

    corrupted = 0
    for left, right in zip(targets[0::2], targets[1::2]):
        if left.payment_set == right.payment_set:
            raise FixtureError(
                f"{left.credit_id} and {right.credit_id} have identical payment sets, so "
                f"swapping them is a no-op and the fixture's expected wrong-match count "
                f"would be wrong"
            )
        for a, b in ((left, right), (right, left)):
            # **Only the linkage is corrupted.** The decomposition and the credit amount
            # stay ``a``'s own, so the verdict remains internally consistent -- it is a
            # correct proof about the wrong payments, which is exactly the failure this
            # fixture models: a matcher that priced the money sensibly and pointed it at
            # the wrong row.
            #
            # Taking ``b``'s decomposition instead was the tempting alternative and is
            # wrong twice over. It would not construct -- ``a``'s credit does not balance
            # against ``b``'s expected credit -- and forcing it to by recomputing the
            # residual would make the saboteur's stated expected values (k wrong matches,
            # coverage untouched) depend on arithmetic it is not trying to test.
            #
            # It also gives gate 8 a free assertion about the scorer: these six rows have
            # payment sets that disagree with truth's, so ``metrics`` must decline to
            # compare their arithmetic at all. The check below pins that denominator.
            by_id[a.credit_id] = Verdict(
                a.credit_id, Outcome.RESOLVED,
                settlement_ids=b.settlement_ids, payment_ids=b.payment_ids,
                tier=a.tier, residual_paise=a.residual_paise,
                credit_amount_paise=a.credit_amount_paise,
                decomposition=a.decomposition,
                note=f"saboteur: crossed with {b.credit_id}",
            )
            corrupted += 1

    if corrupted != k:
        raise FixtureError(f"asked to corrupt {k} credits, corrupted {corrupted}")
    return VerdictFile(
        truth.seed, truth.month, f"fixture:saboteur-k{k}@1",
        tuple(by_id[v.credit_id] for v in base.verdicts),
    )


def zip_fixture(seed: int, month: str, data_dir: Path) -> VerdictFile:
    """Match by row position: ``bank[i]`` to ``payments[i]``. Reads no truth at all.

    The structural shortcut Phase 1's step 7 exists to defeat, kept as a fixture because
    it is the only one here that produces confident, specific, *incorrect* answers.

    Its expected value is exact rather than approximate: at seed 42, n=60 it resolves 21
    of 60, which is the figure ``tools/verify_output.py``'s leak audit reports as
    ``by_row_position``. ``--check`` asserts the two agree. They are independent
    implementations of the same idea, so a disagreement means one of them is wrong and it
    matters which.

    The row order must match the audit's exactly -- both read the CSVs in file order and
    neither sorts. Any re-ordering here silently changes the expected value.
    """
    payments = _read_csv(data_dir / "payments.csv")
    settlements = _read_csv(data_dir / "settlements.csv")
    bank = _read_csv(data_dir / "bank_statement.csv")

    verdicts: list[Verdict] = []
    for i, row in enumerate(bank):
        if i >= len(payments) or i >= len(settlements):
            # More bank rows than payments: the zip has run out of guesses, so it
            # abstains rather than inventing. Phase 7's noise rows reach this branch.
            verdicts.append(
                Verdict(row["row_id"], Outcome.EXCEPTION, reason=Reason.NO_CANDIDATE,
                        note="zip fixture: no row at this position")
            )
            continue
        payment = payments[i]
        credited = int(row["amount_paise"])
        gross = int(payment["gross_paise"])
        # A decomposition computed from data/ alone, and it says the money moved
        # untouched: gross in, gross out, nothing withheld. That claim is what makes the
        # residual below meaningful rather than decorative -- ``credited - gross`` is
        # exactly the remainder this decomposition leaves, so the verdict is internally
        # consistent even where it is factually wrong about which payment it names.
        #
        # It reads no fee column and applies no rate on purpose. This fixture models the
        # *structural* shortcut (position), so giving it a fee model would blur what it
        # measures, and under --fees it will simply carry non-zero residuals -- which the
        # contract permits on a RESOLVED row and Phase 4 treats as a finding.
        decomposition = Decomposition(gross_paise=gross, rule="zip fixture: no deduction modelled")
        residual = credited - gross
        verdicts.append(
            Verdict(
                row["row_id"], Outcome.RESOLVED,
                settlement_ids=(settlements[i]["settlement_id"],),
                payment_ids=(payment["payment_id"],),
                tier=1, residual_paise=residual,
                credit_amount_paise=credited,
                decomposition=decomposition,
                note=f"zip fixture: row position {i}",
            )
        )
    return VerdictFile(seed, month, "fixture:zip@1", tuple(verdicts))


# ---------------------------------------------------------------------------
# Gate 8 -- every fixture scores what it promised
# ---------------------------------------------------------------------------


def build(name: str, truth: Truth, data_dir: Path) -> VerdictFile:
    """Build one fixture by name. ``seed``/``month`` come from truth, the rest does not."""
    if name == "stub":
        return stub(truth.seed, truth.month, data_dir)
    if name == "oracle":
        return oracle(truth)
    if name == "saboteur":
        return saboteur(truth)
    if name == "zip":
        return zip_fixture(truth.seed, truth.month, data_dir)
    raise FixtureError(f"unknown fixture {name!r}")


FIXTURE_NAMES: tuple[str, ...] = ("stub", "oracle", "saboteur", "zip")


def _score_line_via_cli(matches: Path, truth_dir: Path) -> str:
    """Line 1 of the scorer's stdout, as raw text, by running the real CLI.

    Shelling out rather than importing ``score()`` is deliberate: it exercises exit
    codes, argument handling, and the promise that **line 1 of stdout is the metric
    JSON**. An in-process call would leave all three untested, and that line is the
    contract Phase 11 depends on.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "hisaab.scoring",
         "--matches", str(matches), "--truth", str(truth_dir), "--quiet"],
        cwd=ROOT, capture_output=True, text=True,
        env={**_env(), "PYTHONUTF8": "1"},
    )
    if proc.returncode != 0:
        raise FixtureError(
            f"the scorer refused to score (exit {proc.returncode})\n"
            f"{proc.stdout.rstrip()}\n{proc.stderr.rstrip()}"
        )
    return proc.stdout.splitlines()[0] if proc.stdout.strip() else ""


def _score_via_cli(matches: Path, truth_dir: Path) -> dict[str, object]:
    """``_score_line_via_cli``, parsed."""
    first = _score_line_via_cli(matches, truth_dir)
    try:
        return json.loads(first)
    except json.JSONDecodeError as e:
        raise FixtureError(f"line 1 of stdout is not JSON: {first!r} ({e})") from e


def _without_timing(doc: dict[str, object]) -> str:
    """The document minus ``timing``, canonically serialised for comparison."""
    return json.dumps({k: v for k, v in doc.items() if k != "timing"}, sort_keys=True)


def _check_timing_quarantine(run: VerdictFile, truth_dir: Path, tmp: Path) -> None:
    """The metric JSON is identical across two runs that differ only in wall clock.

    Phase 11 quotes the metric block into a report subject to the same reproducibility
    rule as everything else, so a wall clock in the document body would make a
    byte-comparison of two identical runs fail. ``Metrics.as_json`` confines it to
    ``timing``, exactly as ``emit.build_manifest`` does.

    Scoring the same fixture twice would not test that -- the fixtures leave the clock
    unset, so ``timing`` is ``null`` both times and the comparison passes for the wrong
    reason. This scores two verdict files that are identical *except* for the clock, and
    asserts both halves: the body does not move, and the clock is genuinely carried
    rather than silently dropped.
    """
    docs = []
    for seconds in (0.01, 9.99):
        clocked = VerdictFile(
            run.seed, run.month, run.matcher, run.verdicts, wall_clock_seconds=seconds
        )
        path = write_verdicts(tmp / f"clock_{seconds}.json", clocked)
        docs.append(_score_via_cli(path, truth_dir))

    if _without_timing(docs[0]) != _without_timing(docs[1]):
        raise FixtureError(
            "the metric JSON moved when only the wall clock changed -- something "
            "non-deterministic escaped the timing object, and Phase 11's report will "
            "fail its own reproducibility check"
        )
    if docs[0]["timing"] == docs[1]["timing"]:
        raise FixtureError(
            f"both runs reported the same timing ({docs[0]['timing']!r}) despite "
            f"different wall clocks -- the field is being dropped, so the comparison "
            f"above proves nothing"
        )


def _env() -> dict[str, str]:
    import os

    return dict(os.environ)


def _leak_audit_row_position(data_dir: Path, truth: Truth) -> int:
    """``by_row_position`` from the leak audit, as an int. The cross-check's other half."""
    import verify_output  # the sibling tool; tools/ is on sys.path above

    audit = verify_output.leak_audit(
        payments=_read_csv(data_dir / "payments.csv"),
        settlements=_read_csv(data_dir / "settlements.csv"),
        bank=_read_csv(data_dir / "bank_statement.csv"),
        items=_read_csv(data_dir / "settlement_items.csv"),
        truth=truth,
    )
    reported = audit["leak_audit"]["by_row_position"]  # type: ignore[index]
    return int(str(reported).split("/")[0])


def _expectations(truth: Truth, data_dir: Path) -> dict[str, dict[str, object]]:
    """What each fixture must score. Derived from n and K, never from a previous run.

    ``zip``'s ``correct`` is the one value not derivable from first principles, so it is
    cross-checked against the leak audit rather than hard-coded to 21.

    Phase 4 step 5 adds the arithmetic axis. Its *denominator* is the interesting half:
    ``metrics`` compares a decomposition only where the payment sets already agree, so
    ``checked`` is a different count on every fixture and pinning it here is what keeps
    ``0 mismatches`` from being a clean bill of health over an empty comparison.
    """
    n = len(truth.credits)
    planted = len(truth.planted_unresolvable)
    resolvable = n - planted
    zip_correct = _leak_audit_row_position(data_dir, truth)

    # The zip claims every credit equals its payment's gross. Where the position guess is
    # right that agrees with truth term for term *only if* truth withheld nothing, so the
    # expectation below is derived from the answer key rather than assumed. Under --fees it
    # stops holding, and the right response is to refuse rather than to skip the assertion:
    # a gate that quietly stops checking still prints a passing line.
    deducting = [c.credit_id for c in truth.credits
                 if c.decomposition.gross_paise != c.decomposition.expected_credit_paise]
    if deducting:
        raise FixtureError(
            f"{len(deducting)} credit(s) in this answer key withhold something (e.g. "
            f"{deducting[0]}), so the zip fixture's expected arithmetic-mismatch count is "
            f"no longer 0 and is not derivable from n alone. Gate 8 runs against the "
            f"committed clean run; derive the expectation before pointing it at --fees data"
        )

    return {
        "stub": {
            "committed": 0, "correct": 0, "wrong": 0, "missed": resolvable,
            "exceptions": n, "coverage": 0.0, "correctness": None,
            # Nothing was committed, so nothing was checkable: n/a, never 1.0.
            "checked": 0, "mismatches": 0, "agreement": None,
        },
        "oracle": {
            "committed": resolvable, "correct": resolvable, "wrong": 0, "missed": 0,
            "exceptions": planted, "coverage": 1.0 if resolvable else None,
            "correctness": 1.0 if resolvable else None,
            # Copied term for term from the answer key, so the target is 100% on this axis
            # too -- a Phase 4 shortfall is then the matcher's fault, not the key's.
            "checked": resolvable, "mismatches": 0,
            "agreement": 1.0 if resolvable else None,
        },
        "saboteur": {
            "committed": resolvable, "correct": resolvable - SABOTAGE_K,
            "wrong": SABOTAGE_K, "missed": 0, "exceptions": planted,
            "coverage": 1.0, "correctness": (resolvable - SABOTAGE_K) / resolvable,
            # The K crossed rows name payment sets truth disagrees with, so their
            # arithmetic is *not* compared -- it describes different money. This is the
            # assertion that proves the two axes are genuinely independent: correctness
            # falls to (n-K)/n while agreement stays 1.0 over a denominator K smaller.
            "checked": resolvable - SABOTAGE_K, "mismatches": 0, "agreement": 1.0,
        },
        "zip": {
            "committed": n, "correct": zip_correct, "wrong": n - zip_correct,
            "missed": 0, "exceptions": 0, "coverage": 1.0,
            "correctness": zip_correct / n,
            # Only the rows the position guess got right are comparable, and on a clean run
            # they agree: gross in, gross out, nothing withheld on either side.
            "checked": zip_correct, "mismatches": 0,
            "agreement": 1.0 if zip_correct else None,
        },
    }


def check(data_dir: Path, truth_dir: Path, verbose: bool = True) -> dict[str, object]:
    """Gate 8: build all four fixtures, score each through the CLI, assert the table."""
    truth = load_truth(truth_dir)
    expectations = _expectations(truth, data_dir)
    n = len(truth.credits)
    results: dict[str, object] = {}

    if verbose:
        print(f"gate 8 -- four known-answer fixtures, seed {truth.seed}, n={n}")

    with tempfile.TemporaryDirectory(prefix="hisaab-fixtures-") as tmp:
        for name in FIXTURE_NAMES:
            run = build(name, truth, data_dir)
            path = write_verdicts(Path(tmp) / f"{name}.json", run)
            doc = _score_via_cli(path, truth_dir)
            cells = doc["cells"]  # type: ignore[index]
            rates = doc["rates"]  # type: ignore[index]
            want = expectations[name]

            wrong = (
                cells["wrong_match"] + cells["wrong_match_invented"] + cells["lucky_guess"]
            )
            committed = cells["correct"] + wrong
            arithmetic = doc["decomposition"]  # type: ignore[index]
            got = {
                "committed": committed,
                "correct": cells["correct"],
                "wrong": wrong,
                "missed": cells["missed"],
                "exceptions": doc["exceptions"]["count"],  # type: ignore[index]
                "coverage": rates["coverage"],
                "correctness": rates["correctness"],
                # Phase 4 step 5. ``checked`` is asserted alongside ``mismatches`` because
                # a zero mismatch count over a zero denominator is not a passing arithmetic,
                # it is an unrun one -- and it prints identically.
                "checked": arithmetic["checked"],  # type: ignore[index]
                "mismatches": arithmetic["mismatches"],  # type: ignore[index]
                "agreement": rates["decomposition_agreement"],
            }
            for key, expected in want.items():
                actual = got[key]
                if isinstance(expected, float) and isinstance(actual, float):
                    ok = abs(expected - actual) < 1e-9
                else:
                    ok = expected == actual
                if not ok:
                    raise FixtureError(
                        f"fixture {name}: {key} is {actual!r}, expected {expected!r}\n"
                        f"  full cells: {cells}\n  full rates: {rates}"
                    )

            # The identity, on every fixture: nothing dropped, nothing double-counted.
            landed = sum(int(v) for v in cells.values())  # type: ignore[union-attr]
            if landed != doc["totals"]["bank_rows"]:  # type: ignore[index]
                raise FixtureError(
                    f"fixture {name}: {landed} classified verdicts for "
                    f"{doc['totals']['bank_rows']} bank rows"  # type: ignore[index]
                )
            results[name] = got
            if verbose:
                cov = "n/a" if got["coverage"] is None else f"{got['coverage'] * 100:.1f}%"
                corr = (
                    "n/a" if got["correctness"] is None
                    else f"{got['correctness'] * 100:.1f}%"
                )
                # The arithmetic axis prints its denominator, not just its rate. "100%" over
                # nothing and "100%" over 54 rows are different claims, and a reader
                # scanning this table has to be able to tell them apart.
                agree = (
                    "n/a" if got["agreement"] is None
                    else f"{got['agreement'] * 100:.1f}%"
                )
                print(
                    f"    {name:<9} coverage {cov:>6}  correctness {corr:>6}  "
                    f"wrong {got['wrong']:>2}  exceptions {got['exceptions']:>2}  "
                    f"arithmetic {agree:>6} of {got['checked']:>2}   as expected"
                )

        # Acceptance item 7 of .plan/phase2.md: the metric block renders identically
        # from the same seed. The oracle is the fixture Phase 11 quotes, so it is the
        # one that has to hold.
        _check_timing_quarantine(oracle(truth), truth_dir, Path(tmp))
        if verbose:
            print(
                "    reproducibility: the metric JSON is byte-identical outside timing/ "
                "when only the wall clock moves"
            )

    zip_correct = expectations["zip"]["correct"]
    if verbose:
        print(
            f"    cross-check: the zip fixture and tools/verify_output.py's leak audit "
            f"agree at {zip_correct}/{n} by row position"
        )
        print("\ngate 8 passes -- the scorer reports the known answer on all four fixtures")
    results["zip_correct"] = zip_correct
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build or verify the scorer's known-answer fixtures (gate 8).",
        epilog="None of these is a matcher. See the module docstring.",
    )
    p.add_argument("--fixture", choices=FIXTURE_NAMES, help="write one fixture's matches.json")
    p.add_argument("--out", type=Path, default=Path("out/matches.json"),
                   help="where to write it (default: out/matches.json)")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    p.add_argument("--check", action="store_true", help="gate 8: score all four, assert")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if not args.fixture and not args.check:
        p.error("nothing to do: pass --check or --fixture NAME")

    try:
        if args.fixture:
            truth = load_truth(args.truth)
            run = build(args.fixture, truth, args.data)
            written = write_verdicts(args.out, run)
            if not args.quiet:
                counts = run.counts()
                print(
                    f"{args.fixture}: {len(run.verdicts)} verdicts "
                    f"({counts['RESOLVED']} resolved, {counts['EXCEPTION']} exceptions, "
                    f"{counts['IGNORED']} ignored) -> {written}"
                )
                print(f"  score it: python -m hisaab.scoring --matches {written} "
                      f"--truth {args.truth}")
        if args.check:
            check(args.data, args.truth, verbose=not args.quiet)
    except FixtureError as e:
        print(f"FIXTURE FAILED\n  {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
