# Hisaab

**Multi-source payment reconciliation with a measured, honest match rate.**
Razorpay AI Buildathon — Track 04, AI Finance Controller.

*Hisaab* is Hindi for **accounts**, and also for **a reckoning**. Both readings apply: the
tool reconciles payments against settlements against a bank statement, and it reports
what it could not resolve rather than quietly rounding it away.

---

## The claim this project is built to support

Most reconciliation demos report one number: a match rate. That number cannot be checked,
because the person reporting it also decided what counted as a match.

Hisaab reports **two** numbers, and they answer different questions:

| | |
|---|---|
| **Coverage** | how often the matcher committed to an answer |
| **Correctness** | how often the answer it committed to was *right* |

90% coverage with zero wrong matches beats 100% coverage with 85% correctness, and it is
not close. A wrong match silently corrupts the books; an exception just gets a human to
look at it. Any system that averages the two into a single "accuracy" figure has hidden
the one distinction a finance team actually cares about.

Reporting correctness requires knowing the true answer, so the data is **synthetic and
generated with an answer key** (`truth.json`) that the matcher is structurally forbidden
to read. Some cases in it are **planted as unresolvable** — genuinely ambiguous from the
inputs — so "my exception list is honest" becomes a measurement instead of a claim.

**A reported 100% match rate on messy data is a failing signal here, not a win.** It means
the matcher guessed on rows where the inputs contain no answer.

---

## Quick start

Python 3.12. **No dependencies** — standard library only, no virtualenv needed.

```bash
# 1. Generate a month of data plus the answer key (byte-reproducible from the seed)
python -m hisaab.generator --seed 42 --n 60

# 2. Build a known-answer fixture and score it
python tools/fixtures.py --fixture oracle --out out/matches.json
python -m hisaab.scoring --matches out/matches.json --truth truth/
```

Step 1 is required first: `data/` and `truth/` are gitignored, because they regenerate
identically from the seed and versioning them would only add churn.

The scorer prints the metric block:

```
Records processed          60 bank rows (60 gateway, 0 non-gateway)
Run                        seed 42, 2026-08, clean mode, flags: none
Matcher                    fixture:oracle@1

Coverage                   60/60    (100.0%)   how often it committed
Correctness                60/60    (100.0%)   how often it was right
Wrong matches              0   (0 of them on planted-unresolvable rows)
Wrong ignores              0
Correct abstentions        n/a   (0 planted unresolvable in clean mode)
Missed                     0   resolvable, but it abstained

Exceptions                 0
Value in exceptions        ₹0.00
Est. human time to clear   0 min   (vs ~2 h 00 min for the batch by hand)
```

100% is expected *here* and is not the failing signal above: this is clean mode, the
easiest rung of the difficulty dial, and the oracle fixture reads the answer key on
purpose. It exists to prove the target is reachable on this data, so that when the real
matcher falls short the shortfall is unambiguously the matcher's fault and not the
dataset's.

Line 1 of stdout is the same block as JSON, for a caller that parses rather than reads.
`--quiet` prints that line alone.

### Verify the whole thing

```bash
python -m hisaab.generator --seed 42 --n 60   # if you have not already
python tools/acceptance.py
```

Eight gates, one command, exit code is the verdict. Byte-identical output at a fixed
seed across two processes; invariants on three seeds in memory and again re-read from
disk; the leak audit; truth isolation; throughput; the assumptions file; and the four
known-answer fixtures.

---

## Current state — Phase 2 of 13

Honest status, because the interesting part is not built yet:

| | Phase | State |
|---|---|---|
| ✅ | **1. Generator, clean mode** | Done. Three CSVs plus an answer key, 1:1 exact matches |
| ✅ | **2. Scoring harness** | Done. Coverage/correctness, the exception queue, four known-answer fixtures |
| ⬜ | **3. Matcher, Tier 1** | Next. Normalize, block, exact-match — must hit 100% on clean mode |
| ⬜ | 4–8. Fees, batching, adjustments, orphans, planted unresolvables | The difficulty dial, one flag at a time |
| ⬜ | 9–13. Exception ranking, LLM layer, HTML report, holdout, write-up | |

**There is no matcher yet.** Everything scored so far is a fixture with a known answer:
a stub that abstains on every row, an oracle that reads the answer key, a saboteur that
corrupts exactly six matches, and a zip that matches by row position.

The scorer was built **before** the matcher on purpose. Building it second means spending
Phase 3 eyeballing CSVs to decide whether a change helped; building it first turns every
later change into a number that moves.

### The difficulty dial

The generator has thirteen mess flags, all defaulting to off. Clean mode is all of them
off, and it stays in the test set permanently as the regression check: *if it cannot hit
100% here, the code is broken.*

```bash
python -m hisaab.generator --help     # the flags, in difficulty order
```

**They are currently declared but inert** — Phase 1 implemented clean mode only, and
`story.py` does not yet read them. Passing `--fees` today returns unchanged data labelled
as having fees. Phase 4 begins turning them on one at a time.

---

## Three design commitments

These are the parts a judge is most likely to probe, so they are stated rather than
buried.

### 1. The matcher cannot read the answer key

`truth.json` feeds the scorer and nothing else. That is enforced structurally, not
promised:

- It is written to a **separate directory** from the CSVs.
- It is read only through `hisaab/scoring/truth_io.py`, in a package the matching path
  does not import.
- `tools/check_isolation.py` **fails the build** if anything on the matching path imports
  that package or names the truth file in executable code. The guard is already armed for
  `hisaab/matcher/`, which does not exist yet.

The failure mode this prevents is silent and total: a matcher that reads the answer key
still produces a match rate, an exception list, and a confident-looking report. Nothing
crashes. The submission is simply void, and the only way to know is to have checked.

The same discipline applies inside the scorer. `verdict_io.py` — the module that validates
the matcher's output — is deliberately **not** on the truth allowlist: it receives the
credit IDs, seed and month as plain values, so the one scoring job that has no business
seeing the answers structurally cannot.

### 2. The matching engine is deterministic, on purpose

Integer paise throughout, no floats, **no LLM on the match path**. The model's job is
downstream of every decision: explaining a match, triaging exceptions, parsing bank
narration, answering questions.

This is a deliberate answer to "where's the AI?", not an omission. A reconciliation
figure has to be reproducible and defensible — the same seed must give the same number,
and every match must be explainable as arithmetic. A hallucinated journal entry is worse
than an unresolved one.

### 3. Nothing is dropped, and the arithmetic says so

`matched + exceptions = total`, exactly. Enforced in two places: the scorer refuses a
verdict file that misses, duplicates or invents a bank row (naming the offending IDs),
and re-asserts the identity before computing anything. A matcher that omits rows cannot
score well by dropping the hard ones — it does not score at all.

Every assumption behind every number is in **[ASSUMPTIONS.md](ASSUMPTIONS.md)**, marked
🔴 unverified / 🟡 declared / 🟢 guaranteed. The fee rates are 🔴 and say so; the
human-time estimates are 🔴 and say so, including the break-even ratio at which the
comparison stops flattering.

---

## Layout

```
hisaab/
  common/         shared by both sides: money, IDs, reason codes, the verdict contract
  generator/      synthetic payments → settlements → bank statement, plus truth.json
  scoring/        reads truth + a matcher's verdicts, prints the metric block
  matcher/        Phase 3 — the isolation guard is already armed for it
tools/
  acceptance.py       every gate, one command
  fixtures.py         four known-answer matchers (none of them a real matcher)
  check_isolation.py  gate 5: the answer key is unreachable from the matching path
  repro_check.py      byte-identical output at a fixed seed
  verify_output.py    re-reads the written files with an independent parser
```

`hisaab/common/verdict.py` is the interface between the two halves — the matcher writes
that format, the scorer reads it. It lives in `common/` because it touches no truth, so
being importable from the matching path costs nothing.

Every module has a `__main__` self-check (`python -m hisaab.common.money`), and gate 0
runs all of them in dependency order so the first failure is the deepest one.

---

## Reproducibility

The same `--seed` and `--month` produce byte-identical files, verified across two
processes with different hash seeds rather than assumed. Randomness is one named
substream per concern, so adding a mess flag does not shift the data the earlier flags
produced.

**Seeds 1–5 are development seeds. Seed 99 is the holdout** and is not run until Phase 12.
Tuning a tolerance after seeing what it does to your number is the reconciliation
equivalent of test-set leakage, and it is detectable when a judge asks how you picked it.

Wall clock is the one non-deterministic value, and it is quarantined inside a `timing`
object in every document that carries it — so two runs of the same input differ only
there, and the metric block stays byte-comparable.
