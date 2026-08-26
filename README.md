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

# 2. Match it, then score the result against the answer key
python -m hisaab.matcher --data data/ --out out/matches.json
python -m hisaab.scoring --matches out/matches.json --truth truth/
```

Step 1 is required first: `data/` and `truth/` are gitignored, because they regenerate
identically from the seed and versioning them would only add churn.

The scorer prints the metric block:

```
Records processed          60 bank rows (60 gateway, 0 non-gateway)
Run                        seed 42, 2026-08, clean mode, flags: none
Matcher                    tier1@0.3.0

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
easiest rung of the difficulty dial, where one payment becomes one settlement becomes one
bank credit with no fees and no delay. Row 1 of the difficulty dial **requires** 100% —
it is the regression check, not an achievement.

What that number does **not** prove is worth stating in the same breath:

- **The date window is untested.** Every credit lands on its settlement's own date, so
  ±1000 days scores exactly the same as ±0. The window has a real interface and a
  unit-tested tie-break, but no end-to-end run here exercises either.
- **The business-day calendar is exercised only by its own unit test**, including the edge
  where a weekend settlement is zero business days from Monday's credit.
- **The narration parser is not on the match path at all.** That is deliberate, and it is
  gated — see [what Tier 1 refuses to do](#what-tier-1-does-and-what-it-refuses-to-do).

Line 1 of stdout is the same block as JSON, for a caller that parses rather than reads.
`--quiet` prints that line alone.

### Compare against a known answer

The matcher is measured against four fixtures whose scores are known before they run — a
stub that abstains on everything, an oracle that copies the answer key, a saboteur that
corrupts exactly six matches, and a zip that matches by row position:

```bash
python tools/fixtures.py --fixture oracle --out out/oracle.json
python -m hisaab.scoring --matches out/oracle.json --truth truth/
```

The oracle exists to prove the target is *reachable* on this data, so a matcher shortfall
is unambiguously the matcher's fault and not the dataset's. Tier 1 currently matches it.

### Verify the whole thing

```bash
python -m hisaab.generator --seed 42 --n 60   # if you have not already
python tools/acceptance.py
```

Nine gates, one command, exit code is the verdict. Byte-identical output at a fixed
seed across two processes; invariants on three seeds in memory and again re-read from
disk; the leak audit; truth isolation; throughput; the assumptions file; the four
known-answer fixtures; and the matcher at 100/100/0 across three seeds × two sizes,
including the check that blanking every bank narration changes no decision.

---

## Current state — Phase 3 of 13

| | Phase | State |
|---|---|---|
| ✅ | **1. Generator, clean mode** | Done. Three CSVs plus an answer key, 1:1 exact matches |
| ✅ | **2. Scoring harness** | Done. Coverage/correctness, the exception queue, four known-answer fixtures |
| ✅ | **3. Matcher, Tier 1** | Done. Exact `(value_date, net_paise)` join, 100/100/0 on clean mode |
| ⬜ | 4–8. Fees, batching, adjustments, orphans, planted unresolvables | The difficulty dial, one flag at a time |
| ⬜ | 9–13. Exception ranking, LLM layer, HTML report, holdout, write-up | |

The scorer was built **before** the matcher on purpose. Building it second means spending
Phase 3 eyeballing CSVs to decide whether a change helped; building it first turns every
later change into a number that moves.

### What Tier 1 does, and what it refuses to do

One strategy: index settlements by amount, then require the credit's `value_date` to be
within ±0 **business** days of the settlement's `settled_on`. Exactly one candidate
resolves; two or more is `AMBIGUOUS_DUPLICATE_AMOUNT`; none is `NO_CANDIDATE`.
Candidates are **counted**, never taken first — "a candidate exists" and "exactly one
candidate exists" are different facts, and Phase 5's subset-sum depends on the
distinction.

Two shortcuts were available on this data and both are deliberately declined:

- **The UTR tail resolves 60/60 on its own** — sixty distinct tails for sixty
  settlements. Joining on it would score 100% while the amount arithmetic was never
  exercised, and would *stay* at 100% through Phase 4 with no fee model ever written. The
  tail is recorded as corroboration in each verdict's `note` and nothing branches on it.
- **A bare `net_paise` is unique at n=60** — and collides 1–2 times at n=200 and 42–64
  times at n=1000. So the key is the *pair*, the date does real work even at a ±0 window,
  and a wider window would be actively harmful rather than merely unnecessary.

The residual (`credit − Σ gross of matched payments`) is **computed**, not assumed. It is
zero on every row in clean mode; computing it anyway means Phase 4's fee model moves a
number that already exists.

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
  that package or names the truth file in executable code. It scans `hisaab/matcher/`
  statically, so it cannot be defeated by a module that behaves differently when imported.

A sixth check closes a hole the first five left open: **nothing on the matching path may
import `hisaab.generator` either.** That package knows the fee rates, the T+n settlement
cycle and the narration templates — everything the matcher is supposed to infer — so
importing it is reading the answer with extra steps, and the truth-file checks would pass
it silently. Verified by a deliberate violation, not just asserted.

That rule fixes where shared code lives, and the distinction is deliberate: **logic is
shared through `hisaab/common/`, schemas are duplicated on purpose.** The business-day
calendar is shared, because two calendars disagreeing by one day produce a plausible wrong
answer that nothing detects. The five CSV header tuples are copied into the matcher's
loader, because a drifted schema must fail loudly instead of hiding behind a shared symbol.

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
  matcher/        load → normalize → block → tier1 → engine; reads data/, never truth/
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
